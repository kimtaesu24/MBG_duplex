"""
Full-Duplex-Bench v1.0 inference script for finetuned Moshi.

Walks every {data_dir}/{task}/{ID}/input.wav, runs offline inference,
and writes output.wav in the same {ID}/ folder so the benchmark
evaluation scripts find it at the expected path.

Usage:
    python bench_inference.py \\
        --config  path/to/args.yaml \\
        --data-dir /path/to/v1_0 \\
        --tasks candor_pause_handling synthetic_pause_handling \\
                candor_turn_taking icc_backchannel \\
                synthetic_user_interruption \\
        [--ckpt-dir path/to/checkpoint] \\
        [--overwrite] \\
        [--device cuda]
"""

import argparse
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../moshi")))

import tarfile
from pathlib import Path

import numpy as np
import sentencepiece
import sphn
import torch
import yaml
from huggingface_hub import hf_hub_download

import moshi.models.loaders as loaders
import safetensors.torch
from moshi.client_utils import make_log
from moshi.models import LMGen
from moshi.models.lm import load_audio as lm_load_audio
from moshi.models.lm import _iterate_audio as lm_iterate_audio
from moshi.models.lm import encode_from_sphn as lm_encode_from_sphn
from moshi.offline import warmup, decode_tokens_to_pcm


V1_TASKS = [
    "candor_pause_handling",
    "synthetic_pause_handling",
    "candor_turn_taking",
    "icc_backchannel",
    "synthetic_user_interruption",
]


def log(level: str, msg: str):
    print(make_log(level, msg))


def _strip_peft_prefixes(state_dict: dict) -> dict:
    """Strip training wrapper prefixes so keys match the bare LMModel namespace.

    Training wraps: PeftModel(PeftCompatibleWrapper(LMModel))
    Saved keys:     base_model.model.model.<lm_key>
    After strip:    <lm_key>

    .original_module.* entries are frozen originals kept by PEFT — skip them.
    """
    cleaned = {}
    for k, v in state_dict.items():
        if ".original_module." in k:
            continue
        if k.startswith("base_model.model.model."):
            k = k[len("base_model.model.model."):]
        cleaned[k] = v
    return cleaned


def load_checkpoint(lm, ckpt_dir):
    consolidated = os.path.join(ckpt_dir, "consolidated", "consolidated.safetensors")
    lora_path = os.path.join(ckpt_dir, "consolidated", "lora.safetensors")

    if os.path.exists(consolidated):
        log("info", f"Loading full checkpoint from {consolidated}")
        state_dict = safetensors.torch.load_file(consolidated)
        state_dict = _strip_peft_prefixes(state_dict)
        missing, _ = lm.load_state_dict(state_dict, strict=False)
        bc_missing = [k for k in missing if "backchannel" in k]
        face_missing = [k for k in missing if "face_module" in k]
        if bc_missing:
            log("warning", f"Backchannel keys still missing: {bc_missing}")
        if face_missing:
            log("warning", f"Face module keys still missing: {face_missing[:3]}")
        if not bc_missing and not face_missing:
            log("info", "Checkpoint loaded successfully.")

    elif os.path.exists(lora_path):
        log("info", f"Loading LoRA checkpoint from {lora_path}")
        state_dict = safetensors.torch.load_file(lora_path)
        state_dict = _strip_peft_prefixes(state_dict)

        # 1. Load backchannel + face_module weights directly into LMModel.
        #    These modules are full-finetuned (no LoRA) and stored flat in lora.safetensors.
        direct_weights = {k: v for k, v in state_dict.items()
                          if k.startswith("backchannel.") or k.startswith("face_module.")}
        if direct_weights:
            missing, _ = lm.load_state_dict(direct_weights, strict=False)
            bc_missing = [k for k in missing if "backchannel" in k]
            face_missing = [k for k in missing if "face_module" in k]
            if bc_missing:
                log("warning", f"Backchannel keys missing: {bc_missing}")
            if face_missing:
                log("warning", f"Face module keys missing: {face_missing[:3]}")
            if not bc_missing and not face_missing:
                log("info", f"Direct weights loaded ({len(direct_weights)} tensors).")

        # 2. Load LoRA adapter weights via PEFT then merge into base weights.
        #    Training used PeftCompatibleWrapper so saved keys are at depth 3:
        #      base_model.model.model.<lm_key>  →  stripped: <lm_key>
        #    Inference PEFT model (bare LMModel) has depth 2:
        #      base_model.model.<lm_key>
        #    → re-add one "base_model.model." prefix before loading.
        #
        #    target_modules must match training (wrapped_model.py).
        #    In Personaplex, only out_proj and text_linear are plain nn.Linear
        #    that PEFT can wrap — the other names in the regex don't match any
        #    module in the actual graph.  face_module / backchannel are excluded
        #    by the negative lookahead so their linear layers are left as-is.
        lora_weights = {k: v for k, v in state_dict.items()
                        if not k.startswith("backchannel.") and not k.startswith("face_module.")}
        if lora_weights:
            try:
                from peft import get_peft_model, LoraConfig, TaskType
                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=64,
                    lora_alpha=128,
                    target_modules=(
                        r"(?!.*(face_module|backchannel))"
                        r".*(in_proj|out_proj|linear1|linear2|text_linear|input_proj)"
                    ),
                    bias="none",
                )
                peft_lm = get_peft_model(lm, lora_config)
                prefixed = {f"base_model.model.{k}": v for k, v in lora_weights.items()}
                missing_lora, _ = peft_lm.load_state_dict(prefixed, strict=False)
                missing_lora = [k for k in missing_lora if "lora_" in k]
                if missing_lora:
                    log("warning", f"Some LoRA keys missing: {missing_lora[:3]}")
                else:
                    log("info", f"LoRA weights loaded ({len(lora_weights)} tensors).")
                # Merge LoRA deltas into base weights and restore a plain LMModel.
                lm = peft_lm.merge_and_unload()
            except ImportError:
                log("error", "LoRA weights found but PEFT is not installed.")
    else:
        log("warning", f"No checkpoint found in {ckpt_dir}, running base model.")
    return lm


def collect_input_files(data_dir: Path, tasks: list[str], overwrite: bool) -> list[Path]:
    """Return sorted list of input.wav paths that still need output.wav."""
    files = []
    for task in tasks:
        task_dir = data_dir / task
        if not task_dir.exists():
            log("warning", f"Task directory not found, skipping: {task_dir}")
            continue
        for sample_dir in sorted(task_dir.iterdir()):
            inp = sample_dir / "input.wav"
            if not inp.exists():
                continue
            out = sample_dir / "output.wav"
            if out.exists() and not overwrite:
                log("info", f"[SKIP] {inp} (output exists)")
                continue
            files.append(inp)
    return files


def infer_one(
    inp: Path,
    mimi,
    other_mimi,
    lm_gen,
    text_tokenizer,
    frame_size: int,
    device: str,
    voice_prompt_path: str = "",
    vap_mimi=None,
    face_gen: "FaceGenerator | None" = None,
):
    """Run a single input.wav through the model and return (output_pcm, sample_rate).

    If face_gen is provided, also generates face motion and saves it as
    ``inp.parent / "face_motion.npy"``.
    """
    mimi.reset_streaming()
    other_mimi.reset_streaming()
    lm_gen.reset_streaming()

    # Must happen after reset_streaming() and before step_system_prompts() so the
    # embeddings replay and KV-cache restore land on a clean streaming state.
    if voice_prompt_path:
        lm_gen.load_voice_prompt_embeddings(voice_prompt_path)

    if face_gen is not None:
        face_gen.reset()

    lm_gen.step_system_prompts(mimi)
    mimi.reset_streaming()

    user_audio = lm_load_audio(str(inp), mimi.sample_rate)

    if user_audio.ndim == 2 and user_audio.shape[0] == 2:
        # AMI stereo is often [Agent, User] — swap so Ch0 = User (Moshi convention)
        user_audio = user_audio[[1, 0], :]
    elif user_audio.ndim == 2:
        user_audio = user_audio[0:1]   # keep shape (1, T)
    else:
        user_audio = user_audio[np.newaxis]  # (1, T)

    target_samples = user_audio.shape[-1]
    generated_frames = []

    for user_encoded in lm_encode_from_sphn(
        mimi,
        lm_iterate_audio(user_audio, sample_interval_size=frame_size, pad=True),
        max_batch=1,
    ):
        steps = user_encoded.shape[-1]
        for c in range(steps):
            if face_gen is not None:
                result = lm_gen.step(user_encoded[:, :, c : c + 1], return_z=True)
                if result is None or result[0] is None:
                    continue
                tokens, z = result
                face_gen.add_step(tokens, z)
            else:
                tokens = lm_gen.step(user_encoded[:, :, c : c + 1])
                if tokens is None:
                    continue
            generated_frames.append(decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens))

    if not generated_frames:
        return np.zeros(target_samples, dtype=np.float32), mimi.sample_rate

    output_pcm = np.concatenate(generated_frames, axis=-1)

    # Trim or zero-pad to match input length so output is time-synchronous
    if output_pcm.shape[-1] > target_samples:
        output_pcm = output_pcm[:target_samples]
    elif output_pcm.shape[-1] < target_samples:
        output_pcm = np.concatenate(
            [output_pcm, np.zeros(target_samples - output_pcm.shape[-1], dtype=output_pcm.dtype)]
        )

    # Face motion generation (post-loop, uses accumulated tokens/z)
    if face_gen is not None and getattr(face_gen, "_agent_tokens", None):
        try:
            motion = face_gen.generate_numpy(vap_mimi)  # [T_face, 56]
            face_out = inp.parent / "face_motion.npy"
            np.save(str(face_out), motion)
        except Exception as _e:
            import traceback as _tb
            log("warning", f"Face generation failed for {inp.name}: {_e}")
            _tb.print_exc()

    return output_pcm, mimi.sample_rate


def run(args):
    data_dir = Path(args.data_dir)
    device = args.device

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    hf_repo = config.get("moshi_paths", {}).get("hf_repo_id", loaders.DEFAULT_REPO)

    # ── Load models ───────────────────────────────────────────────────────────
    log("info", "Loading Mimi...")
    mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)
    # Dedicated instance for face latent decoding — must not have streaming_forever called on it
    vap_mimi = loaders.get_mimi(mimi_weight, device)

    log("info", "Loading text tokenizer...")
    tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

    log("info", "Loading Moshi LM...")
    moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)

    for k in list(loaders._lm_kwargs.keys()):
        if k.startswith("backchannel_"):
            del loaders._lm_kwargs[k]

    bc_cfg = config.get("backchannel", {})
    if bc_cfg.get("enable", False):
        loaders._lm_kwargs["backchannel_enabled"] = True
        loaders._lm_kwargs["backchannel_module_type"] = bc_cfg.get("module_type", "mlp")
        loaders._lm_kwargs["backchannel_vap_dim"] = bc_cfg.get("vap_dim", 256)
        loaders._lm_kwargs["backchannel_bc_hidden"] = bc_cfg.get("bc_hidden", 512)
        loaders._lm_kwargs["backchannel_gumbel_temp_init"] = bc_cfg.get("gumbel_temp_init", 1.0)
        loaders._lm_kwargs["backchannel_gumbel_temp_min"] = bc_cfg.get("gumbel_temp_min", 0.5)
        loaders._lm_kwargs["backchannel_gumbel_anneal_rate"] = bc_cfg.get("gumbel_anneal_rate", 0.0001)
        if bc_cfg.get("module_type", "mlp") == "vap_gpt":
            loaders._lm_kwargs["backchannel_vap_repo_path"] = bc_cfg.get("vap_gpt_repo_path", "")
            loaders._lm_kwargs["backchannel_vap_checkpoint"] = bc_cfg.get("vap_gpt_checkpoint", None)
            loaders._lm_kwargs["backchannel_vap_channel_layers"] = bc_cfg.get("vap_gpt_channel_layers", 1)
            loaders._lm_kwargs["backchannel_vap_cross_layers"] = bc_cfg.get("vap_gpt_cross_layers", 3)
            loaders._lm_kwargs["backchannel_vap_num_heads"] = bc_cfg.get("vap_gpt_num_heads", 4)
            loaders._lm_kwargs["backchannel_vap_dropout"] = bc_cfg.get("vap_gpt_dropout", 0.1)
        log("info", f"Backchannel module enabled (type={bc_cfg.get('module_type', 'mlp')}).")
    else:
        loaders._lm_kwargs["backchannel_enabled"] = False
        log("info", "Backchannel VAP module disabled.")

    lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=args.cpu_offload)

    if args.ckpt_dir:
        lm = load_checkpoint(lm, args.ckpt_dir)

    lm.eval()

    # ── Face generation module ────────────────────────────────────────────────
    face_gen = None
    face_cfg = config.get("face_gen", {})
    if face_cfg.get("enable", False):
        ckpt_path = face_cfg.get("ckpt_path")
        if ckpt_path is None:
            log("warning", "face_gen.enable=true but face_gen.ckpt_path is null — skipping face generation.")
        else:
            from face_generation import FaceGenerator  # noqa: PLC0415
            face_dir = os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../moshi/moshi/models/face",
            ))
            log("info", f"Loading face generation model from {ckpt_path}")
            face_gen = FaceGenerator.from_checkpoint(
                face_dir=face_dir,
                ckpt_path=ckpt_path,
                device=device,
                max_context_frames=int(face_cfg.get("max_context_frames", 25)),
                hidden_dim=int(face_cfg.get("hidden_dim", 512)),
                layers=int(face_cfg.get("layers", 6)),
                heads=int(face_cfg.get("heads", 8)),
                code_dim=int(face_cfg.get("code_dim", 32)),
                prior_warmup_frames=int(face_cfg.get("prior_warmup_frames", 10)),
            )
            log("info", "Face generation model loaded.")

    # ── Build LMGen ───────────────────────────────────────────────────────────
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        use_sampling=True,
        temp=0.8,
        temp_text=0.7,
        top_k=250,
        top_k_text=25,
    )

    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    log("info", "Warming up...")
    warmup(mimi, other_mimi, lm_gen, device, frame_size)

    # ── Voice / text prompt ───────────────────────────────────────────────────
    voice_prompt_path = "voices/NATM1.pt"
    if not os.path.exists(voice_prompt_path):
        voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=os.path.dirname(voices_tgz))
        voice_prompt_path = os.path.join(os.path.dirname(voices_tgz), "voices/NATM1.pt")

    text_prompt = "<system> You are a highly realistic conversational AI. You should listen attentively. <system>"
    lm_gen.text_prompt_tokens = text_tokenizer.encode(text_prompt)

    log("info", "Model ready.")

    # ── Collect files ─────────────────────────────────────────────────────────
    input_files = collect_input_files(data_dir, args.tasks, args.overwrite)
    log("info", f"Found {len(input_files)} sample(s) to process.")

    # ── Main loop ─────────────────────────────────────────────────────────────
    failed = 0

    with tqdm(input_files, unit="sample") as pbar:
        for inp in pbar:
            out = inp.parent / "output.wav"
            pbar.set_postfix_str(inp.parent.name)
            try:
                pcm, sr = infer_one(
                    inp, mimi, other_mimi, lm_gen, text_tokenizer, frame_size, device,
                    voice_prompt_path=voice_prompt_path,
                    vap_mimi=vap_mimi,
                    face_gen=face_gen,
                )
                sphn.write_wav(str(out), pcm, sr)
            except Exception as e:
                import traceback
                log("warning", f"FAILED {inp}: {e}")
                traceback.print_exc()
                failed += 1

    log("info", f"Done: {len(input_files)-failed}/{len(input_files)} samples ({failed} failed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Full-Duplex-Bench v1.0 inference")
    parser.add_argument("--config", required=True, help="Path to args.yaml / training config")
    parser.add_argument("--data-dir", required=True, help="Root of the v1.0 dataset (contains task sub-folders)")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=V1_TASKS,
        help="Task sub-folder names to process (default: all v1.0 tasks)",
    )
    parser.add_argument("--ckpt-dir", default=None, help="Finetuned checkpoint directory")
    parser.add_argument("--overwrite", action="store_true", help="Re-generate even if output.wav exists")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    run(args)
