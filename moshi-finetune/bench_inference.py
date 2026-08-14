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
import random
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
from moshi.offline import warmup

# Keep benchmark inference on exactly the same model/generation implementation as
# inference.py. Importing it also installs lm.LMModel into the loader.
from inference import (
    LMGen,
    _repair_config_paths,
    _wrap_with_system_tags,
    infer_one,
    load_checkpoint,
    log,
    log_fusion_state,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


V1_TASKS = [
    "candor_pause_handling",
    "synthetic_pause_handling",
    "candor_turn_taking",
    "icc_backchannel",
    "synthetic_user_interruption",
]


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


def run(args):
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        log("warning", "CUDA requested but not available — falling back to CPU.")
        device = "cpu"
    if args.dtype == "auto":
        lm_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    else:
        lm_dtype = getattr(torch, args.dtype)
    log("info", f"device={device}, lm dtype={lm_dtype}")

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if not args.no_auto_fix_config_paths:
        _repair_config_paths(config)
    face_cfg = config.get("face_gen", {})
    face_enabled = bool(face_cfg.get("enable", False))

    if args.ckpt_dir:
        checkpoint_files = (
            Path(args.ckpt_dir) / "consolidated" / "consolidated.safetensors",
            Path(args.ckpt_dir) / "consolidated" / "lora.safetensors",
        )
        if not any(path.is_file() for path in checkpoint_files):
            raise FileNotFoundError(
                f"No checkpoint weights found under --ckpt-dir: {args.ckpt_dir}"
            )

    path_cfg = config.get("moshi_paths", {})
    hf_repo = path_cfg.get("hf_repo_id") or loaders.DEFAULT_REPO

    def resolve_weight(config_key: str, hub_name: str) -> str:
        local_path = path_cfg.get(config_key)
        if local_path:
            if not os.path.isfile(local_path):
                raise FileNotFoundError(
                    f"moshi_paths.{config_key} does not exist: {local_path}"
                )
            log("info", f"Using local {config_key}: {local_path}")
            return local_path
        log("info", f"moshi_paths.{config_key} is unset; downloading {hub_name} from {hf_repo}.")
        return hf_hub_download(hf_repo, hub_name)

    # ── Load models ───────────────────────────────────────────────────────────
    log("info", "Loading Mimi...")
    mimi_weight = resolve_weight("mimi_path", loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)

    log("info", "Loading text tokenizer...")
    tokenizer_path = resolve_weight("tokenizer_path", loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

    log("info", "Loading Moshi LM...")
    moshi_weight = resolve_weight("moshi_path", loaders.MOSHI_NAME)

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
        loaders._lm_kwargs["backchannel_pad_token_id"] = bc_cfg.get("pad_token_id", 3)
        loaders._lm_kwargs["backchannel_epad_token_id"] = bc_cfg.get("epad_token_id", 0)
        loaders._lm_kwargs["backchannel_fusion_trainable"] = bc_cfg.get("fusion_trainable", False)
        loaders._lm_kwargs["backchannel_fusion_bc_init"] = bc_cfg.get("fusion_bc_init", 0.0)
        loaders._lm_kwargs["backchannel_fusion_vap_init"] = bc_cfg.get("fusion_vap_init", 0.0)
        loaders._lm_kwargs["backchannel_fusion_vad_init"] = bc_cfg.get("fusion_vad_init", 0.0)
        loaders._lm_kwargs["backchannel_fusion_bias_init"] = bc_cfg.get("fusion_bias_init", 0.0)
        if bc_cfg.get("module_type", "mlp") == "vap_gpt":
            loaders._lm_kwargs["backchannel_vap_repo_path"] = bc_cfg.get("vap_gpt_repo_path", "")
            loaders._lm_kwargs["backchannel_vap_checkpoint"] = bc_cfg.get("vap_gpt_checkpoint", None)
            loaders._lm_kwargs["backchannel_vap_channel_layers"] = bc_cfg.get("vap_gpt_channel_layers", 1)
            loaders._lm_kwargs["backchannel_vap_cross_layers"] = bc_cfg.get("vap_gpt_cross_layers", 3)
            loaders._lm_kwargs["backchannel_vap_num_heads"] = bc_cfg.get("vap_gpt_num_heads", 4)
            loaders._lm_kwargs["backchannel_vap_dropout"] = bc_cfg.get("vap_gpt_dropout", 0.1)
            loaders._lm_kwargs["backchannel_vap_use_silence_ctx_proj"] = bc_cfg.get(
                "use_silence_ctx_proj", True
            )
        log("info", f"Backchannel module enabled (type={bc_cfg.get('module_type', 'mlp')}).")
    else:
        loaders._lm_kwargs["backchannel_enabled"] = False
        log("info", "Backchannel VAP module disabled.")

    if face_enabled and face_cfg.get("ckpt_path"):
        face_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../moshi/moshi/models/face",
        ))
        loaders._lm_kwargs["face_module_enabled"] = True
        loaders._lm_kwargs["face_module_version"] = int(face_cfg.get("model_version", 1))
        loaders._lm_kwargs["face_module_dir"] = face_dir
        loaders._lm_kwargs["face_module_checkpoint"] = face_cfg.get("ckpt_path")
        loaders._lm_kwargs["face_module_hidden_dim"] = int(face_cfg.get("hidden_dim", 512))
        loaders._lm_kwargs["face_module_layers"] = int(face_cfg.get("layers", 6))
        loaders._lm_kwargs["face_module_heads"] = int(face_cfg.get("heads", 8))
        loaders._lm_kwargs["face_module_code_dim"] = int(face_cfg.get("code_dim", 32))
        loaders._lm_kwargs["face_module_prior_warmup_frames"] = int(
            face_cfg.get("prior_warmup_frames", 10)
        )
    else:
        loaders._lm_kwargs["face_module_enabled"] = False

    lm = loaders.get_moshi_lm(
        moshi_weight,
        device=device,
        dtype=lm_dtype,
        cpu_offload=args.cpu_offload,
    )

    if args.ckpt_dir:
        lm = load_checkpoint(
            lm,
            args.ckpt_dir,
            config.get("lora", {}),
            use_pretrained_depformer=args.use_pretrained_depformer,
        )

    lm.eval()

    # Full-Duplex-Bench consumes output.wav only. Keep lm.face_module loaded when
    # required by the checkpoint, but skip the FaceGenerator wrapper, its token
    # accumulation, the third Mimi model, and motion decoding.
    face_gen = None
    if face_enabled:
        log("info", "Face motion generation disabled for audio-only benchmark evaluation.")

    # ── Build LMGen ───────────────────────────────────────────────────────────
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        use_sampling=True,
        temp=args.temp_audio,
        temp_text=args.temp_text,
        top_k=args.top_k_audio,
        top_k_text=args.top_k_text,
        mimi=mimi,
        bc_context_frames=max(
            1, int(round(float(config.get("duration_sec", 10.0)) * mimi.frame_rate))
        ),
    )
    log_fusion_state(lm)
    if getattr(lm, "backchannel_fusion_trainable", False):
        log(
            "info",
            "Using checkpoint-trained fusion parameters.",
        )

    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    log("info", "Warming up...")
    warmup(mimi, other_mimi, lm_gen, device, frame_size)

    # ── Voice / text prompt ───────────────────────────────────────────────────
    voice_prompt_path = "" if args.no_voice_prompt else "voices/NATM1.pt"
    if voice_prompt_path and not os.path.exists(voice_prompt_path):
        voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=os.path.dirname(voices_tgz))
        voice_prompt_path = os.path.join(os.path.dirname(voices_tgz), "voices/NATM1.pt")

    text_prompt = args.text_prompt if args.text_prompt is not None else config.get("text_prompt", "")
    lm_gen.text_prompt_tokens = (
        text_tokenizer.encode(_wrap_with_system_tags(text_prompt))
        if text_prompt else None
    )

    log("info", "Model ready.")

    # ── Collect files ─────────────────────────────────────────────────────────
    input_files = collect_input_files(data_dir, args.tasks, args.overwrite)
    log("info", f"Found {len(input_files)} sample(s) to process.")

    # ── Main loop ─────────────────────────────────────────────────────────────
    failed = 0

    # already_processed_idx = 600

    with tqdm(input_files, unit="sample") as pbar:
        for i, inp in enumerate(pbar):
            # if i < already_processed_idx:
            #     continue
            out = inp.parent / "output.wav"
            pbar.set_postfix_str(inp.parent.name)
            try:
                pcm, sr, _, _, _ = infer_one(
                    str(inp),
                    mimi, other_mimi, lm_gen,
                    frame_size,
                    voice_prompt_path=voice_prompt_path,
                    face_gen=face_gen,
                    # The benchmark only consumes output.wav. Passing no
                    # tokenizer avoids per-frame text-piece conversion and
                    # accumulation while preserving tokenizer use for prompts.
                    text_tokenizer=None,
                    collect_bc_log=False,
                )
                sphn.write_wav(str(out), pcm, sr)

            except Exception as e:
                import traceback
                log("warning", f"FAILED {inp}: {e}")
                traceback.print_exc()
                failed += 1

    log("info", f"Done: {len(input_files)-failed}/{len(input_files)} samples ({failed} failed)")
    return failed


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
    parser.add_argument(
        "--no-auto-fix-config-paths",
        action="store_true",
        help="Disable automatic rebasing of missing paths copied from another server.",
    )
    parser.add_argument(
        "--use-pretrained-depformer",
        action="store_true",
        help="Keep the pretrained PersonaPlex Depformer instead of loading it from the checkpoint.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-generate even if output.wav exists")
    parser.add_argument("--no-voice-prompt", action="store_true")
    parser.add_argument("--text-prompt", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="auto",
    )
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--temp-audio", type=float, default=0.8)
    parser.add_argument("--top-k-audio", type=int, default=250)
    parser.add_argument("--temp-text", type=float, default=0.7)
    parser.add_argument("--top-k-text", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    sys.exit(1 if run(args) else 0)
