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
        [--suppress-epad] \\
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
from moshi.models import LMGen
from moshi.offline import warmup

# Shared helpers and the canonical per-sample inference function live in test_inference.
from test_inference import log, _strip_peft_prefixes, load_checkpoint, infer_one


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
        mimi=mimi,
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

    text_prompt = config.get("text_prompt", "")
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
                pcm, sr, _, _, _ = infer_one(
                    str(inp),
                    mimi, other_mimi, lm_gen,
                    frame_size,
                    voice_prompt_path=voice_prompt_path,
                    face_gen=face_gen,
                    suppress_epad=args.suppress_epad,
                    collect_bc_log=False,
                )
                sphn.write_wav(str(out), pcm, sr)

                if face_gen is not None and getattr(face_gen, "_agent_tokens", None):
                    try:
                        motion = face_gen.generate_numpy(vap_mimi)
                        face_out = inp.parent / "face_motion.npy"
                        np.save(str(face_out), motion)
                    except Exception as _e:
                        import traceback as _tb
                        log("warning", f"Face generation failed for {inp.name}: {_e}")
                        _tb.print_exc()

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
    parser.add_argument("--suppress-epad", action="store_true",
                        help="Force [EPAD] → [PAD] whenever g_final=0 (VAP says don't speak). "
                             "Word tokens in progress are never replaced.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    run(args)
