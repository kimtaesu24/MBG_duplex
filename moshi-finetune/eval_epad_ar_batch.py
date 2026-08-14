#!/usr/bin/env python3
"""Single-GPU batched autoregressive [EPAD] evaluation for the AMI test set."""

import argparse
from collections import defaultdict, deque
import contextlib
import gc
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import sentencepiece
import torch
import yaml
from huggingface_hub import hf_hub_download
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../moshi")))

import moshi.models.loaders as loaders
from finetune.data.interleaver import InterleavedTokenizer, Interleaver
from moshi.models.lm import load_audio as lm_load_audio
from inference import (
    LMGen,
    _repair_config_paths,
    _wrap_with_system_tags,
    list_jsonl,
    load_checkpoint,
    log,
    fusion_state,
    log_fusion_state,
    set_seed,
)


def select_user_audio(audio: np.ndarray) -> np.ndarray:
    """Select user channel without emitting one log line per file."""
    if audio.ndim != 2:
        return audio[np.newaxis]
    if audio.shape[0] != 2:
        return audio[0:1]
    return audio[1:2]


def parse_args():
    parser = argparse.ArgumentParser("Batched autoregressive AMI EPAD evaluation")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Fine-tuned checkpoint directory (omit with --base-model-only).",
    )
    parser.add_argument(
        "--base-model-only",
        action="store_true",
        help="Evaluate pure nvidia/personaplex-7b-v1 without fine-tuned weights.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--test-jsonl",
        default="./data/stereo_ami_balanced_test/data_with_voice_sample_amplified.jsonl",
    )
    parser.add_argument("--output", default="./result/epad_ar_batch/metrics.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--no-voice-prompt", action="store_true")
    parser.add_argument("--text-prompt", default=None)
    parser.add_argument("--temp-audio", type=float, default=0.8)
    parser.add_argument("--top-k-audio", type=int, default=250)
    parser.add_argument("--temp-text", type=float, default=0.7)
    parser.add_argument("--top-k-text", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def configure_backchannel(config: dict) -> None:
    for key in list(loaders._lm_kwargs):
        if key.startswith("backchannel_"):
            del loaders._lm_kwargs[key]
    bc = config.get("backchannel", {})
    loaders._lm_kwargs["backchannel_enabled"] = bool(bc.get("enable", False))
    if not bc.get("enable", False):
        return
    loaders._lm_kwargs.update(
        backchannel_module_type=bc.get("module_type", "mlp"),
        backchannel_vap_dim=bc.get("vap_dim", 256),
        backchannel_bc_hidden=bc.get("bc_hidden", 512),
        backchannel_gumbel_temp_init=bc.get("gumbel_temp_init", 1.0),
        backchannel_gumbel_temp_min=bc.get("gumbel_temp_min", 0.5),
        backchannel_gumbel_anneal_rate=bc.get("gumbel_anneal_rate", 0.0001),
        backchannel_pad_token_id=bc.get("pad_token_id", 3),
        backchannel_epad_token_id=bc.get("epad_token_id", 0),
        backchannel_fusion_trainable=bc.get("fusion_trainable", False),
        backchannel_fusion_bc_init=bc.get("fusion_bc_init", 0.0),
        backchannel_fusion_vap_init=bc.get("fusion_vap_init", 0.0),
        backchannel_fusion_vad_init=bc.get("fusion_vad_init", 0.0),
        backchannel_fusion_bias_init=bc.get("fusion_bias_init", 0.0),
    )
    if bc.get("module_type") == "vap_gpt":
        loaders._lm_kwargs.update(
            backchannel_vap_repo_path=bc.get("vap_gpt_repo_path", ""),
            backchannel_vap_checkpoint=bc.get("vap_gpt_checkpoint"),
            backchannel_vap_channel_layers=bc.get("vap_gpt_channel_layers", 1),
            backchannel_vap_cross_layers=bc.get("vap_gpt_cross_layers", 3),
            backchannel_vap_num_heads=bc.get("vap_gpt_num_heads", 4),
            backchannel_vap_dropout=bc.get("vap_gpt_dropout", 0.1),
            backchannel_vap_use_silence_ctx_proj=bc.get(
                "use_silence_ctx_proj", True
            ),
        )


def resolve_weight(config: dict, key: str, hub_name: str) -> str:
    path_config = config.get("moshi_paths", {})
    path = path_config.get(key)
    if path and Path(path).is_file():
        return path
    repo = path_config.get("hf_repo_id") or loaders.DEFAULT_REPO
    return hf_hub_download(repo, hub_name)


def load_runtime(args):
    checkpoint = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None
    if args.config:
        config_path = Path(args.config).resolve()
    elif checkpoint is not None:
        config_path = (checkpoint / ".." / ".." / "args.yaml").resolve()
    else:
        config_path = None

    if config_path is not None:
        with config_path.open() as stream:
            config = yaml.safe_load(stream)
        _repair_config_paths(config)
    else:
        config = {
            "moshi_paths": {"hf_repo_id": loaders.DEFAULT_REPO},
            "backchannel": {"enable": False},
            "face_gen": {"enable": False},
            "duration_sec": 10.0,
            "text_prompt": "",
        }

    config.setdefault("face_gen", {})["enable"] = False
    if args.base_model_only:
        config.setdefault("backchannel", {})["enable"] = False

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    mimi_path = resolve_weight(config, "mimi_path", loaders.MIMI_NAME)
    tokenizer_path = resolve_weight(config, "tokenizer_path", loaders.TEXT_TOKENIZER_NAME)
    moshi_path = resolve_weight(config, "moshi_path", loaders.MOSHI_NAME)

    mimi = loaders.get_mimi(mimi_path, device)
    tokenizer_mimi = loaders.get_mimi(mimi_path, device)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
    configure_backchannel(config)
    loaders._lm_kwargs["face_module_enabled"] = False
    lm = loaders.get_moshi_lm(moshi_path, device=device, dtype=dtype)
    if not args.base_model_only:
        if checkpoint is None:
            raise ValueError(
                "--checkpoint-dir is required unless --base-model-only is set."
            )
        lm = load_checkpoint(lm, str(checkpoint), config.get("lora", {}))
    lm.eval()

    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        frame_rate=mimi.frame_rate,
        device=device,
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
    if not args.base_model_only and getattr(lm, "backchannel_fusion_trainable", False):
        log(
            "info",
            "Using checkpoint-trained fusion parameters.",
        )
    text_prompt = args.text_prompt if args.text_prompt is not None else config.get("text_prompt", "")
    lm_gen.text_prompt_tokens = (
        text_tokenizer.encode(_wrap_with_system_tags(text_prompt))
        if text_prompt
        else []
    )
    interleaver = Interleaver(
        text_tokenizer,
        mimi.frame_rate,
        lm.text_padding_token_id,
        lm.end_of_text_padding_id,
        lm.zero_token_id,
        keep_main_only=True,
    )
    interleaved_tokenizer = InterleavedTokenizer(
        tokenizer_mimi,
        interleaver,
        duration_sec=config.get("duration_sec", 10.0),
        vap_manifest_path=config.get("data", {}).get("vap_manifest", ""),
        flame_root="",
        flame_speaker="auto",
    )
    return config, mimi, tokenizer_mimi, lm_gen, interleaved_tokenizer


def build_compatible_batches(samples, batch_size: int, no_voice_prompt: bool):
    """Group by prompt length, then bucket by audio length to limit padding.

    Voice-prompt embeddings cannot be padded safely: every extra embedding would
    become real autoregressive context. Exact-length grouping preserves the
    per-sample prompt while still giving full batches for the dominant group.
    """
    groups = defaultdict(list)
    for sample in samples:
        if no_voice_prompt:
            prompt_length = 0
        else:
            voice_sample = sample.get("voice_sample")
            prompt_path = (
                Path(os.path.splitext(voice_sample)[0] + ".pt")
                if voice_sample
                else None
            )
            if prompt_path is None or not prompt_path.is_file():
                raise FileNotFoundError(
                    f"Voice-prompt cache missing for {sample['path']}: {prompt_path}. "
                    "Generate it first or pass --no-voice-prompt."
                )
            state = torch.load(prompt_path, map_location="cpu", weights_only=True)
            prompt_length = int(state["embeddings"].shape[0])
        groups[prompt_length].append(sample)

    batches = []
    for prompt_length, group in sorted(
        groups.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        # Adjacent samples have similar lengths, reducing zero-padding and making
        # a larger requested batch substantially more memory-efficient.
        group.sort(key=lambda sample: float(sample.get("duration", 0.0)))
        for start in range(0, len(group), batch_size):
            batches.append((prompt_length, group[start : start + batch_size]))
    return batches, {length: len(group) for length, group in groups.items()}


def recover_from_cuda_oom(lm_gen) -> None:
    """Release failed streaming/CUDA-graph state before retrying."""
    try:
        lm_gen._stop_streaming()
    except Exception:
        # An OOM can happen halfway through streaming-state construction.
        pass
    lm_gen._reset_bc_history()
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def evaluate_batch(samples, jsonl_dir, mimi, tokenizer_mimi, lm_gen, tokenizer, no_voice):
    paths, stereo_wavs, user_wavs, prompt_paths = [], [], [], []
    for sample in samples:
        path = Path(sample["path"])
        if not path.is_absolute():
            path = jsonl_dir / path
        path = path.resolve()
        wav = lm_load_audio(str(path), mimi.sample_rate)
        paths.append(path)
        stereo_wavs.append(wav)
        user_wavs.append(select_user_audio(wav))
        voice_sample = sample.get("voice_sample")
        prompt_path = Path(os.path.splitext(voice_sample)[0] + ".pt") if voice_sample else None
        if not no_voice and (prompt_path is None or not prompt_path.is_file()):
            raise FileNotFoundError(
                f"Voice-prompt cache missing for {path}: {prompt_path}. "
                "Generate it first or pass --no-voice-prompt."
            )
        prompt_paths.append(prompt_path)

    batch_size = len(samples)
    max_audio_samples = max(wav.shape[-1] for wav in user_wavs)
    users = np.stack(
        [np.pad(wav[0], (0, max_audio_samples - wav.shape[-1])) for wav in user_wavs]
    )
    device = lm_gen.lm_model.device
    user_codes = tokenizer_mimi.encode(
        torch.as_tensor(users, dtype=torch.float32, device=device).unsqueeze(1)
    )

    channels = stereo_wavs[0].shape[0]
    max_stereo_samples = max(wav.shape[-1] for wav in stereo_wavs)
    stereo = np.stack(
        [
            np.pad(wav, ((0, 0), (0, max_stereo_samples - wav.shape[-1])))
            for wav in stereo_wavs
        ]
    )
    stereo_tensor = torch.as_tensor(stereo, dtype=torch.float32, device=device)
    stereo_codes = tokenizer_mimi.encode(
        stereo_tensor.reshape(batch_size * channels, 1, max_stereo_samples)
    ).view(batch_size, channels, 8, -1)
    gt_samples = [
        tokenizer.tokenize_with_encoded_audio(
            stereo_codes[index],
            0.0,
            str(paths[index]),
            actual_wav_samples=stereo_wavs[index].shape[-1],
        )
        for index in range(batch_size)
    ]

    lm_gen.streaming_forever(batch_size)
    lm_gen._reset_bc_history()
    if no_voice:
        lm_gen.voice_prompt = None
        lm_gen.voice_prompt_audio = None
        lm_gen.voice_prompt_embeddings = None
        lm_gen.voice_prompt_cache = None
    else:
        states = [
            torch.load(path, map_location="cpu", weights_only=True)
            for path in prompt_paths
        ]
        lengths = {state["embeddings"].shape[0] for state in states}
        if len(lengths) != 1:
            raise ValueError(
                "Voice-prompt lengths differ within a batch; reduce --batch-size."
            )
        lm_gen.voice_prompt = "<batch>"
        lm_gen.voice_prompt_audio = None
        lm_gen.voice_prompt_embeddings = torch.cat(
            [state["embeddings"] for state in states], dim=1
        ).to(device)
        lm_gen.voice_prompt_cache = torch.cat(
            [state["cache"] for state in states], dim=0
        ).to(device)
    # LMGen prints four "Done loading ..." messages per batch. Keep the terminal
    # dedicated to the progress bar during metric evaluation.
    with contextlib.redirect_stdout(io.StringIO()):
        lm_gen.step_system_prompts(mimi)

    predictions = [[] for _ in samples]
    for frame in range(user_codes.shape[-1]):
        result = lm_gen.step(user_codes[:, :, frame : frame + 1], return_z=True)
        if result is None or result[0] is None:
            continue
        for index, token_id in enumerate(result[0][:, 0, 0].cpu().tolist()):
            predictions[index].append(token_id)

    counts = [0, 0, 0, 0]
    details = []
    for path, prediction, gt_sample in zip(paths, predictions, gt_samples):
        gt = gt_sample.codes[0, 0].cpu()
        pred = torch.full_like(gt, lm_gen.lm_model.text_padding_token_id)
        copied = min(len(prediction), len(gt))
        if copied:
            pred[:copied] = torch.tensor(prediction[:copied], dtype=gt.dtype)
        valid = gt_sample.valid_mask.cpu().bool()
        pred_epad = pred == lm_gen.lm_model.end_of_text_padding_id
        gt_epad = gt == lm_gen.lm_model.end_of_text_padding_id
        file_counts = [
            int((valid & pred_epad & gt_epad).sum()),
            int((valid & pred_epad & ~gt_epad).sum()),
            int((valid & ~pred_epad & gt_epad).sum()),
            int((valid & ~pred_epad & ~gt_epad).sum()),
        ]
        counts = [left + right for left, right in zip(counts, file_counts)]
        details.append(
            {
                "path": str(path),
                "generated_frames": len(prediction),
                "valid_gt_frames": int(valid.sum()),
                **dict(zip(("tp", "fp", "fn", "tn"), file_counts)),
            }
        )
    return counts, details


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not args.base_model_only and not args.checkpoint_dir:
        raise ValueError(
            "Choose --checkpoint-dir for fine-tuned evaluation or "
            "--base-model-only for pure PersonaPlex."
        )
    set_seed(args.seed)
    config, mimi, tokenizer_mimi, lm_gen, tokenizer = load_runtime(args)
    samples = list_jsonl(args.test_jsonl)
    jsonl_dir = Path(args.test_jsonl).resolve().parent
    batches, prompt_groups = build_compatible_batches(
        samples, args.batch_size, args.no_voice_prompt
    )
    largest_prompt_groups = sorted(
        prompt_groups.items(), key=lambda item: -item[1]
    )[:5]
    log(
        "info",
        f"Prepared {len(batches)} batches from {len(samples)} files; "
        f"requested batch_size={args.batch_size}, "
        f"prompt_group_count={len(prompt_groups)}, largest={largest_prompt_groups}",
    )
    totals = [0, 0, 0, 0]
    details = []
    skipped = []
    work_queue = deque(batches)
    with tqdm(total=len(samples), unit="file", desc="AR EPAD evaluation") as progress:
        while work_queue:
            prompt_length, sample_batch = work_queue.popleft()
            try:
                batch_counts, batch_details = evaluate_batch(
                    sample_batch,
                    jsonl_dir,
                    mimi,
                    tokenizer_mimi,
                    lm_gen,
                    tokenizer,
                    args.no_voice_prompt,
                )
            except torch.OutOfMemoryError as error:
                recover_from_cuda_oom(lm_gen)
                if len(sample_batch) > 1:
                    midpoint = len(sample_batch) // 2
                    # appendleft in reverse order so the first half runs first.
                    work_queue.appendleft((prompt_length, sample_batch[midpoint:]))
                    work_queue.appendleft((prompt_length, sample_batch[:midpoint]))
                    progress.set_postfix(
                        status=f"OOM: {len(sample_batch)} -> "
                        f"{midpoint}+{len(sample_batch) - midpoint}",
                        skipped=len(skipped),
                    )
                    continue

                skipped.append(
                    {
                        "path": sample_batch[0]["path"],
                        "reason": "CUDA out of memory at batch_size=1",
                        "error": str(error),
                    }
                )
                progress.update(1)
                progress.set_postfix(
                    status="OOM: skipped 1 file",
                    skipped=len(skipped),
                )
                continue

            totals = [left + right for left, right in zip(totals, batch_counts)]
            details.extend(batch_details)
            tp_now, fp_now, fn_now, _ = totals
            precision_now = tp_now / (tp_now + fp_now) if tp_now + fp_now else 0.0
            recall_now = tp_now / (tp_now + fn_now) if tp_now + fn_now else 0.0
            f1_now = (
                2 * precision_now * recall_now / (precision_now + recall_now)
                if precision_now + recall_now
                else 0.0
            )
            progress.update(len(sample_batch))
            progress.set_postfix(
                batch=len(sample_batch),
                prompt=prompt_length,
                P=f"{precision_now:.4f}",
                R=f"{recall_now:.4f}",
                F1=f"{f1_now:.4f}",
                skipped=len(skipped),
            )

    tp, fp, fn, tn = totals
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = sum(totals)
    result = {
        "model": (
            "nvidia/personaplex-7b-v1 (pure base)"
            if args.base_model_only
            else str(Path(args.checkpoint_dir).resolve())
        ),
        "base_model_only": args.base_model_only,
        "test_jsonl": str(Path(args.test_jsonl).resolve()),
        "batch_size": args.batch_size,
        "num_files": len(details),
        "num_skipped": len(skipped),
        "num_frames": total,
        "fusion": fusion_state(lm_gen.lm_model),
        **dict(zip(("tp", "fp", "fn", "tn"), totals)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / total if total else 0.0,
        "files": details,
        "skipped": skipped,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("\n" + "=" * 60)
    print("AMI test — Batched Autoregressive EPAD Evaluation")
    print("=" * 60)
    print(f"Model           : {result['model']}")
    print(f"Files evaluated : {len(details):,}")
    print(f"Files skipped   : {len(skipped):,}")
    print(f"Frames evaluated: {total:,}")
    print(f"TP / FP / FN / TN: {tp:,} / {fp:,} / {fn:,} / {tn:,}")
    print(f"Precision       : {precision:.6f}")
    print(f"Recall          : {recall:.6f}")
    print(f"EPAD F1 score   : {f1:.6f}")
    print(f"Accuracy        : {result['accuracy']:.6f}")
    print(f"Result JSON     : {output.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
