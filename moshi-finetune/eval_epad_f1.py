#!/usr/bin/env python3
"""Compute teacher-forced [EPAD] token metrics on an AMI JSONL split.

The positive class is ``model.end_of_text_padding_id``.  Predictions are the
argmax of the model text head, and only positions enabled by ``text_mask`` are
counted.  Counts are accumulated over the whole split before computing micro
precision, recall, and F1, matching the metric used by ``finetune/eval.py``.
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from eval_bc_event import (
    DEFAULT_REPO,
    MIMI_NAME,
    MOSHI_NAME,
    TEXT_TOKENIZER_NAME,
    InterleavedTokenizer,
    Interleaver,
    TrainArgs,
    _lm_kwargs,
    get_mimi,
    load_model,
    patch_depformer_noop,
    pick_free_gpu,
)
from finetune.data.data_loader import build_data_loader
from finetune.loss import epad_confusion_counts, epad_metrics_from_counts


logger = logging.getLogger("eval_epad_f1")
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate teacher-forced [EPAD] token F1 on the AMI test set."
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Checkpoint directory containing consolidated/lora.safetensors.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Training args.yaml (default: <checkpoint-dir>/../../args.yaml).",
    )
    parser.add_argument(
        "--eval-data",
        default="./data/stereo_ami_balanced_test/data_with_voice_sample.jsonl",
        help="AMI test JSONL.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Stop after this many batches; 0 evaluates the complete split.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device such as cuda:0 (default: GPU with most free memory).",
    )
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def resolve_model_paths(args: TrainArgs) -> tuple[str, str, str]:
    # Hyades configs can contain paths from the machine that launched training.
    # A stale configured path should fall back to the local HF cache/repository.
    mimi_path = args.moshi_paths.mimi_path
    moshi_path = args.moshi_paths.moshi_path
    tokenizer_path = args.moshi_paths.tokenizer_path
    mimi_path = mimi_path if mimi_path and Path(mimi_path).is_file() else None
    moshi_path = moshi_path if moshi_path and Path(moshi_path).is_file() else None
    tokenizer_path = (
        tokenizer_path if tokenizer_path and Path(tokenizer_path).is_file() else None
    )
    if mimi_path and moshi_path and tokenizer_path:
        return mimi_path, moshi_path, tokenizer_path

    from huggingface_hub import hf_hub_download

    repo = args.moshi_paths.hf_repo_id or DEFAULT_REPO
    logger.info("Resolving missing base-model files from %s", repo)
    mimi_path = mimi_path or hf_hub_download(repo, MIMI_NAME)
    moshi_path = moshi_path or hf_hub_download(repo, MOSHI_NAME)
    tokenizer_path = tokenizer_path or hf_hub_download(repo, TEXT_TOKENIZER_NAME)
    return mimi_path, moshi_path, tokenizer_path


def repair_stale_training_paths(args: TrainArgs) -> None:
    """Replace known host-specific VAP paths when local equivalents exist."""
    project_parent = Path(__file__).resolve().parents[1]
    vap_root = project_parent / "VoiceActivityProjection"
    vap_checkpoint = (
        vap_root
        / "example"
        / "VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt"
    )
    if args.backchannel.enable and args.backchannel.module_type == "vap_gpt":
        configured_repo = args.backchannel.vap_gpt_repo_path
        if (
            (not configured_repo or not Path(configured_repo).is_dir())
            and vap_root.is_dir()
        ):
            logger.warning("Using local VAP repository: %s", vap_root)
            args.backchannel.vap_gpt_repo_path = str(vap_root)
        configured_checkpoint = args.backchannel.vap_gpt_checkpoint
        if (
            (not configured_checkpoint or not Path(configured_checkpoint).is_file())
            and vap_checkpoint.is_file()
        ):
            logger.warning("Using local VAP checkpoint: %s", vap_checkpoint)
            args.backchannel.vap_gpt_checkpoint = str(vap_checkpoint)

    vap_manifest = project_parent / "vap_dataset" / "manifest.json"
    configured_manifest = args.data.vap_manifest
    if (
        (not configured_manifest or not Path(configured_manifest).is_file())
        and vap_manifest.is_file()
    ):
        logger.warning("Using local VAP manifest: %s", vap_manifest)
        args.data.vap_manifest = str(vap_manifest)


def make_lm_config(args: TrainArgs) -> dict:
    config = dict(_lm_kwargs)
    if args.moshi_paths.config_path:
        with open(args.moshi_paths.config_path, encoding="utf-8") as stream:
            config = json.load(stream)
    config["dep_q"] = 16

    bc = args.backchannel
    if bc.enable:
        config.update(
            backchannel_enabled=True,
            backchannel_vap_dim=bc.vap_dim,
            backchannel_bc_hidden=bc.bc_hidden,
            backchannel_gumbel_temp_init=bc.gumbel_temp_init,
            backchannel_gumbel_temp_min=bc.gumbel_temp_min,
            backchannel_gumbel_anneal_rate=bc.gumbel_anneal_rate,
            backchannel_module_type=bc.module_type,
        )
        if bc.pad_token_id is not None:
            config["backchannel_pad_token_id"] = bc.pad_token_id
        if bc.epad_token_id is not None:
            config["backchannel_epad_token_id"] = bc.epad_token_id
        if bc.module_type == "vap_gpt":
            config.update(
                backchannel_vap_repo_path=bc.vap_gpt_repo_path,
                backchannel_vap_checkpoint=bc.vap_gpt_checkpoint,
                backchannel_vap_channel_layers=bc.vap_gpt_channel_layers,
                backchannel_vap_cross_layers=bc.vap_gpt_cross_layers,
                backchannel_vap_num_heads=bc.vap_gpt_num_heads,
                backchannel_vap_dropout=bc.vap_gpt_dropout,
            )

    # Neither face output nor Depformer audio logits are needed for EPAD.
    config["face_module_enabled"] = False
    return config


@torch.inference_mode()
def evaluate(
    model,
    data_loader,
    args: TrainArgs,
    mimi,
    prompt_ids: list[int],
    param_dtype: torch.dtype,
    device: str,
    max_batches: int,
) -> tuple[torch.Tensor, int]:
    counts = torch.zeros(4, dtype=torch.float64)
    prompt_len = len(prompt_ids)
    num_batches = 0

    for batch in data_loader:
        if max_batches > 0 and num_batches >= max_batches:
            break

        codes = batch.codes.to(device, non_blocking=True)
        if prompt_len:
            prefix = torch.full(
                (codes.shape[0], codes.shape[1], prompt_len),
                model.zero_token_id,
                dtype=codes.dtype,
                device=device,
            )
            prefix[:, 0] = torch.as_tensor(prompt_ids, device=device)
            codes_in = torch.cat((prefix, codes), dim=2)
        else:
            codes_in = codes

        voice_prompt_embs = batch.voice_prompt_embs
        if voice_prompt_embs is not None:
            voice_prompt_embs = voice_prompt_embs.to(device, non_blocking=True)

        # The backchannel module participates in the forward graph that produced
        # this checkpoint.  Supply the same Mimi features as training/eval even
        # though only text_logits are scored here.
        bc_audio_feats = None
        if args.backchannel.enable and args.backchannel.module_type == "vap_gpt":
            agent = mimi.decode_latent(codes[:, 1:9].clamp(min=0)).transpose(1, 2)
            user = mimi.decode_latent(codes[:, 9:17].clamp(min=0)).transpose(1, 2)
            agent = agent.to(dtype=param_dtype)
            user = user.to(dtype=param_dtype)
            if prompt_len:
                zeros = torch.zeros(
                    codes.shape[0],
                    prompt_len,
                    agent.shape[-1],
                    dtype=param_dtype,
                    device=device,
                )
                agent = torch.cat((zeros, agent), dim=1)
                user = torch.cat((zeros, user), dim=1)
            bc_audio_feats = (agent, user)

        output = model(
            codes_in,
            step=0,
            voice_prompt_embs=voice_prompt_embs,
            audio_feat=None,
            gt_face_motion=None,
            mimi=None,
            bc_audio_feats=bc_audio_feats,
        )
        batch_counts = epad_confusion_counts(
            output.text_logits[:, :, prompt_len:],
            codes[:, : model.audio_offset],
            output.text_mask[:, :, prompt_len:],
            model.end_of_text_padding_id,
        )
        counts += batch_counts.double().cpu()
        num_batches += 1
        if num_batches % 10 == 0:
            logger.info(
                "batch %d%s — valid positions: %d",
                num_batches,
                f"/{max_batches}" if max_batches else "",
                int(counts.sum().item()),
            )

    if num_batches == 0:
        raise RuntimeError(
            "The data loader yielded no full batch. Reduce --batch-size or check --eval-data."
        )
    return counts, num_batches


def main() -> None:
    cli = parse_args()
    checkpoint_dir = Path(cli.checkpoint_dir).resolve()
    config_path = (
        Path(cli.config).resolve()
        if cli.config
        else (checkpoint_dir / ".." / ".." / "args.yaml").resolve()
    )
    eval_data = Path(cli.eval_data).resolve()
    for label, path in (
        ("checkpoint", checkpoint_dir),
        ("config", config_path),
        ("eval data", eval_data),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    args = TrainArgs.load(str(config_path), drop_extra_fields=False)
    args.data.eval_data = str(eval_data)
    # Evaluation should be deterministic and must not reshuffle the AMI split.
    args.data.shuffle = False
    repair_stale_training_paths(args)

    device = cli.device or pick_free_gpu()
    if not device.startswith("cuda"):
        raise ValueError("This data pipeline requires a CUDA device.")
    torch.cuda.set_device(device)
    param_dtype = torch.bfloat16 if args.param_dtype == "bfloat16" else torch.float32

    mimi_path, moshi_path, tokenizer_path = resolve_model_paths(args)
    logger.info("Loading Mimi tokenizer on %s", device)
    mimi = get_mimi(mimi_path, device=device)
    mimi.eval()
    for parameter in mimi.parameters():
        parameter.requires_grad = False

    logger.info("Loading checkpoint: %s", checkpoint_dir)
    model = load_model(
        checkpoint_dir,
        args,
        make_lm_config(args),
        moshi_path,
        param_dtype,
        device,
    )
    patch_depformer_noop(model)

    import sentencepiece

    spm = sentencepiece.SentencePieceProcessor()
    if not spm.Load(tokenizer_path):
        raise RuntimeError(f"Could not load tokenizer: {tokenizer_path}")
    prompt_ids = spm.encode(args.text_prompt) if args.text_prompt else []

    interleaver = Interleaver(
        spm,
        mimi.frame_rate,
        model.text_padding_token_id,
        model.end_of_text_padding_id,
        model.zero_token_id,
        keep_main_only=True,
    )
    tokenizer = InterleavedTokenizer(
        mimi,
        interleaver,
        duration_sec=args.duration_sec,
        vap_manifest_path=args.data.vap_manifest,
        flame_root="",
        flame_speaker=args.face_gen.flame_speaker,
    )
    data_loader = build_data_loader(
        instruct_tokenizer=tokenizer,
        args=args.data,
        batch_size=cli.batch_size,
        seed=0,
        rank=0,
        world_size=1,
        is_eval=True,
    )

    counts, num_batches = evaluate(
        model,
        data_loader,
        args,
        mimi,
        prompt_ids,
        param_dtype,
        device,
        cli.max_batches,
    )
    metrics = epad_metrics_from_counts(counts, prefix="epad")
    tp, fp, fn, tn = (int(value) for value in counts.tolist())
    result = {
        "checkpoint": str(checkpoint_dir),
        "eval_data": str(eval_data),
        "batch_size": cli.batch_size,
        "num_batches": num_batches,
        "epad_token_id": int(model.end_of_text_padding_id),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        **metrics,
    }

    print("\n" + "=" * 62)
    print("  AMI test — teacher-forced [EPAD] token metrics")
    print("=" * 62)
    print(f"  batches               : {num_batches}")
    print(f"  valid token positions : {tp + fp + fn + tn:,}")
    print(f"  TP / FP / FN / TN     : {tp:,} / {fp:,} / {fn:,} / {tn:,}")
    print(f"  precision             : {metrics['epad_precision']:.6f}")
    print(f"  recall                : {metrics['epad_recall']:.6f}")
    print(f"  F1                    : {metrics['epad_f1']:.6f}")
    print(f"  accuracy              : {metrics['epad_acc']:.6f}")
    print("=" * 62)

    if cli.output:
        output_path = Path(cli.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Saved metrics to %s", output_path.resolve())


if __name__ == "__main__":
    main()
