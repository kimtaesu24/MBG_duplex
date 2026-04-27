#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from core.data.mimi_flame_data import MimiFlameData
from softvq_continuous_online_train import (
    CausalSoftVQContinuousTransformer,
    cleanup_ddp,
    detect_ddp,
    is_main,
    make_dataset_cfg,
    motion54_to_56,
    move_batch,
)


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default="/home6/duplex/personaplex/moshi/moshi/ARTalk/train_code/outputs/SoftVQContinuousOnline_mimi54_dualtalk_ami_b512/checkpoint_epoch_500.ckpt",
    )
    p.add_argument("--output-dir", default="/home6/duplex/inference")
    p.add_argument("--dualtalk-root", default="/home6/duplex/dataset/dualtalk")
    p.add_argument("--mimi-root", default="/home6/duplex/dataset/mimi_emb")
    p.add_argument("--ami-bc-root", default="/home6/duplex/dataset/ami_flame/bc")
    p.add_argument("--ami-bc-mimi-root", default="/home6/duplex/dataset/mimi_emb")
    p.add_argument("--ami-ut-root", default="/home6/duplex/dataset/ami_flame/ut")
    p.add_argument("--ami-ut-mimi-root", default="/home6/duplex/dataset/mimi_emb")
    p.add_argument("--sources", default="ami")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--clip-length", type=int, default=100)
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--length-mismatch-csv", default="/home6/duplex/inference/length_mismatches.csv")
    p.add_argument("--stream-context-frames", type=int, default=25)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--save-gt", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    ddp, rank, _, world_size, device = detect_ddp()
    torch.set_grad_enabled(False)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    ckpt_args = ckpt.get("args", {})
    model = CausalSoftVQContinuousTransformer(
        hidden_dim=int(ckpt_args.get("hidden_dim", 512)),
        layers=int(ckpt_args.get("layers", 6)),
        heads=int(ckpt_args.get("heads", 8)),
        code_dim=int(ckpt_args.get("code_dim", 32)),
        prior_warmup_frames=int(ckpt_args.get("prior_warmup_frames", 10)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    dataset = MimiFlameData(make_dataset_cfg(args), args.split, debug=False)
    indices = list(range(rank, len(dataset), world_size)) if ddp else list(range(len(dataset)))
    if args.max_samples > 0:
        indices = indices[: args.max_samples]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    out_dir = Path(args.output_dir)
    pred_dir = out_dir / "pred"
    gt_dir = out_dir / "gt"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if args.save_gt:
        gt_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"manifest_rank{rank:02d}.jsonl"

    if is_main(rank):
        print(
            f"[infer] ckpt={args.ckpt} split={args.split} sources={args.sources} "
            f"samples={len(dataset)} world_size={world_size} context={args.stream_context_frames}"
        )

    with manifest_path.open("w") as manifest:
        for local_idx, batch in enumerate(
            tqdm(loader, desc=f"infer r{rank}", disable=not is_main(rank), dynamic_ncols=True)
        ):
            batch = move_batch(batch, device)
            gt = batch["motion_code"]
            audio_feat = batch["audio_feat"]
            llm_feat = batch.get("llm_feat")
            pred = model.generate(
                audio_feat,
                max_frames=gt.shape[1],
                llm_feat=llm_feat,
                max_context_frames=args.stream_context_frames,
            )

            info = batch.get("infos", {})
            key = f"rank{rank:02d}_sample_{local_idx:04d}"
            source = ""
            if isinstance(info, dict):
                if "motion_key" in info:
                    value = info["motion_key"]
                    if isinstance(value, (list, tuple)):
                        value = value[0]
                    key = str(value)
                if "source" in info:
                    value = info["source"]
                    if isinstance(value, (list, tuple)):
                        value = value[0]
                    source = str(value)
            name = safe_name(f"{source}_{key}" if source else key)
            pred_path = pred_dir / f"{name}_pred.npy"
            np.save(pred_path, motion54_to_56(pred[0]).cpu().numpy())
            gt_path = None
            if args.save_gt:
                gt_path = gt_dir / f"{name}_gt.npy"
                np.save(gt_path, motion54_to_56(gt[0]).cpu().numpy())

            manifest.write(
                json.dumps(
                    {
                        "rank": rank,
                        "local_idx": local_idx,
                        "dataset_idx": indices[local_idx],
                        "source": source,
                        "key": key,
                        "frames": int(gt.shape[1]),
                        "pred_path": str(pred_path),
                        "gt_path": str(gt_path) if gt_path is not None else None,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            manifest.flush()

    if ddp:
        dist.barrier()
    if is_main(rank):
        with (out_dir / "manifest.jsonl").open("w") as merged:
            for path in sorted(out_dir.glob("manifest_rank*.jsonl")):
                merged.write(path.read_text())
        print(f"[infer] saved to {out_dir}")
    cleanup_ddp(ddp)


if __name__ == "__main__":
    main()
