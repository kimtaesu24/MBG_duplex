#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def frame_to_motion(item: dict, jaw_dim: int = 1) -> np.ndarray:
    expr = np.asarray(item["expr"], dtype=np.float32)
    if expr.shape[0] < 50:
        expr = np.pad(expr, (0, 50 - expr.shape[0]))
    expr = expr[:50]
    jaw = np.asarray(item["jaw_pose"], dtype=np.float32)[:3]
    if jaw_dim == 1:
        jaw = jaw[:1]
    elif jaw_dim != 3:
        raise ValueError(f"jaw_dim must be 1 or 3, got {jaw_dim}")
    neck = np.asarray(item["neck_pose"], dtype=np.float32)[:3]
    return np.concatenate([expr, jaw, neck], axis=0).astype(np.float32)


def load_motion(path: Path, jaw_dim: int = 1) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if arr.dtype == object:
        return np.stack([frame_to_motion(x, jaw_dim=jaw_dim) for x in arr], axis=0)
    arr = np.asarray(arr, dtype=np.float32)
    expected_dim = 50 + jaw_dim + 3
    if arr.shape[-1] == 56 and jaw_dim == 1:
        arr = np.concatenate([arr[..., :50], arr[..., 50:51], arr[..., 53:56]], axis=-1)
    elif arr.shape[-1] > expected_dim:
        arr = arr[..., :expected_dim]
    if arr.ndim != 2 or arr.shape[-1] != expected_dim:
        raise ValueError(f"Expected [T, {expected_dim}] from {path}, got {arr.shape}")
    return arr


def iter_paths(args):
    if args.dualtalk_root:
        for split in args.dualtalk_splits:
            for path in sorted((Path(args.dualtalk_root) / split).glob("*.npy")):
                if path.name.endswith(".fbank.npy"):
                    continue
                yield path
    if args.candor_root:
        for split in args.candor_splits:
            for path in sorted((Path(args.candor_root) / split).glob("*.npy")):
                if path.name.endswith(".fbank.npy"):
                    continue
                yield path
    for channel, motion_root_arg, mimi_root_arg, splits_arg in (
        ("bc", args.ami_bc_root, args.ami_bc_mimi_root, args.ami_bc_splits),
        ("ut", args.ami_ut_root, args.ami_ut_mimi_root, args.ami_ut_splits),
    ):
        if not motion_root_arg or not mimi_root_arg:
            continue
        motion_root = Path(motion_root_arg)
        mimi_root = Path(mimi_root_arg)
        motion_splits = {"train": "train", "val": "valid", "test": "test"}
        mimi_splits = {"train": f"ami_train_{channel}", "val": f"ami_val_{channel}", "test": f"ami_test_{channel}"}
        for split in splits_arg:
            motion_dir = motion_root / motion_splits.get(split, split)
            mimi_dir = mimi_root / mimi_splits.get(split, split)
            if not motion_dir.exists() or not mimi_dir.exists():
                continue
            for mimi_path in sorted(mimi_dir.glob("*_latent.pt")):
                stem = mimi_path.name[: -len("_latent.pt")]
                motion_stem = f"{stem}_bc" if channel == "bc" and not stem.endswith("_bc") else stem
                motion_path = motion_dir / f"{motion_stem}.npy"
                if motion_path.exists():
                    yield motion_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dualtalk-root", default="/home6/duplex/dataset/dualtalk")
    parser.add_argument("--dualtalk-splits", nargs="+", default=["train"])
    # parser.add_argument("--candor-root", default="/home6/duplex/dataset/candor_flame")
    # parser.add_argument("--candor-splits", nargs="+", default=["ut", "bc"])
    parser.add_argument("--ami-bc-root", default="/home6/duplex/dataset/ami_flame/bc")
    parser.add_argument("--ami-bc-mimi-root", default="/home6/duplex/dataset/mimi_emb")
    parser.add_argument("--ami-bc-splits", nargs="+", default=["train"])
    parser.add_argument("--ami-ut-root", default="/home6/duplex/dataset/ami_flame/ut")
    parser.add_argument("--ami-ut-mimi-root", default="/home6/duplex/dataset/mimi_emb")
    parser.add_argument("--ami-ut-splits", nargs="+", default=["train"])
    parser.add_argument("--output", default="/home6/duplex/dataset/artalk_mimi54_stats.json")
    parser.add_argument("--jaw-dim", type=int, default=1, choices=[1, 3])
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    paths = list(iter_paths(args))
    if args.max_files > 0:
        paths = paths[: args.max_files]
    if not paths:
        raise RuntimeError("No npy files found for stats.")

    count = 0
    motion_dim = 50 + args.jaw_dim + 3
    sum_v = np.zeros(motion_dim, dtype=np.float64)
    sum_sq = np.zeros(motion_dim, dtype=np.float64)
    for path in tqdm(paths, desc="mimi54-stats"):
        motion = load_motion(path, jaw_dim=args.jaw_dim).astype(np.float64)
        count += motion.shape[0]
        sum_v += motion.sum(axis=0)
        sum_sq += np.square(motion).sum(axis=0)
    mean = sum_v / max(count, 1)
    var = np.maximum(sum_sq / max(count, 1) - np.square(mean), 1e-8)
    std = np.sqrt(var)
    out = {
        "motion_mean": mean.astype(float).tolist(),
        "motion_std": std.astype(float).tolist(),
        "num_files": len(paths),
        "num_frames": int(count),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))
    print(f"[mimi54-stats] wrote {output} files={len(paths)} frames={count}")


if __name__ == "__main__":
    main()
