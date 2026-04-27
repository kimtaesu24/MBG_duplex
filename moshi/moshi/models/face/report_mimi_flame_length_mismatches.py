#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def npy_len(path: Path) -> int:
    with path.open("rb") as f:
        version = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format._read_array_header(f, version)
    return int(shape[0])


def mimi_len(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    features = payload["audio_features"] if isinstance(payload, dict) else payload
    if features.dim() == 3 and features.shape[0] == 1:
        features = features[0]
    return int(features.shape[0])


def iter_source(source: str, split: str, data_dir: Path, mimi_dir: Path, prefix: str):
    if not data_dir.exists() or not mimi_dir.exists():
        return
    for motion_path in sorted(data_dir.glob("*.npy")):
        if motion_path.name.endswith(".fbank.npy"):
            continue
        candidates = [
            mimi_dir / f"{motion_path.stem}_latent.pt",
            mimi_dir / f"{prefix}{motion_path.stem}_latent.pt",
            mimi_dir / f"{source}_{motion_path.stem}_latent.pt",
        ]
        mimi_path = next((p for p in candidates if p.exists()), None)
        yield source, split, motion_path, mimi_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dualtalk-root", default="/home/s20225367/dataset/dualtalk")
    parser.add_argument("--dualtalk-splits", nargs="+", default=["train", "ood", "test"])
    parser.add_argument("--mimi-root", default="/home/s20225367/dataset/mimi_emb")
    parser.add_argument("--candor-root", default="/home/s20225367/dataset/candor_flame/bc")
    parser.add_argument("--candor-mimi-root", default="/home/s20225367/dataset/mimi_emb/candor_bc")
    parser.add_argument("--face-per-mimi", type=int, default=2)
    parser.add_argument("--output", default="/home/s20225367/dataset/artalk_mimi54_length_mismatches.csv")
    args = parser.parse_args()

    rows = []
    total = 0
    sources = []
    for split in args.dualtalk_splits:
        sources.extend(
            iter_source(
                "dualtalk",
                split,
                Path(args.dualtalk_root) / split,
                Path(args.mimi_root) / split,
                "",
            )
        )
    sources.extend(
        iter_source(
            "candor_bc",
            "train",
            Path(args.candor_root),
            Path(args.candor_mimi_root),
            "candor_bc_",
        )
    )

    for source, split, motion_path, mimi_path in tqdm(sources, desc="length-check"):
        total += 1
        flame_len = npy_len(motion_path)
        if mimi_path is None:
            rows.append(
                {
                    "split": split,
                    "source": source,
                    "stem": motion_path.stem,
                    "status": "missing_mimi",
                    "flame_len": flame_len,
                    "mimi_len": "",
                    "expected_face_len": "",
                    "diff": "",
                    "motion_path": str(motion_path),
                    "mimi_path": "",
                }
            )
            continue
        mlen = mimi_len(mimi_path)
        expected = mlen * args.face_per_mimi
        if flame_len != expected:
            rows.append(
                {
                    "split": split,
                    "source": source,
                    "stem": motion_path.stem,
                    "status": "length_mismatch",
                    "flame_len": flame_len,
                    "mimi_len": mlen,
                    "expected_face_len": expected,
                    "diff": flame_len - expected,
                    "motion_path": str(motion_path),
                    "mimi_path": str(mimi_path),
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        fieldnames = [
            "split",
            "source",
            "stem",
            "status",
            "flame_len",
            "mimi_len",
            "expected_face_len",
            "diff",
            "motion_path",
            "mimi_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[length-check] total={total} mismatches_or_missing={len(rows)} output={output}")


if __name__ == "__main__":
    main()
