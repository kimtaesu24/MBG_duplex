#!/usr/bin/env python

from __future__ import annotations

import math
import random
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _npy_len(path: Path) -> int:
    with path.open("rb") as f:
        version = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format._read_array_header(f, version)
    return int(shape[0])


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _split_dir(root: Path, split_name: str | None) -> Path:
    if split_name in (None, "", "."):
        return root
    return root / split_name


def _load_mimi_features(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        if "audio_features" in payload:
            features = payload["audio_features"]
        elif "features" in payload:
            features = payload["features"]
        else:
            raise KeyError(f"No audio_features/features in {path}")
    else:
        features = payload
    if features.dim() == 3 and features.shape[0] == 1:
        features = features[0]
    if features.dim() != 2:
        raise ValueError(f"Expected [T, C] Mimi features in {path}, got {tuple(features.shape)}")
    return features.float()


def _flame_object_to_motion(item: dict, jaw_dim: int = 1) -> np.ndarray:
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


def load_flame_motion(path: Path, jaw_dim: int = 1) -> torch.Tensor:
    arr = np.load(path, allow_pickle=True)
    if arr.dtype == object:
        motion = np.stack([_flame_object_to_motion(frame, jaw_dim=jaw_dim) for frame in arr], axis=0)
    else:
        motion = np.asarray(arr, dtype=np.float32)
        if motion.shape[-1] == 56 and jaw_dim == 1:
            motion = np.concatenate([motion[..., :50], motion[..., 50:51], motion[..., 53:56]], axis=-1)
        elif motion.shape[-1] > 50 + jaw_dim + 3:
            motion = motion[..., : 50 + jaw_dim + 3]
    expected_dim = 50 + jaw_dim + 3
    if motion.ndim != 2 or motion.shape[-1] != expected_dim:
        raise ValueError(f"Expected FLAME motion [T, {expected_dim}] from {path}, got {motion.shape}")
    return torch.from_numpy(motion).float()


class MimiFlameData(torch.utils.data.Dataset):
    """ARTalk-compatible dataset for precomputed Mimi embeddings and FLAME npy files.

    Returns the same keys as ARTalkData, plus ``audio_feat``.  The ARTalk generator
    can bypass its internal audio encoder when ``audio_feat`` is present.
    """

    def __init__(self, data_cfg, split, debug=False):
        super().__init__()
        self._split = split
        assert self._split in ["train", "val", "test"], f"Invalid split: {self._split}"
        self._clip_length = int(data_cfg.CLIP_LENGTH)
        self._prev_length = int(data_cfg.PREV_LENGTH)
        self._style_length = int(data_cfg.STYLE_LENGTH)
        self._motion_fps = int(data_cfg.MOTION_FPS)
        self._audio_sample_rate = int(data_cfg.AUDIO_SAMPLE_RATE)
        self._mimi_feature_fps = float(getattr(data_cfg, "MIMI_FEATURE_FPS", 12.5))
        self._face_per_mimi = float(self._motion_fps / self._mimi_feature_fps)
        self._stride = int(getattr(data_cfg, "STRIDE", 50))
        self._jaw_dim = int(getattr(data_cfg, "JAW_DIM", 1))
        self._motion_dim = 50 + self._jaw_dim + 3
        self._mismatch_csv = getattr(data_cfg, "LENGTH_MISMATCH_CSV", None)
        self._data_path = "mimi_flame_sources"
        self._debug = debug

        sources = list(getattr(data_cfg, "SOURCES"))
        records = []
        mismatches = []
        for src in sources:
            source_name = str(src.get("NAME", "source"))
            data_root = _as_path(src["DATA_ROOT"])
            mimi_root = _as_path(src["MIMI_ROOT"])
            split_map = dict(src.get("SPLITS", {}))
            mimi_split_map = dict(src.get("MIMI_SPLITS", split_map))
            data_dir = _split_dir(data_root, split_map.get(split, split))
            mimi_dir = _split_dir(mimi_root, mimi_split_map.get(split, split))
            val_ratio = float(src.get("VAL_RATIO", 0.0))
            stem_prefix = str(src.get("MIMI_STEM_PREFIX", ""))
            stem_suffix_strip = str(src.get("MIMI_STEM_SUFFIX_STRIP", ""))
            length_tolerance = int(src.get("LENGTH_TOLERANCE", 0))
            if not data_dir.exists() or not mimi_dir.exists():
                continue
            npy_paths = [p for p in sorted(data_dir.glob("*.npy")) if not p.name.endswith(".fbank.npy")]
            for npy_path in npy_paths:
                if split in ("val", "test") and val_ratio > 0.0:
                    bucket = (abs(hash(npy_path.stem)) % 10000) / 10000.0
                    if split == "val" and bucket >= val_ratio:
                        continue
                    if split == "test" and bucket < val_ratio:
                        continue
                mimi_stem = npy_path.stem
                if stem_suffix_strip and mimi_stem.endswith(stem_suffix_strip):
                    mimi_stem = mimi_stem[: -len(stem_suffix_strip)]
                candidates = [
                    mimi_dir / f"{npy_path.stem}_latent.pt",
                    mimi_dir / f"{mimi_stem}_latent.pt",
                    mimi_dir / f"{stem_prefix}{mimi_stem}_latent.pt",
                    mimi_dir / f"{source_name}_{mimi_stem}_latent.pt",
                ]
                mimi_path = next((p for p in candidates if p.exists()), None)
                if mimi_path is None:
                    continue
                flame_len = _npy_len(npy_path)
                try:
                    mimi_len = _load_mimi_features(mimi_path).shape[0]
                except Exception:
                    continue
                expected_face_len = int(mimi_len * self._face_per_mimi)
                if abs(flame_len - expected_face_len) > length_tolerance:
                    mismatches.append(
                        {
                            "split": split,
                            "source": source_name,
                            "stem": npy_path.stem,
                            "flame_len": flame_len,
                            "mimi_len": mimi_len,
                            "expected_face_len": expected_face_len,
                            "diff": flame_len - expected_face_len,
                            "motion_path": str(npy_path),
                            "mimi_path": str(mimi_path),
                        }
                    )
                    continue
                seq_len = flame_len
                min_len = max(self._clip_length, self._prev_length, self._style_length) + 1
                if seq_len <= min_len:
                    continue
                records.append(
                    {
                        "source": source_name,
                        "key": npy_path.stem,
                        "motion_path": npy_path,
                        "mimi_path": mimi_path,
                        "seq_len": int(seq_len),
                    }
                )

        if debug:
            records = records[:64]
        self._records = records
        if split == "train":
            windows = []
            for ridx, rec in enumerate(records):
                for start in range(0, rec["seq_len"] - self._clip_length, self._stride):
                    windows.append((ridx, start))
            self._all_data = windows
        else:
            self._all_data = [(ridx, 0) for ridx in range(len(records))]

        if len(self._all_data) == 0:
            raise RuntimeError(f"No MimiFlameData records found for split={split}. Check DATA_ROOT/MIMI_ROOT.")
        if self._mismatch_csv and mismatches:
            mismatch_path = Path(self._mismatch_csv)
            if split != "train":
                mismatch_path = mismatch_path.with_name(f"{mismatch_path.stem}_{split}{mismatch_path.suffix}")
            mismatch_path.parent.mkdir(parents=True, exist_ok=True)
            with mismatch_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "split",
                        "source",
                        "stem",
                        "flame_len",
                        "mimi_len",
                        "expected_face_len",
                        "diff",
                        "motion_path",
                        "mimi_path",
                    ],
                )
                writer.writeheader()
                writer.writerows(mismatches)

    def slice(self, slice, shuffle=False):
        if shuffle:
            random.shuffle(self._all_data)
        self._all_data = self._all_data[:slice]

    def __len__(self):
        return len(self._all_data)

    def __getitem__(self, index):
        rec_idx, base_start = self._all_data[index]
        rec = self._records[rec_idx]
        motion = load_flame_motion(rec["motion_path"], jaw_dim=self._jaw_dim)
        audio_feat = _load_mimi_features(rec["mimi_path"])
        seq_len = min(motion.shape[0], int(math.floor(audio_feat.shape[0] * self._face_per_mimi)))
        motion = motion[:seq_len]

        if self._split == "train":
            start = random.randint(base_start, min(base_start + self._stride, seq_len - self._clip_length))
            end = start + self._clip_length
            curr_motion = motion[start:end]
            prev_start = max(start - self._prev_length, 0)
            prev_motion = motion[prev_start:start]
            if prev_motion.shape[0] < self._prev_length:
                prev_motion = F.pad(prev_motion, (0, 0, self._prev_length - prev_motion.shape[0], 0), value=0)
            style_start = random.randint(0, seq_len - self._style_length)
            style_motion = motion[style_start : style_start + self._style_length]
            feat_start = int(math.floor(start / self._face_per_mimi))
            feat_len = int(math.ceil(self._clip_length / self._face_per_mimi))
            curr_audio_feat = audio_feat[feat_start : feat_start + feat_len]
            if curr_audio_feat.shape[0] < feat_len:
                curr_audio_feat = F.pad(curr_audio_feat, (0, 0, 0, feat_len - curr_audio_feat.shape[0]), value=0)
            llm_feat = curr_audio_feat.new_zeros(curr_audio_feat.shape[0], 4096)
            audio_len = int(self._clip_length * self._audio_sample_rate / self._motion_fps)
            return {
                "audio": torch.zeros(audio_len, dtype=torch.float32),
                "audio_feat": curr_audio_feat.float(),
                "llm_feat": llm_feat.float(),
                "motion_code": curr_motion.float(),
                "prev_motion_code": prev_motion.float(),
                "style_motion_code": style_motion.float(),
                "infos": {"motion_key": rec["key"], "source": rec["source"]},
            }

        style_frame = min(seq_len // 2, seq_len - self._style_length)
        style_motion = motion[style_frame : style_frame + self._style_length]
        return {
            "audio": torch.zeros(int(seq_len * self._audio_sample_rate / self._motion_fps), dtype=torch.float32),
            "audio_feat": audio_feat[: int(math.ceil(seq_len / self._face_per_mimi))].float(),
            "llm_feat": audio_feat.new_zeros(int(math.ceil(seq_len / self._face_per_mimi)), 4096).float(),
            "motion_code": motion.float(),
            "style_motion_code": style_motion.float(),
            "infos": {"motion_key": rec["key"], "source": rec["source"]},
        }

    def sample(self, data_key=None):
        if data_key is None:
            return self[random.randrange(len(self))]
        for idx, rec in enumerate(self._records):
            if rec["key"] == data_key:
                return self._load_val_record(idx)
        raise ValueError(f"Data key {data_key} not found.")

    def _load_val_record(self, rec_idx):
        old_split = self._split
        try:
            self._split = "val"
            return self.__getitem__(rec_idx)
        finally:
            self._split = old_split

    def close(self):
        pass
