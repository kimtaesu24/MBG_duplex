#!/usr/bin/env python
"""Summarise behavioral EPAD statistics from a fusion sweep.

These statistics are screening diagnostics, not a replacement for listening to
the generated audio or evaluating on a labelled held-out set.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


FIELDS = [
    "setting",
    "samples",
    "frames",
    "epad_frames",
    "epad_onsets",
    "word_frames",
    "epad_frame_rate",
    "user_active_epad_rate",
    "user_active_onset_rate",
    "mean_vad_user_at_onset",
    "mean_fusion_prob",
]


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def summarise(setting_dir: Path) -> dict[str, object]:
    records: list[dict] = []
    onsets: list[dict] = []
    sample_count = 0
    for path in sorted(setting_dir.glob("*.json")):
        if path.name.endswith("_vap.json"):
            continue
        try:
            with path.open() as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: skipping unreadable result {path}: {exc}", file=sys.stderr)
            continue
        gate_log = payload.get("bc_gate_log")
        if not isinstance(gate_log, list):
            continue
        sample_count += 1
        records.extend(gate_log)
        previous_epad = False
        for record in gate_log:
            current_epad = record.get("token_label") == "EPAD"
            if current_epad and not previous_epad:
                onsets.append(record)
            previous_epad = current_epad

    epad = [r for r in records if r.get("token_label") == "EPAD"]
    words = [r for r in records if str(r.get("token_label", "")).startswith("WORD(")]

    active_epad = [r for r in epad if float(r.get("vad_user", 0.0)) >= 0.5]
    active_onsets = [r for r in onsets if float(r.get("vad_user", 0.0)) >= 0.5]
    fusion_probs = [
        float(r["fusion_prob"])
        for r in records
        if r.get("fusion_prob") is not None and math.isfinite(float(r["fusion_prob"]))
    ]

    return {
        "setting": setting_dir.name,
        "samples": sample_count,
        "frames": len(records),
        "epad_frames": len(epad),
        "epad_onsets": len(onsets),
        "word_frames": len(words),
        "epad_frame_rate": len(epad) / len(records) if records else float("nan"),
        "user_active_epad_rate": len(active_epad) / len(epad) if epad else float("nan"),
        "user_active_onset_rate": len(active_onsets) / len(onsets) if onsets else float("nan"),
        "mean_vad_user_at_onset": safe_mean(
            [float(r.get("vad_user", 0.0)) for r in onsets]
        ),
        "mean_fusion_prob": safe_mean(fusion_probs),
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <sweep-output-root>")
    root = Path(sys.argv[1]).resolve()
    rows = [summarise(p) for p in sorted(root.iterdir()) if p.is_dir()]
    output = root / "summary.csv"
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output}")
    for row in rows:
        print(
            f"{row['setting']:30s} "
            f"onsets={row['epad_onsets']:4d} "
            f"active_onset_rate={row['user_active_onset_rate']:.3f} "
            f"word_frames={row['word_frames']:5d}"
        )


if __name__ == "__main__":
    main()
