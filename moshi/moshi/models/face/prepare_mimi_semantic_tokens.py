#!/usr/bin/env python
"""Extract mimi token sidecars for the v2 (mimi-token MTP) trainer.

For every ``*_latent.pt`` mimi-feature file used by the online trainers, this
re-encodes the source wav with the SAME Mimi tokenizer that produced the
features (the features were computed as ``mimi.decode_latent(codes)`` but the
codes themselves were never stored -- see moshi/precompute_ami_mimi.py) and
saves the codes next to it as ``{stem}_token.pt``:

    {"audio_tokens": int16 [K, T]}     # level 0 = semantic (WavLM-distilled)

The v2 trainer / MimiFlameData only read level 0, but all K levels are kept
(cheap, int16) so acoustic-token experiments don't need another pass.

Run on a free GPU (encoding is fast; a busy GPU also works but steals memory):

    cd /home6/duplex/personaplex/moshi/moshi/ARTalk/train_code
    CUDA_VISIBLE_DEVICES=0 python prepare_mimi_semantic_tokens.py

Idempotent: existing ``*_token.pt`` files are skipped (``--overwrite`` redoes).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

MOSHI_ROOT = "/home6/duplex/personaplex/moshi"
if MOSHI_ROOT not in sys.path:
    sys.path.insert(0, MOSHI_ROOT)

from moshi.face_pretrain import encode_audio  # noqa: E402
from moshi.models import loaders  # noqa: E402

# (wav dir, mimi-latent dir) pairs, matching make_dataset_cfg in the trainers.
DEFAULT_PAIRS = [
    # dualtalk
    # ("/home6/duplex/dataset/dualtalk/train", "/home6/duplex/dataset/mimi_emb/train"),
    # ("/home6/duplex/dataset/dualtalk/ood", "/home6/duplex/dataset/mimi_emb/ood"),
    # ("/home6/duplex/dataset/dualtalk/test", "/home6/duplex/dataset/mimi_emb/test"),
    # ami utterances
    ("/home6/duplex/dataset/ami_flame/ut/train", "/home6/duplex/dataset/mimi_emb/ami_train_ut"),
    ("/home6/duplex/dataset/ami_flame/ut/valid", "/home6/duplex/dataset/mimi_emb/ami_val_ut"),
    ("/home6/duplex/dataset/ami_flame/ut/test", "/home6/duplex/dataset/mimi_emb/ami_test_ut"),
    # ami backchannels (not in the current training mix, but cheap to cover)
    ("/home6/duplex/dataset/ami_flame/bc/train", "/home6/duplex/dataset/mimi_emb/ami_train_bc"),
    ("/home6/duplex/dataset/ami_flame/bc/valid", "/home6/duplex/dataset/mimi_emb/ami_val_bc"),
    ("/home6/duplex/dataset/ami_flame/bc/test", "/home6/duplex/dataset/mimi_emb/ami_test_bc"),
    # unils (SeamlessInteractionTalk flame56 export)
    # ("/home6/duplex/dataset/unils/SeamlessInteractionTalk/flame56/train", "/home6/duplex/dataset/mimi_emb/unils/train"),
    # ("/home6/duplex/dataset/unils/SeamlessInteractionTalk/flame56/val", "/home6/duplex/dataset/mimi_emb/unils/val"),
    # ("/home6/duplex/dataset/unils/SeamlessInteractionTalk/flame56/test", "/home6/duplex/dataset/mimi_emb/unils/test"),
]


def _latent_len(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    feat = payload["audio_features"] if isinstance(payload, dict) else payload
    return int(feat.shape[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mimi-weight",
        default="/home6/duplex/personaplex/personaplex-7b-v1/tokenizer-e351c8d8-checkpoint125.safetensors",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--filter",
        default="",
        help="only process (wav_dir, mimi_dir) pairs whose mimi_dir path contains this substring, "
        "e.g. --filter unils",
    )
    args = p.parse_args()

    pairs = DEFAULT_PAIRS
    if args.filter:
        pairs = [(w, m) for w, m in DEFAULT_PAIRS if args.filter in m]
        print(f"[mimi-tokens] --filter={args.filter!r} -> {len(pairs)}/{len(DEFAULT_PAIRS)} pairs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mimi-tokens] loading mimi from {args.mimi_weight} on {device}")
    mimi = loaders.get_mimi(args.mimi_weight, device)
    mimi.eval()
    for param in mimi.parameters():
        param.requires_grad = False

    total_done = total_skip = total_missing_wav = total_len_mismatch = 0
    for wav_dir, mimi_dir in pairs:
        wav_dir, mimi_dir = Path(wav_dir), Path(mimi_dir)
        if not wav_dir.is_dir() or not mimi_dir.is_dir():
            print(f"[mimi-tokens] skipping missing pair: {wav_dir} <-> {mimi_dir}")
            continue
        latents = sorted(mimi_dir.glob("*_latent.pt"))
        print(f"[mimi-tokens] {mimi_dir} ({len(latents)} latent files)")
        for latent_path in tqdm(latents, desc=str(mimi_dir), unit="file"):
            stem = latent_path.name[: -len("_latent.pt")]
            token_path = latent_path.with_name(f"{stem}_token.pt")
            if token_path.exists() and not args.overwrite:
                total_skip += 1
                continue
            wav_path = wav_dir / f"{stem}.wav"
            if not wav_path.exists():
                total_missing_wav += 1
                print(f"[mimi-tokens] no wav for {latent_path.name} (expected {wav_path})")
                continue
            codes = encode_audio(mimi, wav_path)  # [K, T] long on cpu
            n_lat = _latent_len(latent_path)
            if codes.shape[1] != n_lat:
                total_len_mismatch += 1
                print(
                    f"[mimi-tokens] length mismatch {latent_path.name}: "
                    f"tokens={codes.shape[1]} latent={n_lat} (saving anyway; "
                    "the dataset pads/truncates to the feature window)"
                )
            torch.save({"audio_tokens": codes.to(torch.int16)}, token_path)
            total_done += 1
    print(
        f"[mimi-tokens] done: wrote={total_done} skipped_existing={total_skip} "
        f"missing_wav={total_missing_wav} length_mismatch={total_len_mismatch}"
    )


if __name__ == "__main__":
    main()
