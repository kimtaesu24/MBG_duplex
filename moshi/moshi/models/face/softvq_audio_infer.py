#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure we can import from the moshi repo root
repo_root = Path(__file__).resolve().parents[3] # /home6/duplex/personaplex/moshi
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from moshi.face_pretrain import encode_audio
from moshi.models import loaders
from moshi.ARTalk.train_code.softvq_continuous_online_train import (
    CausalSoftVQContinuousTransformer,
    motion54_to_56,
)

def parse_args():
    p = argparse.ArgumentParser(description="Inference ArTalk from a single audio file.")
    p.add_argument("--audio", type=str, required=True, help="Path to input .wav file")
    p.add_argument(
        "--ckpt", 
        type=str, 
        default="/home6/duplex/personaplex/moshi/moshi/ARTalk/train_code/outputs/SoftVQContinuousOnline_mimi54_dualtalk_ami_b512/checkpoint_epoch_500.ckpt",
        help="Path to SoftVQ model checkpoint"
    )
    p.add_argument(
        "--mimi-weight", 
        type=str, 
        default="/home6/duplex/personaplex/moshi/moshi/personaplex-7b-v1/tokenizer-e351c8d8-checkpoint125.safetensors",
        help="Path to Mimi tokenizer weights"
    )
    p.add_argument("--output", type=str, default="pred.npy", help="Path to save output .npy motion")
    p.add_argument("--stream-context-frames", type=int, default=25, help="Context windows for generation")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Load Mimi
    print(f"[softvq-audio-infer] Loading Mimi from {args.mimi_weight}")
    mimi = loaders.get_mimi(args.mimi_weight, device)
    mimi.eval()

    # Load SoftVQ Model
    print(f"[softvq-audio-infer] Loading model from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    ckpt_args = ckpt.get("args", {})
    
    model = CausalSoftVQContinuousTransformer(
        hidden_dim=int(ckpt_args.get("hidden_dim", 512)),
        layers=int(ckpt_args.get("layers", 6)),
        heads=int(ckpt_args.get("heads", 8)),
        code_dim=int(ckpt_args.get("code_dim", 32)),
        prior_warmup_frames=int(ckpt_args.get("prior_warmup_frames", 10)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Encode Audio to Mimi embeddings
    print(f"[softvq-audio-infer] Encoding audio: {args.audio}")
    audio_codes = encode_audio(mimi, Path(args.audio))
    # audio_codes is [8, T]
    
    # decode_latent expects [B, K, T]
    audio_feat = mimi.decode_latent(audio_codes.unsqueeze(0).to(device))
    # audio_feat is [1, 512, T]
    audio_feat = audio_feat.transpose(1, 2) # [1, T, 512]
    
    # Generate Motion
    print(f"[softvq-audio-infer] Generating motion...")
    # 1 Mimi frame (12.5 fps) -> 2 face frames (25 fps)
    max_frames = audio_feat.shape[1] * 2
    pred = model.generate(
        audio_feat, 
        max_frames=max_frames,
        max_context_frames=args.stream_context_frames
    )
    # pred is [1, max_frames, 54]
    
    # Convert and Save
    motion_56 = motion54_to_56(pred[0])
    np.save(args.output, motion_56.cpu().numpy())
    print(f"[softvq-audio-infer] Saved to {args.output} (shape: {list(motion_56.shape)})")

if __name__ == "__main__":
    main()
