#!/usr/bin/env python

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

from core.data.mimi_flame_data import MimiFlameData
from core.models.artalk_codec import ARTalkCodec




def detect_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 0, 1, device
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")


def cleanup_ddp(enabled: bool):
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def reduce_metrics(metrics: dict[str, float], device: torch.device, world_size: int) -> dict[str, float]:
    if world_size <= 1:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([metrics[k] for k in keys], device=device, dtype=torch.float32)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= world_size
    return {k: float(v.item()) for k, v in zip(keys, values)}


def reduce_sums(values: dict[str, float], device: torch.device, world_size: int) -> dict[str, float]:
    if world_size <= 1:
        return values
    keys = sorted(values)
    tensor = torch.tensor([values[k] for k in keys], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {k: float(v.item()) for k, v in zip(keys, tensor)}


def local_limit(total: int, rank: int, world_size: int) -> int:
    if total <= 0:
        return 0
    base = total // world_size
    extra = total % world_size
    return base + (1 if rank < extra else 0)


def motion54_to_56(motion: torch.Tensor) -> torch.Tensor:
    if motion.shape[-1] != 54:
        return motion
    exp = motion[..., :50]
    jaw_open = motion[..., 50:51]
    jaw = torch.cat([jaw_open, torch.zeros_like(jaw_open), torch.zeros_like(jaw_open)], dim=-1)
    neck = motion[..., 51:54]
    return torch.cat([exp, jaw, neck], dim=-1)


def make_dataset_cfg(args) -> SimpleNamespace:
    sources = [
        {
            "NAME": "dualtalk",
            "DATA_ROOT": args.dualtalk_root,
            "MIMI_ROOT": args.mimi_root,
            "SPLITS": {"train": "train", "val": "ood", "test": "test"},
            "MIMI_SPLITS": {"train": "train", "val": "ood", "test": "test"},
            "MIMI_STEM_PREFIX": "",
            "LENGTH_TOLERANCE": 2,
        },
        {
            "NAME": "ami_bc",
            "DATA_ROOT": args.ami_bc_root,
            "MIMI_ROOT": args.ami_bc_mimi_root,
            "SPLITS": {"train": "train", "val": "valid", "test": "test"},
            "MIMI_SPLITS": {"train": "ami_train_bc", "val": "ami_val_bc", "test": "ami_test_bc"},
            "MIMI_STEM_SUFFIX_STRIP": "_bc",
            "LENGTH_TOLERANCE": 2,
        },
        {
            "NAME": "ami_ut",
            "DATA_ROOT": args.ami_ut_root,
            "MIMI_ROOT": args.ami_ut_mimi_root,
            "SPLITS": {"train": "train", "val": "valid", "test": "test"},
            "MIMI_SPLITS": {"train": "ami_train_ut", "val": "ami_val_ut", "test": "ami_test_ut"},
            "LENGTH_TOLERANCE": 2,
        },
    ]
    raw_sources = {name.strip() for name in args.sources.split(",") if name.strip()}
    enabled_sources = set()
    for s in raw_sources:
        if s == "ami":
            enabled_sources.update(["ami_bc", "ami_ut"])
        else:
            enabled_sources.add(s)
    sources = [src for src in sources if src["NAME"] in enabled_sources]
    return SimpleNamespace(
        CLIP_LENGTH=args.clip_length,
        PREV_LENGTH=args.clip_length,
        STYLE_LENGTH=args.clip_length,
        STRIDE=args.stride,
        MOTION_FPS=25,
        AUDIO_SAMPLE_RATE=16000,
        MIMI_FEATURE_FPS=12.5,
        JAW_DIM=1,
        LENGTH_MISMATCH_CSV=args.length_mismatch_csv,
        SOURCES=sources,
    )


def load_codec(args, device: torch.device) -> ARTalkCodec:
    cfg = SimpleNamespace(
        MOTION_DIM=54,
        V_CODE_DIM=args.code_dim,
        V_PATCH_NUMS=[1, 10, 20, 50, 100],
        T_DEPTH=args.codec_depth,
        T_NUM_HEADS=args.codec_heads,
        T_HIDDEN_DIM=args.codec_hidden_dim,
        STATS_PATH=args.stats_path,
    )
    codec = ARTalkCodec(cfg, init_submodule=False)
    ckpt = torch.load(args.codec_ckpt, map_location="cpu", weights_only=True)
    codec.load_state_dict(ckpt["model"], strict=True)
    codec.eval().to(device)
    for p in codec.parameters():
        p.requires_grad = False
    return codec


class CausalSoftVQContinuousTransformer(nn.Module):
    def __init__(
        self,
        audio_dim: int = 512,
        motion_dim: int = 54,
        code_dim: int = 32,
        hidden_dim: int = 512,
        layers: int = 6,
        heads: int = 8,
        dropout: float = 0.0,
        max_len: int = 2048,
        exp_res_scale: float = 0.08,
        jaw_res_scale: float = 0.03,
        neck_res_scale: float = 0.01,
        prior_warmup_frames: int = 10,
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.code_dim = code_dim
        self.exp_res_scale = exp_res_scale
        self.jaw_res_scale = jaw_res_scale
        self.neck_res_scale = neck_res_scale
        self.prior_warmup_frames = int(prior_warmup_frames)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.llm_proj = nn.Linear(4096, hidden_dim, bias=False)
        self.motion_proj = nn.Linear(motion_dim + hidden_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        self.phase_emb = nn.Embedding(2, hidden_dim)
        audio_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        motion_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.audio_encoder = nn.TransformerEncoder(audio_layer, num_layers=layers)
        self.motion_encoder = nn.TransformerEncoder(motion_layer, num_layers=layers)
        self.audio_norm = nn.LayerNorm(hidden_dim)
        self.motion_norm = nn.LayerNorm(hidden_dim)
        self.z_head = nn.Linear(hidden_dim, code_dim)
        self.prior_head = nn.Sequential(
            nn.Linear(hidden_dim + code_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, motion_dim),
        )
        self.delta_head = nn.Linear(hidden_dim * 2, motion_dim)
        self.res_head = nn.Linear(hidden_dim * 2, motion_dim)
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + motion_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.start_motion = nn.Parameter(torch.zeros(1, 1, motion_dim))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.res_head.weight)
        nn.init.zeros_(self.res_head.bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, -1.5)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def _expand_audio(self, audio_feat: torch.Tensor, face_len: int) -> torch.Tensor:
        audio_rep = audio_feat.repeat_interleave(2, dim=1)
        if audio_rep.shape[1] < face_len:
            pad = audio_rep.new_zeros(audio_rep.shape[0], face_len - audio_rep.shape[1], audio_rep.shape[2])
            audio_rep = torch.cat([audio_rep, pad], dim=1)
        return audio_rep[:, :face_len]

    def _expand_llm(self, llm_feat: torch.Tensor | None, audio_feat: torch.Tensor, face_len: int) -> torch.Tensor:
        if llm_feat is None:
            llm_feat = audio_feat.new_zeros(audio_feat.shape[0], audio_feat.shape[1], 4096)
        llm_rep = llm_feat.repeat_interleave(2, dim=1)
        if llm_rep.shape[1] < face_len:
            pad = llm_rep.new_zeros(llm_rep.shape[0], face_len - llm_rep.shape[1], llm_rep.shape[2])
            llm_rep = torch.cat([llm_rep, pad], dim=1)
        return llm_rep[:, :face_len]

    def _scale_motion(self, raw: torch.Tensor) -> torch.Tensor:
        exp = self.exp_res_scale * torch.tanh(raw[..., :50])
        jaw = self.jaw_res_scale * torch.tanh(raw[..., 50:51])
        neck = self.neck_res_scale * torch.tanh(raw[..., 51:54])
        return torch.cat([exp, jaw, neck], dim=-1)

    @staticmethod
    def _expand_group_gate(group_gate: torch.Tensor) -> torch.Tensor:
        expr = group_gate[..., 0:1].expand(*group_gate.shape[:-1], 50)
        jaw = group_gate[..., 1:2]
        neck = group_gate[..., 2:3].expand(*group_gate.shape[:-1], 3)
        return torch.cat([expr, jaw, neck], dim=-1)

    def _warmup_candidate_weight(self, face_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.prior_warmup_frames <= 0:
            return torch.ones(1, face_len, 1, device=device, dtype=dtype)
        pos = torch.arange(face_len, device=device, dtype=dtype)
        return (pos / float(self.prior_warmup_frames)).clamp(0.0, 1.0).view(1, face_len, 1)

    def forward(
        self,
        audio_feat: torch.Tensor,
        prev_motion: torch.Tensor,
        llm_feat: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, face_len, _ = prev_motion.shape
        audio_rep = self._expand_audio(audio_feat, face_len)
        llm_rep = self._expand_llm(llm_feat, audio_feat, face_len)
        pos = torch.arange(face_len, device=prev_motion.device).remainder(self.pos_emb.num_embeddings)
        phase = pos.remainder(2)
        pos_phase = self.pos_emb(pos)[None] + self.phase_emb(phase)[None]
        mask = self._causal_mask(face_len, prev_motion.device)

        audio_h = self.audio_proj(audio_rep) + self.llm_proj(llm_rep) + pos_phase
        audio_h = self.audio_encoder(audio_h, mask=mask)
        audio_h = self.audio_norm(audio_h)

        motion_h = self.motion_proj(torch.cat([prev_motion, audio_h], dim=-1)) + pos_phase
        motion_h = self.motion_encoder(motion_h, mask=mask)
        motion_h = self.motion_norm(motion_h)

        z = self.z_head(audio_h)
        prior = self.prior_head(torch.cat([audio_h, z], dim=-1))
        dyn_h = torch.cat([motion_h, audio_h], dim=-1)
        delta = self._scale_motion(self.delta_head(dyn_h))
        residual = self._scale_motion(self.res_head(dyn_h))
        candidate = prev_motion + delta
        group_gate = torch.sigmoid(
            self.gate_head(torch.cat([dyn_h, prev_motion, prior, candidate, candidate - prior], dim=-1))
        )
        gate = self._expand_group_gate(group_gate)
        warmup = self._warmup_candidate_weight(face_len, prev_motion.device, gate.dtype)
        gate = gate * warmup
        group_gate_eff = torch.stack(
            [
                gate[..., :50].mean(dim=-1),
                gate[..., 50],
                gate[..., 51:54].mean(dim=-1),
            ],
            dim=-1,
        )
        base = gate * candidate + (1.0 - gate) * prior
        final = base + residual
        return {
            "pred_motion": final,
            "prior_motion": prior,
            "z_pred": z,
            "delta": delta,
            "residual": residual,
            "gate": gate,
            "group_gate": group_gate,
            "group_gate_eff": group_gate_eff,
        }

    @torch.no_grad()
    def generate(
        self,
        audio_feat: torch.Tensor,
        max_frames: int | None = None,
        llm_feat: torch.Tensor | None = None,
        max_context_frames: int | None = None,
    ) -> torch.Tensor:
        bsz = audio_feat.shape[0]
        face_len = int(audio_feat.shape[1] * 2) if max_frames is None else int(max_frames)
        if max_context_frames is None:
            max_context_frames = self.pos_emb.num_embeddings
        max_context_frames = max(1, min(int(max_context_frames), self.pos_emb.num_embeddings))
        preds = []
        prev_all = self.start_motion.expand(bsz, 1, -1).to(audio_feat)
        for t in range(face_len):
            ctx_start = max(0, prev_all.shape[1] - max_context_frames)
            audio_start = ctx_start // 2
            audio_end = (t // 2) + 1
            llm_prefix = None if llm_feat is None else llm_feat[:, audio_start:audio_end]
            out = self.forward(audio_feat[:, audio_start:audio_end], prev_all[:, ctx_start:], llm_feat=llm_prefix)
            next_motion = out["pred_motion"][:, -1:]
            preds.append(next_motion)
            prev_all = torch.cat([prev_all, next_motion], dim=1)
        return torch.cat(preds, dim=1)


def shifted_motion(gt: torch.Tensor, model: CausalSoftVQContinuousTransformer) -> torch.Tensor:
    start = model.start_motion.expand(gt.shape[0], 1, -1).to(device=gt.device, dtype=gt.dtype)
    return torch.cat([start, gt[:, :-1]], dim=1)


@torch.no_grad()
def codec_quant_to_sum_feat_chunked(codec, motion: torch.Tensor) -> torch.Tensor:
    chunk_len = int(getattr(codec, "patch_nums", [100])[-1])
    if motion.shape[1] <= chunk_len:
        return codec.quant_to_sum_feat(motion)
    chunks = []
    for start in range(0, motion.shape[1], chunk_len):
        chunk = motion[:, start : start + chunk_len]
        chunks.append(codec.quant_to_sum_feat(chunk)[:, : chunk.shape[1]])
    return torch.cat(chunks, dim=1)[:, : motion.shape[1]]


def compute_loss(model, codec, batch, args, autocast_dtype=None):
    gt = batch["motion_code"]
    audio_feat = batch["audio_feat"]
    llm_feat = batch.get("llm_feat")
    prev = shifted_motion(gt, model.module if isinstance(model, DDP) else model)
    with torch.no_grad():
        z_target = codec_quant_to_sum_feat_chunked(codec, gt)
    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
        out = model(audio_feat, prev, llm_feat=llm_feat)
        pred = out["pred_motion"]
        prior = out["prior_motion"]
        loss_motion = F.l1_loss(pred, gt)
        loss_prior = F.l1_loss(prior, gt)
        loss_z = F.mse_loss(out["z_pred"], z_target)
        z_target_bits = (z_target > 0).to(out["z_pred"].dtype)
        loss_z_bce = F.binary_cross_entropy_with_logits(out["z_pred"] * math.sqrt(args.code_dim), z_target_bits)
        z_bit_acc = ((out["z_pred"] > 0) == (z_target > 0)).float().mean()
        loss_jaw = F.l1_loss(pred[..., 50:51], gt[..., 50:51])
        loss_expr = F.l1_loss(pred[..., :50], gt[..., :50])
        loss_neck = F.l1_loss(pred[..., 51:54], gt[..., 51:54])
        loss_prior_expr = F.l1_loss(prior[..., :50], gt[..., :50])
        loss_prior_jaw = F.l1_loss(prior[..., 50:51], gt[..., 50:51])
        loss_prior_neck = F.l1_loss(prior[..., 51:54], gt[..., 51:54])
        pred_vel = pred[:, 1:] - pred[:, :-1]
        gt_vel = gt[:, 1:] - gt[:, :-1]
        loss_vel = F.mse_loss(pred_vel, gt_vel)
        loss_jaw_vel = F.mse_loss(pred_vel[..., 50:51], gt_vel[..., 50:51])
        loss_neck_vel = F.mse_loss(pred_vel[..., 51:54], gt_vel[..., 51:54])
        loss_reg = out["delta"].pow(2).mean() + out["residual"].pow(2).mean()
        gate_target = out["group_gate"].new_tensor(
            [args.gate_target_expr, args.gate_target_jaw, args.gate_target_neck]
        ).view(1, 1, 3)
        gate_group_weight = out["group_gate"].new_tensor(
            [args.gate_loss_expr_weight, args.gate_loss_jaw_weight, args.gate_loss_neck_weight]
        ).view(1, 1, 3)
        loss_gate = ((out["group_gate"] - gate_target) ** 2 * gate_group_weight).mean()
        loss = (
            args.motion_weight * loss_motion
            + args.prior_weight * loss_prior
            + args.z_weight * loss_z
            + args.z_bce_weight * loss_z_bce
            + args.jaw_weight * loss_jaw
            + args.vel_weight * loss_vel
            + args.reg_weight * loss_reg
            + args.gate_weight * loss_gate
        )
    return loss, {
        "loss": float(loss.detach().item()),
        "motion": float(loss_motion.detach().item()),
        "prior": float(loss_prior.detach().item()),
        "z": float(loss_z.detach().item()),
        "z_bce": float(loss_z_bce.detach().item()),
        "z_bit_acc": float(z_bit_acc.detach().item()),
        "jaw": float(loss_jaw.detach().item()),
        "expr": float(loss_expr.detach().item()),
        "neck": float(loss_neck.detach().item()),
        "prior_expr": float(loss_prior_expr.detach().item()),
        "prior_jaw": float(loss_prior_jaw.detach().item()),
        "prior_neck": float(loss_prior_neck.detach().item()),
        "vel": float(loss_vel.detach().item()),
        "jaw_vel": float(loss_jaw_vel.detach().item()),
        "neck_vel": float(loss_neck_vel.detach().item()),
        "reg": float(loss_reg.detach().item()),
        "gate_loss": float(loss_gate.detach().item()),
        "weighted_motion": float((args.motion_weight * loss_motion).detach().item()),
        "weighted_prior": float((args.prior_weight * loss_prior).detach().item()),
        "weighted_z": float((args.z_weight * loss_z).detach().item()),
        "weighted_z_bce": float((args.z_bce_weight * loss_z_bce).detach().item()),
        "weighted_jaw": float((args.jaw_weight * loss_jaw).detach().item()),
        "weighted_vel": float((args.vel_weight * loss_vel).detach().item()),
        "weighted_reg": float((args.reg_weight * loss_reg).detach().item()),
        "weighted_gate": float((args.gate_weight * loss_gate).detach().item()),
        "gate": float(out["group_gate"].detach().mean().item()),
        "gate_expr": float(out["group_gate"][..., 0].detach().mean().item()),
        "gate_jaw": float(out["group_gate"][..., 1].detach().mean().item()),
        "gate_neck": float(out["group_gate"][..., 2].detach().mean().item()),
        "gate_eff": float(out["group_gate_eff"].detach().mean().item()),
        "gate_eff_expr": float(out["group_gate_eff"][..., 0].detach().mean().item()),
        "gate_eff_jaw": float(out["group_gate_eff"][..., 1].detach().mean().item()),
        "gate_eff_neck": float(out["group_gate_eff"][..., 2].detach().mean().item()),
        "delta_abs": float(out["delta"].detach().abs().mean().item()),
        "residual_abs": float(out["residual"].detach().abs().mean().item()),
        "prior_abs": float(prior.detach().abs().mean().item()),
        "pred_abs": float(pred.detach().abs().mean().item()),
    }


@torch.no_grad()
def validate(model, codec, loader, device, args, epoch: int, out_dir: Path, rank: int, world_size: int):
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    total = {
        "teacher_motion": 0.0,
        "teacher_prior": 0.0,
        "teacher_z": 0.0,
        "teacher_z_bce": 0.0,
        "teacher_z_bit_acc": 0.0,
        "teacher_jaw": 0.0,
        "ar_motion": 0.0,
        "ar_jaw": 0.0,
        "ar_vel": 0.0,
        "samples": 0.0,
    }
    save_root = out_dir / f"val_epoch_{epoch:03d}"
    save_val_samples = len(loader.dataset) if args.save_val_samples <= 0 else args.save_val_samples
    if save_val_samples > 0:
        save_root.mkdir(parents=True, exist_ok=True)
    eval_limit = len(loader.dataset) if args.val_batches <= 0 else min(args.val_batches, len(loader.dataset))
    save_limit = min(save_val_samples, len(loader.dataset))
    max_val_iters = max(eval_limit, save_limit)
    pbar = tqdm(
        loader,
        desc=f"val e{epoch}",
        total=max_val_iters,
        disable=not is_main(rank),
        leave=False,
    )
    for idx, batch in enumerate(pbar):
        if idx >= max_val_iters:
            break
        batch = move_batch(batch, device)
        gt = batch["motion_code"]
        audio_feat = batch["audio_feat"]
        llm_feat = batch.get("llm_feat")
        prev = shifted_motion(gt, raw_model)
        z_target = codec_quant_to_sum_feat_chunked(codec, gt)
        out = raw_model(audio_feat, prev, llm_feat=llm_feat)
        teacher_pred = out["pred_motion"]
        ar_pred = raw_model.generate(
            audio_feat,
            max_frames=gt.shape[1],
            llm_feat=llm_feat,
            max_context_frames=args.stream_context_frames,
        )
        teacher_l1 = F.l1_loss(teacher_pred, gt)
        teacher_prior = F.l1_loss(out["prior_motion"], gt)
        teacher_z = F.mse_loss(out["z_pred"], z_target)
        z_target_bits = (z_target > 0).to(out["z_pred"].dtype)
        teacher_z_bce = F.binary_cross_entropy_with_logits(out["z_pred"] * math.sqrt(args.code_dim), z_target_bits)
        teacher_z_bit_acc = ((out["z_pred"] > 0) == (z_target > 0)).float().mean()
        teacher_jaw = F.l1_loss(teacher_pred[..., 50:51], gt[..., 50:51])
        ar_l1 = F.l1_loss(ar_pred, gt)
        ar_jaw = F.l1_loss(ar_pred[..., 50:51], gt[..., 50:51])
        ar_vel = F.mse_loss(ar_pred[:, 1:] - ar_pred[:, :-1], gt[:, 1:] - gt[:, :-1])
        if idx < eval_limit:
            total["teacher_motion"] += float(teacher_l1.item())
            total["teacher_prior"] += float(teacher_prior.item())
            total["teacher_z"] += float(teacher_z.item())
            total["teacher_z_bce"] += float(teacher_z_bce.item())
            total["teacher_z_bit_acc"] += float(teacher_z_bit_acc.item())
            total["teacher_jaw"] += float(teacher_jaw.item())
            total["ar_motion"] += float(ar_l1.item())
            total["ar_jaw"] += float(ar_jaw.item())
            total["ar_vel"] += float(ar_vel.item())
            total["samples"] += 1.0
        if idx < save_limit:
            info = batch.get("infos", {})
            key = f"rank{rank:02d}_sample_{idx:04d}"
            if isinstance(info, dict) and "motion_key" in info:
                v = info["motion_key"]
                if isinstance(v, (list, tuple)):
                    v = v[0]
                key = f"rank{rank:02d}_{v}"
            safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in key)
            np.save(save_root / f"{safe}_pred.npy", motion54_to_56(ar_pred[0]).cpu().numpy())
            np.save(save_root / f"{safe}_teacher_pred.npy", motion54_to_56(teacher_pred[0]).cpu().numpy())
            np.save(save_root / f"{safe}_gt.npy", motion54_to_56(gt[0]).cpu().numpy())
    total = reduce_sums(total, device, world_size)
    metrics = {
        k: total[k] / max(total["samples"], 1.0)
        for k in (
            "teacher_motion",
            "teacher_prior",
            "teacher_z",
            "teacher_z_bce",
            "teacher_z_bit_acc",
            "teacher_jaw",
            "ar_motion",
            "ar_jaw",
            "ar_vel",
        )
    }
    metrics["samples"] = total["samples"]
    raw_model.train()
    return metrics


def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = v
        else:
            out[k] = v
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--codec-ckpt",
        default="/home6/duplex/VAE_ami_dualtalk/checkpoints/iter_100000.pt",
    )
    p.add_argument("--stats-path", default="/home6/duplex/dataset/artalk_mimi54_stats.json")
    p.add_argument("--output-dir", default="/home6/duplex/personaplex/moshi/moshi/ARTalk/train_code/outputs/SoftVQContinuousOnline_mimi54_dualtalk_ami_b512_Fixed_vae")
    p.add_argument("--dualtalk-root", default="/home6/duplex/dataset/dualtalk")
    p.add_argument("--mimi-root", default="/home6/duplex/dataset/mimi_emb")
    p.add_argument("--ami-bc-root", default="/home6/duplex/dataset/ami_flame/bc")
    p.add_argument("--ami-bc-mimi-root", default="/home6/duplex/dataset/mimi_emb")
    p.add_argument("--ami-ut-root", default="/home6/duplex/dataset/ami_flame/ut")
    p.add_argument("--ami-ut-mimi-root", default="/home6/duplex/dataset/mimi_emb")
    p.add_argument("--sources", default="dualtalk,ami_bc,ami_ut")
    p.add_argument("--length-mismatch-csv", default="/home6/duplex/dataset/artalk_mimi54_length_mismatches.csv")
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--clip-length", type=int, default=100)
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--code-dim", type=int, default=32)
    p.add_argument("--codec-hidden-dim", type=int, default=512)
    p.add_argument("--codec-depth", type=int, default=8)
    p.add_argument("--codec-heads", type=int, default=8)
    p.add_argument("--motion-weight", type=float, default=1.0)
    p.add_argument("--prior-weight", type=float, default=0.5)
    p.add_argument("--z-weight", type=float, default=0.2)
    p.add_argument("--z-bce-weight", type=float, default=0.1)
    p.add_argument("--jaw-weight", type=float, default=2.0)
    p.add_argument("--vel-weight", type=float, default=0.5)
    p.add_argument("--reg-weight", type=float, default=0.02)
    p.add_argument("--gate-weight", type=float, default=1.0)
    p.add_argument("--gate-target-expr", type=float, default=0.45)
    p.add_argument("--gate-target-jaw", type=float, default=0.25)
    p.add_argument("--gate-target-neck", type=float, default=0.65)
    p.add_argument("--gate-loss-expr-weight", type=float, default=4.0)
    p.add_argument("--gate-loss-jaw-weight", type=float, default=4.0)
    p.add_argument("--gate-loss-neck-weight", type=float, default=1.0)
    p.add_argument("--prior-warmup-frames", type=int, default=10)
    p.add_argument("--val-batches", type=int, default=0)
    p.add_argument("--val-every-epochs", type=int, default=100)
    p.add_argument("--save-val-samples", type=int, default=0)
    p.add_argument("--save-every-epochs", type=int, default=100)
    p.add_argument("--stream-context-frames", type=int, default=25)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb-project", default="NIPS_duplex_SoftVQ_Continuous")
    p.add_argument("--wandb-run-name", default="softvq_continuous_online_mimi54_dualtalk_ami_b512_ddp6_Fixed_vae")
    args = p.parse_args()
    if not args.codec_ckpt:
        args.codec_ckpt = DEFAULT_CODEC_CKPT
    return args


def main():
    args = parse_args()
    ddp, rank, local_rank, world_size, device = detect_ddp()
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")

    out_dir = Path(args.output_dir)
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        print(
            "[softvq-continuous] model=causal_transformer no_lookahead=True "
            "step=1_mimi_to_2_flame output=54D saved=56D"
        )
        print(f"[softvq-continuous] world_size={world_size} batch_size={args.batch_size} epochs={args.epochs}")

    data_cfg = make_dataset_cfg(args)
    train_ds = MimiFlameData(data_cfg, "train", debug=False)
    val_ds = MimiFlameData(data_cfg, "val", debug=False)
    if is_main(rank):
        print(f"[softvq-continuous] train_windows={len(train_ds)} val_records={len(val_ds)}")

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_eval_ds = Subset(val_ds, list(range(rank, len(val_ds), world_size))) if ddp else val_ds
    val_loader = DataLoader(val_eval_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    codec = load_codec(args, device)
    model = CausalSoftVQContinuousTransformer(
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        code_dim=args.code_dim,
        prior_warmup_frames=args.prior_warmup_frames,
    ).to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else (torch.float16 if args.precision == "fp16" else None)

    wb = None
    if args.wandb and is_main(rank):
        import wandb

        wb = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args), dir=str(out_dir))

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", disable=not is_main(rank), dynamic_ncols=True)
        epoch_sum = {}
        steps = 0
        for batch in pbar:
            if args.max_train_batches > 0 and steps >= args.max_train_batches:
                break
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            loss, metrics = compute_loss(model, codec, batch, args, autocast_dtype=autocast_dtype)
            if args.precision == "fp16":
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
            metrics["grad_norm"] = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
            metrics["lr"] = float(opt.param_groups[0]["lr"])
            global_step += 1
            steps += 1
            for k, v in metrics.items():
                epoch_sum[k] = epoch_sum.get(k, 0.0) + v
            if is_main(rank):
                pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items() if k in ("loss", "motion", "jaw", "z")})
                if wb is not None:
                    wb.log({f"train/{k}": v for k, v in metrics.items()} | {"train/epoch": epoch}, step=global_step)
        epoch_metrics = {k: v / max(steps, 1) for k, v in epoch_sum.items()}
        epoch_metrics = reduce_metrics(epoch_metrics, device, world_size)
        should_save = (
            args.save_every_epochs > 0 and epoch % args.save_every_epochs == 0
        ) or epoch == args.epochs
        if should_save and is_main(rank):
            ckpt_model = model.module if isinstance(model, DDP) else model
            ckpt = {
                "model": ckpt_model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "global_step": global_step,
            }
            torch.save(ckpt, out_dir / f"checkpoint_epoch_{epoch:03d}.ckpt")
        if ddp:
            dist.barrier()
        should_validate = args.val_every_epochs > 0 and (
            epoch % args.val_every_epochs == 0 or epoch == args.epochs
        )
        val_metrics = {}
        if should_validate:
            val_metrics = validate(model, codec, val_loader, device, args, epoch, out_dir, rank, world_size)
        if is_main(rank):
            print(
                f"[softvq-continuous] epoch={epoch} "
                + " ".join(f"train_{k}={v:.4f}" for k, v in sorted(epoch_metrics.items()))
                + " "
                + " ".join(f"val_{k}={v:.4f}" for k, v in sorted(val_metrics.items()))
            )
            if wb is not None:
                log_payload = {f"epoch/train_{k}": v for k, v in epoch_metrics.items()}
                if val_metrics:
                    log_payload.update({f"val/{k}": v for k, v in val_metrics.items()})
                wb.log(log_payload, step=global_step)
        if ddp:
            dist.barrier()
    if wb is not None:
        wb.finish()
    cleanup_ddp(ddp)


if __name__ == "__main__":
    main()



"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 softvq_continuous_online_train.py   --codec-ckpt $CODEC_CKPT   --batch-size 512   --epochs
5000   --precision bf16   --output-dir /home6/duplex/personaplex/moshi/moshi/ARTalk/train_code/outputs/SoftVQContinuousOnline_mimi54_dualtalk_ami_b512   --sour
ces dualtalk,ami_bc,ami_ut   --val-batches 16   --save-val-samples 0   --val-every-epochs 50   --save-every-epochs 50   --num-workers 8   --wandb   --wandb-run
-name softvq_continuous_online_mimi54_dualtalk_ami_b512_ddp6





"""
