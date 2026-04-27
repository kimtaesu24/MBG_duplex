import math

import torch
from torch.nn import functional as F


@torch.no_grad()
def _codec_quant_to_sum_feat_chunked(codec, motion: torch.Tensor) -> torch.Tensor:
    """Run ARTalkCodec.quant_to_sum_feat in 100-frame chunks (codec length constraint)."""
    chunk_len = int(getattr(codec, "patch_nums", [100])[-1])
    chunks = []
    for start in range(0, motion.shape[1], chunk_len):
        chunk = motion[:, start : start + chunk_len]
        actual_len = chunk.shape[1]
        if actual_len < chunk_len:
            pad = chunk.new_zeros(chunk.shape[0], chunk_len - actual_len, chunk.shape[2])
            chunk = torch.cat([chunk, pad], dim=1)
        feat = codec.quant_to_sum_feat(chunk)
        chunks.append(feat[:, :actual_len])
    return torch.cat(chunks, dim=1)[:, : motion.shape[1]]


def compute_face_loss(
    face_outputs: dict,
    gt_face_motion: torch.Tensor,
    codec,
    face_args,
) -> torch.Tensor:
    """Compute the full reference face-motion training loss.

    Replicates the loss from softvq_continuous_online_train.py::compute_loss(),
    including motion L1, prior L1, z MSE/BCE, component-wise jaw L1, velocity,
    regularisation, and gate-target MSE terms.

    Args:
        face_outputs: dict returned by CausalSoftVQContinuousTransformer.forward().
        gt_face_motion: [B, T_face, 54] ground-truth 3DMM at 25 fps.
        codec: frozen ARTalkCodec; used to compute z_target (no_grad).
        face_args: FaceGenArgs dataclass carrying per-loss weights.

    Returns:
        Scalar loss tensor.
    """
    pred   = face_outputs["pred_motion"]    # [B, T, 54]
    prior  = face_outputs["prior_motion"]   # [B, T, 54]
    z_pred = face_outputs["z_pred"]         # [B, T, code_dim]
    delta  = face_outputs["delta"]          # [B, T, 54]
    residual = face_outputs["residual"]     # [B, T, 54]
    group_gate = face_outputs["group_gate"] # [B, T, 3]

    gt = gt_face_motion

    # z-space target (frozen codec, no gradient)
    z_target = _codec_quant_to_sum_feat_chunked(codec, gt)  # [B, T, code_dim]
    z_target = z_target.to(dtype=z_pred.dtype, device=z_pred.device)

    code_dim = z_pred.shape[-1]

    loss_motion = F.l1_loss(pred, gt)
    loss_prior  = F.l1_loss(prior, gt)
    loss_z      = F.mse_loss(z_pred, z_target)

    z_target_bits = (z_target > 0).to(z_pred.dtype)
    loss_z_bce    = F.binary_cross_entropy_with_logits(
        z_pred * math.sqrt(code_dim), z_target_bits
    )

    loss_jaw = F.l1_loss(pred[..., 50:51], gt[..., 50:51])

    pred_vel = pred[:, 1:] - pred[:, :-1]
    gt_vel   = gt[:, 1:]   - gt[:, :-1]
    loss_vel = F.mse_loss(pred_vel, gt_vel)

    loss_reg = delta.pow(2).mean() + residual.pow(2).mean()

    gate_target = group_gate.new_tensor(
        [face_args.gate_target_expr, face_args.gate_target_jaw, face_args.gate_target_neck]
    ).view(1, 1, 3)
    gate_group_weight = group_gate.new_tensor(
        [face_args.gate_loss_expr_weight, face_args.gate_loss_jaw_weight, face_args.gate_loss_neck_weight]
    ).view(1, 1, 3)
    loss_gate = ((group_gate - gate_target) ** 2 * gate_group_weight).mean()

    total = (
        face_args.motion_weight * loss_motion
        + face_args.prior_weight * loss_prior
        + face_args.z_weight * loss_z
        + face_args.z_bce_weight * loss_z_bce
        + face_args.jaw_weight * loss_jaw
        + face_args.vel_weight * loss_vel
        + face_args.reg_weight * loss_reg
        + face_args.gate_weight * loss_gate
    )
    return total


def compute_loss_with_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str,
    first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 1.0,
    text_padding_ids: set[int] | None = None,
):
    target = torch.where(target_mask, target, torch.zeros_like(target))

    weights = target_mask.float()
    if mode == "audio":
        weights[:, 0] *= first_codebook_weight_multiplier
    elif mode == "text":
        assert text_padding_ids is not None
        for id in text_padding_ids:
            weights[target == id] *= text_padding_weight

    logits = logits.reshape(-1, logits.size(-1)).float()
    target = target.reshape(-1)
    weights = weights.reshape(-1)
    mb_loss = F.cross_entropy(logits, target, reduction="none")
    mb_loss = torch.where(weights > 0.0, mb_loss * weights, torch.zeros_like(mb_loss))
    mb_loss = torch.sum(mb_loss) / torch.sum(weights).clamp(min=1e-8)

    return mb_loss
