import logging
from typing import Iterator, Optional

import torch
import torch.cuda
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from torch.nn import functional as F

from finetune.args import TrainArgs

from .data.data_loader import Batch
from .distributed import get_rank, get_world_size
from .loss import compute_loss_with_mask, compute_face_loss
from .utils import TrainState

logger = logging.getLogger("eval")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def evaluate(
    model: FullyShardedDataParallel,
    eval_data_loader: Iterator[Batch],
    state: TrainState,
    args: TrainArgs,
    mimi=None,
    face_codec=None,
    T_p: int = 0,
    prompt_ids: Optional[list] = None,
    param_dtype=None,
):
    if param_dtype is None:
        param_dtype = torch.bfloat16

    num_samples = torch.tensor([0], device="cuda", dtype=torch.long)
    text_loss_accum       = torch.tensor(0.0, device="cuda")
    audio_loss_accum      = torch.tensor(0.0, device="cuda")
    vap_loss_accum        = torch.tensor(0.0, device="cuda")
    commitment_loss_accum = torch.tensor(0.0, device="cuda")
    face_loss_accum       = torch.tensor(0.0, device="cuda")
    bc_event_loss_accum   = torch.tensor(0.0, device="cuda")

    max_eval_batches = max(40 // get_world_size(), 1)
    model.eval()

    for batch in eval_data_loader:
        if num_samples.item() >= max_eval_batches:
            break
        with torch.no_grad():
            codes = batch.codes  # [B, K, T]

            # ── Text prompt prefix ────────────────────────────────────────
            if T_p:
                prefix = torch.full(
                    [codes.shape[0], codes.shape[1], T_p],
                    model.zero_token_id,
                    device=codes.device, dtype=codes.dtype,
                )
                if prompt_ids:
                    prefix[:, 0, :] = torch.tensor(prompt_ids, device=codes.device)
                codes_in = torch.cat([prefix, codes], dim=2)
            else:
                codes_in = codes

            # ── Voice prompt embeddings ───────────────────────────────────
            voice_prompt_embs = batch.voice_prompt_embs
            if voice_prompt_embs is not None:
                voice_prompt_embs = voice_prompt_embs.to(codes.device, non_blocking=True)

            # ── Face generation inputs ────────────────────────────────────
            # mimi is passed directly to the model for VapGPT bc_audio_feats
            # extraction (handled inside lm.forward_train).
            # For face gen, we mirror train.py: teacher-forced audio_feat is
            # precomputed here; generated-audio mode passes mimi to the model.
            audio_feat = None
            gt_face_motion = None
            mimi_for_model = mimi  # always pass mimi so VapGPT bc can auto-extract

            if args.face_gen.enable and mimi is not None:
                if args.face_gen.use_generated_audio_feat:
                    # Model decodes its own predicted codes; mimi_for_model already set.
                    pass
                else:
                    # Teacher-forced: decode GT audio codes → Mimi latents.
                    audio_codes = codes[:, 1:9].clamp(min=0)
                    audio_feat = mimi.decode_latent(audio_codes).transpose(1, 2)  # [B, T, 512]
                    audio_feat = audio_feat.to(dtype=param_dtype)
                    if T_p:
                        zero_feat = torch.zeros(
                            codes.shape[0], T_p, audio_feat.shape[2],
                            device=audio_feat.device, dtype=audio_feat.dtype,
                        )
                        audio_feat = torch.cat([zero_feat, audio_feat], dim=1)

                if batch.face_motion_gt is not None:
                    gt_face_motion = batch.face_motion_gt.to(codes.device, dtype=param_dtype)
                    if T_p:
                        T_face_p = T_p * 2  # face runs at 25 fps = 2× mimi 12.5 fps
                        zero_motion = torch.zeros(
                            codes.shape[0], T_face_p, gt_face_motion.shape[2],
                            device=gt_face_motion.device, dtype=gt_face_motion.dtype,
                        )
                        gt_face_motion = torch.cat([zero_motion, gt_face_motion], dim=1)

            output = model(codes_in, step=state.step,
                           voice_prompt_embs=voice_prompt_embs,
                           audio_feat=audio_feat,
                           gt_face_motion=gt_face_motion,
                           mimi=mimi_for_model,
                           bc_audio_feats=None)  # auto-extracted inside lm when mimi is set

            # Slice off T_p prompt-prefix frames — loss is on the conversation only.
            text_mask  = output.text_mask[:, :, T_p:]
            audio_mask = output.mask[:, :, T_p:]

            text_loss = compute_loss_with_mask(
                output.text_logits[:, :, T_p:],
                codes[:, : model.audio_offset],
                text_mask,
                mode="text",
                text_padding_weight=args.text_padding_weight,
                text_padding_ids={model.text_padding_token_id},
                epad_ids={model.end_of_text_padding_id},
                epad_weight=args.epad_weight,
            )
            audio_loss = compute_loss_with_mask(
                output.logits[:, :, T_p:],
                codes[:, model.audio_offset : model.audio_offset + model.dep_q],
                audio_mask,
                mode="audio",
                first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
            )
            text_loss_accum  += text_loss
            audio_loss_accum += audio_loss

            # ── VAP / commitment / bc-event losses ────────────────────────
            if args.backchannel.enable:
                vap_logits = output.vap_logits[:, T_p:] if output.vap_logits is not None else None
                if vap_logits is not None:
                    vap_targets_tensor = batch.vap_targets
                    if vap_targets_tensor is not None and batch.valid_mask is not None:
                        vap_targets_tensor = vap_targets_tensor.masked_fill(
                            ~batch.valid_mask.to(vap_targets_tensor.device), -100
                        )
                    if vap_targets_tensor is not None:
                        flat_targets = vap_targets_tensor.view(-1).long()
                        n_valid = (flat_targets != -100).sum().item()
                        if n_valid > 0:
                            num_vap_classes = vap_logits.shape[-1]
                            flat_targets = flat_targets.clamp(-100, num_vap_classes - 1)
                            vap_loss = F.cross_entropy(
                                vap_logits.reshape(-1, num_vap_classes),
                                flat_targets,
                                ignore_index=-100,
                            )
                            if not torch.isnan(vap_loss):
                                vap_loss_accum += vap_loss

                if (output.commitment_loss is not None
                        and not torch.isnan(output.commitment_loss)):
                    commitment_loss_accum += output.commitment_loss

                if (args.backchannel.bc_event_loss_weight > 0
                        and output.bc_logits is not None
                        and batch.bc_timing_targets is not None):
                    bc_targets    = batch.bc_timing_targets.to(codes.device)
                    bc_logit_pos  = output.bc_logits[:, T_p:, 1]
                    valid_bc      = bc_targets != -100
                    if valid_bc.any():
                        bce = F.binary_cross_entropy_with_logits(
                            bc_logit_pos[valid_bc],
                            bc_targets[valid_bc],
                            pos_weight=torch.tensor(
                                args.backchannel.bc_focal_pos_weight,
                                device=codes.device, dtype=bc_logit_pos.dtype,
                            ),
                            reduction='none',
                        )
                        p_t = torch.exp(-bce.detach())
                        bc_event_loss = ((1.0 - p_t) ** args.backchannel.bc_focal_gamma * bce).mean()
                        bc_event_loss_accum += bc_event_loss

            # ── Face loss ─────────────────────────────────────────────────
            if (args.face_gen.enable
                    and output.face_outputs is not None
                    and gt_face_motion is not None
                    and face_codec is not None):
                T_face_p = T_p * 2
                valid_face_mask = None
                if T_p or batch.valid_face_frames is not None:
                    T_face = gt_face_motion.shape[1]
                    t_idx = torch.arange(T_face, device=codes.device).unsqueeze(0)
                    if batch.valid_face_frames is not None:
                        valid_face_mask = (t_idx >= T_face_p) & (
                            t_idx < T_face_p + batch.valid_face_frames.to(codes.device).unsqueeze(1)
                        )
                    else:
                        valid_face_mask = t_idx >= T_face_p
                face_loss = compute_face_loss(
                    output.face_outputs, gt_face_motion, face_codec, args.face_gen,
                    valid_face_mask=valid_face_mask,
                )
                if torch.isfinite(face_loss):
                    face_loss_accum += face_loss

        num_samples += 1

    # ── Distributed aggregation ───────────────────────────────────────────
    all_num_samples = [torch.zeros_like(num_samples) for _ in range(get_world_size())]
    torch.distributed.all_gather(all_num_samples, num_samples)
    total_num_samples = int(torch.tensor(all_num_samples).sum().item())
    main_logger_info("Eval finished!")

    if total_num_samples == 0:
        main_logger_info("Warning: eval_data_loader was empty, skipping eval metrics.")
        model.train()
        return

    eval_loss = text_loss_accum + audio_loss_accum
    for t in (eval_loss, text_loss_accum, audio_loss_accum):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    eval_loss       /= total_num_samples
    text_loss_accum  /= total_num_samples
    audio_loss_accum /= total_num_samples

    state.this_eval_loss        = eval_loss.item()
    state.this_eval_perplexity  = (2 ** eval_loss).item()
    state.this_audio_loss       = audio_loss_accum.item()
    state.this_text_loss        = text_loss_accum.item()

    if args.backchannel.enable:
        for t in (vap_loss_accum, commitment_loss_accum):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        vap_loss_accum        /= total_num_samples
        commitment_loss_accum /= total_num_samples
        state.this_vap_loss        = vap_loss_accum.item()
        state.this_commitment_loss = commitment_loss_accum.item()

    if args.face_gen.enable:
        dist.all_reduce(face_loss_accum, op=dist.ReduceOp.SUM)
        face_loss_accum /= total_num_samples
        state.this_face_loss = face_loss_accum.item()

    model.train()
