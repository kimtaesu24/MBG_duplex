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
from .loss import (
    compute_loss_with_mask,
    compute_face_loss,
    epad_confusion_counts,
    epad_metrics_from_counts,
)
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
    vad_loss_accum        = torch.tensor(0.0, device="cuda")
    # LA prior counts, local to this eval run (mirrors train2's running counts).
    bc_class_counts       = torch.ones(3, device="cuda", dtype=torch.float64)
    epad_counts           = torch.zeros(4, device="cuda")  # [tp, fp, fn, tn] for [EPAD]

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
                    if isinstance(prompt_ids[0], list):
                        dataset_idx = batch.get("dataset_idx")
                        if dataset_idx is None:
                            raise ValueError(
                                "Missing dataset_idx in eval batch for multi-prompt evaluation."
                            )
                        ds_indices = dataset_idx.tolist()
                        for b_idx, ds_idx in enumerate(ds_indices):
                            p_tokens = prompt_ids[ds_idx % len(prompt_ids)]
                            curr_len = len(p_tokens)
                            prefix[b_idx, 0, T_p - curr_len:] = torch.tensor(
                                p_tokens, device=codes.device
                            )
                    else:
                        curr_len = len(prompt_ids)
                        prefix[:, 0, T_p - curr_len:] = torch.tensor(
                            prompt_ids, device=codes.device
                        )
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
            gt_face_motion = batch.face_motion_gt
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

                if gt_face_motion is not None:
                    gt_face_motion = gt_face_motion.to(codes.device, dtype=param_dtype)
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

            # [EPAD] 예측 confusion counts 집계
            epad_counts += epad_confusion_counts(
                output.text_logits[:, :, T_p:],
                codes[:, : model.audio_offset],
                text_mask,
                model.end_of_text_padding_id,
            )

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

                # v2.1: 3-class CE from GT text tokens — mirrors train2.py exactly
                # (EPAD = actual [EPAD] token positions; boundary-ignore ±K PADs).
                if (args.backchannel.bc_event_loss_weight > 0
                        and output.bc_logits is not None):
                    txt = codes[:, 0]
                    pad_id = model.text_padding_token_id
                    epad_id = model.end_of_text_padding_id
                    is_epad = txt == epad_id
                    is_pad_t = txt == pad_id
                    is_word = ((~is_pad_t) & (~is_epad)
                               & (txt != 1) & (txt != 2) & (txt >= 0))
                    cls_tgt = torch.full_like(txt, -100)
                    cls_tgt[is_pad_t] = 0
                    cls_tgt[is_word] = 2
                    cls_tgt[is_epad] = 1
                    K = int(getattr(args.backchannel, "bc_onset_ignore_frames", 2))
                    if K > 0 and is_epad.any():
                        dil = F.max_pool1d(
                            is_epad.float().unsqueeze(1), kernel_size=2 * K + 1,
                            stride=1, padding=K,
                        ).squeeze(1).bool()
                        cls_tgt[dil & is_pad_t] = -100
                    if batch.valid_mask is not None:
                        cls_tgt = cls_tgt.masked_fill(~batch.valid_mask.to(codes.device), -100)
                    bc_logits_c = output.bc_logits[:, T_p:]  # [B, T, 3]
                    if (cls_tgt != -100).any():
                        # Logit-Adjustment loss — mirrors train2.py.
                        valid_t = cls_tgt[cls_tgt != -100]
                        bc_class_counts += torch.bincount(valid_t, minlength=3).to(bc_class_counts)
                        log_prior = torch.log(bc_class_counts / bc_class_counts.sum()).to(
                            device=codes.device, dtype=torch.float32,
                        )
                        bc_logits_adj = bc_logits_c.float() + args.backchannel.bc_la_tau * log_prior
                        bc_event_loss = F.cross_entropy(
                            bc_logits_adj.reshape(-1, 3),
                            cls_tgt.reshape(-1),
                            ignore_index=-100,
                        )
                        bc_event_loss_accum += bc_event_loss

                # v2: current-frame VAD BCE (both streams), mirrors train2.py.
                if (args.backchannel.vad_loss_weight > 0
                        and getattr(output, "vad_logits", None) is not None
                        and batch.vad_targets is not None):
                    vad_t = batch.vad_targets.to(codes.device)          # [B, 2, T]
                    vad_l = output.vad_logits[:, T_p:].transpose(1, 2)  # [B, 2, T]
                    if batch.valid_mask is not None:
                        vmask = batch.valid_mask.to(codes.device).unsqueeze(1).expand_as(vad_t)
                    else:
                        vmask = torch.ones_like(vad_t, dtype=torch.bool)
                    if vmask.any():
                        vad_loss_accum += F.binary_cross_entropy_with_logits(
                            vad_l[vmask].float(), vad_t[vmask].float(),
                        )

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

    # [EPAD] 예측 metric: confusion counts를 전체 rank에서 합산 후 acc/recall/f1 계산
    dist.all_reduce(epad_counts, op=dist.ReduceOp.SUM)
    state.this_eval_epad_metrics = epad_metrics_from_counts(epad_counts, prefix="epad")
    main_logger_info(
        "[EPAD] eval "
        f"acc={state.this_eval_epad_metrics['epad_acc']:.4f} "
        f"precision={state.this_eval_epad_metrics['epad_precision']:.4f} "
        f"recall={state.this_eval_epad_metrics['epad_recall']:.4f} "
        f"f1={state.this_eval_epad_metrics['epad_f1']:.4f}"
    )

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
