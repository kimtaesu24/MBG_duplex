import dataclasses
import logging
import os
import sys
import shutil
import contextlib
from contextlib import ExitStack

# Ensure local moshi source takes precedence over installed site-packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../moshi")))

import pprint
from pathlib import Path

# Personaplex rope.py의 apply_rope가 @torch.compile로 감싸져 있는데,
# gradient checkpointing + FSDP와 함께 쓰면 inductor가 FakeTensor 충돌을 일으킴.
# 학습에서는 torch.compile을 비활성화하고 eager 모드로 실행합니다.
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # compile 실패 시 eager로 fallback
os.environ["TORCHDYNAMO_DISABLE"] = "1"      # torch.compile 전역 비활성화

import fire
import torch
import torch.cuda
import torch.distributed as dist
from torch.optim import AdamW, lr_scheduler

from torch.nn import functional as F

from finetune.args import TrainArgs
from finetune.checkpointing import Checkpointer
from finetune.data.data_loader import build_data_loader
from finetune.data.interleaver import InterleavedTokenizer, Interleaver
from finetune.distributed import (
    BACKEND,
    avg_aggregate,
    get_rank,
    get_world_size,
    is_torchrun,
    set_device,
)
from finetune.eval import evaluate
from finetune.loss import compute_loss_with_mask, compute_face_loss
from finetune.monitoring.metrics_logger import (
    MetricsLogger,
    eval_log_msg,
    get_eval_logs,
    get_train_logs,
    train_log_msg,
)
from finetune.monitoring.utils import set_logger
from finetune.utils import TrainState, logged_closing, set_random_seed
from finetune.wrapped_model import get_fsdp_model

# personaplex_MBG/moshi 패키지 (Python 모듈명: moshi)
from moshi.models.loaders import (
    get_mimi,
    get_moshi_lm,
    _lm_kwargs,
    MIMI_NAME,
    MOSHI_NAME,
    TEXT_TOKENIZER_NAME,
    DEFAULT_REPO,
)

logger = logging.getLogger("train")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def train(config: str):
    """진입점: YAML 설정 파일 경로를 인수로 받습니다.

    사용법:
        torchrun --nproc_per_node=<N_GPUS> train.py config/example.yaml
    """
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    set_logger(logging.INFO)

    with ExitStack() as exit_stack:
        _train(args, exit_stack)
    logger.info("학습 종료!")


def _train(args: TrainArgs, exit_stack: ExitStack):
    # ── 1. 초기 설정 ──────────────────────────────────────────────────────
    set_random_seed(args.seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # NCCL 초기화
    if "LOCAL_RANK" in os.environ:
        set_device()
        logger.info("분산 통신 초기화 중...")
        dist.init_process_group(backend=BACKEND)
    else:
        logger.error(
            "PyTorch 분산 환경이 초기화되지 않았습니다. "
            "torchrun으로 실행하세요."
        )

    # ── 2. run_dir 초기화 ─────────────────────────────────────────────────
    main_logger_info(f"Run dir: {args.run_dir}")
    run_dir = Path(args.run_dir)

    if is_torchrun():
        if run_dir.exists() and not args.overwrite_run_dir:
            raise RuntimeError(
                f"Run dir {run_dir} 이미 존재합니다. "
                f"`run_dir` 이름을 변경하거나 `overwrite_run_dir: true`를 설정하세요."
            )
        elif run_dir.exists():
            main_logger_info(f"Run dir 삭제 중: {run_dir}...")
            shutil.rmtree(run_dir)

    # (LoRA 모드는 args.py 및 wrapped_model.py에서 처리됨)
    pass

    dist.barrier()
    run_dir.mkdir(exist_ok=True, parents=True)

    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        args.save(args_path)

    main_logger_info(f"TrainArgs:\n{pprint.pformat(dataclasses.asdict(args))}")

    # ── 3. 로거 초기화 ────────────────────────────────────────────────────
    metrics_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="train",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(metrics_logger, "metrics_logger"))

    eval_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="eval",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(eval_logger, "eval_logger"))

    # ── 4. Mimi(오디오 토크나이저) & 텍스트 토크나이저 로드 ─────────────
    main_logger_info("Mimi 및 텍스트 토크나이저 로드 중 (personaplex)...")
    hf_repo = args.moshi_paths.hf_repo_id or DEFAULT_REPO

    mimi_path = args.moshi_paths.mimi_path
    moshi_path = args.moshi_paths.moshi_path
    tokenizer_path = args.moshi_paths.tokenizer_path

    # 로컬 경로 미설정 시 HuggingFace에서 다운로드
    if mimi_path is None or moshi_path is None or tokenizer_path is None:
        main_logger_info(
            f"경로 미설정. HF 레포에서 다운로드 중: {hf_repo}"
        )
        try:
            from huggingface_hub import hf_hub_download
            if mimi_path is None:
                mimi_path = hf_hub_download(hf_repo, MIMI_NAME)
                main_logger_info(f"  mimi → {mimi_path}")
            if moshi_path is None:
                moshi_path = hf_hub_download(hf_repo, MOSHI_NAME)
                main_logger_info(f"  moshi → {moshi_path}")
            if tokenizer_path is None:
                tokenizer_path = hf_hub_download(hf_repo, TEXT_TOKENIZER_NAME)
                main_logger_info(f"  tokenizer → {tokenizer_path}")
        except Exception as e:
            raise RuntimeError(
                f"{hf_repo} 에서 다운로드 실패: {e}\n"
                "config에 moshi_paths.mimi_path, moshi_paths.moshi_path, "
                "moshi_paths.tokenizer_path를 직접 지정해주세요."
            )

    mimi = get_mimi(mimi_path, device="cuda")
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    # ── 5. LM 설정 결정 ───────────────────────────────────────────────────
    lm_config = dict(_lm_kwargs)
    if args.moshi_paths.config_path is not None:
        import json
        with open(args.moshi_paths.config_path, "r") as f:
            lm_config = json.load(f)
        main_logger_info(f"커스텀 LM 설정 로드: {args.moshi_paths.config_path}")
    else:
        main_logger_info("기본 _lm_kwargs 사용 (personaplex loaders)")

    # get_moshi_lm과 동일하게 dep_q=16 오버라이드
    # (_lm_kwargs 기본값은 dep_q=8이지만 체크포인트는 dep_q=16으로 학습됨)
    lm_config["dep_q"] = 16

    # ── 5-1. Backchannel / VAP 설정 ──────────────────────────────────────
    if args.backchannel.enable:
        lm_config["backchannel_enabled"] = True
        lm_config["backchannel_vap_dim"] = args.backchannel.vap_dim
        lm_config["backchannel_bc_hidden"] = args.backchannel.bc_hidden
        lm_config["backchannel_gumbel_temp_init"] = args.backchannel.gumbel_temp_init
        lm_config["backchannel_gumbel_temp_min"] = args.backchannel.gumbel_temp_min
        lm_config["backchannel_gumbel_anneal_rate"] = args.backchannel.gumbel_anneal_rate
        if args.backchannel.pad_token_id is not None:
            lm_config["backchannel_pad_token_id"] = args.backchannel.pad_token_id
        if args.backchannel.epad_token_id is not None:
            lm_config["backchannel_epad_token_id"] = args.backchannel.epad_token_id

        if args.backchannel.module_type == "mlp":
            main_logger_info("Backchannel MLP VAP 모듈 활성화 (LM 내부)")
            lm_config["backchannel_module_type"] = "mlp"
        elif args.backchannel.module_type == "vap_gpt":
            main_logger_info("VapGPT Backchannel 모듈 활성화 (LM 내부, z_s 입력)")
            lm_config["backchannel_module_type"] = "vap_gpt"
            lm_config["backchannel_vap_repo_path"] = args.backchannel.vap_gpt_repo_path
            lm_config["backchannel_vap_checkpoint"] = args.backchannel.vap_gpt_checkpoint
            lm_config["backchannel_vap_channel_layers"] = args.backchannel.vap_gpt_channel_layers
            lm_config["backchannel_vap_cross_layers"] = args.backchannel.vap_gpt_cross_layers
            lm_config["backchannel_vap_num_heads"] = args.backchannel.vap_gpt_num_heads
            lm_config["backchannel_vap_dropout"] = args.backchannel.vap_gpt_dropout
            main_logger_info(
                f"VapGPTBackchannelModule 설정 완료 "
                f"(checkpoint={args.backchannel.vap_gpt_checkpoint})"
            )
        else:
            raise ValueError(
                f"알 수 없는 backchannel.module_type: '{args.backchannel.module_type}'. "
                "지원 값: 'mlp', 'vap_gpt'"
            )

    # ── 5-2. Face Generation 설정 ─────────────────────────────────────────
    if args.face_gen.enable:
        face_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../moshi/moshi/models/face",
        ))
        lm_config["face_module_enabled"] = True
        lm_config["face_module_dir"] = face_dir
        lm_config["face_module_checkpoint"] = args.face_gen.ckpt_path
        lm_config["face_module_hidden_dim"] = args.face_gen.hidden_dim
        lm_config["face_module_layers"] = args.face_gen.layers
        lm_config["face_module_heads"] = args.face_gen.heads
        lm_config["face_module_code_dim"] = args.face_gen.code_dim
        lm_config["face_module_prior_warmup_frames"] = args.face_gen.prior_warmup_frames
        main_logger_info(
            f"Face generation module 활성화 "
            f"(checkpoint={args.face_gen.ckpt_path})"
        )

    # ── 6. LM 모델 로드 및 FSDP 샤딩 ─────────────────────────────────────
    main_logger_info("Personaplex LM 모델 로드 중...")
    model = get_fsdp_model(args, moshi_path, lm_config)

    # ── 6-1. ARTalkCodec (VAE) 로드 (face_gen 활성 시) ────────────────────
    # The codec is frozen and used only to compute z_target = quant_to_sum_feat(gt_face_motion).
    # It is NOT wrapped in FSDP — each rank holds a full copy (codec is small, ~50M params).
    face_codec = None
    if args.face_gen.enable and args.face_gen.codec_ckpt_path:
        from types import SimpleNamespace
        face_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../moshi/moshi/models/face",
        ))
        if face_dir not in sys.path:
            sys.path.insert(0, face_dir)
        from core.models.artalk_codec import ARTalkCodec  # noqa: PLC0415
        codec_cfg = SimpleNamespace(
            MOTION_DIM=54,
            V_CODE_DIM=args.face_gen.code_dim,
            V_PATCH_NUMS=[1, 10, 20, 50, 100],
            T_DEPTH=args.face_gen.codec_depth,
            T_NUM_HEADS=args.face_gen.codec_heads,
            T_HIDDEN_DIM=args.face_gen.codec_hidden_dim,
            STATS_PATH=args.face_gen.codec_stats_path or "",
        )
        face_codec = ARTalkCodec(codec_cfg, init_submodule=False)
        _codec_ckpt = torch.load(args.face_gen.codec_ckpt_path, map_location="cpu", weights_only=True)
        face_codec.load_state_dict(_codec_ckpt["model"], strict=True)
        face_codec.eval().cuda()
        for p in face_codec.parameters():
            p.requires_grad = False
        main_logger_info(f"ARTalkCodec loaded from: {args.face_gen.codec_ckpt_path}")

    # ── 7. 텍스트 토크나이저 & Interleaver 구성 ───────────────────────────
    main_logger_info("텍스트 토크나이저 로드 중...")
    import sentencepiece
    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(tokenizer_path)

    interleaver = Interleaver(
        spm,
        mimi.frame_rate,
        model.text_padding_token_id,
        model.end_of_text_padding_id,
        model.zero_token_id,
        keep_main_only=True,
    )
    interleaved_tokenizer = InterleavedTokenizer(
        mimi, interleaver, duration_sec=args.duration_sec,
        vap_manifest_path=args.data.vap_manifest,
        flame_root=args.face_gen.flame_root if args.face_gen.enable else "",
        flame_speaker=args.face_gen.flame_speaker,
    )

    # ── 8. 데이터 로더 구성 ───────────────────────────────────────────────
    data_loader = build_data_loader(
        instruct_tokenizer=interleaved_tokenizer,
        args=args.data,
        batch_size=args.batch_size,
        seed=args.seed,
        rank=get_rank(),
        world_size=get_world_size(),
        is_eval=False,
    )

    # eval_data_loader은 매 평가 시마다 새로 생성합니다.
    # is_eval=True의 유한 이터레이터를 재사용하면 백그라운드 스레드가 데이터셋을
    # 소진한 후 종료되고, 이후 _q.get()이 영구 블록됩니다.

    # ── 9. dtype 설정 ─────────────────────────────────────────────────────
    param_dtype = getattr(torch, args.param_dtype)

    # ── 10. 옵티마이저 & 스케줄러 ─────────────────────────────────────────
    trainable_params = list(model.parameters())
    optimizer = AdamW(
        trainable_params,
        lr=args.optim.lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.optim.lr,
        total_steps=args.max_steps,
        pct_start=args.optim.pct_start,
    )

    state = TrainState(args.max_steps)

    # ── 11. 체크포인터 초기화 ─────────────────────────────────────────────
    if args.do_ckpt:
        checkpointer = Checkpointer(
            model=model,
            state=state,
            config=lm_config,
            run_dir=run_dir,
            optimizer=optimizer,
            num_ckpt_keep=args.num_ckpt_keep,
            full_finetuning=args.full_finetuning,
        )

    # ── 12. 학습 준비 ──────────────────────────────────────────────────────
    # FSDP MixedPrecision이 bf16 변환을 내부적으로 처리합니다.
    # prepare_mixed_precision은 불필요(fp32 복사본을 별도 유지하여 메모리 낭비)
    scaler = None  # bf16 사용 시 GradScaler 불필요

    # ── 13. 학습 루프 ─────────────────────────────────────────────────────
    model.train()
    torch.cuda.empty_cache()
    main_logger_info("학습 시작!")

    while state.step < args.max_steps:
        state.start_step()
        is_last_step = state.step == args.max_steps

        loss = torch.tensor([0.0], device="cuda")
        vap_loss_val = torch.tensor([0.0], device="cuda")
        face_loss_val = torch.tensor([0.0], device="cuda")
        bc_stats_accum: dict | None = None
        n_batch_tokens: int = 0
        n_real_tokens: int = 0

        for i in range(args.num_microbatches):
            batch = next(data_loader)
            codes = batch.codes  # [B, K, T]

            # 마지막 microbatch 전까지는 FSDP all-reduce 억제 (단일 GPU는 no_sync 없으므로 스킵)
            sync_ctx = (
                contextlib.nullcontext()
                if (i == args.num_microbatches - 1 or not hasattr(model, "no_sync"))
                else model.no_sync()
            )
            with sync_ctx:
                # FSDP의 forward pre-hook이 정상 발동하도록 __call__ -> forward 호출
                voice_prompt_embs = batch.voice_prompt_embs
                if voice_prompt_embs is not None:
                    voice_prompt_embs = voice_prompt_embs.to(codes.device, non_blocking=True)  # int64 mimi codes

                # ── Face generation inputs ────────────────────────────────
                # Decode agent audio codes to Mimi latent features.
                # Uses frozen mimi (no grad) — safe to call outside the model.
                audio_feat = None
                gt_face_motion = None
                if args.face_gen.enable:
                    with torch.no_grad():
                        # codes[:, 1:9] = first 8 agent audio codebooks (Mimi's n_q).
                        # Clamp to min=0 so zero_token_id (-1) padding positions don't
                        # produce out-of-bounds indices inside F.embedding in the VQ decoder.
                        audio_codes = codes[:, 1:9].clamp(min=0)
                        audio_feat = mimi.decode_latent(audio_codes).transpose(1, 2)  # [B, T, 512]
                        audio_feat = audio_feat.to(dtype=param_dtype)
                    if batch.face_motion_gt is not None:
                        gt_face_motion = batch.face_motion_gt.to(codes.device, dtype=param_dtype)

                output = model(codes, step=state.step, voice_prompt_embs=voice_prompt_embs,
                               audio_feat=audio_feat, gt_face_motion=gt_face_motion)

                # Silence-padded frames (mimi tokens from zero-padded waveform) are kept
                # in audio/text loss intentionally: they provide backbone regularization,
                # teaching the model to suppress output after real content ends.
                # Text at padded positions is already zero_token_id (-1) → text_mask=False
                # regardless; the audio silence tokens are the meaningful signal here.
                text_mask = output.text_mask
                audio_mask = output.mask

                text_loss = compute_loss_with_mask(
                    output.text_logits,
                    codes[:, : model.audio_offset],
                    text_mask,
                    mode="text",
                    text_padding_weight=args.text_padding_weight,
                    text_padding_ids={
                        model.text_padding_token_id,
                        model.end_of_text_padding_id,
                    },
                )
                audio_loss = compute_loss_with_mask(
                    output.logits,
                    codes[:, model.audio_offset : model.audio_offset + model.dep_q],
                    audio_mask,
                    mode="audio",
                    first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
                )

                mb_loss = text_loss + audio_loss

                # ── VAP 보조 손실 ─────────────────────────────────────────
                vap_loss = None
                if args.backchannel.enable:
                    vap_logits = output.vap_logits  # [B, T, 256] from BackchannelModule inside LM

                    if vap_logits is not None:
                        vap_targets_tensor = batch.vap_targets  # [B, T]
                        # Mask silence-padded frames: set to ignore_index so cross_entropy skips them
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
                                    vap_logits.view(-1, num_vap_classes),
                                    flat_targets,
                                    ignore_index=-100,
                                )
                                if not torch.isnan(vap_loss):
                                    mb_loss = mb_loss + args.backchannel.vap_loss_weight * vap_loss
                                    vap_loss_val += vap_loss.detach()
                            # else: no valid VAP targets in this batch — skip
                        else:
                            # Fallback: entropy regularization when no manifest targets
                            vap_probs = F.softmax(vap_logits, dim=-1)
                            vap_entropy = -(vap_probs * (vap_probs + 1e-8).log()).sum(dim=-1).mean()
                            vap_loss = -vap_entropy
                            mb_loss = mb_loss + args.backchannel.vap_loss_weight * vap_loss
                            vap_loss_val += vap_loss.detach()

                    # Alt 3: Commitment loss — bc_mlp + silence_gate alignment with depformer
                    if output.commitment_loss is not None and not torch.isnan(output.commitment_loss):
                        mb_loss = mb_loss + args.backchannel.commitment_loss_weight * output.commitment_loss

                # ── Face motion reconstruction loss (full reference loss) ──
                face_loss = None
                if (args.face_gen.enable and output.face_outputs is not None
                        and gt_face_motion is not None and face_codec is not None):
                    # Build [B, T_face] bool mask from per-sample valid frame counts
                    valid_face_mask = None
                    if batch.valid_face_frames is not None:
                        T_face = gt_face_motion.shape[1]
                        t_idx = torch.arange(T_face, device=codes.device).unsqueeze(0)  # [1, T_face]
                        valid_face_mask = t_idx < batch.valid_face_frames.to(codes.device).unsqueeze(1)  # [B, T_face]
                    face_loss = compute_face_loss(
                        output.face_outputs, gt_face_motion, face_codec, args.face_gen,
                        valid_face_mask=valid_face_mask,
                    )
                    if torch.isfinite(face_loss):
                        mb_loss = mb_loss + args.face_gen.face_loss_weight * face_loss
                        face_loss_val += face_loss.detach()
                    else:
                        logger.warning(f"[step {state.step}] Non-finite face_loss={face_loss.item():.4f}, skipping.")

                n_batch_tokens += text_mask.numel() + audio_mask.numel()
                n_real_tokens += (
                    torch.sum(text_mask).item() + torch.sum(audio_mask).item()
                )

                mb_loss.backward()

            loss += mb_loss.detach()
            if args.backchannel.enable and output.bc_stats is not None:
                if bc_stats_accum is None:
                    bc_stats_accum = {k: v.clone() for k, v in output.bc_stats.items()}
                else:
                    for k in bc_stats_accum:
                        bc_stats_accum[k] = bc_stats_accum[k] + output.bc_stats[k]
            del output, mb_loss, text_loss, audio_loss
            if args.backchannel.enable and vap_loss is not None:
                del vap_loss
            if args.face_gen.enable:
                del audio_feat, gt_face_motion, face_loss

        if args.num_microbatches > 1:
            loss /= args.num_microbatches
            if args.backchannel.enable:
                vap_loss_val /= args.num_microbatches
                if bc_stats_accum is not None:
                    for k in bc_stats_accum:
                        bc_stats_accum[k] = bc_stats_accum[k] / args.num_microbatches
            if args.face_gen.enable:
                face_loss_val /= args.num_microbatches
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.div_(args.num_microbatches)

        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(list(model.parameters()), args.max_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # 메모리 즉시 해제

        last_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        loss_item = loss.item()
        avg_loss = avg_aggregate(loss_item)
        if args.backchannel.enable:
            state.this_vap_loss = avg_aggregate(vap_loss_val.item())
        if args.face_gen.enable:
            state.this_face_loss = avg_aggregate(face_loss_val.item())

        # 평가
        if args.do_eval and (
            (args.eval_freq > 0 and state.step % args.eval_freq == 0) or is_last_step
        ):
            eval_data_loader = build_data_loader(
                instruct_tokenizer=interleaved_tokenizer,
                args=args.data,
                batch_size=args.batch_size,
                seed=None,
                rank=get_rank(),
                world_size=get_world_size(),
                is_eval=True,
            )
            evaluate(model, eval_data_loader, state, args)
            del eval_data_loader

            eval_logs = get_eval_logs(
                state.step,
                avg_loss,
                state.this_eval_perplexity,
                state.this_eval_loss,
            )
            main_logger_info(eval_log_msg(eval_logs))
            eval_logger.log(eval_logs, step=state.step)

        # 타이밍
        state.end_step(n_batch_tokens)

        if args.backchannel.enable and bc_stats_accum is not None and state.step % args.log_freq == 0:
            state.this_bc_stats = {k: avg_aggregate(v.item()) for k, v in bc_stats_accum.items()}

        if state.step % args.log_freq == 0:
            train_logs = get_train_logs(
                state,
                avg_loss,
                n_real_tokens,
                last_lr,
                torch.cuda.max_memory_allocated(),
                torch.cuda.memory_allocated(),
                args,
            )
            main_logger_info(train_log_msg(state, logs=train_logs, loss=avg_loss))
            metrics_logger.log(train_logs, step=state.step)

        # 주기적 CUDA 캐시 비우기 (메모리 단편화 방지)
        if state.step % 200 == 0:
            torch.cuda.empty_cache()

        # 체크포인트 저장
        if args.do_ckpt and (
            (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
        ):
            torch.cuda.empty_cache()
            checkpointer.save_checkpoint(
                save_only_lora=args.lora.enable,
                dtype=param_dtype,
            )

    main_logger_info("학습 완료!")


if __name__ == "__main__":
    """사용법: torchrun --nproc_per_node=<N_GPUS> train.py config/example.yaml"""
    fire.Fire(train)
