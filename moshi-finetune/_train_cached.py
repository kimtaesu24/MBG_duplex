import dataclasses
import logging
import os
import sys
import shutil
import contextlib
from contextlib import ExitStack
from pathlib import Path
import pprint
from tqdm import tqdm
import sentencepiece as spm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../moshi")))

# rope.py의 apply_rope가 @torch.compile로 감싸져 있는데,
# gradient checkpointing + FSDP와 함께 쓰면 inductor가 FakeTensor 충돌을 일으킴.
# 학습에서는 torch.compile을 비활성화하고 eager 모드로 실행합니다.
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # compile 실패 시 eager로 fallback
os.environ["TORCHDYNAMO_DISABLE"] = "1"      # torch.compile 전역 비활성화

import ast
import fire
import torch
import torch.cuda
import torch.distributed as dist
from torch.optim import AdamW, lr_scheduler
from torch.nn import functional as F

from finetune.args import TrainArgs
from finetune.checkpointing import Checkpointer
from finetune.data.cached_dataset import build_cached_loader
from finetune.distributed import (
    BACKEND,
    avg_aggregate,
    get_rank,
    get_world_size,
    is_torchrun,
    set_device,
)
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
from finetune.eval import evaluate

from moshi.models.loaders import (
    get_mimi,
    _lm_kwargs,
    DEFAULT_REPO,
)

os.environ["WANDB_DIR"] = os.path.abspath("./wandb_logs")
os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

logger = logging.getLogger("train")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def is_main(rank: int) -> bool:
    return rank == 0


def train(config: str, cached_train: str = None, cached_valid: str = None):
    """Entry point for training with cached dataset shards.

    사용법:
        torchrun --nproc_per_node=<N_GPUS> train_merged.py config/example.yaml \\
            [--cached_train /path/to/train] [--cached_valid /path/to/valid]
    """
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    set_logger(logging.INFO)

    # 명령행 인자가 없으면 설정 파일의 값을 기본값으로 사용
    train_path = cached_train or args.data.train_data
    valid_path = cached_valid or args.data.eval_data

    with ExitStack() as exit_stack:
        _train(args, exit_stack, cached_train=train_path, cached_valid=valid_path)
    logger.info("학습 종료!")


def _train(args: TrainArgs, exit_stack: ExitStack, cached_train: str, cached_valid: str):
    # ── 1. 초기 설정 ──────────────────────────────────────────────────────
    set_random_seed(args.seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if "LOCAL_RANK" in os.environ:
        set_device()
        rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend=BACKEND)
    else:
        rank = 0
        logger.error("torchrun으로 실행하세요.")

    main_logger_info(f"Run dir: {args.run_dir}")
    run_dir = Path(args.run_dir)
    if is_main(rank):
        if run_dir.exists() and args.overwrite_run_dir:
            shutil.rmtree(run_dir)
        run_dir.mkdir(exist_ok=True, parents=True)

    dist.barrier()
    args.save(run_dir / "args.yaml")
    main_logger_info(f"TrainArgs:\n{pprint.pformat(dataclasses.asdict(args))}")

    # ── 2. 로거 & 체크포인터 ────────────────────────────────────────────────
    metrics_logger = MetricsLogger(
        run_dir, tag="train", is_master=is_main(rank),
        wandb_args=args.wandb, config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(metrics_logger, "metrics_logger"))

    eval_logger = MetricsLogger(
        run_dir, tag="eval", is_master=is_main(rank),
        wandb_args=args.wandb, config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(eval_logger, "eval_logger"))

    # ── 3. 텍스트 토크나이저 & 다중 프롬프트 사전 토크나이즈 ──────────────
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(args.moshi_paths.tokenizer_path)

    # args.text_prompt 가 단일 문자열이거나 리스트일 수 있음.
    # cached loader 배치의 dataset_idx를 통해 배치 내 샘플별로 프롬프트를 선택한다.
    prompts: list[str] = []
    if isinstance(args.text_prompt, str):
        if args.text_prompt.strip().startswith("[") and args.text_prompt.strip().endswith("]"):
            try:
                parsed = ast.literal_eval(args.text_prompt)
                prompts = parsed if isinstance(parsed, list) else [args.text_prompt]
            except Exception:
                prompts = [args.text_prompt]
        else:
            prompts = [args.text_prompt]
    elif isinstance(args.text_prompt, list):
        prompts = args.text_prompt

    prompt_ids_list: list[list[int]] = [tokenizer.encode(p) for p in prompts]
    T_p_list: list[int] = [len(ids) for ids in prompt_ids_list]
    T_p_max: int = max(T_p_list) if T_p_list else 0

    if T_p_max:
        main_logger_info(
            f"텍스트 프롬프트 활성화: {len(prompts)}개 프롬프트 (최대 {T_p_max}토큰 prefix)"
        )
        for i, (p, length) in enumerate(zip(prompts, T_p_list)):
            main_logger_info(f"  [{i}] ({length} tokens): {p!r}")
    else:
        main_logger_info("텍스트 프롬프트 없음 — prefix 삽입 생략")

    # 편의 alias: 단일 프롬프트 경로와의 호환
    # (T_p_max 를 T_p 로 쓰는 부분은 T_p_max 로 통일)
    T_p = T_p_max

    # ── 4. Mimi(오디오 토크나이저) 로드 ──────────────────────────────────
    main_logger_info("Mimi 로드 중...")
    mimi = get_mimi(args.moshi_paths.mimi_path, device="cuda")
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    # ── 5. LM 설정 결정 ───────────────────────────────────────────────────
    lm_config = dict(_lm_kwargs)
    lm_config["dep_q"] = 16

    # ── 5-1. Backchannel / VAP 설정 ───────────────────────────────────────
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
            main_logger_info("Backchannel MLP VAP 모듈 활성화")
            lm_config["backchannel_module_type"] = "mlp"
        elif args.backchannel.module_type == "vap_gpt":
            main_logger_info("VapGPT Backchannel 모듈 활성화 (z_s 입력)")
            lm_config["backchannel_module_type"] = "vap_gpt"
            lm_config["backchannel_vap_repo_path"] = args.backchannel.vap_gpt_repo_path
            lm_config["backchannel_vap_checkpoint"] = args.backchannel.vap_gpt_checkpoint
            lm_config["backchannel_vap_channel_layers"] = args.backchannel.vap_gpt_channel_layers
            lm_config["backchannel_vap_cross_layers"] = args.backchannel.vap_gpt_cross_layers
            lm_config["backchannel_vap_num_heads"] = args.backchannel.vap_gpt_num_heads
            lm_config["backchannel_vap_dropout"] = args.backchannel.vap_gpt_dropout
            lm_config["backchannel_vap_use_silence_ctx_proj"] = getattr(args.backchannel, "use_silence_ctx_proj", True)
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
        lm_config["mimi_enabled"] = True
        lm_config["mimi_checkpoint"] = args.moshi_paths.mimi_path
        main_logger_info(
            f"Face generation module 활성화 (checkpoint={args.face_gen.ckpt_path})"
        )

    # ── 6. LM 모델 로드 및 FSDP 샤딩 ─────────────────────────────────────
    main_logger_info("LM 모델 로드 중...")
    model = get_fsdp_model(args, args.moshi_paths.moshi_path, lm_config)

    # ── 6-1. ARTalkCodec (face VAE) 로드 ──────────────────────────────────
    # 동결 상태로 유지하며 face loss의 z_target 계산에만 사용.
    # 작은 모델이므로 FSDP 없이 각 rank가 full copy 유지.
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
        _codec_ckpt = torch.load(
            args.face_gen.codec_ckpt_path, map_location="cpu", weights_only=True
        )
        face_codec.load_state_dict(_codec_ckpt["model"], strict=True)
        face_codec.eval().cuda()
        for p in face_codec.parameters():
            p.requires_grad = False
        main_logger_info(f"ARTalkCodec loaded from: {args.face_gen.codec_ckpt_path}")

    # ── 7. 캐시 기반 데이터 로더 구성 ────────────────────────────────────
    main_logger_info(f"Cached train loader: {cached_train}")
    main_logger_info(f"Cached valid loader: {cached_valid}")
    data_loader = build_cached_loader(cached_train, batch_size=args.batch_size, shuffle=True)
    valid_loader = build_cached_loader(cached_valid, batch_size=args.batch_size, shuffle=False)

    steps_per_epoch = max(1, len(data_loader) // args.num_microbatches)

    # ── 8. dtype / 옵티마이저 / 스케줄러 ─────────────────────────────────
    param_dtype = getattr(torch, args.param_dtype)

    optimizer = AdamW(
        model.parameters(),
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

    # ── 9. 체크포인터 ─────────────────────────────────────────────────────
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

    # ── 10. 학습 루프 ─────────────────────────────────────────────────────
    model.train()
    torch.cuda.empty_cache()
    main_logger_info("Cached 학습 시작!")

    device = torch.device("cuda")
    loader_iter = iter(data_loader)

    pbar = tqdm(
        range(state.step, args.max_steps),
        desc="Training",
        disable=not is_main(rank),
    )

    while state.step < args.max_steps:
        state.start_step()
        is_last_step = state.step == args.max_steps
        curr_epoch = state.step / steps_per_epoch

        # ── 손실 누적 변수 초기화 ──────────────────────────────────────────
        loss_val = torch.tensor(0.0, device=device)
        vap_loss_val = torch.tensor(0.0, device=device)
        commitment_loss_val = torch.tensor(0.0, device=device)
        face_loss_val = torch.tensor(0.0, device=device)
        bc_event_loss_val = torch.tensor(0.0, device=device)
        bc_stats_accum: dict | None = None
        face_loss_details: dict = {}
        n_real_tokens: int = 0
        n_batch_tokens: int = 0
        face_loss_skipped_no_data: int = 0
        face_loss_skipped_nonfinite: int = 0

        optimizer.zero_grad(set_to_none=True)

        for i in range(args.num_microbatches):
            if is_main(rank):
                pbar.set_description(
                    f"Step {state.step} [Epoch {curr_epoch:.2f}] [Data]"
                )

            # ── 캐시 배치 fetch (cyclic) ───────────────────────────────────
            try:
                batch = next(loader_iter)
            except StopIteration:
                if hasattr(data_loader.sampler, "set_epoch"):
                    data_loader.sampler.set_epoch(state.step // len(data_loader))
                loader_iter = iter(data_loader)
                batch = next(loader_iter)

            codes = batch["audio_codes"].to(device)  # [B, K, T]

            # ── 다중 텍스트 프롬프트 prefix 삽입 ──────────────────────────
            # dataset_idx 를 이용해 배치 내 샘플마다 다른 프롬프트를 선택한다.
            if T_p_max:
                prefix = torch.full(
                    [codes.shape[0], codes.shape[1], T_p_max],
                    model.zero_token_id,
                    device=device, dtype=codes.dtype,
                )
                ds_indices = batch["dataset_idx"].tolist()
                for b_idx, ds_idx in enumerate(ds_indices):
                    p_idx = ds_idx % len(prompt_ids_list)
                    p_tokens = prompt_ids_list[p_idx]
                    curr_len = len(p_tokens)
                    # 가장 긴 프롬프트 기준으로 우측 정렬
                    prefix[b_idx, 0, T_p_max - curr_len:] = torch.tensor(
                        p_tokens, device=device
                    )
                codes_in = torch.cat([prefix, codes], dim=2)  # [B, K, T_p_max+T]
            else:
                codes_in = codes

            # ── voice_prompt_codes (optional) ─────────────────────────────
            voice_prompt_codes = batch.get("voice_prompt_codes")
            if voice_prompt_codes is not None:
                voice_prompt_codes = voice_prompt_codes.to(device)

            sync_ctx = (
                contextlib.nullcontext()
                if i == args.num_microbatches - 1
                else model.no_sync()
            )

            with sync_ctx:
                if is_main(rank):
                    pbar.set_description(f"Step {state.step} [Forward]")

                # ── Face generation inputs ────────────────────────────────
                audio_feat = None
                gt_face_motion = None

                if args.face_gen.enable:
                    # 캐시에 저장된 user 채널 Mimi latent 피처를 teacher-forcing 에 사용.
                    # cached_dataset 에서 audio_latents[:, 0] = user 채널.
                    raw_audio_latent = batch.get("audio_latents")
                    if raw_audio_latent is not None:
                        audio_feat = raw_audio_latent[:, 0].to(device, dtype=param_dtype)
                        if T_p_max:
                            zero_feat = torch.zeros(
                                audio_feat.shape[0], T_p_max, audio_feat.shape[2],
                                device=device, dtype=audio_feat.dtype,
                            )
                            audio_feat = torch.cat([zero_feat, audio_feat], dim=1)

                    raw_face = batch.get("face_motion")
                    if raw_face is not None:
                        gt_face_motion = raw_face.to(device, dtype=param_dtype)
                        if T_p_max:
                            # face 25fps = 2× mimi 12.5fps → T_p_max mimi frames = 2*T_p_max face frames
                            T_face_p = T_p_max * 2
                            zero_motion = torch.zeros(
                                gt_face_motion.shape[0], T_face_p, gt_face_motion.shape[2],
                                device=device, dtype=gt_face_motion.dtype,
                            )
                            gt_face_motion = torch.cat([zero_motion, gt_face_motion], dim=1)
                    else:
                        face_loss_skipped_no_data += 1

                # ── VapGPT per-speaker Mimi features ─────────────────────
                bc_audio_feats = None
                if (
                    args.backchannel.enable
                    and args.backchannel.module_type == "vap_gpt"
                ):
                    with torch.no_grad():
                        _agent = mimi.decode_latent(
                            codes[:, 1:9].clamp(min=0)
                        ).transpose(1, 2)
                        _user = mimi.decode_latent(
                            codes[:, 9:17].clamp(min=0)
                        ).transpose(1, 2)
                        _agent = _agent.to(dtype=param_dtype)
                        _user = _user.to(dtype=param_dtype)
                        if T_p_max:
                            _zero = torch.zeros(
                                codes.shape[0], T_p_max, _agent.shape[2],
                                device=device, dtype=_agent.dtype,
                            )
                            _agent = torch.cat([_zero, _agent], dim=1)
                            _user = torch.cat([_zero, _user], dim=1)
                    bc_audio_feats = (_agent, _user)

                # ── Forward ───────────────────────────────────────────────
                output = model(
                    codes_in,
                    step=state.step,
                    voice_prompt_codes=voice_prompt_codes,
                    audio_feat=audio_feat,
                    gt_face_motion=gt_face_motion,
                    bc_audio_feats=bc_audio_feats,
                )

                if is_main(rank):
                    pbar.set_description(
                        f"Step {state.step} [Epoch {curr_epoch:.2f}] [Loss]"
                    )

                # ── prefix 구간 제외 마스크 ────────────────────────────────
                text_mask = output.text_mask[:, :, T_p_max:]
                audio_mask = output.mask[:, :, T_p_max:]
                if audio_mask.shape[1] == 1:
                    audio_mask = audio_mask.expand(-1, model.dep_q, -1)

                # ── 기본 Text / Audio Loss ────────────────────────────────
                text_loss = compute_loss_with_mask(
                    output.text_logits[:, :, T_p_max:],
                    codes[:, : model.audio_offset],
                    text_mask,
                    mode="text",
                    text_padding_weight=args.text_padding_weight,
                    text_padding_ids={model.text_padding_token_id},
                    epad_ids={model.end_of_text_padding_id},
                    epad_weight=args.epad_weight,
                )
                audio_loss = compute_loss_with_mask(
                    output.logits[:, :, T_p_max:],
                    codes[:, model.audio_offset: model.audio_offset + model.dep_q],
                    audio_mask,
                    mode="audio",
                    first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
                )
                mb_loss = text_loss + audio_loss

                # ── VAP 보조 손실 ─────────────────────────────────────────
                vap_loss = None
                if args.backchannel.enable:
                    vap_logits = (
                        output.vap_logits[:, T_p_max:]
                        if output.vap_logits is not None
                        else None
                    )
                    if vap_logits is not None:
                        vap_targets_tensor = batch.get("vap_targets")
                        valid_mask_tensor = batch.get("valid_mask")
                        if vap_targets_tensor is not None:
                            vap_targets_tensor = vap_targets_tensor.to(device)
                            if valid_mask_tensor is not None:
                                vap_targets_tensor = vap_targets_tensor.masked_fill(
                                    ~valid_mask_tensor.to(device), -100
                                )
                            min_t = min(
                                vap_logits.shape[1],
                                vap_targets_tensor.shape[1],
                            )
                            flat_targets = vap_targets_tensor[:, :min_t].reshape(-1).long()
                            n_valid = (flat_targets != -100).sum().item()
                            if n_valid > 0:
                                num_vap_classes = vap_logits.shape[-1]
                                flat_targets = flat_targets.clamp(-100, num_vap_classes - 1)
                                vap_loss = F.cross_entropy(
                                    vap_logits[:, :min_t].reshape(-1, num_vap_classes),
                                    flat_targets,
                                    ignore_index=-100,
                                )
                                if not torch.isnan(vap_loss):
                                    mb_loss = mb_loss + args.backchannel.vap_loss_weight * vap_loss
                                    vap_loss_val += vap_loss.detach()
                        else:
                            # Fallback: entropy 정규화
                            vap_probs = F.softmax(vap_logits, dim=-1)
                            vap_entropy = -(vap_probs * (vap_probs + 1e-8).log()).sum(dim=-1).mean()
                            vap_loss = -vap_entropy
                            mb_loss = mb_loss + args.backchannel.vap_loss_weight * vap_loss
                            vap_loss_val += vap_loss.detach()

                    # Commitment loss
                    if (
                        output.commitment_loss is not None
                        and not torch.isnan(output.commitment_loss)
                    ):
                        mb_loss = (
                            mb_loss
                            + args.backchannel.commitment_loss_weight
                            * output.commitment_loss
                        )
                        commitment_loss_val += output.commitment_loss.detach()

                # ── Direct BC event supervision (focal BCE) ───────────────
                if (
                    args.backchannel.enable
                    and args.backchannel.bc_event_loss_weight > 0
                    and output.bc_logits is not None
                ):
                    bc_timing_targets = batch.get("bc_timing_targets")
                    if bc_timing_targets is not None:
                        bc_targets = bc_timing_targets.to(device)  # [B, T]
                        bc_logit_pos = output.bc_logits[:, T_p_max:, 1]  # [B, T]
                        valid_bc = bc_targets != -100
                        if valid_bc.any():
                            bce = F.binary_cross_entropy_with_logits(
                                bc_logit_pos[valid_bc],
                                bc_targets[valid_bc],
                                pos_weight=torch.tensor(
                                    args.backchannel.bc_focal_pos_weight,
                                    device=device, dtype=bc_logit_pos.dtype,
                                ),
                                reduction="none",
                            )
                            p_t = torch.exp(-bce.detach())
                            bc_event_loss = (
                                (1.0 - p_t) ** args.backchannel.bc_focal_gamma * bce
                            ).mean()
                            mb_loss = (
                                mb_loss
                                + args.backchannel.bc_event_loss_weight * bc_event_loss
                            )
                            bc_event_loss_val += bc_event_loss.detach()

                # ── Face motion reconstruction loss ───────────────────────
                face_loss = None
                if args.face_gen.enable:
                    if output.face_outputs is None:
                        if gt_face_motion is not None and is_main(rank):
                            logger.warning(
                                f"[step {state.step}] face_outputs is None but "
                                f"gt_face_motion is not None — "
                                f"face_codec={'ok' if face_codec else 'MISSING'}."
                            )
                    elif gt_face_motion is not None and face_codec is not None:
                        T_face_p = T_p_max * 2
                        valid_face_frames = batch.get("valid_face_frames")
                        valid_face_mask = None
                        if T_p_max or valid_face_frames is not None:
                            T_face = gt_face_motion.shape[1]
                            t_idx = torch.arange(T_face, device=device).unsqueeze(0)
                            if valid_face_frames is not None:
                                valid_face_mask = (t_idx >= T_face_p) & (
                                    t_idx
                                    < T_face_p
                                    + valid_face_frames.to(device).unsqueeze(1)
                                )
                            else:
                                valid_face_mask = t_idx >= T_face_p
                        if (
                            valid_face_mask is not None
                            and not valid_face_mask.any()
                            and is_main(rank)
                        ):
                            logger.warning(
                                f"[step {state.step}] valid_face_mask is all-False — "
                                f"T_face_p={T_face_p}, T_face={gt_face_motion.shape[1]}. "
                                f"Face loss will be 0."
                            )

                        face_result = compute_face_loss(
                            output.face_outputs,
                            gt_face_motion,
                            face_codec,
                            args.face_gen,
                            valid_face_mask=valid_face_mask,
                        )
                        # compute_face_loss 가 (loss, details) 튜플을 반환하는 경우 대응
                        if isinstance(face_result, tuple):
                            face_loss, f_details = face_result
                            for k, v in f_details.items():
                                face_loss_details[k] = (
                                    face_loss_details.get(k, 0.0)
                                    + v / args.num_microbatches
                                )
                        else:
                            face_loss = face_result

                        if torch.isfinite(face_loss):
                            if face_loss.item() == 0.0 and is_main(rank):
                                logger.warning(
                                    f"[step {state.step}] face_loss == 0.0 "
                                    f"(valid frames: "
                                    f"{valid_face_mask.sum().item() if valid_face_mask is not None else 'N/A'}). "
                                    f"Check FLAME data quality."
                                )
                            mb_loss = mb_loss + args.face_gen.face_loss_weight * face_loss
                            face_loss_val += face_loss.detach()
                        else:
                            face_loss_skipped_nonfinite += 1
                            logger.warning(
                                f"[step {state.step}] Non-finite face_loss="
                                f"{face_loss.item():.4f}, skipping."
                            )

                if torch.isnan(mb_loss):
                    raise RuntimeError(f"[RANK {rank}] NaN detected at step {state.step}!")

                n_real_tokens += text_mask.sum().item() + audio_mask.sum().item()
                n_batch_tokens += text_mask.numel() + audio_mask.numel()

                if is_main(rank):
                    pbar.set_description(
                        f"Step {state.step} [Epoch {curr_epoch:.2f}] [Backward]"
                    )
                (mb_loss / args.num_microbatches).backward()
                loss_val += mb_loss.detach() / args.num_microbatches

            # BC stats 누적
            if args.backchannel.enable and output.bc_stats is not None:
                if bc_stats_accum is None:
                    bc_stats_accum = {k: v.clone() for k, v in output.bc_stats.items()}
                else:
                    for k in bc_stats_accum:
                        bc_stats_accum[k] = bc_stats_accum[k] + output.bc_stats[k]

            # 명시적 메모리 해제
            del output, mb_loss, text_loss, audio_loss
            if vap_loss is not None:
                del vap_loss
            if bc_audio_feats is not None:
                del bc_audio_feats
            if args.face_gen.enable:
                del audio_feat, gt_face_motion, face_loss

        # ── 그래디언트 클리핑 & 옵티마이저 스텝 ─────────────────────────
        if is_main(rank):
            pbar.set_description(f"Step {state.step} [Optimizer]")

        # model.clip_grad_norm_(args.max_norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        # VapGPT warm-up: 초반 스텝에서 사전학습된 GPT/vap_head 가중치 보호
        if (
            args.backchannel.enable
            and args.backchannel.module_type == "vap_gpt"
            and args.backchannel.bc_warmup_steps > 0
            and state.step <= args.backchannel.bc_warmup_steps
        ):
            for name, p in model.named_parameters():
                if p.grad is not None and (
                    "backchannel.ar_channel" in name
                    or "backchannel.ar." in name
                    or "backchannel.vap_head" in name
                ):
                    p.grad.zero_()
            if state.step == args.backchannel.bc_warmup_steps and is_main(rank):
                logger.info(
                    f"[step {state.step}] VapGPT warm-up 완료 — "
                    "GPT/vap_head 그래디언트 재활성화."
                )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # 메모리 즉시 해제

        last_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        state.step += 1

        # ── 분산 평균 집계 ────────────────────────────────────────────────
        avg_loss = avg_aggregate(loss_val.item())
        if args.backchannel.enable:
            state.this_vap_loss = avg_aggregate(
                vap_loss_val.item() / args.num_microbatches
            )
            state.this_commitment_loss = avg_aggregate(
                commitment_loss_val.item() / args.num_microbatches
            )
        if args.face_gen.enable:
            state.this_face_loss = avg_aggregate(
                face_loss_val.item() / args.num_microbatches
            )

        # ── BC stats 평균 계산 ────────────────────────────────────────────
        if (
            args.backchannel.enable
            and bc_stats_accum is not None
            and state.step % args.log_freq == 0
        ):
            state.this_bc_stats = {
                k: avg_aggregate(v.item() / args.num_microbatches)
                for k, v in bc_stats_accum.items()
            }

        # ── 주기적 CUDA 캐시 비우기 ───────────────────────────────────────
        if state.step % 200 == 0:
            torch.cuda.empty_cache()

        state.end_step(n_batch_tokens)

        # ── 로깅 ──────────────────────────────────────────────────────────
        if state.step % args.log_freq == 0:
            if args.face_gen.enable and is_main(rank):
                total_mb = args.num_microbatches
                if face_loss_skipped_no_data > 0:
                    logger.warning(
                        f"[step {state.step}] face_loss skipped in "
                        f"{face_loss_skipped_no_data}/{total_mb} microbatch(es): "
                        f"face_motion 키 없음 (cached batch)."
                    )
                if face_loss_skipped_nonfinite > 0:
                    logger.warning(
                        f"[step {state.step}] face_loss skipped in "
                        f"{face_loss_skipped_nonfinite}/{total_mb} microbatch(es): "
                        f"NaN 또는 Inf."
                    )

            train_logs = get_train_logs(
                state,
                avg_loss,
                n_real_tokens,
                last_lr,
                torch.cuda.max_memory_allocated(),
                torch.cuda.memory_allocated(),
                args,
                commitment_loss=state.this_commitment_loss if args.backchannel.enable else None,
                vap_loss=state.this_vap_loss if args.backchannel.enable else None,
                face_loss=state.this_face_loss if args.face_gen.enable else None,
            )
            train_logs["epoch"] = curr_epoch

            if is_main(rank):
                logger.info(
                    f"Step {state.step} | {train_log_msg(state, train_logs, avg_loss)}"
                )
                metrics_logger.log(train_logs, step=state.step)
                if face_loss_details:
                    metrics_logger.log(face_loss_details, step=state.step)
                if args.backchannel.enable and args.backchannel.bc_event_loss_weight > 0:
                    bc_evt = avg_aggregate(bc_event_loss_val.item())
                    logger.info(
                        f"[step {state.step}] bc_event_loss={bc_evt:.4f}"
                    )

        # ── 평가 ──────────────────────────────────────────────────────────
        if args.do_eval and (
            (args.eval_freq > 0 and state.step % args.eval_freq == 0) or is_last_step
        ):
            if is_main(rank):
                pbar.set_description(
                    f"Step {state.step} [Epoch {curr_epoch:.2f}] [Eval]"
                )
            evaluate(
                model, valid_loader, state, args,
                mimi=mimi, face_codec=face_codec,
                T_p=T_p, prompt_ids=prompt_ids_list if T_p else None,
                param_dtype=param_dtype,
            )
            if is_main(rank):
                eval_logs = get_eval_logs(
                    state.step,
                    avg_loss,
                    perplexity=state.this_eval_perplexity,
                    eval_loss=state.this_eval_loss,
                    text_eval_loss=getattr(state, "this_text_loss", None),
                    audio_eval_loss=getattr(state, "this_audio_loss", None),
                )
                main_logger_info(eval_log_msg(eval_logs))
                eval_logger.log(eval_logs, step=state.step)

            # valid_loader 는 재사용하므로 이터레이터를 리셋할 필요 없음.
            # (build_cached_loader 는 매 iteration 마다 새로 iterate 가능)
            loader_iter = iter(data_loader)  # train loader만 안전하게 재설정

        # ── 체크포인트 ────────────────────────────────────────────────────
        if args.do_ckpt and (
            (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
        ):
            torch.cuda.empty_cache()
            checkpointer.save_checkpoint(
                save_only_lora=args.lora.enable, dtype=param_dtype
            )

        # ── tqdm 업데이트 ─────────────────────────────────────────────────
        if is_main(rank):
            pbar.set_postfix({
                "loss": f"{loss_val.item():.4f}",
                "commit": f"{commitment_loss_val.item() / args.num_microbatches:.4f}",
                "vap": f"{vap_loss_val.item() / args.num_microbatches:.4f}",
                "face": f"{face_loss_val.item() / args.num_microbatches:.4f}",
            })
            pbar.update(1)

    main_logger_info("학습 완료!")


if __name__ == "__main__":
    fire.Fire(train)