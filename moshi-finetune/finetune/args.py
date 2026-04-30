import logging
import os
from dataclasses import dataclass, field

from simple_parsing.helpers import Serializable

from .data.args import DataArgs


@dataclass
class BackchannelArgs(Serializable):
    """Backchannel / VAP 모듈 설정.

    module_type 선택:
      "mlp"     – 기존 BackchannelModule (LM 내부 MLP 기반 VAP + 토큰 게이팅).
      "vap_gpt" – 외부 VapGPT 모델을 보조 VAP 손실로 사용 (LM 내 BackchannelModule 비활성).
    """
    enable: bool = False
    module_type: str = "mlp"  # "mlp" | "vap_gpt"

    # ── MLP 모드 전용 ─────────────────────────────────────────────────────
    vap_dim: int = 256
    bc_hidden: int = 512
    gumbel_temp_init: float = 1.0
    gumbel_temp_min: float = 0.5
    gumbel_anneal_rate: float = 0.0001
    pad_token_id: int | None = None   # None이면 card (= initial_token_id) 사용
    epad_token_id: int | None = None  # None이면 card - 1 사용

    # ── 공통 ──────────────────────────────────────────────────────────────
    vap_loss_weight: float = 0.1
    commitment_loss_weight: float = 0.05  # Alt 3: bc_mlp + silence_gate alignment with depformer

    # ── VapGPT 모드 전용 ──────────────────────────────────────────────────
    # VoiceActivityProjection 리포지토리 루트 경로
    vap_gpt_repo_path: str = "/home2/s20235100/Conversational-AI/VoiceActivityProjection"
    # 사전학습 VapGPT 체크포인트 (.pt 또는 .ckpt).
    # 기본값: VAP 리포에 포함된 공개 사전학습 가중치 (.pt = clean state dict).
    # None으로 설정하면 랜덤 초기화 + CPC 단독 사전학습 가중치만 사용.
    vap_gpt_checkpoint: str | None = (
        "/home2/s20235100/Conversational-AI/VoiceActivityProjection"
        "/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt"
    )
    # True면 VapGPT 파라미터 고정 (추론 전용; LM만 업데이트됨)
    vap_gpt_freeze: bool = True
    # VapGPT 내부 프레임률 (기본 50 fps)
    vap_gpt_frame_hz: int = 50
    # VapGPT 가 기대하는 오디오 샘플레이트 (Hz)
    vap_gpt_sample_rate: int = 16000
    # GPT 아키텍처 하이퍼파라미터 (사전학습 모델과 일치해야 함)
    vap_gpt_dim: int = 256
    vap_gpt_channel_layers: int = 1
    vap_gpt_cross_layers: int = 3
    vap_gpt_num_heads: int = 4
    vap_gpt_dropout: float = 0.1
    # CPC 인코더 고정 여부 (사전학습 가중치 사용 시 True 권장)
    # checkpoint_path가 설정된 경우 CPC는 체크포인트에서 로드되므로 별도 다운로드 없음.
    vap_gpt_freeze_encoder: bool = True
    # 대안 체크포인트: Lightning .ckpt 포맷도 지원
    # "/home2/s20235100/Conversational-AI/VoiceActivityProjection/example/50hz_48_10s-epoch20-val_1.85.ckpt"


@dataclass
class LoraArgs(Serializable):
    """Personaplex LMModel은 LoRA 레이어를 내장하지 않으므로,
    enable=False / full_finetuning=True 조합이 기본입니다.
    향후 LoRA 주입 지원 시 이 클래스를 확장할 수 있습니다."""
    enable: bool = False
    rank: int = 64
    scaling: float = 2.0
    ft_embed: bool = False  # True이면 임베딩 레이어도 학습

    def __post_init__(self) -> None:
        if self.enable:
            assert self.rank > 0
            assert self.scaling > 0.0


@dataclass
class OptimArgs(Serializable):
    lr: float = 1e-5
    weight_decay: float = 0.1
    pct_start: float = 0.05


@dataclass
class WandbArgs(Serializable):
    project: str | None = None  # None이면 wandb 미사용
    offline: bool = False
    key: str | None = None
    run_name: str | None = None
    local_dir: str = "/dev/shm/wandb"  # wandb staging 파일 저장 위치 (/tmp은 용량 부족, RAM tmpfs 사용)

    def __post_init__(self) -> None:
        if self.project is not None:
            try:
                import wandb  # noqa: F401
            except ImportError:
                raise ImportError(
                    "`wandb` not installed. Either make sure `wandb` is installed or set `wandb:project` to None."
                )
            if len(self.project) == 0:
                raise ValueError("`wandb.project` must not be an empty string.")


@dataclass
class ModelPaths(Serializable):
    """Personaplex 모델 가중치 경로 설정.
    local path를 지정하거나, hf_repo_id만 지정해 HuggingFace에서 자동 다운로드합니다.
    """
    hf_repo_id: str | None = "nvidia/personaplex-7b-v1"
    mimi_path: str | None = None      # tokenizer-e351c8d8-checkpoint125.safetensors
    moshi_path: str | None = None     # model.safetensors
    tokenizer_path: str | None = None # tokenizer_spm_32k_3.model
    config_path: str | None = None    # 사용자 정의 LM 설정 JSON (선택)


@dataclass
class FaceGenArgs(Serializable):
    """Face generation module settings (CausalSoftVQContinuousTransformer).

    Controlled by face_gen.enable in config.  When disabled the face module
    is never loaded, so no checkpoint path is required.
    """
    enable: bool = False

    # Path to a CausalSoftVQContinuousTransformer checkpoint (.ckpt).
    # Must be set when enable=True.
    ckpt_path: str | None = None

    # Model architecture — must match the saved checkpoint.
    # Defaults are read from the checkpoint's saved args when ckpt_path is given,
    # so these only need to be set if the checkpoint lacks an "args" key.
    hidden_dim: int = 512
    layers: int = 6
    heads: int = 8
    code_dim: int = 32
    prior_warmup_frames: int = 10

    # Sliding context window (in face frames at 25 fps) for autoregressive generation.
    max_context_frames: int = 25

    # ── ARTalkCodec (VAE) for z-space loss computation ────────────────────
    # Frozen codec used only to compute z_target = quant_to_sum_feat(gt_face_motion).
    codec_ckpt_path: str | None = None   # e.g. /home6/duplex/VAE_ami_dualtalk/checkpoints/iter_100000.pt
    codec_stats_path: str | None = None  # e.g. /home6/duplex/dataset/artalk_mimi54_stats.json
    # Codec architecture — must match the saved codec checkpoint.
    codec_hidden_dim: int = 512
    codec_depth: int = 8
    codec_heads: int = 8

    # ── FLAME / 3DMM data root ─────────────────────────────────────────────
    # Root directory containing FLAME .npy files structured as:
    #   {flame_root}/{speaker}/{split}/{stem}_{speaker}.npy
    # where speaker ∈ {"bc", "ut"} and split ∈ {"train", "valid", "test"}.
    flame_root: str = ""
    flame_speaker: str = "bc"  # primary speaker suffix for the agent channel

    # ── Per-component loss weights (from reference pretraining) ───────────
    # Overall weight applied to the sum of all face sub-losses.
    face_loss_weight: float = 1.0
    # Sub-loss weights (matching softvq_continuous_online_train.py defaults).
    motion_weight: float = 1.0       # L1 on pred_motion vs gt
    prior_weight: float = 0.5        # L1 on prior_motion vs gt
    z_weight: float = 0.2            # MSE on z_pred vs z_target (from codec)
    z_bce_weight: float = 0.1        # binary-CE on z_pred bits vs z_target bits
    jaw_weight: float = 2.0          # extra L1 weight on jaw dimension [50:51]
    vel_weight: float = 0.5          # MSE on frame velocity
    reg_weight: float = 0.02         # regularisation on delta and residual magnitude
    gate_weight: float = 1.0         # gate target MSE loss
    gate_target_expr: float = 0.45
    gate_target_jaw: float = 0.25
    gate_target_neck: float = 0.65
    gate_loss_expr_weight: float = 4.0
    gate_loss_jaw_weight: float = 4.0
    gate_loss_neck_weight: float = 1.0


@dataclass
class TrainArgs(Serializable):
    data: DataArgs

    run_dir: str  # 체크포인트와 로그가 저장될 디렉토리 (존재하지 않아야 함)
    moshi_paths: ModelPaths = field(default_factory=ModelPaths)

    # 손실 가중치
    first_codebook_weight_multiplier: float = 1.0
    text_padding_weight: float = 0.5
    epad_weight: float = 1.0  # EPAD(end-of-text-padding) 토큰 손실 가중치 (PAD와 별도)

    # 텍스트 프롬프트 (에이전트 지시문)
    # 비어 있으면 프롬프트 없이 학습 (기존 동작 유지).
    # 설정 시 매 학습 스텝에서 codes의 앞에 접두사로 삽입되어
    # inference의 step_system_prompts() 와 동일한 방식으로 LM에 제공됩니다.
    text_prompt: str = ""

    optim: OptimArgs = field(default_factory=OptimArgs)
    seed: int = 0
    num_microbatches: int = 1   # 그래디언트 누적 횟수

    duration_sec: float = 10.0  # 학습 세그먼트 길이(초)
    batch_size: int = 1
    max_norm: float = 1.0       # 그래디언트 클리핑 최대 norm
    max_steps: int = 1000
    log_freq: int = 10

    # 체크포인트
    ckpt_freq: int = 0          # 0이면 마지막 스텝에만 저장
    save_adapters: bool = True
    do_ckpt: bool = True
    num_ckpt_keep: int | None = 3

    # 평가
    eval_freq: int = 0
    do_eval: bool = False

    # 효율
    gradient_checkpointing: bool = True

    world_size: int | None = field(init=False, default=None)

    # 로깅
    wandb: WandbArgs = field(default_factory=WandbArgs)

    # LoRA (Personaplex는 기본 비활성화)
    lora: LoraArgs = field(default_factory=LoraArgs)
    # Personaplex는 LoRA 미지원이므로 full_finetuning=True가 기본값
    full_finetuning: bool = True

    # Backchannel VAP
    backchannel: BackchannelArgs = field(default_factory=BackchannelArgs)

    # Face generation (inference-only; not used during training)
    face_gen: FaceGenArgs = field(default_factory=FaceGenArgs)

    param_dtype: str = "bfloat16"
    overwrite_run_dir: bool = False

    def __post_init__(self) -> None:
        assert getattr(self, "world_size", None) is None
        self.world_size = int(os.environ.get("WORLD_SIZE", -1))

        if self.wandb.offline:
            command = f"cd {self.run_dir}; wandb sync --sync-all"
            logging.info(f"to sync wandb offline, run: {command}")

        assert self.num_microbatches >= 1
        assert self.num_ckpt_keep is None or self.num_ckpt_keep >= 1

        # Personaplex: LoRA 및 full_finetuning 지원
        if not self.lora.enable and not self.full_finetuning:
            logging.warning(
                "LoRA is disabled and full_finetuning is False. "
                "Forcing full_finetuning=True for Personaplex model."
            )
            self.full_finetuning = True
