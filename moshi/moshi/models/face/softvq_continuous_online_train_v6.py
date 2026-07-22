#!/usr/bin/env python
"""v6 trainer: v2+blink (v4 lineage) PLUS reactive-listening conditioning.

Adds two capabilities on top of the v4 production model
(``softvq_continuous_online_train_v2.py`` with the blink event head +
eyelid-aperture loss -- see docs/version_history_v2_to_v6.md):

  1. A causal PARTNER-AUDIO encoder stream.  In real duplex deployment the
     interlocutor's audio is live, so this is a legitimate inference-time
     input (unlike VAP below).  It lets listening motion react to what the
     OTHER speaker is saying instead of only settling via the EOS token.
  2. A VAP (Voice Activity Projection) auxiliary look-ahead head, predicting
     an 8-bit future voice-activity vector (200/400/600/800 ms bins x
     {own, partner}) from own+partner audio_h -- mirroring the mimi-token
     MTP head's "predict now, consume the prediction" pattern.  VAP LABELS
     ARE TRAIN-TIME ONLY (~21% coverage on dualtalk; ami_ut has none): the
     loss is masked to samples with a label, and at inference the model
     never sees a VAP feature -- only its own predicted probabilities are
     injected into the motion path, exactly like MTP tokens are never the
     real future audio at inference.

NOTE: v6 is a deliberately NEW architecture (extra input stream + heads), so
unlike v2's own checkpoints it is NOT expected to load v2/v4 checkpoints --
it trains from scratch.  See the open data-provenance caveat (VAP channel
order) in docs/version_history_v2_to_v6.md before scaling up training.
"""

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
            # v6: dualtalk is the only source with paired speaker1/speaker2
            # files (partner audio resolved generically in the loader) and
            # precomputed VAP labels; ami_bc/ami_ut simply omit VAP_ROOT.
            "VAP_ROOT": args.vap_root,
            "VAP_SPLITS": {"train": "train_vap", "val": "ood_vap", "test": "test_vap"},
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
        # Look-ahead training targets = real mimi semantic tokens (``*_token.pt``
        # sidecars produced by prepare_mimi_semantic_tokens.py).
        RETURN_AUDIO_TOKENS=args.lookahead_frames > 0,
        # v6 listening-mode extras (see class docstring / docs/version_history_v2_to_v6.md).
        RETURN_PARTNER_AUDIO=True,
        RETURN_VAP=args.vap_weight > 0.0,
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
        lookahead_frames: int = 0,
        token_vocab: int = 2048,
        la_temp: float = 0.5,
        chunk_frames: int = 1,
        partner_layers: int = 2,
        use_vap: bool = True,
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.code_dim = code_dim
        self.exp_res_scale = exp_res_scale
        self.jaw_res_scale = jaw_res_scale
        self.neck_res_scale = neck_res_scale
        self.prior_warmup_frames = int(prior_warmup_frames)
        # Block-causal audio context: the audio encoder lets each frame attend
        # bidirectionally within its ``chunk_frames``-frame block (and causally to
        # all earlier blocks), so audio_h carries up to (chunk-1) frames of real
        # future audio -> anticipatory coarticulation + locally-coherent audio_h ->
        # smoother motion.  The MOTION encoder stays strictly causal (frame-shifted
        # prev_motion would otherwise create an intra-block circular dependency at
        # generation).  chunk_frames=1 reproduces the original strictly-causal model.
        self.chunk_frames = max(1, int(chunk_frames))
        # Strictly-causal predictive look-ahead (MTP) over MIMI SEMANTIC TOKENS.
        # Unlike v1 (which learned its own VQ over audio_h -- a moving target that
        # needed EMA decay + dead-code restarts to stay usable), the prediction
        # target here is Mimi's first-codebook token stream: a FIXED, phoneme-like
        # tokenisation (WavLM-distilled) that the duplex pipeline produces anyway.
        # ``lookahead_frames`` counts MIMI frames (12.5 Hz), so each horizon step
        # is 2 face frames = 80 ms of anticipation.  From each causal face frame
        # the MTP module predicts the next K mimi tokens with a horizon-chained
        # recurrence, and the motion path reads the predicted-future slots through
        # cross-attention -> anticipation with zero added latency.  At inference
        # the realised mimi tokens verify the predictions (speculative decode).
        self.lookahead_frames = int(lookahead_frames)
        self.use_lookahead = self.lookahead_frames > 0
        self.token_vocab = int(token_vocab)
        self.la_temp = float(la_temp)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.llm_proj = nn.Linear(4096, hidden_dim, bias=False)
        # v6: motion path also reads the PARTNER audio stream (see class
        # docstring); prev_motion + own audio_h + partner_h.
        self.motion_proj = nn.Linear(motion_dim + hidden_dim * 2, hidden_dim)
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
        # v6: causal PARTNER-audio encoder (interlocutor's own mimi features --
        # live in real duplex deployment, unlike VAP below).  Deliberately
        # smaller/strictly-causal (no block-causal smoothing): it only needs to
        # carry coarse "what is the other person doing" context for reactive
        # listening, not fine lip coarticulation.
        self.partner_layers = max(1, int(partner_layers))
        partner_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.partner_proj = nn.Linear(audio_dim, hidden_dim)
        self.partner_encoder = nn.TransformerEncoder(partner_layer, num_layers=self.partner_layers)
        self.partner_norm = nn.LayerNorm(hidden_dim)
        # Learned bias substituted for partner_h when a record has no partner
        # file (e.g. every ami_ut sample): lets the model tell "no interlocutor
        # signal available" apart from "interlocutor is genuinely silent".
        self.partner_missing_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        # v6: VAP auxiliary look-ahead head -- predicts an 8-bit future
        # voice-activity vector (4 bins x {own,partner}) from own+partner
        # audio_h.  TRAIN-TIME signal only (see class docstring): the motion
        # path is conditioned on the model's OWN sigmoid predictions, never on
        # the raw vap_target, so nothing at inference requires a VAP feature.
        self.use_vap = bool(use_vap)
        if self.use_vap:
            self.vap_trunk = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
            self.vap_head = nn.Linear(hidden_dim, 8)
            self.vap_feat_proj = nn.Linear(8, hidden_dim)
            self.vap_inj_proj = nn.Linear(hidden_dim, hidden_dim)  # zero-init, see _init_weights
        # v6: explicit CONVERSATIONAL-ROLE conditioning -- disambiguates WHY
        # own audio is quiet.  Own-mic-quiet is ambiguous between "turn just
        # ended" (EOS, settle) and "listening to the partner" (should react);
        # without an explicit signal the model has to infer this from audio_h
        # alone.  Unlike blink (label needs GT motion) and VAP (precomputed,
        # train-only sidecar), the role label is a pure function of own+partner
        # RAW audio energy vs a causal EMA baseline -- IDENTICAL at train and
        # inference, no sampling/mismatch step needed (see _role_step).
        # role id: 0=speaking (own active), 1=listening (own quiet, partner
        # active), 2=mutual silence / no partner signal.  Zero-init -> old
        # behaviour reproduced exactly until trained.
        self.use_role = True
        self.role_embed = nn.Embedding(3, hidden_dim)
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
        # End-of-audio token (audio hidden space): used wherever the audio stream
        # has ended (training augmentation + inference past the provided audio) so
        # the model learns to settle to a neutral/held pose instead of hallucinating
        # motion from zero-padded audio.  Lets the duplex pipeline mark turn-end.
        self.eos_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        # Stochastic blink event head: blinks are audio-independent sparse events,
        # so a deterministic L1 regressor never INITIATES one in free-running AR
        # (the conditional median is "eyes open"; teacher forcing only continues
        # blinks already present in the GT prev frames).  Fix: predict a per-frame
        # Bernoulli blink state (labels come from a linear FLAME eyelid probe over
        # expr50), condition the motion path on the teacher-forced GT label in
        # training, and SAMPLE the flag at inference -> the decoder closes the
        # eyes because the flag says so.  Zero-init blink_embed keeps existing
        # checkpoints byte-identical until fine-tuned.  (Modules always exist;
        # with --blink-weight 0 they get no gradient, fine on single GPU -- under
        # DDP either train them or set find_unused_parameters=True.)
        self.blink_embed = nn.Embedding(2, hidden_dim)
        self.blink_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        # Linear eyelid probe (prepare_eyelid_probe.py; filled by load_eyelid_probe
        # or restored from a checkpoint): with pose zeroed, FLAME eyelid landmark
        # differences are exactly affine in expr50, so
        #   aperture(expr) = mean_p || eyelid_base_p + expr @ eyelid_weight_p ||.
        self.register_buffer("eyelid_base", torch.zeros(4, 3))
        self.register_buffer("eyelid_weight", torch.zeros(4, 50, 3))
        if self.use_lookahead:
            # Horizon-chained MTP: a shared trunk produces the horizon-1 state; each
            # further horizon is conditioned on the soft embedding of the previous
            # horizon's prediction through a GRU step (predictions stay a joint
            # rollout instead of K independent marginals -> far horizons stay sharp).
            self.mtp_trunk = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
            self.mtp_rnn = nn.GRUCell(hidden_dim, hidden_dim)
            self.mtp_out = nn.Linear(hidden_dim, self.token_vocab)
            # Embedding of mimi tokens used for the anticipation feature, for the
            # horizon chaining, and (via spec_feat_head) for speculative decode.
            self.la_token_embed = nn.Embedding(self.token_vocab, hidden_dim)
            # Cross-attention injection: the motion stream queries K future-token
            # slots (sharpened expected embedding + horizon embedding) so it can
            # read WHICH horizon carries an upcoming event, instead of receiving a
            # single softmax-averaged vector over the whole horizon.
            self.la_horizon_emb = nn.Embedding(self.lookahead_frames, hidden_dim)
            self.la_q_norm = nn.LayerNorm(hidden_dim)
            # Manual multi-head attention (1 query x K slots per frame).  NOT
            # nn.MultiheadAttention: its SDPA path launches one grid entry per
            # sequence, and the flattened B*T batch (1024x100 ~ 102k sequences)
            # exceeds the CUDA grid limit (65535) -> "invalid configuration
            # argument".  Plain batched matmuls have no such limit.
            self.la_heads = 4
            self.la_q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.la_k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.la_v_proj = nn.Linear(hidden_dim, hidden_dim)
            # Projects the attended anticipation before injecting into motion path.
            self.la_proj = nn.Linear(hidden_dim, hidden_dim)
            # Maps a token embedding to a hypothesised audio_h (trained against the
            # realised audio_h) so speculative decode can splice predicted tokens
            # into the motion path before the real audio frame arrives.
            self.spec_feat_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
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
        if self.use_lookahead:
            # Start with no look-ahead contribution so early training is stable.
            nn.init.zeros_(self.la_proj.weight)
            nn.init.zeros_(self.la_proj.bias)
        # Blink starts silent (zero embedding) and rare (logit -4 ~ p 0.018/frame)
        # so untrained/fine-tuned checkpoints behave exactly as before.
        nn.init.zeros_(self.blink_embed.weight)
        nn.init.zeros_(self.blink_head[-1].weight)
        nn.init.constant_(self.blink_head[-1].bias, -4.0)
        # VAP injection starts silent (zero-init), same stabilisation pattern
        # as la_proj: early training is unaffected by an untrained predictor.
        if self.use_vap:
            nn.init.zeros_(self.vap_inj_proj.weight)
            nn.init.zeros_(self.vap_inj_proj.bias)
        nn.init.zeros_(self.role_embed.weight)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    @staticmethod
    def _block_causal_mask(length: int, chunk: int, device: torch.device) -> torch.Tensor:
        """Block-causal mask (True = disallowed).

        A query attends to every key whose block index is <= its own, i.e. all
        earlier blocks (causal) plus *its whole* block (bidirectional within the
        block).  ``chunk<=1`` falls back to a strict causal mask.
        """
        if chunk <= 1:
            return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)
        blk = torch.arange(length, device=device) // chunk          # block index per pos
        return blk[None, :] > blk[:, None]                          # key block ahead of query block

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

    def _sample_pos_offset(self, face_len: int) -> int:
        """Random EVEN absolute-position offset for training-time position
        augmentation.  Training clips are short (clip_length frames) so positions
        ``clip_length..max_len`` of ``pos_emb`` would otherwise never be trained ->
        absolute-position streaming decode produces garbage past the clip length.
        Jittering the start position covers the whole table.  The offset is kept
        EVEN so each position keeps its (pos parity <-> phase) pairing (matches the
        offset=0 inference path)."""
        if not self.training:
            return 0
        max_off = self.pos_emb.num_embeddings - face_len
        if max_off <= 0:
            return 0
        return 2 * int(torch.randint(0, max_off // 2 + 1, (1,)).item())

    def _pos_phase(self, face_len: int, pos_offset: int, device: torch.device) -> torch.Tensor:
        base = torch.arange(face_len, device=device)
        phase = base.remainder(2)                                    # true sub-frame parity
        pos = (base + pos_offset).remainder(self.pos_emb.num_embeddings)
        return self.pos_emb(pos)[None] + self.phase_emb(phase)[None]

    # ------------------------------------------------------------------ #
    # Blink event head: labels, conditioning, and sampling.               #
    # ------------------------------------------------------------------ #
    def load_eyelid_probe(self, path: str):
        probe = torch.load(path, map_location="cpu", weights_only=True)
        self.eyelid_base.copy_(probe["base"].to(self.eyelid_base))
        self.eyelid_weight.copy_(probe["weight"].to(self.eyelid_weight))

    @property
    def blink_ready(self) -> bool:
        return bool(self.eyelid_weight.abs().sum().item() > 0)

    def eyelid_aperture(self, motion: torch.Tensor) -> torch.Tensor:
        """Motion [..., 54] -> eyelid aperture [...] in meters (linear FLAME probe)."""
        expr = motion[..., :50].float()
        d = self.eyelid_base + torch.einsum("...e,pec->...pc", expr, self.eyelid_weight)
        return d.norm(dim=-1).mean(dim=-1)

    def blink_labels(self, motion: torch.Tensor, thresh: float = 0.6) -> torch.Tensor:
        """Per-frame blink label [B, T]: 1 where the eyelids close below
        ``thresh`` x the clip's median aperture (same dip rule as offline eval)."""
        ap = self.eyelid_aperture(motion)
        med = ap.median(dim=1, keepdim=True).values.clamp_min(1e-6)
        return (ap < thresh * med).long()

    def _blink_logit(self, audio_h: torch.Tensor, prev_motion: torch.Tensor, pos_phase: torch.Tensor,
                     partner_h_t: torch.Tensor | None = None) -> torch.Tensor:
        """Blink logit from the pre-conditioning motion input (same features the
        batched path uses, recomputed so callers can sample BEFORE the step)."""
        if partner_h_t is None:
            partner_h_t = self.partner_missing_embed.to(audio_h.dtype).expand(audio_h.shape[0], audio_h.shape[1], -1)
        motion_in = self.motion_proj(torch.cat([prev_motion, audio_h, partner_h_t], dim=-1)) + pos_phase
        return self.blink_head(motion_in).squeeze(-1)

    @staticmethod
    def _init_blink_state(bsz: int, refractory: int, device: torch.device) -> dict:
        return {
            "active": torch.zeros(bsz, dtype=torch.bool, device=device),
            "since": torch.full((bsz,), refractory, dtype=torch.long, device=device),
        }

    def _sample_blink(self, logit: torch.Tensor, state: dict, temp: float,
                      refractory: int, stochastic: bool) -> torch.Tensor:
        """One per-frame blink decision [B].  A new blink may only START once the
        previous one ended >= ``refractory`` frames ago; an active blink continues
        while the head keeps voting 1 (closed-run length is learned, not fixed)."""
        prob = torch.sigmoid(logit.float() / max(temp, 1e-4))
        if stochastic:
            vote = torch.rand_like(prob) < prob
        else:
            vote = prob > 0.5
        allow_new = state["since"] >= refractory
        flag = torch.where(state["active"], vote, vote & allow_new)
        state["active"] = flag
        state["since"] = torch.where(flag, torch.zeros_like(state["since"]), state["since"] + 1)
        return flag.long()

    def _encode_audio(
        self,
        audio_feat: torch.Tensor,
        llm_feat: torch.Tensor | None,
        face_len: int,
        pos_offset: int = 0,
        eos_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Causal audio encoder -> contextualised audio_h ([B, face_len, H]).

        ``eos_mask`` [B, face_len] (bool) marks frames whose audio has ended -> the
        learned ``eos_embed`` replaces the projected audio there.  At eval, frames
        past the provided audio are auto-marked EOS (so generation beyond the audio
        settles instead of running on zero padding)."""
        audio_rep = self._expand_audio(audio_feat, face_len)
        llm_rep = self._expand_llm(llm_feat, audio_feat, face_len)
        pos_phase = self._pos_phase(face_len, pos_offset, audio_feat.device)
        a_hidden = self.audio_proj(audio_rep)
        if eos_mask is None and not self.training:
            real = audio_feat.shape[1] * 2  # face frames covered by real audio
            if real < face_len:
                idxs = torch.arange(face_len, device=audio_feat.device)
                eos_mask = (idxs >= real)[None].expand(audio_feat.shape[0], face_len)
        if eos_mask is not None:
            a_hidden = torch.where(eos_mask[..., None], self.eos_embed.to(a_hidden.dtype), a_hidden)
        mask = self._block_causal_mask(face_len, self.chunk_frames, audio_feat.device)
        audio_h = a_hidden + self.llm_proj(llm_rep) + pos_phase
        audio_h = self.audio_encoder(audio_h, mask=mask)
        audio_h = self.audio_norm(audio_h)
        return audio_h

    def _encode_partner(
        self,
        partner_audio_feat: torch.Tensor | None,
        partner_available: torch.Tensor | None,
        face_len: int,
        pos_offset: int,
        template: torch.Tensor,
    ) -> torch.Tensor:
        """Causal partner-audio encoder -> [B, face_len, H].

        ``partner_audio_feat``/``partner_available`` may be None (no partner
        stream requested at all, e.g. legacy callers) -- returns the missing
        embedding broadcast for every frame, same as a per-record unavailable
        flag would.  ``template`` supplies device/dtype/batch when
        ``partner_audio_feat`` is None.
        """
        bsz = template.shape[0]
        device = template.device
        if partner_audio_feat is None:
            return self.partner_missing_embed.to(template.dtype).expand(bsz, face_len, -1)
        rep = self._expand_audio(partner_audio_feat, face_len)
        pos_phase = self._pos_phase(face_len, pos_offset, device)
        h = self.partner_proj(rep) + pos_phase
        mask = self._causal_mask(face_len, device)
        h = self.partner_encoder(h, mask=mask)
        h = self.partner_norm(h)
        if partner_available is not None:
            avail = partner_available.to(h.dtype).view(bsz, 1, 1)
            h = avail * h + (1.0 - avail) * self.partner_missing_embed.to(h.dtype)
        return h

    # ------------------------------------------------------------------ #
    # Conversational-role conditioning: rule-based (no learned params),  #
    # identical computation in batched training and incremental streaming. #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _init_activity_state(bsz: int, device: torch.device) -> dict:
        return {
            "own_ema": torch.zeros(bsz, device=device),
            "partner_ema": torch.zeros(bsz, device=device),
        }

    @staticmethod
    def _role_step(
        own_feat_t: torch.Tensor,
        partner_feat_t: torch.Tensor | None,
        partner_avail_t: torch.Tensor | None,
        state: dict,
        decay: float = 0.98,
        ratio: float = 1.5,
    ) -> tuple[torch.Tensor, dict]:
        """One frame of role inference from RAW (pre-encoder) mimi features.

        ``own_feat_t``/``partner_feat_t`` are [B, D] (partner may be None: no
        interlocutor stream at all).  "Active" = current L2 energy exceeds
        ``ratio`` x a causal EMA baseline of past energy (bootstraps to
        "active" for a frame or two at t=0 -- negligible, pre-warmup anyway).
        Returns role id [B] long: 0=speaking, 1=listening, 2=mutual silence.
        """
        bsz = own_feat_t.shape[0]
        device = own_feat_t.device
        own_energy = own_feat_t.float().norm(dim=-1)
        own_active = own_energy > (state["own_ema"] * ratio).clamp_min(1e-3)
        state["own_ema"] = decay * state["own_ema"] + (1.0 - decay) * own_energy
        if partner_feat_t is not None:
            partner_energy = partner_feat_t.float().norm(dim=-1)
            partner_active = partner_energy > (state["partner_ema"] * ratio).clamp_min(1e-3)
            state["partner_ema"] = decay * state["partner_ema"] + (1.0 - decay) * partner_energy
            if partner_avail_t is not None:
                partner_active = partner_active & partner_avail_t
        else:
            partner_active = torch.zeros(bsz, dtype=torch.bool, device=device)
        role = torch.where(
            own_active,
            torch.zeros(bsz, dtype=torch.long, device=device),
            torch.where(
                partner_active,
                torch.ones(bsz, dtype=torch.long, device=device),
                torch.full((bsz,), 2, dtype=torch.long, device=device),
            ),
        )
        return role, state

    def _role_labels_batched(
        self,
        audio_feat: torch.Tensor,
        partner_audio_feat: torch.Tensor | None,
        partner_available: torch.Tensor | None,
        face_len: int,
    ) -> torch.Tensor:
        """Batched counterpart of ``_role_step`` over a whole clip -> [B, face_len] long.

        Runs at MIMI-frame granularity (own/partner audio's native rate) then
        repeat_interleave(2) to face rate, matching ``_expand_audio``.  A
        Python loop over ``T_mimi`` (~50 for a 100-frame clip) of cheap tensor
        ops -- negligible next to the transformer passes in the same forward.
        """
        bsz, tm, _ = audio_feat.shape
        device = audio_feat.device
        state = self._init_activity_state(bsz, device)
        roles = []
        for i in range(tm):
            partner_feat_i = None
            if partner_audio_feat is not None:
                partner_feat_i = (
                    partner_audio_feat[:, i] if i < partner_audio_feat.shape[1]
                    else partner_audio_feat.new_zeros(bsz, partner_audio_feat.shape[-1])
                )
            role_i, state = self._role_step(audio_feat[:, i], partner_feat_i, partner_available, state)
            roles.append(role_i)
        role_mimi = torch.stack(roles, dim=1)  # [B, Tm]
        role_face = role_mimi.repeat_interleave(2, dim=1)
        if role_face.shape[1] < face_len:
            role_face = F.pad(role_face, (0, face_len - role_face.shape[1]), value=2)
        return role_face[:, :face_len]

    def _role_at_step(
        self,
        audio_feat: torch.Tensor,
        partner_audio_feat: torch.Tensor | None,
        partner_available: torch.Tensor | None,
        t: int,
        state: dict,
    ) -> tuple[torch.Tensor, dict]:
        """Incremental counterpart of ``_role_labels_batched`` for absolute
        face-frame ``t`` (same ``ai = t // 2`` indexing as ``_encode_audio_step``)."""
        bsz = audio_feat.shape[0]
        ai = t // 2
        own_feat_t = (
            audio_feat[:, ai] if ai < audio_feat.shape[1]
            else audio_feat.new_zeros(bsz, audio_feat.shape[-1])
        )
        partner_feat_t = None
        if partner_audio_feat is not None:
            partner_feat_t = (
                partner_audio_feat[:, ai] if ai < partner_audio_feat.shape[1]
                else partner_audio_feat.new_zeros(bsz, partner_audio_feat.shape[-1])
            )
        return self._role_step(own_feat_t, partner_feat_t, partner_available, state)

    def _vap_predict(self, audio_h: torch.Tensor, partner_h: torch.Tensor) -> torch.Tensor:
        """Predict the 8-bit future voice-activity vector -> logits [B, T, 8].

        TRAIN-TIME target only; see class docstring.  Only own+partner audio_h
        are used (no motion/pose leakage), so the same call is valid whether
        audio_h is real (batched/cached) or a speculative candidate.
        """
        state = self.vap_trunk(torch.cat([audio_h, partner_h], dim=-1))
        return self.vap_head(state)

    def _vap_feature(self, vap_logits: torch.Tensor) -> torch.Tensor:
        """Predicted-probability feature injected into the motion path.
        ``vap_inj_proj`` is zero-init so this starts silent (see _init_weights).
        """
        probs = torch.sigmoid(vap_logits.float()).to(vap_logits.dtype)
        return self.vap_inj_proj(self.vap_feat_proj(probs))

    def _mtp_predict(self, audio_h: torch.Tensor) -> torch.Tensor:
        """Horizon-chained multi-token prediction -> [B, T, K, vocab].

        From each causal frame, horizon 1 is read off the trunk state; every later
        horizon k is conditioned on the soft embedding of the horizon-(k-1)
        prediction via one GRU step, so the K predictions form a joint rollout.
        """
        bsz, length, hidden = audio_h.shape
        state = self.mtp_trunk(audio_h).reshape(bsz * length, hidden)
        logits_steps = []
        for k in range(self.lookahead_frames):
            logits_k = self.mtp_out(state)  # [B*T, vocab]
            logits_steps.append(logits_k)
            if k + 1 < self.lookahead_frames:
                # Explicit dtype keeps autocast from upcasting the [B*T, vocab]
                # softmax to fp32 (vocab=2048 -> the fp32 copies saved for backward
                # OOM'd a 44GB card at batch 1024).
                soft_embed = F.softmax(logits_k, dim=-1, dtype=logits_k.dtype) @ self.la_token_embed.weight
                state = self.mtp_rnn(soft_embed, state)
        return torch.stack(logits_steps, dim=1).view(bsz, length, self.lookahead_frames, self.token_vocab)

    def _lookahead_feature(self, motion_in: torch.Tensor, mtp_logits: torch.Tensor) -> torch.Tensor:
        """Cross-attend the motion stream to the K predicted-future token slots.

        Each slot is the temperature-sharpened expected token embedding of one
        horizon plus its horizon embedding (sharpening keeps a multimodal future
        from collapsing to a nonexistent "average" token).  ``la_proj`` is
        zero-initialised so the injection starts silent.
        """
        bsz, length, k_h, _ = mtp_logits.shape
        # Explicit dtype: keep the [B, T, K, vocab] softmax out of fp32 (see
        # _mtp_predict); bf16 softmax after the internal max-subtraction is fine.
        probs = F.softmax(mtp_logits / max(self.la_temp, 1e-4), dim=-1, dtype=mtp_logits.dtype)
        fut = probs @ self.la_token_embed.weight  # [B, T, K, H]
        slots = fut + self.la_horizon_emb.weight.view(1, 1, k_h, -1)
        nh = self.la_heads
        dk = slots.shape[-1] // nh
        q = self.la_q_proj(self.la_q_norm(motion_in)).view(bsz, length, nh, 1, dk)
        k = self.la_k_proj(slots).view(bsz, length, k_h, nh, dk).permute(0, 1, 3, 2, 4)
        v = self.la_v_proj(slots).view(bsz, length, k_h, nh, dk).permute(0, 1, 3, 2, 4)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dk)  # [B, T, nh, 1, K]
        att = torch.matmul(scores.softmax(dim=-1), v)  # [B, T, nh, 1, dk]
        return self.la_proj(att.reshape(bsz, length, nh * dk))

    def forward(
        self,
        audio_feat: torch.Tensor,
        prev_motion: torch.Tensor,
        llm_feat: torch.Tensor | None = None,
        eos_mask: torch.Tensor | None = None,
        blink_cond: torch.Tensor | None = None,
        partner_audio_feat: torch.Tensor | None = None,
        partner_available: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        face_len = prev_motion.shape[1]
        pos_offset = self._sample_pos_offset(face_len)  # 0 at eval
        audio_h = self._encode_audio(audio_feat, llm_feat, face_len, pos_offset, eos_mask)
        partner_h = self._encode_partner(partner_audio_feat, partner_available, face_len, pos_offset, audio_h)
        role_cond = None
        if self.use_role:
            role_cond = self._role_labels_batched(audio_feat, partner_audio_feat, partner_available, face_len)
        return self._decode_motion(audio_h, prev_motion, pos_offset, blink_cond=blink_cond, partner_h=partner_h,
                                   role_cond=role_cond)

    def _decode_motion(self, audio_h: torch.Tensor, prev_motion: torch.Tensor, pos_offset: int = 0,
                       blink_cond: torch.Tensor | None = None, partner_h: torch.Tensor | None = None,
                       role_cond: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Motion path + heads given (real or hypothesised) audio_h.

        Accepting audio_h directly lets the speculative decoder reuse real frames
        and splice in candidate frames built from predicted tokens.
        """
        bsz, face_len, _ = prev_motion.shape
        pos_phase = self._pos_phase(face_len, pos_offset, prev_motion.device)
        mask = self._causal_mask(face_len, prev_motion.device)
        if partner_h is None:
            partner_h = self.partner_missing_embed.to(audio_h.dtype).expand(bsz, face_len, -1)

        # Strictly-causal predictive look-ahead (MTP): predict the next K mimi
        # tokens from each causal frame and let the motion path cross-attend to
        # the predicted-future slots.  Targets come from the dataset (real mimi
        # semantic tokens), so nothing here needs a learned tokeniser.
        mtp_logits = None
        motion_h = self.motion_proj(torch.cat([prev_motion, audio_h, partner_h], dim=-1)) + pos_phase
        # Blink logit comes from the PRE-conditioning input (the flag must be
        # decidable before it is injected); conditioning is teacher-forced GT
        # labels in training and the sampled flag at inference.
        blink_logit = self.blink_head(motion_h).squeeze(-1)  # [B, T]
        if blink_cond is not None:
            motion_h = motion_h + self.blink_embed(blink_cond)
        # Conversational-role conditioning (speaking/listening/mutual-silence):
        # rule-derived, identical at train and inference -- see _role_step.
        if role_cond is not None:
            motion_h = motion_h + self.role_embed(role_cond)
        if self.use_lookahead:
            mtp_logits = self._mtp_predict(audio_h)
            motion_h = motion_h + self._lookahead_feature(motion_h, mtp_logits)
        vap_logits = None
        if self.use_vap:
            vap_logits = self._vap_predict(audio_h, partner_h)
            motion_h = motion_h + self._vap_feature(vap_logits)
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
            "mtp_logits": mtp_logits,
            "audio_h": audio_h,
            "blink_logit": blink_logit,
            "vap_logits": vap_logits,
        }

    # ------------------------------------------------------------------ #
    # KV-cached streaming path (absolute positions, train-consistent).    #
    # Reuses the trained nn.TransformerEncoder weights; only the          #
    # attention is recomputed incrementally so past frames are not        #
    # re-encoded every step (O(W) per frame instead of O(W^2)).           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _encoder_layer_step(layer, x_new, cache_k, cache_v, causal_new: bool = True):
        """One ``nn.TransformerEncoderLayer(norm_first=True)`` step with KV cache.

        ``cache_k``/``cache_v`` are [B, heads, T_past, head_dim] (or None).  Returns
        the new tokens' output and the full (past+new) K/V to store as the cache.
        ``causal_new=False`` lets the new tokens attend to each other bidirectionally
        (used for block-causal audio encoding: the whole new block is one block, and
        all cached keys are earlier blocks -> full attention over past+block).
        """
        attn = layer.self_attn
        num_heads = attn.num_heads
        bsz, n_new, dim = x_new.shape
        head_dim = dim // num_heads
        h = layer.norm1(x_new)
        qkv = F.linear(h, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, n_new, num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, n_new, num_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, n_new, num_heads, head_dim).transpose(1, 2)
        if cache_k is not None:
            k = torch.cat([cache_k, k], dim=2)
            v = torch.cat([cache_v, v], dim=2)
        if causal_new:
            # Explicit cache-aware causal mask: new query i (i in 0..n_new-1) may
            # attend to all cached keys plus new keys up to itself.  (SDPA's
            # is_causal aligns the triangle top-left, wrong when q_len != k_len.)
            k_len = k.shape[2]
            t_past = k_len - n_new
            qi = torch.arange(n_new, device=x_new.device).view(n_new, 1)
            kj = torch.arange(k_len, device=x_new.device).view(1, k_len)
            allow = kj <= (t_past + qi)  # [n_new, k_len] bool, True = attend
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)
        else:
            # Block step: cached keys are all earlier blocks (fully visible) and the
            # new tokens form one block (bidirectional) -> attend to everything.
            attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, n_new, dim)
        x = x_new + attn.out_proj(attn_out)
        h2 = layer.norm2(x)
        x = x + layer.linear2(F.gelu(layer.linear1(h2)))
        return x, k, v

    def _run_encoder_step(self, encoder, x_new, cache, causal_new: bool = True):
        new_cache = []
        for idx, layer in enumerate(encoder.layers):
            ck, cv = (None, None) if cache is None else cache[idx]
            x_new, k, v = self._encoder_layer_step(layer, x_new, ck, cv, causal_new=causal_new)
            new_cache.append((k, v))
        return x_new, new_cache

    def _pos_phase_at(self, t: int, device: torch.device) -> torch.Tensor:
        pos = t % self.pos_emb.num_embeddings
        idx = torch.tensor([pos], device=device)
        return (self.pos_emb(idx) + self.phase_emb(idx.remainder(2)))[None]  # [1,1,H]

    def _encode_audio_step(self, audio_feat, llm_feat, t, cache):
        """Incremental audio encoder for absolute face-frame ``t`` -> audio_h_t."""
        bsz = audio_feat.shape[0]
        ai = t // 2  # face frame t <- mimi feature t//2 (repeat_interleave(2))
        if ai < audio_feat.shape[1]:
            a_hidden = self.audio_proj(audio_feat[:, ai : ai + 1])
        else:
            # past the provided audio -> end-of-audio token (settle)
            a_hidden = self.eos_embed.to(audio_feat.dtype).expand(bsz, 1, -1)
        x = a_hidden + self._pos_phase_at(t, audio_feat.device)
        if llm_feat is not None and ai < llm_feat.shape[1]:
            x = x + self.llm_proj(llm_feat[:, ai : ai + 1])
        x, new_cache = self._run_encoder_step(self.audio_encoder, x, cache)
        return self.audio_norm(x), new_cache

    def _encode_audio_block(self, audio_feat, llm_feat, t0, blk, cache):
        """Block-causal incremental audio encoder: encode the ``blk`` frames
        starting at absolute frame ``t0`` as one block (bidirectional within the
        block, full attention to cached earlier blocks).  Returns [B, blk, H]."""
        bsz = audio_feat.shape[0]
        feats = []
        for j in range(blk):
            t = t0 + j
            ai = t // 2  # face frame t <- mimi feature t//2 (repeat_interleave(2))
            if ai < audio_feat.shape[1]:
                a_hidden = self.audio_proj(audio_feat[:, ai : ai + 1])
            else:
                a_hidden = self.eos_embed.to(audio_feat.dtype).expand(bsz, 1, -1)  # end-of-audio
            xj = a_hidden + self._pos_phase_at(t, audio_feat.device)
            if llm_feat is not None and ai < llm_feat.shape[1]:
                xj = xj + self.llm_proj(llm_feat[:, ai : ai + 1])
            feats.append(xj)
        x = torch.cat(feats, dim=1)  # [B, blk, H]
        x, new_cache = self._run_encoder_step(self.audio_encoder, x, cache, causal_new=False)
        return self.audio_norm(x), new_cache

    def _encode_partner_step(self, partner_audio_feat, t, cache):
        """Incremental partner-audio encoder for absolute face-frame ``t``.

        Strictly causal, one frame at a time (no block smoothing -- see
        ``_encode_partner``).  ``partner_audio_feat`` may be None (no
        interlocutor stream at all) -> returns the missing embedding, cache
        untouched (None stays None so the caller keeps skipping this path)."""
        if partner_audio_feat is None:
            return None, cache
        bsz = partner_audio_feat.shape[0]
        ai = t // 2
        if ai < partner_audio_feat.shape[1]:
            h = self.partner_proj(partner_audio_feat[:, ai : ai + 1])
        else:
            h = self.partner_missing_embed.to(partner_audio_feat.dtype).expand(bsz, 1, -1)
        x = h + self._pos_phase_at(t, partner_audio_feat.device)
        x, new_cache = self._run_encoder_step(self.partner_encoder, x, cache)
        return self.partner_norm(x), new_cache

    def _decode_motion_step(self, audio_h_t, prev_motion_t, t, cache, blink_t=None, partner_h_t=None,
                            role_t=None):
        """Incremental counterpart of ``_decode_motion`` for a single frame ``t``.

        ``blink_t`` [B] long is the blink flag decided by the caller (sampled via
        ``_blink_logit`` + ``_sample_blink`` BEFORE the step); None = no injection.
        ``partner_h_t`` [B,1,H] is this frame's partner-audio encoding, or None
        if no interlocutor stream is being used (falls back to the missing
        embedding, same as the batched path).  ``role_t`` [B] long is this
        frame's conversational-role id (see ``_role_step``); None = no injection.
        """
        if partner_h_t is None:
            partner_h_t = self.partner_missing_embed.to(audio_h_t.dtype).expand(audio_h_t.shape[0], 1, -1)
        pos_phase = self._pos_phase_at(t, audio_h_t.device)
        motion_in = self.motion_proj(torch.cat([prev_motion_t, audio_h_t, partner_h_t], dim=-1)) + pos_phase
        if blink_t is not None:
            motion_in = motion_in + self.blink_embed(blink_t)[:, None, :]
        if role_t is not None:
            motion_in = motion_in + self.role_embed(role_t)[:, None, :]
        mtp_logits = None
        if self.use_lookahead:
            mtp_logits = self._mtp_predict(audio_h_t)
            motion_in = motion_in + self._lookahead_feature(motion_in, mtp_logits)
        if self.use_vap:
            vap_logits_t = self._vap_predict(audio_h_t, partner_h_t)
            motion_in = motion_in + self._vap_feature(vap_logits_t)
        x, new_cache = self._run_encoder_step(self.motion_encoder, motion_in, cache)
        motion_h = self.motion_norm(x)

        z = self.z_head(audio_h_t)
        prior = self.prior_head(torch.cat([audio_h_t, z], dim=-1))
        dyn_h = torch.cat([motion_h, audio_h_t], dim=-1)
        delta = self._scale_motion(self.delta_head(dyn_h))
        residual = self._scale_motion(self.res_head(dyn_h))
        candidate = prev_motion_t + delta
        group_gate = torch.sigmoid(
            self.gate_head(torch.cat([dyn_h, prev_motion_t, prior, candidate, candidate - prior], dim=-1))
        )
        gate = self._expand_group_gate(group_gate)
        if self.prior_warmup_frames > 0:
            warmup = min(t / float(self.prior_warmup_frames), 1.0)
            gate = gate * warmup
        final = gate * candidate + (1.0 - gate) * prior + residual
        out = {
            "pred_motion": final,
            "prior_motion": prior,
            "mtp_logits": mtp_logits,
            "audio_h": audio_h_t,
        }
        return out, new_cache

    @staticmethod
    def _truncate_cache(cache, max_len):
        if cache is None or max_len is None:
            return cache
        return [(k[:, :, -max_len:, :], v[:, :, -max_len:, :]) for (k, v) in cache]

    @torch.no_grad()
    def generate_cached(
        self,
        audio_feat: torch.Tensor,
        max_frames: int | None = None,
        llm_feat: torch.Tensor | None = None,
        max_context_frames: int | None = None,
        blink_sample: bool = True,
        blink_temp: float = 1.0,
        blink_refractory: int = 12,
        partner_audio_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Streaming decode with per-layer KV cache (absolute positions).

        Numerically equivalent to a full-context ``generate`` that uses absolute
        positions; past frames are encoded once and cached.  ``max_context_frames``
        evicts the oldest cached K/V (true sliding-window attention, positions kept).
        Blink flags are sampled per frame from the blink head (``blink_sample=False``
        thresholds at p>0.5 instead -> deterministic); models without a loaded
        eyelid probe skip blink conditioning entirely.  ``partner_audio_feat`` is
        the interlocutor's live Mimi features (real duplex input, unlike VAP,
        which is never fed here) -- None reproduces the no-listening-signal
        v4 behaviour exactly (missing embedding, same as an unpaired record).
        """
        bsz = audio_feat.shape[0]
        face_len = int(audio_feat.shape[1] * 2) if max_frames is None else int(max_frames)
        cap = None if max_context_frames is None else max(1, int(max_context_frames))
        C = self.chunk_frames
        audio_cache = None
        motion_cache = None
        partner_cache = None
        prev_motion = self.start_motion.expand(bsz, 1, -1).to(audio_feat)
        if self.blink_ready:
            blink_state = self._init_blink_state(bsz, blink_refractory, audio_feat.device)
        else:
            blink_state = None
        role_state = self._init_activity_state(bsz, audio_feat.device) if self.use_role else None
        preds = []
        t = 0
        while t < face_len:
            blk = min(C, face_len - t)
            # Encode the whole audio block at once (block-causal -> each frame sees
            # the block's real future audio); then decode motion frame-by-frame
            # (the motion path stays strictly causal).
            audio_h_blk, audio_cache = self._encode_audio_block(audio_feat, llm_feat, t, blk, audio_cache)
            for j in range(blk):
                audio_h_t = audio_h_blk[:, j : j + 1]
                partner_h_t, partner_cache = self._encode_partner_step(partner_audio_feat, t + j, partner_cache)
                if blink_state is None:
                    blink_t = None
                else:
                    logit = self._blink_logit(audio_h_t, prev_motion, self._pos_phase_at(t + j, audio_feat.device),
                                              partner_h_t=partner_h_t)
                    blink_t = self._sample_blink(logit[:, 0], blink_state, blink_temp, blink_refractory, blink_sample)
                if role_state is None:
                    role_t = None
                else:
                    role_t, role_state = self._role_at_step(audio_feat, partner_audio_feat, None, t + j, role_state)
                out, motion_cache = self._decode_motion_step(
                    audio_h_t, prev_motion, t + j, motion_cache, blink_t=blink_t, partner_h_t=partner_h_t,
                    role_t=role_t,
                )
                prev_motion = out["pred_motion"]
                preds.append(prev_motion)
            audio_cache = self._truncate_cache(audio_cache, cap)
            motion_cache = self._truncate_cache(motion_cache, cap)
            partner_cache = self._truncate_cache(partner_cache, cap)
            t += blk
        return torch.cat(preds, dim=1)

    @staticmethod
    def _expand_cache(cache, k):
        return [(ck.repeat_interleave(k, dim=0), cv.repeat_interleave(k, dim=0)) for ck, cv in cache]

    @torch.no_grad()
    def generate_speculative_cached(
        self,
        audio_feat: torch.Tensor,
        mimi_token: torch.Tensor,
        max_frames: int | None = None,
        llm_feat: torch.Tensor | None = None,
        max_context_frames: int | None = None,
        topk: int = 4,
        accept_threshold: float = 0.5,
        spec_temp: float = 0.5,
        blink_sample: bool = True,
        blink_temp: float = 1.0,
        blink_refractory: int = 12,
        partner_audio_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """KV-cached streaming decode with MTP candidates + soft speculative
        verification against the REALISED mimi token stream.

        ``mimi_token`` [B, T_mimi] is the semantic (first-codebook) token of the
        same audio -- in the duplex pipeline it exists anyway, since the audio
        arrives through the Mimi encoder.  Each face frame maps to mimi frame
        ``t // 2``.  For frame t+1 the model proposes candidate tokens (the known
        realised token when t+1 stays in the same mimi frame; the horizon-1 MTP
        top-k when t+1 opens a new one), precomputes a candidate mouth shape per
        token by splicing ``spec_feat_head(la_token_embed(token))`` into a forked
        motion cache, then blends candidates by predicted probability x acoustic
        closeness once the real frame arrives.  Confidence = weight mass on
        candidates matching the realised token; low-confidence frames fall back
        smoothly to the exact recompute (conf=0 equals ``generate_cached``).
        """
        assert self.use_lookahead, "generate_speculative_cached requires lookahead_frames > 0"
        bsz = audio_feat.shape[0]
        face_len = int(audio_feat.shape[1] * 2) if max_frames is None else int(max_frames)
        cap = None if max_context_frames is None else max(1, int(max_context_frames))
        topk = max(1, min(int(topk), self.token_vocab))
        n_mimi = mimi_token.shape[1]

        C = self.chunk_frames
        audio_cache = None
        motion_cache = None
        partner_cache = None
        prev_motion = self.start_motion.expand(bsz, 1, -1).to(audio_feat)
        cand_tokens = cand_probs = cand_motions = None
        audio_h_blk = None
        # Partner audio is REAL (not speculative -- the interlocutor's audio up
        # to "now" is fully known in a live duplex stream), so its cache just
        # advances one frame ahead of the committed motion, same pattern as
        # the block-causal audio buffer.  partner_h_next feeds both the t+1
        # candidate precompute AND the t+1 real step, no forking needed.
        partner_h_next, partner_cache = self._encode_partner_step(partner_audio_feat, 0, partner_cache)
        if self.blink_ready:
            blink_state = self._init_blink_state(bsz, blink_refractory, audio_feat.device)
        else:
            blink_state = None
        # Role is real & deterministic (not speculative), so -- like partner_h_next
        # -- it just advances one frame ahead in lockstep with the committed
        # timeline; no propose/reuse-on-mismatch logic needed (unlike blink).
        if self.use_role:
            role_state = self._init_activity_state(bsz, audio_feat.device)
            role_next, role_state = self._role_at_step(audio_feat, partner_audio_feat, None, 0, role_state)
        else:
            role_state = None
            role_next = None
        next_blink = None  # frame t's flag, decided at frame t-1's proposal time
        preds = []
        hit_mass = 0.0
        recompute_count = 0.0
        steps = 0.0
        for t in range(face_len):
            # Block-causal audio: encode the whole block when entering it, then serve
            # frames from the buffer (matches training; chunk=1 -> per-frame as before).
            if t % C == 0:
                blk = min(C, face_len - t)
                audio_h_blk, audio_cache = self._encode_audio_block(audio_feat, llm_feat, t, blk, audio_cache)
                audio_cache = self._truncate_cache(audio_cache, cap)
            audio_h_t = audio_h_blk[:, (t % C) : (t % C) + 1]
            realized_h = audio_h_t[:, 0]
            m = t // 2
            if m < n_mimi:
                realized_token = mimi_token[:, m]
            else:
                # Past the provided token stream (EOS territory): nothing to verify
                # against -> candidates never match -> conf 0 -> exact path.
                realized_token = mimi_token.new_full((bsz,), -1)
            # Real step always advances the committed motion cache (defines history).
            partner_h_t = partner_h_next  # computed for frame t at the end of iteration t-1 (or t==0 above)
            role_t = role_next  # ditto
            if blink_state is None:
                blink_t = None
            elif next_blink is None:  # t == 0: nothing proposed yet
                logit = self._blink_logit(audio_h_t, prev_motion, self._pos_phase_at(t, audio_feat.device),
                                          partner_h_t=partner_h_t)
                blink_t = self._sample_blink(logit[:, 0], blink_state, blink_temp, blink_refractory, blink_sample)
            else:
                # Reuse the flag decided at proposal time so the exact recompute
                # matches the candidates it is blended with.
                blink_t = next_blink
            out, motion_cache = self._decode_motion_step(audio_h_t, prev_motion, t, motion_cache, blink_t=blink_t,
                                                          partner_h_t=partner_h_t, role_t=role_t)
            exact_motion = out["pred_motion"]

            if cand_tokens is None:
                next_motion = exact_motion
                recompute_count += float(bsz)
            else:
                cand_embed = self.spec_feat_head(self.la_token_embed(cand_tokens))  # [B, k, H]
                dist = (cand_embed - realized_h[:, None]).pow(2).mean(dim=-1)
                weight = cand_probs * torch.softmax(-dist / max(spec_temp, 1e-4), dim=1)
                weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
                spec_motion = (weight[..., None] * cand_motions).sum(dim=1, keepdim=True)
                exact_match = (cand_tokens == realized_token[:, None]).to(weight.dtype)
                conf = (weight * exact_match).sum(dim=1)
                hit_mass += float(conf.sum().item())
                steps += float(bsz)
                if bool((conf >= accept_threshold).all().item()):
                    next_motion = spec_motion
                else:
                    c = conf.view(bsz, 1, 1)
                    next_motion = c * spec_motion + (1.0 - c) * exact_motion
                    recompute_count += float(bsz)

            preds.append(next_motion)
            prev_motion = next_motion
            motion_cache = self._truncate_cache(motion_cache, cap)

            # Propose + precompute candidate mouth shapes for frame t+1.
            if (t + 1) // 2 == m and m < n_mimi:
                # Frame t+1 shares this frame's mimi token -> the "candidate" is the
                # realised token itself (certain), only its audio_h is hypothesised.
                cand_tokens = realized_token.clamp(min=0)[:, None]  # [B, 1]
                cand_probs = torch.ones(bsz, 1, device=audio_feat.device, dtype=audio_h_t.dtype)
            else:
                next_logits = out["mtp_logits"][:, 0, 0]  # horizon-1 -> next mimi frame
                topv, cand_tokens = next_logits.softmax(dim=-1).topk(topk, dim=-1)
                cand_probs = topv / topv.sum(dim=1, keepdim=True).clamp_min(1e-8)
            k_now = cand_tokens.shape[1]
            cand_embed = self.spec_feat_head(self.la_token_embed(cand_tokens))  # [B, k, H]
            fork = self._expand_cache(motion_cache, k_now)
            prev_rep = prev_motion.repeat_interleave(k_now, dim=0)
            cand_audio_h = cand_embed.reshape(bsz * k_now, 1, -1)
            # Partner audio for t+1 is REAL (already available), not part of the
            # speculation -- advance its cache once (bsz, unforked) and reuse for
            # every candidate plus the eventual real step at the next iteration.
            partner_h_next, partner_cache = self._encode_partner_step(partner_audio_feat, t + 1, partner_cache)
            partner_cache = self._truncate_cache(partner_cache, cap)
            partner_h_next_rep = None if partner_h_next is None else partner_h_next.repeat_interleave(k_now, dim=0)
            # Role for t+1: same "advance once, real not speculative" treatment.
            if role_state is not None:
                role_next, role_state = self._role_at_step(audio_feat, partner_audio_feat, None, t + 1, role_state)
            role_next_rep = None if role_next is None else role_next.repeat_interleave(k_now, dim=0)
            if blink_state is None:
                cand_blink = None
            else:
                # Decide frame t+1's blink ONCE here (using the top candidate's
                # hypothesised audio_h -- the decision barely depends on audio)
                # so all candidates and the later exact recompute share the flag.
                logit = self._blink_logit(cand_embed[:, :1], prev_motion, self._pos_phase_at(t + 1, audio_feat.device),
                                          partner_h_t=partner_h_next)
                next_blink = self._sample_blink(logit[:, 0], blink_state, blink_temp, blink_refractory, blink_sample)
                cand_blink = next_blink.repeat_interleave(k_now, dim=0)
            cand_out, _ = self._decode_motion_step(cand_audio_h, prev_rep, t + 1, fork, blink_t=cand_blink,
                                                    partner_h_t=partner_h_next_rep, role_t=role_next_rep)
            cand_motions = cand_out["pred_motion"].view(bsz, k_now, -1)

        motion = torch.cat(preds, dim=1)
        stats = {
            "spec_conf": hit_mass / max(steps, 1.0),
            "recompute_rate": recompute_count / max(face_len * bsz, 1.0),
            "topk": float(topk),
        }
        return motion, stats

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

    def _mtp_targets(
        self,
        length: int,
        horizon: int,
        mimi_token: torch.Tensor,
        eos_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Targets + validity for MTP horizon ``horizon`` (in mimi frames).

        Face frame ``t`` lives in mimi frame ``t // 2``; the horizon-k target is
        ``mimi_token[t // 2 + k]``.  A position is invalid when the target falls
        past the token stream (or is the -1 pad), or when either the source face
        frame or the target's face frames lie in the EOS-augmented tail (the
        model sees ``eos_embed`` there, and the on-disk tokens continue with real
        speech the model must NOT be asked to predict).
        """
        n_mimi = mimi_token.shape[1]
        device = mimi_token.device
        tgt_idx = torch.arange(length, device=device) // 2 + horizon  # [T]
        in_range = tgt_idx < n_mimi
        tgt = mimi_token[:, tgt_idx.clamp(max=max(n_mimi - 1, 0))]  # [B, T]
        valid = in_range[None, :] & (tgt >= 0)
        if eos_mask is not None:
            src_real = ~eos_mask  # [B, T]
            tgt_face = (2 * tgt_idx).clamp(max=length - 1)
            valid = valid & src_real & ~eos_mask[:, tgt_face]
        return tgt.clamp(min=0), valid

    def mtp_loss(
        self,
        mtp_logits: torch.Tensor,
        mimi_token: torch.Tensor,
        eos_mask: torch.Tensor | None = None,
        horizon_gamma: float = 0.8,
    ) -> torch.Tensor:
        """Multi-token-prediction cross-entropy against REAL mimi semantic tokens.

        ``mtp_logits`` is [B, T, K, vocab]: face position ``t``, horizon ``k``
        predicts the mimi token of mimi frame ``t // 2 + k``.  Horizons are
        weighted by ``horizon_gamma ** (k-1)`` (normalised): near horizons matter
        most (horizon 1 is what speculative decode verifies), far horizons act as
        an auxiliary anticipation signal.
        """
        bsz, length, n_steps, vocab = mtp_logits.shape
        per_step = []
        for k in range(1, n_steps + 1):
            tgt, valid = self._mtp_targets(length, k, mimi_token, eos_mask)
            ce = F.cross_entropy(
                mtp_logits[:, :, k - 1, :].reshape(-1, vocab),
                tgt.reshape(-1),
                reduction="none",
            ).reshape(bsz, length)
            valid_f = valid.to(ce.dtype)
            per_step.append((ce * valid_f).sum() / valid_f.sum().clamp_min(1.0))
        weights = torch.tensor(
            [horizon_gamma ** k for k in range(n_steps)],
            device=mtp_logits.device,
            dtype=mtp_logits.dtype,
        )
        weights = weights / weights.sum().clamp_min(1e-8)
        return (torch.stack(per_step) * weights).sum()

    @torch.no_grad()
    def mtp_top1_acc(self, mtp_logits: torch.Tensor, mimi_token: torch.Tensor,
                     eos_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Top-1 accuracy of the horizon-1 head (verification hit-rate proxy)."""
        length = mtp_logits.shape[1]
        tgt, valid = self._mtp_targets(length, 1, mimi_token, eos_mask)
        pred = mtp_logits[:, :, 0, :].argmax(dim=-1)
        valid_f = valid.to(torch.float32)
        hit = (pred == tgt).to(torch.float32) * valid_f
        return hit.sum() / valid_f.sum().clamp_min(1.0)

    def spec_feat_loss(self, audio_h: torch.Tensor, mimi_token: torch.Tensor,
                       eos_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Grounding loss for speculative decode: ``spec_feat_head`` must map each
        frame's REAL token embedding to that frame's realised ``audio_h`` (target
        detached -- this trains the token->feature bridge and semantically anchors
        ``la_token_embed``, not the audio encoder)."""
        bsz, length, _ = audio_h.shape
        tgt, valid = self._mtp_targets(length, 0, mimi_token, eos_mask)  # this frame's token
        pred_h = self.spec_feat_head(self.la_token_embed(tgt))
        err = (pred_h - audio_h.detach()).pow(2).mean(dim=-1)  # [B, T]
        valid_f = valid.to(err.dtype)
        return (err * valid_f).sum() / valid_f.sum().clamp_min(1.0)


def shifted_motion(gt: torch.Tensor, model: CausalSoftVQContinuousTransformer) -> torch.Tensor:
    start = model.start_motion.expand(gt.shape[0], 1, -1).to(device=gt.device, dtype=gt.dtype)
    return torch.cat([start, gt[:, :-1]], dim=1)


def expand_vap_target_to_face_rate(vap_target: torch.Tensor, face_len: int) -> torch.Tensor:
    """Mimi-rate VAP labels [B,Tm,8] -> face-rate labels [B,Tf,8].

    The model uses the project-wide alignment ``face t <- mimi t//2``.  Repeat
    every 80-ms VAP label for its two 40-ms face frames, then explicitly pad
    invalid (-1) if an odd/end-truncated face sequence is ever encountered.
    """
    if vap_target.ndim != 3 or vap_target.shape[-1] != 8:
        raise ValueError(f"expected vap_target [B,T,8], got {tuple(vap_target.shape)}")
    vap_face = vap_target.repeat_interleave(2, dim=1)
    if vap_face.shape[1] < face_len:
        pad = vap_face.new_full((vap_face.shape[0], face_len - vap_face.shape[1], 8), -1)
        vap_face = torch.cat([vap_face, pad], dim=1)
    return vap_face[:, :face_len]


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
    # EOS augmentation: for a fraction of samples, cut the audio at a random frame
    # and HOLD the last pose for the rest, marking the tail with the EOS token.  This
    # teaches "audio ended -> settle/hold" so the model closes gracefully at turn-end
    # instead of hallucinating (inference auto-uses EOS past the provided audio).
    eos_mask = None
    if getattr(args, "eos_prob", 0.0) > 0.0:
        bsz, length = gt.shape[0], gt.shape[1]
        do = torch.rand(bsz, device=gt.device) < args.eos_prob
        cut = torch.randint(max(1, length // 4), length, (bsz,), device=gt.device)
        cut = torch.where(do, cut, torch.full_like(cut, length))  # length = no EOS
        idxs = torch.arange(length, device=gt.device)[None, :]
        eos_mask = idxs >= cut[:, None]  # [B, T] True past the cut
        held = gt[torch.arange(bsz, device=gt.device), (cut - 1).clamp(min=0)]  # [B, 54]
        gt = torch.where(eos_mask[..., None], held[:, None, :], gt)
    raw_model = model.module if isinstance(model, DDP) else model
    # Blink supervision: labels from the GT eyelid aperture (linear FLAME probe),
    # teacher-forced into the motion path so the decoder learns flag -> close.
    blink_cond = None
    if getattr(args, "blink_weight", 0.0) > 0.0:
        if not raw_model.blink_ready:
            raise RuntimeError(
                "--blink-weight > 0 needs the eyelid probe loaded "
                "(run prepare_eyelid_probe.py; main() loads assets/eyelid_probe.pt)"
            )
        blink_cond = raw_model.blink_labels(gt, thresh=args.blink_thresh)
    prev = shifted_motion(gt, raw_model)
    partner_audio_feat = batch.get("partner_audio_feat")
    partner_available = batch.get("partner_available")
    with torch.no_grad():
        z_target = codec_quant_to_sum_feat_chunked(codec, gt)
    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
        out = model(audio_feat, prev, llm_feat=llm_feat, eos_mask=eos_mask, blink_cond=blink_cond,
                    partner_audio_feat=partner_audio_feat, partner_available=partner_available)
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
        loss_blink = pred.new_zeros(())
        blink_rate = pred.new_zeros(())
        blink_recall = pred.new_zeros(())
        if blink_cond is not None:
            pos_weight = out["blink_logit"].new_tensor(args.blink_pos_weight)
            loss_blink = F.binary_cross_entropy_with_logits(
                out["blink_logit"].float(), blink_cond.float(), pos_weight=pos_weight
            )
            blink_rate = blink_cond.float().mean()
            pos = blink_cond.bool()
            if bool(pos.any()):
                blink_recall = ((out["blink_logit"] > 0) & pos).float().sum() / pos.float().sum()
        # Eyelid-aperture matching (mm): the eyelid dims are a tiny slice of the
        # 50-dim expr L1 and blink frames are ~3% of data, so flag->closure depth
        # trains too slowly (and the expr gate ~0.5 halves the candidate's closure
        # against the blink-blind prior).  The linear probe makes aperture
        # differentiable, so supervise the FINAL output's aperture directly,
        # upweighting closed frames -- gradients flow through gate/delta/residual.
        loss_blink_ap = pred.new_zeros(())
        if blink_cond is not None and getattr(args, "blink_ap_weight", 0.0) > 0.0:
            ap_pred = raw_model.eyelid_aperture(pred) * 1000.0
            ap_gt = raw_model.eyelid_aperture(gt) * 1000.0
            ap_w = 1.0 + args.blink_ap_boost * blink_cond.float()
            loss_blink_ap = (ap_w * (ap_pred - ap_gt).abs()).sum() / ap_w.sum()
        # VAP auxiliary look-ahead loss: TRAIN-TIME ONLY (see module docstring).
        # ``vap_target`` arrives at Mimi rate [B,Tm,8]; logits are face-rate
        # [B,Tf,8], so apply the same 1 Mimi -> 2 face alignment as audio_h.
        # A target is -1 wherever the record has no VAP sidecar or
        # the future window ran past the clip -- both cases masked out of the
        # BCE, exactly like the -1 padding convention used for mimi tokens.
        loss_vap = pred.new_zeros(())
        vap_frame_frac = pred.new_zeros(())
        if raw_model.use_vap and out["vap_logits"] is not None and "vap_target" in batch:
            vap_target = expand_vap_target_to_face_rate(
                batch["vap_target"], out["vap_logits"].shape[1]
            )
            vap_valid = (vap_target >= 0) & batch.get(
                "vap_available", torch.ones(vap_target.shape[0], dtype=torch.bool, device=vap_target.device)
            )[:, None, None]
            if bool(vap_valid.any()):
                loss_vap = F.binary_cross_entropy_with_logits(
                    out["vap_logits"][vap_valid].float(), vap_target[vap_valid].float()
                )
            vap_frame_frac = vap_valid.any(dim=-1).float().mean()
        loss_mtp = pred.new_zeros(())
        loss_spec_feat = pred.new_zeros(())
        mtp_acc = pred.new_zeros(())
        if raw_model.use_lookahead and out["mtp_logits"] is not None:
            if "audio_token" in batch:
                mimi_token = batch["audio_token"]
            else:
                raise KeyError(
                    "look-ahead training needs mimi semantic tokens: the dataset must be "
                    "built with RETURN_AUDIO_TOKENS (run prepare_mimi_semantic_tokens.py first)"
                )
            loss_mtp = raw_model.mtp_loss(
                out["mtp_logits"], mimi_token, eos_mask=eos_mask, horizon_gamma=args.mtp_horizon_gamma
            )
            loss_spec_feat = raw_model.spec_feat_loss(out["audio_h"], mimi_token, eos_mask=eos_mask)
            mtp_acc = raw_model.mtp_top1_acc(out["mtp_logits"], mimi_token, eos_mask=eos_mask)
        loss = (
            args.motion_weight * loss_motion
            + args.prior_weight * loss_prior
            + args.z_weight * loss_z
            + args.z_bce_weight * loss_z_bce
            + args.jaw_weight * loss_jaw
            + args.vel_weight * loss_vel
            + args.reg_weight * loss_reg
            + args.gate_weight * loss_gate
            + args.lookahead_weight * loss_mtp
            + args.spec_feat_weight * loss_spec_feat
            + args.blink_weight * loss_blink
            + getattr(args, "blink_ap_weight", 0.0) * loss_blink_ap
            + getattr(args, "vap_weight", 0.0) * loss_vap
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
        "mtp": float(loss_mtp.detach().item()),
        "spec_feat": float(loss_spec_feat.detach().item()),
        "mtp_acc": float(mtp_acc.detach().item()),
        "weighted_mtp": float((args.lookahead_weight * loss_mtp).detach().item()),
        "weighted_spec_feat": float((args.spec_feat_weight * loss_spec_feat).detach().item()),
        "blink": float(loss_blink.detach().item()),
        "blink_rate": float(blink_rate.detach().item()),
        "blink_recall": float(blink_recall.detach().item()),
        "weighted_blink": float((args.blink_weight * loss_blink).detach().item()),
        "blink_ap": float(loss_blink_ap.detach().item()),
        "weighted_blink_ap": float((getattr(args, "blink_ap_weight", 0.0) * loss_blink_ap).detach().item()),
        "vap": float(loss_vap.detach().item()),
        "vap_frame_frac": float(vap_frame_frac.detach().item()),
        "weighted_vap": float((getattr(args, "vap_weight", 0.0) * loss_vap).detach().item()),
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
        "teacher_mtp_acc": 0.0,
        "spec_hit_rate": 0.0,
        "spec_motion": 0.0,
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
        if raw_model.blink_ready:
            blink_cond = raw_model.blink_labels(gt, thresh=args.blink_thresh)
        else:
            blink_cond = None
        partner_audio_feat = batch.get("partner_audio_feat")
        partner_available = batch.get("partner_available")
        out = raw_model(audio_feat, prev, llm_feat=llm_feat, blink_cond=blink_cond,
                        partner_audio_feat=partner_audio_feat, partner_available=partner_available)
        teacher_pred = out["pred_motion"]
        # Autoregressive eval is capped to keep validation fast on long utterances.
        ar_frames = gt.shape[1] if args.ar_eval_frames <= 0 else min(gt.shape[1], args.ar_eval_frames)
        gt_ar = gt[:, :ar_frames]
        partner_ar = None if partner_audio_feat is None else partner_audio_feat[:, : (ar_frames + 1) // 2]
        ar_pred = raw_model.generate_cached(
            audio_feat,
            max_frames=ar_frames,
            llm_feat=llm_feat,
            max_context_frames=args.stream_context_frames,
            blink_temp=args.blink_temp,
            blink_refractory=args.blink_refractory,
            partner_audio_feat=partner_ar,
        )
        teacher_l1 = F.l1_loss(teacher_pred, gt)
        teacher_prior = F.l1_loss(out["prior_motion"], gt)
        teacher_z = F.mse_loss(out["z_pred"], z_target)
        z_target_bits = (z_target > 0).to(out["z_pred"].dtype)
        teacher_z_bce = F.binary_cross_entropy_with_logits(out["z_pred"] * math.sqrt(args.code_dim), z_target_bits)
        teacher_z_bit_acc = ((out["z_pred"] > 0) == (z_target > 0)).float().mean()
        teacher_jaw = F.l1_loss(teacher_pred[..., 50:51], gt[..., 50:51])
        ar_l1 = F.l1_loss(ar_pred, gt_ar)
        ar_jaw = F.l1_loss(ar_pred[..., 50:51], gt_ar[..., 50:51])
        ar_vel = F.mse_loss(ar_pred[:, 1:] - ar_pred[:, :-1], gt_ar[:, 1:] - gt_ar[:, :-1])
        teacher_mtp_acc = torch.zeros((), device=device)
        spec_hit_rate = torch.zeros((), device=device)
        spec_l1 = ar_l1
        if raw_model.use_lookahead and out["mtp_logits"] is not None and "audio_token" in batch:
            mimi_token = batch["audio_token"]
            teacher_mtp_acc = raw_model.mtp_top1_acc(out["mtp_logits"], mimi_token)
            if args.val_speculative:
                spec_pred, spec_stats = raw_model.generate_speculative_cached(
                    audio_feat,
                    mimi_token,
                    max_frames=ar_frames,
                    llm_feat=llm_feat,
                    max_context_frames=args.stream_context_frames,
                    topk=args.spec_topk,
                    blink_temp=args.blink_temp,
                    blink_refractory=args.blink_refractory,
                    partner_audio_feat=partner_ar,
                )
                spec_hit_rate = torch.tensor(spec_stats["spec_conf"], device=device)
                spec_l1 = F.l1_loss(spec_pred, gt_ar)
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
            total["teacher_mtp_acc"] += float(teacher_mtp_acc.item())
            total["spec_hit_rate"] += float(spec_hit_rate.item())
            total["spec_motion"] += float(spec_l1.item())
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
            "teacher_mtp_acc",
            "spec_hit_rate",
            "spec_motion",
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
    p.add_argument("--output-dir", default="/home6/duplex/personaplex/moshi/moshi/ARTalk/train_code/outputs/SoftVQ_v6_listening")
    p.add_argument("--resume", default="",
                   help="resume/fine-tune from a checkpoint .ckpt (loads model weights only, "
                        "strict=False so new params like eos_embed start fresh; no optimizer state "
                        "is restored). Continues epoch numbering from the checkpoint.")
    p.add_argument("--dualtalk-root", default="/home6/duplex/dataset/dualtalk")
    p.add_argument("--mimi-root", default="/home6/duplex/dataset/mimi_emb")
    p.add_argument("--ami-bc-root", default="/home6/duplex/dataset/ami_flame/bc")
    p.add_argument("--ami-bc-mimi-root", default="/home6/duplex/dataset/mimi_emb")
    p.add_argument("--ami-ut-root", default="/home6/duplex/dataset/ami_flame/ut")
    p.add_argument("--ami-ut-mimi-root", default="/home6/duplex/dataset/mimi_emb")
    p.add_argument("--vap-root", default="/home6/duplex/dataset/dualtalk_processed/vap",
                   help="VAP (Voice Activity Projection) label root; dualtalk only, "
                        "TRAIN-TIME auxiliary signal (never fed at inference)")
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
    # MTP predictive look-ahead over mimi semantic tokens (strictly causal).
    p.add_argument("--lookahead-frames", type=int, default=4,
                   help="number of future MIMI frames (12.5 Hz) the MTP module predicts; "
                        "1 mimi frame = 2 face frames = 80 ms (0 disables look-ahead)")
    p.add_argument("--lookahead-weight", type=float, default=0.5,
                   help="weight of the MTP cross-entropy loss")
    p.add_argument("--token-vocab", type=int, default=2048,
                   help="mimi codebook cardinality (semantic level; personaplex tokenizer = 2048)")
    p.add_argument("--mtp-horizon-gamma", type=float, default=0.8,
                   help="per-horizon MTP loss decay: horizon k is weighted gamma^(k-1) (normalised)")
    p.add_argument("--la-temp", type=float, default=0.5,
                   help="softmax temperature sharpening the predicted-token distribution before "
                        "its expected embedding is injected into the motion path")
    p.add_argument("--spec-feat-weight", type=float, default=0.1,
                   help="weight of the token->audio_h grounding loss used by speculative decode")
    p.add_argument("--chunk-frames", type=int, default=1,
                   help="block-causal audio context: each frame sees its chunk-frame block "
                        "of real future audio (1 = original strictly-causal model)")
    p.add_argument("--eos-prob", type=float, default=0.0,
                   help="fraction of training samples augmented with an end-of-audio (EOS) "
                        "tail: audio cut + last pose held, so the model settles at turn-end "
                        "(0 disables; inference auto-uses EOS past the provided audio)")
    # Stochastic blink event head (blinks are audio-independent -> deterministic
    # regression never initiates them in AR; see the model-class comment).
    p.add_argument("--blink-weight", type=float, default=0.0,
                   help="weight of the blink-event BCE loss; > 0 also teacher-forces the GT "
                        "blink flag into the motion path (0 = feature off, old behaviour)")
    p.add_argument("--blink-pos-weight", type=float, default=10.0,
                   help="BCE positive-class weight (closed frames are ~3-6%% of data)")
    p.add_argument("--blink-thresh", type=float, default=0.6,
                   help="a frame counts as blinking when eyelid aperture < thresh x clip median")
    p.add_argument("--blink-ap-weight", type=float, default=0.0,
                   help="weight of the differentiable eyelid-aperture L1 (mm, via the linear "
                        "probe) on the final output -- forces blink closure DEPTH, which the "
                        "expr L1 alone trains too slowly (0 = off)")
    p.add_argument("--blink-ap-boost", type=float, default=9.0,
                   help="extra weight on closed frames in the aperture L1 (closed frames get "
                        "1+boost, open frames 1)")
    p.add_argument("--blink-temp", type=float, default=1.0,
                   help="sampling temperature for the blink Bernoulli at inference")
    p.add_argument("--blink-refractory", type=int, default=12,
                   help="minimum frames between sampled blinks at inference (~0.5 s)")
    p.add_argument("--eyelid-probe",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "eyelid_probe.pt"),
                   help="linear FLAME eyelid probe from prepare_eyelid_probe.py (blink labels)")
    # v6: reactive listening (partner-audio stream, live at inference) + VAP
    # auxiliary look-ahead (train-time only -- see module docstring).
    p.add_argument("--partner-layers", type=int, default=2,
                   help="depth of the causal partner-audio encoder (smaller than the "
                        "own-audio encoder -- only needs coarse listening context)")
    p.add_argument("--use-vap", action=argparse.BooleanOptionalAction, default=True,
                   help="enable the VAP auxiliary look-ahead head architecture "
                        "(independent of --vap-weight, which controls whether it gets "
                        "gradient; keep enabled so --resume can fine-tune it in later)")
    p.add_argument("--vap-weight", type=float, default=0.05,
                   help="weight of the VAP auxiliary BCE loss (0 = feature off, no VAP "
                        "labels loaded, model falls back to the missing-embedding path "
                        "everywhere -- same as v4 behaviour)")
    p.add_argument("--spec-topk", type=int, default=4,
                   help="number of candidate tokens kept per frame in speculative decoding")
    p.add_argument("--ar-eval-frames", type=int, default=250,
                   help="cap autoregressive eval length during validation (0 = full sequence)")
    p.add_argument("--val-speculative", action=argparse.BooleanOptionalAction, default=True,
                   help="also run cached speculative decode during validation")
    p.add_argument("--val-batches", type=int, default=0)
    p.add_argument("--val-every-epochs", type=int, default=100)
    p.add_argument("--save-val-samples", type=int, default=0)
    p.add_argument("--save-every-epochs", type=int, default=100)
    p.add_argument("--stream-context-frames", type=int, default=25)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb-project", default="NIPS_duplex_SoftVQ_Continuous")
    p.add_argument("--wandb-run-name", default="softvq_v6_listening")
    args = p.parse_args()
    if not args.codec_ckpt:
        raise ValueError("--codec-ckpt is required")
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
        lookahead_desc = (
            f"mtp_lookahead={args.lookahead_frames}mimi_frames"
            f"(mimi_semantic_vocab{args.token_vocab},chained,xattn,causal)"
            if args.lookahead_frames > 0
            else "no_lookahead=True"
        )
        chunk_desc = (
            f"block_causal_audio(chunk={args.chunk_frames}f)" if args.chunk_frames > 1 else "strict_causal"
        )
        eos_desc = f"eos_aug={args.eos_prob}" if args.eos_prob > 0 else "no_eos"
        print(
            f"[softvq-continuous] model=causal_transformer {lookahead_desc} {chunk_desc} {eos_desc} "
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
    start_epoch = 1
    resume_ck = None
    if args.resume:
        resume_ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        # Restore architecture-defining args from the checkpoint so the rebuilt model
        # matches it exactly (prevents silent strict=False mismatches if the user
        # forgets e.g. --lookahead-frames). Training hyper-params stay from the CLI.
        ck_args = resume_ck.get("args", {})
        for k in ("hidden_dim", "layers", "heads", "code_dim", "prior_warmup_frames",
                  "lookahead_frames", "token_vocab", "la_temp", "chunk_frames",
                  "partner_layers", "use_vap"):
            if k in ck_args:
                setattr(args, k, ck_args[k])
    model = CausalSoftVQContinuousTransformer(
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        code_dim=args.code_dim,
        prior_warmup_frames=args.prior_warmup_frames,
        lookahead_frames=args.lookahead_frames,
        token_vocab=args.token_vocab,
        la_temp=args.la_temp,
        chunk_frames=args.chunk_frames,
        partner_layers=args.partner_layers,
        use_vap=args.use_vap,
    ).to(device)
    if resume_ck is not None:
        missing, unexpected = model.load_state_dict(resume_ck["model"], strict=False)
        start_epoch = int(resume_ck.get("epoch", 0)) + 1
        if is_main(rank):
            print(
                f"[softvq-continuous] resumed from {args.resume} @ epoch {resume_ck.get('epoch')} "
                f"(arch from ckpt: lookahead={args.lookahead_frames} vocab={args.token_vocab} chunk={args.chunk_frames}) "
                f"-> start_epoch={start_epoch} | missing={list(missing)} unexpected={list(unexpected)}"
            )
    if args.blink_weight > 0 and not model.blink_ready:
        if os.path.exists(args.eyelid_probe):
            model.load_eyelid_probe(args.eyelid_probe)
            if is_main(rank):
                print(f"[softvq-continuous] loaded eyelid probe {args.eyelid_probe} (blink head active)")
        else:
            raise FileNotFoundError(
                f"--blink-weight > 0 but eyelid probe not found at {args.eyelid_probe}; "
                "run: python prepare_eyelid_probe.py"
            )
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
    for epoch in range(start_epoch, args.epochs + 1):
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
