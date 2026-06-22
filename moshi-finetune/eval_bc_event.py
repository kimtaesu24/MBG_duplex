#!/usr/bin/env python3
"""Evaluate bc_event prediction performance of a trained model.

Loads a fine-tuned Personaplex checkpoint (LoRA or full), runs the eval data
through the model, collects bc_logits vs bc_timing_targets, and reports:

  * Focal BCE loss (same formulation as training)
  * Binary classification metrics at a configurable threshold:
      accuracy, precision, recall, F1, specificity
  * Threshold-independent: AUROC, AUPRC
  * Positive-rate statistics (label base-rate vs predicted rate)
  * Per-threshold PR/ROC curves (optional --curves flag)

Usage (single GPU):
    python eval_bc_event.py \\
        --checkpoint_dir output/<run>/checkpoints/checkpoint_XXXXXX \\
        [--config output/<run>/args.yaml]          # default: checkpoint_dir/../../args.yaml
        [--eval_data ./data/.../data.jsonl]         # default: from config
        [--batch_size 16]
        [--max_batches 200]
        [--threshold 0.5]
        [--device cuda:0]
        [--curves]                                  # print PR / ROC tables
        [--output results/bc_eval_XXXXXX.json]
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

# Ensure local moshi source takes precedence over installed packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../moshi")))

import torch._dynamo
torch._dynamo.config.suppress_errors = True
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
import safetensors

from finetune.args import TrainArgs
from finetune.data.data_loader import build_data_loader
from finetune.data.interleaver import InterleavedTokenizer, Interleaver
from moshi.models.loaders import (
    get_mimi,
    _lm_kwargs,
    MIMI_NAME,
    MOSHI_NAME,
    TEXT_TOKENIZER_NAME,
    DEFAULT_REPO,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("eval_bc_event")


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def pick_free_gpu() -> str:
    """Return the CUDA device string with the most free memory."""
    best_idx, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free, best_idx = free, i
    device = f"cuda:{best_idx}"
    logger.info(f"Auto-selected {device} ({best_free / 1024**3:.1f} GiB free)")
    return device


def patch_depformer_noop(model) -> None:
    """Replace forward_depformer_training with a zero-tensor stub.

    bc_logits are produced by the backchannel module from the main transformer
    hidden states — the depformer (audio codebook generation) runs *after* that
    and is not needed for bc_event evaluation.  Skipping it cuts ~50% of the
    per-batch compute.
    """
    import types

    def _noop_depformer(self, sequence, transformer_out, text_token_emb=None):
        B, _, T = sequence.shape
        # Return zeros of shape [B, dep_q, T, card] — never used in bc_event eval.
        return torch.zeros(
            B, self.dep_q, T, self.card,
            device=transformer_out.device, dtype=transformer_out.dtype,
        )

    # Traverse wrapper layers (PEFT / PeftCompatibleWrapper) to reach LMModel.
    lm = model
    for attr in ("base_model", "model"):
        if hasattr(lm, attr):
            lm = getattr(lm, attr)
    # Keep unwrapping until we find forward_depformer_training.
    while not hasattr(lm, "forward_depformer_training") and hasattr(lm, "model"):
        lm = lm.model

    if hasattr(lm, "forward_depformer_training"):
        lm.forward_depformer_training = types.MethodType(_noop_depformer, lm)
        logger.info("Depformer patched to no-op (bc_event eval only).")
    else:
        logger.warning("Could not locate forward_depformer_training — depformer NOT patched.")


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _focal_bce(logits: torch.Tensor, targets: torch.Tensor, gamma: float, pos_weight: float) -> float:
    """Focal BCE matching the training loss in train.py / eval.py."""
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype),
        reduction="none",
    )
    p_t = torch.exp(-bce.detach())
    focal = ((1.0 - p_t) ** gamma * bce).mean()
    return focal.item()


def compute_metrics(
    all_logits: np.ndarray,   # [N] raw logit for positive class
    all_targets: np.ndarray,  # [N] float 0/1
    threshold: float,
    gamma: float,
    pos_weight: float,
) -> dict:
    """Compute a comprehensive set of bc_event metrics from collected predictions."""
    N = len(all_logits)
    assert len(all_targets) == N

    logits_t = torch.from_numpy(all_logits).float()
    targets_t = torch.from_numpy(all_targets).float()

    probs = torch.sigmoid(logits_t).numpy()
    preds = (probs >= threshold).astype(np.int32)
    labels = all_targets.astype(np.int32)

    # ── Basic counts ──────────────────────────────────────────────────────
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy  = (tp + tn) / N
    specificity = tn / (tn + fp + 1e-9)

    pos_rate  = labels.mean()
    pred_rate = preds.mean()

    # ── AUROC (hand-rolled, no sklearn dependency required) ───────────────
    auroc = _compute_auroc(probs, labels)

    # ── AUPRC ─────────────────────────────────────────────────────────────
    auprc = _compute_auprc(probs, labels)

    # ── Loss ──────────────────────────────────────────────────────────────
    focal_loss = _focal_bce(logits_t, targets_t, gamma=gamma, pos_weight=pos_weight)
    plain_bce  = F.binary_cross_entropy_with_logits(logits_t, targets_t).item()

    return dict(
        n_samples=N,
        pos_rate=float(pos_rate),
        pred_pos_rate=float(pred_rate),
        threshold=threshold,
        # per-threshold
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        specificity=specificity,
        # threshold-independent
        auroc=auroc,
        auprc=auprc,
        # loss
        focal_bce_loss=focal_loss,
        plain_bce_loss=plain_bce,
    )


def _compute_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC without sklearn using trapezoidal rule on sorted scores."""
    sorted_idx = np.argsort(-probs)
    labels_sorted = labels[sorted_idx]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp_cumsum = np.cumsum(labels_sorted)
    fp_cumsum = np.cumsum(1 - labels_sorted)

    tpr = tp_cumsum / n_pos
    fpr = fp_cumsum / n_neg

    # Prepend origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    auroc = float(np.trapz(tpr, fpr))
    return auroc


def _compute_auprc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Area under precision-recall curve (average precision)."""
    sorted_idx = np.argsort(-probs)
    labels_sorted = labels[sorted_idx]

    n_pos = labels.sum()
    if n_pos == 0:
        return float("nan")

    tp_cumsum = np.cumsum(labels_sorted)
    fp_cumsum = np.cumsum(1 - labels_sorted)

    precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-12)
    recall_curve    = tp_cumsum / n_pos

    # Prepend (recall=0, precision=1)
    precision_curve = np.concatenate([[1.0], precision_curve])
    recall_curve    = np.concatenate([[0.0], recall_curve])

    auprc = float(np.trapz(precision_curve, recall_curve))
    return abs(auprc)  # trapz can be negative when recall decreases


def _pr_curve_table(probs: np.ndarray, labels: np.ndarray, n_points: int = 20) -> list[dict]:
    """Return a list of {threshold, precision, recall, f1} dicts at evenly-spaced thresholds."""
    rows = []
    for thr in np.linspace(0.0, 1.0, n_points + 1)[1:]:
        preds  = (probs >= thr).astype(np.int32)
        lbls   = labels.astype(np.int32)
        tp = int(((preds == 1) & (lbls == 1)).sum())
        fp = int(((preds == 1) & (lbls == 0)).sum())
        fn = int(((preds == 0) & (lbls == 1)).sum())
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f = 2 * p * r / (p + r + 1e-9)
        rows.append(dict(threshold=round(float(thr), 2), precision=round(p, 4),
                         recall=round(r, 4), f1=round(f, 4)))
    return rows


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_base_model(lm_config: dict, moshi_path: str, param_dtype: torch.dtype):
    """Instantiate LMModel and load the pretrained base weights."""
    from moshi.models.lm import LMModel
    with torch.device("cpu"):
        model = LMModel(device="cpu", dtype=param_dtype, **lm_config)
    with safetensors.safe_open(moshi_path, framework="pt", device="cpu") as f:
        base_sd = {k: f.get_tensor(k) for k in f.keys()}
    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    if missing:
        logger.warning(f"Base weights — {len(missing)} missing keys")
    if unexpected:
        logger.warning(f"Base weights — {len(unexpected)} unexpected keys")
    del base_sd
    return model


def load_model(
    checkpoint_dir: Path,
    args: TrainArgs,
    lm_config: dict,
    moshi_path: str,
    param_dtype: torch.dtype,
    device: str,
):
    """Load the model from a consolidated checkpoint directory.

    Supports both:
      * LoRA hybrid checkpoints  (lora.safetensors  — adapters + backchannel)
      * Full-finetune checkpoints (consolidated.safetensors)
    """
    lora_ckpt = checkpoint_dir / "consolidated" / "lora.safetensors"
    full_ckpt = checkpoint_dir / "consolidated" / "consolidated.safetensors"

    # ── Base model ──────────────────────────────────────────────────────
    model = _load_base_model(lm_config, moshi_path, param_dtype)

    if lora_ckpt.exists():
        # ── Hybrid LoRA + full-finetune (backchannel / face_module) ────
        logger.info(f"LoRA checkpoint detected: {lora_ckpt}")

        from peft import get_peft_model, LoraConfig, TaskType

        target_modules = (
            r"(?!.*(face_module|backchannel))"
            r".*(in_proj|out_proj|linear1|linear2|text_linear|input_proj|linear_in|linear_out)"
        )
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=args.lora.rank,
            lora_alpha=args.lora.scaling * args.lora.rank,
            target_modules=target_modules,
            bias="none",
        )

        class PeftCompatibleWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.model = m

            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.model, name)

            def forward(self, input_ids=None, *a, **kw):
                return self.model(input_ids, step=kw.get("step", 0),
                                  voice_prompt_embs=kw.get("voice_prompt_embs"),
                                  audio_feat=kw.get("audio_feat"),
                                  gt_face_motion=kw.get("gt_face_motion"),
                                  mimi=kw.get("mimi"),
                                  bc_audio_feats=kw.get("bc_audio_feats"))

        model = PeftCompatibleWrapper(model)
        model = get_peft_model(model, peft_config)
        model = model.to(param_dtype)

        with safetensors.safe_open(str(lora_ckpt), framework="pt", device="cpu") as f:
            adapter_sd = {k: f.get_tensor(k) for k in f.keys()}
        missing, unexpected = model.load_state_dict(adapter_sd, strict=False)
        n_loaded = len(adapter_sd) - len(unexpected)
        logger.info(f"Loaded {n_loaded}/{len(adapter_sd)} tensors from LoRA checkpoint "
                    f"({len(missing)} missing, {len(unexpected)} unexpected)")
        del adapter_sd

    elif full_ckpt.exists():
        # ── Full-finetune checkpoint ─────────────────────────────────────
        logger.info(f"Full checkpoint detected: {full_ckpt}")
        with safetensors.safe_open(str(full_ckpt), framework="pt", device="cpu") as f:
            full_sd = {k: f.get_tensor(k) for k in f.keys()}
        missing, unexpected = model.load_state_dict(full_sd, strict=False)
        logger.info(f"Loaded checkpoint ({len(missing)} missing, {len(unexpected)} unexpected)")
        del full_sd

    else:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir}/consolidated/  "
            f"(expected lora.safetensors or consolidated.safetensors)"
        )

    model = model.to(device).to(param_dtype)
    model.eval()
    logger.info("Model loaded and moved to eval mode.")
    return model


# ---------------------------------------------------------------------------
# Forward pass + collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model,
    data_loader,
    args: TrainArgs,
    mimi,
    prompt_ids: list,
    T_p: int,
    param_dtype: torch.dtype,
    device: str,
    max_batches: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over eval batches and collect bc_logits / bc_targets.

    Returns:
        logits_all: [N] float32 numpy array — raw logit for the positive class
        targets_all: [N] float32 numpy array — 0/1 binary labels
    """
    logits_list  = []
    targets_list = []
    n_batches    = 0

    for batch in data_loader:
        if n_batches >= max_batches:
            break

        codes = batch.codes.to(device)  # [B, K, T]

        # Text prompt prefix
        if T_p:
            prefix = torch.full(
                [codes.shape[0], codes.shape[1], T_p],
                model.zero_token_id, device=device, dtype=codes.dtype,
            )
            prefix[:, 0, :] = torch.tensor(prompt_ids, device=device)
            codes_in = torch.cat([prefix, codes], dim=2)
        else:
            codes_in = codes

        voice_prompt_embs = batch.voice_prompt_embs
        if voice_prompt_embs is not None:
            voice_prompt_embs = voice_prompt_embs.to(device)

        # VapGPT per-speaker audio features (same as training loop)
        bc_audio_feats = None
        if args.backchannel.enable and args.backchannel.module_type == "vap_gpt":
            _agent = mimi.decode_latent(codes[:, 1:9].clamp(min=0)).transpose(1, 2)
            _user  = mimi.decode_latent(codes[:, 9:17].clamp(min=0)).transpose(1, 2)
            _agent = _agent.to(device=device, dtype=param_dtype)
            _user  = _user.to(device=device, dtype=param_dtype)
            if T_p:
                _zero = torch.zeros(codes.shape[0], T_p, _agent.shape[2],
                                    device=device, dtype=param_dtype)
                _agent = torch.cat([_zero, _agent], dim=1)
                _user  = torch.cat([_zero, _user],  dim=1)
            bc_audio_feats = (_agent, _user)

        output = model(
            codes_in,
            step=0,
            voice_prompt_embs=voice_prompt_embs,
            audio_feat=None,
            gt_face_motion=None,
            mimi=None,
            bc_audio_feats=bc_audio_feats,
        )

        if output.bc_logits is None or batch.bc_timing_targets is None:
            n_batches += 1
            continue

        bc_logit_pos = output.bc_logits[:, T_p:, 1]  # [B, T] — positive logit
        bc_targets   = batch.bc_timing_targets.to(device)  # [B, T]

        valid_mask = bc_targets != -100
        if valid_mask.any():
            logits_list.append(bc_logit_pos[valid_mask].float().cpu().numpy())
            targets_list.append(bc_targets[valid_mask].float().cpu().numpy())

        n_batches += 1
        if n_batches % 10 == 0:
            n_collected = sum(len(x) for x in logits_list)
            logger.info(f"  batch {n_batches}/{max_batches} — {n_collected:,} frames collected")

    if not logits_list:
        raise RuntimeError("No valid bc_timing_targets found in eval data. "
                           "Check vap_manifest path in the config.")

    logits_all  = np.concatenate(logits_list,  axis=0)
    targets_all = np.concatenate(targets_list, axis=0)
    logger.info(f"Collected {len(logits_all):,} frames from {n_batches} batches.")
    return logits_all, targets_all


# ---------------------------------------------------------------------------
# Batch count helper
# ---------------------------------------------------------------------------

def _count_batches(eval_data_path: str, duration_sec: float, batch_size: int) -> int:
    """Count expected number of batches by reading the eval JSONL metadata."""
    total_segments = 0
    with open(eval_data_path) as f:
        for line in f:
            d = json.loads(line).get("duration", duration_sec)
            total_segments += math.ceil(d / duration_sec)
    return math.ceil(total_segments / batch_size)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate bc_event prediction performance")
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="Path to a checkpoint directory containing consolidated/")
    p.add_argument("--config", type=str, default=None,
                   help="Path to args.yaml. Defaults to checkpoint_dir/../../args.yaml")
    p.add_argument("--eval_data", type=str,
                   default="./data/stereo_ami_balanced_test/data_with_voice_sample.jsonl",
                   help="Eval data path (JSONL). Defaults to the AMI test split.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_batches", type=int, default=0,
                   help="Max number of eval batches (set 0 for unlimited)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold for binary classification metrics")
    p.add_argument("--device", type=str, default=None,
                   help="CUDA device (e.g. cuda:1). Defaults to GPU with most free memory.")
    p.add_argument("--curves", action="store_true",
                   help="Print PR curve table at 20 thresholds")
    p.add_argument("--output", type=str, default=None,
                   help="Optional path to save JSON results")
    return p.parse_args()


def main():
    cli = parse_args()

    checkpoint_dir = Path(cli.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    # ── Config ──────────────────────────────────────────────────────────
    config_path = cli.config or str(checkpoint_dir / ".." / ".." / "args.yaml")
    config_path = str(Path(config_path).resolve())
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading config: {config_path}")
    args: TrainArgs = TrainArgs.load(config_path, drop_extra_fields=False)

    if not args.backchannel.enable:
        raise ValueError("backchannel.enable is False in config — no bc_event to evaluate.")
    if args.backchannel.bc_event_loss_weight <= 0:
        logger.warning("bc_event_loss_weight <= 0 in config. "
                       "The model may not have been trained with this loss.")

    if cli.eval_data:
        args.data.eval_data = cli.eval_data

    device = cli.device or pick_free_gpu()
    torch.cuda.set_device(device)
    param_dtype = torch.bfloat16 if args.param_dtype == "bfloat16" else torch.float32

    # Compute exact batch count from the eval JSONL so the loop is always finite.
    if cli.max_batches > 0:
        max_batches = cli.max_batches
    else:
        max_batches = _count_batches(args.data.eval_data, args.duration_sec, cli.batch_size)
        logger.info(f"Eval JSONL → {max_batches} batches (batch_size={cli.batch_size})")

    # ── Model paths ─────────────────────────────────────────────────────
    hf_repo = args.moshi_paths.hf_repo_id or DEFAULT_REPO
    mimi_path    = args.moshi_paths.mimi_path
    moshi_path   = args.moshi_paths.moshi_path
    tokenizer_path = args.moshi_paths.tokenizer_path

    if mimi_path is None or moshi_path is None or tokenizer_path is None:
        logger.info(f"Downloading missing model files from HuggingFace: {hf_repo}")
        from huggingface_hub import hf_hub_download
        if mimi_path is None:
            mimi_path = hf_hub_download(hf_repo, MIMI_NAME)
        if moshi_path is None:
            moshi_path = hf_hub_download(hf_repo, MOSHI_NAME)
        if tokenizer_path is None:
            tokenizer_path = hf_hub_download(hf_repo, TEXT_TOKENIZER_NAME)

    # ── Mimi ────────────────────────────────────────────────────────────
    logger.info("Loading Mimi tokenizer...")
    mimi = get_mimi(mimi_path, device=device)
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    # ── LM config ───────────────────────────────────────────────────────
    lm_config = dict(_lm_kwargs)
    if args.moshi_paths.config_path is not None:
        with open(args.moshi_paths.config_path) as f:
            lm_config = json.load(f)
    lm_config["dep_q"] = 16

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
            lm_config["backchannel_module_type"] = "mlp"
        elif args.backchannel.module_type == "vap_gpt":
            lm_config["backchannel_module_type"] = "vap_gpt"
            lm_config["backchannel_vap_repo_path"] = args.backchannel.vap_gpt_repo_path
            lm_config["backchannel_vap_checkpoint"] = args.backchannel.vap_gpt_checkpoint
            lm_config["backchannel_vap_channel_layers"] = args.backchannel.vap_gpt_channel_layers
            lm_config["backchannel_vap_cross_layers"] = args.backchannel.vap_gpt_cross_layers
            lm_config["backchannel_vap_num_heads"] = args.backchannel.vap_gpt_num_heads
            lm_config["backchannel_vap_dropout"] = args.backchannel.vap_gpt_dropout

    # Face module is not needed for bc_event evaluation; disable to save memory.
    lm_config["face_module_enabled"] = False

    # ── Load model ──────────────────────────────────────────────────────
    logger.info(f"Loading model from: {checkpoint_dir}")
    model = load_model(checkpoint_dir, args, lm_config, moshi_path, param_dtype, device)
    patch_depformer_noop(model)

    # ── Text prompt ─────────────────────────────────────────────────────
    import sentencepiece
    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(tokenizer_path)
    prompt_ids: list[int] = spm.encode(args.text_prompt) if args.text_prompt else []
    T_p: int = len(prompt_ids)
    if T_p:
        logger.info(f"Text prompt: {T_p} tokens")

    # ── Data loader ─────────────────────────────────────────────────────
    interleaver = Interleaver(
        spm,
        mimi.frame_rate,
        model.text_padding_token_id,
        model.end_of_text_padding_id,
        model.zero_token_id,
        keep_main_only=True,
    )
    interleaved_tokenizer = InterleavedTokenizer(
        mimi, interleaver,
        duration_sec=args.duration_sec,
        vap_manifest_path=args.data.vap_manifest,
        flame_root="",  # face gen not needed
        flame_speaker=args.face_gen.flame_speaker,
    )

    logger.info(f"Building eval data loader (eval_data={args.data.eval_data})...")
    eval_loader = build_data_loader(
        instruct_tokenizer=interleaved_tokenizer,
        args=args.data,
        batch_size=cli.batch_size,
        seed=0,
        rank=0,
        world_size=1,
        is_eval=True,
    )

    # ── Collect predictions ─────────────────────────────────────────────
    limit_str = str(cli.max_batches) if cli.max_batches > 0 else "unlimited"
    logger.info(f"Running inference (batches={limit_str}, batch_size={cli.batch_size})...")
    logits_all, targets_all = collect_predictions(
        model=model,
        data_loader=eval_loader,
        args=args,
        mimi=mimi,
        prompt_ids=prompt_ids,
        T_p=T_p,
        param_dtype=param_dtype,
        device=device,
        max_batches=max_batches,
    )

    # ── Metrics ─────────────────────────────────────────────────────────
    logger.info("Computing metrics...")
    metrics = compute_metrics(
        logits_all, targets_all,
        threshold=cli.threshold,
        gamma=args.backchannel.bc_focal_gamma,
        pos_weight=args.backchannel.bc_focal_pos_weight,
    )

    # ── Report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  BC-Event Prediction Evaluation")
    print(f"  checkpoint : {checkpoint_dir}")
    print(f"  eval_data  : {args.data.eval_data}")
    print("=" * 62)
    print(f"  Frames evaluated      : {metrics['n_samples']:>12,}")
    print(f"  True positive rate    : {metrics['pos_rate']:>12.4f}  ({metrics['pos_rate']*100:.2f}%)")
    print(f"  Predicted pos rate    : {metrics['pred_pos_rate']:>12.4f}  (thr={cli.threshold})")
    print("-" * 62)
    print("  [ Loss ]")
    print(f"  Focal BCE loss        : {metrics['focal_bce_loss']:>12.4f}")
    print(f"  Plain BCE loss        : {metrics['plain_bce_loss']:>12.4f}")
    print("-" * 62)
    print(f"  [ Classification @ threshold = {cli.threshold} ]")
    print(f"  TP / FP / FN / TN     : {metrics['tp']:>6} / {metrics['fp']:>6} / {metrics['fn']:>6} / {metrics['tn']:>6}")
    print(f"  Precision             : {metrics['precision']:>12.4f}")
    print(f"  Recall                : {metrics['recall']:>12.4f}")
    print(f"  F1 score              : {metrics['f1']:>12.4f}")
    print(f"  Accuracy              : {metrics['accuracy']:>12.4f}")
    print(f"  Specificity           : {metrics['specificity']:>12.4f}")
    print("-" * 62)
    print("  [ Threshold-independent ]")
    print(f"  AUROC                 : {metrics['auroc']:>12.4f}")
    print(f"  AUPRC                 : {metrics['auprc']:>12.4f}")
    print("=" * 62 + "\n")

    if cli.curves:
        probs = 1.0 / (1.0 + np.exp(-logits_all))
        table = _pr_curve_table(probs, targets_all, n_points=20)
        print("  Precision-Recall curve (20 thresholds):")
        print(f"  {'thr':>6}  {'P':>7}  {'R':>7}  {'F1':>7}")
        print("  " + "-" * 32)
        for row in table:
            print(f"  {row['threshold']:>6.2f}  {row['precision']:>7.4f}  "
                  f"{row['recall']:>7.4f}  {row['f1']:>7.4f}")
        print()

    # ── Save JSON ───────────────────────────────────────────────────────
    results = {
        "checkpoint_dir": str(checkpoint_dir),
        "eval_data": args.data.eval_data,
        "batch_size": cli.batch_size,
        "max_batches": cli.max_batches,
        "threshold": cli.threshold,
        "focal_gamma": args.backchannel.bc_focal_gamma,
        "focal_pos_weight": args.backchannel.bc_focal_pos_weight,
        "metrics": metrics,
    }
    if cli.curves:
        probs = 1.0 / (1.0 + np.exp(-logits_all))
        results["pr_curve"] = _pr_curve_table(probs, targets_all, n_points=20)

    if cli.output:
        out_path = Path(cli.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
