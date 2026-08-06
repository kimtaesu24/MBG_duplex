"""
Offline inference script over a test dataset (e.g. data.jsonl).
Loads the finetuned Moshi checkpoint and iteratively runs inference, saving the decoded audio and text tokens.
"""

import argparse
import os
import random
import sys

# Ensure local moshi source takes precedence over installed site-packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../moshi")))

import json
import tarfile
from pathlib import Path
from typing import Optional, List
import yaml

import numpy as np
import torch
import sentencepiece
import torch.nn.functional as F
import sphn
import safetensors.torch
from huggingface_hub import hf_hub_download

# Local imports from moshi and finetune
import moshi.models.loaders as loaders
from face_generation import FaceGenerator
from moshi.models import MimiModel
from moshi.client_utils import make_log
# v2: lm2 (vap_gpt_module-backed LMModel + LMGen with 3-class PAD/EPAD/WORD decision)
import moshi.models.lm as lm2
from moshi.models.lm import LMGen
from moshi.models.lm import load_audio as lm_load_audio
from moshi.models.lm import _iterate_audio as lm_iterate_audio
from moshi.models.lm import encode_from_sphn as lm_encode_from_sphn
from moshi.offline import warmup, decode_tokens_to_pcm

# loaders.get_moshi_lm instantiates the module-global `LMModel` — swap in the v2
# class so the pretrained/finetuned checkpoint loads into lm2.LMModel instead.
loaders.LMModel = lm2.LMModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def log(level: str, msg: str):
    print(make_log(level, msg))


def _wrap_with_system_tags(text: str) -> str:
    """Match the original PersonaPlex offline prompt normalization."""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def _repair_config_paths(config: dict) -> None:
    """Replace paths from the training server with this server's known paths."""
    finetune_dir = Path(__file__).resolve().parent
    conversational_ai_dir = finetune_dir.parent.parent
    replacements = {
        ("backchannel", "vap_gpt_repo_path"):
            str(conversational_ai_dir / "VoiceActivityProjection"),
        ("backchannel", "vap_gpt_checkpoint"):
            str(conversational_ai_dir / "VoiceActivityProjection" / "example"
                / "VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt"),
        ("face_gen", "ckpt_path"):
            str(finetune_dir / "checkpoint_epoch_125_v7.ckpt"),
        ("face_gen", "codec_ckpt_path"):
            str(finetune_dir / "mimi_codec_50000.pt"),
        ("face_gen", "codec_stats_path"):
            str(finetune_dir / "mimi_stats.json"),
        ("face_gen", "flame_root"):
            "/home6/duplex/dataset/ami_flame",
        ("data", "vap_manifest"):
            str(conversational_ai_dir / "vap_dataset" / "manifest.json"),
        # None makes resolve_weight() use the configured Hugging Face repository.
        ("moshi_paths", "mimi_path"): None,
        ("moshi_paths", "moshi_path"): None,
        ("moshi_paths", "tokenizer_path"): None,
    }

    for (section, key), new_path in replacements.items():
        section_cfg = config.get(section)
        if not isinstance(section_cfg, dict) or key not in section_cfg:
            continue
        old_path = section_cfg.get(key)
        if old_path is None or old_path == "":
            continue
        if isinstance(old_path, str) and os.path.exists(
            os.path.expandvars(os.path.expanduser(old_path))
        ):
            continue
        if old_path != new_path:
            section_cfg[key] = new_path
            log("warning", f"Replaced {section}.{key}: {old_path} -> {new_path}")


def list_jsonl(jsonl_path: str):
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            samples.append(json.loads(line))
    return samples

def _label_int_to_bits(label_int, num_bits=8):
    """Converts an integer label to its binary bit representation."""
    bits = []
    for i in range(num_bits):
        bits.append((label_int >> (num_bits - 1 - i)) & 1)
    return bits


def _select_user_audio(audio: np.ndarray, audio_path: str) -> np.ndarray:
    """Select the same user channel for inference and saved comparison audio."""
    if audio.ndim != 2:
        return audio[np.newaxis]
    if audio.shape[0] != 2:
        return audio[0:1]

    is_ami_stereo = any(
        part == "stereo_ami_balanced" or part.startswith("stereo_ami_balanced_")
        for part in Path(audio_path).parts
    )
    if is_ami_stereo:
        # This AMI test set is intentionally oriented with the user on ch1.
        log("info", f"AMI channel order: using original ch1 as user input for {audio_path}.")
        return audio[1:2]

    # DualTalk convention: ch0=target agent, ch1=user.
    log("info", f"Using original ch1 as user input for {audio_path}.")
    return audio[1:2]


def _strip_peft_prefixes(state_dict: dict) -> dict:
    """Strip training wrapper prefixes so keys match the bare LMModel namespace.

    Training wraps: PeftModel(PeftCompatibleWrapper(LMModel))
    Saved keys:     base_model.model.model.<lm_key>
    After strip:    <lm_key>

    .original_module.* entries are frozen originals kept by PEFT — skip them.
    """
    cleaned = {}
    for k, v in state_dict.items():
        if ".original_module." in k:
            continue
        if k.startswith("base_model.model.model."):
            k = k[len("base_model.model.model."):]
        elif k.startswith("base_model.model."):
            k = k[len("base_model.model."):]
        cleaned[k] = v
    return cleaned


def _is_depformer_weight(key: str) -> bool:
    """Return whether a state-dict key belongs to the complete Depformer path.

    ``linears`` are the per-codebook output heads used exclusively by the
    Depformer, so they must be restored together with the transformer and its
    input embeddings/projections.
    """
    root = key.split(".", 1)[0]
    return root == "depformer" or root.startswith("depformer_") or root == "linears"


def load_checkpoint(
    lm,
    ckpt_dir,
    lora_config: Optional[dict] = None,
    use_pretrained_depformer: bool = False,
):
    """Loads the finetuned checkpoint (consolidated or lora-only)."""
    consolidated_path = os.path.join(ckpt_dir, "consolidated", "consolidated.safetensors")
    lora_path = os.path.join(ckpt_dir, "consolidated", "lora.safetensors")

    if os.path.exists(consolidated_path):
        log("info", f"Loading full checkpoint from {consolidated_path}")
        state_dict = safetensors.torch.load_file(consolidated_path)
        state_dict = _strip_peft_prefixes(state_dict)
        if use_pretrained_depformer:
            depformer_keys = {k for k in state_dict if _is_depformer_weight(k)}
            state_dict = {k: v for k, v in state_dict.items() if k not in depformer_keys}
            log(
                "info",
                f"Keeping pretrained PersonaPlex Depformer weights "
                f"({len(depformer_keys)} checkpoint tensors skipped).",
            )
        missing, unexpected = lm.load_state_dict(state_dict, strict=False)
        bc_dropped = [k for k in unexpected if "backchannel" in k]
        if bc_dropped and getattr(lm, "backchannel", None) is None:
            raise RuntimeError(
                f"Checkpoint contains {len(bc_dropped)} backchannel tensors, but the "
                "backchannel module is disabled by the config."
            )
        bc_missing = [k for k in missing if "backchannel" in k]
        face_missing = [k for k in missing if "face_module" in k]
        if bc_missing:
            raise RuntimeError(f"Backchannel checkpoint keys missing: {bc_missing}")
        if face_missing:
            raise RuntimeError(f"Face checkpoint keys missing: {face_missing[:3]}")
        if not bc_missing and not face_missing:
            log("info", "Checkpoint loaded successfully.")

    elif os.path.exists(lora_path):
        log("info", f"Loading LoRA checkpoint from {lora_path}")
        state_dict = safetensors.torch.load_file(lora_path)
        state_dict = _strip_peft_prefixes(state_dict)
        if use_pretrained_depformer:
            depformer_keys = {k for k in state_dict if _is_depformer_weight(k)}
            state_dict = {k: v for k, v in state_dict.items() if k not in depformer_keys}
            log(
                "info",
                f"Keeping pretrained PersonaPlex Depformer weights "
                f"({len(depformer_keys)} checkpoint tensors skipped).",
            )

        # 1. Load backchannel + face_module weights directly into LMModel.
        #    These modules are full-finetuned (no LoRA) and stored flat in lora.safetensors.
        def _is_direct_module_key(k: str) -> bool:
            segments = k.split(".")
            return any(seg in ("backchannel", "face_module") or seg.startswith("backchannel") or seg.startswith("face_module") for seg in segments)

        direct_weights = {k: v for k, v in state_dict.items() if _is_direct_module_key(k)}
        has_bc_weights = any("backchannel" in k.split(".") for k in direct_weights)
        has_face_weights = any("face_module" in k.split(".") for k in direct_weights)
        if getattr(lm, "backchannel", None) is not None and not has_bc_weights:
            raise RuntimeError("Backchannel is enabled, but the checkpoint has no backchannel weights.")
        if getattr(lm, "face_module", None) is not None and not has_face_weights:
            raise RuntimeError("Face module is enabled, but the checkpoint has no face weights.")
        if direct_weights:
            missing, unexpected = lm.load_state_dict(direct_weights, strict=False)
            bc_dropped = [k for k in unexpected if "backchannel" in k]
            if bc_dropped and getattr(lm, "backchannel", None) is None:
                raise RuntimeError(
                    f"Checkpoint contains {len(bc_dropped)} backchannel tensors, but the "
                    "backchannel module is disabled by the config."
                )
            bc_missing = [k for k in missing if "backchannel" in k]
            face_missing = [k for k in missing if "face_module" in k]
            if bc_missing:
                raise RuntimeError(f"Backchannel checkpoint keys missing: {bc_missing}")
            if face_missing:
                raise RuntimeError(f"Face checkpoint keys missing: {face_missing[:3]}")
            if not bc_missing and not face_missing:
                log("info", f"Direct weights loaded ({len(direct_weights)} tensors).")

        # 2. Load LoRA adapter weights via PEFT then merge into base weights.
        #    Training used PeftCompatibleWrapper so saved keys are at depth 3:
        #      base_model.model.model.<lm_key>  →  stripped: <lm_key>
        #    Inference PEFT model (bare LMModel) has depth 2:
        #      base_model.model.<lm_key>
        #    → re-add one "base_model.model." prefix before loading.
        #
        #    target_modules must match training (wrapped_model.py).
        lora_weights = {k: v for k, v in state_dict.items() if not _is_direct_module_key(k)}
        if lora_weights:
            try:
                from peft import get_peft_model, LoraConfig, TaskType
                # Exclude depformer if the checkpoint has no depformer LoRA keys
                # (happens when freeze_depformer=true was used during training).
                extra_exclude = "|depformer" if not any("depformer" in k for k in lora_weights) else ""
                lora_cfg = lora_config or {}
                lora_rank = int(lora_cfg.get("rank", 64))
                lora_scaling = float(lora_cfg.get("scaling", 2.0))
                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_rank,
                    lora_alpha=lora_scaling * lora_rank,
                    target_modules=(
                        rf"(?!.*(face_module|backchannel{extra_exclude}))"
                        r".*(in_proj|out_proj|linear1|linear2|text_linear|input_proj|linear_in|linear_out)"
                    ),
                    bias="none",
                )
                log("info", f"Constructing LoRA adapter (rank={lora_rank}, scaling={lora_scaling}).")
                peft_lm = get_peft_model(lm, lora_config)
                prefixed = {f"base_model.model.{k}": v for k, v in lora_weights.items()}
                missing_lora, _ = peft_lm.load_state_dict(prefixed, strict=False)
                missing_lora = [k for k in missing_lora if "lora_" in k]
                if missing_lora:
                    raise RuntimeError(
                        f"LoRA checkpoint is incompatible; missing keys: {missing_lora[:3]}"
                    )
                else:
                    log("info", f"LoRA weights loaded ({len(lora_weights)} tensors).")
                # Merge LoRA deltas into base weights and restore a plain LMModel.
                lm = peft_lm.merge_and_unload()
            except ImportError as exc:
                raise RuntimeError(
                    "LoRA weights were found, but PEFT is not installed."
                ) from exc
    else:
        raise FileNotFoundError(
            f"No consolidated.safetensors or lora.safetensors found in {ckpt_dir}"
        )
    return lm


def infer_one(
    inp_path: str,
    mimi,
    other_mimi,
    lm_gen,
    frame_size: int,
    voice_prompt_path: str = "",
    face_gen=None,
    text_tokenizer=None,
    collect_bc_log: bool = False,
    original_personaplex: bool = False,
):
    """Run offline inference for a single input WAV.

    Returns (output_pcm, sample_rate, generated_text_tokens, bc_gate_log, has_audio).
    generated_text_tokens and bc_gate_log are empty lists when collect_bc_log=False.
    If face_gen is provided, add_step() is called per frame; the caller is responsible
    for invoking face_gen.generate_numpy() afterwards.
    """
    mimi.reset_streaming()
    other_mimi.reset_streaming()
    lm_gen.reset_streaming()

    if voice_prompt_path:
        lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
    else:
        # Explicitly clear any voice prompt from a previous sample so an empty
        # path truly means "no voice prompt" (training-consistent for datasets
        # without voice_sample entries).
        lm_gen.voice_prompt = None
        lm_gen.voice_prompt_audio = None
        lm_gen.voice_prompt_embeddings = None
        lm_gen.voice_prompt_cache = None

    if face_gen is not None:
        face_gen.reset()

    lm_gen.step_system_prompts(mimi)
    mimi.reset_streaming()

    user_audio = _select_user_audio(
        lm_load_audio(inp_path, mimi.sample_rate),
        inp_path,
    )

    target_samples = user_audio.shape[-1]
    generated_frames = []
    generated_text_tokens = []
    bc_gate_log = []
    special_token_map = {0: "EPAD", 1: "BOS", 2: "EOS", 3: "PAD"}

    for user_encoded in lm_encode_from_sphn(
        mimi,
        lm_iterate_audio(user_audio, sample_interval_size=frame_size, pad=True),
        max_batch=1,
    ):
        steps = user_encoded.shape[-1]
        for c in range(steps):
            if original_personaplex:
                # Original offline.py calls LMGen.step() without the lm2-only
                # return_z extension.
                tokens = lm_gen.step(user_encoded[:, :, c : c + 1])
                z = None
                if tokens is None:
                    continue
            else:
                result = lm_gen.step(user_encoded[:, :, c : c + 1], return_z=True)
                if result is None or result[0] is None:
                    continue
                tokens, z = result
            # LMGen computes the controller for the same token before returning it.
            # Read it immediately; delaying this lookup by one loop iteration shifts
            # every gate/fusion record one frame away from the token it controlled.
            _current_bc_result = (
                getattr(lm_gen.lm_model, "_last_bc_result", None)
                if getattr(lm_gen.lm_model, "backchannel", None) is not None
                else None
            )

            if face_gen is not None:
                face_gen.add_step(tokens, z)

            _PAD  = lm_gen.lm_model.text_padding_token_id
            _EPAD = lm_gen.lm_model.end_of_text_padding_id

            # Note: [EPAD] → [PAD] suppression (when the backchannel gate says "don't speak")
            # is handled inside LMGen.process_transformer_output, controlled by the
            # suppress_epad flag passed at LMGen construction. It must run there so the
            # substitution is written to the model's autoregressive cache; doing it here on
            # the returned tokens would only affect decoding/logging, not generation.

            generated_frames.append(decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens))

            # Text-token collection is independent of backchannel logging: collect
            # whenever a tokenizer is available (e.g. backbone-only runs), so the JSON
            # "text_tokens" is populated even when backchannel is disabled.
            if text_tokenizer is not None:
                text_id = tokens[0, 0, 0].item()
                if text_id in special_token_map:
                    generated_text_tokens.append(special_token_map[text_id])
                else:
                    generated_text_tokens.append(
                        text_tokenizer.id_to_piece(text_id).replace("▁", " ")
                    )
                _label = "PAD" if text_id == _PAD else ("EPAD" if text_id == _EPAD else f"WORD({text_id})")
                # v2 bc log: 3-class head (0=PAD, 1=EPAD, 2=WORD) + current-frame VAD.
                if collect_bc_log and _current_bc_result is not None:
                    _probs = _current_bc_result.bc_logits[0, 0].float().softmax(-1)  # [3]
                    _p_pad, _p_epad, _p_word = (_probs[0].item(), _probs[1].item(), _probs[2].item())
                    _pred_cls = int(_probs.argmax().item())
                    _pred_label = ("PAD", "EPAD", "WORD")[_pred_cls]
                    # Actual gate decision from LMGen (threshold rule may differ from argmax).
                    _gate = (int(_current_bc_result.gate[0, 0].item())
                             if _current_bc_result.gate is not None else int(_pred_cls == 1))
                    _vad = torch.sigmoid(_current_bc_result.vad_logits[0, 0].float())  # [2] (user, agent)
                    _vad_user, _vad_agent = _vad[0].item(), _vad[1].item()
                    _implicit_lo = _current_bc_result.implicit_log_odds[0, 0].item()
                    _explicit_lo = _current_bc_result.explicit_log_odds[0, 0].item()
                    _vap_near = _current_bc_result.vap_agent_near[0, 0].item()
                    _fusion_p = _current_bc_result.fusion_prob[0, 0].item()
                    print(f"token={_label:14s} | gate={_gate} argmax={_pred_label:4s} "
                          f"p_pad={_p_pad:.3f} p_epad={_p_epad:.3f} p_word={_p_word:.3f} "
                          f"| implicit_lo={_implicit_lo:.3f} explicit_lo={_explicit_lo:.3f} "
                          f"fusion_p={_fusion_p:.3f} vap_near={_vap_near:.3f} "
                          f"vad_user={_vad_user:.3f} vad_agent={_vad_agent:.3f}")
                    bc_gate_log.append({
                        "token_id": text_id,
                        "token_label": _label,
                        "pred_cls": _pred_cls,
                        "pred_label": _pred_label,
                        "p_pad": round(_p_pad, 4),
                        "p_epad": round(_p_epad, 4),
                        "p_word": round(_p_word, 4),
                        "gate": _gate,
                        "vad_user": round(_vad_user, 4),
                        "vad_agent": round(_vad_agent, 4),
                        "implicit_log_odds": round(_implicit_lo, 4),
                        "explicit_log_odds": round(_explicit_lo, 4),
                        "vap_agent_near": round(_vap_near, 4),
                        "fusion_prob": round(_fusion_p, 4),
                    })

    if not generated_frames:
        return np.zeros(target_samples, dtype=np.float32), mimi.sample_rate, [], [], False

    output_pcm = np.concatenate(generated_frames, axis=-1)
    if output_pcm.shape[-1] > target_samples:
        output_pcm = output_pcm[:target_samples]
    elif output_pcm.shape[-1] < target_samples:
        output_pcm = np.concatenate([
            output_pcm,
            np.zeros(target_samples - output_pcm.shape[-1], dtype=output_pcm.dtype),
        ])

    return output_pcm, mimi.sample_rate, generated_text_tokens, bc_gate_log, True


def run_test_inference(args):
    # Base-model mode keeps the lm2 classes but follows the original PersonaPlex
    # sampling/step path. Prompt formatting remains independently selectable.
    original_personaplex = args.base_model_only
    seed = args.seed
    if seed is None:
        seed = -1 if original_personaplex else 42
    if seed != -1:
        set_seed(seed)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        log("warning", "CUDA requested but not available — falling back to CPU.")
        device = "cpu"
    # LM dtype: bfloat16 on GPU (checkpoint native), float32 on CPU (bf16 CPU
    # kernels are slow/incomplete). Override with --dtype if needed.
    if args.dtype == "auto":
        lm_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    else:
        lm_dtype = getattr(torch, args.dtype)
    log("info", f"device={device}, lm dtype={lm_dtype}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    log("info", f"Reading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if not args.no_auto_fix_config_paths:
        _repair_config_paths(config)
    if original_personaplex:
        config.setdefault("backchannel", {})["enable"] = False
        config.setdefault("face_gen", {})["enable"] = False
        args.ckpt_dir = None
        log(
            "info",
            "Pretrained-only mode: finetuned checkpoint, backchannel, and face module disabled.",
        )
    if original_personaplex:
        log("info", "Original PersonaPlex compatibility branch enabled (using lm2 classes).")
    elif args.ckpt_dir:
        checkpoint_files = (
            Path(args.ckpt_dir) / "consolidated" / "consolidated.safetensors",
            Path(args.ckpt_dir) / "consolidated" / "lora.safetensors",
        )
        if not any(path.is_file() for path in checkpoint_files):
            raise FileNotFoundError(
                f"No checkpoint weights found under --ckpt-dir: {args.ckpt_dir}"
            )
        
    path_cfg = config.get("moshi_paths", {})
    hf_repo = path_cfg.get("hf_repo_id") or loaders.DEFAULT_REPO

    def resolve_weight(config_key: str, hub_name: str) -> str:
        local_path = path_cfg.get(config_key)
        if local_path:
            if not os.path.isfile(local_path):
                raise FileNotFoundError(
                    f"moshi_paths.{config_key} does not exist: {local_path}"
                )
            log("info", f"Using local {config_key}: {local_path}")
            return local_path
        log("info", f"moshi_paths.{config_key} is unset; downloading {hub_name} from {hf_repo}.")
        return hf_hub_download(hf_repo, hub_name)
    
    # 1) Load Mimi
    log("info", "Loading mimi...")
    mimi_weight = resolve_weight("mimi_path", loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)
    # VAP tokenization용 별도 인스턴스 — streaming_forever를 호출하지 않아 batch 크기 제약 없음
    vap_mimi = None if original_personaplex else loaders.get_mimi(mimi_weight, device)
    
    # 2) Load Text Tokenizer
    tokenizer_path = resolve_weight("tokenizer_path", loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
    
    # 3) Load Moshi
    log("info", "Loading Moshi base model...")
    moshi_weight = resolve_weight("moshi_path", loaders.MOSHI_NAME)
    
    # Reset backchannel keys in loaders._lm_kwargs to avoid contamination
    for k in list(loaders._lm_kwargs.keys()):
        if k.startswith("backchannel_"):
            del loaders._lm_kwargs[k]

    # Inject backchannel config into _lm_kwargs (same pattern as train.py)
    bc_enabled = False
    bc_cfg = config.get("backchannel", {})
    if bc_cfg.get("enable", False):
        bc_enabled = True
        loaders._lm_kwargs["backchannel_enabled"] = True
        loaders._lm_kwargs["backchannel_module_type"] = bc_cfg.get("module_type", "mlp")
        loaders._lm_kwargs["backchannel_vap_dim"] = bc_cfg.get("vap_dim", 256)
        loaders._lm_kwargs["backchannel_bc_hidden"] = bc_cfg.get("bc_hidden", 512)
        loaders._lm_kwargs["backchannel_gumbel_temp_init"] = bc_cfg.get("gumbel_temp_init", 1.0)
        loaders._lm_kwargs["backchannel_gumbel_temp_min"] = bc_cfg.get("gumbel_temp_min", 0.5)
        loaders._lm_kwargs["backchannel_gumbel_anneal_rate"] = bc_cfg.get("gumbel_anneal_rate", 0.0001)
        loaders._lm_kwargs["backchannel_fusion_trainable"] = bc_cfg.get("fusion_trainable", False)
        loaders._lm_kwargs["backchannel_fusion_bc_init"] = bc_cfg.get("fusion_bc_init", 0.0)
        loaders._lm_kwargs["backchannel_fusion_vap_init"] = bc_cfg.get("fusion_vap_init", 0.0)
        loaders._lm_kwargs["backchannel_fusion_vad_init"] = bc_cfg.get("fusion_vad_init", 0.0)
        loaders._lm_kwargs["backchannel_fusion_bias_init"] = bc_cfg.get("fusion_bias_init", 0.0)
        loaders._lm_kwargs["backchannel_pad_token_id"] = bc_cfg.get("pad_token_id", 3)
        loaders._lm_kwargs["backchannel_epad_token_id"] = bc_cfg.get("epad_token_id", 0)
        if bc_cfg.get("module_type", "mlp") == "vap_gpt":
            loaders._lm_kwargs["backchannel_vap_repo_path"] = bc_cfg.get("vap_gpt_repo_path", "")
            loaders._lm_kwargs["backchannel_vap_checkpoint"] = bc_cfg.get("vap_gpt_checkpoint", None)
            loaders._lm_kwargs["backchannel_vap_channel_layers"] = bc_cfg.get("vap_gpt_channel_layers", 1)
            loaders._lm_kwargs["backchannel_vap_cross_layers"] = bc_cfg.get("vap_gpt_cross_layers", 3)
            loaders._lm_kwargs["backchannel_vap_num_heads"] = bc_cfg.get("vap_gpt_num_heads", 4)
            loaders._lm_kwargs["backchannel_vap_dropout"] = bc_cfg.get("vap_gpt_dropout", 0.1)
            loaders._lm_kwargs["backchannel_vap_use_silence_ctx_proj"] = bc_cfg.get(
                "use_silence_ctx_proj", True
            )
        log("info", f"Backchannel module enabled (type={bc_cfg.get('module_type', 'mlp')}).")
    else:
        loaders._lm_kwargs["backchannel_enabled"] = False
        log("info", "Backchannel VAP module disabled.")
        if not original_personaplex and args.epad_control != "none":
            log("warning", f"--epad-control={args.epad_control} has no effect without "
                           "the backchannel module; using backbone output unchanged.")
        if args.suppress_epad:
            log("warning", "--suppress-epad has NO effect without the backchannel module "
                           "(gate suppression lives inside the module branch).")

    # Inject face_gen config into _lm_kwargs BEFORE get_moshi_lm so that
    # lm.face_module is instantiated and load_checkpoint can overwrite its
    # weights with the end-to-end trained face_module.* from lora.safetensors.
    face_cfg = config.get("face_gen", {})
    if face_cfg.get("enable", False) and face_cfg.get("ckpt_path"):
        face_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../moshi/moshi/models/face",
        ))
        loaders._lm_kwargs["face_module_enabled"] = True
        loaders._lm_kwargs["face_module_version"] = int(face_cfg.get("model_version", 1))
        loaders._lm_kwargs["face_module_dir"] = face_dir
        loaders._lm_kwargs["face_module_checkpoint"] = face_cfg.get("ckpt_path")
        loaders._lm_kwargs["face_module_hidden_dim"] = int(face_cfg.get("hidden_dim", 512))
        loaders._lm_kwargs["face_module_layers"] = int(face_cfg.get("layers", 6))
        loaders._lm_kwargs["face_module_heads"] = int(face_cfg.get("heads", 8))
        loaders._lm_kwargs["face_module_code_dim"] = int(face_cfg.get("code_dim", 32))
        loaders._lm_kwargs["face_module_prior_warmup_frames"] = int(face_cfg.get("prior_warmup_frames", 10))
        log("info", f"Face module configured in _lm_kwargs (pretrained ckpt={face_cfg.get('ckpt_path')})")
    else:
        loaders._lm_kwargs["face_module_enabled"] = False

    lm = loaders.get_moshi_lm(moshi_weight, device=device, dtype=lm_dtype, cpu_offload=args.cpu_offload)
    
    # Apply fine-tuned checkpoint
    if args.ckpt_dir:
        lm = load_checkpoint(
            lm,
            args.ckpt_dir,
            config.get("lora", {}),
            use_pretrained_depformer=args.use_pretrained_depformer,
        )
    
    lm.eval()
    log("info", "Model loaded successfully.")
    
    # 4) Construct LMGen
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen_kwargs = dict(
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        use_sampling=True,
        temp=args.temp_audio,
        temp_text=args.temp_text,
        top_k=args.top_k_audio,
        top_k_text=args.top_k_text,
        mimi=mimi,
        suppress_epad=args.suppress_epad,
        bc_context_frames=max(
            1, int(round(float(config.get("duration_sec", 10.0)) * mimi.frame_rate))
        ),
        epad_control=args.epad_control,
        fusion_bc_weight=args.fusion_bc_weight,
        fusion_vap_weight=args.fusion_vap_weight,
        fusion_vad_weight=args.fusion_vad_weight,
        fusion_threshold=args.fusion_threshold,
    )
    if not original_personaplex:
        lm_gen_kwargs.update(
            mimi=mimi,
            suppress_epad=args.suppress_epad,
            bc_context_frames=max(
                1, int(round(float(config.get("duration_sec", 10.0)) * mimi.frame_rate))
            ),
            epad_control=args.epad_control,
            fusion_bc_weight=args.fusion_bc_weight,
            fusion_vap_weight=args.fusion_vap_weight,
            fusion_vad_weight=args.fusion_vad_weight,
            fusion_threshold=args.fusion_threshold,
        )
    lm_gen = LMGen(lm, **lm_gen_kwargs)
    if not original_personaplex:
        if lm.backchannel_fusion_trainable:
            log(
                "info",
                f"EPAD control={args.epad_control}, learned fusion "
                f"bc={lm.backchannel_fusion_bc_weight.item():.6f}, "
                f"vap={lm.backchannel_fusion_vap_weight.item():.6f}, "
                f"vad={lm.backchannel_fusion_vad_weight.item():.6f}, "
                f"bias={lm.backchannel_fusion_bias.item():.6f}, "
                f"context_frames={lm_gen.bc_context_frames}",
            )
        else:
            log(
                "info",
                f"EPAD control={args.epad_control}, bc_weight={args.fusion_bc_weight}, "
                f"vap_weight={args.fusion_vap_weight}, vad_weight={args.fusion_vad_weight}, "
                f"context_frames={lm_gen.bc_context_frames}",
            )
    
    # Set streaming mode (critical for LMGen to work)
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)
    
    # Warmup CUDA graphs
    log("info", "Warming up the model...")
    warmup(mimi, other_mimi, lm_gen, device, frame_size)
    
    # Default Voice Prompt setup (used as fallback when a sample has no voice_sample entry)
    default_voice_prompt_path = "voices/NATM1.pt"
    if not os.path.exists(default_voice_prompt_path):
        voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=os.path.dirname(voices_tgz))
        default_voice_prompt_path = os.path.join(os.path.dirname(voices_tgz), "voices/NATM1.pt")

    # Build per-sample voice prompt lookup from the test JSONL (mirrors training dataset.py).
    # Each JSONL entry may have a "voice_sample" field pointing to an audio file whose
    # corresponding .pt cache (embeddings + KV cache) lives at the same path with .pt extension.
    voice_sample_lookup: dict[str, str] = {}  # {audio_basename -> .pt path}
    with open(args.test_jsonl, "r") as _f:
        for _line in _f:
            if not _line.strip():
                continue
            _entry = json.loads(_line)
            _vs = _entry.get("voice_sample")
            if _vs:
                _pt = os.path.splitext(_vs)[0] + ".pt"
                voice_sample_lookup[os.path.basename(_entry["path"])] = _pt
    log("info", f"Voice prompt lookup built: {len(voice_sample_lookup)} per-sample entries")
    if len(voice_sample_lookup) == 0:
        log("warning",
            "No voice_sample entries found in the test JSONL. "
            "All samples will use the default voice prompt. "
            "To use per-sample voice prompts, pass data_with_voice_sample.jsonl as --test-jsonl.")

    text_prompt = (
        args.text_prompt
        if args.text_prompt is not None
        else config.get("text_prompt", "")
    )
    lm_gen.text_prompt_tokens = (
        text_tokenizer.encode(_wrap_with_system_tags(text_prompt))
        if text_prompt else None
    )
    if text_prompt:
        log(
            "info",
            f"텍스트 프롬프트 적용 "
            f"({len(lm_gen.text_prompt_tokens)}토큰): {text_prompt!r}",
        )
    else:
        log("info", "텍스트 프롬프트 없음")

    # 5-a) Use the end-to-end trained face_module from the finetuned checkpoint.
    # lm.face_module was instantiated with pretrained weights by get_moshi_lm()
    # (via face_module_checkpoint in _lm_kwargs) and then overwritten with the
    # trained face_module.* tensors by load_checkpoint() above.
    face_gen = None
    if face_cfg.get("enable", False):
        if lm.face_module is not None:
            log("info", "Using end-to-end trained face_module from finetuned checkpoint.")
            lm.face_module.eval()
            face_gen = FaceGenerator(
                lm.face_module,
                max_context_frames=int(face_cfg.get("max_context_frames", 25)),
            )
        else:
            log("warning", "face_gen.enable=true but lm.face_module is None — skipping face generation.")

    # 5) Process Dataset
    samples = list_jsonl(args.test_jsonl)
    
    if args.sample_idx is not None:
        samples = [samples[args.sample_idx]]
        log("info", f"Testing single sample at index {args.sample_idx}")
    elif args.input_wav is not None:
        samples = [s for s in samples if s["path"] == args.input_wav]
        log("info", f"Testing single sample with path {args.input_wav}")
    else:
        if args.world_size > 1:
            samples = samples[args.rank::args.world_size]
            log("info", f"Rank {args.rank}/{args.world_size}: Found {len(samples)} samples to test out of total.")
        else:
            log("info", f"Found {len(samples)} samples to test.")
    
    # Resolve base directory for relative WAV paths in the jsonl
    jsonl_dir = os.path.dirname(os.path.abspath(args.test_jsonl))
    
    interleaved_tokenizer = None
    if bc_enabled:
        # Only the optional GT-VAP comparison needs the training tokenizer.
        from finetune.data.interleaver import Interleaver, InterleavedTokenizer
        interleaver = Interleaver(
            text_tokenizer,
            mimi.frame_rate,
            lm.text_padding_token_id,
            lm.end_of_text_padding_id,
            lm.zero_token_id,
            keep_main_only=True,
        )
        interleaved_tokenizer = InterleavedTokenizer(
            vap_mimi,
            interleaver,
            duration_sec=config.get("duration_sec", 15.0),
            vap_manifest_path=config.get("data", {}).get("vap_manifest", ""),
            flame_root=face_cfg.get("flame_root", "") if face_cfg.get("enable", False) else "",
            flame_speaker=face_cfg.get("flame_speaker", "bc"),
        )

    for i, sample in enumerate(samples):
        input_wav = sample["path"]
        # Resolve relative paths against the jsonl directory
        if not os.path.isabs(input_wav):
            input_wav = os.path.join(jsonl_dir, input_wav)
        base_name = os.path.basename(input_wav).replace(".wav", "")
        out_text = os.path.join(args.output_dir, f"{base_name}.json")
        out_vap = os.path.join(args.output_dir, f"{base_name}_vap.json")

        log("info", f"[{i+1}/{len(samples)}] Processing {input_wav}...")

        # Determine voice prompt path for this sample
        audio_basename = os.path.basename(input_wav)
        per_sample_pt = voice_sample_lookup.get(audio_basename)
        if args.no_voice_prompt:
            # Match training regimes that had no voice prompts (e.g. DualTalk jsonl
            # lacks voice_sample): skip the NATM1 fallback so inference context is
            # consistent with what the finetune actually saw.
            voice_prompt = ""
            log("info", "  Voice prompt: DISABLED (--no-voice-prompt)")
        elif per_sample_pt and os.path.exists(per_sample_pt):
            voice_prompt = per_sample_pt
            log("info", f"  Voice prompt: {per_sample_pt}")
        else:
            voice_prompt = default_voice_prompt_path
            if per_sample_pt:
                log("warning", f"  Voice prompt .pt not found ({per_sample_pt}), using default")
            else:
                log("info", f"  No voice_sample entry for {audio_basename}, using default")

        output_pcm, _, generated_text_tokens, bc_gate_log, has_audio = infer_one(
            input_wav,
            mimi, other_mimi, lm_gen,
            frame_size,
            voice_prompt_path=voice_prompt,
            face_gen=face_gen,
            text_tokenizer=text_tokenizer,
            collect_bc_log=bc_enabled,
            original_personaplex=original_personaplex,
        )

        if has_audio:
            # Save the exact same user channel that was fed to inference.
            input_pcm = _select_user_audio(
                lm_load_audio(input_wav, mimi.sample_rate),
                input_wav,
            )[0]

            # 1) Save input WAV (mono)
            out_input_wav = os.path.join(args.output_dir, f"{base_name}_input.wav")
            sphn.write_wav(out_input_wav, input_pcm, mimi.sample_rate)

            # 2) Save output WAV (mono)
            out_output_wav = os.path.join(args.output_dir, f"{base_name}_output.wav")
            sphn.write_wav(out_output_wav, output_pcm, mimi.sample_rate)

            # 3) Save merged WAV (stereo: ch0=input, ch1=output)
            out_merged_wav = os.path.join(args.output_dir, f"{base_name}_merged.wav")
            merged_pcm = np.stack([input_pcm, output_pcm], axis=0)
            sphn.write_wav(out_merged_wav, merged_pcm, mimi.sample_rate)

            # 4) Save text tokens + BC gate analysis log
            with open(out_text, 'w') as f:
                json.dump({
                    "text_tokens": generated_text_tokens,
                    "bc_gate_log": bc_gate_log,
                }, f, indent=2, ensure_ascii=False)
            log("info", f"Saved text tokens + BC gate log → {out_text}")
            log("info", f"Saved {out_input_wav}, {out_output_wav}, {out_merged_wav}")

            # 5) Optional face motion generation
            if face_gen is not None:
                try:
                    motion = face_gen.generate_numpy(vap_mimi)
                    out_face = os.path.join(args.output_dir, f"{base_name}_face_motion.npy")
                    np.save(out_face, motion)
                    log("info", f"Saved face motion to {out_face} (shape: {list(motion.shape)})")
                except Exception as _e:
                    import traceback as _tb
                    log("warning", f"Face generation failed: {_e}")
                    _tb.print_exc()
        else:
            log("warning", f"No generated frames for {input_wav}")
        
        # --- Ground Truth Forward Pass for VAP ---
        if bc_enabled:
            try:
                wav_np = lm_load_audio(input_wav, mimi.sample_rate)
                # 채널 swap은 InterleavedTokenizer 내부에서 처리하므로 stereo 그대로 전달
                sample_data = interleaved_tokenizer(wav_np, 0.0, input_wav)
                codes = sample_data.codes.to(device)  # already [1, K, T] from InterleavedTokenizer

                output = lm.forward_train(codes, mimi=mimi)
                if output.vap_logits is not None:
                    vap_probs = F.softmax(output.vap_logits[0], dim=-1)  # [T, 256]
                    vap_preds = vap_probs.argmax(dim=-1).cpu().tolist()
                    
                    vap_targets_list = None
                    if sample_data.vap_targets is not None:
                        vap_targets_list = sample_data.vap_targets[0].cpu().tolist()
                    
                    # Build per-token VAP records
                    num_tokens = len(vap_preds)
                    frame_dur_ms = 1000.0 / mimi.frame_rate  # e.g. 80ms
                    
                    per_token_records = []
                    for t in range(num_tokens):
                        record = {
                            "token_idx": t,
                            "time_ms": round(t * frame_dur_ms, 1),
                            "vap_pred": vap_preds[t],
                            "vap_pred_bits": _label_int_to_bits(vap_preds[t], 8),
                        }
                        if vap_targets_list is not None:
                            gt = vap_targets_list[t]
                            record["vap_gt"] = gt
                            if gt != -100:
                                record["vap_gt_bits"] = _label_int_to_bits(gt, 8)
                                record["match"] = (vap_preds[t] == gt)
                            else:
                                record["vap_gt_bits"] = None
                                record["match"] = None
                        per_token_records.append(record)
                    
                    # Summary stats
                    valid_pairs = [(r["vap_pred"], r["vap_gt"]) for r in per_token_records
                                   if r.get("vap_gt") is not None and r["vap_gt"] != -100]
                    accuracy = sum(1 for p, g in valid_pairs if p == g) / max(len(valid_pairs), 1)
                    
                    vap_output = {
                        "file": input_wav,
                        "num_tokens": num_tokens,
                        "num_valid_targets": len(valid_pairs),
                        "accuracy": round(accuracy, 4),
                        "frame_duration_ms": frame_dur_ms,
                        "tokens": per_token_records,
                    }
                    
                    with open(out_vap, 'w') as f:
                        json.dump(vap_output, f, indent=2, ensure_ascii=False)
                    log("info", f"Saved VAP per-token predictions to {out_vap} (acc={accuracy:.4f})")
            except Exception as e:
                import traceback
                log("warning", f"VAP forward pass failed: {e}")
                traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test Dataset Inference")
    parser.add_argument("--config", type=str, default="./output/exp1_backbone_only/args.yaml", help="Path to args.yaml or training config yaml")
    parser.add_argument(
        "--no-auto-fix-config-paths",
        action="store_true",
        help="Disable automatic rebasing of missing paths copied from another server.",
    )
    parser.add_argument("--test-jsonl", type=str, default='./experiments_ami/ami_test_10.jsonl', help="Path to data.jsonl for the test dataset")
    parser.add_argument("--output-dir", type=str, default="./result/exp1_backbone_only/400", help="Directory to save generated outputs")
    parser.add_argument("--ckpt-dir", type=str, default="./output/exp1_backbone_only/checkpoints/checkpoint_000400", help="Directory containing consolidated/lora.safetensors")
    parser.add_argument(
        "--base-model-only",
        action="store_true",
        help="Use pretrained PersonaPlex weights with lm2 while following the original "
             "offline inference behavior; disables checkpoint/backchannel/face modules.",
    )
    parser.add_argument(
        "--use-pretrained-depformer",
        action="store_true",
        help="Keep the Depformer (including its embeddings/input projections and output heads) "
             "from the pretrained PersonaPlex/Moshi model instead of loading it from --ckpt-dir.",
    )
    parser.add_argument("--sample-idx", type=int, default=None, help="Process only a specific index in the JSONL")
    parser.add_argument("--input-wav", type=str, default=None, help="Process only a specific WAV path in the JSONL")
    parser.add_argument("--device", type=str, default="cuda",
                        help='"cuda", "cuda:N", or "cpu"')
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float32", "bfloat16", "float16"],
                        help='LM weight dtype. "auto" = bfloat16 on GPU / float32 on CPU.')
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--no-voice-prompt", action="store_true",
                        help="Disable the voice prompt entirely (no NATM1 fallback). Use when the "
                             "training data had no voice_sample entries (e.g. DualTalk) so the "
                             "inference context matches training.")
    parser.add_argument(
        "--text-prompt",
        type=str,
        default=None,
        help="Override config.text_prompt; PersonaPlex <system> tags are added if missing.",
    )
    parser.add_argument("--temp-audio", type=float, default=0.8,
                        help="Audio sampling temperature (PersonaPlex default: 0.8).")
    parser.add_argument("--top-k-audio", type=int, default=250,
                        help="Audio sampling top-k (PersonaPlex default: 250).")
    parser.add_argument("--temp-text", type=float, default=0.7, help="Text sampling temperature.")
    parser.add_argument("--top-k-text", type=int, default=25, help="Text sampling top-k.")
    parser.add_argument("--suppress-epad", action="store_true",
                        help="Force [EPAD] → [PAD] whenever g_final=0 (VAP says don't speak). "
                             "Legacy mode only; word tokens in progress are never replaced.")
    parser.add_argument(
        "--epad-control", choices=("none", "legacy", "fusion"), default="none",
        help="EPAD controller: none=backbone only, legacy=hard BC-head gate, "
             "fusion=training-free BC/VAP/VAD residual on the backbone EPAD logit "
             "followed by normal text sampling (default).",
    )
    parser.add_argument("--fusion-bc-weight", type=float, default=1.0,
                        help="Weight on explicit BC EPAD-vs-PAD log-odds.")
    parser.add_argument("--fusion-vap-weight", type=float, default=0.0,
                        help="Optional weight on marginal agent-near VAP log-odds.")
    parser.add_argument("--fusion-vad-weight", type=float, default=0.0,
                        help="Optional weight on user-quiet VAD log-odds.")
    parser.add_argument("--fusion-threshold", type=float, default=0.5,
                        help="Deprecated compatibility option; residual fusion samples "
                             "from adjusted logits and does not use a hard threshold.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--world-size", type=int, default=1, help="Total number of processes for inference")
    parser.add_argument("--rank", type=int, default=0, help="Rank of the current process")
    
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    run_test_inference(args)
