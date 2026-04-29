"""
Offline inference script over a test dataset (e.g. data.jsonl).
Loads the finetuned Moshi checkpoint and iteratively runs inference, saving the decoded audio and text tokens.
"""

import argparse
import os
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
from moshi.models import LMGen, MimiModel
from moshi.client_utils import make_log
from moshi.models.lm import load_audio as lm_load_audio
from moshi.models.lm import _iterate_audio as lm_iterate_audio
from moshi.models.lm import encode_from_sphn as lm_encode_from_sphn
from moshi.offline import warmup, decode_tokens_to_pcm


def log(level: str, msg: str):
    print(make_log(level, msg))

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
        cleaned[k] = v
    return cleaned


def load_checkpoint(lm, ckpt_dir):
    """Loads the finetuned checkpoint (consolidated or lora-only)."""
    consolidated_path = os.path.join(ckpt_dir, "consolidated", "consolidated.safetensors")
    lora_path = os.path.join(ckpt_dir, "consolidated", "lora.safetensors")

    if os.path.exists(consolidated_path):
        log("info", f"Loading full checkpoint from {consolidated_path}")
        state_dict = safetensors.torch.load_file(consolidated_path)
        state_dict = _strip_peft_prefixes(state_dict)
        missing, _ = lm.load_state_dict(state_dict, strict=False)
        bc_missing = [k for k in missing if "backchannel" in k]
        face_missing = [k for k in missing if "face_module" in k]
        if bc_missing:
            log("warning", f"Backchannel keys still missing after load: {bc_missing}")
        if face_missing:
            log("warning", f"Face module keys still missing after load: {face_missing[:3]}")
        if not bc_missing and not face_missing:
            log("info", "Checkpoint loaded successfully.")

    elif os.path.exists(lora_path):
        log("info", f"Loading LoRA checkpoint from {lora_path}")
        state_dict = safetensors.torch.load_file(lora_path)
        state_dict = _strip_peft_prefixes(state_dict)

        # 1. Load backchannel + face_module weights directly into LMModel.
        #    These modules are full-finetuned (no LoRA) and stored flat in lora.safetensors.
        direct_weights = {k: v for k, v in state_dict.items()
                          if k.startswith("backchannel.") or k.startswith("face_module.")}
        if direct_weights:
            missing, _ = lm.load_state_dict(direct_weights, strict=False)
            bc_missing = [k for k in missing if "backchannel" in k]
            face_missing = [k for k in missing if "face_module" in k]
            if bc_missing:
                log("warning", f"Backchannel keys missing: {bc_missing}")
            if face_missing:
                log("warning", f"Face module keys missing: {face_missing[:3]}")
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
        #    In Personaplex, only out_proj and text_linear are plain nn.Linear
        #    that PEFT can wrap — the other names in the regex don't match any
        #    module in the actual graph.  face_module / backchannel are excluded
        #    by the negative lookahead so their linear layers are left as-is.
        lora_weights = {k: v for k, v in state_dict.items()
                        if not k.startswith("backchannel.") and not k.startswith("face_module.")}
        if lora_weights:
            try:
                from peft import get_peft_model, LoraConfig, TaskType
                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=64,
                    lora_alpha=128,
                    target_modules=(
                        r"(?!.*(face_module|backchannel))"
                        r".*(in_proj|out_proj|linear1|linear2|text_linear|input_proj)"
                    ),
                    bias="none",
                )
                peft_lm = get_peft_model(lm, lora_config)
                prefixed = {f"base_model.model.{k}": v for k, v in lora_weights.items()}
                missing_lora, _ = peft_lm.load_state_dict(prefixed, strict=False)
                missing_lora = [k for k in missing_lora if "lora_" in k]
                if missing_lora:
                    log("warning", f"Some LoRA keys missing: {missing_lora[:3]}")
                else:
                    log("info", f"LoRA weights loaded ({len(lora_weights)} tensors).")
                # Merge LoRA deltas into base weights and restore a plain LMModel.
                lm = peft_lm.merge_and_unload()
            except ImportError:
                log("error", "LoRA weights found but PEFT is not installed.")
    else:
        log("warning", f"No checkpoint found in {ckpt_dir}, running base model.")
    return lm

def run_test_inference(args):
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)
    
    log("info", f"Reading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    hf_repo = config.get("moshi_paths", {}).get("hf_repo_id", loaders.DEFAULT_REPO)
    
    # 1) Load Mimi
    log("info", "Loading mimi...")
    mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)
    # VAP tokenization용 별도 인스턴스 — streaming_forever를 호출하지 않아 batch 크기 제약 없음
    vap_mimi = loaders.get_mimi(mimi_weight, device)
    
    # 2) Load Text Tokenizer
    tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
    
    # 3) Load Moshi
    log("info", "Loading Moshi base model...")
    moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
    
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
        if bc_cfg.get("module_type", "mlp") == "vap_gpt":
            loaders._lm_kwargs["backchannel_vap_repo_path"] = bc_cfg.get("vap_gpt_repo_path", "")
            loaders._lm_kwargs["backchannel_vap_checkpoint"] = bc_cfg.get("vap_gpt_checkpoint", None)
            loaders._lm_kwargs["backchannel_vap_channel_layers"] = bc_cfg.get("vap_gpt_channel_layers", 1)
            loaders._lm_kwargs["backchannel_vap_cross_layers"] = bc_cfg.get("vap_gpt_cross_layers", 3)
            loaders._lm_kwargs["backchannel_vap_num_heads"] = bc_cfg.get("vap_gpt_num_heads", 4)
            loaders._lm_kwargs["backchannel_vap_dropout"] = bc_cfg.get("vap_gpt_dropout", 0.1)
        log("info", f"Backchannel module enabled (type={bc_cfg.get('module_type', 'mlp')}).")
    else:
        loaders._lm_kwargs["backchannel_enabled"] = False
        log("info", "Backchannel VAP module disabled.")

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

    lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=args.cpu_offload)
    
    # Apply fine-tuned checkpoint
    if args.ckpt_dir:
        lm = load_checkpoint(lm, args.ckpt_dir)
    
    lm.eval()
    log("info", "Model loaded successfully.")
    
    # 4) Construct LMGen
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        use_sampling=True,
        temp=0.8,
        temp_text=0.7,
        top_k=250,
        top_k_text=25,
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

    text_prompt = "<system> You are a highly realistic conversational AI. You should listen attentively. <system>"
    lm_gen.text_prompt_tokens = text_tokenizer.encode(text_prompt)

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
        log("info", f"Found {len(samples)} samples to test.")
    
    # Resolve base directory for relative WAV paths in the jsonl
    jsonl_dir = os.path.dirname(os.path.abspath(args.test_jsonl))
    
    # Initialize tokenizer fully to get vap_targets
    from finetune.data.interleaver import Interleaver, InterleavedTokenizer
    interleaver = Interleaver(
        text_tokenizer,
        mimi.frame_rate,
        lm.text_padding_token_id,
        lm.end_of_text_padding_id,
        lm.zero_token_id,
        keep_main_only=True,
    )
    # Using 15s duration arbitrarily to fetch codes and VAP targets
    interleaved_tokenizer = InterleavedTokenizer(
        vap_mimi, interleaver, duration_sec=config.get("duration_sec", 15.0),
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
        out_text = os.path.join(args.output_dir, f"out_{base_name}.json")
        out_vap = os.path.join(args.output_dir, f"out_{base_name}_vap.json")
        
        log("info", f"[{i+1}/{len(samples)}] Processing {input_wav}...")
        
        # --- Autoregressive Generation ---
        # Reset streaming
        mimi.reset_streaming()
        other_mimi.reset_streaming()
        lm_gen.reset_streaming()

        # Load per-sample voice prompt (mirrors training: JSONL voice_sample -> .pt cache).
        # Must happen after reset_streaming() and before step_system_prompts() so the
        # embeddings replay and KV-cache restore land on a clean streaming state.
        audio_basename = os.path.basename(input_wav)
        per_sample_pt = voice_sample_lookup.get(audio_basename)
        if per_sample_pt and os.path.exists(per_sample_pt):
            lm_gen.load_voice_prompt_embeddings(per_sample_pt)
            log("info", f"  Voice prompt: {per_sample_pt}")
        else:
            lm_gen.load_voice_prompt_embeddings(default_voice_prompt_path)
            if per_sample_pt:
                log("warning", f"  Voice prompt .pt not found ({per_sample_pt}), using default")
            else:
                log("info", f"  No voice_sample entry for {audio_basename}, using default")

        # Warmup and system prompts
        lm_gen.step_system_prompts(mimi)
        mimi.reset_streaming()
        
        # Stream User Audio
        user_audio = lm_load_audio(input_wav, mimi.sample_rate)
        if user_audio.ndim == 2 and user_audio.shape[0] == 2:
            # Swap channels: original AMI stereo is often [Agent, User] or simply reversed
            # By swapping [1, 0], we make Ch0 the User and Ch1 the Agent.
            # Moshi normally expects User in Ch0 for its input iterator.
            log("info", f"Swapping channels for {input_wav}...")
            user_audio = user_audio[[1, 0], :]
            
        generated_frames = []
        generated_text_tokens = []
        bc_gate_log = []
        special_token_map = {0: 'EPAD', 1: 'BOS', 2: 'EOS', 3: 'PAD'}

        if face_gen is not None:
            face_gen.reset()

        for user_encoded in lm_encode_from_sphn(
            mimi,
            lm_iterate_audio(user_audio, sample_interval_size=frame_size, pad=True),
            max_batch=1,
        ):
            steps = user_encoded.shape[-1]
            for c in range(steps):
                if face_gen is not None:
                    result = lm_gen.step(user_encoded[:, :, c:c+1], return_z=True)
                    if result is None or result[0] is None:
                        continue
                    tokens, z = result
                    face_gen.add_step(tokens, z)
                else:
                    result = lm_gen.step(user_encoded[:, :, c:c+1], return_z=True)
                    if result is None or result[0] is None:
                        continue
                    tokens, z = result

                generated_frames.append(decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens))

                text_id = tokens[0, 0, 0].item()
                if text_id in special_token_map:
                    generated_text_tokens.append(special_token_map[text_id])
                else:
                    _t = text_tokenizer.id_to_piece(text_id).replace("▁", " ")
                    generated_text_tokens.append(_t)

                # Level 2: BC gate analysis — log per-step gate internals
                _PAD  = lm_gen.lm_model.text_padding_token_id
                _EPAD = lm_gen.lm_model.end_of_text_padding_id
                _label = "PAD" if text_id == _PAD else ("EPAD" if text_id == _EPAD else f"WORD({text_id})")
                _bc_module = getattr(lm_gen.lm_model, "backchannel", None)
                if _bc_module is not None and z is not None:
                    with torch.no_grad():
                        _bc = _bc_module(z, emb_cb0=lm_gen.lm_model.depformer_text_emb, step=999_999)
                    _y_bc   = _bc.bc_logits[0, 0].softmax(-1)[1].item()
                    _s_pad  = _bc.silence_gate_logits[0, 0].softmax(-1)[1].item()
                    _g_soft = _y_bc * _s_pad
                    _g_final = int(_bc.bc_gate[0, 0].item())
                    print(f"[BC] step={c:4d} | token={_label:14s} | y_bc={_y_bc:.3f}  s_pad={_s_pad:.3f}  g_soft={_g_soft:.3f}  g_final={_g_final}")
                    bc_gate_log.append({
                        "step": c,
                        "token_id": text_id,
                        "token_label": _label,
                        "y_bc": round(_y_bc, 4),
                        "s_pad": round(_s_pad, 4),
                        "g_soft": round(_g_soft, 4),
                        "g_final": _g_final,
                    })
                    
        # Save outputs
        if generated_frames:
            output_pcm = np.concatenate(generated_frames, axis=-1)
            # Load input audio as numpy for saving
            input_pcm = user_audio
            if input_pcm.ndim == 2:
                input_pcm = input_pcm[0]  # mono
            target_samples = input_pcm.shape[-1]
            
            # Trim/pad output to match input length
            if output_pcm.shape[-1] > target_samples:
                output_pcm = output_pcm[:target_samples]
            elif output_pcm.shape[-1] < target_samples:
                output_pcm = np.concatenate([
                    output_pcm,
                    np.zeros(target_samples - output_pcm.shape[-1], dtype=output_pcm.dtype)
                ], axis=-1)
            
            # 1) Save input WAV (mono)
            out_input_wav = os.path.join(args.output_dir, f"{base_name}_input.wav")
            sphn.write_wav(out_input_wav, input_pcm, mimi.sample_rate)
            
            # 2) Save output WAV (mono)
            out_output_wav = os.path.join(args.output_dir, f"{base_name}_output.wav")
            sphn.write_wav(out_output_wav, output_pcm, mimi.sample_rate)
            
            # 3) Save merged WAV (stereo: ch0=input, ch1=output)
            out_merged_wav = os.path.join(args.output_dir, f"{base_name}_merged.wav")
            merged_pcm = np.stack([input_pcm, output_pcm], axis=0)  # [2, T]
            sphn.write_wav(out_merged_wav, merged_pcm, mimi.sample_rate)
            
            # 4) Save text tokens + BC gate analysis log into same file
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
                    motion = face_gen.generate_numpy(vap_mimi)  # [T_face, 56]
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

                output = lm.forward_train(codes)
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
    parser.add_argument("--config", type=str, required=True, help="Path to args.yaml or training config yaml")
    parser.add_argument("--test-jsonl", type=str, required=True, help="Path to data.jsonl for the test dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save generated outputs")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Directory containing consolidated/lora.safetensors")
    parser.add_argument("--sample-idx", type=int, default=None, help="Process only a specific index in the JSONL")
    parser.add_argument("--input-wav", type=str, default=None, help="Process only a specific WAV path in the JSONL")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    run_test_inference(args)
