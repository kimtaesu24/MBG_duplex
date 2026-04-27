"""Face generation wrapper for CausalSoftVQContinuousTransformer.

Accumulates per-step (tokens, z) during autoregressive inference and
produces face motion via model.generate() at the end of each sample.
"""

from __future__ import annotations

import sys
from typing import List

import numpy as np
import torch


class FaceGenerator:
    """Wraps CausalSoftVQContinuousTransformer for per-sample face motion generation.

    Usage:
        gen = FaceGenerator.from_checkpoint(face_dir, ckpt_path, device,
                                            max_context_frames=25)
        gen.reset()
        for step ...:
            tokens, z = lm_gen.step(..., return_z=True)
            gen.add_step(tokens, z)
        motion = gen.generate(vap_mimi)   # [1, T_face, 54]
        np.save(path, motion[0].cpu().numpy())
        gen.reset()
    """

    def __init__(self, model, max_context_frames: int = 25):
        self.model = model
        self.max_context_frames = max_context_frames
        self._agent_tokens: List[torch.Tensor] = []  # each [B, 8, 1]
        self._z: List[torch.Tensor] = []              # each [B, 1, 4096]

    @classmethod
    def from_checkpoint(
        cls,
        face_dir: str,
        ckpt_path: str,
        device,
        max_context_frames: int = 25,
        **overrides,
    ) -> "FaceGenerator":
        """Load CausalSoftVQContinuousTransformer from a checkpoint.

        Architecture hyperparams are read from ckpt["args"] when present;
        overrides (hidden_dim, layers, heads, code_dim, prior_warmup_frames)
        take precedence over both ckpt["args"] and built-in defaults.
        """
        if face_dir not in sys.path:
            sys.path.insert(0, face_dir)
        from softvq_continuous_online_train import CausalSoftVQContinuousTransformer  # noqa: PLC0415

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        a = ckpt.get("args", {})

        model = CausalSoftVQContinuousTransformer(
            hidden_dim=int(overrides.get("hidden_dim", a.get("hidden_dim", 512))),
            layers=int(overrides.get("layers", a.get("layers", 6))),
            heads=int(overrides.get("heads", a.get("heads", 8))),
            code_dim=int(overrides.get("code_dim", a.get("code_dim", 32))),
            prior_warmup_frames=int(
                overrides.get("prior_warmup_frames", a.get("prior_warmup_frames", 10))
            ),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        return cls(model, max_context_frames=max_context_frames)

    def reset(self) -> None:
        """Clear accumulated steps (call before each new sample)."""
        self._agent_tokens.clear()
        self._z.clear()

    def add_step(self, tokens: torch.Tensor, z: torch.Tensor) -> None:
        """Accumulate one autoregressive step.

        tokens: [B, dep_q+1, 1]  full output from lm_gen.step()
        z:      [B, 1, 4096]     transformer_out from lm_gen.step(return_z=True)
        """
        self._agent_tokens.append(tokens[:, 1:9].clone())  # [B, 8, 1]
        self._z.append(z.clone())                           # [B, 1, 4096]

    @torch.no_grad()
    def generate(self, mimi) -> torch.Tensor:
        """Decode accumulated audio tokens and generate face motion.

        mimi: MimiModel instance (vap_mimi is ideal — no streaming state).
        Returns [B, T_face, 54] float32 on the model's device.
        """
        if not self._agent_tokens:
            raise RuntimeError("No steps accumulated. Call add_step() before generate().")

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        agent_codes = torch.cat(self._agent_tokens, dim=-1).to(device)  # [B, 8, T]
        llm_feat = torch.cat(self._z, dim=1).to(device=device, dtype=dtype)  # [B, T, 4096]

        # Decode discrete codes → continuous Mimi latents
        audio_feat = mimi.decode_latent(agent_codes).transpose(1, 2).to(dtype=dtype)  # [B, T, 512]

        motion = self.model.generate(
            audio_feat,
            llm_feat=llm_feat,
            max_context_frames=self.max_context_frames,
        )
        return motion  # [B, T_face, 54]

    def generate_numpy(self, mimi) -> np.ndarray:
        """Convenience wrapper: returns motion as a numpy array [T_face, 54]."""
        return self.generate(mimi)[0].cpu().float().numpy()
