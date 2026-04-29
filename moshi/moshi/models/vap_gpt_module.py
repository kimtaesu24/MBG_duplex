# VapGPTBackchannelModule — drop-in replacement for BackchannelModule.
#
# Architecture difference vs. BackchannelModule:
#   BackchannelModule  : z_s → simple MLP → VAP logits
#   VapGPTBackchannelModule: z_s → two linear projections (User/Agent pseudo-streams)
#                            → VapGPT's ar_channel (per-stream GPT)
#                            → VapGPT's ar (cross-stream GPTStereo)
#                            → vap_head → VAP logits
#
# The LM produces a SINGLE hidden-state stream z_s (it has no separate speaker
# channels).  We resolve the stereo requirement by projecting z_s twice:
#
#   proj_user (z_s)  → x_user   — extracts User-side contextual features
#   proj_agent(z_s)  → x_agent  — extracts Agent-side contextual features
#
# Both streams are then processed by VapGPT's GPT layers, which model
# cross-stream turn-taking dynamics.  Pretrained VapGPT checkpoint weights
# can be loaded for ar_channel, ar, and vap_head; proj_* and bc_mlp are always
# trained from scratch.
#
# Output interface is identical to BackchannelModule (BackchannelOutput).


''' paste code from vap.modules.py '''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from typing import Dict, Optional, Tuple


def ffn_block(
    din: int,
    dff: int,
    activation: str = "GELU",
    dropout: float = 0.0,
    bias: bool = False,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(din, dff, bias=bias),
        getattr(nn, activation)(),
        nn.Dropout(p=dropout),
        nn.Linear(dff, din, bias=bias),
    )


class MultiHeadAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float, bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim

        # key, query, value projections for all heads
        self.key = nn.Linear(dim, dim, bias=bias)
        self.query = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)

        # head re-shapers
        self.unstack_heads = Rearrange("b t (h d) -> b h t d", h=self.num_heads)
        self.stack_heads = Rearrange("b h t d -> b t (h d)")

        # regularization
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # output projection
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.scale = 1.0 / math.sqrt(dim)

    def get_scores(self, q: torch.Tensor, k: torch.Tensor):
        """
        Arguments:
            q: (B, heads, T, D)
            k: (B, heads, T, D)

        Return:
            QK:     (B, heads, T, T)
        """
        return torch.einsum("bhid,bhjd->bhij", q, k)

    @staticmethod
    def prepare_causal_mask(T, device="cpu", dtype=torch.float32):
        mask = torch.tril(torch.ones((T, T), device=device, dtype=dtype)).view(
            1, 1, T, T
        )
        mask.requires_grad_(False)
        return mask

    def mask_scores(self, qk: torch.Tensor, mask=None):
        T = qk.size(-1)
        if mask is None:
            mask = MultiHeadAttention.prepare_causal_mask(
                T, device=qk.device, dtype=qk.dtype
            )
        qk = qk.masked_fill(mask == 0, float("-inf"))
        return qk

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, D = Q.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.unstack_heads(self.key(K))  # (B, heads, T, D_head)
        q = self.unstack_heads(self.query(Q))  # (B, heads, T, D_head)
        v = self.unstack_heads(self.value(V))  # (B, heads, T, D_head)

        # QK
        att = self.get_scores(q, k) * self.scale  #  (B, nh, T, T)
        att = self.mask_scores(att, mask)
        att = F.softmax(att, dim=-1)

        # Softmax, dropout, values
        y = self.attn_drop(att) @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # re-assemble all head outputs side by side
        y = self.stack_heads(y)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class MultiHeadAttentionAlibi(MultiHeadAttention):
    def __init__(self, dim: int, num_heads: int, dropout: float, bias: bool = False):
        super().__init__(dim, num_heads, dropout, bias)
        # self.m = torch.tensor(MultiHeadAttentionAlibi.get_slopes(num_heads))
        self.register_parameter(
            "m",
            nn.Parameter(torch.tensor(MultiHeadAttentionAlibi.get_slopes(num_heads))),
        )
        self.m.requires_grad_(False)
        self.mask = None

    @staticmethod
    def get_slopes(n):
        """
        * aLiBi slopes for heads.
        * m in Figure 3.
        * Source:
            - https://github.com/ofirpress/attention_with_linear_biases/blob/5b327adc6d131e28b40ba58906b30bb469483519/fairseq/models/transformer.py#L742

        Comments:

        In the paper, we only train models that have 2^a heads for some a. This function has
        some good properties that only occur when the input is a power of 2.
        To maintain that even closest_power_of_2 = 2**math.floor(math.log2(n))
        when the number of heads is not a power of 2, we use this workaround.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        # In the paper, we only train models that have 2^a heads for some a. This function has
        # some good properties that only occur when the input is a power of 2. To maintain that even
        # when the number of heads is not a power of 2, we use this workaround.
        if math.log2(n).is_integer():
            slopes = get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            slopes = (
                get_slopes_power_of_2(closest_power_of_2)
                + MultiHeadAttentionAlibi.get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )
        return slopes

    @staticmethod
    def get_relative_bias_matrix(n, num_heads, device="cpu", dtype=torch.float32):
        """Relative Bias matrix for aLiBi embeddings"""
        return (
            torch.arange(n, device=device, dtype=dtype)
            .view(1, 1, -1)
            .expand(1, num_heads, -1)
        )

    def get_alibi_mask(self, T: int, device="cpu", dtype=torch.float32):
        rel_bias_mat = MultiHeadAttentionAlibi.get_relative_bias_matrix(
            T, self.num_heads, device, dtype
        )
        alibi = rel_bias_mat * self.m.unsqueeze(0).unsqueeze(-1).to(device)

        # Causal mask (standard GPT pask)
        # lower triangle = 1
        # upper triangle = 0
        mask = MultiHeadAttention.prepare_causal_mask(T, device, dtype)  # (1, 1, T, T)
        # Repeat to get a mask for each head
        mask = mask.repeat(1, self.num_heads, 1, 1)  # (1, num_heads, T, T)
        # fill "future" information with negative infinity
        mask.masked_fill_(mask == 0, float("-inf"))

        # Add causality mask to alibi  (1, num_heads, T, T)
        alibi = alibi.unsqueeze(-2) + mask
        # alibi.requires_grad_(False)  # this should not be trained
        alibi = alibi.detach()
        return alibi

    def mask_scores(self, qk: torch.Tensor, mask=None):
        T = qk.size(-1)
        if mask is None:
            if self.mask is None or self.mask.shape[-1] < T:
                mask = self.get_alibi_mask(T, device=qk.device, dtype=qk.dtype)
                self.mask = mask
            else:
                mask = self.mask[..., :T, :T]

        # add aLiBi-mask to qk (see Figure 3.)
        # Addition/translation does not effect softmax (over each row)
        # mentioned in the original representation
        qk = qk + mask.to(qk.device)
        return qk


class TransformerLayer(nn.Module):
    """
    Transformer Layer

    Using pre-layer-normalization: https://arxiv.org/pdf/2002.04745.pdf
    Inspiration: https://nn.labml.ai/transformers/models.html
    AliBI Attention: https://ofir.io/train_short_test_long.pdf
    """

    def __init__(
        self,
        dim: int = 256,
        ffn_dim: int = 768,
        num_heads: int = 4,
        ffn_activation: str = "GELU",
        dropout: float = 0.1,
        cross_attention: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.cross_attention = cross_attention

        self.dropout = nn.Dropout(p=dropout)
        self.ln_self_attn = nn.LayerNorm(dim)
        self.ln_ffnetwork = nn.LayerNorm(dim)
        self.mha = MultiHeadAttentionAlibi(
            dim=dim, num_heads=num_heads, dropout=dropout
        )
        self.ffnetwork = ffn_block(
            dim, ffn_dim, activation=ffn_activation, dropout=dropout
        )

        if cross_attention:
            self.ln_src_attn = nn.LayerNorm(dim)
            self.mha_cross = MultiHeadAttentionAlibi(
                dim=dim, num_heads=num_heads, dropout=dropout
            )

    def forward(
        self,
        x: torch.Tensor,
        src: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Using pre-layer-normalization: https://arxiv.org/pdf/2002.04745.pdf
        """

        # Self-attention
        z = self.ln_self_attn(x)
        self_attn, self_attn_weights = self.mha(Q=z, K=z, V=z, mask=mask)

        # Residual
        x = x + self.dropout(self_attn)

        # Cross-attention
        cross_attn_weights = None
        if self.cross_attention and src is not None:
            z = self.ln_src_attn(x)
            # https://nn.labml.ai/transformers/models.html#section-16
            # Don't normalize src... why?
            cross_attn, cross_attn_weights = self.mha_cross(
                Q=z, K=src, V=src, mask=mask
            )
            x = x + self.dropout(cross_attn)

        x = x + self.dropout(self.ffnetwork(self.ln_ffnetwork(x)))
        return x, self_attn_weights, cross_attn_weights


class TransformerStereoLayer(TransformerLayer):
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        # sa1w: self-attention-weights 1
        # ca1w: cross-attention-weights 1
        z1, sa1w, ca1w = super().forward(x=x1, src=x2, mask=mask)
        z2, sa2w, ca2w = super().forward(x=x2, src=x1, mask=mask)
        return z1, z2, [sa1w, ca1w, sa2w, ca2w]


class GPT(nn.Module):
    """
    GPT like transformer Decoder-only class.

    * Uses AliBi attention (no positional embeddings or max-sequence-length)
    """

    def __init__(
        self,
        dim: int,
        dff_k: int = 3,
        num_layers: int = 4,
        num_heads: int = 4,
        activation: str = "GELU",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.dff = int(dim * dff_k)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = dropout

        self._build_layers()
        self.apply(self._init_weights)

    def _build_layers(self):
        layers = []
        for _ in range(self.num_layers):
            layers.append(
                TransformerLayer(
                    dim=self.dim,
                    ffn_dim=self.dff,
                    num_heads=self.num_heads,
                    ffn_activation=self.activation,
                    dropout=self.dropout,
                )
            )
        self.layers = nn.ModuleList(layers)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self, x: torch.Tensor, attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        all_attention = []

        for layer in self.layers:
            x, self_attn_weights, _ = layer(x)
            if attention:
                all_attention.append(self_attn_weights)

        ret = {"x": x}

        if attention:
            self_attn_weights = torch.stack(all_attention, dim=1)
            ret["attn"] = self_attn_weights

        return ret


class GPTStereo(GPT):
    def _build_layers(self):
        layers = []
        for _ in range(self.num_layers):
            layers.append(
                TransformerStereoLayer(
                    dim=self.dim,
                    ffn_dim=self.dff,
                    num_heads=self.num_heads,
                    ffn_activation=self.activation,
                    dropout=self.dropout,
                    cross_attention=True,
                )
            )
        self.layers = nn.ModuleList(layers)

        # Combine output from both 'towers'
        self.combinator = Combinator(dim=self.dim, activation="GELU")

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, attention: bool = False
    ) -> Dict[str, torch.Tensor]:

        self_attn_a = []
        self_attn_b = []
        cross_attn_a = []
        cross_attn_b = []
        for layer in self.layers:
            x1, x2, attn_list = layer(x1=x1, x2=x2)
            if attention:
                # [sa1w, ca1w, sa2w, ca2w] = attn_list
                self_attn_a.append(attn_list[0])
                cross_attn_a.append(attn_list[1])
                self_attn_b.append(attn_list[2])
                cross_attn_b.append(attn_list[3])

        x = self.combinator(x1, x2)
        ret = {"x": x, "x1": x1, "x2": x2}

        if attention:
            # B, num_layers, num_heads, N, N
            self_attn_a = torch.stack(self_attn_a, dim=1)  # stack on layer dim
            self_attn_b = torch.stack(self_attn_b, dim=1)  # stack on layer dim
            cross_attn_a = torch.stack(cross_attn_a, dim=1)  # stack on layer dim
            cross_attn_b = torch.stack(cross_attn_b, dim=1)  # stack on layer dim
            ret["self_attn"] = torch.stack([self_attn_a, self_attn_b], dim=1)
            ret["cross_attn"] = torch.stack([cross_attn_a, cross_attn_b], dim=1)
        return ret


class Combinator(nn.Module):
    """
    Combines the "ego-centric" representations from identical 'towers'
    processing channel 0 and 1. The towers are identical (shared weights)
    and therefore channel agnostic, e.g. they don't know if they process information
    from the view of speaker A or B.

    Here we have specific layers associated with each channel to join the representations
    into a single coherent space with channel information included.
    """

    def __init__(self, dim: int, activation: str = "GELU"):
        super().__init__()
        self.dim = dim

        # Channel information
        self.h0_a = nn.Linear(dim, dim, bias=False)  # Channel 0
        self.h0_b = nn.Linear(dim, dim, bias=False)  # Channel 1
        self.ln = nn.LayerNorm(self.dim)

        # Activation
        self.activation = getattr(nn, activation)()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Combines the hidden states from identical 'towers' which have processed
        each channel from an 'ego-centric' view. However, the towers are channel agnostic
        by default (shared weights) so in this step we process the information from channel 0, 1
        separately into a joint representation.

        The final representation will (see GPTStereo -> ProjectionModel) go into a final linear
        layer to produce logits.
        """

        # Channel specific information
        ha = self.activation(self.ln(self.h0_a(x1)))
        hb = self.activation(self.ln(self.h0_b(x2)))
        h = ha + hb  # combine estimations from both parties
        return h






######## code for backchannel #########
import sys
from .backchannel_vap import BackchannelOutput, gumbel_softmax_st, compute_temperature


def _load_vap_state_dict(path: str) -> dict:
    """Load a VapGPT state dict from a .pt or .ckpt file.

    .pt  – plain state dict, load directly.
    .ckpt – Lightning checkpoint: extract state_dict, strip 'net.' prefix,
            drop 'VAP.codebook' entries, rename
            'vap_head.projection_head' → 'vap_head'.
    """
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
        cleaned = {}
        for k, v in sd.items():
            if "VAP.codebook" in k:
                continue
            if "vap_head" in k:
                k = k.replace("vap_head.projection_head", "vap_head")
            cleaned[k.replace("net.", "")] = v
        return cleaned
    return raw  # plain .pt


class VapGPTBackchannelModule(nn.Module):
    """Backchannel prediction using VapGPT's cross-stream GPT architecture.

    Replaces BackchannelModule inside LMModel.  Input is the LM's single hidden-
    state stream z_s; the CPC audio encoder is NOT used.

    Two linear projections split z_s into pseudo User / Agent streams, which are
    then processed by VapGPT's per-stream GPT (ar_channel) and cross-stream
    GPTStereo (ar) to model turn-taking dynamics.

    Args:
        lm_dim: Dimension of z_s (LM hidden states).
        depformer_dim: DepthTransformer embedding dimension (for BC gate output).
        card: Audio codebook vocabulary size.
        vap_repo_path: Path to the VoiceActivityProjection repository root.
        checkpoint_path: Optional pretrained VapGPT checkpoint (.pt or .ckpt).
            ar_channel, ar, and vap_head weights are loaded from it; proj_* and
            bc_mlp are always random-initialised.
        vap_dim: Hidden dimension of VapGPT's GPT layers (default 256).
        channel_layers: Number of per-stream GPT layers (ar_channel).
        cross_layers: Number of cross-stream GPTStereo layers (ar).
        num_heads: Attention heads in GPT layers.
        dropout: Dropout probability.
        bc_hidden: Hidden size of the BC gate MLP.
        pad_token_id / epad_token_id: Silence / backchannel token IDs.
        gumbel_temp_init / gumbel_temp_min / gumbel_anneal_rate:
            Gumbel-Softmax temperature schedule (same as BackchannelModule).
    """

    def __init__(
        self,
        lm_dim: int,
        depformer_dim: int,
        card: int,
        vap_repo_path: str,
        checkpoint_path: Optional[str] = None,
        vap_dim: int = 256,
        channel_layers: int = 1,
        cross_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        bc_hidden: int = 512,
        pad_token_id: Optional[int] = None,
        epad_token_id: Optional[int] = None,
        gumbel_temp_init: float = 1.0,
        gumbel_temp_min: float = 0.5,
        gumbel_anneal_rate: float = 0.0001,
    ):
        super().__init__()

        self.lm_dim = lm_dim
        self.depformer_dim = depformer_dim
        self.card = card
        self.vap_dim = vap_dim
        self.pad_token_id = pad_token_id if pad_token_id is not None else card - 2
        self.epad_token_id = epad_token_id if epad_token_id is not None else card - 1
        assert self.pad_token_id < card
        assert self.epad_token_id < card

        self.gumbel_temp_init = gumbel_temp_init
        self.gumbel_temp_min = gumbel_temp_min
        self.gumbel_anneal_rate = gumbel_anneal_rate

        # ── Pseudo-speaker projections ────────────────────────────────────
        # z_s carries fused context for both speakers.  Two independent linear
        # projections let the model learn to extract user-side vs. agent-side
        # features from the same hidden state.
        self.proj_user = nn.Linear(lm_dim, vap_dim, bias=False)   # User stream
        self.proj_agent = nn.Linear(lm_dim, vap_dim, bias=False)  # Agent stream

        # ── VapGPT GPT layers ─────────────────────────────────────────────
        if vap_repo_path not in sys.path:
            sys.path.insert(0, vap_repo_path)

        # Per-stream autoregressive GPT (shared weights, applied to each stream)
        self.ar_channel = GPT(
            dim=vap_dim,
            dff_k=3,
            num_layers=channel_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        # Cross-stream turn-taking GPT
        self.ar = GPTStereo(
            dim=vap_dim,
            dff_k=3,
            num_layers=cross_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ── VAP head ──────────────────────────────────────────────────────
        # 256 = 2^(2 * n_bins) with n_bins=4 (VapGPT default)
        self.vap_head = nn.Linear(vap_dim, 256)

        # ── BC gate MLP (agent-stream context → binary BC decision) ───────
        # Uses out["x2"] (agent-side output from GPTStereo) so the gate is
        # conditioned on what the agent "knows" about the conversation so far.
        self.bc_mlp = nn.Sequential(
            nn.Linear(vap_dim, bc_hidden),
            nn.ReLU(),
            nn.Linear(bc_hidden, 2),
        )

        # ── Alt 1: Silence gate MLP ───────────────────────────────────────
        # Operates on out["x"].detach() (combined VapGPT output — richer than raw z_s).
        # Detach prevents gate gradients from reaching VapGPT/Helium.
        # Trained purely by commitment loss (Alt 3) in lm.py.
        self.silence_gate_mlp = nn.Sequential(
            nn.Linear(vap_dim, bc_hidden // 2),
            nn.ReLU(),
            nn.Linear(bc_hidden // 2, 2),
        )

        # ── Load pretrained VapGPT weights (GPT layers + vap_head) ───────
        if checkpoint_path is not None:
            self._load_pretrained(checkpoint_path)

    def _load_pretrained(self, checkpoint_path: str) -> None:
        """Load ar_channel, ar, and vap_head weights from a pretrained checkpoint."""
        print(f"[VapGPTBackchannelModule] Loading pretrained checkpoint: {checkpoint_path}")
        full_sd = _load_vap_state_dict(checkpoint_path)

        # Filter to only the GPT and VAP-head keys (skip encoder.*)
        own_sd = self.state_dict()
        to_load = {}
        for k, v in full_sd.items():
            if k.startswith("ar_channel.") or k.startswith("ar.") or k.startswith("vap_head."):
                if k in own_sd and own_sd[k].shape == v.shape:
                    to_load[k] = v

        missing_in_ckpt = [k for k in own_sd if k.startswith(("ar_channel.", "ar.", "vap_head.")) and k not in to_load]
        print(f"[VapGPTBackchannelModule] Loaded {len(to_load)} GPT/vap_head tensors from checkpoint.")
        if missing_in_ckpt:
            print(f"[VapGPTBackchannelModule] {len(missing_in_ckpt)} keys not found in checkpoint (random init).")

        own_sd.update(to_load)
        self.load_state_dict(own_sd)

    def get_temperature(self, step: int) -> float:
        return compute_temperature(step, self.gumbel_temp_init, self.gumbel_temp_min, self.gumbel_anneal_rate)

    def forward(
        self,
        z_s: torch.Tensor,
        emb_cb0: nn.Module,
        step: int = 0,
    ) -> BackchannelOutput:
        """Backchannel prediction from LM hidden states.

        Args:
            z_s:     [B, T, lm_dim] LM backbone hidden states.
            emb_cb0: PAD/EPAD를 조회할 임베딩 테이블. PAD/EPAD가 텍스트 어휘에
                     속하므로 호출 측에서 depformer_text_emb를 전달해야 함.
                     bc_embeddings는 cb_index=0 입력 교체에 사용됨 (Alt 2).
            step:    training step for temperature annealing.

        Returns:
            BackchannelOutput — bc_embeddings, vap_logits, bc_gate,
                                bc_logits, silence_gate_logits.
        """
        B, T, _ = z_s.shape
        temp = self.get_temperature(step)
        device = z_s.device

        # ── Step 1: Project z_s → two pseudo-speaker streams ─────────────
        x_user = self.proj_user(z_s)   # [B, T, vap_dim]
        x_agent = self.proj_agent(z_s) # [B, T, vap_dim]

        # ── Step 2: Per-stream GPT (shared weights) ───────────────────────
        h_user = self.ar_channel(x_user)["x"]   # [B, T, vap_dim]
        h_agent = self.ar_channel(x_agent)["x"] # [B, T, vap_dim]

        # ── Step 3: Cross-stream GPTStereo ────────────────────────────────
        # out["x"]  = combined turn-taking representation (for VAP head + silence gate)
        # out["x2"] = agent-side context after cross-attention (for BC gate)
        out = self.ar(h_user, h_agent)  # {"x", "x1", "x2"}

        # ── Step 4: VAP logits ────────────────────────────────────────────
        vap_logits = self.vap_head(out["x"])  # [B, T, 256]

        # ── Step 5: BC gate (agent-side context → PAD / EPAD decision) ───
        z_bc = self.bc_mlp(out["x2"])  # [B, T, 2]
        y_bc_onehot = gumbel_softmax_st(z_bc, temperature=temp, hard=True)
        y_bc = y_bc_onehot[..., 1]    # [B, T], hard {0,1} for g_final

        # ── Alt 1: Silence gate ───────────────────────────────────────────
        s_pad_logits = self.silence_gate_mlp(out["x"].detach())  # [B, T, 2]
        s_pad_onehot = gumbel_softmax_st(s_pad_logits, temperature=temp, hard=True)
        s_pad = s_pad_onehot[..., 1]  # [B, T], hard {0,1} for g_final

        # Hard gate used at inference for discrete token replacement.
        g_final = s_pad * y_bc  # [B, T]

        # Soft gate for training embedding — product of raw softmax probs (no Gumbel, no ST).
        # Both gates contribute gradient at every timestep; no blocking from the other being 0.
        y_bc_soft = F.softmax(z_bc, dim=-1)[..., 1]          # [B, T]
        s_pad_soft = F.softmax(s_pad_logits, dim=-1)[..., 1]  # [B, T]
        g_soft = y_bc_soft * s_pad_soft                        # [B, T], always in (0,1)

        pad_ids = torch.full((1,), self.pad_token_id, device=device, dtype=torch.long)
        epad_ids = torch.full((1,), self.epad_token_id, device=device, dtype=torch.long)
        pad_emb = emb_cb0(pad_ids)    # [1, depformer_dim]
        epad_emb = emb_cb0(epad_ids)  # [1, depformer_dim]
        g_soft_exp = g_soft.unsqueeze(-1)
        bc_token_emb = g_soft_exp * epad_emb + (1.0 - g_soft_exp) * pad_emb  # [B, T, depformer_dim]
        return BackchannelOutput(
            bc_embeddings=bc_token_emb,
            vap_logits=vap_logits,
            bc_gate=g_final,
            bc_logits=z_bc,
            silence_gate_logits=s_pad_logits,
        )
