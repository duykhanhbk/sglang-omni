# SPDX-License-Identifier: Apache-2.0
"""Torch-native building blocks for the SpeechifyTTS port.

These are framework-free re-implementations of the shared ``speechify`` layers
that vllm-omni couples to vLLM internals (EncoderOnlyAttention, the Triton
varlen depthwise-conv custom op, ``fused_experts``, ``support_torch_compile``).
Keeping them dependency-light lets the encoder / decoder / vocoder run inside a
self-contained decode loop while still loading the *exact* converted
``simba3-moe4b-vllm-v5-streaming`` safetensors (same parameter names).

Numerical-equivalence notes vs. the vllm-omni layers:

- :class:`RMSNorm` matches ``vllm.model_executor.layers.layernorm.RMSNorm``
  (normalize in fp32, scale by ``weight``, cast back).
- :class:`SwiGLUMLP` matches ``speechify.mlp.MLP`` eval path
  (``down_proj(silu(gate_proj(x)) * up_proj(x))``).
- :class:`ConformerConv` matches ``speechify.cnn.Conformer``: the Triton varlen
  causal depthwise conv is exactly a left-padded (K-1) depthwise ``conv1d`` for a
  single sequence, with SiLU fused after the depthwise stage.
- :class:`Attention` is a plain SDPA wrapper; the encoder uses it bidirectionally
  and the decoder uses it causally / for cross-attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Standard RMSNorm (matches vLLM's non-fused RMSNorm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x * self.weight.float()
        return x.to(orig_dtype)


class SwiGLUMLP(nn.Module):
    """Gated MLP: ``down_proj(act(gate_proj(x)) * up_proj(x))``.

    ``activation_fn`` mirrors ``speechify.mlp.MLP``: ``swiglu`` -> SiLU gate,
    ``geglu`` -> tanh-approx GELU gate. ``relu``/``silu`` are non-gated.
    """

    def __init__(
        self,
        dim: int,
        mult: int = 4,
        activation_fn: str = "swiglu",
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        self.activation_fn = activation_fn
        self.gated = activation_fn in ("swiglu", "geglu")
        if self.gated:
            self.gate_proj = nn.Linear(dim, inner_dim, bias=False)
        self.up_proj = nn.Linear(dim, inner_dim, bias=False)
        self.down_proj = nn.Linear(inner_dim, dim, bias=False)
        if activation_fn in ("swiglu", "silu"):
            self.act_fn = F.silu
        elif activation_fn == "geglu":
            self.act_fn = lambda t: F.gelu(t, approximate="tanh")
        elif activation_fn == "relu":
            self.act_fn = F.relu
        else:
            raise ValueError(f"Unsupported activation_fn: {activation_fn!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(self.act_fn(self.up_proj(x)))


class _GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


def _calc_same_padding(kernel_size: int) -> tuple[int, int]:
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class _DepthWiseConv1d(nn.Module):
    """Mirrors ``speechify.cnn.DepthWiseConv1d`` (wraps inner ``conv``)."""

    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        kernel_size: int,
        padding: tuple[int, int],
    ) -> None:
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, self.padding))


class ConformerConv(nn.Module):
    """Conformer conv module (parameter-name compatible with vllm-omni).

    Sequence of ops on ``[T, hidden]``:
      pointwise_conv1 (1x1) -> GLU (channel) -> causal depthwise conv (+SiLU)
      -> pointwise_conv2 (1x1).

    A per-request ``conv_state`` of the last ``K-1`` pre-conv frames may be
    passed for streaming/AR-incremental decoding; otherwise a single full
    sequence is left-padded (causal) or center-padded (non-causal).
    """

    def __init__(
        self,
        dim: int,
        is_causal: bool = True,
        expansion_factor: int = 2,
        kernel_size: int = 31,
        activation: str = "swish",
    ) -> None:
        super().__init__()
        inner_dim = dim * expansion_factor
        self.kernel_size = kernel_size
        self.is_causal = is_causal
        self.inner_dim = inner_dim
        self.padding = (0, 0) if is_causal else _calc_same_padding(kernel_size)
        self.pointwise_conv1 = nn.Conv1d(dim, inner_dim * 2, 1)
        self.glu = _GLU(dim=1)
        self.depthwise_conv = _DepthWiseConv1d(
            inner_dim, inner_dim, kernel_size=kernel_size, padding=self.padding
        )
        self.fuse_silu = activation == "swish"
        self.pointwise_conv2 = nn.Conv1d(inner_dim, dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Args: ``x`` ``[T, hidden]``. Returns ``([T, hidden], new_conv_state)``.

        When ``is_causal`` and a ``conv_state`` (``[inner_dim, K-1]``) is given,
        the depthwise conv is primed with the cached tail so AR steps stay
        consistent with a full-sequence prefill.
        """
        x_conv = x.t().unsqueeze(0)  # [1, hidden, T]
        x_conv = self.pointwise_conv1(x_conv)  # [1, inner_dim*2, T]
        x_conv = self.glu(x_conv)  # [1, inner_dim, T]

        new_state: torch.Tensor | None = None
        k = self.kernel_size
        if self.is_causal:
            if conv_state is None:
                left = F.pad(x_conv, (k - 1, 0))
            else:
                left = torch.cat([conv_state.unsqueeze(0), x_conv], dim=2)
            if k > 1:
                new_state = left[:, :, -(k - 1):].squeeze(0).detach()
            x_conv = self.depthwise_conv.conv(left)
        else:
            x_conv = self.depthwise_conv(x_conv)

        if self.fuse_silu:
            x_conv = F.silu(x_conv)

        x_conv = self.pointwise_conv2(x_conv)  # [1, hidden, T]
        return x_conv.squeeze(0).t(), new_state


class Attention(nn.Module):
    """Multi-head attention via SDPA (parameter-name compatible).

    Supports self-attention (``kv_dim is None``) and cross-attention. Optional
    per-head RMSNorm (``qk_norm``) and an additive attention bias (T5 relative
    bias / cross-attention alignment bias).
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_head_dim: int,
        query_pre_attn_scalar: float,
        kv_dim: int | None = None,
        qk_norm: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_head_dim = kv_head_dim
        self.scaling = query_pre_attn_scalar ** -0.5
        self.qk_norm = qk_norm
        kv_in = hidden_size if kv_dim is None else kv_dim
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * kv_head_dim, bias=False)
        self.k_proj = nn.Linear(kv_in, num_kv_heads * kv_head_dim, bias=False)
        self.v_proj = nn.Linear(kv_in, num_kv_heads * kv_head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * kv_head_dim, hidden_size, bias=False)
        if qk_norm:
            self.q_norm = RMSNorm(kv_head_dim, eps=eps)
            self.k_norm = RMSNorm(kv_head_dim, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_states: torch.Tensor | None = None,
        is_causal: bool = False,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``hidden_states`` ``[Tq, hidden]``, ``kv_states`` ``[Tk, kv_dim]``.

        Returns ``[Tq, hidden]``. ``attn_bias`` broadcasts to
        ``[heads, Tq, Tk]`` and is added to the scores (used for T5 relative
        bias / cross-alignment bias).
        """
        kv_src = hidden_states if kv_states is None else kv_states
        tq = hidden_states.shape[0]
        tk = kv_src.shape[0]
        q = self.q_proj(hidden_states).view(tq, self.num_attention_heads, self.kv_head_dim)
        k = self.k_proj(kv_src).view(tk, self.num_kv_heads, self.kv_head_dim)
        v = self.v_proj(kv_src).view(tk, self.num_kv_heads, self.kv_head_dim)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # [heads, T, hd]
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        if self.num_kv_heads != self.num_attention_heads:
            rep = self.num_attention_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=0)
            v = v.repeat_interleave(rep, dim=0)
        attn_mask = attn_bias
        if attn_bias is not None and attn_bias.dtype != q.dtype:
            attn_mask = attn_bias.to(q.dtype)
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            attn_mask=None if attn_mask is None else attn_mask.unsqueeze(0),
            is_causal=is_causal and attn_mask is None,
            scale=self.scaling,
        ).squeeze(0)  # [heads, Tq, hd]
        out = out.transpose(0, 1).reshape(tq, self.num_attention_heads * self.kv_head_dim)
        return self.o_proj(out)
