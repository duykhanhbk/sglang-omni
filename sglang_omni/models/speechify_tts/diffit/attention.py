from typing import Optional, Dict, Any, Tuple, Union
import torch
import torch.nn as nn
from typeguard import typechecked as typechecker
from jaxtyping import Float, Int, jaxtyped
import jaxtyping as jt
from transformers.cache_utils import Cache
from einops import rearrange
import math

from .rope import apply_rotary_pos_emb
from .attentions import ALL_ATTENTION_FUNCTIONS


class RMSNorm(nn.Module):
    """Minimal RMSNorm matching vLLM's ``RMSNorm`` weight layout (gamma stored
    as ``weight``, normalization in fp32, ``(1 + 0)`` gain — i.e. plain
    multiply by ``weight``). Used for optional QK-norm inside attention."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x.to(in_dtype))


def get_flash_attn_version():
    return None


def flash_attn_varlen_func(*args, **kwargs):  # pragma: no cover - sdpa path used
    raise NotImplementedError(
        "flash_attn_varlen_func is unavailable in the torch-native diffit port; "
        "the diffusion vocoder uses the SDPA attention backend."
    )

def _repeat_kv_impl(
    hidden_states: torch.Tensor,
    n_rep: int,
) -> torch.Tensor:
    """GQA key/value repeat — compile-friendly (no jaxtyping decorator)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@jaxtyped(typechecker=typechecker)
def repeat_kv(
    hidden_states: Float[torch.Tensor, "batch num_kv_heads seqlen head_dim"],
    n_rep: int,
) -> Float[torch.Tensor, "batch num_attn_heads seqlen head_dim"]:
    """
    Repeats the key and value hidden states to match the number of query heads.

    This function is used in Grouped-Query Attention (GQA) to expand the key and value
    tensors from the number of key-value heads to the number of attention heads. It's
    an efficient equivalent of `torch.repeat_interleave(x, dim=1, repeats=n_rep)`.

    Args:
        hidden_states: The key or value tensor to be repeated, with shape
                       (batch, num_key_value_heads, seqlen, head_dim).
        n_rep: The number of times to repeat each key-value head (kv_groups).

    Returns:
        The expanded tensor with shape (batch, num_attention_heads, seqlen, head_dim).
    """
    return _repeat_kv_impl(hidden_states, n_rep)


class SelfAttention(nn.Module):
    """
    Multi-Head or Grouped-Query Self-Attention layer.

    This module implements the scaled dot-product attention mechanism, with support
    for Grouped-Query Attention (GQA), Rotary Position Embeddings (RoPE),
    optional key/query normalization, and various attention implementations like
    Flash Attention.

    Args:
        hidden_size: The dimensionality of the input and output hidden states.
        num_attention_heads: The number of query heads.
        num_kv_heads: The number of key and value heads. For GQA, this is less than
                      `num_attention_heads`. For MHA, this is equal.
        query_pre_attn_scalar: A scalar value used to scale the query tensor before
                               the attention mechanism.
        head_dim: The dimensionality of each query head.
        kv_head_dim: The dimensionality of each key/value head.
        layer_idx: The index of the current layer, used for KV caching.
        attention_dropout: Dropout probability for the attention scores.
        attn_logit_softcapping: A value for softcapping the attention logits.
        is_causal: If True, applies a causal mask to the attention scores.
        qk_norm: If True, applies RMSNorm to query and key tensors before attention.
        attn_implementation: The specific attention function to use (e.g., "sdpa",
                             "flash_attention_2").
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        query_pre_attn_scalar: float,
        head_dim: int,
        kv_head_dim: int,
        layer_idx: int,
        attention_dropout: float = 0.0,
        attn_logit_softcapping: float = 0.0,
        is_causal: bool = True,
        qk_norm: bool = False,
        attn_gate: bool = False,
        attn_implementation: str = "flash_attention_2",
        bias: bool = False,
        sliding_window: Optional[Tuple[int, int]] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_head_dim = kv_head_dim
        self.kv_groups = num_attention_heads // num_kv_heads
        self.attention_dropout = attention_dropout
        self.dropout = attention_dropout
        self.scaling = query_pre_attn_scalar**-0.5
        self.attn_logit_softcapping = attn_logit_softcapping
        self.is_causal = is_causal  # will be used in attention interface
        self.qk_norm = qk_norm
        self.attn_gate = attn_gate
        self.attn_implementation = attn_implementation
        self.layer_idx = layer_idx
        self.sliding_window = sliding_window
        self.alibi_slopes = alibi_slopes
        self.fa_version = get_flash_attn_version()

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.kv_head_dim * (2 if self.attn_gate else 1),
            bias=bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.kv_head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.kv_head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.kv_head_dim, self.hidden_size, bias=bias
        )

        if self.qk_norm:
            self.q_norm = RMSNorm(hidden_size=self.head_dim, eps=1e-5)
            self.k_norm = RMSNorm(hidden_size=self.kv_head_dim, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        max_seqlen: int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, None]:
        """Packed (no-padding) forward for encoder self-attention.

        Args:
            hidden_states: [total_tokens, hidden_size]
            query_start_loc: [num_seqs + 1] int32 cumulative sequence lengths
            max_seqlen: int, maximum sequence length in batch
            position_embeddings: optional (cos, sin) each [1, total_tokens, head_dim],
                                  None when using NoPE (default encoder config)

        Returns:
            ([total_tokens, hidden_size], None)
        """
        total_tokens = hidden_states.shape[0]

        # QKV projections: [total_tokens, hidden] -> [total_tokens, num_heads, head_dim]
        query_states = self.q_proj(hidden_states).view(total_tokens, self.num_attention_heads, self.kv_head_dim)
        key_states   = self.k_proj(hidden_states).view(total_tokens, self.num_kv_heads, self.kv_head_dim)
        value_states = self.v_proj(hidden_states).view(total_tokens, self.num_kv_heads, self.kv_head_dim)

        if self.qk_norm:
            query_states = self.q_norm(query_states)
            key_states   = self.k_norm(key_states)

        if position_embeddings is not None:
            cos, sin = position_embeddings  # [1, total_tokens, head_dim]
            # apply_rotary_pos_emb expects [batch, num_heads, seq_len, head_dim]
            q_4d = query_states.unsqueeze(0).transpose(1, 2)   # [1, nH, total, head_dim]
            k_4d = key_states.unsqueeze(0).transpose(1, 2)
            q_4d, k_4d = apply_rotary_pos_emb(q_4d, k_4d, cos, sin)
            query_states = q_4d.squeeze(0).transpose(0, 1)     # [total, nH, head_dim]
            key_states   = k_4d.squeeze(0).transpose(0, 1)

        # GQA repeat
        if self.kv_groups > 1:
            key_states   = key_states.repeat_interleave(self.kv_groups, dim=1)
            value_states = value_states.repeat_interleave(self.kv_groups, dim=1)

        flash_kwargs = {"fa_version": self.fa_version}
        if self.attn_logit_softcapping > 0:
            flash_kwargs["softcap"] = self.attn_logit_softcapping
        attn_output = flash_attn_varlen_func(
            query_states, key_states, value_states,
            cu_seqlens_q=query_start_loc,
            cu_seqlens_k=query_start_loc,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.scaling,
            causal=False,
            **flash_kwargs,
        )  # [total_tokens, num_heads, head_dim]

        attn_output = attn_output.reshape(total_tokens, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    def forward_naive(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.Tensor] = None,
        cond_q: Optional[torch.Tensor] = None,
        cond_k: Optional[torch.Tensor] = None,
        cond_v: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        """Batched forward for non-vLLM contexts (e.g. diffusion self-attention).

        Uses the standard HF attention interface with attention_mask, RoPE,
        and optional KV cache, unlike the packed vLLM forward() above.
        """
        batch_size, seq_len = hidden_states.shape[:-1]

        query_shape = (batch_size, -1, self.num_attention_heads, self.kv_head_dim)
        kv_shape = (batch_size, -1, self.num_kv_heads, self.kv_head_dim)

        if self.attn_gate:
            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(
                    *query_shape[:-1], query_shape[-1] * 2
                ),
                2,
                dim=-1,
            )
            gate = gate.reshape(batch_size, seq_len, -1)
        else:
            query_states = self.q_proj(hidden_states)

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Apply optional conditioning (additive bias to Q/K/V before reshape)
        if cond_q is not None:
            query_states = query_states + cond_q
        if cond_k is not None:
            key_states = key_states + cond_k
        if cond_v is not None:
            value_states = value_states + cond_v

        query_states = query_states.view(query_shape).transpose(1, 2)
        key_states = key_states.view(kv_shape).transpose(1, 2)
        value_states = value_states.view(kv_shape).transpose(1, 2)

        if self.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # Length of cached history (committed frames from prior chunks)
        # BEFORE this step's update — needed to build the streaming
        # bottom-right causal mask below.
        streaming_base_len = 0
        is_streaming = past_key_values is not None
        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            if position_embeddings is not None:
                cos, sin = position_embeddings
                cache_kwargs.update({"sin": sin, "cos": cos})

            streaming_base_len = past_key_values.get_seq_length(self.layer_idx)

            cl = kwargs.get("commit_len")
            if cl is not None and cl < seq_len:
                # Only commit the prefix to persistent cache; keep the
                # uncommitted tail for this step's attention only.
                committed_k = key_states[:, :, :cl, :]
                committed_v = value_states[:, :, :cl, :]
                tail_k = key_states[:, :, cl:, :]
                tail_v = value_states[:, :, cl:, :]

                committed_cache_kwargs = dict(cache_kwargs)
                if cache_position is not None:
                    committed_cache_kwargs["cache_position"] = cache_position[:cl]

                cached_k, cached_v = past_key_values.update(
                    committed_k, committed_v, self.layer_idx, committed_cache_kwargs
                )
                key_states = torch.cat([cached_k, tail_k], dim=2)
                value_states = torch.cat([cached_v, tail_v], dim=2)
            else:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

        key_states = repeat_kv(key_states, self.kv_groups)
        value_states = repeat_kv(value_states, self.kv_groups)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.attn_implementation]
        sliding_window_override = kwargs.pop("sliding_window_override", None)
        effective_sliding_window = (
            self.sliding_window if sliding_window_override is None else sliding_window_override
        )
        effective_is_causal = self.is_causal and effective_sliding_window is None

        if is_streaming and self.is_causal:
            # Streaming KV-cache self-attention. ``key_states`` is laid out
            # as ``[history(base) ; window(seq_len)]`` with every position
            # carrying real data (DynamicCache holds no garbage), so the
            # logical key index equals the array index ``j`` and query
            # ``i`` lives at logical position ``base + i``. PyTorch's
            # ``is_causal=True`` aligns top-left (wrong when q_len != k_len),
            # so we build the bottom-right causal mask explicitly, widened
            # by ``sw_future`` for lookahead layers (DiffiTv3 blocks 0/9
            # have sliding_window=(-1, 32); 0 elsewhere). This mirrors the
            # vllm-omni static-step mask without the static-slab garbage
            # bookkeeping.
            S_total = key_states.shape[2]
            base = streaming_base_len
            if effective_sliding_window is not None:
                left, sw_future = effective_sliding_window
            else:
                left, sw_future = -1, 0
            dtype_min = torch.finfo(query_states.dtype).min
            j = torch.arange(S_total, device=query_states.device)
            i = torch.arange(seq_len, device=query_states.device)
            logical_q = (base + i)[:, None]            # [Q, 1]
            allowed = j[None, :] <= (logical_q + sw_future)  # [Q, S]
            if left is not None and left >= 0:
                allowed = allowed & (j[None, :] >= (logical_q - left))
            stream_mask = torch.where(
                allowed,
                torch.zeros((), dtype=query_states.dtype, device=query_states.device),
                torch.full((), dtype_min, dtype=query_states.dtype, device=query_states.device),
            )[None, None, :, :]                        # [1, 1, Q, S]
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                stream_mask,
                dropout=0.0,
                scaling=self.scaling,
                is_causal=False,
                softcap=self.attn_logit_softcapping,
                **kwargs,
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.scaling,
                is_causal=effective_is_causal,
                softcap=self.attn_logit_softcapping,
                **kwargs,
            )

        attn_output = attn_output.reshape(batch_size, seq_len, -1).contiguous()
        if self.attn_gate:
            attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class CrossAttention(nn.Module):
    """Multi-Head or Grouped-Query Cross-Attention layer.

    This module implements the scaled dot-product attention mechanism with
    support for Grouped-Query Attention (GQA), optional key/query
    normalization, and various attention implementations like Flash Attention.

    This also specifically supports cross-attention alignment bias for
    Text-to-Speech models.

    Args:
        hidden_size: The dimensionality of the input and output hidden states.
        num_attention_heads: The number of query heads.
        num_kv_heads: The number of key and value heads. For GQA, this is
                      less than `num_attention_heads`. For MHA, this is equal.
        query_pre_attn_scalar: A scalar value used to scale the query tensor
                               before the attention mechanism.
        head_dim: The dimensionality of each query head.
        kv_head_dim: The dimensionality of each key/value head.
        layer_idx: The index of the current layer, used for KV caching.
        attention_dropout: Dropout probability for the attention scores.
        attn_logit_softcapping: A value for softcapping the attention logits.
        qk_norm: If True, applies RMSNorm to query and key tensors before attention.
        attn_implementation: The specific attention function to use (e.g.,
                             "sdpa", "flash_attention_2"). Defaults to
                             "flash_attention_2".
        use_alignment_bias: Whether to use alignment attention bias.
        relative_attention_num_buckets: Number of buckets for alignment bias.
        relative_attention_max_distance: Max distance for alignment bias.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        query_pre_attn_scalar: float,
        head_dim: int,
        kv_head_dim: int,
        layer_idx: int,
        kv_dim: int | None = None,
        attention_dropout: float = 0.0,
        attn_logit_softcapping: float = 0.0,
        qk_norm: bool = False,
        attn_gate: bool = False,
        attn_implementation: str = "flash_attention_2",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_head_dim = kv_head_dim
        self.kv_groups = num_attention_heads // num_kv_heads
        self.attention_dropout = attention_dropout
        self.dropout = attention_dropout
        self.scaling = query_pre_attn_scalar**-0.5
        self.is_causal = False
        self.qk_norm = qk_norm
        self.attn_gate = attn_gate
        self.layer_idx = layer_idx
        self.kv_dim = kv_dim if kv_dim is not None else hidden_size

        # If alignment bias is used, it requires adding a bias tensor to the
        # attention scores, which is not supported by FlashAttention.
        # Therefore, we fall back to the PyTorch's native SDPA.
        # The attn_logit_softcapping is also disabled in this case as SDPA not support it.
        self.attn_implementation = attn_implementation
        self.attn_logit_softcapping = (
            attn_logit_softcapping if self.attn_implementation != "sdpa" else 0.0
        )

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.kv_head_dim * (2 if self.attn_gate else 1),
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.kv_dim, self.num_kv_heads * self.kv_head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.kv_dim, self.num_kv_heads * self.kv_head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.kv_head_dim, self.hidden_size, bias=False
        )

        if self.qk_norm:
            self.q_norm = RMSNorm(hidden_size=self.head_dim, eps=1e-5)
            self.k_norm = RMSNorm(hidden_size=self.kv_head_dim, eps=1e-5)

    def project_kv(
        self,
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project encoder hidden states to K,V and apply K norm if needed.

        Returns (key_states, value_states) in [batch, num_kv_heads, enc_len, head_dim] format,
        ready to be passed as cached_key_states/cached_value_states to forward().
        """
        encoder_input_shape = encoder_hidden_states.shape[:-1]
        encoder_hidden_shape = (*encoder_input_shape, -1, self.kv_head_dim)
        key_states = (
            self.k_proj(encoder_hidden_states)
            .view(encoder_hidden_shape)
            .transpose(1, 2)
            .contiguous()
        )
        value_states = (
            self.v_proj(encoder_hidden_states)
            .view(encoder_hidden_shape)
            .transpose(1, 2)
            .contiguous()
        )
        if self.qk_norm:
            key_states = self.k_norm(key_states).contiguous()
        return key_states, value_states

    def make_cross_attention_mask(
        self,
        query_states: Float[torch.Tensor, "batch query_len dim"],
        encoder_attention_mask: Int[torch.Tensor, "batch key_len"],
    ) -> Float[torch.Tensor, "batch 1 query_len key_len"]:
        """Creates an attention mask for cross-attention from an encoder mask."""
        if self.attn_implementation == "sdpa":
            # Expand mask to [batch, 1, 1, key_len] for broadcasting
            bidirectional_mask = encoder_attention_mask[:, None, None, :]
            # Convert to additive mask with large negative values for padded positions
            additive_bidirectional_mask = (
                1.0 - bidirectional_mask.to(query_states.dtype)
            ) * torch.finfo(query_states.dtype).min
            return additive_bidirectional_mask
        else:
            # For flash-attention-2, the cross-attention accept [B, T] with 1 is non-mask, 0 is mask
            return encoder_attention_mask

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        hidden_states: Float[torch.Tensor, "batch seq_len hidden_size"],
        encoder_hidden_states: Float[torch.Tensor, "batch enc_seq_len enc_hidden_size"] | None = None,
        encoder_attention_mask: Int[torch.Tensor, "batch enc_seq_len"] | None = None,
        past_key_values: Cache | None = None,
        # if cross_attention_bias is provided, it must also take into account the encoder_attention_mask
        cross_attention_bias: (
            Float[torch.Tensor, "batch n_kv_heads seq_len enc_seq_len"] | None
        ) = None,
        return_cross_attentions: Optional[bool] = False,
        cached_key_states: torch.Tensor | None = None,
        cached_value_states: torch.Tensor | None = None,
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, "batch seq_len hidden_size"],
        Tuple[
            Float[torch.Tensor, "batch seq_len hidden_size"],
            Optional[Float[torch.Tensor, "batch num_heads seq_len enc_seq_len"]] | None,
        ],
    ]:
        batch_size, seq_len = hidden_states.shape[:-1]
        query_shape = (batch_size, -1, self.num_attention_heads, self.kv_head_dim)

        if self.attn_gate:
            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(*query_shape[:-1], query_shape[-1] * 2), 2, dim=-1
            )
            gate = gate.reshape(batch_size, seq_len, -1)
        else:
            query_states = self.q_proj(hidden_states)
        query_states = query_states.view(query_shape).transpose(1, 2)

        # K,V: use pre-projected cache, past_key_values, or project fresh
        if cached_key_states is not None:
            key_states = cached_key_states
            value_states = cached_value_states
            kv_already_normed = True
        elif past_key_values is not None and past_key_values.get_seq_length(self.layer_idx) > 0:
            key_states = past_key_values.layers[self.layer_idx].keys
            value_states = past_key_values.layers[self.layer_idx].values
            kv_already_normed = False
        else:
            encoder_input_shape = encoder_hidden_states.shape[:-1]
            encoder_hidden_shape = (*encoder_input_shape, -1, self.kv_head_dim)
            key_states = (
                self.k_proj(encoder_hidden_states)
                .view(encoder_hidden_shape)
                .transpose(1, 2)
            )
            value_states = (
                self.v_proj(encoder_hidden_states)
                .view(encoder_hidden_shape)
                .transpose(1, 2)
            )
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx
                )
            kv_already_normed = False

        if self.qk_norm:
            query_states = self.q_norm(query_states)
            if not kv_already_normed:
                key_states = self.k_norm(key_states)

        # Repeat keys/values if using different number of heads for queries and keys/values
        key_states = repeat_kv(key_states, self.kv_groups)
        value_states = repeat_kv(value_states, self.kv_groups)

        # Cross attention is not well supported by flash attention,
        # especially for key padding mask and custom attention bias.
        # Note: softcap is not supported in sdpa_attention_forward.
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.attn_implementation]
        if cross_attention_bias is not None:
            cross_attention_mask = cross_attention_bias
        elif encoder_attention_mask is not None:
            cross_attention_mask = self.make_cross_attention_mask(
                hidden_states, encoder_attention_mask
            )
        else:
            cross_attention_mask = None

        if cross_attention_mask is not None and self.attn_implementation == "sdpa":
            key_length = key_states.shape[2]
            query_length = hidden_states.shape[1]
            cross_attention_mask = cross_attention_mask[
                :, :, -query_length:, :key_length
            ]
            cross_attention_mask = repeat_kv(cross_attention_mask, self.kv_groups)

        attn_output = attention_interface(
            module=self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=cross_attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            softcap=self.attn_logit_softcapping,
            return_cross_attentions=return_cross_attentions,
            **kwargs,
        )
        if return_cross_attentions:
            attn_output, attn_weights = attn_output
        elif type(attn_output) is tuple:
            attn_output = attn_output[0]

        attn_output = attn_output.reshape(batch_size, seq_len, -1).contiguous()
        if self.attn_gate:
            attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        if return_cross_attentions:
            return attn_output, attn_weights
        else:
            return attn_output

    def forward_stream(
        self,
        hidden_states: torch.Tensor,
        cached_key_states: torch.Tensor,
        cached_value_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compile-friendly cross-attention with pre-projected K/V.

        No jaxtyping, no Optional branches, no past_key_values / DynamicCache.

        Args:
            hidden_states: ``[B, T, hidden_size]``
            cached_key_states: ``[B, num_kv_heads, S, kv_head_dim]`` pre-projected keys.
            cached_value_states: ``[B, num_kv_heads, S, kv_head_dim]`` pre-projected values.
            encoder_attention_mask: ``[B, S]`` 0/1 mask.

        Returns:
            ``[B, T, hidden_size]``
        """
        batch_size, seq_len = hidden_states.shape[:2]

        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(batch_size, -1, self.num_attention_heads, self.kv_head_dim).transpose(1, 2)

        key_states = _repeat_kv_impl(cached_key_states, self.kv_groups)
        value_states = _repeat_kv_impl(cached_value_states, self.kv_groups)

        if self.qk_norm:
            query_states = self.q_norm(query_states)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.attn_implementation]
        cross_attention_mask = self.make_cross_attention_mask(
            hidden_states, encoder_attention_mask
        )

        attn_output = attention_interface(
            module=self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=cross_attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            softcap=self.attn_logit_softcapping,
        )
        if type(attn_output) is tuple:
            attn_output = attn_output[0]

        attn_output = attn_output.reshape(batch_size, seq_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


def test_self_attn():
    from speechify_tts.layers.rope import RotaryEmbedding

    positional_embedding = (
        RotaryEmbedding(
            rope_type="default",
            max_position_embeddings=2048,
            rope_config={
                "rope_theta": 100000,
                "hidden_size": 768,
                "num_attention_heads": 8,
                "partial_rotary_factor": 1.0,
            },
        )
        .cuda()
        .to(torch.bfloat16)
    )
    self_attention = (
        SelfAttention(
            hidden_size=768,
            num_attention_heads=8,
            num_kv_heads=2,
            query_pre_attn_scalar=1.0,
            head_dim=96,
            kv_head_dim=96,  # MUST equal to hidden_size // num_attention_heads for RoPE models
            attention_dropout=0.0,
            attn_logit_softcapping=50.0,
            is_causal=True,
            qk_norm=True,
            attn_implementation="flash_attention_2",
        )
        .cuda()
        .to(torch.bfloat16)
    )

    x = torch.randn(1, 720, 768).cuda().to(torch.bfloat16)
    attention_mask = torch.ones(1, 720).cuda().to(torch.int32)
    position_ids = torch.arange(0, 720).unsqueeze(0).cuda().to(torch.int32)
    cos, sin = positional_embedding(x, position_ids)
    out, _ = self_attention(x, (cos, sin), attention_mask=attention_mask)
    assert out.shape == (1, 720, 768), f"unexpected shape {out.shape}"


def test_cross_attn():
    cross_attention = (
        CrossAttention(
            hidden_size=768,
            num_attention_heads=8,
            num_kv_heads=2,
            query_pre_attn_scalar=1.0,
            head_dim=96,
            kv_head_dim=24,
            attention_dropout=0.0,
            attn_logit_softcapping=50.0,
            qk_norm=False,
            attn_implementation="flash_attention_2",
        )
        .cuda()
        .to(torch.bfloat16)
    )

    x = torch.randn(1, 720, 768).cuda().to(torch.bfloat16)

    encoder_hidden_states = torch.randn(1, 512, 768).cuda().to(torch.bfloat16)
    encoder_attention_mask = torch.ones(1, 512).cuda().to(torch.int32)
    cross_alignments = torch.ones(1, 720).cuda().to(torch.float32)

    out, _ = cross_attention(
        x,
        encoder_hidden_states,
        encoder_attention_mask,
        cross_alignment=cross_alignments,
    )
    assert out.shape == (1, 720, 768), f"unexpected shape {out.shape}"


class T5RelativeAttentionBias(nn.Module):
    def __init__(
        self,
        relative_attention_num_buckets: int,
        relative_attention_max_distance: int,
        n_heads: int,
    ):
        super().__init__()

        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.n_heads = n_heads

        self.relative_attention_bias = nn.Embedding(
            relative_attention_num_buckets, n_heads
        )

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> Int[torch.Tensor, "..."]:
        """Calculates relative position buckets, adapted from T5's implementation.

        This function maps continuous relative position values into a discrete
        set of buckets, which is useful for creating position-aware attention biases.
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )

        # Values within half of the buckets are mapped directly.
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Values beyond that are mapped logarithmically to the remaining buckets.
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position.to(torch.long), relative_position_if_large
        )
        return relative_buckets

    @staticmethod
    def _make_additive_cross_attention_mask(
        encoder_attention_mask: Int[torch.Tensor, "batch key_len"],  #  0/1 values
        dtype,
    ) -> Float[torch.Tensor, "batch 1 1 key_len"]:
        """Creates an attention mask for cross-attention from an encoder mask."""
        # Expand mask to [batch, 1, 1, key_len] for broadcasting
        bidirectional_mask = encoder_attention_mask[:, None, None, :]
        # Convert to additive mask with large negative values for padded positions
        additive_bidirectional_mask = (
            1.0 - bidirectional_mask.to(dtype)
        ) * torch.finfo(dtype).min
        return additive_bidirectional_mask

    @jaxtyped(typechecker=typechecker)
    def compute_bias(
        self, query_length: int, key_length: int, device=None, cache_position=None
    ) -> jt.Float[torch.Tensor, "batch n_kv_heads query_len key_len"]:
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        if cache_position is None:
            context_position = torch.arange(
                query_length, dtype=torch.long, device=device
            )[:, None]
        else:
            context_position = cache_position[:, None].to(device)
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
            None, :
        ]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    # Distance range covered by the precomputed alignment-bias table:
    # d = memory_pos - trunc(align) for encoder lengths up to this value.
    _ALIGN_BIAS_TABLE_RANGE = 1024

    def _alignment_bias_table(self, device: torch.device) -> torch.Tensor:
        """[2R, n_heads] table with row r = embedding(bucket(r - R)).

        ``compute_bias_for_cross_alignment`` casts the alignment to int64
        before the bucket math, so the bias is a pure function of the
        integer distance ``d = memory_pos - trunc(align)``.  Precomputing
        ``emb(bucket(d))`` collapses the per-step bucket math (~12 tiny
        kernels inside the decode CUDA graph) into one gather.  Built
        lazily after weights load; weights are frozen at inference.
        """
        tbl = getattr(self, "_align_bias_tbl", None)
        if tbl is None or tbl.device != device:
            R = self._ALIGN_BIAS_TABLE_RANGE
            d = torch.arange(-R, R, dtype=torch.long, device=device)
            buckets = self._relative_position_bucket(
                d,
                bidirectional=True,
                num_buckets=self.relative_attention_num_buckets,
                max_distance=self.relative_attention_max_distance,
            ).clamp(0, self.relative_attention_num_buckets - 1)
            tbl = self.relative_attention_bias.weight.index_select(0, buckets)
            self._align_bias_tbl = tbl
        return tbl

    @jaxtyped(typechecker=typechecker)
    def compute_bias_for_cross_alignment(
        self,
        alignment: Float[torch.Tensor, "batch query_len"],
        encoder_attention_mask: Int[torch.Tensor, "batch key_len"],
    ) -> jt.Float[torch.Tensor, "batch n_kv_heads query_len key_len"]:
        """Computes the binned relative position bias for cross-attention.

        Runs on the AR decode hot path inside the FULL CUDA graph every
        step, so it gathers from the precomputed distance table instead of
        re-running the bucket math (identical values — the original code
        casts the alignment to int64 before bucketing, see
        _alignment_bias_table).  Falls back to direct computation for
        encoder lengths beyond the table range.
        """
        device = alignment.device
        key_length = encoder_attention_mask.shape[1]
        bsz, query_len = alignment.shape
        R = self._ALIGN_BIAS_TABLE_RANGE

        if key_length <= R:
            tbl = self._alignment_bias_table(device)  # [2R, n_heads]
            memory_position = torch.arange(
                key_length, dtype=torch.long, device=device
            ).view(1, 1, key_length)
            # trunc-toward-zero cast, matching the original
            # ``context_position.to(memory_position.dtype)``.
            align_long = alignment.to(torch.long)[:, :, None]
            # Clamp keeps NaN-derived garbage indices in range during CUDA
            # graph capture (same protection as the original bucket clamp).
            idx = (memory_position - align_long + R).clamp(0, 2 * R - 1)
            bias_values = tbl.index_select(0, idx.view(-1)).view(
                bsz, query_len, key_length, self.n_heads
            )
        else:
            # Fallback: direct bucket math (original implementation).
            memory_position = torch.arange(
                key_length, dtype=torch.long, device=device
            ).view(1, 1, key_length)
            relative_position = memory_position - alignment[:, :, None].to(
                memory_position.dtype
            )
            relative_position_bucket = self._relative_position_bucket(
                relative_position,
                bidirectional=True,
                num_buckets=self.relative_attention_num_buckets,
                max_distance=self.relative_attention_max_distance,
            ).clamp(0, self.relative_attention_num_buckets - 1)
            # Using index_select is found to be faster than a direct
            # embedding lookup. (Found by @minh)
            bias_values = self.relative_attention_bias.weight.index_select(
                0, relative_position_bucket.view(-1)
            ).view(bsz, query_len, key_length, self.n_heads)

        # [batch, n_kv_heads, query_len, key_len]
        bias_values = rearrange(bias_values, "b q k h -> b h q k")

        additive_cross_attention_mask = self._make_additive_cross_attention_mask(
            encoder_attention_mask, dtype=bias_values.dtype
        )
        bias_values = bias_values + additive_cross_attention_mask

        return bias_values


if __name__ == "__main__":
    test_self_attn()
    test_cross_attn()
