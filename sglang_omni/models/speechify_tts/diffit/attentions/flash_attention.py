"""Flash attention backend using vLLM's built-in flash_attn_varlen_func.

No dependency on the external ``flash-attn`` package — vLLM bundles its own
flash attention implementation that this module delegates to.
"""

from typing import Optional

import torch
import torch.nn.functional as F

try:
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_flash_attn_version,
    )

    _FLASH_ATTN_VER = get_flash_attn_version()
except Exception:  # pragma: no cover - torch-native diffit uses SDPA
    flash_attn_varlen_func = None
    _FLASH_ATTN_VER = None

    def get_flash_attn_version():
        return None

# ---------------------------------------------------------------------------
# Unpad / pad helpers (pure-PyTorch, no external deps)
# ---------------------------------------------------------------------------

def _unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Remove padding tokens from a batched tensor.

    Args:
        hidden_states: [batch, seq_len, ...]
        attention_mask: [batch, seq_len] with 1=valid, 0=padding

    Returns:
        (unpadded, indices, cu_seqlens, max_seqlen)
        - unpadded: [total_valid_tokens, ...]
        - indices: [total_valid_tokens] flat indices into the original tensor
        - cu_seqlens: [batch+1] cumulative sequence lengths (int32)
        - max_seqlen: int
    """
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = int(seqlens.max())
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    # Flatten first two dims then index
    flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    unpadded = flat[indices]
    return unpadded, indices, cu_seqlens, max_seqlen


def _pad_input(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Re-pad unpadded tokens back into a batched tensor.

    Args:
        hidden_states: [total_valid_tokens, ...]
        indices: [total_valid_tokens] flat indices
        batch_size: original batch size
        seq_len: original sequence length

    Returns:
        [batch_size, seq_len, ...]
    """
    output = torch.zeros(
        batch_size * seq_len, *hidden_states.shape[1:],
        device=hidden_states.device, dtype=hidden_states.dtype,
    )
    output[indices] = hidden_states
    return output.view(batch_size, seq_len, *hidden_states.shape[1:])


# ---------------------------------------------------------------------------
# Unpad query/key/value together (handles cross-attention shapes)
# ---------------------------------------------------------------------------

def _unpad_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    tuple[int, int],
]:
    """Unpad Q/K/V using the key-side attention mask.

    Returns:
        (query, key, value, indices_q,
         (cu_seqlens_q, cu_seqlens_k),
         (max_seqlen_q, max_seqlen_k))
    """
    seqlens_k = attention_mask.sum(-1, dtype=torch.int32)
    indices_k, cu_seqlens_k, max_seqlen_k = (
        torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten(),
        F.pad(
            torch.cumsum(seqlens_k, 0, dtype=torch.int32),
            (1, 0),
        ),
        int(seqlens_k.max()),
    )

    batch_size, kv_seq_len = key.shape[0], key.shape[1]
    # Truncate K/V if larger than mask (static cache)
    if kv_seq_len > attention_mask.shape[-1]:
        sl = attention_mask.shape[-1]
        key = key[:, :sl]
        value = value[:, :sl]

    key = key.reshape(-1, *key.shape[2:])[indices_k]
    value = value.reshape(-1, *value.shape[2:])[indices_k]

    if query_length == kv_seq_len:
        query = query.reshape(-1, *query.shape[2:])[indices_k]
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_q = max_seqlen_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query.device,
        )
        indices_q = cu_seqlens_q[:-1]
        query = query.squeeze(1)
    else:
        # Cross-attention: query has different length — treat all query tokens as valid
        all_ones = torch.ones(
            batch_size, query_length,
            dtype=attention_mask.dtype, device=attention_mask.device,
        )
        query, indices_q, cu_seqlens_q, max_seqlen_q = _unpad_input(query, all_ones)

    return (
        query, key, value, indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_q, max_seqlen_k),
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Flash-attention forward using vLLM's built-in kernel.

    Drop-in replacement for the old ``flash_attention_forward`` that imported
    from the external ``flash-attn`` package.

    Args:
        module: Attention module (must have ``is_causal`` attribute).
        query:  [batch, heads, query_len, head_dim]
        key:    [batch, heads, key_len, head_dim]
        value:  [batch, heads, key_len, head_dim]
        attention_mask: [batch, key_len] 0/1 padding mask, or None.
        dropout: Dropout probability (ignored for FA3).
        scaling: Softmax scale (default: 1/sqrt(head_dim)).
        sliding_window: Not supported by vLLM FA wrapper (ignored).
        softcap: Logit soft-capping value.
    """
    is_causal = getattr(module, "is_causal", False)
    fa_version = _FLASH_ATTN_VER

    # FA expects [batch, seq, heads, head_dim]
    seq_len = query.shape[2]
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    flash_kwargs: dict = {"fa_version": fa_version}
    if softcap is not None and softcap > 0:
        flash_kwargs["softcap"] = softcap

    if attention_mask is not None:
        # Unpad sequences and use varlen kernel
        batch_size = query.shape[0]
        (
            query, key, value, indices_q,
            (cu_q, cu_k), (max_q, max_k),
        ) = _unpad_qkv(query, key, value, attention_mask, seq_len)

        attn_output = flash_attn_varlen_func(
            query, key, value,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=int(max_q),
            max_seqlen_k=int(max_k),
            softmax_scale=scaling,
            causal=is_causal,
            **flash_kwargs,
        )
        attn_output = _pad_input(attn_output, indices_q, batch_size, seq_len)
    else:
        # No mask — build uniform cu_seqlens from batch shape
        batch_size, q_len = query.shape[0], query.shape[1]
        k_len = key.shape[1]

        cu_q = torch.arange(
            0, (batch_size + 1) * q_len, q_len,
            dtype=torch.int32, device=query.device,
        )
        cu_k = torch.arange(
            0, (batch_size + 1) * k_len, k_len,
            dtype=torch.int32, device=query.device,
        )

        # Pack into [total, heads, head_dim]
        query = query.reshape(-1, *query.shape[2:])
        key = key.reshape(-1, *key.shape[2:])
        value = value.reshape(-1, *value.shape[2:])

        attn_output = flash_attn_varlen_func(
            query, key, value,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=q_len,
            max_seqlen_k=k_len,
            softmax_scale=scaling,
            causal=is_causal,
            **flash_kwargs,
        )
        attn_output = attn_output.view(batch_size, q_len, *attn_output.shape[1:])

    return attn_output, None
