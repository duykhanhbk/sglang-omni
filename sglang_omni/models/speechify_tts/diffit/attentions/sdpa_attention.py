from typing import Optional, Any, Union
import math
import torch

from transformers.utils import logging


logger = logging.get_logger(__name__)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def vanilla_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight.float(), dim=-1).to(value.dtype)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight


def _build_sdpa_mask(
    attention_mask: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    """Build a 4D additive attention mask for SDPA from a 2D 0/1 padding mask.

    Combines padding mask with causal mask when both are needed.
    Returns a float mask where masked positions are -inf and valid positions are 0.

    Args:
        attention_mask: [batch, key_len] with 1=valid, 0=padding
        query: query tensor (used for dtype and query_len)
        key: key tensor (used for key_len)
        is_causal: whether to apply causal (lower-triangular) masking
    """
    dtype = query.dtype
    batch_size = attention_mask.shape[0]
    query_len = query.shape[2]
    key_len = key.shape[2]

    # Convert padding mask: [batch, key_len] -> [batch, 1, 1, key_len]
    # 0/1 -> additive: 0 for valid, -inf for padding
    pad_mask = (1.0 - attention_mask.to(dtype))[:, None, None, :] * torch.finfo(dtype).min

    if not is_causal:
        return pad_mask

    # Build causal mask: [1, 1, query_len, key_len]
    causal_mask = torch.full(
        (1, 1, query_len, key_len), torch.finfo(dtype).min,
        dtype=dtype, device=query.device,
    )
    causal_mask = causal_mask.triu(diagonal=key_len - query_len + 1)

    # Combine: broadcast addition merges both masks
    return pad_mask + causal_mask


_XATTN_DUMP_DIR = None
_XATTN_DUMP_COUNT = 0
_XATTN_DUMP_MAX = 0


def _maybe_dump_xattn(path_tag, query, key, value, mask, scale,
                      attn_output, attention_weight) -> None:
    """Debug-only (CALIB_*-style) per-call dump of the decode-step
    weights-returning cross-attention, for offline replay of the fused
    kernel against the eager reference on live tensors.  Enable with
    ``SPEECHIFY_XATTN_DUMP_DIR=/path`` (``SPEECHIFY_XATTN_DUMP_MAX``
    caps call count, default 5000).  Never set in production configs.
    Only fires on the q_len==1 decode path; no-op when tracing/compiling.
    """
    global _XATTN_DUMP_DIR, _XATTN_DUMP_COUNT, _XATTN_DUMP_MAX
    if _XATTN_DUMP_DIR is None:
        import os
        d = os.environ.get("SPEECHIFY_XATTN_DUMP_DIR", "")
        _XATTN_DUMP_DIR = d or ""
        if d:
            os.makedirs(d, exist_ok=True)
            _XATTN_DUMP_MAX = int(
                os.environ.get("SPEECHIFY_XATTN_DUMP_MAX", "5000"))
    if not _XATTN_DUMP_DIR or _XATTN_DUMP_COUNT >= _XATTN_DUMP_MAX:
        return
    if torch.compiler.is_compiling():
        return
    if query.shape[2] != 1:
        return
    i = _XATTN_DUMP_COUNT
    _XATTN_DUMP_COUNT = i + 1
    torch.save(
        {
            "tag": path_tag,
            "query": query.detach().cpu(),
            "key": key.detach().cpu(),
            "value": value.detach().cpu(),
            "mask": None if mask is None else mask.detach().cpu(),
            "scale": scale,
            "attn_output": attn_output.detach().cpu(),
            "attention_weight": (
                None if attention_weight is None
                else attention_weight.detach().cpu()),
        },
        f"{_XATTN_DUMP_DIR}/xattn_{i:05d}.pt",
    )


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    return_cross_attentions: Optional[bool] = False,
    **kwargs,
) -> Union[torch.Tensor, tuple[torch.Tensor, Any]]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask", None) is not None:
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    # Fused single-kernel path for the decode-step weights-returning case
    # (TTS text cross-attention): replaces the ~10-kernel eager chain below
    # AND the three .contiguous() copies.  The earlier live divergence
    # (shorter audio despite exact synthetic parity) was the kernel
    # assuming a unit-stride bias S dim: the live alignment bias is a
    # permute of a [..., S, H] tensor (stride(S)==H, stride(H)==1), so
    # every s>0 read the wrong bias value — synthetic tests built
    # contiguous masks and could not see it.  Fixed by threading
    # stride_bs through the kernel; replay of 576 dumped live calls now
    # matches eager bit-exactly on weights (max 1 ulp) with 0 argmax
    # mismatches (scripts/replay_xattn_dump.py in the outer repo,
    # SPEECHIFY_XATTN_DUMP_DIR to regenerate dumps).  Not bit-exact on
    # outputs by design (reduction order) — calibration-gated.
    _FUSED_DECODE_XATTN_ENABLED = False  # torch-native port: Triton fused kernel unavailable
    if (
        _FUSED_DECODE_XATTN_ENABLED
        and return_cross_attentions
        and not kwargs.get("output_attentions", False)
    ):
        from vllm_omni.model_executor.layers.speechify.fused_decode_cross_attn import (
            fused_decode_cross_attn,
            fused_decode_cross_attn_supported,
        )

        _is_causal_static = (
            is_causal
            if is_causal is not None
            else (getattr(module, "is_causal", False) and query.shape[2] > 1)
        )
        if fused_decode_cross_attn_supported(
            query, key, attention_mask, dropout, bool(_is_causal_static)
        ):
            bias4d = (
                attention_mask[:, :, :, : key.shape[-2]]
                if attention_mask is not None
                else None
            )
            scale = (
                scaling if scaling is not None else query.shape[-1] ** -0.5
            )
            attn_output, attention_weight = fused_decode_cross_attn(
                query, key, value, bias4d, scale
            )
            _maybe_dump_xattn(
                "fused", query, key, value, bias4d, scale,
                attn_output, attention_weight,
            )
            # attn_output is already [B, 1(q), H, D] — the layout the
            # epilogue below produces via transpose(1, 2).contiguous().
            return attn_output, attention_weight

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # Resolve is_causal from the module when not explicitly provided.
    module_is_causal = getattr(module, "is_causal", False)
    if is_causal is None:
        is_causal = module_is_causal and query.shape[2] > 1

    # Convert 2D padding masks to 4D additive masks, combining with causal
    # mask when needed.  This makes SDPA a drop-in replacement for
    # flash_attention_2 which handles 2D masks + causal flag independently.
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = _build_sdpa_mask(attention_mask, query, key, is_causal)
        is_causal = False  # causal already baked into the mask
    elif attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()
    if not return_cross_attentions:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
        )
        attention_weight = None
    else:
        attn_output, attention_weight = vanilla_scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout, scale=scaling, is_causal=is_causal
        )
        _maybe_dump_xattn(
            "eager", query, key, value, attention_mask,
            scaling if scaling is not None else query.shape[-1] ** -0.5,
            attn_output, attention_weight,
        )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attention_weight
