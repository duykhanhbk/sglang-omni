import logging
import math
import os
from typing import List, Literal, Optional, Tuple, Union
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Integer
from transformers.cache_utils import DynamicCache

from .timestep import CombinedTimestepLabelEmbeddings
from .normalization import LayerNorm
from .diffit_cache_utils import DiffitPastKeyValues
from .attention import SelfAttention, CrossAttention
from .mlp import MLP
from .cnn import DepthWiseConv1d

# Per-block activation dump for stage-4 calibration, parallel hook to
# the one in ``diffusion_runtime/diffit_static_step.py``. Set
# ``SPEECHIFY_DIFFIT_BLOCK_DUMP=<dir>`` to save x after each block of
# every call as ``<dir>/call{N}_block{L}.pt``. Same format as the
# static-step hook so a single ``compare_blocks.py`` consumes both.
_BLOCK_DUMP_DIR = os.environ.get("SPEECHIFY_DIFFIT_BLOCK_DUMP")
_BLOCK_DUMP_CALL_IDX = 0

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class CachedConv1d(nn.Module):
    """Compile-friendly Conv1d with explicit fixed-shape streaming cache.

    Offline and streaming paths are split into separate methods so that
    torch.compile only traces the path actually used — no Python-level
    branching on ``cache is None`` / ``use_cache``.

    Streaming cache is always a fixed-shape ``[B, C_in, left_context]`` tensor
    (never ``None``), updated via a rolling window.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        is_causal: bool = False,
        **kwargs,
    ):
        super().__init__()
        dilation = int(kwargs.get("dilation", 1))
        self.is_causal = bool(is_causal)
        self.kernel_size = int(kernel_size)
        self.dilation = dilation

        self.left_context = (
            self.dilation * (self.kernel_size - 1) if self.is_causal else int(padding)
        )
        self.cache_len = self.left_context  # backward-compat alias

        conv_padding = 0 if self.is_causal else int(padding)
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=conv_padding, **kwargs,
        )

    # -- cache helpers --------------------------------------------------------

    def init_cache(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        w = self.conv.weight
        return torch.zeros(
            batch_size, self.conv.in_channels, self.left_context,
            device=device or w.device, dtype=dtype or w.dtype,
        )

    def init_cache_like(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.shape[0], x.shape[1], self.left_context)

    # -- forward paths --------------------------------------------------------

    def forward_offline(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_causal and self.left_context > 0:
            x = F.pad(x, (self.left_context, 0))
        return self.conv(x)

    def _roll_cache(
        self, cache: torch.Tensor, x: torch.Tensor, commit_len: int,
    ) -> torch.Tensor:
        left = self.left_context
        if left == 0:
            return cache[..., :0]
        if commit_len == 0:
            return cache
        committed = x[..., :commit_len]
        if commit_len >= left:
            return committed[..., -left:].contiguous()
        keep = left - commit_len
        return torch.cat([cache[..., -keep:], committed], dim=-1).contiguous()

    def forward_stream(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        commit_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Streaming path. cache must be ``[B, C_in, left_context]``, never None."""
        T = x.shape[-1]
        commit_len = T if commit_len is None else int(commit_len)

        y = self.conv(torch.cat([cache, x], dim=-1))
        if not self.is_causal and self.left_context > 0:
            y = y[..., self.left_context : self.left_context + T]

        new_cache = self._roll_cache(cache, x, commit_len)
        return y, new_cache

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        commit_len: Optional[int] = None,
    ):
        """Compatibility wrapper. For compile hot-paths call forward_stream directly."""
        if not use_cache:
            return self.forward_offline(x)
        if cache is None:
            cache = self.init_cache_like(x)
        return self.forward_stream(x, cache, commit_len)

class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block: self-attention + latent projection + feed-forward.

    Parameters:
        latent_channels (`int`): The number of channels in the encoder hidden states.
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function for feed-forward.
        attention_bias (`bool`, *optional*, defaults to `False`): Bias in attention projections.
    """

    def __init__(
        self,
        latent_channels: Optional[int],
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        activation_fn: Literal['geglu', 'swiglu', 'relu', 'silu'] = "geglu",
        ff_mult: float = 2.67,
        attention_bias: bool = False,
        norm_elementwise_affine: bool = True,
        layer_idx: int = 0,
        is_causal: bool = False,
        sliding_window: Optional[Tuple[int, int]] = None,
        attn_implementation: str = "flash_attention",
        qk_norm: bool = False,
    ):
        super().__init__()
        self.has_latent_projection = latent_channels is not None

        # sliding_window: (left_past, right_future) frame counts; -1 means unlimited
        # 1. Self-Attention (uses clean layers/attention.py SelfAttention)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.attn1 = SelfAttention(
            hidden_size=dim,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_attention_heads,  # MHA (same as before)
            query_pre_attn_scalar=float(attention_head_dim),
            head_dim=attention_head_dim,
            kv_head_dim=attention_head_dim,
            layer_idx=layer_idx,
            attention_dropout=dropout,
            is_causal=is_causal,
            sliding_window=sliding_window,
            attn_implementation=attn_implementation,
            bias=attention_bias,
            qk_norm=qk_norm,
        )

        # Conditioning and latent projection (only for denoiser blocks)
        if self.has_latent_projection:
            self.cond_q_proj = nn.Linear(dim, dim, bias=False)
            self.cond_k_proj = nn.Linear(dim, dim, bias=False)
            self.cond_v_proj = nn.Linear(dim, dim, bias=False)
            self.latent_projection = nn.Linear(latent_channels, dim, bias=False)
            self.latent_norm = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

        # 2. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = MLP(
            dim=dim,
            mult=ff_mult,
            dropout_rate=dropout,
            activation_fn=activation_fn,
        )

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "batch seq_len dim"],
        *,
        attention_mask: Optional[Float[torch.Tensor, "batch seq_len"]] = None,
        encoder_hidden_states: Optional[Float[torch.Tensor, "batch seq_len latent_channels"]] = None,
        cond_embeddings: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[DiffitPastKeyValues, DynamicCache]] = None,
        commit_len: Optional[int] = None,
        sliding_window_override: Optional[Tuple[int, int]] = None,
    ) -> Float[torch.Tensor, "batch seq_len dim"]:
        """Forward pass: self-attention + optional latent projection + feed-forward.

        Args:
            hidden_states: Input sequence.
            attention_mask: 0/1 key-padding mask for self-attention.
            encoder_hidden_states: Latent conditioning (only used when ``has_latent_projection``).
            cond_embeddings: Timestep + speaker conditioning broadcast into Q/K/V.
            past_key_values: KV cache for streaming self-attention (DiffitPastKeyValues or HF DynamicCache).
            commit_len: Frames committed into the cache this chunk.
            sliding_window_override: If provided, overrides the layer's sliding_window for this forward pass.

        Returns:
            Updated hidden states with residual connections applied.
        """
        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)

        self_cache = None
        if past_key_values is not None:
            self_cache = getattr(past_key_values, 'self_attention_cache', past_key_values)

        cond_q = cond_k = cond_v = None
        if self.has_latent_projection and cond_embeddings is not None:
            cond_q = self.cond_q_proj(cond_embeddings)
            cond_k = self.cond_k_proj(cond_embeddings)
            cond_v = self.cond_v_proj(cond_embeddings)

        attn_output, _ = self.attn1.forward_naive(
            norm_hidden_states,
            attention_mask=attention_mask,
            past_key_values=self_cache,
            commit_len=commit_len,
            cond_q=cond_q,
            cond_k=cond_k,
            cond_v=cond_v,
            sliding_window_override=sliding_window_override,
        )
        hidden_states = attn_output + hidden_states

        # 2. Latent projection (only for denoiser blocks)
        if self.has_latent_projection:
            hidden_states = self.latent_norm(self.latent_projection(encoder_hidden_states)).to(hidden_states.dtype) + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states


class DiffiTResBlock(nn.Module):
    r"""
    DiffiT residual transformer block: Norm + SiLU + DepthWiseConv1d + BasicTransformerBlock + CrossAttention.
    """

    def __init__(
        self,
        latent_channels: int,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        activation_fn: Literal['geglu', 'swiglu', 'relu', 'silu'] = "geglu",
        ff_mult: float = 2.67,
        attention_bias: bool = False,
        norm_elementwise_affine: bool = True,
        output_scale_factor: float = 1.0,
        in_channels: int = 768,
        is_causal: bool = True,
        sliding_window: Optional[Tuple[int, int]] = None,
        layer_idx: int = 0,
        enable_speech_prompt_cross_attention: bool = True,
        attn_implementation: str = "flash_attention",
        conv_is_causal: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.enable_speech_prompt_cross_attention = enable_speech_prompt_cross_attention

        # RMSNorm instead of GroupNorm for streaming compatibility
        # GroupNorm computes stats across time dimension, breaking streaming
        # RMSNorm normalizes per-position along feature dimension
        self.norm = LayerNorm(in_channels)
        # swish activation
        self.act_fn = nn.SiLU()
        self.conv = CachedConv1d(in_channels, in_channels, kernel_size=3, padding=1, is_causal=conv_is_causal)

        self.diffit_block = BasicTransformerBlock(
            latent_channels=latent_channels,
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            dropout=dropout,
            activation_fn=activation_fn,
            ff_mult=ff_mult,
            attention_bias=attention_bias,
            norm_elementwise_affine=norm_elementwise_affine,
            layer_idx=layer_idx,
            is_causal=is_causal,
            sliding_window=sliding_window,
            attn_implementation=attn_implementation,
            qk_norm=qk_norm,
        )

        if enable_speech_prompt_cross_attention:
            self.speech_prompt_norm = nn.LayerNorm(dim, elementwise_affine=False)
            self.speech_prompt_attn = CrossAttention(
                hidden_size=dim,
                num_attention_heads=num_attention_heads,
                num_kv_heads=num_attention_heads,
                query_pre_attn_scalar=float(attention_head_dim),
                head_dim=attention_head_dim,
                kv_head_dim=attention_head_dim,
                layer_idx=layer_idx,
                kv_dim=latent_channels,
                attention_dropout=dropout,
                attn_implementation="sdpa",
                qk_norm=qk_norm,
            )

        self.output_scale_factor = output_scale_factor

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "batch time dim"],
        attention_mask: Optional[Float[torch.Tensor, "batch time"]] = None,
        encoder_hidden_states: Optional[Float[torch.Tensor, "batch time latent_channels"]] = None,
        speech_prompt_hidden_states: Optional[Float[torch.Tensor, "batch T_prompt latent_channels"]] = None,
        speech_prompt_attention_mask: Optional[Float[torch.Tensor, "batch T_prompt"]] = None,
        cond_embeddings: Optional[torch.Tensor] = None,
        past_key_values: Optional[DiffitPastKeyValues] = None,
        use_cache: bool = False,
        commit_len: Optional[int] = None,
        sliding_window_override: Optional[Tuple[int, int]] = None,
    ) -> Float[torch.Tensor, "batch time dim"]:
        """Residual block: Norm -> SiLU -> CachedConv1d -> Transformer -> CrossAttention.

        Args:
            hidden_states: Input features ``[B, T, F]``.
            attention_mask: 0/1 key-padding mask for self-attention.
            encoder_hidden_states: Latent conditioning ``[B, T, F]``.
            speech_prompt_hidden_states: Encoded speech prompt for cross-attention.
            speech_prompt_attention_mask: Valid-token mask for speech prompt.
            cond_embeddings: Timestep + speaker conditioning.
            past_key_values: Typed cache object for self-attn / cross-attn / conv states.
            use_cache: Whether to return updated cache.
            commit_len: Number of new frames to commit into cache for this chunk.
            sliding_window_override: If provided, overrides the layer's sliding_window for this forward pass.

        Returns:
            Output tensor with residual connection, same shape as *hidden_states*.
        """
        # hidden_states: [B, T, F]
        attn_hidden_states = hidden_states
        attn_mask = attention_mask
        prompt_attn_mask = speech_prompt_attention_mask

        # RMSNorm expects [*, hidden_size], so we keep [B, T, F] format
        attn_hidden_states = self.norm(attn_hidden_states)  # [B, T, F]
        attn_hidden_states = self.act_fn(attn_hidden_states)

        # Causal Conv1d with optional caching
        attn_hidden_states = attn_hidden_states.permute(0, 2, 1)  # [B, F, T]
        if use_cache:
            conv_cache = past_key_values.get_conv_cache(self.layer_idx)
            if conv_cache is None:
                conv_cache = self.conv.init_cache_like(attn_hidden_states)
            attn_hidden_states, conv_cache = self.conv.forward_stream(
                attn_hidden_states, conv_cache, commit_len=commit_len,
            )
            past_key_values.set_conv_cache(self.layer_idx, conv_cache)
        else:
            attn_hidden_states = self.conv.forward_offline(attn_hidden_states)
        attn_hidden_states = attn_hidden_states.permute(0, 2, 1)  # [B, T, F]

        # Apply safe masking to remove artifacts from padding leakage in Conv/Norm
        if attn_mask is not None:
            attn_hidden_states = attn_hidden_states * attn_mask.to(attn_hidden_states.dtype).unsqueeze(-1)

        # DiffiT Transformer block with caching
        attn_hidden_states = self.diffit_block(
            attn_hidden_states,
            attention_mask=attn_mask,
            encoder_hidden_states=encoder_hidden_states,
            cond_embeddings=cond_embeddings,
            past_key_values=past_key_values if use_cache else None,
            commit_len=commit_len,
            sliding_window_override=sliding_window_override,
        )

        # Cross-attention conditioning from a separately encoded speech prompt.
        if self.enable_speech_prompt_cross_attention and speech_prompt_hidden_states is not None:
            norm_hidden_states = self.speech_prompt_norm(attn_hidden_states)
            prompt_cache = None
            if use_cache and past_key_values is not None:
                prompt_cache = past_key_values.prompt_cross_attention_cache
            prompt_attn_out = self.speech_prompt_attn(
                norm_hidden_states,
                encoder_hidden_states=speech_prompt_hidden_states,
                encoder_attention_mask=prompt_attn_mask,
                past_key_values=prompt_cache,
            )
            attn_hidden_states = attn_hidden_states + prompt_attn_out

        # Residual connection
        output_states = (attn_hidden_states + hidden_states) / self.output_scale_factor
        return output_states


class SpeechPromptQueryPooler(nn.Module):
    """Compress a variable-length speech prompt into ``num_queries`` learned tokens.

    Mirrors ``speechify_tts.models.diffitv3.SpeechPromptQueryPooler`` (training-time
    class).  Trained as part of the diffusion model: a fixed bank of learnable
    ``query_tokens`` cross-attends into the speech-prompt encoder output (already
    passed through ``speech_prompt_encoder``), then a residual feed-forward.

    The pooled output replaces the raw prompt in every downstream cross-attention
    of the denoiser, and its mean over the query dimension is used to build the
    timestep / speaker conditioning vector (see
    :meth:`DiffiTModelV3._build_timestep_cond_from_speech_prompt`).

    Why this is critical for inference quality:
      Without query pooling the denoiser cross-attends to the **raw 468-frame**
      prompt at 24 kHz/512-hop, which is far longer than the 32-token prompt the
      model was trained against.  Attention scores end up diffuse, speaker
      conditioning bleeds across many keys, and the generated voice does not
      match the reference.
    """

    def __init__(
        self,
        latent_channels: int,
        num_queries: int,
        num_attention_heads: int,
        dropout: float = 0.0,
        ff_mult: float = 2.67,
        activation_fn: Literal['geglu', 'swiglu', 'relu', 'silu'] = "geglu",
        qk_norm: bool = False,
    ):
        super().__init__()
        if num_queries <= 0:
            raise ValueError(f"num_queries must be > 0, got {num_queries}")
        if latent_channels % num_attention_heads != 0:
            raise ValueError(
                "latent_channels must be divisible by num_attention_heads for prompt query pooling, "
                f"got latent_channels={latent_channels}, num_attention_heads={num_attention_heads}.",
            )

        self.num_queries = int(num_queries)
        attention_head_dim = latent_channels // num_attention_heads

        # NOTE: ``randn(...) * 0.02`` here is the *training-time* initialiser.
        # At inference these are overwritten by the checkpointed ``query_tokens``
        # tensor in :func:`DiffiTModelV3.load_state_dict`, so the seed used here
        # does not affect the loaded model — only the random init when no
        # weights are loaded yet (e.g. unit tests).
        self.query_tokens = nn.Parameter(torch.randn(self.num_queries, latent_channels) * 0.02)
        self.query_norm = LayerNorm(latent_channels)
        self.context_norm = LayerNorm(latent_channels)
        self.cross_attn = CrossAttention(
            hidden_size=latent_channels,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_attention_heads,
            query_pre_attn_scalar=float(attention_head_dim),
            head_dim=attention_head_dim,
            kv_head_dim=attention_head_dim,
            layer_idx=0,
            kv_dim=latent_channels,
            attention_dropout=dropout,
            attn_implementation="sdpa",
            qk_norm=qk_norm,
        )
        self.ff_norm = LayerNorm(latent_channels)
        self.ff = MLP(
            dim=latent_channels,
            mult=ff_mult,
            dropout_rate=dropout,
            activation_fn=activation_fn,
        )

    def forward(
        self,
        prompt_hidden_states: Float[torch.Tensor, "batch prompt_len latent_channels"],
        prompt_attention_mask: Optional[Float[torch.Tensor, "batch prompt_len"]],
    ) -> Tuple[
        Float[torch.Tensor, "batch num_queries latent_channels"],
        Float[torch.Tensor, "batch num_queries"],
    ]:
        bsz, prompt_len, _ = prompt_hidden_states.shape
        query_bank = self.query_tokens.unsqueeze(0).expand(bsz, -1, -1).to(
            device=prompt_hidden_states.device,
            dtype=prompt_hidden_states.dtype,
        )
        pooled_mask = torch.ones(
            bsz, self.num_queries,
            device=prompt_hidden_states.device, dtype=prompt_hidden_states.dtype,
        )

        # Edge case: empty prompt — fall back to the learned query bank.
        if prompt_len == 0:
            return query_bank, pooled_mask

        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones(
                bsz, prompt_len,
                device=prompt_hidden_states.device, dtype=prompt_hidden_states.dtype,
            )
        else:
            prompt_attention_mask = prompt_attention_mask.to(
                device=prompt_hidden_states.device, dtype=prompt_hidden_states.dtype,
            )

        valid_rows = prompt_attention_mask.sum(dim=1) > 0
        safe_prompt_mask = prompt_attention_mask
        if not bool(torch.all(valid_rows)):
            # Force at least one valid key per row to avoid NaNs in attention.
            safe_prompt_mask = prompt_attention_mask.clone()
            safe_prompt_mask[~valid_rows, 0] = 1.0

        pooled_states = query_bank + self.cross_attn(
            self.query_norm(query_bank),
            encoder_hidden_states=self.context_norm(prompt_hidden_states),
            encoder_attention_mask=safe_prompt_mask,
        )
        pooled_states = pooled_states + self.ff(self.ff_norm(pooled_states))

        if not bool(torch.all(valid_rows)):
            # Rows with no valid prompt frames fall back to the learned bank
            # (avoids propagating garbage from masked-out positions).
            pooled_states = torch.where(
                valid_rows[:, None, None],
                pooled_states,
                query_bank,
            )

        return pooled_states, pooled_mask


class SimpleTransformerEncoder(nn.Module):
    """Stacked Transformer encoder reusing :class:`BasicTransformerBlock`.

    No cross-attention, no timestep conditioning, no conv layers.
    Drop-in replacement for T5Stack in the latent / speech-prompt encoder role.

    Args:
        d_model: Hidden dimension of each block.
        num_layers: Number of stacked transformer blocks.
        num_attention_heads: Heads for multi-head self-attention.
        dropout: Dropout probability.
        activation_fn: Feed-forward activation (``geglu``, ``swiglu``, ``relu``, ``silu``).
        ff_mult: Feed-forward hidden-dim multiplier.
        is_causal: If ``True``, apply causal masking in self-attention.
        attn_implementation: Backend for attention (``flash_attention_2``, ``sdpa``, …).
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_attention_heads: int,
        dropout: float = 0.0,
        activation_fn: Literal['geglu', 'swiglu', 'relu', 'silu'] = "geglu",
        ff_mult: float = 2.67,
        is_causal: bool = False,
        attn_implementation: str = "flash_attention",
        qk_norm: bool = False,
    ):
        super().__init__()
        self.is_causal = is_causal
        attention_head_dim = d_model // num_attention_heads

        self.blocks = nn.ModuleList([
            BasicTransformerBlock(
                latent_channels=None,
                dim=d_model,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                activation_fn=activation_fn,
                ff_mult=ff_mult,
                is_causal=is_causal,
                layer_idx=i,
                attn_implementation=attn_implementation,
                qk_norm=qk_norm,
            )
            for i in range(num_layers)
        ])
        self.final_norm = LayerNorm(d_model)

    def forward(
        self,
        inputs_embeds: Float[torch.Tensor, "batch seq_len d_model"],
        attention_mask: Optional[Float[torch.Tensor, "batch seq_len"]] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[DynamicCache] = None,
        commit_len: Optional[int] = None,
    ) -> Tuple[Float[torch.Tensor, "batch seq_len d_model"], Optional[DynamicCache]]:
        """Encode a sequence through all transformer blocks.

        Args:
            inputs_embeds: Input embeddings.
            attention_mask: 0/1 key-padding mask.
            use_cache: Enable KV-cache (only effective when ``is_causal=True``).
            past_key_values: Existing KV-cache to continue from.
            commit_len: Frames committed into cache this chunk.

        Returns:
            ``(hidden_states, past_key_values)`` — cache is ``None`` when
            ``use_cache`` is ``False`` or the encoder is bidirectional.
        """
        hidden_states = inputs_embeds

        if use_cache and not self.is_causal:
            use_cache = False
            past_key_values = None

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values if use_cache else None,
                commit_len=commit_len,
            )

        hidden_states = self.final_norm(hidden_states)

        return hidden_states, past_key_values if use_cache else None


class Transformer2DModel(nn.Module):
    """Main denoiser transformer: proj_in -> timestep embed -> N x DiffiTResBlock -> proj_out.

    Despite the "2D" name (inherited from diffusers conventions), input/output are
    1-D sequences in ``[B, C, T]`` layout and internally transposed to ``[B, T, C]``.
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        latent_channels: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        ff_mult: float = 2.67,
        norm_elementwise_affine: bool = True,
        cond_proj_dim: Optional[int] = None,
        num_diffusion_timesteps: int = 1000,
        n_layers_for_cond=None,
        is_causal: bool = True,
        sliding_window: Optional[Tuple[int, int]] = None,
        sliding_window_layer_indices: Optional[List[int]] = None,
        cross_attention_layer_indices: Optional[List[int]] = None,
        causal_conv_layer_indices: Optional[List[int]] = None,
        attn_implementation: str = "flash_attention",
        qk_norm: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.cond_proj_dim = cond_proj_dim

        self.n_layers_for_cond = num_layers if n_layers_for_cond is None else n_layers_for_cond

        # 2. Define input layers
        self.in_channels = in_channels

        # RMSNorm instead of GroupNorm for streaming compatibility
        # GroupNorm normalizes across time dimension, breaking streaming
        assert in_channels is not None, "in_channels must be specified"
        self.norm = LayerNorm(in_channels)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. Define the combine label embedding and time embedding block
        self.timestep_label_combiner = CombinedTimestepLabelEmbeddings(
            num_classes=num_diffusion_timesteps,
            embedding_dim=inner_dim,
            cond_proj_dim=cond_proj_dim,
        )

        # 4. Define transformer blocks.
        # is_causal=True: all layers use causal attention; layers in
        # sliding_window_layer_indices additionally get a sliding_window (left_past, right_future).
        self._lookahead_set = set(sliding_window_layer_indices or [])
        _lookahead_set = self._lookahead_set
        if cross_attention_layer_indices is None:
            self.cross_attention_layer_indices = list(range(num_layers))
        else:
            self.cross_attention_layer_indices = sorted(set(int(i) for i in cross_attention_layer_indices))
            invalid_cross_attention_layer_indices = [
                i for i in self.cross_attention_layer_indices if i < 0 or i >= num_layers
            ]
            if invalid_cross_attention_layer_indices:
                raise ValueError(
                    "cross_attention_layer_indices contains invalid layer indices "
                    f"(num_layers={num_layers}): {invalid_cross_attention_layer_indices}"
                )
        _speech_prompt_cross_attn_set = set(self.cross_attention_layer_indices)
        _causal_conv_set = set(causal_conv_layer_indices or [])
        self.transformer_blocks = nn.ModuleList(
            [
                DiffiTResBlock(
                    latent_channels,
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    ff_mult=ff_mult,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    in_channels=in_channels,
                    is_causal=is_causal,
                    sliding_window=sliding_window if d in _lookahead_set else None,
                    layer_idx=d,
                    enable_speech_prompt_cross_attention=d in _speech_prompt_cross_attn_set,
                    attn_implementation=attn_implementation,
                    conv_is_causal=d in _causal_conv_set,
                    qk_norm=qk_norm,
                )
                for d in range(num_layers)
            ]
        )

        self.final_norm = LayerNorm(inner_dim)
        self.proj_out = nn.Linear(inner_dim, out_channels)
        self.out_channels = out_channels

    def _infer_commit_length(
        self,
        seq_len: int,
        use_cache: bool,
        block_num_frames: Optional[int] = None,
    ) -> int:
        if not use_cache:
            return seq_len
        if block_num_frames is None:
            raise ValueError("`block_num_frames` is required when `use_cache=True`.")
        return max(1, min(int(block_num_frames), seq_len))

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "batch in_channels time"],
        encoder_hidden_states: Optional[Float[torch.Tensor, "batch time latent_channels"]] = None,
        speech_prompt_hidden_states: Optional[Float[torch.Tensor, "batch T_prompt latent_channels"]] = None,
        speech_prompt_attention_mask: Optional[Float[torch.Tensor, "batch T_prompt"]] = None,
        timestep: Optional[torch.Tensor] = None,
        timestep_cond: Optional[Float[torch.Tensor, "batch cond_dim"]] = None,
        attention_mask: Optional[Float[torch.Tensor, "batch time"]] = None,
        return_hidden_states: bool = False,
        past_key_values: Optional[DiffitPastKeyValues] = None,
        use_cache: bool = False,
        block_num_frames: Optional[int] = None,
        sliding_window_override: Optional[Tuple[int, int]] = None,
    ):
        """Run the denoiser transformer stack.

        Args:
            hidden_states: Noisy input ``[B, C, T]`` (transposed internally to ``[B, T, C]``).
            encoder_hidden_states: Encoded latent conditioning ``[B, T, F]``.
            speech_prompt_hidden_states: Encoded speech prompt for cross-attention.
            speech_prompt_attention_mask: Valid-token mask for speech prompt.
            timestep: Diffusion timestep indices.
            timestep_cond: Speaker / global conditioning. Shape ``[B, F]`` for a single
                embedding or ``[B, M, F]`` for per-layer embeddings.
            attention_mask: 0/1 key-padding mask.
            return_hidden_states: If ``True``, also return per-layer hidden states.
            past_key_values: Typed cache for streaming self-attn / cross-attn / conv states.
            use_cache: Whether to return updated caches.
            block_num_frames: Current block size in mel frames (required when ``use_cache=True``).
            sliding_window_override: If provided, overrides the sliding_window for lookahead layers.

        Returns:
            ``hidden_states`` or a tuple depending on flags:
            - ``(hidden_states, past_key_values)`` when ``use_cache``
            - ``(hidden_states, all_hidden_states)`` when ``return_hidden_states``
            - ``(hidden_states, all_hidden_states, past_key_values)`` when both
        """
        # 1. Input: [B, C, T] → [B, T, inner_dim]
        hidden_states = hidden_states.transpose(1, 2)  # [B, T, C]
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)     # [B, T, inner_dim]

        if len(timestep_cond.shape) == 2:
            cond_embeddings = self.timestep_label_combiner(timestep, timestep_cond)  # [B, inner_dim]

        all_hidden_states = []
        if use_cache and past_key_values is None:
            past_key_values = DiffitPastKeyValues.create(num_layers=len(self.transformer_blocks))

        seq_len = hidden_states.shape[1]
        commit_len = self._infer_commit_length(
            seq_len=seq_len,
            use_cache=use_cache,
            block_num_frames=block_num_frames,
        )

        # 2. Blocks
        # Skip dumping inside a CUDA-graph capture region — .cpu() copy
        # plus file I/O would break capture or be baked in as a no-op.
        global _BLOCK_DUMP_CALL_IDX
        _dump_this_call = (
            _BLOCK_DUMP_DIR is not None
            and not torch.cuda.is_current_stream_capturing()
        )
        if _dump_this_call:
            _BLOCK_DUMP_CALL_IDX += 1
            os.makedirs(_BLOCK_DUMP_DIR, exist_ok=True)
            torch.save(
                {"x": hidden_states.detach().cpu(),
                 "commit_len": commit_len, "B": hidden_states.shape[0],
                 "T": hidden_states.shape[1]},
                f"{_BLOCK_DUMP_DIR}/call{_BLOCK_DUMP_CALL_IDX:04d}_block-1_input.pt",
            )
        for idx, block in enumerate(self.transformer_blocks):
            if len(timestep_cond.shape) == 3:
                # Only apply the timestep_cond (spk embedding) for first few layers
                M = timestep_cond.shape[1]
                cond_embeddings = self.timestep_label_combiner(
                    timestep, timestep_cond=timestep_cond[:, idx % M, :] if idx < self.n_layers_for_cond else None
                )  # [B, inner_dim]

            block_sw = sliding_window_override if idx in self._lookahead_set else None
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                speech_prompt_hidden_states=speech_prompt_hidden_states,
                speech_prompt_attention_mask=speech_prompt_attention_mask,
                cond_embeddings=cond_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
                commit_len=commit_len,
                sliding_window_override=block_sw,
            )
            if _dump_this_call:
                torch.save(
                    {"x": hidden_states.detach().cpu()},
                    f"{_BLOCK_DUMP_DIR}/call{_BLOCK_DUMP_CALL_IDX:04d}_block{idx:02d}.pt",
                )

            if return_hidden_states:
                all_hidden_states.append(hidden_states)

        # 3. Output: [B, T, inner_dim] → [B, out_channels, T]
        hidden_states = self.final_norm(hidden_states)
        hidden_states = self.proj_out(hidden_states)              # [B, T, out_channels]
        hidden_states = hidden_states.transpose(1, 2).contiguous() # [B, out_channels, T]

        # Handle various return combinations
        if return_hidden_states and use_cache:
            return hidden_states, all_hidden_states, past_key_values
        if return_hidden_states:
            return hidden_states, all_hidden_states
        if use_cache:
            return hidden_states, past_key_values
        return hidden_states


class DiffiTModelV3(nn.Module):
    def __init__(
        self,
        in_channels=512,
        content_in_channels=512,
        output_channels=100,
        input_channels=100,
        latent_channels=512,
        num_layers=12,
        attention_head_dim=64,
        num_attention_heads=12,
        cond_proj_dim=None,  # usually, spk embedding or cfg embedding
        dropout=0.1,
        num_diffusion_timesteps=1000,
        use_prompted_mel=False,
        input_layer_config=None,
        do_finetuning_extra_conds_model=False,
        _reuse_ar_extra_cond_for_diffusion=True,
        use_last_latents=True,
        use_token_latents=False,
        unconditioned_percentage=0.1,
        # Causal attention configuration:
        # - is_causal=True: all main transformer layers use causal self-attention, except
        #   layers at sliding_window_layer_indices which use causal + sliding_window lookahead.
        #   sliding_window is (left_past, right_future) frame counts; -1 means unlimited.
        #   The aligned_latent_encoder follows is_causal (causal with KV caching for streaming).
        #   The speech_prompt_encoder is always bidirectional.
        # - is_causal=False: all layers are bidirectional (sliding_window must be None).
        is_causal: bool = True,
        sliding_window: Optional[Tuple[int, int]] = None,
        sliding_window_layer_indices: Optional[List[int]] = None,
        # Controls speech-prompt cross-attention injection in denoiser layers.
        # If None, enable in all layers.
        cross_attention_layer_indices: Optional[List[int]] = None,
        # Controls which denoiser layers use causal (no lookahead) convolutions.
        # If None, all layers use non-causal (symmetric) convolutions.
        causal_conv_layer_indices: Optional[List[int]] = None,
        streaming_block_size: int = 32,
        # Encoder config
        latent_encoder_num_layers: int = 6,
        latent_encoder_num_heads: int = 8,
        speech_prompt_encoder_num_layers: Optional[int] = None,
        speech_prompt_encoder_num_heads: Optional[int] = None,
        # Query-pool the speech-prompt encoder output into this many learned
        # tokens (0 = disabled).  Required to match training distribution for
        # MoE 4B Unified recipes (YAML sets 32).  See SpeechPromptQueryPooler.
        speech_prompt_query_pooling_num: int = 0,
        # Bound speech-prompt encoder magnitudes via x → c*tanh(x/c) before
        # cross-attention.  0 = disabled.  YAML sets 15.0 for the MoE 4B
        # recipe — without this, attention scores blow up because the
        # encoder's hidden states are not magnitude-regularised.
        speech_prompt_encoder_output_soft_cap: float = 0.0,
        ff_mult: float = 2.67,
        attn_implementation: str = "flash_attention",
        qk_norm: bool = False,
        # Streaming inference parameters
        streaming_noise_seed: int = 42,        # Fixed seed for deterministic streaming noise
        max_streaming_frames: int = 30000,     # Max frames (~5 min at 100fps)
        diffusion_upsample_factor: int = 2,
        compile_denoiser: bool = False,        # legacy, no longer used
        **kwargs,
    ):
        super().__init__()

        if not is_causal and sliding_window is not None:
            raise ValueError(
                "sliding_window must be None when is_causal=False. "
                "Sliding window lookahead is only meaningful with causal attention."
            )

        self.streaming_block_size = streaming_block_size
        self.is_causal = is_causal
        self.sliding_window = sliding_window
        self.sliding_window_layer_indices = sliding_window_layer_indices or []
        self.cross_attention_layer_indices = cross_attention_layer_indices

        self.input_layer, self.n_mel_channels = self._prepare_input_layer(input_layer_config, output_channels)
        self.latent_channels = latent_channels
        self.use_last_latents = use_last_latents
        self.use_token_latents = use_token_latents
        self.cond_proj_dim = cond_proj_dim

        if self.use_last_latents == False:
            # use weighted sum learnable weights
            # TODO (@minh): fix the hardcoded 6 later
            self.layer_weights = nn.Parameter(torch.ones(6) / 6)  # 6 last layer.

        # Single combined projection: [aligned_latents; aligned_encoder_latents] → latent_channels
        self.input_combined_proj = nn.Linear(in_channels + content_in_channels, latent_channels)
        self.input_combined_norm = LayerNorm(latent_channels)

        if latent_channels != self.n_mel_channels:
            self.prefix_prompt_mels_proj = nn.Linear(self.n_mel_channels, latent_channels, bias=False)
        else:
            self.prefix_prompt_mels_proj = nn.Identity()

        in_channels = latent_channels

        self.transformer = Transformer2DModel(
            latent_channels=latent_channels,
            in_channels=in_channels,
            out_channels=output_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            attention_bias=True,
            activation_fn="geglu",
            ff_mult=ff_mult,
            norm_elementwise_affine=False,
            dropout=dropout,
            cond_proj_dim=cond_proj_dim,
            num_diffusion_timesteps=num_diffusion_timesteps,
            is_causal=is_causal,
            sliding_window=sliding_window,
            sliding_window_layer_indices=self.sliding_window_layer_indices,
            cross_attention_layer_indices=self.cross_attention_layer_indices,
            causal_conv_layer_indices=causal_conv_layer_indices,
            attn_implementation=attn_implementation,
            qk_norm=qk_norm,
        )

        self.inp_block = CachedConv1d(input_channels, in_channels, kernel_size=3, padding=1)

        if use_prompted_mel:
            # Non-causal for prompt (fully available reference) - symmetric padding
            self.prompt_block = nn.Conv1d(input_channels, in_channels, kernel_size=3, padding=1)
            self.integrating_conv = nn.Conv1d(in_channels * 2, in_channels, kernel_size=1)

        # Classifier-free guidance
        self.unconditioned_percentage = unconditioned_percentage
        self.unconditioned_latents = nn.Parameter(torch.randn(1, latent_channels, 1))
        self.unconditioned_speech_prompt_latents = nn.Parameter(torch.randn(1, 1, latent_channels))

        if cond_proj_dim is not None:
            self.unconditioned_embeds = nn.Parameter(torch.randn(1, cond_proj_dim))

        self.aligned_latent_encoder = SimpleTransformerEncoder(
            d_model=latent_channels,
            num_layers=latent_encoder_num_layers,
            num_attention_heads=latent_encoder_num_heads,
            dropout=dropout,
            ff_mult=ff_mult,
            is_causal=is_causal,
            attn_implementation=attn_implementation,
            qk_norm=qk_norm,
        )
        self.speech_prompt_encoder = SimpleTransformerEncoder(
            d_model=latent_channels,
            num_layers=speech_prompt_encoder_num_layers or latent_encoder_num_layers,
            num_attention_heads=speech_prompt_encoder_num_heads or latent_encoder_num_heads,
            dropout=dropout,
            ff_mult=ff_mult,
            is_causal=False,
            attn_implementation=attn_implementation,
            qk_norm=qk_norm,
        )

        # Speech-prompt regularisation: clamp magnitudes (soft-cap) and
        # compress to a fixed-length query bank (query pooler).  Both are
        # NO-OPS unless the corresponding kwargs are set.  See the
        # SpeechPromptQueryPooler docstring for why they matter for inference.
        self.speech_prompt_query_pooling_num = int(speech_prompt_query_pooling_num)
        self.speech_prompt_encoder_output_soft_cap = float(
            speech_prompt_encoder_output_soft_cap or 0.0,
        )
        if self.speech_prompt_query_pooling_num > 0:
            self.speech_prompt_query_pooler = SpeechPromptQueryPooler(
                latent_channels=latent_channels,
                num_queries=self.speech_prompt_query_pooling_num,
                num_attention_heads=(
                    speech_prompt_encoder_num_heads or latent_encoder_num_heads
                ),
                dropout=dropout,
                ff_mult=ff_mult,
                qk_norm=qk_norm,
            )
            # If the diffusion's cond_proj_dim is different from latent_channels,
            # the prompt-derived timestep_cond needs an explicit projection.
            # For our MoE 4B recipe both are 1024 so the projection is identity.
            if cond_proj_dim is not None and latent_channels != cond_proj_dim:
                self.speech_prompt_timestep_cond_proj = nn.Linear(
                    latent_channels, cond_proj_dim, bias=False,
                )
            else:
                self.speech_prompt_timestep_cond_proj = None
        else:
            self.speech_prompt_query_pooler = None
            self.speech_prompt_timestep_cond_proj = None

        self.use_prompted_mel = use_prompted_mel
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.do_finetuning_extra_conds_model = do_finetuning_extra_conds_model
        self._reuse_ar_extra_cond_for_diffusion = _reuse_ar_extra_cond_for_diffusion

        # Streaming inference settings
        self._streaming_noise_seed = streaming_noise_seed
        self._max_streaming_frames = max_streaming_frames
        self.diffusion_upsample_factor = int(diffusion_upsample_factor)
        self._register_streaming_noise()

        # Static cache: always use pre-allocated KV caches for streaming
        # denoiser. Cap is set to match the AR ``max_tokens`` ceiling
        # via ``diffusion_upsample_factor``. Keep the default AR-token
        # ceiling aligned with deploy yaml ``default_sampling_params.max_tokens``.
        max_ar_tokens = int(os.environ.get("VLLM_DIFFIT_MAX_AR_TOKENS", "960"))
        num_future_frames = int(kwargs.get("num_future_frames", 8))
        stream_t = int(streaming_block_size) + num_future_frames
        max_write = max_ar_tokens * self.diffusion_upsample_factor + stream_t
        self._static_cache_max_seq_len = 1 << (max_write - 1).bit_length()

        # Denoiser + aligned-encoder architecture specs, populated from
        # ``config.json`` by ``load_denoiser_spec`` / ``load_encoder_spec``
        # at model-load time. The cuda-graph runtime reads these for
        # cache sizing; the legacy ``.plan`` engine artifacts they used
        # to come from are gone (the runtime is captured CUDA graphs +
        # eager PyTorch now).
        self._denoiser_spec = None        # DenoiserSpec | None
        self._denoiser_plan_meta = None   # DenoiserPlanMetadata | None
        self._encoder_spec = None         # EncoderSpec | None

    def load_denoiser_spec(self, diffusion_dir: str) -> None:
        """Read denoiser architecture spec + plan metadata from ``config.json``.

        Args:
            diffusion_dir: Path containing ``config.json``.
        """
        from .diffit_specs import (
            load_denoiser_spec,
        )

        spec, plan_meta = load_denoiser_spec(
            diffusion_dir,
            block_num_frames=self.streaming_block_size,
            max_cache_frames=self._static_cache_max_seq_len,
        )
        self._denoiser_spec = spec
        self._denoiser_plan_meta = plan_meta
        logger.info(
            "DiffiT denoiser spec: max_cache_frames=%d, window_frames=%d, "
            "commit_frames=%d, prompt_len=%d",
            plan_meta.max_cache_frames, plan_meta.window_frames,
            plan_meta.commit_frames, plan_meta.prompt_len,
        )

    def load_encoder_spec(self, diffusion_dir: str) -> None:
        """Read aligned-encoder architecture spec from ``config.json``."""
        import json
        import os

        from .diffit_specs import (
            load_encoder_spec,
        )

        config_path = os.path.join(diffusion_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        num_future_frames = int(config.get("num_future_frames", 0))
        window_frames = self.streaming_block_size + num_future_frames
        # Token count = window_frames / frame_repeat (frame_repeat=2 for this model)
        max_seq_len = window_frames // 2

        spec, _plan_meta = load_encoder_spec(
            diffusion_dir,
            seq_len=max_seq_len,
            max_cache_tokens=512,
        )
        self._encoder_spec = spec
        logger.info(
            "DiffiT aligned-encoder spec: max_cache_tokens=%d",
            spec.max_cache_tokens,
        )

    def get_diffit_specs(self):
        """Return the cached architecture specs for external consumers.

        Reads from the cached spec / plan metadata that
        ``load_denoiser_spec`` and ``load_encoder_spec`` store on
        ``self``.
        """
        from .diffit_specs import (
            DiffiTSpecs,
        )
        if self._denoiser_spec is None or self._encoder_spec is None:
            raise RuntimeError(
                "DiffiT specs not initialized — call load_denoiser_spec() "
                "and load_encoder_spec() first.",
            )
        return DiffiTSpecs(
            denoiser_spec=self._denoiser_spec,
            encoder_spec=self._encoder_spec,
            prompt_len=self._denoiser_plan_meta.prompt_len,
            window_frames=self._denoiser_plan_meta.window_frames,
        )

    def _register_streaming_noise(self):
        generator = torch.Generator().manual_seed(self._streaming_noise_seed)
        noise = torch.randn(
            1, self.output_channels, self._max_streaming_frames,
            generator=generator
        )
        self.register_buffer("streaming_noise", noise, persistent=False)

    def get_streaming_noise(
        self,
        shape: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Float[torch.Tensor, "batch channels time"]:
        """Return deterministic noise from the pre-registered buffer, or generate on-the-fly."""
        B, C, T = shape
        if T > self._max_streaming_frames:
            return self._generate_extended_noise(shape, device, dtype)
        return self.streaming_noise[:, :C, :T].expand(B, -1, -1).to(device=device, dtype=dtype)

    def _generate_extended_noise(
        self,
        shape: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Float[torch.Tensor, "batch channels time"]:
        """Generate noise for sequences longer than the pre-allocated buffer."""
        B, C, T = shape
        generator = torch.Generator(device='cpu').manual_seed(self._streaming_noise_seed)
        noise = torch.randn(1, C, T, generator=generator)
        return noise.expand(B, -1, -1).to(device=device, dtype=dtype)

    @property
    def hop_length(self):
        """
        Return the hop length (downsampling factor) of the input layer.
        For TargetMelSpectrogram, this is typically 256.
        For FlowVAEInputLayer, this is the product of downsampling_ratios (typically 512).
        """
        return getattr(self.input_layer, "hop_length", 256)

    def set_melspec_calculator(self, melspec_calculator):
        self.melspec_calculator = melspec_calculator

    def set_dvae_model(self, dvae):
        self.dvae = dvae

    def set_gpt_model(self, gpt_tts):
        self.gpt_tts = gpt_tts

    def set_targetmelspec_calculator(self, targetmelspec_calculator):
        self.targetmelspec_calculator = targetmelspec_calculator

    def get_conditioning(self, x):
        return torch.zeros(1024)

    def set_extra_conds_model(self, extra_conds_model):
        self.extra_conds_model = extra_conds_model

    @staticmethod
    def _prepare_input_layer(input_layer_config, output_channels):
        """Prepare input layer (mel spectrogram) from config."""
        if input_layer_config is None:
            input_layer_config = {
                "class_name": "TargetMelSpectrogram",
                "n_mel_channels": 100,
                "sampling_rate": 24000,
                "mel_fmax": 12000,
                "do_normalization": True,
            }

        # Resolve input layer class from config name
        def _load_flow_vae():
            from .flow_vae import FlowVAE
            return FlowVAE

        def _load_flow_vae_input_layer():
            from .flow_vae import FlowVAEInputLayer
            return FlowVAEInputLayer

        def _load_target_mel():
            from ..mel_spectrogram import TargetMelSpectrogram
            return TargetMelSpectrogram

        _input_layer_registry = {
            "FlowVAE": _load_flow_vae,
            "FlowVAEInputLayer": _load_flow_vae_input_layer,
            "TargetMelSpectrogram": _load_target_mel,
        }
        class_name = input_layer_config["class_name"]
        if class_name not in _input_layer_registry:
            raise ValueError(f"Unknown input_layer class: {class_name}")
        input_layer_class = _input_layer_registry[class_name]()
        input_layer_config_copy = copy.deepcopy(input_layer_config)
        if hasattr(input_layer_config_copy, "to_dict"):
            input_layer_config_copy = input_layer_config_copy.to_dict()
        input_layer_config_copy.pop("class_name", None)
        input_layer = input_layer_class(**input_layer_config_copy)
        input_layer.requires_grad_(False)

        n_mel_channels = input_layer_config.get(
            "n_mel_channels", getattr(input_layer, "n_mel_channels", output_channels)
        )
        return input_layer, n_mel_channels

    @staticmethod
    def _build_lengths_mask(
        batch_size: int,
        seq_len: int,
        device: torch.device,
        lengths: Optional[Integer[torch.Tensor, "batch"]] = None,
    ) -> Float[torch.Tensor, "batch seq_len"]:
        """Build a 0/1 mask from per-example lengths (all-ones when *lengths* is ``None``)."""
        if lengths is None:
            return torch.ones(batch_size, seq_len, device=device, dtype=torch.float32)
        lengths = lengths.to(device=device).long()
        lengths = torch.clamp(lengths, min=0, max=seq_len)
        return (torch.arange(seq_len, device=device)[None, :] < lengths[:, None]).float()

    @staticmethod
    def _soft_cap(x: torch.Tensor, soft_cap: float = 0.0) -> torch.Tensor:
        """Apply ``x → c * tanh(x / c)`` to bound magnitudes to ``±c``.

        ``soft_cap <= 0`` disables the cap and returns ``x`` unchanged.  Used
        on the speech-prompt encoder output to prevent unbounded activations
        from saturating downstream cross-attention.  Ported from
        ``speechify_tts.models.diffitv3.DiffiTModelV3._soft_cap``.
        """
        if soft_cap <= 0.0:
            return x
        x = x / soft_cap
        x = torch.tanh(x)
        x = x * soft_cap
        return x

    def _build_timestep_cond_from_speech_prompt(
        self,
        speech_prompt_hidden_states: Float[
            torch.Tensor, "batch prompt_len latent_channels"
        ],
        speech_prompt_attention_mask: Optional[
            Float[torch.Tensor, "batch prompt_len"]
        ],
    ) -> Float[torch.Tensor, "batch cond_dim"]:
        """Mean-pool the (already soft-capped) speech-prompt states into a
        single ``[B, cond_dim]`` vector that replaces the external speaker
        embedding when query pooling is enabled.

        Mirrors ``DiffiTModelV3._build_timestep_cond_from_speech_prompt`` in
        the training code.  When ``cond_proj_dim != latent_channels`` a
        learned linear projection is applied.
        """
        if speech_prompt_hidden_states.shape[1] == 0:
            raise ValueError(
                "speech_prompt_hidden_states has zero prompt length; cannot "
                "derive timestep_cond from speech prompt.",
            )

        if speech_prompt_attention_mask is None:
            pooled = speech_prompt_hidden_states.mean(dim=1)
        else:
            mask = speech_prompt_attention_mask.to(
                device=speech_prompt_hidden_states.device,
                dtype=speech_prompt_hidden_states.dtype,
            )
            masked_sum = (speech_prompt_hidden_states * mask.unsqueeze(-1)).sum(dim=1)
            counts = mask.sum(dim=1, keepdim=True)
            pooled = masked_sum / counts.clamp(min=1.0)
            has_valid = counts.squeeze(-1) > 0
            if not bool(torch.all(has_valid)):
                pooled = torch.where(
                    has_valid[:, None],
                    pooled,
                    speech_prompt_hidden_states.mean(dim=1),
                )

        if self.speech_prompt_timestep_cond_proj is not None:
            pooled = self.speech_prompt_timestep_cond_proj(pooled)
        return pooled

    def encode_prompt(
        self,
        prefix_prompt_mels: Float[torch.Tensor, "batch T_prompt n_mels"],
        prefix_prompt_mels_lengths: Optional[Integer[torch.Tensor, "batch"]],
    ) -> Tuple[
        Float[torch.Tensor, "batch prompt_len_out latent_channels"],
        Float[torch.Tensor, "batch prompt_len_out"],
    ]:
        """Project prompt mels, run them through the speech-prompt encoder,
        optionally compress to a fixed-length query bank, and apply the
        soft-cap.

        The pooled+capped output is what every downstream consumer in the
        diffusion runtime (cross-attention preprojection, K/V cache, etc.)
        sees as the prompt — so by the time it leaves this method it is
        guaranteed to match the training distribution.

        Returns:
            ``(prompt_hidden_states, prompt_mask)``.  Shape is
            ``[B, T_prompt, latent_channels]`` when query pooling is off,
            and ``[B, num_queries, latent_channels]`` when it is on.
        """
        prompt_latents = self.prefix_prompt_mels_proj(prefix_prompt_mels)
        prompt_mask = self._build_lengths_mask(
            batch_size=prompt_latents.shape[0],
            seq_len=prompt_latents.shape[1],
            device=prompt_latents.device,
            lengths=prefix_prompt_mels_lengths,
        )
        prompt_hidden_states, _ = self.speech_prompt_encoder(
            inputs_embeds=prompt_latents,
            attention_mask=prompt_mask,
        )
        # Order mirrors the GPTTTS reference (`_encode_prompt` → query
        # pooler, then the main forward applies `_soft_cap`).  We collapse
        # both into `encode_prompt` so the downstream K/V preprojection
        # always sees the soft-capped, pooled prompt.
        if self.speech_prompt_query_pooler is not None:
            prompt_hidden_states, prompt_mask = self.speech_prompt_query_pooler(
                prompt_hidden_states,
                prompt_mask,
            )
        if self.speech_prompt_encoder_output_soft_cap > 0.0:
            prompt_hidden_states = self._soft_cap(
                prompt_hidden_states,
                soft_cap=self.speech_prompt_encoder_output_soft_cap,
            )
        return prompt_hidden_states, prompt_mask

    def _make_uncond_batch(
        self,
        batch_size: int,
        mel_frames: int,
        prompt_frames: Optional[int] = None,
    ) -> Tuple[
        Float[torch.Tensor, "batch latent_channels mel_frames"],
        Optional[Float[torch.Tensor, "batch prompt_frames latent_channels"]],
        Optional[Float[torch.Tensor, "batch cond_proj_dim"]],
    ]:
        """Create unconditional (learned) substitutes for CFG-dropped conditions."""
        uncond_latents = self.unconditioned_latents.expand(batch_size, -1, mel_frames)  # [B, F, T]
        uncond_prompt = None
        if prompt_frames is not None:
            uncond_prompt = self.unconditioned_speech_prompt_latents.expand(batch_size, prompt_frames, -1)
        uncond_tcond = self.unconditioned_embeds.expand(batch_size, -1) if self.cond_proj_dim is not None else None
        return uncond_latents, uncond_prompt, uncond_tcond

    def run_latent_encoder(
        self,
        noisy_latents: Float[torch.Tensor, "batch channels time"],
        aligned_latents: Optional[Float[torch.Tensor, "batch ..."]] = None,
        aligned_encoder_latents: Optional[Float[torch.Tensor, "batch M F_enc"]] = None,
        aligned_latents_mask: Optional[Float[torch.Tensor, "batch M"]] = None,
        prefix_prompt_mels: Optional[Float[torch.Tensor, "batch T_prompt n_mels"]] = None,
        prefix_prompt_mels_lengths: Optional[Integer[torch.Tensor, "batch"]] = None,
    ) -> Tuple[
        Float[torch.Tensor, "batch latent_channels time"],
        None,
        Optional[Float[torch.Tensor, "batch T_prompt latent_channels"]],
    ]:
        """Encode aligned latents and optionally encode the speech prompt.

        Args:
            noisy_latents: Noisy input ``[B, C, T]`` — used only to determine target *T*.
            aligned_latents: AR decoder latents ``[B, M, F]`` or ``[B, layers, M, F]``.
            aligned_encoder_latents: AR encoder latents ``[B, M, F]``.
            aligned_latents_mask: Valid-token mask for aligned latents.
            prefix_prompt_mels: Prompt mel spectrogram ``[B, T_prompt, n_mels]``.
            prefix_prompt_mels_lengths: Per-example prompt lengths.

        Returns:
            ``(encoder_hidden_states, None, speech_prompt_hidden_states)``
            where ``encoder_hidden_states`` has shape ``[B, F, T]``.
        """
        # 1. Apply weighted layer sum if needed
        if self.use_last_latents:
            if aligned_latents.ndim == 4:
                aligned_latents = aligned_latents[:, -1, ...]  # [B, M, F]
        else:
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            aligned_latents = (aligned_latents * norm_weights.view(-1, 1, 1)).sum(dim=1)

        B, M, _ = aligned_latents.shape
        _, _, T = noisy_latents.shape

        if aligned_latents_mask is not None:
            aligned_latents_mask = aligned_latents_mask.to(device=aligned_latents.device, dtype=aligned_latents.dtype)

        # Combine decoder + encoder latents and project
        combined = torch.cat([aligned_latents, aligned_encoder_latents], dim=-1)  # [B, M, 2*F_in]
        combined = self.input_combined_norm(self.input_combined_proj(combined))  # [B, M, F]

        encoded, _ = self.aligned_latent_encoder(
            inputs_embeds=combined,
            attention_mask=aligned_latents_mask,
        )  # [B, M, F]

        # Interpolate to mel frames: [B, F, T]
        encoder_hidden_states = F.interpolate(
            encoded.transpose(1, 2), size=T, mode="nearest"
        )  # [B, F, T]

        speech_prompt_hidden_states = None
        if prefix_prompt_mels is not None:
            speech_prompt_hidden_states, _ = self.encode_prompt(prefix_prompt_mels, prefix_prompt_mels_lengths)

        return encoder_hidden_states, None, speech_prompt_hidden_states

    def preproject_prompt_kv(
        self,
        speech_prompt_hidden_states: torch.Tensor,
    ) -> List[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Pre-project prompt K/V for every denoiser layer's cross-attention.

        Returns a list (one entry per layer) of ``(key, value)`` tuples in
        ``[B, num_heads, prompt_len, head_dim]`` format, ready to be passed
        as ``cached_key_states`` / ``cached_value_states`` to
        :class:`CrossAttention`. Layers without cross-attention get ``None``.
        """
        prompt_kv_per_layer: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for block in self.transformer.transformer_blocks:
            if block.enable_speech_prompt_cross_attention:
                k, v = block.speech_prompt_attn.project_kv(speech_prompt_hidden_states)
                prompt_kv_per_layer.append((k, v))
            else:
                prompt_kv_per_layer.append(None)
        return prompt_kv_per_layer

    def cache_dims(self) -> dict:
        """Return dimensions needed to pre-allocate static KV caches."""
        t = self.transformer
        return dict(
            num_layers=len(t.transformer_blocks),
            num_heads=t.num_attention_heads,
            head_dim=t.attention_head_dim,
            inner_dim=t.inner_dim,
        )

    def _apply_cfg_training_dropout(
        self,
        encoder_hidden_states: Float[torch.Tensor, "batch latent_channels time"],
        speech_prompt_hidden_states: Optional[Float[torch.Tensor, "batch T_prompt latent_channels"]],
        timestep_cond: Float[torch.Tensor, "batch cond_dim"],
    ) -> Tuple[
        Float[torch.Tensor, "batch latent_channels time"],
        Optional[Float[torch.Tensor, "batch T_prompt latent_channels"]],
        Float[torch.Tensor, "batch cond_dim"],
    ]:
        """Apply CFG dropout during training — randomly replace conditions with learned unconditional embeddings."""
        unconditioned_batches = (
            torch.rand(
                (encoder_hidden_states.shape[0], 1, 1),
                device=encoder_hidden_states.device,
            )
            < self.unconditioned_percentage
        )
        uncond_latents, uncond_prompt, uncond_tcond = self._make_uncond_batch(
            batch_size=encoder_hidden_states.shape[0],
            mel_frames=encoder_hidden_states.shape[-1],
            prompt_frames=speech_prompt_hidden_states.shape[1] if speech_prompt_hidden_states is not None else None,
        )
        encoder_hidden_states = torch.where(
            unconditioned_batches,
            uncond_latents,
            encoder_hidden_states,
        )
        if speech_prompt_hidden_states is not None:
            speech_prompt_hidden_states = torch.where(
                unconditioned_batches,
                uncond_prompt,
                speech_prompt_hidden_states,
            )
        if uncond_tcond is not None:
            timestep_cond = torch.where(
                unconditioned_batches.squeeze(-1),
                uncond_tcond,
                timestep_cond,
            )
        return encoder_hidden_states, speech_prompt_hidden_states, timestep_cond

    def _prepare_cfg_inference_batch(
        self,
        encoder_hidden_states: Float[torch.Tensor, "batch latent_channels time"],
        speech_prompt_hidden_states: Optional[Float[torch.Tensor, "batch T_prompt latent_channels"]],
        speech_prompt_attention_mask: Optional[Float[torch.Tensor, "batch T_prompt"]],
        timestep_cond: Float[torch.Tensor, "batch cond_dim"],
        hidden_states: Float[torch.Tensor, "batch channels time"],
        timestep: Float[torch.Tensor, "batch"],
        hidden_states_mask: Float[torch.Tensor, "batch 1 time"],
    ):
        """Prepare a doubled batch for CFG inference — concat conditioned and unconditional halves."""
        uncond_latents, uncond_prompt, uncond_tcond = self._make_uncond_batch(
            batch_size=encoder_hidden_states.shape[0],
            mel_frames=encoder_hidden_states.shape[-1],
            prompt_frames=speech_prompt_hidden_states.shape[1] if speech_prompt_hidden_states is not None else None,
        )
        if uncond_tcond is None:
            raise RuntimeError("conditioning_free inference requires cond_proj_dim to be set.")

        encoder_hidden_states = torch.cat([encoder_hidden_states, uncond_latents], dim=0)
        timestep_cond = torch.cat([timestep_cond, uncond_tcond], dim=0)

        if speech_prompt_hidden_states is not None:
            speech_prompt_hidden_states = torch.cat(
                [speech_prompt_hidden_states, uncond_prompt], dim=0
            )
            if speech_prompt_attention_mask is not None:
                speech_prompt_attention_mask = torch.cat(
                    [speech_prompt_attention_mask, speech_prompt_attention_mask], dim=0
                )

        hidden_states = torch.cat([hidden_states, hidden_states], dim=0)
        timestep = torch.cat([timestep, timestep], dim=0)
        hidden_states_mask = torch.cat([hidden_states_mask, hidden_states_mask], dim=0)

        return (
            encoder_hidden_states,
            speech_prompt_hidden_states,
            speech_prompt_attention_mask,
            timestep_cond,
            hidden_states,
            timestep,
            hidden_states_mask,
        )

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "batch n_mels L"],
        timestep: Optional[Float[torch.Tensor, "batch"]] = None,
        prompted_mels: Optional[Float[torch.Tensor, "batch n_mels L"]] = None,
        timestep_cond: Optional[Float[torch.Tensor, "batch F"]] = None,
        hidden_states_mask: Optional[Float[torch.Tensor, "batch 1 L"]] = None,
        mel_codes_hidden_states: Optional[Float[torch.Tensor, "batch channels T"]] = None,
        speech_prompt_hidden_states: Optional[Float[torch.Tensor, "batch M latent_channels"]] = None,
        speech_prompt_attention_mask: Optional[Float[torch.Tensor, "batch M"]] = None,
        aligned_latents: Optional[Float[torch.Tensor, "batch M latent_channels"]] = None,
        aligned_encoder_latents: Optional[Float[torch.Tensor, "batch M latent_channels"]] = None,
        aligned_latents_mask: Optional[Float[torch.Tensor, "batch M"]] = None,
        conditioning_free: bool = False,
        cfg_training: bool = True,
        return_hidden_states: bool = False,
        prefix_prompt_mels: Optional[Float[torch.Tensor, "batch T_prompt n_mels"]] = None,
        prefix_prompt_mels_lengths: Optional[Integer[torch.Tensor, "batch"]] = None,
        past_key_values: Optional[DiffitPastKeyValues] = None,
        use_cache: bool = False,
        block_num_frames: Optional[int] = None,
        sliding_window_override: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        """Full forward pass: encode latents, apply CFG, run denoiser transformer.

        Args:
            hidden_states: Noisy mel latents ``[B, n_mels, L]``.
            timestep: Diffusion timestep indices ``[B]``.
            prompted_mels: Prompted mel spectrograms (only when ``use_prompted_mel``).
            timestep_cond: Speaker / global conditioning ``[B, F]``.
            hidden_states_mask: Padding mask ``[B, 1, L]``.
            mel_codes_hidden_states: Pre-encoded latent conditioning ``[B, F, T]``
                (bypasses ``run_latent_encoder`` when provided).
            speech_prompt_hidden_states: Pre-encoded speech prompt ``[B, M, F]``.
            speech_prompt_attention_mask: Valid-token mask for speech prompt.
            aligned_latents: AR decoder latents ``[B, M, F]`` (or ``[B, layers, M, F]``).
            aligned_encoder_latents: AR encoder latents ``[B, M, F]``.
            aligned_latents_mask: Valid-token mask for aligned latents.
            conditioning_free: Use CFG at inference time.
            cfg_training: Apply CFG dropout during training.
            return_hidden_states: Also return per-layer hidden states.
            prefix_prompt_mels: Raw prompt mels to encode via ``speech_prompt_encoder``.
            prefix_prompt_mels_lengths: Per-example prompt lengths.
            past_key_values: Typed cache for streaming.
            use_cache: Whether to maintain and return KV-cache.
            block_num_frames: Current block size in mel frames.
            sliding_window_override: If provided, overrides the sliding_window for lookahead
                layers. During training with ``self.sliding_window`` configured, a random
                right lookahead in ``[0, max_right]`` is sampled automatically when this is
                ``None``.

        Returns:
            ``output`` or a tuple depending on *return_hidden_states* / *use_cache* flags.
        """
        # 0. Determine effective sliding window for this forward pass
        if sliding_window_override is None and self.sliding_window is not None and self.training:
            max_right = self.sliding_window[1]
            random_right = torch.randint(0, max_right + 1, (1,)).item()
            sliding_window_override = (self.sliding_window[0], random_right)

        # 1. Handle noisy sample (hidden_states): [B, 100, T]
        B, _, T = hidden_states.shape
        encoder_hidden_states = mel_codes_hidden_states

        # CausalConv1d handles padding internally
        hidden_states = self.inp_block(hidden_states)  # [B, 1024, T]
        if self.use_prompted_mel:
            # prompt_block is non-causal (prompt is fully available)
            prompted_mels = self.prompt_block(prompted_mels)  # [B, 1024, T]
            hidden_states = torch.concat([hidden_states, prompted_mels], dim=1)

        if aligned_latents_mask is not None:
            aligned_latents_mask = aligned_latents_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if speech_prompt_attention_mask is not None:
            speech_prompt_attention_mask = speech_prompt_attention_mask.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
        if hidden_states_mask is None:
            hidden_states_mask = torch.ones(
                B, 1, hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype
            )
        else:
            hidden_states_mask = hidden_states_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)

        # 2. Apply latent encoders (aligned causal + speech prompt bidirectional)
        if encoder_hidden_states is None and aligned_latents is not None:
            encoder_hidden_states, _, speech_prompt_hidden_states = self.run_latent_encoder(
                noisy_latents=hidden_states,
                aligned_latents=aligned_latents,
                aligned_encoder_latents=aligned_encoder_latents,
                aligned_latents_mask=aligned_latents_mask,
                prefix_prompt_mels=prefix_prompt_mels,
                prefix_prompt_mels_lengths=prefix_prompt_mels_lengths,
            )
        elif speech_prompt_hidden_states is None and prefix_prompt_mels is not None:
            speech_prompt_hidden_states, speech_prompt_attention_mask = self.encode_prompt(
                prefix_prompt_mels, prefix_prompt_mels_lengths
            )

        if (
            speech_prompt_hidden_states is not None
            and speech_prompt_attention_mask is None
        ):
            speech_prompt_attention_mask = self._build_lengths_mask(
                batch_size=speech_prompt_hidden_states.shape[0],
                seq_len=speech_prompt_hidden_states.shape[1],
                device=speech_prompt_hidden_states.device,
                lengths=prefix_prompt_mels_lengths,
            )

        # Apply CFG (Classifier-Free Guidance)
        if self.training and self.unconditioned_percentage > 0 and cfg_training:
            (
                encoder_hidden_states,
                speech_prompt_hidden_states,
                timestep_cond,
            ) = self._apply_cfg_training_dropout(
                encoder_hidden_states,
                speech_prompt_hidden_states,
                timestep_cond,
            )

        if (not self.training) and conditioning_free:
            (
                encoder_hidden_states,
                speech_prompt_hidden_states,
                speech_prompt_attention_mask,
                timestep_cond,
                hidden_states,
                timestep,
                hidden_states_mask,
            ) = self._prepare_cfg_inference_batch(
                encoder_hidden_states,
                speech_prompt_hidden_states,
                speech_prompt_attention_mask,
                timestep_cond,
                hidden_states,
                timestep,
                hidden_states_mask,
            )

        # 4. Concatenate hidden states and mel codes
        if self.use_prompted_mel:
            hidden_states = self.integrating_conv(hidden_states)  # [B, 1024, T]

        # Transpose encoder_hidden_states from [B, F, T] to [B, T, F]
        encoder_hidden_states = encoder_hidden_states.transpose(1, 2)

        # 5. Run forward of the main model (layer-wise causal + lookahead self-attention)
        outs = self.transformer(
            hidden_states,
            attention_mask=hidden_states_mask.squeeze(1),  # [b, T]
            encoder_hidden_states=encoder_hidden_states,
            speech_prompt_hidden_states=speech_prompt_hidden_states,
            speech_prompt_attention_mask=speech_prompt_attention_mask,
            timestep=timestep,  # [B]
            timestep_cond=timestep_cond,  # [B, F]
            return_hidden_states=return_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            block_num_frames=block_num_frames,
            sliding_window_override=sliding_window_override,
        )

        present_key_values = None
        if return_hidden_states and use_cache:
            outs, all_hidden_states, present_key_values = outs
        elif return_hidden_states:
            outs, all_hidden_states = outs
        elif use_cache:
            outs, present_key_values = outs

        if self.output_channels > self.input_channels:
            outs = outs[:, : self.input_channels, :]

        if return_hidden_states:
            if use_cache:
                return outs * hidden_states_mask, all_hidden_states, present_key_values
            return outs * hidden_states_mask, all_hidden_states

        if use_cache:
            return outs, present_key_values
        return outs
