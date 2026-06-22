# SPDX-License-Identifier: Apache-2.0
"""SpeechifyTTS speaker tower: ``UnifiedSpkEmbeddingWithDec`` (inference-only).

Vendored from the GPTTTS / vllm-omni inference ports
(``speechify/speaker_embedding.py`` + ``speaker_embedding_unified.py``). The
module hierarchy / parameter names match the converted checkpoint's
``extra_conds_model.*`` keys verbatim so a Unified checkpoint loads cleanly.

Pipeline (per 800-frame mel chunk)::

    mel [B, F, T]
      -> conv_downsample2d -> x_down [B, T', dim]
      -> feats_predictor (detached) -> per-feature logits -> resolve_for_inference
      -> encoder (conformer) -> enc_output
      -> decoder (cross-attn from spk_feature_keys) -> out_norm(o(...)) -> identity tokens
      -> cat([identity tokens, feature tokens]) -> spk_features [B, nb_spk+nb_feat, 1024]

Only the inference surface (``get_spk_features`` / ``extract_spk_features`` /
``update_feature_embeddings``) is implemented; training paths are omitted.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Optional, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from sglang_omni.models.speechify_tts.mel_spectrogram import TargetMelSpectrogram


# --------------------------------------------------------------------------- #
# Conformer building blocks (from speaker_embedding.py)                       #
# --------------------------------------------------------------------------- #
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def calc_same_padding(kernel_size: int) -> tuple[int, int]:
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in: int, chan_out: int, kernel_size: int, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.padding)
        return self.conv(x)


class Scale(nn.Module):
    def __init__(self, scale: float, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module, remove_args: tuple = ()):
        super().__init__()
        self.remove_args = remove_args
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.norm(x)
        for optional_arg in self.remove_args:
            kwargs.pop(optional_arg, None)
        return self.fn(x, **kwargs)


class Attention(nn.Module):
    """Multi-head attention via SDPA, with optional relative position bias."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        max_pos_emb: int = 128,
        v_dim: int | None = None,
        k_dim: int | None = None,
        use_flash_att: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        v_dim = default(v_dim, dim)
        k_dim = default(k_dim, v_dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(k_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(v_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        if max_pos_emb > 0:
            self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, heads)
        self.dropout = nn.Dropout(dropout)
        self._cached_pos_bias: torch.Tensor | None = None
        self._cached_pos_bias_len: int = -1

    def _get_pos_bias(self, n, device, dtype):
        if self._cached_pos_bias is not None and self._cached_pos_bias_len == n:
            return self._cached_pos_bias.to(dtype)
        max_pos_emb = self.max_pos_emb
        seq = torch.arange(n, device=device)
        dist = seq.unsqueeze(1) - seq.unsqueeze(0)
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        bias = self.rel_pos_emb(dist).permute(2, 0, 1).unsqueeze(0)
        self._cached_pos_bias = bias
        self._cached_pos_bias_len = n
        return bias.to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        extended_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, n = x.shape[0], x.shape[1]
        h, d = self.heads, self.dim_head
        has_context = exists(context)
        context = default(context, x)

        q = self.to_q(x).reshape(B, -1, h, d).transpose(1, 2)
        k = self.to_k(context).reshape(B, -1, h, d).transpose(1, 2)
        v = self.to_v(context).reshape(B, -1, h, d).transpose(1, 2)

        attn_mask = None
        if self.max_pos_emb > 0 and not has_context:
            attn_mask = self._get_pos_bias(n, x.device, q.dtype)

        if exists(mask) or exists(context_mask):
            if not exists(mask):
                mask = torch.ones(B, n, device=x.device)
            context_mask = (
                default(context_mask, mask)
                if not has_context
                else default(
                    context_mask,
                    torch.ones(B, context.shape[1], device=x.device),
                )
            )
            pad_mask = mask.unsqueeze(1).unsqueeze(3) * context_mask.unsqueeze(
                1
            ).unsqueeze(2)
            neg_inf = torch.tensor(
                -torch.finfo(q.dtype).max, dtype=q.dtype, device=x.device
            )
            pad_mask = torch.where(pad_mask.bool(), torch.zeros_like(neg_inf), neg_inf)
            attn_mask = attn_mask + pad_mask if attn_mask is not None else pad_mask

        if exists(extended_mask):
            neg_inf = torch.tensor(
                -torch.finfo(q.dtype).max, dtype=q.dtype, device=x.device
            )
            ext = torch.where(extended_mask.bool(), torch.zeros_like(neg_inf), neg_inf)
            attn_mask = attn_mask + ext if attn_mask is not None else ext

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, n, -1)
        return self.dropout(self.to_out(out))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim: int,
        causal: bool = False,
        expansion_factor: int = 2,
        kernel_size: int = 31,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n c -> b c n"),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(
                inner_dim, inner_dim, kernel_size=kernel_size, padding=padding
            ),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange("b c n -> b n c"),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, conditioning=None) -> torch.Tensor:
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        use_post_norm: bool = False,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        conv_dropout: float = 0.0,
        conv_causal: bool = False,
        max_pos_emb_self: int = 128,
        use_flash_att: bool = False,
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            max_pos_emb=max_pos_emb_self,
            use_flash_att=use_flash_att,
        )
        self.conv = ConformerConvModule(
            dim=dim,
            causal=conv_causal,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        self.post_norm = nn.LayerNorm(dim) if use_post_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_extended_mask: bool = True,
        debug_layer_idx: int = -1,
    ) -> torch.Tensor:
        x = self.ff1(x) + x
        if use_extended_mask:
            x = self.attn(x, extended_mask=mask) + x
        else:
            x = self.attn(x, mask=mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        return self.post_norm(x)


class ConformerNoConvDecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        enc_dim: int | None = None,
        use_post_norm: bool = False,
        use_flash_att: bool = False,
    ):
        super().__init__()
        if enc_dim is None:
            enc_dim = dim
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            max_pos_emb=-1,
            use_flash_att=use_flash_att,
        )
        self.cross_attn = Attention(
            dim=dim,
            k_dim=enc_dim,
            v_dim=enc_dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            max_pos_emb=-1,
            use_flash_att=use_flash_att,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = PreNorm(dim, self.attn)
        self.cross_attn = PreNorm(dim, self.cross_attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        self.post_norm = nn.LayerNorm(dim) if use_post_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.ff1(x) + x
        x = self.attn(x) + x
        x = (
            self.cross_attn(
                x, context=encoder_hidden_states, context_mask=encoder_attention_mask
            )
            + x
        )
        x = self.ff2(x) + x
        return self.post_norm(x)


# --------------------------------------------------------------------------- #
# Feature processor + Unified speaker model (from speaker_embedding_unified.py)#
# --------------------------------------------------------------------------- #
def sinusoidal_encode(
    values: torch.Tensor,
    embedding_dim: int,
    min_value: float,
    max_value: float,
    min_period: float = 0.006,
    max_period: float = 16.0,
) -> torch.Tensor:
    clamped = values.float().clamp(min=min_value, max=max_value)
    normalized = (clamped - min_value) / (max_value - min_value + 1e-8)
    half_dim = embedding_dim // 2
    periods = torch.exp(
        torch.linspace(
            math.log(min_period),
            math.log(max_period),
            half_dim,
            device=values.device,
            dtype=torch.float32,
        )
    )
    freqs = (2.0 * math.pi) / periods
    args = normalized.unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


@dataclasses.dataclass
class EmbeddingFeature:
    class Type(str, enum.Enum):
        CONTINUOUS_BUCKETED = "continuous_bucketed"
        CONTINUOUS = "continuous"
        CATEGORICAL = "categorical"

    name: str
    min_value: Union[float, int]
    max_value: Union[float, int]
    missing_value: Union[float, int]
    nb_buckets: int = 0
    clamp_boundary_buckets: bool = False
    apply_grad_reversal: bool = True
    feature_type: Type = Type.CONTINUOUS_BUCKETED
    noise_sigma: float = 0.0
    grad_reversal_buckets: int = 16
    percentile_cdf: Optional[list[float]] = None

    def __post_init__(self) -> None:
        self.feature_type = EmbeddingFeature.Type(self.feature_type)
        if (
            self.feature_type == EmbeddingFeature.Type.CONTINUOUS_BUCKETED
            and self.nb_buckets <= 0
        ):
            raise ValueError(
                f"nb_buckets must be > 0 for continuous_bucketed feature '{self.name}'"
            )
        if self.percentile_cdf is not None:
            if len(self.percentile_cdf) < 2:
                raise ValueError(
                    f"percentile_cdf must have >= 2 values for feature '{self.name}'"
                )
            if self.feature_type == EmbeddingFeature.Type.CATEGORICAL:
                raise ValueError(
                    f"percentile_cdf is not supported for categorical '{self.name}'"
                )

    @property
    def is_bucketed(self) -> bool:
        return self.feature_type in (
            EmbeddingFeature.Type.CONTINUOUS_BUCKETED,
            EmbeddingFeature.Type.CATEGORICAL,
        )

    @property
    def is_continuous_sinusoidal(self) -> bool:
        return self.feature_type == EmbeddingFeature.Type.CONTINUOUS

    @property
    def uses_percentile_space(self) -> bool:
        return self.percentile_cdf is not None

    @property
    def num_classes(self) -> int:
        if self.feature_type == EmbeddingFeature.Type.CATEGORICAL:
            return int(self.max_value - self.min_value) + 1
        if self.feature_type == EmbeddingFeature.Type.CONTINUOUS_BUCKETED:
            return self.nb_buckets
        raise ValueError(f"num_classes not applicable for continuous '{self.name}'")

    @property
    def embedding_size(self) -> int:
        return self.num_classes + 1


class MeanPooledPredictor(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim, eps=1e-5)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            mask_f = mask.float().unsqueeze(-1)
            x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        x = self.norm(x.to(self.norm.weight.dtype))
        x = self.act(self.fc1(x))
        return self.fc2(x)


class AttentionPooledPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 32,
        num_heads: int = 4,
    ):
        super().__init__()
        assert input_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        self.k_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.norm = nn.LayerNorm(input_dim, eps=1e-5)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.query.expand(B, -1, -1).reshape(B, 1, H, D).transpose(1, 2).contiguous()
        k = self.k_proj(x).reshape(B, T, H, D).transpose(1, 2).contiguous()
        v = self.v_proj(x).reshape(B, T, H, D).transpose(1, 2).contiguous()
        attn_mask = None
        if mask is not None:
            attn_mask = (
                torch.where(mask.bool().unsqueeze(1).unsqueeze(2), 0.0, float("-inf"))
                .to(dtype=x.dtype)
                .contiguous()
            )
        pooled = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        pooled = pooled.transpose(1, 2).reshape(B, -1)
        pooled = self.norm(pooled.to(self.norm.weight.dtype))
        return self.fc2(self.act(self.fc1(pooled)))


class FeatureProcessor(nn.Module):
    def __init__(
        self,
        features_config: list[Union[dict, EmbeddingFeature]],
        dec_dim: int,
        predictor_input_dim: int,
        sinusoidal_dim: int = 256,
    ):
        super().__init__()
        self.dec_dim = dec_dim
        self.sinusoidal_dim = sinusoidal_dim
        self.features_config: list[EmbeddingFeature] = [
            f if isinstance(f, EmbeddingFeature) else EmbeddingFeature(**f)
            for f in features_config
        ]
        self._features_by_name = {f.name: f for f in self.features_config}

        bucketed_feats = [f for f in self.features_config if f.is_bucketed]
        continuous_feats = [
            f for f in self.features_config if f.is_continuous_sinusoidal
        ]

        self.feature_embeddings = nn.ModuleDict(
            {
                feat.name: nn.Embedding(feat.embedding_size, dec_dim)
                for feat in bucketed_feats
            }
        )
        for emb in self.feature_embeddings.values():
            nn.init.zeros_(emb.weight)

        self.continuous_feature_projections = nn.ModuleDict(
            {
                feat.name: nn.Sequential(
                    nn.Linear(sinusoidal_dim, dec_dim),
                    nn.SiLU(),
                    nn.Linear(dec_dim, dec_dim),
                )
                for feat in continuous_feats
            }
        )

        for feat in bucketed_feats:
            nc = feat.num_classes
            if feat.feature_type == EmbeddingFeature.Type.CATEGORICAL:
                self.register_buffer(
                    f"_bucket_centers_{feat.name}",
                    torch.arange(nc, dtype=torch.float) + feat.min_value,
                )
            elif feat.uses_percentile_space:
                boundaries = torch.linspace(0.0, 1.0, nc + 1)[1:-1]
                step = 1.0 / nc
                centers = (torch.arange(nc, dtype=torch.float) + 0.5) * step
                self.register_buffer(f"_boundaries_{feat.name}", boundaries)
                self.register_buffer(f"_bucket_centers_{feat.name}", centers)
            else:
                boundaries = torch.linspace(feat.min_value, feat.max_value, nc + 1)[1:-1]
                step = (feat.max_value - feat.min_value) / nc
                centers = feat.min_value + (
                    torch.arange(nc, dtype=torch.float) + 0.5
                ) * step
                self.register_buffer(f"_boundaries_{feat.name}", boundaries)
                self.register_buffer(f"_bucket_centers_{feat.name}", centers)

        for feat in bucketed_feats + continuous_feats:
            if feat.uses_percentile_space:
                self.register_buffer(
                    f"_percentile_cdf_{feat.name}",
                    torch.tensor(feat.percentile_cdf, dtype=torch.float32),
                )

        for feat in continuous_feats:
            if feat.apply_grad_reversal:
                nc = feat.grad_reversal_buckets
                if feat.uses_percentile_space:
                    boundaries = torch.linspace(0.0, 1.0, nc + 1)[1:-1]
                else:
                    boundaries = torch.linspace(feat.min_value, feat.max_value, nc + 1)[
                        1:-1
                    ]
                self.register_buffer(f"_gr_boundaries_{feat.name}", boundaries)

        def _predictor_output_dim(feat: EmbeddingFeature) -> int:
            return 1 if feat.is_continuous_sinusoidal else feat.num_classes

        self.feature_predictor_heads = nn.ModuleDict(
            {
                feat.name: AttentionPooledPredictor(
                    predictor_input_dim, output_dim=_predictor_output_dim(feat)
                )
                for feat in self.features_config
            }
        )

        def _grad_rev_output_dim(feat: EmbeddingFeature) -> int:
            if feat.is_continuous_sinusoidal:
                return feat.grad_reversal_buckets
            return feat.num_classes

        self.grad_reversal_predictors = nn.ModuleDict(
            {
                feat.name: MeanPooledPredictor(
                    predictor_input_dim, output_dim=_grad_rev_output_dim(feat)
                )
                for feat in self.features_config
                if feat.apply_grad_reversal
            }
        )

    def _bucketize(self, value: torch.Tensor, feat: EmbeddingFeature) -> torch.Tensor:
        missing_mask = value == feat.missing_value
        if feat.feature_type == EmbeddingFeature.Type.CATEGORICAL:
            bucket_idx = (value - feat.min_value).long()
        else:
            boundaries = getattr(self, f"_boundaries_{feat.name}")
            buck_value = (
                self._raw_to_percentile(value, feat)
                if feat.uses_percentile_space
                else value
            )
            bucket_idx = torch.bucketize(buck_value, boundaries)
        return torch.where(
            missing_mask, torch.full_like(bucket_idx, feat.num_classes), bucket_idx
        )

    def _raw_to_percentile(
        self, value: torch.Tensor, feat: EmbeddingFeature
    ) -> torch.Tensor:
        cdf = getattr(self, f"_percentile_cdf_{feat.name}")
        n = cdf.shape[0]
        scale = torch.linspace(0.0, 1.0, n, device=cdf.device)
        x = value.float().clamp(cdf[0], cdf[-1])
        idx = torch.searchsorted(cdf.contiguous(), x.contiguous()).clamp(1, n - 1)
        x0, x1 = cdf[idx - 1], cdf[idx]
        y0, y1 = scale[idx - 1], scale[idx]
        t = (x - x0) / (x1 - x0).clamp(min=1e-6)
        return (y0 + t * (y1 - y0)).clamp(0.0, 1.0)

    def _percentile_to_raw(
        self, pct: torch.Tensor, feat: EmbeddingFeature
    ) -> torch.Tensor:
        cdf = getattr(self, f"_percentile_cdf_{feat.name}")
        n = cdf.shape[0]
        scale = torch.linspace(0.0, 1.0, n, device=cdf.device)
        p = pct.float().clamp(0.0, 1.0)
        idx = torch.searchsorted(scale.contiguous(), p.contiguous()).clamp(1, n - 1)
        y0, y1 = scale[idx - 1], scale[idx]
        x0, x1 = cdf[idx - 1], cdf[idx]
        t = (p - y0) / (y1 - y0).clamp(min=1e-6)
        return x0 + t * (x1 - x0)

    def _logits_to_value(
        self, logits: torch.Tensor, feat: EmbeddingFeature
    ) -> torch.Tensor:
        if feat.is_continuous_sinusoidal:
            pct = logits.squeeze(-1)
            if feat.uses_percentile_space:
                return self._percentile_to_raw(pct, feat)
            return pct
        centers = getattr(self, f"_bucket_centers_{feat.name}")
        if feat.feature_type == EmbeddingFeature.Type.CATEGORICAL:
            return logits.argmax(dim=-1).float() + feat.min_value
        probs = torch.softmax(logits, dim=-1)
        value = (probs * centers).sum(dim=-1)
        if feat.uses_percentile_space:
            return self._percentile_to_raw(value, feat)
        return value

    def _encode_continuous(
        self, value: torch.Tensor, feat: EmbeddingFeature, is_training: bool = False
    ) -> torch.Tensor:
        if feat.uses_percentile_space:
            enc_value = self._raw_to_percentile(value, feat)
            enc_min, enc_max = 0.0, 1.0
        else:
            enc_value = value
            enc_min, enc_max = feat.min_value, feat.max_value
        sinusoidal = sinusoidal_encode(enc_value, self.sinusoidal_dim, enc_min, enc_max)
        projection = self.continuous_feature_projections[feat.name]
        return projection(sinusoidal.to(projection[0].weight.dtype))

    def _clamp_continuous_value(
        self, value: torch.Tensor, feat: EmbeddingFeature
    ) -> torch.Tensor:
        if not feat.clamp_boundary_buckets or feat.noise_sigma <= 0:
            return value
        margin = feat.noise_sigma / 2.0
        if feat.uses_percentile_space:
            pct_lo, pct_hi = margin, 1.0 - margin
            if pct_lo >= pct_hi:
                return value
            lo_raw = self._percentile_to_raw(
                torch.tensor([pct_lo], device=value.device), feat
            ).item()
            hi_raw = self._percentile_to_raw(
                torch.tensor([pct_hi], device=value.device), feat
            ).item()
            return value.clamp(min=lo_raw, max=hi_raw)
        lo = feat.min_value + margin
        hi = feat.max_value - margin
        if lo >= hi:
            lo, hi = feat.min_value, feat.max_value
        return value.clamp(min=lo, max=hi)

    def _clamp_boundary_bucket(
        self, bucket_idx: torch.Tensor, feat: EmbeddingFeature
    ) -> torch.Tensor:
        if (
            feat.feature_type == EmbeddingFeature.Type.CONTINUOUS_BUCKETED
            and feat.clamp_boundary_buckets
            and feat.num_classes > 2
        ):
            return bucket_idx.clamp(1, feat.num_classes - 2)
        return bucket_idx

    def encode_feature(
        self,
        value: torch.Tensor,
        feat: EmbeddingFeature,
        is_training: bool = False,
        apply_clamp: bool = False,
    ) -> torch.Tensor:
        if feat.is_continuous_sinusoidal:
            if apply_clamp:
                value = self._clamp_continuous_value(value, feat)
            return self._encode_continuous(value, feat, is_training=is_training)
        bucket_idx = self._bucketize(value, feat)
        if apply_clamp:
            bucket_idx = self._clamp_boundary_bucket(bucket_idx, feat)
        return self.feature_embeddings[feat.name](bucket_idx)

    def resolve_for_inference(
        self,
        predicted_logits: dict[str, torch.Tensor],
        features: Optional[dict[str, torch.Tensor]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if features is None:
            features = {}
        resolved_features: dict[str, torch.Tensor] = {}
        feature_embeds: dict[str, torch.Tensor] = {}
        for feat in self.features_config:
            value = features.get(feat.name)
            is_predicted = value is None
            if feat.is_continuous_sinusoidal:
                if value is not None:
                    resolved_features[feat.name] = value.float()
                else:
                    resolved_features[feat.name] = self._logits_to_value(
                        predicted_logits[feat.name], feat
                    )
                feature_embeds[feat.name] = self.encode_feature(
                    resolved_features[feat.name],
                    feat,
                    is_training=False,
                    apply_clamp=is_predicted,
                )
            else:
                if value is not None:
                    resolved_features[feat.name] = value.float()
                    bucket_indices = self._bucketize(value, feat)
                else:
                    resolved_features[feat.name] = self._logits_to_value(
                        predicted_logits[feat.name], feat
                    )
                    bucket_indices = self._clamp_boundary_bucket(
                        self._bucketize(resolved_features[feat.name], feat), feat
                    )
                feature_embeds[feat.name] = self.feature_embeddings[feat.name](
                    bucket_indices
                )
        return feature_embeds, resolved_features


_INPUT_LAYER_REGISTRY: dict[str, type[nn.Module]] = {
    "TargetMelSpectrogram": TargetMelSpectrogram,
}


class UnifiedSpkEmbeddingWithDec(nn.Module):
    """Inference-only port of GPTTTS ``UnifiedSpkEmbeddingWithDec``."""

    def __init__(
        self,
        features_config: list[Union[dict, EmbeddingFeature]],
        in_channels: int = 100,
        dim: int = 512,
        nb_speaker_features: int = 8,
        dec_dim: int = 1024,
        features_dim: int = 1024,
        encoder_num_layers: int = 6,
        decoder_num_layers: int = 6,
        feature_predictor_num_layers: int = 4,
        conv1_dim: int = 64,
        conv2_dim: int = 128,
        input_layer_config: Optional[dict] = None,
        use_flash_att: bool = True,
        use_post_norm: bool = False,
        use_post_norm_encoder: bool = False,
        use_out_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.nb_speaker_features = nb_speaker_features
        self.features_dim = features_dim
        self.dec_dim = dec_dim

        if input_layer_config is None:
            input_layer_config = {
                "class_name": "TargetMelSpectrogram",
                "n_mel_channels": 80,
                "sampling_rate": 16000,
                "mel_fmax": 8000,
                "do_normalization": True,
            }

        self.feature_processor = FeatureProcessor(
            features_config=features_config, dec_dim=dec_dim, predictor_input_dim=dim
        )

        self.spk_feature_keys = nn.Parameter(
            torch.randn(1, nb_speaker_features, dec_dim)
        )

        self._downsample_factor = 4
        self.conv_downsample2d = nn.Sequential(
            einops.layers.torch.Rearrange("b f t -> b () f t"),
            nn.Conv2d(1, conv1_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv1_dim, conv2_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            einops.layers.torch.Rearrange("b c f t -> b t (c f)"),
            nn.Linear(conv2_dim * in_channels // 4, dim),
        )

        self.feats_predictor = nn.ModuleList(
            [
                ConformerBlock(
                    dim=self.dim,
                    dim_head=64,
                    heads=8,
                    ff_mult=4,
                    conv_expansion_factor=2,
                    conv_kernel_size=32,
                    conv_causal=False,
                    use_post_norm=use_post_norm,
                    use_flash_att=use_flash_att,
                )
                for _ in range(feature_predictor_num_layers)
            ]
        )

        self.encoder = nn.ModuleList(
            [
                ConformerBlock(
                    dim=self.dim,
                    dim_head=64,
                    heads=8,
                    ff_mult=4,
                    conv_expansion_factor=2,
                    conv_kernel_size=32,
                    conv_causal=False,
                    use_post_norm=use_post_norm_encoder,
                    use_flash_att=use_flash_att,
                )
                for _ in range(encoder_num_layers)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                ConformerNoConvDecoderBlock(
                    dim=dec_dim,
                    enc_dim=dim,
                    dim_head=64,
                    heads=8,
                    ff_mult=4,
                    use_post_norm=use_post_norm,
                    use_flash_att=use_flash_att,
                )
                for _ in range(decoder_num_layers)
            ]
        )

        self.o = (
            nn.Linear(dec_dim, features_dim)
            if features_dim != dec_dim
            else nn.Identity()
        )
        self.out_norm = nn.LayerNorm(features_dim) if use_out_norm else nn.Identity()

        input_layer_cls_name = input_layer_config.get(
            "class_name", "TargetMelSpectrogram"
        )
        if input_layer_cls_name not in _INPUT_LAYER_REGISTRY:
            raise ValueError(
                f"Unsupported input_layer class_name '{input_layer_cls_name}'."
            )
        layer_kwargs = {
            k: v for k, v in input_layer_config.items() if k != "class_name"
        }
        self.input_layer = _INPUT_LAYER_REGISTRY[input_layer_cls_name](**layer_kwargs)
        self.input_layer.requires_grad_(False)

    @property
    def features_config(self) -> list[EmbeddingFeature]:
        return self.feature_processor.features_config

    def _run_mel_predictor(self, x_down, enc_mask):
        feats = x_down.detach()
        for layer in self.feats_predictor:
            feats = layer(feats, enc_mask, use_extended_mask=False)
        fp = self.feature_processor
        return {
            feat.name: fp.feature_predictor_heads[feat.name](feats, enc_mask)
            for feat in self.features_config
        }

    def _run_decoder(self, enc_output, enc_mask, feature_embeds, B):
        dec_input = self.spk_feature_keys.expand(B, -1, -1)
        for layer in self.decoder:
            dec_input = layer(
                x=dec_input,
                encoder_hidden_states=enc_output,
                encoder_attention_mask=enc_mask,
            )
        spk_features = self.out_norm(self.o(dec_input))
        if self.features_config:
            feat_embeds_t = torch.cat(
                [feature_embeds[f.name].unsqueeze(1) for f in self.features_config],
                dim=1,
            )
            spk_features = torch.cat(
                [spk_features, self.out_norm(self.o(feat_embeds_t))], dim=1
            )
        return spk_features.contiguous()

    def _extract_spk_features(self, x, features=None, mask=None):
        B = x.shape[0]
        enc_mask = None if mask is None else mask[:, :: self._downsample_factor]
        x_down = self.conv_downsample2d(x)
        predicted_logits = self._run_mel_predictor(x_down, enc_mask)
        feature_embeds, resolved_features = (
            self.feature_processor.resolve_for_inference(predicted_logits, features)
        )
        enc_output = x_down
        for layer in self.encoder:
            enc_output = layer(enc_output, enc_mask, use_extended_mask=False)
        spk_features = self._run_decoder(enc_output, enc_mask, feature_embeds, B)
        return spk_features, resolved_features

    @torch.inference_mode()
    def get_spk_features(self, x, features=None, mask=None):
        return self._extract_spk_features(x, features, mask)

    def extract_spk_features(self, x, features=None, mask=None):
        return self._extract_spk_features(x, features, mask)

    def update_feature_embeddings(
        self, emb: torch.Tensor, features_to_update: list[dict[str, float]]
    ) -> torch.Tensor:
        if len(features_to_update) != emb.shape[0]:
            raise ValueError(
                f"Expected {emb.shape[0]} feature dicts, got {len(features_to_update)}"
            )
        B = emb.shape[0]
        device = emb.device
        result = emb.clone()
        fp = self.feature_processor
        for feat_offset, feat in enumerate(self.features_config):
            token_idx = self.nb_speaker_features + feat_offset
            for b in range(B):
                if feat.name not in features_to_update[b]:
                    continue
                value = torch.tensor(
                    [features_to_update[b][feat.name]], device=device, dtype=emb.dtype
                )
                raw_emb = fp.encode_feature(
                    value, feat, is_training=False, apply_clamp=True
                ).unsqueeze(1)
                new_emb = self.out_norm(self.o(raw_emb))
                result[b, token_idx, :] = new_emb.squeeze(0).squeeze(0)
        return result
