"""DiffiT architecture spec dataclasses.

Historically this file contained the full TRT engine builders
(``RawTRTDenoiserV3StepBuilder``, ``RawTRTEncoderBuilder`` and their
weight loaders) along with the spec dataclasses used by the runtime
to size caches. After the cuda-graph rewrite, none of the diffusion
TRT engines are mmapped or executed — the builders / weight loaders /
network helpers are dead code and have been removed. What remains:

* ``DenoiserSpec`` — denoiser architecture spec, read from
  ``config.json``. Used by the cuda-graph runtime to size KV / conv
  caches and to know cross-attention / sliding-window layer indices.
* ``EncoderSpec`` — same for the aligned-latent encoder.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple


DEFAULT_PROMPT_LEN = 256


@dataclass(frozen=True)
class DenoiserSpec:
    """Architecture spec matching the production DiffiTModelV3 denoiser."""

    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    latent_channels: int
    input_channels: int
    out_channels: int
    ff_mult: float
    cross_attention_layers: Tuple[int, ...]
    full_chunk_layers: Tuple[int, ...]
    causal_conv_layers: Tuple[int, ...]
    max_cache_frames: int = 512
    conv_kernel_size: int = 3
    norm_elementwise_affine: bool = False

    @property
    def kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def ff_inner_dim(self) -> int:
        return int(self.hidden_size * self.ff_mult)

    def conv_cache_len(self, layer_idx: int) -> int:
        if layer_idx in self.causal_conv_layers:
            return self.conv_kernel_size - 1
        return 1

    def conv_padding(self, layer_idx: int) -> int:
        return 0 if layer_idx in self.causal_conv_layers else 1

    def inp_conv_cache_len(self) -> int:
        return 1

    def inp_conv_padding(self) -> int:
        return 1

    def with_max_cache_frames(self, max_cache_frames: int) -> "DenoiserSpec":
        return replace(self, max_cache_frames=max_cache_frames)

    @classmethod
    def from_config(cls, diffusion_dir: str) -> "DenoiserSpec":
        import json
        import os

        with open(os.path.join(diffusion_dir, "config.json")) as f:
            cfg = json.load(f)
        num_heads = cfg["num_attention_heads"]
        head_dim = cfg["attention_head_dim"]
        num_layers = cfg["num_layers"]
        hidden = num_heads * head_dim
        cross_attn_indices = cfg.get("cross_attention_layer_indices")
        if cross_attn_indices is None:
            cross_attn_indices = list(range(num_layers))
        sliding_window_indices = cfg.get("sliding_window_layer_indices") or []
        causal_conv_indices = cfg.get("causal_conv_layer_indices") or []
        return cls(
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            hidden_size=hidden,
            latent_channels=cfg.get("latent_channels", hidden),
            input_channels=cfg.get("input_channels", 64),
            out_channels=cfg.get("output_channels", 64),
            ff_mult=cfg.get("ff_mult", 2.67),
            cross_attention_layers=tuple(sorted(cross_attn_indices)),
            full_chunk_layers=tuple(sorted(sliding_window_indices)),
            causal_conv_layers=tuple(sorted(causal_conv_indices)),
            norm_elementwise_affine=cfg.get("norm_elementwise_affine", False),
        )


@dataclass(frozen=True)
class EncoderSpec:
    """Architecture spec for the aligned_latent_encoder (SimpleTransformerEncoder)."""

    num_layers: int
    num_heads: int
    head_dim: int
    hidden_size: int
    ff_mult: float
    max_cache_tokens: int = 512
    norm_elementwise_affine: bool = True

    @property
    def ff_inner_dim(self) -> int:
        return int(self.hidden_size * self.ff_mult)

    def with_max_cache_tokens(self, n: int) -> "EncoderSpec":
        return replace(self, max_cache_tokens=n)

    @classmethod
    def from_config(cls, diffusion_dir: str) -> "EncoderSpec":
        import json
        import os

        with open(os.path.join(diffusion_dir, "config.json")) as f:
            cfg = json.load(f)
        num_heads = cfg.get("latent_encoder_num_heads", 16)
        head_dim = cfg.get("attention_head_dim", 64)
        return cls(
            num_layers=cfg.get("latent_encoder_num_layers", 6),
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_size=num_heads * head_dim,
            ff_mult=cfg.get("ff_mult", 2.67),
            norm_elementwise_affine=True,
        )
