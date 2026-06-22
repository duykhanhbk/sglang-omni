"""Spec + plan-metadata producers for the diffusion stage.

Historically this module built and persisted TRT engines for the
denoiser and aligned-latent encoder, caching them as ``.plan`` files
alongside a ``.plan.json`` sidecar carrying the cache-size metadata
the runtime needs.

After the cuda-graph rewrite no TRT engine is mmapped or executed at
inference time; the runtime only needs the metadata (max_cache_frames,
window_frames, commit_frames, prompt_len). These functions now just
synthesize the metadata + spec in memory from ``config.json``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from sglang_omni.models.speechify_tts.diffit.diffit_specs.builder import (
    DEFAULT_PROMPT_LEN,
    DenoiserSpec,
    EncoderSpec,
)

logger = logging.getLogger(__name__)


def _default_max_batch_size() -> int:
    """Authoritative batch-size source for spec metadata.

    ``VLLM_DIFFIT_MAX_SLOTS`` is the runtime knob (set by the deploy
    yaml); falls back to ``VLLM_DIFFIT_MAX_BATCH_SIZE`` (legacy name)
    or 8 (the production default).
    """
    val = os.environ.get("VLLM_DIFFIT_MAX_SLOTS")
    if val is None:
        val = os.environ.get("VLLM_DIFFIT_MAX_BATCH_SIZE", "8")
    return int(val)


DEFAULT_MAX_BATCH_SIZE = _default_max_batch_size()


@dataclass(frozen=True)
class DenoiserPlanMetadata:
    batch_size: int
    window_frames: int
    commit_frames: int
    prompt_len: int
    max_cache_frames: int


def compute_plan_metadata(
    diffusion_dir: str,
    *,
    block_num_frames: int,
    max_cache_frames: int,
    batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    prompt_len: int = DEFAULT_PROMPT_LEN,
) -> DenoiserPlanMetadata:
    """Compute denoiser plan metadata from streaming-block / cache sizes."""
    # window_frames = block_num_frames + num_future_frames (from config.json).
    # commit_frames = block_num_frames (one full streaming block per ODE step).
    import json
    with open(os.path.join(diffusion_dir, "config.json")) as f:
        cfg = json.load(f)
    num_future_frames = int(cfg.get("num_future_frames", 0))
    return DenoiserPlanMetadata(
        batch_size=batch_size,
        window_frames=block_num_frames + num_future_frames,
        commit_frames=block_num_frames,
        prompt_len=prompt_len,
        max_cache_frames=max_cache_frames,
    )


def load_denoiser_spec(
    diffusion_dir: str,
    *,
    batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    block_num_frames: int,
    max_cache_frames: int,
    prompt_len: int = DEFAULT_PROMPT_LEN,
) -> "tuple[DenoiserSpec, DenoiserPlanMetadata]":
    """Return ``(spec, metadata)`` for the denoiser."""
    metadata = compute_plan_metadata(
        diffusion_dir,
        block_num_frames=block_num_frames,
        max_cache_frames=max_cache_frames,
        batch_size=batch_size,
        prompt_len=prompt_len,
    )
    spec = DenoiserSpec.from_config(diffusion_dir).with_max_cache_frames(
        metadata.max_cache_frames,
    )
    return spec, metadata


@dataclass(frozen=True)
class EncoderPlanMetadata:
    seq_len: int
    max_cache_tokens: int = 512
    batch_size: int = DEFAULT_MAX_BATCH_SIZE


def load_encoder_spec(
    diffusion_dir: str,
    *,
    seq_len: int,
    max_cache_tokens: int = 512,
    batch_size: int = DEFAULT_MAX_BATCH_SIZE,
) -> "tuple[EncoderSpec, EncoderPlanMetadata]":
    """Return ``(spec, metadata)`` for the aligned encoder."""
    metadata = EncoderPlanMetadata(
        seq_len=seq_len,
        max_cache_tokens=max_cache_tokens,
        batch_size=batch_size,
    )
    spec = EncoderSpec.from_config(diffusion_dir).with_max_cache_tokens(
        max_cache_tokens,
    )
    return spec, metadata
