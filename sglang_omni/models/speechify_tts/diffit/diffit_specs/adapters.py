"""DiffiT spec container exposed to the vocoder."""

from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.models.speechify_tts.diffit.diffit_specs.builder import (
    DenoiserSpec,
    EncoderSpec,
)


@dataclass(frozen=True)
class DiffiTSpecs:
    """Public configuration surface for DiffiT cache sizing.

    Exposed via ``DiffiTModelV3.get_diffit_specs()`` so the vocoder
    doesn't need to reach into private spec internals.
    """

    denoiser_spec: DenoiserSpec
    encoder_spec: EncoderSpec
    prompt_len: int
    window_frames: int
