"""DiffiT architecture spec metadata.

Post-cuda-graph-rewrite: no TRT engines are built, mmapped, or
executed at inference time. What remains:

* ``DenoiserSpec`` / ``EncoderSpec`` — architecture specs read from
  ``config.json``. Used by the cuda-graph runtime to size KV / conv
  caches and route cross-attention layers.
* ``load_denoiser_spec`` / ``load_encoder_spec`` — return
  ``(spec, plan_metadata)`` pairs computed from ``config.json``.
* ``DiffiTSpecs`` — public container exposed to the vocoder.
"""

from sglang_omni.models.speechify_tts.diffit.diffit_specs.adapters import (
    DiffiTSpecs,
)
from sglang_omni.models.speechify_tts.diffit.diffit_specs.builder import (
    DenoiserSpec,
    EncoderSpec,
)
from sglang_omni.models.speechify_tts.diffit.diffit_specs.engine import (
    DenoiserPlanMetadata,
    EncoderPlanMetadata,
    load_denoiser_spec,
    load_encoder_spec,
)

__all__ = [
    "DenoiserPlanMetadata",
    "DenoiserSpec",
    "DiffiTSpecs",
    "EncoderPlanMetadata",
    "EncoderSpec",
    "load_denoiser_spec",
    "load_encoder_spec",
]
