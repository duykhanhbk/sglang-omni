# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for SpeechifyTTS (speechify_t5_tts).

3-stage pipeline:

    preprocessing -> tts_engine -> vocoder

- ``preprocessing`` runs the T5-style text encoder and Unified speaker-embedding
  extraction on the AR GPU, producing the decoder prompt + cross-attention
  state. Heavy work is kept here so the AR loop is not stalled.
- ``tts_engine`` is the Gemma-style MoE autoregressive decoder. It emits mel
  codes plus the per-step ``latent`` / ``aligned_encoder_latent`` / ``alignment``
  tensors the diffusion vocoder consumes, and streams them to the vocoder.
- ``vocoder`` runs the DiffiTv3 flow-matching diffusion + FlowVAE to produce
  24 kHz PCM. It accepts streamed chunks before the AR finishes.

Streaming (``stream_to`` / ``can_accept_stream_before_payload``) mirrors the
vllm-omni ``async_chunk: true`` topology: the AR decoder pushes 4-mel-code
chunks to the vocoder while it keeps generating, giving low time-to-first-audio.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.speechify_tts"

# Default per-process GPU budget when the three stages are colocated on one GPU.
_AR_MEM_FRACTION_STATIC = 0.45


def _stages(*, gpu: int = 0, dtype: str = "bfloat16") -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={"gpu_id": gpu, "dtype": dtype},
            gpu=gpu,
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={"gpu_id": gpu, "dtype": dtype},
            gpu=gpu,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={"gpu_id": gpu, "dtype": dtype},
            gpu=gpu,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


class SpeechifyTTSPipelineConfig(PipelineConfig):
    """Single-GPU 3-stage SpeechifyTTS pipeline (MoE 4B MTL and dense EN)."""

    architecture: ClassVar[str] = "SpeechifyT5TTSEncoderForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "SpeechifyT5TTSDecoderForConditionalGeneration",
        "SpeechifyTTSForConditionalGeneration",
    )

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "tts_engine"}

    model_path: str
    stages: list[StageConfig] = Field(default_factory=_stages)

    # AR-engine CUDA-graph knobs (MoE decoder full graphs). Default on.
    cuda_graph: bool = True
    cuda_graph_max_bs: int = 8

    # Diffusion-vocoder CUDA-graph knobs (bucketed denoiser / VAE graphs).
    diffusion_cuda_graph: bool = True
    diffusion_kv_cache_frames: int = 512
    diffusion_chunk_size: int = 4
    diffusion_min_free_gb: float = 3.0

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        if self.diffusion_min_free_gb < 0:
            raise ValueError(
                "diffusion_min_free_gb must be >= 0 (0 disables the VRAM guard); "
                f"got {self.diffusion_min_free_gb}"
            )
        if self.cuda_graph_max_bs < 1:
            raise ValueError("cuda_graph_max_bs must be >= 1")
        for stage in self.stages:
            if stage.factory.endswith("create_sglang_tts_engine_executor"):
                stage.factory_args.setdefault("cuda_graph", self.cuda_graph)
                stage.factory_args.setdefault(
                    "cuda_graph_max_bs", self.cuda_graph_max_bs
                )
            elif stage.factory.endswith("create_vocoder_executor"):
                stage.factory_args.setdefault("cuda_graph", self.diffusion_cuda_graph)
                stage.factory_args.setdefault(
                    "kv_cache_frames", self.diffusion_kv_cache_frames
                )
                stage.factory_args.setdefault("chunk_size", self.diffusion_chunk_size)
                stage.factory_args.setdefault(
                    "cuda_graph_min_free_gb", self.diffusion_min_free_gb
                )

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = SpeechifyTTSPipelineConfig
