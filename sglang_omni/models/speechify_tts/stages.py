# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the SpeechifyTTS pipeline.

Three factories referenced by ``config.py``:

- ``create_preprocessing_executor`` -> SimpleScheduler running the text encoder
  + Unified speaker-embedding extraction (request boundary fully implemented;
  the encoder/speaker forward is deferred to the GPU port).
- ``create_sglang_tts_engine_executor`` -> OmniScheduler around the SGLang MoE
  AR decoder. The SGLang ``ServerArgs`` wiring below mirrors the vllm-omni MoE
  4B deploy yaml (full CUDA graphs on the decoder, mem fraction 0.45, AR
  sampling temp 0.8 / top_p 0.8 / top_k 10 / repetition_penalty 2.0). The model
  instantiation + alignment-stop runner is deferred to the GPU port.
- ``create_vocoder_executor`` -> streaming DiffiTv3 + FlowVAE scheduler
  (deferred to the GPU port).

Deferred boundaries raise ``NotImplementedError`` with a pointer rather than
failing obscurely; everything up to that line is real and reviewable.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from sglang_omni.models.speechify_tts.configuration import (
    register_speechify_hf_configs,
)
from sglang_omni.models.speechify_tts.request_builders import (
    cleanup_prepared_speechify_request,
    preprocess_speechify_payload,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

logger = logging.getLogger(__name__)

# SGLang AR architecture key registered in sglang_model_runner._register_omni_model.
_AR_MODEL_ARCH = "SpeechifyTTSDecoder"


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def _resolve_ar_subdir(checkpoint_dir: str) -> str:
    """SpeechifyTTS ships AR weights under ``ar/`` (vllm-omni layout)."""
    ar_dir = os.path.join(checkpoint_dir, "ar")
    return ar_dir if os.path.isdir(ar_dir) else checkpoint_dir


def _resolve_diffusion_subdir(checkpoint_dir: str) -> str:
    diff_dir = os.path.join(checkpoint_dir, "diffusion")
    return diff_dir if os.path.isdir(diff_dir) else checkpoint_dir


# --------------------------------------------------------------------------- #
# Stage 0: preprocessing (text encoder + speaker extraction)                  #
# --------------------------------------------------------------------------- #
def create_preprocessing_executor(
    model_path: str,
    *,
    gpu_id: int = 0,
    dtype: str = "bfloat16",
) -> SimpleScheduler:
    """Threaded preprocessing: request mapping + encoder + speaker extraction.

    The request-boundary mapping is active immediately; the encoder + speaker
    forward is wired through ``set_speechify_preprocessing_context`` once the
    deep model internals are ported (see ``request_builders`` /package docstring).
    """
    register_speechify_hf_configs()
    # TODO(GPU port): load the T5-style text encoder + Unified speaker extractor
    # onto cuda:{gpu_id} and call set_speechify_preprocessing_context(...).
    del dtype, gpu_id
    return SimpleScheduler(
        preprocess_speechify_payload,
        abort_callback=cleanup_prepared_speechify_request,
    )


# --------------------------------------------------------------------------- #
# Stage 1: tts_engine (Gemma-style MoE AR decoder)                            #
# --------------------------------------------------------------------------- #
def build_speechify_ar_server_args(
    checkpoint_dir: str,
    *,
    dtype: str = "bfloat16",
    cuda_graph: bool = True,
    cuda_graph_max_bs: int = 8,
    max_model_len: int = 4096,
    server_args_overrides: dict[str, Any] | None = None,
):
    """Build SGLang ServerArgs for the SpeechifyTTS MoE AR decoder.

    Mirrors ``speechify_t5_tts_moe4b_production.yaml`` stage 1: full CUDA graphs
    on the MoE decoder, mem fraction 0.45, single-stream batch, and the
    ``fused_experts`` MoE path (``VLLM_OMNI_DISABLE_GROUPED_MM`` analog is the
    SGLang FusedMoE default).
    """
    from sglang_omni.scheduling.generation_batch_policy import (
        build_default_cuda_graph_bs,
        sync_cuda_graph_bs_with_max_bs,
    )
    from sglang_omni.scheduling.sglang_backend import build_sglang_server_args

    overrides: dict[str, Any] = {
        "cuda_graph_bs": build_default_cuda_graph_bs(cuda_graph_max_bs),
        "cuda_graph_max_bs": cuda_graph_max_bs,
        "disable_cuda_graph": not cuda_graph,
        "disable_overlap_schedule": True,
        "dtype": dtype,
        "mem_fraction_static": 0.45,
        "max_running_requests": cuda_graph_max_bs,
        "trust_remote_code": True,
        "sampling_backend": "pytorch",
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)
        sync_cuda_graph_bs_with_max_bs(overrides, server_args_overrides)

    return build_sglang_server_args(
        checkpoint_dir,
        context_length=max_model_len,
        **overrides,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    gpu_id: int = 0,
    dtype: str = "bfloat16",
    cuda_graph: bool = True,
    cuda_graph_max_bs: int = 8,
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    """Stand up the SGLang MoE AR decoder engine (OmniScheduler)."""
    register_speechify_hf_configs()
    checkpoint_dir = _resolve_checkpoint(model_path)
    ar_dir = _resolve_ar_subdir(checkpoint_dir)

    server_args = build_speechify_ar_server_args(
        ar_dir,
        dtype=dtype,
        cuda_graph=cuda_graph,
        cuda_graph_max_bs=cuda_graph_max_bs,
        server_args_overrides=server_args_overrides,
    )
    logger.info(
        "SpeechifyTTS AR server_args ready (ar_dir=%s, cuda_graph=%s, max_bs=%s)",
        ar_dir,
        cuda_graph,
        cuda_graph_max_bs,
    )

    # TODO(GPU port): the remainder mirrors qwen3_tts/fishaudio_s2_pro:
    #   1. want_cuda_graph = not server_args.disable_cuda_graph; flip off for load
    #   2. create_sglang_infrastructure(server_args, gpu_id,
    #          model_arch_override="SpeechifyTTSDecoder")
    #   3. install the conformer-conv state cache + dual cross-attn KV buffers
    #   4. set_speechify_preprocessing_context(...) so preprocessing can encode
    #   5. model_worker.model_runner.init_device_graphs() when want_cuda_graph
    #   6. return OmniScheduler(..., model_runner=SpeechifyTTSModelRunner(...),
    #          request_builder, result_adapter,
    #          abort_callback=cleanup_prepared_speechify_request)
    raise NotImplementedError(
        "SpeechifyTTS AR engine instantiation is deferred to the GPU port. The "
        "SGLang ServerArgs above are fully wired; the SpeechifyTTSDecoder model "
        "class, conformer-conv cache, dual cross-attention KV, alignment-stop "
        "runner, and OmniScheduler hook-up remain to be ported. See "
        "sglang_omni/models/speechify_tts/sglang_model.py (to be added)."
    )


# --------------------------------------------------------------------------- #
# Stage 2: vocoder (DiffiTv3 diffusion + FlowVAE, streaming)                   #
# --------------------------------------------------------------------------- #
def create_vocoder_executor(
    model_path: str,
    *,
    gpu_id: int = 0,
    dtype: str = "bfloat16",
    cuda_graph: bool = True,
    kv_cache_frames: int = 512,
    chunk_size: int = 4,
    cuda_graph_min_free_gb: float = 3.0,
) -> Any:
    """Streaming diffusion vocoder scheduler (deferred to the GPU port)."""
    register_speechify_hf_configs()
    checkpoint_dir = _resolve_checkpoint(model_path)
    diffusion_dir = _resolve_diffusion_subdir(checkpoint_dir)
    logger.info(
        "SpeechifyTTS vocoder config (diffusion_dir=%s, cuda_graph=%s, "
        "kv_cache_frames=%s, chunk_size=%s)",
        diffusion_dir,
        cuda_graph,
        kv_cache_frames,
        chunk_size,
    )
    del dtype, gpu_id, cuda_graph_min_free_gb
    # TODO(GPU port): load DiffiTModelV3 + FlowVAEInputLayer from diffusion_dir,
    # build the bucketed denoiser/VAE CUDA-graph runners (streaming_block_size 32),
    # and return a streaming scheduler that consumes chunk_size-mel-code chunks
    # (latent / aligned_encoder_latent / alignment) and emits 24 kHz PCM +
    # speechmarks, accepting streamed chunks before the AR finishes.
    raise NotImplementedError(
        "SpeechifyTTS diffusion vocoder is deferred to the GPU port. See "
        "sglang_omni/models/speechify_tts/diffusion_vocoder.py (to be added)."
    )
