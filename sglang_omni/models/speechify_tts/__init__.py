# SPDX-License-Identifier: Apache-2.0
"""SpeechifyTTS (speechify_t5_tts) model package for sglang-omni.

A 3-stage zero-shot voice-cloning TTS pipeline ported from vllm-omni:

- preprocessing : T5-style text encoder + Unified speaker-embedding extraction.
- tts_engine    : Gemma-style MoE autoregressive decoder (mel codes + alignment).
- vocoder       : DiffiTv3 flow-matching diffusion + FlowVAE -> 24 kHz PCM.

The subpackage is auto-discovered by ``sglang_omni.models.registry`` via the
``EntryClass`` exported from :mod:`sglang_omni.models.speechify_tts.config`.
"""
