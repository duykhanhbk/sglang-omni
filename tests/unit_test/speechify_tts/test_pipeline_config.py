# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for the SpeechifyTTS pipeline + HF config wiring."""

from __future__ import annotations

from sglang_omni.models.speechify_tts.config import SpeechifyTTSPipelineConfig
from sglang_omni.models.speechify_tts.configuration import (
    SpeechifyT5TTSConfig,
    register_speechify_hf_configs,
)


def _config() -> SpeechifyTTSPipelineConfig:
    return SpeechifyTTSPipelineConfig(model_path="/tmp/simba3-moe4b")


def test_pipeline_has_three_streaming_stages():
    cfg = _config()
    names = [s.name for s in cfg.stages]
    assert names == ["preprocessing", "tts_engine", "vocoder"]

    by_name = {s.name: s for s in cfg.stages}
    assert by_name["preprocessing"].next == "tts_engine"
    # Engine streams chunks to the vocoder; vocoder accepts pre-payload streams.
    assert by_name["tts_engine"].next == "vocoder"
    assert by_name["tts_engine"].stream_to == ["vocoder"]
    assert by_name["vocoder"].terminal is True
    assert by_name["vocoder"].can_accept_stream_before_payload is True


def test_pipeline_architecture_and_aliases():
    assert (
        SpeechifyTTSPipelineConfig.architecture
        == "SpeechifyT5TTSEncoderForConditionalGeneration"
    )
    assert (
        "SpeechifyT5TTSDecoderForConditionalGeneration"
        in SpeechifyTTSPipelineConfig.architecture_aliases
    )


def test_cuda_graph_knobs_propagate_to_factory_args():
    cfg = SpeechifyTTSPipelineConfig(
        model_path="/tmp/x",
        cuda_graph=False,
        cuda_graph_max_bs=4,
        diffusion_kv_cache_frames=256,
        diffusion_chunk_size=8,
    )
    eng = next(s for s in cfg.stages if s.name == "tts_engine").factory_args
    voc = next(s for s in cfg.stages if s.name == "vocoder").factory_args
    assert eng["cuda_graph"] is False
    assert eng["cuda_graph_max_bs"] == 4
    assert voc["kv_cache_frames"] == 256
    assert voc["chunk_size"] == 8


def test_supports_uploaded_voice_references():
    assert _config().supports_uploaded_voice_references() is True


def test_hf_config_derived_token_ids_moe4b():
    register_speechify_hf_configs()
    cfg = SpeechifyT5TTSConfig(
        vocab_size=17062,
        number_text_tokens=15006,
        number_mel_codes=2048,
        enable_speaking_rate_ift=True,
        speaking_rate_vocab_size=5,
        decoder_start_token_id=17061,
        decoder_config={
            "hidden_size": 1536,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_experts": 48,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 1920,
            "num_dense_layers": 2,
        },
        extra_conds_config={
            "class_name": "UnifiedSpkEmbeddingWithDec",
            "nb_speaker_features": 6,
            "features_config": [
                {"name": "adv_a"},
                {"name": "adv_d"},
                {"name": "adv_v"},
                {"name": "spk_rate"},
            ],
        },
    )
    # speaking-rate IFT adds 5 to mel codes -> decoder EOS at 2053.
    assert cfg.effective_number_mel_codes == 2053
    assert cfg.decoder_eos_token_id == 2053
    assert cfg.audio_token_offset == 15007
    assert cfg.full_vocab_eos_token_id == 15006 + 2053 + 1
    # Top-level probes resolve to the decoder backbone.
    assert cfg.num_attention_heads == 12
    assert cfg.hidden_size == 1536
    assert cfg.num_hidden_layers == 12
    # Unified speaker tower exposes 6 + 4 cross-attention tokens.
    assert cfg.extra_conds_config.is_unified is True
    assert cfg.extra_conds_config.num_speaker_tokens == 10
    # MoE knobs survive the round trip.
    assert cfg.decoder_config.num_experts == 48
    assert cfg.decoder_config.num_experts_per_tok == 4
    assert cfg.decoder_config.num_dense_layers == 2
