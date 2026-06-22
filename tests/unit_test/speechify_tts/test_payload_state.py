# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for the SpeechifyTTS cross-stage state serialization."""

from __future__ import annotations

import torch

from sglang_omni.models.speechify_tts.payload_types import SpeechifyTTSState


def test_roundtrip_scalar_fields():
    state = SpeechifyTTSState(
        text="hello",
        decoder_prompt_token_ids=[17061],
        text_seq_len=7,
        temperature=0.7,
        top_p=0.85,
        top_k=12,
        repetition_penalty=1.8,
        max_new_tokens=1500,
        streaming=True,
        speaking_rate=1.25,
        body_end_anchor_offset=1,
        align_stop_offset=1,
    )
    out = SpeechifyTTSState.from_dict(state.to_dict())
    assert out.text == "hello"
    assert out.decoder_prompt_token_ids == [17061]
    assert out.text_seq_len == 7
    assert out.temperature == 0.7
    assert out.top_k == 12
    assert out.max_new_tokens == 1500
    assert out.streaming is True
    assert out.speaking_rate == 1.25
    assert out.body_end_anchor_offset == 1


def test_tensors_are_detached_to_cpu_in_payload():
    latents = torch.randn(8, 1536)
    mel = torch.randint(0, 2048, (8,))
    state = SpeechifyTTSState(
        text="x",
        text_hidden_states=torch.randn(5, 1536),
        speaker_embedding=torch.randn(10, 1024),
        decoder_latents=latents,
        mel_codes=mel,
        alignments=torch.randn(8),
    )
    data = state.to_dict()
    # to_dict keeps tensors (detached, CPU) for the SHM relay, not python lists.
    assert isinstance(data["decoder_latents"], torch.Tensor)
    assert data["decoder_latents"].device.type == "cpu"
    assert not data["decoder_latents"].requires_grad

    out = SpeechifyTTSState.from_dict(data)
    assert torch.allclose(out.decoder_latents, latents)
    assert torch.equal(out.mel_codes, mel)
    assert tuple(out.speaker_embedding.shape) == (10, 1024)


def test_optional_fields_omitted_when_unset():
    data = SpeechifyTTSState(text="x").to_dict()
    for key in (
        "ref_audio",
        "speaking_rate",
        "speaker_embedding",
        "mel_codes",
        "audio_samples",
        "body_end_mel_codes",
        "finish_reason",
    ):
        assert key not in data


def test_from_dict_tolerates_non_dict():
    out = SpeechifyTTSState.from_dict(None)
    assert out.text == ""
    assert out.sample_rate == 24000
