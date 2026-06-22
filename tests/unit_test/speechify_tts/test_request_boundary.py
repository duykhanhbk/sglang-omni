# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for the SpeechifyTTS request boundary + abort cleanup."""

from __future__ import annotations

import pytest

from sglang_omni.models.speechify_tts import request_builders as rb
from sglang_omni.models.speechify_tts.request_builders import (
    SPEECHIFY_DEFAULT_REPETITION_PENALTY,
    SPEECHIFY_DEFAULT_TEMPERATURE,
    SPEECHIFY_DEFAULT_TOP_K,
    build_speechify_tts_state,
    cleanup_prepared_speechify_request,
)
from sglang_omni.proto import OmniRequest, StagePayload


def _payload(inputs, params=None, tts_params=None, request_id="r1"):
    request = OmniRequest(
        inputs=inputs,
        params=params or {},
        metadata={"task": "tts", "tts_params": tts_params or {}},
    )
    return StagePayload(request_id=request_id, request=request, data=None)


def test_endpoint_sampling_defaults_do_not_override_model_defaults():
    # speech_service always fills temperature/top_k; without explicit markers
    # the SpeechifyTTS defaults must win.
    state = build_speechify_tts_state(
        _payload(
            "Hello world",
            params={"temperature": 0.99, "top_k": 30, "repetition_penalty": 1.1},
            tts_params={"ref_audio": "/tmp/ref.wav"},
        )
    )
    assert state.temperature == SPEECHIFY_DEFAULT_TEMPERATURE
    assert state.top_k == SPEECHIFY_DEFAULT_TOP_K
    assert state.repetition_penalty == SPEECHIFY_DEFAULT_REPETITION_PENALTY


def test_explicit_generation_params_are_honored():
    state = build_speechify_tts_state(
        _payload(
            {"text": "Hi", "references": [{"ref_audio": "gs://b/x.wav"}]},
            params={
                "temperature": 0.5,
                "top_k": 7,
                "repetition_penalty": 1.3,
                "max_new_tokens": 1000,
                "stream": True,
            },
            tts_params={
                "explicit_generation_params": ["temperature", "top_k", "repetition_penalty"],
                "speed": 1.4,
            },
        )
    )
    assert state.temperature == 0.5
    assert state.top_k == 7
    assert state.repetition_penalty == 1.3
    assert state.max_new_tokens == 1000  # not part of generic default set
    assert state.speaking_rate == 1.4
    assert state.streaming is True
    assert state.ref_audio == "gs://b/x.wav"


def test_reference_resolution_prefers_inputs_then_tts_params():
    s1 = build_speechify_tts_state(
        _payload({"text": "x", "references": [{"audio_path": "/a.wav"}]})
    )
    assert s1.ref_audio == "/a.wav"
    s2 = build_speechify_tts_state(_payload("x", tts_params={"ref_audio": "/b.wav"}))
    assert s2.ref_audio == "/b.wav"


def test_missing_reference_raises():
    with pytest.raises(ValueError, match="voice reference"):
        build_speechify_tts_state(_payload("Hi", tts_params={"voice": "default"}))


def test_empty_text_raises():
    with pytest.raises(ValueError, match="non-empty"):
        build_speechify_tts_state(_payload("   ", tts_params={"ref_audio": "/r.wav"}))


def test_uploaded_voice_name_satisfies_reference_requirement():
    # A non-default named voice (uploaded) is a valid reference.
    state = build_speechify_tts_state(_payload("Hi", tts_params={"voice": "my_voice"}))
    assert state.ref_audio is None
    assert state.text == "Hi"


def test_mtl_alignment_stop_defaults():
    state = build_speechify_tts_state(_payload("Hi", tts_params={"ref_audio": "/r.wav"}))
    assert state.body_end_anchor_offset == 1
    assert state.align_stop_offset == 1
    assert state.alignment_plateau_max_steps == 30


def test_abort_cleanup_is_idempotent_and_covers_race_paths():
    rb.set_speechify_preprocessing_context()  # reset bookkeeping
    # Path A: abort with nothing stashed is a no-op.
    cleanup_prepared_speechify_request("missing")
    # Path B: abort while in-flight marks it so a late finish is dropped.
    with rb._LOCK:
        rb._INFLIGHT_REQUESTS.add("r2")
    cleanup_prepared_speechify_request("r2")
    cleanup_prepared_speechify_request("r2")  # idempotent
    with rb._LOCK:
        assert "r2" in rb._ABORTED_REQUESTS
    # Path C: a stashed result is freed on abort.
    with rb._LOCK:
        rb._PREPARED_REQUESTS["r3"] = object()  # type: ignore[assignment]
    cleanup_prepared_speechify_request("r3")
    with rb._LOCK:
        assert "r3" not in rb._PREPARED_REQUESTS
