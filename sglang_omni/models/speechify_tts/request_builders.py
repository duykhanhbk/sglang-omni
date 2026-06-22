# SPDX-License-Identifier: Apache-2.0
"""Request mapping helpers for SpeechifyTTS.

This module owns the *request boundary*: turning an incoming ``/v1/audio/speech``
request (lowered to an :class:`OmniRequest` by ``speech_service.py``) into a
:class:`SpeechifyTTSState`. It deliberately keeps all heterogeneous-input
normalization, sampling-default preservation, and reference validation here so a
bad request fails before anything touches the GPU.

The GPU-coupled pieces (running the text encoder + speaker-embedding extraction
in preprocessing, and building/!resolving the SGLang AR request) are wired
through a :class:`_PreprocessingContext` set by the stage factory. Until the
deep model internals land (see ``stages.py`` and the package docstring) those
paths raise :class:`NotImplementedError` with a pointer, while the pure request
mapping below is fully functional and unit-testable without a GPU.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from sglang_omni.models.speechify_tts.payload_types import SpeechifyTTSState
from sglang_omni.proto import StagePayload

# SpeechifyTTS sampling defaults (mirrors the vllm-omni MoE 4B deploy yaml:
# temperature 0.8, top_p 0.8, top_k 10, repetition_penalty 2.0, max_tokens 2048).
SPEECHIFY_DEFAULT_TEMPERATURE = 0.8
SPEECHIFY_DEFAULT_TOP_P = 0.8
SPEECHIFY_DEFAULT_TOP_K = 10
SPEECHIFY_DEFAULT_REPETITION_PENALTY = 2.0
SPEECHIFY_DEFAULT_MAX_NEW_TOKENS = 2048

# MTL (MoE 4B) alignment-stop policy defaults; the preprocessing factory may
# overwrite these from the loaded checkpoint config.
SPEECHIFY_DEFAULT_BODY_END_ANCHOR_OFFSET = 1
SPEECHIFY_DEFAULT_ALIGN_STOP_OFFSET = 1
SPEECHIFY_DEFAULT_ALIGNMENT_PLATEAU_MAX_STEPS = 30

_PREPARED_MARKER = "_speechify_tts_prepared_request"
_REFERENCE_AUDIO_FIELDS = ("ref_audio", "audio_path", "audio")


# --------------------------------------------------------------------------- #
# Request -> state (pure; no GPU)                                              #
# --------------------------------------------------------------------------- #
def _coerce_inputs(inputs: Any) -> tuple[str, list[dict[str, Any]]]:
    """Normalize the heterogeneous ``inputs`` shapes into (text, references)."""
    if inputs is None:
        return "", []
    if isinstance(inputs, str):
        return inputs, []
    if isinstance(inputs, dict):
        text = inputs.get("text") or inputs.get("input") or ""
        refs = inputs.get("references") or []
        if not isinstance(refs, list):
            refs = [refs]
        return str(text), [r for r in refs if isinstance(r, dict)]
    raise ValueError(f"Unsupported SpeechifyTTS inputs type: {type(inputs)!r}")


def _resolve_reference_audio(
    references: list[dict[str, Any]], tts_params: dict[str, Any]
) -> Any | None:
    """Pick a reference-audio descriptor from inputs.references or tts_params."""
    for ref in references:
        for field_name in _REFERENCE_AUDIO_FIELDS:
            value = ref.get(field_name)
            if value:
                return value
        if ref.get("data") is not None:
            return ref
    for field_name in _REFERENCE_AUDIO_FIELDS:
        value = tts_params.get(field_name)
        if value:
            return value
    return None


def _explicit_fields(tts_params: dict[str, Any]) -> set[str]:
    explicit = tts_params.get("explicit_generation_params")
    if isinstance(explicit, (list, tuple, set)):
        return {str(field) for field in explicit}
    return set()


def _resolve_sampling(
    params: dict[str, Any], tts_params: dict[str, Any]
) -> dict[str, Any]:
    """Preserve user-set sampling vs endpoint-filled defaults.

    ``speech_service._build_sampling_params`` always fills a generic default set
    (0.8 / 0.8 / 30 / 1.1), so a value being present in ``params`` does NOT mean
    the user asked for it. Only fields named in ``explicit_generation_params``
    are honored; everything else falls back to the SpeechifyTTS defaults.
    """
    explicit = _explicit_fields(tts_params)
    out = {
        "temperature": SPEECHIFY_DEFAULT_TEMPERATURE,
        "top_p": SPEECHIFY_DEFAULT_TOP_P,
        "top_k": SPEECHIFY_DEFAULT_TOP_K,
        "repetition_penalty": SPEECHIFY_DEFAULT_REPETITION_PENALTY,
        "max_new_tokens": SPEECHIFY_DEFAULT_MAX_NEW_TOKENS,
        "seed": None,
    }
    if "temperature" in explicit and params.get("temperature") is not None:
        out["temperature"] = float(params["temperature"])
    if "top_p" in explicit and params.get("top_p") is not None:
        out["top_p"] = float(params["top_p"])
    if "top_k" in explicit and params.get("top_k") is not None:
        out["top_k"] = int(params["top_k"])
    if "repetition_penalty" in explicit and params.get("repetition_penalty") is not None:
        out["repetition_penalty"] = float(params["repetition_penalty"])
    # max_new_tokens / seed are not part of the generic default set, so honor
    # them whenever present.
    if params.get("max_new_tokens") is not None:
        out["max_new_tokens"] = int(params["max_new_tokens"])
    if params.get("seed") is not None:
        out["seed"] = params["seed"]
    return out


def _resolve_speaking_rate(tts_params: dict[str, Any]) -> float | None:
    """OpenAI ``speed`` (and explicit ``speaking_rate``) -> speaking rate."""
    rate = tts_params.get("speaking_rate")
    if rate is None:
        rate = tts_params.get("speed")
    if rate is None:
        return None
    try:
        return float(rate)
    except (TypeError, ValueError):
        return None


def build_speechify_tts_state(payload: StagePayload) -> SpeechifyTTSState:
    """Map an incoming request into a :class:`SpeechifyTTSState`.

    Raises ``ValueError`` for malformed input or a missing voice reference.
    """
    request = payload.request
    inputs = request.inputs
    params = request.params if isinstance(request.params, dict) else {}
    metadata = request.metadata if isinstance(request.metadata, dict) else {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        tts_params = {}

    text, references = _coerce_inputs(inputs)
    if not text.strip():
        raise ValueError("SpeechifyTTS requires non-empty input text")

    ref_audio = _resolve_reference_audio(references, tts_params)
    voice = tts_params.get("voice")
    if ref_audio is None and (not voice or str(voice).lower() == "default"):
        raise ValueError(
            "SpeechifyTTS is a zero-shot voice-cloning model and requires a voice "
            "reference: pass ref_audio (path/URL/base64) or an uploaded voice name."
        )

    sampling = _resolve_sampling(params, tts_params)
    streaming = bool(params.get("stream") or request.metadata.get("stream"))

    state = SpeechifyTTSState(
        text=text,
        ref_audio=ref_audio,
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        top_k=sampling["top_k"],
        repetition_penalty=sampling["repetition_penalty"],
        max_new_tokens=sampling["max_new_tokens"],
        seed=sampling["seed"],
        streaming=streaming,
        body_end_anchor_offset=SPEECHIFY_DEFAULT_BODY_END_ANCHOR_OFFSET,
        align_stop_offset=SPEECHIFY_DEFAULT_ALIGN_STOP_OFFSET,
        alignment_plateau_max_steps=SPEECHIFY_DEFAULT_ALIGNMENT_PLATEAU_MAX_STEPS,
    )
    rate = _resolve_speaking_rate(tts_params)
    if rate is not None:
        # Raw rate; the speaker tower maps it to a speaking_rate token at encode.
        state.speaking_rate = rate
    return state


# --------------------------------------------------------------------------- #
# Preprocessing context + abort bookkeeping (GPU-coupled handoff)             #
# --------------------------------------------------------------------------- #
@dataclass
class _PreprocessingContext:
    """Handles to the text encoder + speaker extractor used by preprocessing.

    Populated by ``create_preprocessing_executor`` once the deep model internals
    are available. ``text_encoder`` / ``speaker_extractor`` are ``None`` in the
    serving-skeleton build.
    """

    text_encoder: Any = None
    speaker_extractor: Any = None
    decoder_start_token_id: int | None = None


_PREPROCESSING_CONTEXT: _PreprocessingContext | None = None
_PREPARED_REQUESTS: dict[str, SpeechifyTTSState] = {}
_INFLIGHT_REQUESTS: set[str] = set()
_ABORTED_REQUESTS: set[str] = set()
_LOCK = threading.Lock()


def set_speechify_preprocessing_context(
    *,
    text_encoder: Any = None,
    speaker_extractor: Any = None,
    decoder_start_token_id: int | None = None,
) -> None:
    global _PREPROCESSING_CONTEXT
    with _LOCK:
        _PREPROCESSING_CONTEXT = _PreprocessingContext(
            text_encoder=text_encoder,
            speaker_extractor=speaker_extractor,
            decoder_start_token_id=decoder_start_token_id,
        )
        _PREPARED_REQUESTS.clear()
        _INFLIGHT_REQUESTS.clear()
        _ABORTED_REQUESTS.clear()


def cleanup_prepared_speechify_request(request_id: str) -> None:
    """Idempotent abort cleanup for the preprocessing -> AR handoff stash.

    Frees state on all three race paths (preprocessing aborts before handoff;
    AR aborts before consuming; preprocessing finishes after the abort).
    """
    rid = str(request_id)
    with _LOCK:
        if _PREPARED_REQUESTS.pop(rid, None) is not None:
            return
        if rid in _INFLIGHT_REQUESTS:
            _ABORTED_REQUESTS.add(rid)


def pop_prepared_speechify_request(payload: StagePayload) -> SpeechifyTTSState | None:
    data = payload.data if isinstance(payload.data, dict) else {}
    marker = data.get(_PREPARED_MARKER)
    if marker is None:
        return None
    with _LOCK:
        return _PREPARED_REQUESTS.pop(str(marker), None)


# --------------------------------------------------------------------------- #
# Preprocessing compute_fn (encoder + speaker extraction) — GPU-coupled       #
# --------------------------------------------------------------------------- #
def preprocess_speechify_payload(payload: StagePayload) -> StagePayload:
    """Build state, then run the text encoder + speaker extraction.

    The request-boundary mapping (``build_speechify_tts_state``) is always
    executed so invalid requests fail fast. The encoder / speaker-extraction
    step requires the deep model internals; until those land it raises
    ``NotImplementedError``.
    """
    rid = str(payload.request_id)
    state = build_speechify_tts_state(payload)

    with _LOCK:
        context = _PREPROCESSING_CONTEXT
        if context is not None:
            _INFLIGHT_REQUESTS.add(rid)

    if context is None or context.text_encoder is None:
        raise NotImplementedError(
            "SpeechifyTTS preprocessing requires the text-encoder + speaker "
            "extractor (deferred to the GPU port). Wire them via "
            "set_speechify_preprocessing_context() in create_preprocessing_executor."
        )

    try:
        # Deep internals (deferred): run encoder + speaker extraction, then set
        #   state.text_hidden_states, state.text_seq_len,
        #   state.speaker_embedding, state.speaking_rate_token,
        #   state.speech_prompt_mels, state.decoder_prompt_token_ids.
        raise NotImplementedError(
            "SpeechifyTTS encoder + speaker-embedding forward not yet ported "
            "(see package docstring: deep model internals)."
        )
    finally:
        with _LOCK:
            _INFLIGHT_REQUESTS.discard(rid)
            aborted = rid in _ABORTED_REQUESTS
            _ABORTED_REQUESTS.discard(rid)
            if not aborted:
                _PREPARED_REQUESTS[rid] = state


# --------------------------------------------------------------------------- #
# AR scheduler adapters — GPU-coupled                                         #
# --------------------------------------------------------------------------- #
def make_speechify_scheduler_adapters(*, model: Any):
    """Build StagePayload <-> SGLang AR request adapters for SpeechifyTTS.

    Deferred to the GPU port: turning the prepared state into an SGLang ``Req``
    with the forced speaking-rate first token, alignment-stop wiring, and the
    masked-audio logits processor; and adapting the AR result (mel codes +
    per-step latent/aligned_encoder_latent/alignment) back into a payload.
    """

    def request_builder(payload: StagePayload):
        del payload
        raise NotImplementedError(
            "SpeechifyTTS AR request builder not yet ported (deep model internals)."
        )

    def result_adapter(data: Any):
        del data
        raise NotImplementedError(
            "SpeechifyTTS AR result adapter not yet ported (deep model internals)."
        )

    _ = model
    return request_builder, result_adapter


def _now() -> float:
    return time.perf_counter()
