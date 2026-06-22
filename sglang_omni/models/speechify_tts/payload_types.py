# SPDX-License-Identifier: Apache-2.0
"""SpeechifyTTS pipeline state passed between stages.

State accumulates as a request moves preprocessing -> tts_engine -> vocoder.
Tensors are kept as detached CPU tensors in :meth:`to_dict` so the SHM relay can
transfer them between stage processes without an expensive ``.tolist()`` round
trip (mirrors :class:`MossTTSLocalState`). The AR decoder emits one ``latent`` /
``aligned_encoder_latent`` / ``alignment`` row per mel-code step; the diffusion
vocoder consumes those rows in chunks of ``chunk_size`` (default 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _as_tensor(value: Any) -> Any:
    if value is None or isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


@dataclass
class SpeechifyTTSState:
    """Per-request state for the SpeechifyTTS 3-stage pipeline."""

    # -- Original request (preprocessing input) ---------------------------
    text: str = ""
    normalized_text: str | None = None
    ref_audio: Any | None = None  # path / gs:// uri / handle (preprocessing only)

    # -- Decoder prompt / cross-attention state (preprocessing -> engine) --
    decoder_prompt_token_ids: list[int] = field(default_factory=list)
    text_hidden_states: Any | None = None  # [text_seq_len, enc_hidden]
    text_seq_len: int = 0
    speaker_embedding: Any | None = None  # [num_speaker_tokens, spk_dim]
    speaking_rate: float | None = None  # raw requested rate (OpenAI speed)
    speaking_rate_token: int | None = None  # forced first sampled token
    speech_prompt_mels: Any | None = None  # diffusion VAE prompt latents

    # -- Sampling / generation params -------------------------------------
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 10
    repetition_penalty: float = 2.0
    max_new_tokens: int = 2048
    seed: int | None = None
    streaming: bool = True

    # -- Alignment-based stop policy (recipe-gated) -----------------------
    predict_alignment: bool = True
    body_end_anchor_offset: int = 1
    align_stop_offset: int = 1
    alignment_plateau_max_steps: int = 30

    # -- From tts_engine (mel codes + diffusion conditioning) -------------
    mel_codes: Any | None = None  # [num_steps] long
    decoder_latents: Any | None = None  # [num_steps, dec_hidden]
    aligned_encoder_latents: Any | None = None  # [num_steps, enc_hidden]
    alignments: Any | None = None  # [num_steps] float
    body_end_mel_codes: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0
    finish_reason: str | None = None

    # -- From vocoder -----------------------------------------------------
    audio_samples: Any | None = None
    sample_rate: int = 24000
    speechmarks: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "decoder_prompt_token_ids": list(self.decoder_prompt_token_ids),
            "text_seq_len": int(self.text_seq_len),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "top_k": int(self.top_k),
            "repetition_penalty": float(self.repetition_penalty),
            "max_new_tokens": int(self.max_new_tokens),
            "streaming": bool(self.streaming),
            "predict_alignment": bool(self.predict_alignment),
            "body_end_anchor_offset": int(self.body_end_anchor_offset),
            "align_stop_offset": int(self.align_stop_offset),
            "alignment_plateau_max_steps": int(self.alignment_plateau_max_steps),
            "sample_rate": int(self.sample_rate),
        }
        for key in (
            "normalized_text",
            "ref_audio",
            "speaking_rate",
            "speaking_rate_token",
            "seed",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        for key in (
            "text_hidden_states",
            "speaker_embedding",
            "speech_prompt_mels",
            "mel_codes",
            "decoder_latents",
            "aligned_encoder_latents",
            "alignments",
            "audio_samples",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = _to_cpu(value)
        if self.body_end_mel_codes is not None:
            data["body_end_mel_codes"] = int(self.body_end_mel_codes)
        if self.prompt_tokens:
            data["prompt_tokens"] = int(self.prompt_tokens)
        if self.completion_tokens:
            data["completion_tokens"] = int(self.completion_tokens)
        if self.engine_time_s:
            data["engine_time_s"] = float(self.engine_time_s)
        if self.finish_reason is not None:
            data["finish_reason"] = self.finish_reason
        if self.speechmarks:
            data["speechmarks"] = list(self.speechmarks)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "SpeechifyTTSState":
        if not isinstance(data, dict):
            data = {}
        return cls(
            text=str(data.get("text", "")),
            normalized_text=data.get("normalized_text"),
            ref_audio=data.get("ref_audio"),
            decoder_prompt_token_ids=list(data.get("decoder_prompt_token_ids", [])),
            text_hidden_states=_as_tensor(data.get("text_hidden_states")),
            text_seq_len=int(data.get("text_seq_len", 0) or 0),
            speaker_embedding=_as_tensor(data.get("speaker_embedding")),
            speaking_rate=data.get("speaking_rate"),
            speaking_rate_token=data.get("speaking_rate_token"),
            speech_prompt_mels=_as_tensor(data.get("speech_prompt_mels")),
            temperature=float(data.get("temperature", 0.8)),
            top_p=float(data.get("top_p", 0.8)),
            top_k=int(data.get("top_k", 10)),
            repetition_penalty=float(data.get("repetition_penalty", 2.0)),
            max_new_tokens=int(data.get("max_new_tokens", 2048)),
            seed=data.get("seed"),
            streaming=bool(data.get("streaming", True)),
            predict_alignment=bool(data.get("predict_alignment", True)),
            body_end_anchor_offset=int(data.get("body_end_anchor_offset", 1)),
            align_stop_offset=int(data.get("align_stop_offset", 1)),
            alignment_plateau_max_steps=int(
                data.get("alignment_plateau_max_steps", 30)
            ),
            mel_codes=_as_tensor(data.get("mel_codes")),
            decoder_latents=_as_tensor(data.get("decoder_latents")),
            aligned_encoder_latents=_as_tensor(data.get("aligned_encoder_latents")),
            alignments=_as_tensor(data.get("alignments")),
            body_end_mel_codes=data.get("body_end_mel_codes"),
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
            engine_time_s=float(data.get("engine_time_s", 0.0) or 0.0),
            finish_reason=data.get("finish_reason"),
            audio_samples=data.get("audio_samples"),
            sample_rate=int(data.get("sample_rate", 24000) or 24000),
            speechmarks=list(data.get("speechmarks", [])),
        )
