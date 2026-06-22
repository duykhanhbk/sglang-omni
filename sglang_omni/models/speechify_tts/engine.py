# SPDX-License-Identifier: Apache-2.0
"""Self-contained offline SpeechifyTTS engine (text + reference voice -> audio).

Ties the validated torch-native components into one synthesizer:

    tokenizer -> text encoder -> (voice extractor) -> MoE AR decode loop
              -> DiffiTv3 + FlowVAE diffusion vocoder -> 24 kHz waveform

This is the eager reference path used by the web demo / CLI and by the
sglang-omni stage executors. The streaming + CUDA-graph runtime wraps these
same modules in later phases; the numerics here are the ground truth.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

from .ar_loop import generate as ar_generate
from .configuration import SpeechifyT5TTSConfig, register_speechify_hf_configs
from .decoder import SpeechifyDecoder
from .encoder import SpeechifyTextEncoder
from .tokenizer import SpeechifyTTSTokenizer
from .vocoder import SpeechifyVocoder
from .voice_extractor import VoiceFeatureExtractor

logger = logging.getLogger(__name__)


def _resolve_ar_subdir(checkpoint_dir: str) -> str:
    ar = os.path.join(checkpoint_dir, "ar")
    return ar if os.path.isdir(ar) else checkpoint_dir


@dataclass
class SynthesisResult:
    audio: torch.Tensor          # [T_audio] float32 (cpu)
    sample_rate: int
    num_mel_codes: int
    stop_reason: str
    speaking_rate: float
    timings: dict | None = None  # per-phase wall times (seconds)
    marks: list | None = None    # word speechmarks [{value,startIndex,endIndex,startTime(ms)}]


def _compute_speechmarks(
    text: str, alignments: list[float], text_len: int, frame_dur_ms: float,
) -> list[dict]:
    """Derive word-level speechmarks from the AR alignment trajectory.

    The decoder's per-frame alignment tracks the text-token position being
    spoken; mapping it (proportionally) onto whitespace-delimited word char
    spans yields a monotonic start time per word. This is dependency-free
    (no ICU) and accurate enough for live word highlighting; it captures the
    pacing/pauses the model actually produced rather than a flat estimate.
    """
    import re

    words = [
        {"value": m.group(), "start": m.start(), "end": m.end()}
        for m in re.finditer(r"\S+", text)
    ]
    if not words or not alignments:
        return []
    n = len(alignments)
    char_total = max(len(text), 1)
    tl = max(text_len, 1)
    marks: list[dict] = []
    fi = 0
    for w in words:
        target_frac = w["start"] / char_total
        while fi < n and (alignments[fi] / tl) < target_frac:
            fi += 1
        marks.append({
            "value": w["value"].strip(),
            "startIndex": w["start"],
            "endIndex": w["end"],
            "startTime": int(round(min(fi, n - 1) * frame_dur_ms)),
        })
    return marks


class SpeechifyTTSEngine:
    """Offline text-to-speech engine for the SpeechifyTTS MoE 4B MTL recipe."""

    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        register_speechify_hf_configs()
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        ar_dir = _resolve_ar_subdir(model_path)
        self.config = SpeechifyT5TTSConfig.from_pretrained(ar_dir)

        from safetensors.torch import load_file

        items = list(load_file(os.path.join(ar_dir, "model.safetensors"), device="cpu").items())

        logger.info("Loading SpeechifyTTS text encoder + AR decoder from %s", ar_dir)
        self.encoder = SpeechifyTextEncoder(self.config)
        self.encoder.load_weights(items)
        self.encoder = self.encoder.eval().to(device=device, dtype=dtype)

        self.decoder = SpeechifyDecoder(self.config)
        self.decoder.load_weights(items)
        self.decoder = self.decoder.eval().to(device=device, dtype=dtype)

        self.tokenizer = SpeechifyTTSTokenizer(ar_dir)

        logger.info("Loading DiffiTv3 + FlowVAE vocoder")
        self.vocoder = SpeechifyVocoder(model_path, device=device, dtype=dtype)
        self.sample_rate = self.vocoder.sample_rate

        self.voice_extractor = VoiceFeatureExtractor(ar_dir, device=device, dtype=dtype)
        # FlowVAE encodes the reference into diffusion speech-prompt latents.
        self.voice_extractor.set_vae(self.vocoder.input_layer)

    @torch.inference_mode()
    def warmup(self, texts: tuple[str, ...] = (
        "Hello there, this is a short warmup utterance.",
        "This is a slightly longer warmup utterance that exercises the "
        "autoregressive decode loop, the diffusion vocoder, and the speaker "
        "tower so the very first real request does not pay cold CUDA costs.",
    )) -> None:
        """Run a couple of full ``synthesize`` passes through every stage (voice
        extractor, encoder, AR loop, vocoder) so the first real request is not
        ~4x slower paying one-time CUDA module-load / cuBLAS heuristic costs.

        Uses a synthetic reference waveform (band-limited noise) so no reference
        file is required."""
        g = torch.Generator().manual_seed(0)
        ref = (torch.randn(3 * 24000, generator=g) * 0.05).clamp(-1, 1)
        for txt in texts:
            try:
                self.synthesize(txt, ref, max_new_tokens=256, seed=0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("warmup pass failed (non-fatal): %s", exc)
        torch.cuda.synchronize()
        logger.info("SpeechifyTTS warmup complete (%d passes)", len(texts))

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        reference_audio: str | torch.Tensor,
        *,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 10,
        repetition_penalty: float = 2.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
        speaking_rate: float | None = None,
    ) -> SynthesisResult:
        """Synthesize ``text`` in the voice of ``reference_audio``."""
        import time

        def _now() -> float:
            torch.cuda.synchronize()
            return time.perf_counter()

        t0 = _now()
        vf = self.voice_extractor.extract(reference_audio)
        spk = vf.speaker_embedding.to(self.device)
        rate = speaking_rate if speaking_rate is not None else vf.speaking_rate
        t_voice = _now()

        ids = torch.tensor(self.tokenizer.encode(text), device=self.device)
        text_hidden = self.encoder(ids).to(self.dtype)
        text_mask = torch.ones(1, text_hidden.shape[0], dtype=torch.long, device=self.device)
        t_enc = _now()

        ar = ar_generate(
            self.decoder, self.config, text_hidden, text_mask,
            spk.to(self.dtype), rate,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
            seed=seed,
        )
        t_ar = _now()
        if ar.latents.shape[0] == 0:
            raise RuntimeError("AR decoder produced no audio codes")

        audio = self.vocoder.generate(
            ar.latents, ar.aligned_encoder_latents, spk,
            vf.speech_prompt_mels, seed=seed,
        )
        t_voc = _now()
        n = len(ar.codes)
        frame_dur_ms = (
            self.vocoder.diffusion_upsample_factor * self.vocoder.hop_length
            / self.sample_rate * 1000.0
        )
        marks = _compute_speechmarks(text, ar.alignments, text_hidden.shape[0], frame_dur_ms)
        timings = {
            "voice_s": round(t_voice - t0, 3),
            "encoder_s": round(t_enc - t_voice, 3),
            "ar_s": round(t_ar - t_enc, 3),
            "ar_tok_s": round(n / max(t_ar - t_enc, 1e-6), 1),
            "vocoder_s": round(t_voc - t_ar, 3),
            "total_s": round(t_voc - t0, 3),
        }
        return SynthesisResult(
            audio=audio[0].float().cpu(),
            sample_rate=self.sample_rate,
            num_mel_codes=n,
            stop_reason=ar.stop_reason,
            speaking_rate=float(rate),
            timings=timings,
            marks=marks,
        )
