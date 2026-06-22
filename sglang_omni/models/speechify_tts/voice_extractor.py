# SPDX-License-Identifier: Apache-2.0
"""Reference-audio voice feature extractor for SpeechifyTTS.

Ports the speaker-embedding + speaking-rate path of vllm-omni
``voice_cache/extractor.py`` for the Unified family. Produces the AR-side
conditioning (``speaker_embedding`` + ``speaking_rate``) from a reference
waveform; the diffusion ``speech_prompt_mels`` (FlowVAE encode) is produced by
the vocoder stage and wired in via :meth:`VoiceFeatureExtractor.set_vae`.

GPTTTS ``extract_extra_cond_embeds`` parity:
  1. mel = TargetMelSpectrogram(audio @ 24 kHz)
  2. chunk mel into 800-frame segments (repeat-pad when shorter)
  3. ``get_spk_features`` per chunk, mean-aggregate identity tokens
  4. re-embed the *scalar*-averaged feature values via
     ``update_feature_embeddings`` (averaging discrete bucket tokens is OOD)
  5. speaking rate = mean of the unified ``spk_rate`` head, clamped to
     the AR bucketizer range [1.5, 3.5]
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.models.speechify_tts.speaker_tower import UnifiedSpkEmbeddingWithDec

logger = logging.getLogger(__name__)

_CHUNK = 800
_TARGET_SR = 24000
_AR_RATE_MIN, _AR_RATE_MAX = 1.5, 3.5


@dataclass
class VoiceFeatures:
    speaker_embedding: torch.Tensor  # [num_tokens, features_dim] (cpu float)
    speaking_rate: float
    speech_prompt_mels: torch.Tensor | None = None  # [T', 64] FlowVAE latents


def build_speaker_model(ar_dir: str, device: str = "cuda", dtype=torch.bfloat16):
    """Construct + load ``UnifiedSpkEmbeddingWithDec`` from an AR checkpoint dir."""
    from safetensors.torch import load_file

    with open(os.path.join(ar_dir, "config.json")) as f:
        ec = json.load(f)["extra_conds_config"]
    input_layer_config = {
        "class_name": "TargetMelSpectrogram",
        "n_mel_channels": ec.get("n_mel_channels", 100),
        "sampling_rate": ec.get("sampling_rate", 24000),
        "mel_fmax": ec.get("mel_fmax", 12000),
        "do_normalization": ec.get("do_normalization", True),
    }
    features_config = ec.get("features_config")
    if not features_config:
        raise ValueError(
            "extra_conds_config.features_config missing; cannot build the "
            "Unified speaker model. Re-export the AR HF folder with the trainer "
            "features_config embedded."
        )
    model = UnifiedSpkEmbeddingWithDec(
        features_config=features_config,
        in_channels=ec.get("in_channels", 100),
        dim=ec.get("dim", 512),
        nb_speaker_features=ec.get("nb_speaker_features", 6),
        dec_dim=ec.get("dec_dim", 1024),
        features_dim=ec.get("features_dim", 1024),
        encoder_num_layers=ec.get("encoder_num_layers", 6),
        decoder_num_layers=ec.get("decoder_num_layers", 6),
        feature_predictor_num_layers=ec.get("feature_predictor_num_layers", 6),
        conv1_dim=ec.get("conv1_dim", 64),
        conv2_dim=ec.get("conv2_dim", 128),
        input_layer_config=input_layer_config,
        use_flash_att=False,
        use_post_norm=ec.get("use_post_norm", False),
        use_post_norm_encoder=ec.get("use_post_norm_encoder", False),
        use_out_norm=ec.get("use_out_norm", True),
    )
    weights = _load_safetensors_dir(ar_dir)
    state = {
        k[len("extra_conds_model.") :]: v
        for k, v in weights.items()
        if k.startswith("extra_conds_model.")
    }
    result = model.load_state_dict(state, strict=False)
    nonbuf_missing = [
        k
        for k in result.missing_keys
        if not any(
            s in k
            for s in (
                "_boundaries_",
                "_percentile_cdf_",
                "_gr_boundaries_",
                "_bucket_centers_",
            )
        )
    ]
    if nonbuf_missing:
        raise RuntimeError(
            f"speaker model missing non-buffer weights: {nonbuf_missing[:8]}"
        )
    if result.unexpected_keys:
        logger.warning(
            "speaker model: %d unexpected keys; sample %s",
            len(result.unexpected_keys),
            result.unexpected_keys[:5],
        )
    return model.eval().to(dtype).to(device)


def _load_safetensors_dir(model_dir: str) -> dict:
    from safetensors.torch import load_file

    consolidated = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(consolidated):
        return load_file(consolidated, device="cpu")
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = (json.load(f).get("weight_map") or {})
        merged: dict = {}
        for shard_name in sorted(set(weight_map.values())):
            merged.update(load_file(os.path.join(model_dir, shard_name), device="cpu"))
        return merged
    raise FileNotFoundError(f"no safetensors weights under {model_dir!r}")


def load_audio_24k(path: str) -> torch.Tensor:
    """Decode an audio file to a mono float32 [T] tensor at 24 kHz."""
    import numpy as np

    try:
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    except Exception:  # pragma: no cover - fall back to librosa/audioread
        import librosa

        audio, sr = librosa.load(path, sr=None, mono=True)
        audio = audio.astype("float32")
    if sr != _TARGET_SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=_TARGET_SR)
    return torch.from_numpy(np.ascontiguousarray(audio))


class VoiceFeatureExtractor:
    """Run the Unified speaker tower on a reference waveform (GPU-resident)."""

    def __init__(self, ar_dir: str, device: str = "cuda", dtype=torch.bfloat16):
        self.ar_dir = ar_dir
        self.device = device
        self.dtype = dtype
        self._spk = build_speaker_model(ar_dir, device=device, dtype=dtype)
        self._vae = None  # set by the vocoder stage for speech_prompt_mels
        self.diff_audio_seconds = 5.0

    def set_vae(self, vae) -> None:
        """Attach a FlowVAE input layer so ``extract`` can emit prompt latents."""
        self._vae = vae

    @torch.inference_mode()
    def extract(self, ref_audio: str | torch.Tensor) -> VoiceFeatures:
        if isinstance(ref_audio, str):
            audio = load_audio_24k(ref_audio)
        else:
            audio = ref_audio.detach().float().reshape(-1).cpu()
        audio_t = audio.to(self.device).unsqueeze(0)

        mel = self._spk.input_layer(audio_t.float())
        mel_len = mel.shape[-1]
        if mel_len < _CHUNK:
            rf = int(_CHUNK / max(mel_len, 1) + 1.0)
            chunked = mel.repeat(1, 1, rf)[..., :_CHUNK]
        else:
            n = mel_len // _CHUNK
            trimmed = mel[..., : n * _CHUNK]
            fdim = trimmed.shape[1]
            chunked = (
                trimmed.view(1, fdim, n, _CHUNK)
                .permute(0, 2, 1, 3)
                .reshape(n, fdim, _CHUNK)
            )

        feats, resolved = self._spk.get_spk_features(
            x=chunked.to(self.dtype), features=None, mask=None
        )
        n_per_batch = chunked.shape[0]
        spk_emb = feats.view(1, n_per_batch, *feats.shape[1:]).mean(dim=1)
        if resolved:
            avg: list[dict[str, float]] = [{}]
            for name, vals in resolved.items():
                avg[0][name] = float(vals.view(1, n_per_batch).float().mean().item())
            spk_emb = self._spk.update_feature_embeddings(spk_emb, avg)
            rate = float(avg[0].get("spk_rate", _AR_RATE_MIN))
        else:
            rate = _AR_RATE_MIN
        speaking_rate = max(_AR_RATE_MIN, min(_AR_RATE_MAX, rate))

        speech_prompt_mels = None
        if self._vae is not None:
            diff_len = int(self.diff_audio_seconds * _TARGET_SR)
            wav = audio_t.float()
            if wav.shape[-1] > diff_len:
                wav = wav[..., :diff_len]
            else:
                wav = torch.nn.functional.pad(wav, (0, diff_len - wav.shape[-1]))
            latents = self._vae.encode(wav.to(self.dtype))  # [1, 64, T']
            speech_prompt_mels = latents.transpose(-1, -2).float().contiguous().cpu()[0]

        return VoiceFeatures(
            speaker_embedding=spk_emb.squeeze(0).float().cpu(),
            speaking_rate=speaking_rate,
            speech_prompt_mels=speech_prompt_mels,
        )
