# SPDX-License-Identifier: Apache-2.0
"""Torch-native DiffiTv3 + FlowVAE diffusion vocoder driver (offline / eager).

Wraps the ported :class:`DiffiTModelV3` (``diffit/diffitv3_core.py``) and its
``FlowVAEInputLayer`` to turn AR-decoder latents into a waveform. This mirrors
the cache-free offline path of the vllm-omni
``SpeechifyT5TTSDiffusionVocoder`` (``process_window_offline`` +
``_process_offline_finish``):

  1. encode the reference speech prompt (FlowVAE latents -> speech-prompt
     encoder -> query pooling + soft cap);
  2. build the speaker timestep conditioning from the pooled prompt when the
     recipe uses query pooling (``_reuse_ar_extra_cond_for_diffusion=false``);
  3. run the flow-matching Euler ODE (``n_diffusion_steps`` steps over
     ``t in [0, 1]``) calling the denoiser each step;
  4. denormalize and FlowVAE-decode the latent to audio.

The streaming / CUDA-graph slot machinery from vllm-omni is intentionally not
ported here; this driver is the eager reference that the scheduler + graph
runtime will wrap in later phases.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

from .diffit.diffitv3_core import DiffiTModelV3

_DEPRECATED_DIFFIT_KEYS = ("use_alibi", "cfg_training", "unconditioned_percentage")


def _normalize_speaker_emb(spk: torch.Tensor) -> torch.Tensor:
    """Reduce a speaker embedding of any rank to ``[1, cond_dim]`` (mean-pool
    leading seq dims), matching the vllm-omni vocoder helper."""
    while spk.dim() > 2:
        spk = spk.mean(dim=1)
    if spk.dim() == 1:
        spk = spk.unsqueeze(0)
    if spk.shape[0] != 1:
        spk = spk.mean(dim=0, keepdim=True)
    return spk


def _remove_weight_norm(module: nn.Module) -> int:
    count = 0
    for m in module.modules():
        try:
            torch.nn.utils.remove_weight_norm(m)
            count += 1
        except ValueError:
            pass
    return count


class SpeechifyVocoder:
    """DiffiTv3 + FlowVAE diffusion vocoder (offline eager reference)."""

    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        diffusion_dir = os.path.join(model_path, "diffusion")
        with open(os.path.join(diffusion_dir, "config.json")) as f:
            cfg = json.load(f)

        for k in _DEPRECATED_DIFFIT_KEYS:
            cfg.pop(k, None)
        self.n_diffusion_steps = int(cfg.pop("n_diffusion_steps", 2))
        self.num_future_frames = int(cfg.pop("num_future_frames", 8))
        self.conditioning_free = bool(
            cfg.pop("conditioning_free", cfg.pop("use_cfg", False))
        )
        self.cfg_scale = float(cfg.pop("cfg_scale", 1.0))
        self.diffusion_upsample_factor = int(cfg.pop("diffusion_upsample_factor", 2))
        cfg.pop("mel_length_compression", None)
        cfg.pop("class_name", None)
        cfg["compile_denoiser"] = False
        cfg["diffusion_upsample_factor"] = self.diffusion_upsample_factor

        self.diffit = DiffiTModelV3(**cfg)
        self.diffit.diffusion_upsample_factor = self.diffusion_upsample_factor

        from safetensors.torch import load_file

        state = load_file(os.path.join(diffusion_dir, "model.safetensors"))
        info = self.diffit.load_state_dict(state, strict=False)
        if info.missing_keys or info.unexpected_keys:
            raise RuntimeError(
                f"DiffiTv3 weight mismatch: missing={info.missing_keys[:6]} "
                f"unexpected={info.unexpected_keys[:6]}"
            )
        self.diffit = self.diffit.eval().to(device=device, dtype=dtype)

        # Materialize weight_norm params as plain weights (eager decode path).
        il = self.diffit.input_layer
        for name in ("decoder", "encoder"):
            sub = getattr(il, name, None)
            if sub is not None:
                _remove_weight_norm(sub)

        self.input_layer = il
        self.sample_rate = int(getattr(il, "sampling_rate", 24000))
        self.hop_length = int(getattr(il, "hop_length", 512))
        self._uses_query_pooling = (
            int(getattr(self.diffit, "speech_prompt_query_pooling_num", 0) or 0) > 0
        )

    # -- reference prompt -------------------------------------------------- #
    @torch.inference_mode()
    def encode_prompt(self, speech_prompt_mels: torch.Tensor):
        """``speech_prompt_mels`` [T', latent_channels] (FlowVAE latents) ->
        (prompt_hidden [1, P, F], prompt_mask [1, P])."""
        mels = speech_prompt_mels.to(device=self.device, dtype=self.dtype)
        if mels.dim() == 2:
            mels = mels.unsqueeze(0)
        return self.diffit.encode_prompt(mels, None)

    # -- diffusion --------------------------------------------------------- #
    @torch.inference_mode()
    def generate(
        self,
        decoder_latents: torch.Tensor,           # [M, F]
        aligned_encoder_latents: torch.Tensor,   # [M, F]
        speaker_embedding: torch.Tensor,         # [Ns, cond_dim] or [cond_dim]
        speech_prompt_mels: torch.Tensor | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Run the full offline ODE + VAE decode. Returns audio ``[1, T_audio]``."""
        dev, dt = self.diffit.parameters().__next__().device, self.diffit.parameters().__next__().dtype
        M = decoder_latents.shape[0]
        T = M * self.diffusion_upsample_factor

        aligned = decoder_latents.to(dev, dt).unsqueeze(0)            # [1, M, F]
        aligned_enc = aligned_encoder_latents.to(dev, dt).unsqueeze(0)  # [1, M, F]

        prompt_hs = prompt_mask = None
        if speech_prompt_mels is not None:
            prompt_hs, prompt_mask = self.encode_prompt(speech_prompt_mels)

        # Speaker timestep conditioning: pooled-prompt mean for query-pooling
        # recipes (matches training when _reuse_ar_extra_cond_for_diffusion=false),
        # otherwise the AR speaker embedding.
        if self._uses_query_pooling and prompt_hs is not None:
            spk = self.diffit._build_timestep_cond_from_speech_prompt(prompt_hs, prompt_mask)
        else:
            spk = _normalize_speaker_emb(speaker_embedding.to(dev, dt))

        noise = self.diffit.get_streaming_noise(
            (1, self.diffit.output_channels, T), dev, dt,
        )
        if seed is not None:
            g = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(1, self.diffit.output_channels, T, generator=g).to(dev, dt)
        x = noise

        t_span = torch.linspace(0, 1, self.n_diffusion_steps + 1, device=dev, dtype=dt)
        for step_i in range(self.n_diffusion_steps):
            timestep = t_span[step_i] * torch.ones(1, device=dev, dtype=dt)
            v = self.diffit.forward(
                hidden_states=x,
                timestep=timestep,
                timestep_cond=spk,
                aligned_latents=aligned,
                aligned_encoder_latents=aligned_enc,
                speech_prompt_hidden_states=prompt_hs,
                speech_prompt_attention_mask=prompt_mask,
                conditioning_free=False,
                cfg_training=False,
                use_cache=False,
            )
            dt_step = t_span[step_i + 1] - t_span[step_i]
            x = x + dt_step * v

        audio = self.input_layer.decode(x)  # denormalizes internally -> [1, 1, T*hop]
        return audio.squeeze(1).float()      # [1, T_audio]
