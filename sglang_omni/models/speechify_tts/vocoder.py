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

    # -- streaming (chunked diffusion + VAE) ------------------------------ #
    def new_streaming_session(
        self,
        speaker_embedding: torch.Tensor,
        speech_prompt_mels: torch.Tensor | None = None,
    ) -> "StreamingVocoderSession":
        """Create a stateful streaming session for incremental vocoding.

        The session keeps the denoiser KV/conv caches, the aligned-encoder
        KV cache and the FlowVAE decoder conv cache across blocks so audio
        can be emitted block-by-block (low TTFA) with cross-block
        continuity — the torch-native equivalent of vllm-omni's
        ``DiffusionOrchestrator.process_window`` streaming path.
        """
        return StreamingVocoderSession(self, speaker_embedding, speech_prompt_mels)

    @torch.inference_mode()
    def streaming_generate(
        self,
        decoder_latents: torch.Tensor,           # [M, F]
        aligned_encoder_latents: torch.Tensor,   # [M, F]
        speaker_embedding: torch.Tensor,
        speech_prompt_mels: torch.Tensor | None = None,
    ):
        """Generator yielding audio chunks ``[1, n_samples]`` block-by-block.

        Convenience wrapper around :class:`StreamingVocoderSession` that
        feeds an already-complete latent sequence in one shot. Used for
        validation / offline-vs-streaming parity; the engine drives the
        session incrementally instead.
        """
        sess = self.new_streaming_session(speaker_embedding, speech_prompt_mels)
        sess.push(decoder_latents, aligned_encoder_latents)
        for chunk in sess.drain():
            yield chunk
        tail = sess.finish()
        if tail is not None and tail.numel() > 0:
            yield tail


def _snapshot_self_conv_cache(pkv):
    """Snapshot the denoiser self-attention seq length + conv caches so the
    intermediate ODE steps can be rewound (only the final step's commit
    sticks). The prompt cross-attention cache is intentionally untouched —
    it is write-once and shared across steps/blocks."""
    sa = pkv.self_attention_cache
    base = sa.get_seq_length(0) if len(sa.layers) > 0 else 0
    conv = [c.clone() if c is not None else None for c in pkv.conv_cache]
    return base, conv


def _restore_self_conv_cache(pkv, snap) -> None:
    base, conv = snap
    sa = pkv.self_attention_cache
    for layer in sa.layers:
        if layer.keys is not None and layer.keys.shape[-2] > base:
            layer.keys = layer.keys[:, :, :base, :].contiguous()
            layer.values = layer.values[:, :, :base, :].contiguous()
    pkv.conv_cache = [c.clone() if c is not None else None for c in conv]


class StreamingVocoderSession:
    """Incremental block-wise diffusion + VAE vocoding for one request.

    Usage::

        sess = vocoder.new_streaming_session(spk_emb, prompt_mels)
        sess.push(decoder_latents_chunk, aligned_enc_chunk)   # repeatable
        for audio in sess.drain():                            # ready blocks
            stream(audio)
        tail = sess.finish()                                  # flush remainder

    Frame bookkeeping (mel-frame space = AR-token space x ``upsample``):
      * ``commit_frames``  = ``streaming_block_size`` (committed per block)
      * ``future_frames``  = ``num_future_frames``    (lookahead, recomputed)
      * window             = ``commit_frames + future_frames`` mel frames
      * one AR token       = ``upsample`` mel frames
    """

    def __init__(self, voc: SpeechifyVocoder, speaker_embedding, speech_prompt_mels):
        from .diffit.diffit_cache_utils import DiffitPastKeyValues

        self.voc = voc
        self.dev = next(voc.diffit.parameters()).device
        self.dt = next(voc.diffit.parameters()).dtype
        self.up = int(voc.diffusion_upsample_factor)
        self.commit_tokens = max(1, voc.diffit.streaming_block_size // self.up)
        self.lookahead_tokens = max(0, -(-voc.num_future_frames // self.up))  # ceil
        self.n_steps = voc.n_diffusion_steps
        self.out_channels = voc.diffit.output_channels

        # One-time conditioning (prompt encode + speaker timestep cond).
        prompt_hs = prompt_mask = None
        if speech_prompt_mels is not None:
            prompt_hs, prompt_mask = voc.encode_prompt(speech_prompt_mels)
        self.prompt_hs = prompt_hs
        self.prompt_mask = prompt_mask
        if voc._uses_query_pooling and prompt_hs is not None:
            self.spk = voc.diffit._build_timestep_cond_from_speech_prompt(prompt_hs, prompt_mask)
        else:
            self.spk = _normalize_speaker_emb(speaker_embedding.to(self.dev, self.dt))

        # Persistent streaming caches.
        n_layers = len(voc.diffit.transformer.transformer_blocks)
        self.denoiser_pkv = DiffitPastKeyValues.create(num_layers=n_layers)
        self.enc_pkv = None          # DynamicCache, lazily created by the encoder
        self.vae_cache = None        # dict, created by decode_streaming

        # Latent buffers (accumulate AR latents; consumed block-by-block).
        self._dec_buf: list[torch.Tensor] = []
        self._enc_buf: list[torch.Tensor] = []
        self._n_tokens = 0           # tokens pushed so far
        self._committed_tokens = 0   # tokens already vocoded/committed
        self._finished = False

    # -- public API ------------------------------------------------------- #
    def push(self, decoder_latents: torch.Tensor, aligned_encoder_latents: torch.Tensor) -> None:
        """Append a chunk of AR latents (``[m, F]``) to the pending buffer."""
        self._dec_buf.append(decoder_latents.to(self.dev, self.dt))
        self._enc_buf.append(aligned_encoder_latents.to(self.dev, self.dt))
        self._n_tokens += int(decoder_latents.shape[0])

    @torch.inference_mode()
    def drain(self):
        """Yield audio for every block whose commit+lookahead tokens are ready."""
        while True:
            need = self._committed_tokens + self.commit_tokens + self.lookahead_tokens
            if self._n_tokens < need:
                return
            yield self._vocode_block(self.commit_tokens, self.lookahead_tokens, is_last=False)

    @torch.inference_mode()
    def finish(self) -> torch.Tensor | None:
        """Flush all remaining committed tokens (no future lookahead)."""
        if self._finished:
            return None
        self._finished = True
        remaining = self._n_tokens - self._committed_tokens
        if remaining <= 0:
            return None
        return self._vocode_block(remaining, 0, is_last=True)

    # -- internals -------------------------------------------------------- #
    def _latents(self):
        dec = torch.cat(self._dec_buf, dim=0) if len(self._dec_buf) > 1 else self._dec_buf[0]
        enc = torch.cat(self._enc_buf, dim=0) if len(self._enc_buf) > 1 else self._enc_buf[0]
        # Collapse to a single contiguous tensor to avoid re-concat each block.
        self._dec_buf = [dec]
        self._enc_buf = [enc]
        return dec, enc

    def _vocode_block(self, commit_tokens: int, lookahead_tokens: int, is_last: bool) -> torch.Tensor:
        voc = self.voc
        diffit = voc.diffit
        dec_all, enc_all = self._latents()

        tok0 = self._committed_tokens
        m_win = commit_tokens + lookahead_tokens
        tok1 = tok0 + m_win
        alat = dec_all[tok0:tok1].unsqueeze(0)   # [1, m_win, F]
        aenc = enc_all[tok0:tok1].unsqueeze(0)   # [1, m_win, F]

        commit_frames = commit_tokens * self.up
        T_win = m_win * self.up
        future_frames = lookahead_tokens * self.up
        frame0 = tok0 * self.up

        # --- aligned encoder (streaming KV cache, one pass per block) ---
        combined = torch.cat([alat, aenc], dim=-1)                     # [1, m_win, 2F]
        combined = diffit.input_combined_norm(diffit.input_combined_proj(combined))
        encoded, self.enc_pkv = diffit.aligned_latent_encoder(
            inputs_embeds=combined,
            attention_mask=None,
            use_cache=True,
            past_key_values=self.enc_pkv,
            commit_len=commit_tokens,
        )                                                              # [1, m_win, F]
        enc_hs = torch.nn.functional.interpolate(
            encoded.transpose(1, 2), size=T_win, mode="nearest",
        )                                                              # [1, F, T_win]

        # --- diffusion ODE (Euler), denoiser self-attn/conv KV cache ---
        noise = diffit.get_streaming_noise(
            (1, self.out_channels, frame0 + T_win), self.dev, self.dt,
        )[..., frame0:frame0 + T_win]
        x = noise
        t_span = torch.linspace(0, 1, self.n_steps + 1, device=self.dev, dtype=self.dt)
        snap = _snapshot_self_conv_cache(self.denoiser_pkv)
        for step_i in range(self.n_steps):
            if step_i > 0:
                _restore_self_conv_cache(self.denoiser_pkv, snap)
            timestep = t_span[step_i] * torch.ones(1, device=self.dev, dtype=self.dt)
            v, _ = diffit.forward(
                hidden_states=x,
                timestep=timestep,
                timestep_cond=self.spk,
                mel_codes_hidden_states=enc_hs,
                speech_prompt_hidden_states=self.prompt_hs,
                speech_prompt_attention_mask=self.prompt_mask,
                conditioning_free=False,
                cfg_training=False,
                use_cache=True,
                past_key_values=self.denoiser_pkv,
                block_num_frames=commit_frames,
            )
            x = x + (t_span[step_i + 1] - t_span[step_i]) * v

        # --- streaming VAE decode (FlowVAE decoder conv cache) ---
        audio, self.vae_cache = voc.input_layer.decode_streaming(
            x, cache=self.vae_cache, lookahead=future_frames,
        )                                                              # [1, 1, commit_frames*hop]

        self._committed_tokens = tok1 - lookahead_tokens
        return audio.squeeze(1).float()                                # [1, n_samples]
