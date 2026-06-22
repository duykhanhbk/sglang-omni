# SPDX-License-Identifier: Apache-2.0
"""Mel spectrogram computation for SpeechifyTTS speaker extraction.

Direct port of the GPTTTS / vllm-omni ``TacotronSTFT`` + ``TargetMelSpectrogram``
(``speechify_tts/utils/utils.py``). Exact-match Tacotron normalization constants
and the "recreate hann_window every forward" rule (so a bf16-cast parent module
does not contaminate the STFT window). Pure torch + librosa (mel filterbank).
"""

from __future__ import annotations

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

TACOTRON_MEL_MAX = 2.3143386840820312
TACOTRON_MEL_MIN = -11.512925148010254


def normalize_tacotron_mel(mel: torch.Tensor) -> torch.Tensor:
    return 2 * ((mel - TACOTRON_MEL_MIN) / (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN)) - 1


def denormalize_tacotron_mel(norm_mel: torch.Tensor) -> torch.Tensor:
    return ((norm_mel + 1) / 2) * (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN) + TACOTRON_MEL_MIN


class TacotronSTFT(nn.Module):
    def __init__(
        self,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        sampling_rate=22050,
        mel_fmin=0.0,
        mel_fmax=8000,
        center=False,
        device="cpu",
        return_real_imag=False,
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.n_fft = filter_length
        self.hop_size = hop_length
        self.win_size = win_length
        self.fmin = mel_fmin
        self.fmax = mel_fmax
        self.center = center
        self.return_real_imag = return_real_imag
        # mel_basis stays float32 (from_numpy is not converted by Module._apply).
        self._mel_basis_cpu = torch.from_numpy(
            librosa.filters.mel(
                sr=self.sampling_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mel_channels,
                fmin=self.fmin,
                fmax=self.fmax,
            )
        ).float()

    def mel_spectrogram(self, y, return_real_imag=False, return_stft=False):
        y = F.pad(
            y.unsqueeze(1),
            (
                int((self.n_fft - self.hop_size) / 2),
                int((self.n_fft - self.hop_size) / 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)
        hann_window = torch.hann_window(self.win_size).to(y.device)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=hann_window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        stft = torch.abs(spec)[0]
        spec = torch.view_as_real(spec)
        mel_basis = self._mel_basis_cpu.to(spec.device)
        amp_spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        mel_amp_spec = torch.matmul(mel_basis, amp_spec)
        mel_amp_spec = self.spectral_normalize_torch(mel_amp_spec)
        if return_real_imag:
            real = spec[:, :, :, 0]
            imag = spec[:, :, :, 1]
            log_amplitude = torch.log(
                torch.abs(torch.sqrt(torch.pow(real, 2) + torch.pow(imag, 2))) + 1e-5
            )
            phase = torch.atan2(imag, real)
            return mel_amp_spec, (log_amplitude, phase, real, imag)
        elif return_stft:
            return mel_amp_spec, stft
        return mel_amp_spec

    def spectral_normalize_torch(self, magnitudes):
        return self.dynamic_range_compression_torch(magnitudes)

    def dynamic_range_compression_torch(self, x, C=1, clip_val=1e-5):
        return torch.log(torch.clamp(x, min=clip_val) * C)

    def forward(self, y, return_real_imag=False):
        return self.mel_spectrogram(y, return_real_imag)


class TargetMelSpectrogram(nn.Module):
    def __init__(
        self,
        n_mel_channels=80,
        sampling_rate=22050,
        mel_fmax=8000,
        do_normalization=True,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        **kwargs,
    ):
        super().__init__()
        self.stft = TacotronSTFT(
            n_mel_channels=n_mel_channels,
            sampling_rate=sampling_rate,
            mel_fmax=mel_fmax,
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
        )
        self.do_normalization = do_normalization
        self.sampling_rate = sampling_rate
        self._hop_length = hop_length

    @property
    def hop_length(self):
        return self._hop_length

    def forward(self, inp, return_real_imag=False, return_stft=False):
        if len(inp.shape) == 3:
            inp = inp.squeeze(1)
        self.stft = self.stft.to(inp.device)
        if return_real_imag is True:
            mel, complex_info = self.stft.mel_spectrogram(inp, return_real_imag=True)
            if self.do_normalization:
                mel = normalize_tacotron_mel(mel)
            return mel, complex_info
        elif return_stft is True:
            mel, stft = self.stft.mel_spectrogram(inp, return_stft=True)
            if self.do_normalization:
                mel = normalize_tacotron_mel(mel)
            return mel, stft
        mel = self.stft.mel_spectrogram(inp)
        if self.do_normalization:
            mel = normalize_tacotron_mel(mel)
        return mel
