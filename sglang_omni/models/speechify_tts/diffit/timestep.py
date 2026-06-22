"""Timestep embedding for diffusion models.

Ported from GPTTTS speechify_tts.layers.timestep.
"""

import torch
import torch.nn as nn


class CombinedTimestepLabelEmbeddings(nn.Module):
    def __init__(self, num_classes, embedding_dim, cond_proj_dim=None):
        super().__init__()
        self.emb = nn.Embedding(num_classes, embedding_dim)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, embedding_dim, bias=False)
        else:
            self.cond_proj = None

        self.embedding_dim = embedding_dim

    # Frequency basis cache keyed by (half_dim, dtype, device).  The basis
    # is deterministic, so rebuilding it on CPU (arange/exp) and shipping
    # it H2D on every call is pure overhead — py-spy showed this at ~7% of
    # the diffusion-stage CPU under c=16 load (one call per ODE step).
    # The cached tensor is computed with the exact same CPU ops as before
    # and moved to the device once, so the numerics are bit-identical.
    _FREQ_BASIS_CACHE: dict = {}

    def float_to_embedding(
        self,
        durations,
        scale_factor=1000.0,
        embedding_dim=512,
        dtype=torch.bfloat16,
        device="cpu",
    ):
        durations = durations * scale_factor
        half_dim = embedding_dim // 2
        cache_key = (half_dim, dtype, str(device))
        freqs = self._FREQ_BASIS_CACHE.get(cache_key)
        if freqs is None:
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            freqs = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb).to(device=device, dtype=dtype)
            self._FREQ_BASIS_CACHE[cache_key] = freqs
        emb = durations.to(dtype)[:, :, None] * freqs[None, None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb.to(device)

    def forward(self, timestep, timestep_cond=None):
        if timestep.dtype != torch.long:
            timesteps_proj = self.float_to_embedding(
                timestep.unsqueeze(1),
                embedding_dim=self.embedding_dim,
                dtype=torch.bfloat16,
                device=timestep.device,
            )
        else:
            timesteps_proj = self.emb(timestep[:, None])

        if timestep_cond is not None and self.cond_proj is not None:
            timesteps_proj += self.cond_proj(timestep_cond).unsqueeze(1)

        return timesteps_proj
