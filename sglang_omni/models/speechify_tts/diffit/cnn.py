# SPDX-License-Identifier: Apache-2.0
"""Pure-torch CNN helpers used by the DiffiTv3 denoiser.

Only the modules required by ``diffitv3_core`` are kept here. The original
vllm-omni ``cnn.py`` also carried a Triton varlen causal depthwise conv used
by the AR conformer; that path is decoder-side and is not imported by the
diffusion vocoder, so it is intentionally dropped to keep this dependency
free of Triton / vLLM custom ops.
"""

from typing import Tuple

import torch
import torch.nn as nn


class GLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


def calc_same_padding(kernel_size: int) -> Tuple[int, int]:
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class DepthWiseConv1d(nn.Module):
    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        kernel_size: int,
        padding: Tuple[int, int],
        is_causal: bool = False,
    ):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, self.padding)
        return self.conv(x)
