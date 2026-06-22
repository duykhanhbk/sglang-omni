import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from typing import Any, Dict, Optional, Tuple, Literal
import types


class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Position Embeddings (RoPE) with support for various scaling strategies.

    This module computes the cosine and sine embeddings required for RoPE, which are
    then applied to query and key tensors in a transformer's attention mechanism. It
    supports different RoPE types like linear, dynamic, yarn, etc., through a
    configurable backend.

    Args:
        rope_type: The type of RoPE scaling to use.
        max_position_embeddings: The maximum sequence length that this model might
            ever be used with.
        rope_config: A dictionary containing configuration parameters for the chosen
            `rope_type`.
        device: The device to initialize tensors on.
    """

    def __init__(
        self,
        rope_type: Literal["default", "linear", "dynamic", "yarn", "longrope", "llama3"] = "default",
        max_position_embeddings: int = 2048,
        rope_config: Dict[str, Any] = {
            "rope_theta": 100000,
            "hidden_size": 768,
            "num_attention_heads": 8,
            "partial_rotary_factor": 1.0,
        },
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.rope_type = rope_type
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        self.config = types.SimpleNamespace(**rope_config)

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(
        self, x: Tensor, position_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Computes the cosine and sine embeddings for the given positions.

        Args:
            x: An input tensor, used primarily to determine the device and dtype for
               the output embeddings. Its shape is not directly used in computation.
            position_ids: A tensor of shape (batch_size, sequence_length) containing
                          the positions for which to compute embeddings.

        Returns:
            A tuple containing the cosine and sine embeddings, each of shape
            (batch_size, sequence_length, head_dim).
        """
        # self.inv_freq: (d_head_half)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        )  # (batch, d_head_half, 1)
        position_ids_expanded = position_ids[:, None, :].float()  # (batch, 1, seq_len)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            # freqs: (batch, d_head_half, seq_len)
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
                1, 2
            )  # (batch, seq_len, d_head_half)
            emb = torch.cat((freqs, freqs), dim=-1)  # (batch, seq_len, d_head)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Optional[Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[Tensor, Tensor]:
    """Torch-native equivalent of ``LigerRopeFunction.apply``.

    ``cos``/``sin`` arrive as ``[batch, seq, head_dim]``; unsqueeze along the
    head dim so they broadcast over ``[batch, heads, seq, head_dim]`` query and
    key tensors. Identical math to the fused Liger kernel (rotate-half RoPE).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed
