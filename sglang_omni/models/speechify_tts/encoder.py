# SPDX-License-Identifier: Apache-2.0
"""Torch-native SpeechifyTTS text encoder (T5-style conformer).

Port of ``vllm_omni/.../speechify_t5_tts/encoder.py`` + ``SpeechifyT5Encoder``
without the vLLM ``EncoderOnlyAttention`` / ``@support_torch_compile`` coupling.
Parameter names match the converted checkpoint so weights load directly:

    model.embed_tokens.weight
    model.text_encoder.layers.{i}.pre_self_attn_layernorm.weight        (self-attn layers)
    model.text_encoder.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    model.text_encoder.layers.{i}.pre_conv_layernorm.weight
    model.text_encoder.layers.{i}.conv.pointwise_conv1.{weight,bias}
    model.text_encoder.layers.{i}.conv.depthwise_conv.conv.{weight,bias}
    model.text_encoder.layers.{i}.conv.pointwise_conv2.{weight,bias}
    model.text_encoder.layers.{i}.pre_mlp_layernorm.weight
    model.text_encoder.layers.{i}.mlp.{gate,up,down}_proj.weight

For the MoE 4B MTL recipe: 12 layers, self-attention at [2, 5, 8, 11]
(bidirectional, "nope" positions, no qk-norm), conformer conv on every layer
(causal, kernel 4), SwiGLU MLP (mult 5). ``use_aux_context`` is False so the
T5 relative bias / aux cross-attention paths are intentionally omitted.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sglang_omni.models.speechify_tts.layers import (
    Attention,
    ConformerConv,
    RMSNorm,
    SwiGLUMLP,
)


class _EncoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.has_self_attention = layer_idx in config.self_attention_layer_indices
        if self.has_self_attention:
            self.pre_self_attn_layernorm = RMSNorm(config.hidden_size, eps=1e-5)
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                kv_head_dim=config.kv_head_dim,
                query_pre_attn_scalar=config.query_pre_attn_scalar,
                qk_norm=config.qk_norm,
            )

        conv_indices = config.conv_layer_indices
        if conv_indices is None:
            conv_indices = list(range(config.num_layers))
        self.has_conv = layer_idx in conv_indices and config.conv_kernel_size > 0
        if self.has_conv:
            self.pre_conv_layernorm = RMSNorm(config.hidden_size, eps=1e-5)
            self.conv = ConformerConv(
                dim=config.hidden_size,
                is_causal=config.is_conv_causal,
                expansion_factor=2,
                kernel_size=config.conv_kernel_size,
                activation="swish",
            )

        self.pre_mlp_layernorm = RMSNorm(config.hidden_size, eps=1e-5)
        self.mlp = SwiGLUMLP(
            dim=config.hidden_size,
            mult=config.mlp_multiplier,
            activation_fn=config.mlp_activation_fn,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.has_self_attention:
            residual = hidden_states
            hidden_states = self.pre_self_attn_layernorm(hidden_states)
            hidden_states = self.self_attn(hidden_states, is_causal=False)
            hidden_states = residual + hidden_states

        if self.has_conv:
            residual = hidden_states
            hidden_states = self.pre_conv_layernorm(hidden_states)
            conv_out, _ = self.conv(hidden_states)
            hidden_states = residual + conv_out

        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SpeechifyTextEncoder(nn.Module):
    """Standalone text encoder. Forward maps token IDs -> ``[T, hidden]``."""

    def __init__(self, config) -> None:
        super().__init__()
        # Accept either the top-level SpeechifyT5TTSConfig or the encoder sub-config.
        self.full_config = config
        encoder_config = getattr(config, "encoder_config", config)
        self.config = encoder_config
        vocab_size = getattr(config, "vocab_size", None) or encoder_config.vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, encoder_config.hidden_size)
        self.layers = nn.ModuleList(
            [_EncoderLayer(encoder_config, i) for i in range(encoder_config.num_layers)]
        )

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() > 1:
            input_ids = input_ids.reshape(-1)
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

    def load_weights(self, weights) -> set[str]:
        """Load ``model.embed_tokens.*`` / ``model.text_encoder.*`` weights.

        ``weights`` is an iterable of ``(name, tensor)``. Returns the set of
        local parameter names that were populated.
        """
        param_dict = dict(self.named_parameters())
        param_dict.update(dict(self.named_buffers()))
        loaded: set[str] = set()
        for name, weight in weights:
            if name.startswith("model.embed_tokens."):
                local = name[len("model.") :]
            elif name.startswith("model.text_encoder."):
                local = name[len("model.text_encoder.") :]
            else:
                continue
            param = param_dict.get(local)
            if param is None:
                continue
            if tuple(param.shape) != tuple(weight.shape):
                raise ValueError(
                    f"shape mismatch for {name} -> {local}: "
                    f"model {tuple(param.shape)} vs ckpt {tuple(weight.shape)}"
                )
            param.data.copy_(weight.to(param.dtype))
            loaded.add(local)
        return loaded
