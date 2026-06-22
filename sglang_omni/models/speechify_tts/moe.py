# SPDX-License-Identifier: Apache-2.0
"""Torch-native Sparse-MoE block for the SpeechifyTTS AR decoder.

Framework-free re-implementation of vllm-omni ``speechify/moe.py`` with the
vLLM ``fused_experts`` Triton kernel replaced by a plain grouped matmul over
the routed experts. Routing math is bit-identical:

    sigmoid(gate(x)) -> (optional expert_bias-corrected) top-k
    -> optional norm_topk_prob renormalization -> routed_scaling_factor

Parameter layout matches the converted checkpoint exactly so weights load
without renaming:

    mlp.gate.weight                 [num_experts, hidden]
    mlp.experts.gate_up_proj        [num_experts, 2*inter, hidden]
    mlp.experts.down_proj           [num_experts, hidden, inter]
    mlp.expert_bias                 [num_experts] (float32 buffer)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def is_moe_layer(
    layer_idx: int, num_experts: int, num_dense_layers: int = 0, moe_layer_stride: int = 1
) -> bool:
    if num_experts <= 1:
        return False
    if layer_idx < num_dense_layers:
        return False
    return (layer_idx - num_dense_layers) % moe_layer_stride == 0


class MoeExperts(nn.Module):
    def __init__(self, num_experts: int, hidden_dim: int, intermediate_dim: int,
                 activation: str = "swiglu") -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_dim, hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_dim, intermediate_dim)
        )
        self.activation = activation

    def _act(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        if self.activation == "geglu":
            return F.gelu(gate, approximate="tanh") * up
        return F.silu(gate) * up  # swiglu / silu

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        # hidden_states [N, H], top_k_index/weights [N, k]
        #
        # Vectorised grouped matmul: gather the routed experts' weight matrices
        # and batch the two projections with ``bmm``. This is sync-free and
        # launch-light (critical for token-by-token decode, where the previous
        # per-expert Python loop issued a ``.any()`` host sync per expert per
        # layer). Numerics are identical: each (token, slot) pair multiplies by
        # its expert's weights and the weighted results are summed per token.
        n, h = hidden_states.shape
        k = top_k_index.shape[1]
        flat_e = top_k_index.reshape(-1)                              # [n*k]
        flat_w = top_k_weights.reshape(-1).to(hidden_states.dtype)    # [n*k]
        x = hidden_states.unsqueeze(1).expand(n, k, h).reshape(n * k, h)  # [n*k, H]
        gu_w = self.gate_up_proj.index_select(0, flat_e)             # [n*k, 2*inter, H]
        gu = torch.bmm(gu_w, x.unsqueeze(-1)).squeeze(-1)            # [n*k, 2*inter]
        act = self._act(gu)                                          # [n*k, inter]
        dn_w = self.down_proj.index_select(0, flat_e)               # [n*k, H, inter]
        y = torch.bmm(dn_w, act.unsqueeze(-1)).squeeze(-1)          # [n*k, H]
        y = y * flat_w.unsqueeze(-1)
        tok_idx = torch.arange(n, device=hidden_states.device).repeat_interleave(k)
        out = hidden_states.new_zeros(n, h)
        out.index_add_(0, tok_idx, y)
        return out


class SparseMoeBlock(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int,
                 moe_intermediate_size: int, routed_scaling_factor: float = 1.0,
                 norm_topk_prob: bool = True, use_expert_bias: bool = False,
                 mlp_activation_fn: str = "swiglu") -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.use_expert_bias = use_expert_bias
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = MoeExperts(
            num_experts=num_experts, hidden_dim=hidden_size,
            intermediate_dim=moe_intermediate_size, activation=mlp_activation_fn,
        )
        if use_expert_bias:
            self.register_buffer("expert_bias", torch.zeros(num_experts, dtype=torch.float32))

    @classmethod
    def from_decoder_config(cls, dc) -> "SparseMoeBlock":
        return cls(
            hidden_size=dc.hidden_size, num_experts=dc.num_experts,
            top_k=dc.num_experts_per_tok, moe_intermediate_size=dc.moe_intermediate_size,
            routed_scaling_factor=dc.routed_scaling_factor, norm_topk_prob=dc.norm_topk_prob,
            use_expert_bias=dc.use_expert_bias, mlp_activation_fn=dc.mlp_activation_fn,
        )

    def route_tokens_to_experts(self, router_logits: torch.Tensor):
        routing_weights = router_logits.float().sigmoid()
        if self.use_expert_bias:
            scores = routing_weights + self.expert_bias
            _, selected = torch.topk(scores, k=self.top_k, dim=-1)
            routing_weights = torch.gather(routing_weights, 1, selected)
        else:
            routing_weights, selected = torch.topk(routing_weights, k=self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / (
                routing_weights.sum(dim=-1, keepdim=True) + 1e-6
            )
        routing_weights = routing_weights * self.routed_scaling_factor
        return selected.to(torch.int64), routing_weights.to(torch.float32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        selected, weights = self.route_tokens_to_experts(self.gate(flat))
        out = self.experts(flat, selected, weights)
        return out.reshape(shape)
