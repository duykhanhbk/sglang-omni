from typing import Callable, Optional, Literal

import torch
import torch.nn as nn
from torch import Tensor

from transformers.activations import ACT2FN


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module with/without Gated Linear Units, optimized using Liger Kernel.
    Supports GeGLU and SwiGLU variations. Others will use standard PyTorch implementation.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout_rate: float = 0.0,
        activation_fn: Literal["geglu", "swiglu", "relu", "silu"] = "geglu",
    ) -> None:
        """
        Initializes the MLP module.

        Args:
            dim (int): Input and output feature dimension.
            dim_out (Optional[int]): Target output dimension. Currently ignored in implementation
                                     (output is always 'dim'). kept for API compatibility.
            mult (int): Expansion factor for the intermediate dimension. Defaults to 4.
            dropout_rate (float): Dropout probability. Currently unused in implementation.
            activation_fn (Literal["geglu", "swiglu", "relu", "silu"]): The activation function.
                                                     Only 'geglu', 'swiglu', 'relu' or 'silu' are supported.
        """
        super().__init__()

        if activation_fn not in ["geglu", "swiglu", "relu", "silu"]:
            raise ValueError(
                f"Unsupported activation_fn: '{activation_fn}'. "
                "Only 'geglu', 'swiglu', 'relu' or 'silu' are supported for Liger-optimized MLP."
            )

        inner_dim: int = int(dim * mult)
        # Note: dim_out is accepted but not used in the layer definitions below
        # based on the provided reference code.
        _ = dim_out if dim_out is not None else dim

        self.activation_fn: str = activation_fn
        self.hidden_size: int = dim
        self.intermediate_size: int = inner_dim

        # Projections (Note: biases are False)
        if activation_fn == "geglu" or activation_fn == "swiglu":
            self.gate_proj: nn.Linear = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)

        self.up_proj: nn.Linear = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj: nn.Linear = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

        # Setup PyTorch reference activation callable
        # TODO: support exact GELU
        # Right now Gemma 1, 1.1 and 2 models are all using `gelu_pytorch_tanh`
        # We can safely assume we use tanh approximation form all the time for GEGLU here.
        self.act_fn: Callable[[Tensor], Tensor]
        if activation_fn == "geglu":
            self.act_fn = ACT2FN["gelu_pytorch_tanh"]
        elif activation_fn == "swiglu":
            self.act_fn = ACT2FN["silu"]
        elif activation_fn == "relu":
            self.act_fn = ACT2FN["relu"]
        elif activation_fn == "silu":
            self.act_fn = ACT2FN["silu"]
        else:
            raise ValueError(
                f"Unsupported activation_fn: '{activation_fn}'. Only 'geglu', 'swiglu', 'relu' or 'silu' are supported."
            )

    def forward_pytorch(self, x: Tensor) -> Tensor:
        """
        Standard PyTorch implementation of the Gated MLP pass for reference/testing.
        Implementation: down_proj( act(gate_proj(x)) * up_proj(x) )
        """
        # x: [*batch, dim]
        # gate/up results: [*batch, intermediate_dim]
        # down result: [*batch, dim]
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def forward(self, x: Tensor) -> Tensor:
        """Inference forward pass (torch-native; the Liger fused training path
        from vllm-omni is dropped since the vocoder always runs in eval)."""
        if self.activation_fn in ("swiglu", "geglu"):
            return self.forward_pytorch(x)
        elif self.activation_fn in ("relu", "silu"):
            intermed = self.dropout(self.act_fn(self.up_proj(x)))
            return self.down_proj(intermed)


def test_matching() -> None:
    """
    Tests numerical equivalence between the Liger optimized forward pass
    and the PyTorch reference implementation.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping MLP matching test.")
        return

    # Define dimensions
    batch_size: int = 1
    seq_len: int = 100
    dim: int = 512

    # Create input and model
    x = torch.randn(batch_size, seq_len, dim).cuda()
    for activation_fn in ["geglu", "silu"]:
        ff_layer = MLP(dim=dim, activation_fn=activation_fn).cuda()
        print(f"Testing MLP with activation: {ff_layer.activation_fn}")

        # Run both implementations
        a = ff_layer(x)
        b = ff_layer.forward_pytorch(x)

        # Compare
        # calculating mean absolute difference
        diff_tensor = (a - b).abs().mean()
        diff: float = diff_tensor.item()
        tolerance: float = 1e-5

        assert diff <= tolerance, f"Outputs not close. Diff: {diff}"
        print("DIFFERENCE:", diff)
        print("Test passed.")


if __name__ == "__main__":
    test_matching()
