import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from jaxtyping import Float, jaxtyped
from torch import Tensor
from typeguard import typechecked as typechecker


class RMSNorm(nn.Module):
    """
    A Liger-optimized RMSNorm layer.

    Args:
        hidden_size (int): The size of the hidden dimension.
        eps (float): A small value added to the denominator for numerical stability.
        offset (float): An offset value, typically 0.0 for standard RMSNorm.
        casting_mode (str): The casting mode for the operation, e.g., "llama" or "gemma".
            NOTE:  "llama" mode will match to t5 layernorm.
        init_fn (str): The initialization function for the weight parameter, either "ones" or "zeros".
        in_place (bool): Whether to perform the operation in-place.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        offset: float = 0.0,
        casting_mode: str = "llama",
        init_fn: str = "ones",
        in_place: bool = True,
    ) -> None:
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"

        self.weight: nn.Parameter = nn.Parameter(
            torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size)
        )
        self.variance_epsilon: float
        self.offset: float
        self.casting_mode: str
        self.in_place: bool

        (
            self.variance_epsilon,
            self.offset,
            self.casting_mode,
            self.in_place,
        ) = (
            eps,
            offset,
            casting_mode,
            in_place,
        )

    @jaxtyped(typechecker=typechecker)
    def forward(self, hidden_states: Float[Tensor, "*batch hidden_size"]) -> Float[Tensor, "*batch hidden_size"]:
        """RMSNorm forward via ``F.rms_norm`` — matches GPTTTS canonical
        and the TRT engine build. The ``offset != 0`` fallback path is
        kept inline because no current call site uses it.
        """
        if self.offset == 0.0:
            return F.rms_norm(
                hidden_states, self.weight.shape,
                self.weight, self.variance_epsilon,
            )
        in_dtype = hidden_states.dtype
        if self.casting_mode == "llama":
            x_f32 = hidden_states.to(torch.float32)
            variance = x_f32.pow(2).mean(-1, keepdim=True)
            x_f32 = x_f32 * torch.rsqrt(variance + self.variance_epsilon)
            return x_f32.to(in_dtype) * (self.weight + self.offset)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        x = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return x * (self.weight + self.offset)

    def extra_repr(self) -> str:
        """Provides a string representation of the module's configuration."""
        return (
            f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, offset={self.offset}, in_place={self.in_place}"
        )


class LayerNorm(nn.Module):
    """
    A Liger-optimized LayerNorm module. Only support elementwise_affine=True.

    This module applies Layer Normalization over a mini-batch of inputs. It is
    designed to be a drop-in replacement for torch.nn.LayerNorm, but utilizes a
    custom, high-performance kernel for the computation.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        init_fn: Literal["ones", "zeros"] = "ones",
    ) -> None:
        """
        Initializes the LigerLayerNorm module.

        Args:
            hidden_size (int): The size of the last dimension of the input tensor.
            eps (float): A value added to the denominator for numerical stability.
            elementwise_affine (bool): If True, this module has learnable affine
                parameters (weight and bias).
            init_fn (Literal["ones", "zeros"]): Initialization for the weight parameter.
                'ones' for standard LayerNorm, 'zeros' for specific architectures.
        """
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"

        assert elementwise_affine, "LigerLayerNorm only supports elementwise_affine=True"

        self.hidden_size: int = hidden_size
        self.eps: float = eps
        self.elementwise_affine: bool = elementwise_affine

        self.weight = nn.Parameter(torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size))
        self.bias = nn.Parameter(torch.zeros((hidden_size,)))

    @jaxtyped(typechecker=typechecker)
    def forward(self, hidden_states: Float[Tensor, "*batch hidden_size"]) -> Float[Tensor, "*batch hidden_size"]:
        """LayerNorm forward via ``F.layer_norm`` — matches GPTTTS
        canonical and the TRT engine build.
        """
        return F.layer_norm(
            hidden_states, (self.hidden_size,),
            weight=self.weight, bias=self.bias, eps=self.eps,
        )

    def extra_repr(self) -> str:
        """Provides a string representation of the module's configuration."""
        return f"{self.hidden_size}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


def test_matching_rmsnorm() -> None:
    """
    Tests the numerical equivalence between LigerRMSNorm and T5LayerNorm from a reference implementation.
    """
    # This import is local to the test function as it's a dev dependency.
    from speechify_tts.experimental.reduction_t5 import T5LayerNorm

    # Constants for the test
    hidden_dim = 512
    epsilon = 1e-5
    batch_size = 1
    seq_len = 100

    # Ensure CUDA is available for the test
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    # Initialize layers
    t5_layernorm: T5LayerNorm = T5LayerNorm(hidden_dim, eps=epsilon).cuda()
    liger_layernorm: RMSNorm = RMSNorm(hidden_dim, eps=epsilon).cuda()

    # Create a random input tensor
    x: Float[Tensor, "batch_size seq_len hidden_dim"] = torch.randn(batch_size, seq_len, hidden_dim).cuda()

    # Get outputs from both layers
    a: Float[Tensor, "batch_size seq_len hidden_dim"] = t5_layernorm(x)
    b: Float[Tensor, "batch_size seq_len hidden_dim"] = liger_layernorm(x)

    # Calculate and check the difference
    diff: float = (a - b).abs().mean().item()
    tolerance = 1e-5

    print(f"DIFFERENCE: {diff}")
    assert diff <= tolerance, f"Difference {diff} exceeds tolerance {tolerance}"
    print("Test passed: LigerRMSNorm matches T5LayerNorm.")


@jaxtyped(typechecker=typechecker)
def test_matching_layernorm() -> None:
    """
    Tests numerical equivalence between LigerLayerNorm and torch.nn.LayerNorm.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping LayerNorm matching test.")
        return

    # --- Test Parameters ---
    hidden_dim: int = 512
    epsilon: float = 1e-5
    batch_size: int = 4
    seq_len: int = 128
    tolerance: float = 1e-5

    # --- Create random input ---
    x: Float[Tensor, "b s d"] = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32).cuda()

    # --- Test Case 1: With Elementwise Affine ---
    print("-" * 30)
    print("Testing with elementwise_affine=True")

    torch_layernorm = nn.LayerNorm(hidden_dim, eps=epsilon, elementwise_affine=True).cuda()
    liger_layernorm = LayerNorm(hidden_dim, eps=epsilon, elementwise_affine=True).cuda()

    # Ensure parameters are identical for a fair comparison
    with torch.no_grad():
        liger_layernorm.weight.copy_(torch_layernorm.weight)
        liger_layernorm.bias.copy_(torch_layernorm.bias)

    # Get outputs
    a: Float[Tensor, "b s d"] = torch_layernorm(x)
    b: Float[Tensor, "b s d"] = liger_layernorm(x)

    # Compare
    diff_affine: float = (a - b).abs().mean().item()
    print(f"DIFFERENCE (affine=True): {diff_affine}")
    assert diff_affine <= tolerance, f"Difference {diff_affine} exceeds tolerance {tolerance}"
    print("Test passed.")


if __name__ == "__main__":
    test_matching_rmsnorm()
    test_matching_layernorm()
