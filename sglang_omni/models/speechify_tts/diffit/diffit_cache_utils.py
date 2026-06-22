from typing import Optional, List
import torch
from transformers.cache_utils import DynamicCache


class DiffitPastKeyValues:
    """Cache container for DiffiT PyTorch forward path.

    Wraps DynamicCache for self-attention + per-layer conv caches.
    Used only by Transformer2DModel.forward(use_cache=True).

    The production TRT batched path uses slotted cache pools directly
    (SlottedKVCachePool, SlottedEncoderKVPool, SlottedPromptKVStore)
    and never touches this class.
    """

    def __init__(
        self,
        self_attention_cache: DynamicCache,
        conv_cache: List[Optional[torch.Tensor]],
        num_layers: int,
        prompt_cross_attention_cache: Optional[DynamicCache] = None,
        aligned_encoder_cache: Optional[DynamicCache] = None,
        inp_conv_cache: Optional[torch.Tensor] = None,
    ):
        self.self_attention_cache = self_attention_cache
        self.prompt_cross_attention_cache = prompt_cross_attention_cache or DynamicCache()
        self.aligned_encoder_cache = aligned_encoder_cache or DynamicCache()
        self.conv_cache = conv_cache
        self.num_layers = num_layers
        self.inp_conv_cache = inp_conv_cache

    @classmethod
    def create(
        cls,
        num_layers: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "DiffitPastKeyValues":
        del device, dtype  # DynamicCache tensors are materialized lazily on first update.
        return cls(
            self_attention_cache=DynamicCache(),
            conv_cache=[None for _ in range(num_layers)],
            num_layers=num_layers,
            prompt_cross_attention_cache=DynamicCache(),
            aligned_encoder_cache=DynamicCache(),
        )

    def get_seq_length(self) -> int:
        return self.self_attention_cache.get_seq_length(0)

    def get_conv_cache(self, layer_idx: int) -> Optional[torch.Tensor]:
        if layer_idx >= len(self.conv_cache):
            return None
        return self.conv_cache[layer_idx]

    def set_conv_cache(self, layer_idx: int, value: Optional[torch.Tensor]) -> None:
        if layer_idx >= len(self.conv_cache):
            self.conv_cache.extend([None] * (layer_idx + 1 - len(self.conv_cache)))
        self.conv_cache[layer_idx] = value

    def get_prompt_cross_attention_seq_length(self) -> int:
        return self.prompt_cross_attention_cache.get_seq_length(0)

    def get_aligned_encoder_seq_length(self) -> int:
        return self.aligned_encoder_cache.get_seq_length(0)
