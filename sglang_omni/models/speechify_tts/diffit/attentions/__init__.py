from .flash_attention import flash_attention_forward
from .sdpa_attention import sdpa_attention_forward

ALL_ATTENTION_FUNCTIONS = {
    "flash_attention": flash_attention_forward,
    # Backward-compat aliases — both map to the same vLLM-builtin FA impl
    "flash_attention_2": flash_attention_forward,
    "flash_attention_3": flash_attention_forward,
    "sdpa": sdpa_attention_forward,
}

__all__ = [
    "flash_attention_forward",
    "sdpa_attention_forward",
]