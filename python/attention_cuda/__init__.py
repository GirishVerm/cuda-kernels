"""
Custom CUDA Attention Kernels for LLM Inference

High-performance attention implementations with progressive optimizations:
- Naive: Baseline implementation
- Tiled: Shared memory optimization
- Flash: IO-aware FlashAttention-style implementation
"""

__version__ = "0.1.0"

from .attention import (
    AttentionNaive,
    AttentionTiled,
    AttentionFlash,
    attention_naive,
    attention_tiled,
    attention_flash,
)

__all__ = [
    "AttentionNaive",
    "AttentionTiled",
    "AttentionFlash",
    "attention_naive",
    "attention_tiled",
    "attention_flash",
]

