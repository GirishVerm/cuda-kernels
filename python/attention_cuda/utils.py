"""
Utility functions for attention benchmarking and testing
"""

import torch
import numpy as np
from typing import Tuple, Optional


def generate_test_inputs(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate random Q, K, V tensors for testing
    
    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension per head
        device: Device to create tensors on
        dtype: Data type
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (Q, K, V) tensors
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    shape = (batch_size, num_heads, seq_len, head_dim)
    
    Q = torch.randn(shape, device=device, dtype=dtype)
    K = torch.randn(shape, device=device, dtype=dtype)
    V = torch.randn(shape, device=device, dtype=dtype)
    
    return Q, K, V


def compute_attention_pytorch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    Reference PyTorch implementation of attention
    
    Args:
        Q, K, V: Input tensors
        scale: Optional scaling factor
    
    Returns:
        Attention output
    """
    if scale is None:
        scale = 1.0 / np.sqrt(Q.shape[-1])
    
    # Q @ K^T
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Softmax
    attention_weights = torch.softmax(scores, dim=-1)
    
    # Weighted sum of values
    output = torch.matmul(attention_weights, V)
    
    return output


def check_correctness(
    output_custom: torch.Tensor,
    output_reference: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3
) -> Tuple[bool, float]:
    """
    Check if custom implementation matches reference
    
    Args:
        output_custom: Output from custom kernel
        output_reference: Output from reference implementation
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        Tuple of (is_correct, max_error)
    """
    max_error = torch.max(torch.abs(output_custom - output_reference)).item()
    is_correct = torch.allclose(output_custom, output_reference, rtol=rtol, atol=atol)
    
    return is_correct, max_error


def measure_memory_usage(func, *args, **kwargs) -> Tuple[torch.Tensor, float]:
    """
    Measure peak GPU memory usage during function execution
    
    Args:
        func: Function to measure
        *args, **kwargs: Arguments to pass to function
    
    Returns:
        Tuple of (output, memory_mb)
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    output = func(*args, **kwargs)
    
    torch.cuda.synchronize()
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return output, memory_mb


def estimate_tflops(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    time_ms: float
) -> float:
    """
    Estimate TFLOPS for attention computation
    
    Attention requires:
    - Q @ K^T: 2 * B * H * N^2 * D FLOPs
    - Softmax: ~5 * B * H * N^2 FLOPs (approximate)
    - Attn @ V: 2 * B * H * N^2 * D FLOPs
    
    Args:
        batch_size: Batch size
        num_heads: Number of heads
        seq_len: Sequence length
        head_dim: Head dimension
        time_ms: Execution time in milliseconds
    
    Returns:
        TFLOPS
    """
    # Q @ K^T and Attn @ V
    matmul_flops = 2 * 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    
    # Softmax (approximate)
    softmax_flops = 5 * batch_size * num_heads * seq_len * seq_len
    
    total_flops = matmul_flops + softmax_flops
    
    # Convert to TFLOPS
    time_s = time_ms / 1000.0
    tflops = (total_flops / time_s) / 1e12
    
    return tflops


def get_gpu_info() -> dict:
    """
    Get GPU information
    
    Returns:
        Dictionary with GPU details
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }

