"""
PyTorch wrapper for custom CUDA attention kernels
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    import attention_cuda_kernels
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA kernels not available. Please build the extension first.")


class AttentionFunction(torch.autograd.Function):
    """Base class for attention autograd functions"""
    
    @staticmethod
    def check_inputs(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        """Validate input tensors"""
        assert Q.is_cuda, "Q must be a CUDA tensor"
        assert K.is_cuda, "K must be a CUDA tensor"
        assert V.is_cuda, "V must be a CUDA tensor"
        assert Q.dtype == torch.float32, "Only float32 supported currently"
        assert Q.shape == K.shape == V.shape, "Q, K, V must have same shape"
        assert len(Q.shape) == 4, "Expected 4D tensors [batch, num_heads, seq_len, head_dim]"


class NaiveAttentionFunction(AttentionFunction):
    """Autograd function for naive attention"""
    
    @staticmethod
    def forward(ctx, Q, K, V, scale=None):
        AttentionFunction.check_inputs(Q, K, V)
        
        if not CUDA_AVAILABLE:
            # Fallback to PyTorch
            return torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        
        output = attention_cuda_kernels.naive_forward(Q, K, V, scale)
        ctx.save_for_backward(Q, K, V, output)
        ctx.scale = scale
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        if not CUDA_AVAILABLE:
            raise NotImplementedError("CUDA kernels not available")
        
        Q, K, V, output = ctx.saved_tensors
        # Note: For full backward support, we'd need to save attention weights
        # For now, using PyTorch autograd
        Q.requires_grad = K.requires_grad = V.requires_grad = True
        with torch.enable_grad():
            out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
            grad_Q, grad_K, grad_V = torch.autograd.grad(out, (Q, K, V), grad_output)
        
        return grad_Q, grad_K, grad_V, None


class TiledAttentionFunction(AttentionFunction):
    """Autograd function for tiled attention"""
    
    @staticmethod
    def forward(ctx, Q, K, V, scale=None):
        AttentionFunction.check_inputs(Q, K, V)
        
        if not CUDA_AVAILABLE:
            return torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        
        output = attention_cuda_kernels.tiled_forward(Q, K, V, scale)
        ctx.save_for_backward(Q, K, V, output)
        ctx.scale = scale
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, output = ctx.saved_tensors
        # Using PyTorch autograd for backward
        Q.requires_grad = K.requires_grad = V.requires_grad = True
        with torch.enable_grad():
            out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
            grad_Q, grad_K, grad_V = torch.autograd.grad(out, (Q, K, V), grad_output)
        
        return grad_Q, grad_K, grad_V, None


class FlashAttentionFunction(AttentionFunction):
    """Autograd function for FlashAttention-style implementation"""
    
    @staticmethod
    def forward(ctx, Q, K, V, scale=None):
        AttentionFunction.check_inputs(Q, K, V)
        
        if not CUDA_AVAILABLE:
            return torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        
        output = attention_cuda_kernels.flash_forward(Q, K, V, scale)
        ctx.save_for_backward(Q, K, V)
        ctx.scale = scale
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V = ctx.saved_tensors
        # Using PyTorch autograd for backward
        Q.requires_grad = K.requires_grad = V.requires_grad = True
        with torch.enable_grad():
            out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
            grad_Q, grad_K, grad_V = torch.autograd.grad(out, (Q, K, V), grad_output)
        
        return grad_Q, grad_K, grad_V, None


# Functional API
def attention_naive(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    Naive attention implementation (baseline)
    
    Args:
        Q: Query tensor [batch, num_heads, seq_len, head_dim]
        K: Key tensor [batch, num_heads, seq_len, head_dim]
        V: Value tensor [batch, num_heads, seq_len, head_dim]
        scale: Optional scaling factor (default: 1/sqrt(head_dim))
    
    Returns:
        Output tensor [batch, num_heads, seq_len, head_dim]
    """
    return NaiveAttentionFunction.apply(Q, K, V, scale)


def attention_tiled(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    Tiled attention with shared memory optimization
    
    Args:
        Q: Query tensor [batch, num_heads, seq_len, head_dim]
        K: Key tensor [batch, num_heads, seq_len, head_dim]
        V: Value tensor [batch, num_heads, seq_len, head_dim]
        scale: Optional scaling factor (default: 1/sqrt(head_dim))
    
    Returns:
        Output tensor [batch, num_heads, seq_len, head_dim]
    """
    return TiledAttentionFunction.apply(Q, K, V, scale)


def attention_flash(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    FlashAttention-style IO-aware implementation
    
    Args:
        Q: Query tensor [batch, num_heads, seq_len, head_dim]
        K: Key tensor [batch, num_heads, seq_len, head_dim]
        V: Value tensor [batch, num_heads, seq_len, head_dim]
        scale: Optional scaling factor (default: 1/sqrt(head_dim))
    
    Returns:
        Output tensor [batch, num_heads, seq_len, head_dim]
    """
    return FlashAttentionFunction.apply(Q, K, V, scale)


# Module API
class AttentionNaive(nn.Module):
    """Naive attention module"""
    
    def __init__(self, scale: Optional[float] = None):
        super().__init__()
        self.scale = scale
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return attention_naive(Q, K, V, self.scale)


class AttentionTiled(nn.Module):
    """Tiled attention module"""
    
    def __init__(self, scale: Optional[float] = None):
        super().__init__()
        self.scale = scale
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return attention_tiled(Q, K, V, self.scale)


class AttentionFlash(nn.Module):
    """FlashAttention-style module"""
    
    def __init__(self, scale: Optional[float] = None):
        super().__init__()
        self.scale = scale
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return attention_flash(Q, K, V, self.scale)

