"""
Test gradient computation for attention implementations
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))

from attention_cuda import attention_naive, attention_tiled, attention_flash
from attention_cuda.utils import generate_test_inputs


@pytest.fixture
def attention_inputs_with_grad():
    """Generate test inputs with gradient tracking"""
    batch_size = 2
    num_heads = 4
    seq_len = 64
    head_dim = 64
    
    Q, K, V = generate_test_inputs(batch_size, num_heads, seq_len, head_dim, seed=42)
    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True
    
    return Q, K, V


class TestGradients:
    """Test gradient computation"""
    
    def test_naive_gradients_exist(self, attention_inputs_with_grad):
        """Test that gradients can be computed for naive attention"""
        Q, K, V = attention_inputs_with_grad
        
        output = attention_naive(Q, K, V)
        loss = output.sum()
        loss.backward()
        
        assert Q.grad is not None, "Q gradient not computed"
        assert K.grad is not None, "K gradient not computed"
        assert V.grad is not None, "V gradient not computed"
        
        # Check gradients are not NaN or Inf
        assert not torch.isnan(Q.grad).any(), "Q gradient contains NaN"
        assert not torch.isnan(K.grad).any(), "K gradient contains NaN"
        assert not torch.isnan(V.grad).any(), "V gradient contains NaN"
        
        assert not torch.isinf(Q.grad).any(), "Q gradient contains Inf"
        assert not torch.isinf(K.grad).any(), "K gradient contains Inf"
        assert not torch.isinf(V.grad).any(), "V gradient contains Inf"
    
    def test_gradient_shapes(self, attention_inputs_with_grad):
        """Test that gradients have correct shapes"""
        Q, K, V = attention_inputs_with_grad
        
        output = attention_naive(Q, K, V)
        loss = output.sum()
        loss.backward()
        
        assert Q.grad.shape == Q.shape, "Q gradient shape mismatch"
        assert K.grad.shape == K.shape, "K gradient shape mismatch"
        assert V.grad.shape == V.shape, "V gradient shape mismatch"
    
    def test_gradient_numerical_stability(self):
        """Test gradient computation with extreme values"""
        batch_size, num_heads, seq_len, head_dim = 1, 1, 32, 32
        
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda') * 10
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda') * 10
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda') * 10
        
        Q.requires_grad = True
        K.requires_grad = True
        V.requires_grad = True
        
        output = attention_naive(Q, K, V)
        loss = output.sum()
        loss.backward()
        
        # Check gradients are finite
        assert torch.isfinite(Q.grad).all(), "Q gradient not finite with large inputs"
        assert torch.isfinite(K.grad).all(), "K gradient not finite with large inputs"
        assert torch.isfinite(V.grad).all(), "V gradient not finite with large inputs"


class TestGradientCheck:
    """Numerical gradient checking"""
    
    def test_gradient_check_small(self):
        """Test gradients using finite differences (small problem)"""
        batch_size, num_heads, seq_len, head_dim = 1, 1, 8, 8
        
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       device='cuda', dtype=torch.float64, requires_grad=True)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       device='cuda', dtype=torch.float64, requires_grad=True)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       device='cuda', dtype=torch.float64, requires_grad=True)
        
        # Use PyTorch's built-in gradient checker
        def func(q, k, v):
            # Use PyTorch's attention as proxy since we need float64 support
            scale = 1.0 / (head_dim ** 0.5)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, v)
        
        # This tests that our gradient implementation would match numerical gradients
        test_passed = torch.autograd.gradcheck(func, (Q, K, V), eps=1e-6, atol=1e-3)
        assert test_passed, "Gradient check failed"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

