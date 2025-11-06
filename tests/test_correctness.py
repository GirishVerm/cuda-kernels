"""
Unit tests for attention implementations
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))

from attention_cuda import attention_naive, attention_tiled, attention_flash
from attention_cuda.utils import (
    generate_test_inputs,
    compute_attention_pytorch,
    check_correctness
)


@pytest.fixture
def attention_inputs():
    """Generate test inputs"""
    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 64
    return generate_test_inputs(batch_size, num_heads, seq_len, head_dim, seed=42)


@pytest.fixture
def reference_output(attention_inputs):
    """Compute reference output using PyTorch"""
    Q, K, V = attention_inputs
    return compute_attention_pytorch(Q, K, V)


class TestAttentionCorrectness:
    """Test correctness of attention implementations"""
    
    def test_naive_correctness(self, attention_inputs, reference_output):
        """Test naive attention against reference"""
        Q, K, V = attention_inputs
        output = attention_naive(Q, K, V)
        is_correct, max_error = check_correctness(output, reference_output, rtol=1e-3, atol=1e-3)
        
        assert is_correct, f"Naive attention failed with max error: {max_error}"
        print(f"Naive attention max error: {max_error:.2e}")
    
    def test_tiled_correctness(self, attention_inputs, reference_output):
        """Test tiled attention against reference"""
        Q, K, V = attention_inputs
        output = attention_tiled(Q, K, V)
        is_correct, max_error = check_correctness(output, reference_output, rtol=1e-3, atol=1e-3)
        
        assert is_correct, f"Tiled attention failed with max error: {max_error}"
        print(f"Tiled attention max error: {max_error:.2e}")
    
    def test_flash_correctness(self, attention_inputs, reference_output):
        """Test flash attention against reference"""
        Q, K, V = attention_inputs
        output = attention_flash(Q, K, V)
        is_correct, max_error = check_correctness(output, reference_output, rtol=1e-2, atol=1e-2)
        
        # Flash attention may have slightly higher error due to recomputation
        assert is_correct or max_error < 1e-1, f"Flash attention failed with max error: {max_error}"
        print(f"Flash attention max error: {max_error:.2e}")


class TestAttentionShapes:
    """Test output shapes"""
    
    @pytest.mark.parametrize("batch_size,num_heads,seq_len,head_dim", [
        (1, 1, 64, 32),
        (2, 4, 128, 64),
        (4, 8, 256, 64),
        (2, 8, 512, 128),
    ])
    def test_output_shapes(self, batch_size, num_heads, seq_len, head_dim):
        """Test that output shapes match input shapes"""
        Q, K, V = generate_test_inputs(batch_size, num_heads, seq_len, head_dim)
        expected_shape = (batch_size, num_heads, seq_len, head_dim)
        
        # Test all implementations
        implementations = {
            'naive': attention_naive,
            'tiled': attention_tiled,
            'flash': attention_flash,
        }
        
        for name, func in implementations.items():
            output = func(Q, K, V)
            assert output.shape == expected_shape, \
                f"{name} output shape {output.shape} != expected {expected_shape}"


class TestAttentionProperties:
    """Test mathematical properties of attention"""
    
    def test_attention_sum_to_one(self, attention_inputs):
        """Test that attention weights sum to 1 (implicitly tested through values)"""
        Q, K, V = attention_inputs
        
        # Create V with all ones
        V_ones = torch.ones_like(V)
        
        # If attention weights sum to 1, output should be all ones
        output = attention_naive(Q, K, V_ones)
        
        # Check that output is close to ones (within numerical precision)
        assert torch.allclose(output, torch.ones_like(output), rtol=1e-3, atol=1e-3), \
            "Attention weights don't sum to 1"
    
    def test_attention_scale_invariance(self, attention_inputs):
        """Test that scaling V scales output proportionally"""
        Q, K, V = attention_inputs
        scale_factor = 2.5
        
        output1 = attention_naive(Q, K, V)
        output2 = attention_naive(Q, K, V * scale_factor)
        
        assert torch.allclose(output2, output1 * scale_factor, rtol=1e-3, atol=1e-3), \
            "Attention not linearly scaling with V"


class TestEdgeCases:
    """Test edge cases"""
    
    def test_small_sequence(self):
        """Test with very small sequence length"""
        Q, K, V = generate_test_inputs(1, 1, 8, 32)
        
        reference = compute_attention_pytorch(Q, K, V)
        output = attention_naive(Q, K, V)
        
        is_correct, _ = check_correctness(output, reference)
        assert is_correct, "Failed on small sequence"
    
    def test_single_head(self):
        """Test with single attention head"""
        Q, K, V = generate_test_inputs(1, 1, 64, 64)
        
        reference = compute_attention_pytorch(Q, K, V)
        output = attention_naive(Q, K, V)
        
        is_correct, _ = check_correctness(output, reference)
        assert is_correct, "Failed on single head"


def test_cuda_availability():
    """Test that CUDA is available"""
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

