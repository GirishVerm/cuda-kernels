#!/usr/bin/env python3
"""
Compare different attention implementations for correctness and performance
"""

import torch
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from attention_cuda import attention_naive, attention_tiled, attention_flash
from attention_cuda.utils import (
    generate_test_inputs,
    compute_attention_pytorch,
    check_correctness,
    measure_memory_usage
)


def compare_correctness(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 512,
    head_dim: int = 64
):
    """
    Compare correctness of all implementations against PyTorch reference
    """
    print(f"\n{'='*80}")
    print("CORRECTNESS COMPARISON")
    print(f"{'='*80}")
    print(f"Config: batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}\n")
    
    # Generate test inputs
    Q, K, V = generate_test_inputs(batch_size, num_heads, seq_len, head_dim, seed=42)
    
    # Reference implementation
    print("Computing PyTorch reference...")
    reference = compute_attention_pytorch(Q, K, V)
    
    implementations = {
        'naive': attention_naive,
        'tiled': attention_tiled,
        'flash': attention_flash,
    }
    
    print(f"\n{'Implementation':<15} {'Max Error':<15} {'Status':<10}")
    print("-" * 45)
    
    for name, func in implementations.items():
        try:
            output = func(Q, K, V)
            is_correct, max_error = check_correctness(output, reference)
            
            status = "✓ PASS" if is_correct else "✗ FAIL"
            print(f"{name:<15} {max_error:<15.2e} {status:<10}")
            
        except Exception as e:
            print(f"{name:<15} {'ERROR':<15} ✗ FAIL")
            print(f"  Error: {e}")


def compare_memory(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 512,
    head_dim: int = 64
):
    """
    Compare memory usage of implementations
    """
    print(f"\n{'='*80}")
    print("MEMORY USAGE COMPARISON")
    print(f"{'='*80}")
    print(f"Config: batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}\n")
    
    Q, K, V = generate_test_inputs(batch_size, num_heads, seq_len, head_dim)
    
    # Calculate theoretical input size
    input_size_mb = (Q.numel() * 3 * 4) / 1024 / 1024  # 3 tensors, 4 bytes per float32
    
    implementations = {
        'pytorch': compute_attention_pytorch,
        'naive': attention_naive,
        'tiled': attention_tiled,
        'flash': attention_flash,
    }
    
    print(f"Input tensors size: {input_size_mb:.2f} MB\n")
    print(f"{'Implementation':<15} {'Peak Memory (MB)':<20} {'Overhead (MB)':<15}")
    print("-" * 55)
    
    for name, func in implementations.items():
        try:
            output, memory_mb = measure_memory_usage(func, Q, K, V)
            overhead = memory_mb - input_size_mb
            
            print(f"{name:<15} {memory_mb:<20.2f} {overhead:<15.2f}")
            
        except Exception as e:
            print(f"{name:<15} ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description='Compare attention implementations')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--head-dim', type=int, default=64)
    parser.add_argument('--check', type=str, default='all',
                       choices=['all', 'correctness', 'memory'],
                       help='What to check')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    
    if args.check in ['all', 'correctness']:
        compare_correctness(
            args.batch_size,
            args.num_heads,
            args.seq_len,
            args.head_dim
        )
    
    if args.check in ['all', 'memory']:
        compare_memory(
            args.batch_size,
            args.num_heads,
            args.seq_len,
            args.head_dim
        )
    
    print()


if __name__ == '__main__':
    main()

