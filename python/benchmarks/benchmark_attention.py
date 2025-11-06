#!/usr/bin/env python3
"""
Comprehensive benchmarking suite for attention implementations

Measures:
- Latency (mean, std, min, max)
- Throughput (TFLOPS)
- Memory bandwidth utilization
- GPU utilization
"""

import torch
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Callable
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from attention_cuda import attention_naive, attention_tiled, attention_flash
from attention_cuda.utils import (
    generate_test_inputs,
    compute_attention_pytorch,
    estimate_tflops,
    get_gpu_info
)


def benchmark_implementation(
    func: Callable,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    num_warmup: int = 10,
    num_iterations: int = 100
) -> Dict[str, float]:
    """
    Benchmark a single attention implementation
    
    Args:
        func: Attention function to benchmark
        Q, K, V: Input tensors
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
    
    Returns:
        Dictionary with benchmark results
    """
    # Warmup
    for _ in range(num_warmup):
        _ = func(Q, K, V)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        output = func(Q, K, V)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    batch_size, num_heads, seq_len, head_dim = Q.shape
    tflops = estimate_tflops(batch_size, num_heads, seq_len, head_dim, np.mean(times))
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'tflops': tflops,
    }


def benchmark_suite(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    num_warmup: int = 10,
    num_iterations: int = 100
) -> Dict:
    """
    Run complete benchmark suite
    
    Args:
        batch_size, num_heads, seq_len, head_dim: Tensor dimensions
        num_warmup: Warmup iterations
        num_iterations: Benchmark iterations
    
    Returns:
        Dictionary with all results
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking: batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")
    print(f"{'='*80}")
    
    # Generate inputs
    Q, K, V = generate_test_inputs(batch_size, num_heads, seq_len, head_dim)
    
    implementations = {
        'pytorch': compute_attention_pytorch,
        'naive': attention_naive,
        'tiled': attention_tiled,
        'flash': attention_flash,
    }
    
    results = {
        'config': {
            'batch_size': batch_size,
            'num_heads': num_heads,
            'seq_len': seq_len,
            'head_dim': head_dim,
        },
        'implementations': {}
    }
    
    for name, func in implementations.items():
        print(f"\nBenchmarking {name}...")
        try:
            bench_results = benchmark_implementation(
                func, Q, K, V, num_warmup, num_iterations
            )
            results['implementations'][name] = bench_results
            
            print(f"  Mean latency: {bench_results['mean_ms']:.3f} ms")
            print(f"  Throughput: {bench_results['tflops']:.2f} TFLOPS")
            
        except Exception as e:
            print(f"  Error: {e}")
            results['implementations'][name] = {'error': str(e)}
    
    # Compute speedups relative to PyTorch
    if 'pytorch' in results['implementations'] and 'mean_ms' in results['implementations']['pytorch']:
        pytorch_time = results['implementations']['pytorch']['mean_ms']
        for name in ['naive', 'tiled', 'flash']:
            if name in results['implementations'] and 'mean_ms' in results['implementations'][name]:
                custom_time = results['implementations'][name]['mean_ms']
                speedup = pytorch_time / custom_time
                results['implementations'][name]['speedup_vs_pytorch'] = speedup
                print(f"\n{name} speedup vs PyTorch: {speedup:.2f}x")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark attention implementations')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size (default=1 for GTX 1650 Ti)')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--seq-lengths', type=str, default='128,256,512,1024',
                       help='Comma-separated sequence lengths (default adjusted for GTX 1650 Ti 4GB VRAM)')
    parser.add_argument('--head-dim', type=int, default=64, help='Head dimension')
    parser.add_argument('--num-warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--num-iterations', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    # Check GPU availability
    gpu_info = get_gpu_info()
    print("\n" + "="*80)
    print("GPU Information")
    print("="*80)
    for key, value in gpu_info.items():
        print(f"{key}: {value}")
    
    if not gpu_info['available']:
        print("\nERROR: No CUDA GPU available!")
        return
    
    # Parse sequence lengths
    seq_lengths = [int(x.strip()) for x in args.seq_lengths.split(',')]
    
    # Run benchmarks
    all_results = {
        'gpu_info': gpu_info,
        'benchmarks': []
    }
    
    for seq_len in seq_lengths:
        results = benchmark_suite(
            args.batch_size,
            args.num_heads,
            seq_len,
            args.head_dim,
            args.num_warmup,
            args.num_iterations
        )
        all_results['benchmarks'].append(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")
    
    # Print summary
    print("\nSummary Table:")
    print(f"{'Seq Len':<10} {'PyTorch':<12} {'Naive':<12} {'Tiled':<12} {'Flash':<12}")
    print("-" * 60)
    
    for bench in all_results['benchmarks']:
        seq_len = bench['config']['seq_len']
        impls = bench['implementations']
        
        row = [f"{seq_len:<10}"]
        for name in ['pytorch', 'naive', 'tiled', 'flash']:
            if name in impls and 'mean_ms' in impls[name]:
                row.append(f"{impls[name]['mean_ms']:<12.3f}")
            else:
                row.append(f"{'ERROR':<12}")
        
        print("".join(row))
    
    print("\nSpeedup vs PyTorch:")
    print(f"{'Seq Len':<10} {'Naive':<12} {'Tiled':<12} {'Flash':<12}")
    print("-" * 50)
    
    for bench in all_results['benchmarks']:
        seq_len = bench['config']['seq_len']
        impls = bench['implementations']
        
        row = [f"{seq_len:<10}"]
        for name in ['naive', 'tiled', 'flash']:
            if name in impls and 'speedup_vs_pytorch' in impls[name]:
                speedup = impls[name]['speedup_vs_pytorch']
                row.append(f"{speedup:<12.2f}x")
            else:
                row.append(f"{'-':<12}")
        
        print("".join(row))


if __name__ == '__main__':
    main()

