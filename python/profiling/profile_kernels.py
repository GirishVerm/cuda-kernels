#!/usr/bin/env python3
"""
Profile CUDA kernels using PyTorch profiler and CUDA events

For advanced profiling, use:
  - NVIDIA Nsight Compute: ncu --set full python profile_kernels.py
  - NVIDIA Nsight Systems: nsys profile python profile_kernels.py
"""

import torch
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from attention_cuda import attention_naive, attention_tiled, attention_flash
from attention_cuda.utils import generate_test_inputs, get_gpu_info


def profile_with_pytorch_profiler(
    func,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    name: str,
    output_dir: Path
):
    """
    Profile using PyTorch's built-in profiler
    """
    print(f"\nProfiling {name} with PyTorch profiler...")
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # Warmup
        for _ in range(10):
            _ = func(Q, K, V)
        
        torch.cuda.synchronize()
        
        # Profile
        for _ in range(100):
            _ = func(Q, K, V)
        
        torch.cuda.synchronize()
    
    # Print results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Export Chrome trace
    trace_path = output_dir / f"trace_{name}.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace saved to: {trace_path}")
    
    return prof


def profile_with_cuda_events(
    func,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    name: str,
    num_iterations: int = 100
):
    """
    Profile using CUDA events for accurate timing
    """
    print(f"\nProfiling {name} with CUDA events...")
    
    # Warmup
    for _ in range(10):
        _ = func(Q, K, V)
    
    torch.cuda.synchronize()
    
    # Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Profile
    times = []
    for _ in range(num_iterations):
        start_event.record()
        _ = func(Q, K, V)
        end_event.record()
        
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    
    import numpy as np
    times = np.array(times)
    
    print(f"  Mean: {np.mean(times):.3f} ms")
    print(f"  Std:  {np.std(times):.3f} ms")
    print(f"  Min:  {np.min(times):.3f} ms")
    print(f"  Max:  {np.max(times):.3f} ms")
    
    return times


def profile_memory_usage(
    func,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    name: str
):
    """
    Profile memory usage
    """
    print(f"\nProfiling {name} memory usage...")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Run function
    _ = func(Q, K, V)
    torch.cuda.synchronize()
    
    # Get memory stats
    allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    reserved = torch.cuda.memory_reserved() / 1024**2    # MB
    peak = torch.cuda.max_memory_allocated() / 1024**2   # MB
    
    print(f"  Allocated: {allocated:.2f} MB")
    print(f"  Reserved:  {reserved:.2f} MB")
    print(f"  Peak:      {peak:.2f} MB")
    
    return {
        'allocated_mb': allocated,
        'reserved_mb': reserved,
        'peak_mb': peak
    }


def main():
    parser = argparse.ArgumentParser(description='Profile attention kernels')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size (default=1 for GTX 1650 Ti)')
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length (max 1024 for GTX 1650 Ti)')
    parser.add_argument('--head-dim', type=int, default=64)
    parser.add_argument('--output-dir', type=str, default='profiling_results')
    parser.add_argument('--impl', type=str, choices=['all', 'naive', 'tiled', 'flash'],
                       default='all', help='Which implementation to profile')
    
    args = parser.parse_args()
    
    # Check GPU
    gpu_info = get_gpu_info()
    print("\n" + "="*80)
    print("GPU Information")
    print("="*80)
    for key, value in gpu_info.items():
        print(f"{key}: {value}")
    
    if not gpu_info['available']:
        print("\nERROR: No CUDA GPU available!")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Profiling Configuration")
    print(f"{'='*80}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num heads: {args.num_heads}")
    print(f"Seq length: {args.seq_len}")
    print(f"Head dim: {args.head_dim}")
    print(f"Output dir: {output_dir}")
    
    # Generate inputs
    Q, K, V = generate_test_inputs(
        args.batch_size,
        args.num_heads,
        args.seq_len,
        args.head_dim
    )
    
    implementations = {}
    if args.impl == 'all':
        implementations = {
            'naive': attention_naive,
            'tiled': attention_tiled,
            'flash': attention_flash,
        }
    else:
        if args.impl == 'naive':
            implementations['naive'] = attention_naive
        elif args.impl == 'tiled':
            implementations['tiled'] = attention_tiled
        elif args.impl == 'flash':
            implementations['flash'] = attention_flash
    
    # Profile each implementation
    for name, func in implementations.items():
        print(f"\n{'='*80}")
        print(f"Profiling: {name}")
        print(f"{'='*80}")
        
        try:
            # PyTorch profiler
            profile_with_pytorch_profiler(func, Q, K, V, name, output_dir)
            
            # CUDA events
            profile_with_cuda_events(func, Q, K, V, name)
            
            # Memory usage
            profile_memory_usage(func, Q, K, V, name)
            
        except Exception as e:
            print(f"Error profiling {name}: {e}")
    
    print(f"\n{'='*80}")
    print("Profiling Instructions")
    print(f"{'='*80}")
    print("\nFor detailed kernel-level profiling, use NVIDIA Nsight tools:")
    print("\n1. Nsight Compute (kernel profiling):")
    print(f"   ncu --set full --target-processes all \\")
    print(f"       --export ncu_report \\")
    print(f"       python {__file__}")
    print("\n2. Nsight Systems (system-wide profiling):")
    print(f"   nsys profile --trace=cuda,nvtx --output=nsys_report \\")
    print(f"       python {__file__}")
    print("\n3. View Chrome traces:")
    print(f"   Open chrome://tracing in Chrome browser")
    print(f"   Load: {output_dir}/trace_*.json")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()

