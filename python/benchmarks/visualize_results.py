#!/usr/bin/env python3
"""
Visualize benchmark results
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
sns.set_palette("husl")


def load_results(filepath: str) -> dict:
    """Load benchmark results from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_latency_comparison(results: dict, output_dir: Path):
    """Plot latency comparison across implementations"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    seq_lens = []
    implementations = {}
    
    for bench in results['benchmarks']:
        seq_len = bench['config']['seq_len']
        seq_lens.append(seq_len)
        
        for impl_name, impl_data in bench['implementations'].items():
            if 'mean_ms' in impl_data:
                if impl_name not in implementations:
                    implementations[impl_name] = []
                implementations[impl_name].append(impl_data['mean_ms'])
    
    x = np.arange(len(seq_lens))
    width = 0.2
    
    for i, (name, values) in enumerate(implementations.items()):
        offset = width * (i - len(implementations) / 2)
        ax.bar(x + offset, values, width, label=name)
    
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Attention Latency Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lens)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'latency_comparison.png'}")
    plt.close()


def plot_throughput_comparison(results: dict, output_dir: Path):
    """Plot throughput (TFLOPS) comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    seq_lens = []
    implementations = {}
    
    for bench in results['benchmarks']:
        seq_len = bench['config']['seq_len']
        seq_lens.append(seq_len)
        
        for impl_name, impl_data in bench['implementations'].items():
            if 'tflops' in impl_data:
                if impl_name not in implementations:
                    implementations[impl_name] = []
                implementations[impl_name].append(impl_data['tflops'])
    
    for name, values in implementations.items():
        ax.plot(seq_lens, values, marker='o', linewidth=2, markersize=8, label=name)
    
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Throughput (TFLOPS)', fontsize=12)
    ax.set_title('Attention Throughput Comparison', fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'throughput_comparison.png'}")
    plt.close()


def plot_speedup(results: dict, output_dir: Path):
    """Plot speedup relative to PyTorch"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    seq_lens = []
    implementations = {}
    
    for bench in results['benchmarks']:
        seq_len = bench['config']['seq_len']
        seq_lens.append(seq_len)
        
        for impl_name, impl_data in bench['implementations'].items():
            if impl_name != 'pytorch' and 'speedup_vs_pytorch' in impl_data:
                if impl_name not in implementations:
                    implementations[impl_name] = []
                implementations[impl_name].append(impl_data['speedup_vs_pytorch'])
    
    x = np.arange(len(seq_lens))
    width = 0.25
    
    for i, (name, values) in enumerate(implementations.items()):
        offset = width * (i - len(implementations) / 2)
        bars = ax.bar(x + offset, values, width, label=name)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}x',
                   ha='center', va='bottom', fontsize=9)
    
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='PyTorch baseline')
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Speedup vs PyTorch', fontsize=12)
    ax.set_title('Speedup Relative to PyTorch', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lens)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'speedup_comparison.png'}")
    plt.close()


def plot_latency_scaling(results: dict, output_dir: Path):
    """Plot how latency scales with sequence length"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    seq_lens = []
    implementations = {}
    
    for bench in results['benchmarks']:
        seq_len = bench['config']['seq_len']
        seq_lens.append(seq_len)
        
        for impl_name, impl_data in bench['implementations'].items():
            if 'mean_ms' in impl_data:
                if impl_name not in implementations:
                    implementations[impl_name] = []
                implementations[impl_name].append(impl_data['mean_ms'])
    
    # Linear scale
    for name, values in implementations.items():
        ax1.plot(seq_lens, values, marker='o', linewidth=2, markersize=8, label=name)
    
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('Latency Scaling (Linear)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log-log scale
    for name, values in implementations.items():
        ax2.loglog(seq_lens, values, marker='o', linewidth=2, markersize=8, label=name, base=2)
    
    # Add O(N^2) reference line
    ref_x = np.array(seq_lens)
    ref_y = ref_x**2 * implementations['pytorch'][0] / seq_lens[0]**2
    ax2.loglog(ref_x, ref_y, 'k--', alpha=0.5, linewidth=1, label='O(N²) reference')
    
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Latency (ms)', fontsize=12)
    ax2.set_title('Latency Scaling (Log-Log)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_scaling.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'latency_scaling.png'}")
    plt.close()


def generate_summary_report(results: dict, output_dir: Path):
    """Generate text summary report"""
    report = []
    report.append("="*80)
    report.append("ATTENTION BENCHMARK SUMMARY")
    report.append("="*80)
    report.append("")
    
    # GPU info
    if 'gpu_info' in results:
        report.append("GPU Information:")
        for key, value in results['gpu_info'].items():
            report.append(f"  {key}: {value}")
        report.append("")
    
    # Results table
    report.append("Latency Results (ms):")
    report.append("-" * 80)
    header = f"{'Seq Len':<10} {'PyTorch':<12} {'Naive':<12} {'Tiled':<12} {'Flash':<12}"
    report.append(header)
    report.append("-" * 80)
    
    for bench in results['benchmarks']:
        seq_len = bench['config']['seq_len']
        impls = bench['implementations']
        
        row = [f"{seq_len:<10}"]
        for name in ['pytorch', 'naive', 'tiled', 'flash']:
            if name in impls and 'mean_ms' in impls[name]:
                row.append(f"{impls[name]['mean_ms']:<12.3f}")
            else:
                row.append(f"{'N/A':<12}")
        
        report.append("".join(row))
    
    report.append("")
    report.append("Speedup vs PyTorch:")
    report.append("-" * 80)
    header = f"{'Seq Len':<10} {'Naive':<12} {'Tiled':<12} {'Flash':<12}"
    report.append(header)
    report.append("-" * 80)
    
    for bench in results['benchmarks']:
        seq_len = bench['config']['seq_len']
        impls = bench['implementations']
        
        row = [f"{seq_len:<10}"]
        for name in ['naive', 'tiled', 'flash']:
            if name in impls and 'speedup_vs_pytorch' in impls[name]:
                speedup = impls[name]['speedup_vs_pytorch']
                row.append(f"{speedup:<12.2f}x")
            else:
                row.append(f"{'N/A':<12}")
        
        report.append("".join(row))
    
    report.append("")
    report.append("="*80)
    
    report_text = "\n".join(report)
    
    # Print to console
    print("\n" + report_text)
    
    # Save to file
    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write(report_text)
    
    print(f"\nSaved: {output_dir / 'summary_report.txt'}")


def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('--input', type=str, default='benchmark_results.json',
                       help='Input JSON file with benchmark results')
    parser.add_argument('--output-dir', type=str, default='benchmark_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.input)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    print(f"Output directory: {output_dir}\n")
    
    # Generate plots
    plot_latency_comparison(results, output_dir)
    plot_throughput_comparison(results, output_dir)
    plot_speedup(results, output_dir)
    plot_latency_scaling(results, output_dir)
    
    # Generate summary report
    generate_summary_report(results, output_dir)
    
    print(f"\n✓ All visualizations generated successfully!")


if __name__ == '__main__':
    main()

