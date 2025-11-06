# Profiling Guide

Complete guide to profiling CUDA attention kernels and interpreting results.

## Table of Contents
- [Quick Start](#quick-start)
- [Profiling Tools](#profiling-tools)
- [Interpreting Results](#interpreting-results)
- [Common Issues](#common-issues)

## Quick Start

### Basic Benchmarking
```bash
# Run benchmarks
python python/benchmarks/benchmark_attention.py \
    --seq-lengths 128,256,512,1024,2048 \
    --output benchmark_results.json

# Visualize results
python python/benchmarks/visualize_results.py \
    --input benchmark_results.json \
    --output-dir plots/
```

### PyTorch Profiler
```bash
# Profile with PyTorch's built-in profiler
python python/profiling/profile_kernels.py \
    --seq-len 512 \
    --impl flash

# View Chrome trace at chrome://tracing
```

### NVIDIA Nsight Compute
```bash
# Detailed kernel profiling
ncu --set full \
    --target-processes all \
    --export ncu_report \
    python python/profiling/profile_kernels.py --impl flash

# View in GUI
ncu-ui ncu_report.ncu-rep
```

### NVIDIA Nsight Systems
```bash
# System-wide timeline profiling
nsys profile \
    --trace=cuda,nvtx \
    --output=nsys_report \
    --force-overwrite true \
    python python/profiling/profile_kernels.py

# View in GUI
nsys-ui nsys_report.nsys-rep
```

## Profiling Tools

### 1. PyTorch Profiler

**Purpose**: Quick profiling with Python API

**Usage**:
```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    output = attention_flash(Q, K, V)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**Outputs**:
- Console table with timing breakdown
- Chrome trace for visualization
- Memory profiling data

**Best For**: Initial investigation, Python-level overhead

### 2. CUDA Events

**Purpose**: Accurate GPU timing

**Usage**:
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
output = attention_flash(Q, K, V)
end.record()

torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

**Advantages**:
- Accurate GPU time measurement
- Low overhead
- Simple API

**Best For**: Benchmarking, comparing implementations

### 3. NVIDIA Nsight Compute (ncu)

**Purpose**: Detailed kernel-level profiling

**Key Metrics**:
- **Memory Throughput**: GB/s for global, L2, shared memory
- **Compute Throughput**: FLOP/s achieved
- **Occupancy**: Active warps / max warps
- **Warp Stall Reasons**: Why warps aren't executing
- **Memory Access Patterns**: Coalescing efficiency

**Common Commands**:
```bash
# Full metrics
ncu --set full script.py

# Specific metrics
ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct \
    script.py

# Roofline analysis
ncu --section SpeedOfLight script.py

# Compare two runs
ncu --set full --export baseline script.py
ncu --set full --export optimized script.py
# Then compare in GUI
```

**Best For**: Optimizing individual kernels, finding bottlenecks

### 4. NVIDIA Nsight Systems (nsys)

**Purpose**: System-wide timeline profiling

**Shows**:
- Kernel launches and durations
- Memory copies
- CPU-GPU interactions
- CUDA API calls
- Thread activities

**Common Commands**:
```bash
# Basic profiling
nsys profile python script.py

# With options
nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=cpu \
    --cuda-memory-usage=true \
    python script.py
```

**Best For**: Understanding overall execution flow, finding idle time

## Interpreting Results

### Memory-Bound Kernels

**Indicators**:
- Memory throughput near peak (~1.5 TB/s on A100)
- Low compute throughput
- High memory access latency
- Nsight: "Memory Bound" in Speed of Light

**Example Output**:
```
Memory Throughput: 1200 GB/s (80% of peak)
Compute Throughput: 5 TFLOPS (3% of peak)
```

**Optimization Strategy**:
1. Reduce memory accesses (caching, tiling)
2. Improve memory coalescing
3. Use shared memory
4. Increase arithmetic intensity

### Compute-Bound Kernels

**Indicators**:
- High compute throughput
- Low memory throughput relative to compute
- High SM utilization
- Nsight: "Compute Bound" in Speed of Light

**Example Output**:
```
Memory Throughput: 200 GB/s (13% of peak)
Compute Throughput: 150 TFLOPS (75% of peak)
```

**Optimization Strategy**:
1. Increase ILP (instruction-level parallelism)
2. Use fast math
3. Better warp occupancy
4. Reduce control flow divergence

### Occupancy Analysis

**Formula**:
```
Occupancy = Active Warps / Max Warps per SM
```

**Common Limiters**:
1. **Registers**: Too many registers per thread
2. **Shared Memory**: Too much SMEM per block
3. **Block Size**: Too few or too many threads

**Example**:
```
Theoretical Occupancy: 100%
Achieved Occupancy: 62%
Limiter: Registers (64 per thread, 32 max for 100%)
```

**Optimal Occupancy**: Usually 50-75% is sufficient

### Warp Stall Analysis

**Common Stall Reasons**:

1. **Memory Throttle**: Waiting for memory
   - Solution: Improve memory access patterns

2. **Execution Dependency**: Instruction dependencies
   - Solution: Increase ILP, reorder instructions

3. **Memory Dependency**: Waiting for data
   - Solution: Prefetch, hide latency with more work

4. **Synchronization**: Waiting at __syncthreads()
   - Solution: Reduce sync points, balance work

**Nsight Output**:
```
Warp Stall Reasons:
  Memory Throttle: 45%
  Execution Dependency: 25%
  Synchronization: 15%
  Other: 15%
```

### Roofline Analysis

**Interpretation**:
```
       |   /
       |  / Compute Bound
FLOPS  | /
       |/
       /|_______________ Memory Bound
       |
      Operational Intensity (FLOPs/Byte)
```

**Your Kernel Position**:
- **Below and left**: Memory optimization needed
- **Below and right**: Compute optimization needed
- **On line**: Well-optimized for current algorithm
- **Above line**: Impossible (check measurement)

**Improving Position**:
- **Move right**: Increase arithmetic intensity (reuse data)
- **Move up**: Optimize what's limiting you

## Common Issues

### Issue 1: Low Memory Throughput

**Symptoms**:
```
Memory Throughput: 150 GB/s (10% of peak)
Global Memory Load Efficiency: 25%
```

**Causes**:
- Uncoalesced memory access
- Wrong access patterns
- Memory bank conflicts

**Debug**:
```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
              l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio \
    script.py
```

**Fix**: Ensure consecutive threads access consecutive memory

### Issue 2: Low Occupancy

**Symptoms**:
```
Achieved Occupancy: 12.5%
Limiter: Shared Memory
```

**Causes**:
- Too much shared memory per block
- Too many registers per thread
- Block size not multiple of 32

**Fix**:
- Reduce shared memory usage
- Use `__launch_bounds__` to limit registers
- Adjust block dimensions

### Issue 3: Long Kernel Duration

**Symptoms**:
```
Kernel Time: 50ms (expected ~5ms)
```

**Debug Strategy**:
1. Check if correct GPU being used
2. Verify work distribution
3. Look for serialization
4. Check for memory allocations in kernel

### Issue 4: Numerical Issues

**Symptoms**:
- NaN or Inf in output
- Results differ from CPU
- Unstable training

**Causes**:
- Softmax overflow
- Division by zero
- Loss of precision

**Debug**:
```python
# Check for NaN/Inf
assert not torch.isnan(output).any()
assert not torch.isinf(output).any()

# Compare with double precision
output_fp64 = kernel_fp64(Q_fp64, K_fp64, V_fp64)
error = torch.max(torch.abs(output - output_fp64.float()))
```

## Performance Checklist

### Memory Optimization
- [ ] Memory accesses are coalesced
- [ ] Using shared memory for frequently accessed data
- [ ] No bank conflicts in shared memory
- [ ] Memory transactions are efficient

### Compute Optimization
- [ ] Good occupancy (50-75%)
- [ ] Using warp-level primitives where possible
- [ ] Minimal branch divergence
- [ ] Unrolled critical loops

### Algorithm
- [ ] Kernel fusion applied where beneficial
- [ ] Optimal work distribution
- [ ] Minimal synchronization points
- [ ] Data reuse maximized

### Configuration
- [ ] Block size is multiple of warp size
- [ ] Grid size saturates GPU
- [ ] Register/SMEM usage balanced
- [ ] Correct compile flags enabled

## Example Analysis Session

### Step 1: Initial Benchmark
```bash
python benchmark_attention.py --seq-len 512
# Result: 10ms per iteration
```

### Step 2: Profile with PyTorch
```bash
python profile_kernels.py --impl flash
# Observation: Kernel takes 8ms, 2ms overhead
```

### Step 3: Deep Dive with Nsight Compute
```bash
ncu --set full script.py
# Key findings:
# - Memory throughput: 60% of peak
# - Occupancy: 45%
# - Limiter: Shared memory
```

### Step 4: Optimize
- Reduce shared memory usage
- Increase block size
- Improve memory access pattern

### Step 5: Verify Improvement
```bash
python benchmark_attention.py --seq-len 512
# Result: 5ms per iteration (2x speedup!)
```

## References

- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [PyTorch Profiler](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
- [CUDA Profiling Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)

