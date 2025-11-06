# Optimization Techniques

This document details the specific optimization techniques used in our CUDA attention kernels.

## Table of Contents
- [Memory Optimizations](#memory-optimizations)
- [Compute Optimizations](#compute-optimizations)
- [Algorithm-Level Optimizations](#algorithm-level-optimizations)
- [Profiling and Analysis](#profiling-and-analysis)

## Memory Optimizations

### 1. Shared Memory Tiling

**Problem**: Global memory access is expensive (400-800 cycles latency)

**Solution**: Cache frequently accessed data in shared memory (20-40 cycles)

**Implementation**:
```cuda
__shared__ float Q_tile[TILE_SIZE][TILE_SIZE + 1];  // +1 avoids bank conflicts
__shared__ float K_tile[TILE_SIZE][TILE_SIZE + 1];

// Cooperative loading
if (threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {
    Q_tile[threadIdx.y][threadIdx.x] = Q[...];
    K_tile[threadIdx.y][threadIdx.x] = K[...];
}
__syncthreads();

// Use tile data multiple times
```

**Impact**: ~3x reduction in global memory traffic

### 2. Memory Coalescing

**Problem**: Uncoalesced access wastes bandwidth

**Good Pattern** (Coalesced):
```cuda
// Threads access consecutive memory
float val = data[blockIdx.x * blockDim.x + threadIdx.x];
```

**Bad Pattern** (Strided):
```cuda
// Threads access with stride
float val = data[threadIdx.x * stride];
```

**Verification**: Check with Nsight Compute:
```
Global Memory Load Efficiency: 100% (coalesced) vs 12.5% (strided on 32-byte)
```

### 3. Bank Conflict Avoidance

**Problem**: Shared memory organized in 32 banks. Conflicts serialize access.

**Solution**: Add padding to avoid conflicts
```cuda
// Without padding - potential conflicts
__shared__ float data[32][32];

// With padding - no conflicts
__shared__ float data[32][33];  // +1 column
```

**Why It Works**: Offsets each row, breaking conflict patterns

### 4. Register Pressure Management

**Problem**: Too many registers reduces occupancy

**Strategy**:
```cuda
// Use arrays when beneficial
float acc[4] = {0.0f};  // Compiler may optimize to registers

// Prefer compiler to manage
#pragma unroll
for (int i = 0; i < 4; i++) {
    acc[i] += ...;
}
```

**Check**: Use `--ptxas-options=-v` to see register usage

## Compute Optimizations

### 1. Warp-Level Primitives

**Warp Shuffle for Reductions**:
```cuda
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

**Benefits**:
- No shared memory needed
- Single instruction per iteration
- ~10x faster than shared memory reduction

**Use Cases**: Sum, max, any associative operation within warp

### 2. Instruction-Level Parallelism

**Technique**: Unroll loops to expose parallelism
```cuda
#pragma unroll
for (int i = 0; i < 4; i++) {
    result += a[i] * b[i];
}
```

**Compiler generates**:
```cuda
result += a[0] * b[0];
result += a[1] * b[1];  // Can execute in parallel
result += a[2] * b[2];
result += a[3] * b[3];
```

**Impact**: Better instruction pipeline utilization

### 3. Fast Math Operations

**Compiler Flags**:
```python
'--use_fast_math',  # Enables fast but less precise math
```

**Trade-offs**:
- `__expf()` instead of `exp()`: 2x faster, slightly less accurate
- `-ftz=true`: Flush denormals to zero
- `-prec-div=false`: Faster division

**When to Use**: Performance-critical sections where slight precision loss acceptable

### 4. Vectorized Memory Access

**Concept**: Load multiple values in single instruction
```cuda
// Load 4 floats at once (float4)
float4* data_vec = reinterpret_cast<float4*>(data);
float4 val = data_vec[idx];

// Equivalent to 4 scalar loads but faster
```

**Requirements**:
- Aligned memory access (16-byte for float4)
- Consecutive data needed

## Algorithm-Level Optimizations

### 1. Online Softmax (FlashAttention)

**Traditional Approach**:
```
1. Compute all scores: O(N²) memory
2. Find max
3. Compute exp and sum
4. Normalize
```

**Online Approach**:
```
1. Process in tiles
2. Maintain running max and sum
3. Update incrementally
4. Never materialize full matrix: O(N) memory
```

**Algorithm**:
```python
max_score = -inf
sum_exp = 0
output = 0

for each tile:
    new_max = max(max_score, tile_max)
    
    # Rescale previous results
    scale = exp(max_score - new_max)
    output *= scale
    sum_exp *= scale
    
    # Add new tile contribution
    tile_exp = exp(tile - new_max)
    output += tile_exp @ V_tile
    sum_exp += sum(tile_exp)
    
    max_score = new_max

output /= sum_exp
```

**Memory Savings**: O(N²) → O(N)

### 2. Kernel Fusion

**Problem**: Multiple kernel launches have overhead

**Solution**: Fuse operations into single kernel
```cuda
// Before: 3 separate kernels
matmul_kernel(Q, K, scores);    // Launch overhead
softmax_kernel(scores, attn);   // Launch overhead  
matmul_kernel(attn, V, output); // Launch overhead

// After: 1 fused kernel
attention_kernel(Q, K, V, output);  // Single launch
```

**Benefits**:
- Reduced kernel launch overhead
- Intermediate results stay in registers/shared memory
- Better data locality

### 3. Work Distribution

**Naive**: One thread per query
```cuda
int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
// Each thread does entire query
```

**Better**: Distribute across dimensions
```cuda
int q_idx = blockIdx.x * TILE_M + threadIdx.y;
int d_idx = threadIdx.x;
// Threads collaborate on query
```

**Best**: Hierarchical decomposition
```cuda
// Block processes tile of queries
// Warp processes subset
// Thread processes elements
```

## Profiling and Analysis

### 1. Metrics to Track

**Memory Metrics**:
- Global Memory Throughput (GB/s)
- L2 Cache Hit Rate
- Shared Memory Bank Conflicts
- Memory Efficiency (useful bytes / transferred bytes)

**Compute Metrics**:
- Achieved Occupancy
- SM Efficiency
- IPC (Instructions Per Cycle)
- Warp Execution Efficiency

**Performance Metrics**:
- Kernel Duration
- Grid/Block Configuration Efficiency
- Register Usage per Thread

### 2. Profiling with Nsight Compute

**Basic Profile**:
```bash
ncu --set full --target-processes all \
    --export report python script.py
```

**Key Sections to Check**:
1. **Memory Workload Analysis**
   - Look for high L2 miss rate
   - Check for uncoalesced access

2. **Compute Workload Analysis**
   - Check SM utilization
   - Look for warp stalls

3. **Occupancy**
   - Should be 50-75% for most kernels
   - Check limiters (registers, shared memory, blocks)

### 3. Roofline Analysis

**Concept**: Plot attained performance vs operational intensity

```
            Compute Bound
           /
          /
Peak    /
FLOPS  /
      /__________________ Memory Bound
      
      Memory Bandwidth
```

**Your Kernel**: Plot point based on:
- Arithmetic Intensity = FLOPs / Bytes Transferred
- Achieved Performance = FLOPs / Time

**Interpretation**:
- Below memory bound line: optimize memory access
- Below compute bound line: optimize computation
- Near roofline: well-optimized

### 4. Memory Bandwidth Analysis

**Theoretical Peak** (A100):
- HBM Bandwidth: 1.5 TB/s
- L2 Bandwidth: ~6 TB/s
- Shared Memory: ~15 TB/s per SM

**Measure Achieved**:
```python
achieved_bandwidth = (bytes_read + bytes_written) / time
utilization = achieved_bandwidth / theoretical_peak
```

**Target**: 60-80% of peak for memory-bound kernels

## Optimization Workflow

### Step-by-Step Process

1. **Implement Correctly**
   - Start with simple, correct implementation
   - Validate against reference

2. **Profile to Find Bottleneck**
   - Use Nsight Compute
   - Identify limiting factor (memory/compute)

3. **Apply Targeted Optimization**
   - If memory-bound: reduce memory traffic
   - If compute-bound: improve ILP, use fast math

4. **Measure Impact**
   - Compare before/after metrics
   - Verify correctness maintained

5. **Iterate**
   - Move to next bottleneck
   - Diminishing returns - know when to stop

### Common Pitfalls

❌ **Premature Optimization**: Optimize before profiling
✅ **Profile First**: Measure, then optimize bottleneck

❌ **Over-Optimization**: 2% gain for 50% code complexity
✅ **Pragmatic**: Focus on 10x+ opportunities

❌ **Ignoring Correctness**: Fast but wrong
✅ **Test Thoroughly**: Maintain correctness throughout

❌ **Single Configuration**: Optimize for one size only
✅ **Representative Workloads**: Test multiple configurations

## Performance Expectations

### Typical Speedups

| Optimization | Expected Speedup | Effort |
|--------------|------------------|--------|
| Memory coalescing | 2-5x | Low |
| Shared memory tiling | 2-3x | Medium |
| Warp-level primitives | 1.5-2x | Low |
| Kernel fusion | 1.5-3x | Medium |
| Online softmax | 2-4x | High |
| FP16 (not implemented) | 2-4x | Medium |

### Architecture-Specific Considerations

**V100** (Volta):
- 32 FP32 cores per SM
- 128 KB shared memory per SM
- Tensor Cores for mixed precision

**A100** (Ampere):
- 64 FP32 cores per SM
- 164 KB shared memory per SM
- 3rd gen Tensor Cores
- Better async copy

**H100** (Hopper):
- Thread block clusters
- TMA (Tensor Memory Accelerator)
- 4th gen Tensor Cores
- Transformer Engine

## References

- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Dissecting the NVIDIA Volta GPU Architecture](https://developer.nvidia.com/blog/inside-volta/)

