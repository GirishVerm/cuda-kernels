# Design Decisions

This document explains the key design decisions in implementing custom CUDA kernels for attention mechanisms.

## Table of Contents
- [Overall Architecture](#overall-architecture)
- [Implementation Strategy](#implementation-strategy)
- [Memory Management](#memory-management)
- [Performance Considerations](#performance-considerations)

## Overall Architecture

### Three-Tier Implementation Approach

We implement three versions of attention with increasing optimization levels:

1. **Naive Implementation** - Correctness baseline
2. **Tiled Implementation** - Memory hierarchy optimization
3. **FlashAttention-style** - IO-aware implementation

This progressive approach demonstrates:
- Understanding of GPU architecture fundamentals
- Ability to identify and optimize bottlenecks
- Knowledge of state-of-the-art techniques

## Implementation Strategy

### 1. Naive Attention

**Purpose**: Establish correctness baseline and understand the problem

**Approach**:
```
for each query position:
    compute scores = Q[i] @ K^T
    apply softmax
    compute output = attention @ V
```

**Characteristics**:
- Materializes full attention matrix: O(N²) memory
- Simple implementation prioritizing correctness
- Serves as reference for optimization validation

**Key Code Elements**:
- Direct implementation of mathematical formula
- Uses global memory for all operations
- Numerically stable softmax (max subtraction)

### 2. Tiled Attention

**Purpose**: Optimize memory access patterns using shared memory

**Approach**:
```
Load Q, K, V tiles into shared memory
Process attention in blocks
Reduce global memory traffic
```

**Optimizations**:
1. **Shared Memory Tiling**: Cache tiles in fast on-chip memory
2. **Coalesced Access**: Threads access consecutive memory locations
3. **Reduced Bandwidth**: ~3x fewer global memory accesses

**Design Decisions**:
- Tile size: 32x32 (one warp, good occupancy)
- Bank conflict avoidance: +1 padding in shared memory
- Block size tuned for occupancy vs shared memory usage

### 3. FlashAttention-Style

**Purpose**: Minimize HBM traffic through IO-aware algorithm

**Approach**:
```
for each Q tile:
    initialize output accumulator
    for each K,V tile:
        compute attention for tile
        update output incrementally
        maintain running statistics
```

**Key Innovation**: Never materialize full attention matrix

**Optimizations**:
1. **Online Softmax**: Incremental computation without full materialization
2. **Recomputation**: Trade compute for memory in backward pass
3. **Tiling Strategy**: Carefully orchestrated memory access
4. **Kernel Fusion**: Combine operations to reduce memory round-trips

## Memory Management

### Memory Hierarchy Utilization

```
Registers (fastest, smallest)
    ↓
Shared Memory (fast, 48-163 KB per SM)
    ↓
L2 Cache (medium, shared across SMs)
    ↓
Global Memory / HBM (slow, large)
```

### Optimization Strategy

1. **Maximize Register Usage**
   - Store frequently accessed scalars
   - Loop accumulators
   - Intermediate computation results

2. **Leverage Shared Memory**
   - Cache Q, K, V tiles
   - Avoid bank conflicts with padding
   - Size tiles to fit in available SMEM

3. **Minimize Global Memory Access**
   - Coalesced reads/writes (128-byte transactions)
   - Reduce round-trips through tiling
   - Reuse data across thread blocks

4. **Memory Coalescing**
   ```
   // Good: Consecutive threads access consecutive addresses
   float val = data[threadIdx.x + blockIdx.x * blockDim.x];
   
   // Bad: Strided or random access
   float val = data[threadIdx.x * stride];
   ```

## Performance Considerations

### Occupancy vs Resource Usage

**Tradeoff**: More threads per SM vs more resources per thread

**Strategy**:
- Target 50-75% occupancy (not always 100%)
- Balance shared memory, registers, and threads
- Use `--ptxas-options=-v` to check resource usage

### Warp-Level Programming

**Warp**: Group of 32 threads executing in lockstep

**Techniques**:
1. **Warp Shuffle**: Fast intra-warp communication
   ```cuda
   __shfl_down_sync(0xffffffff, val, offset)
   ```

2. **Warp Reductions**: Efficient parallel sum/max
   ```cuda
   float warpReduceSum(float val) {
       for (int offset = 16; offset > 0; offset /= 2)
           val += __shfl_down_sync(0xffffffff, val, offset);
       return val;
   }
   ```

3. **Branch Divergence**: Minimize within warps
   - Ensure uniform control flow when possible
   - Use predication instead of branches

### Numerical Stability

**Softmax Stability**:
```cuda
// Numerically unstable
exp(x) / sum(exp(x))

// Stable version
max_val = max(x)
exp(x - max_val) / sum(exp(x - max_val))
```

**Why It Matters**:
- Prevents overflow with large logits
- Essential for long sequences
- Maintains accuracy across precision levels

### Kernel Launch Configuration

**Grid Dimensions**: How to split work across blocks
```cuda
dim3 grid(num_queries / BLOCK_SIZE, num_heads, batch_size);
dim3 block(BLOCK_SIZE, BLOCK_SIZE);
```

**Considerations**:
1. Enough blocks to saturate GPU
2. Block size multiple of warp size (32)
3. Balance parallelism and resource usage

## Comparison with State-of-the-Art

### vs PyTorch Native Attention

**PyTorch**: Uses cuBLAS/cuDNN for matrix operations
- Highly optimized GEMM kernels
- General-purpose, not attention-specific

**Our Implementation**: Attention-specific optimizations
- Fused operations reduce memory traffic
- Custom tiling strategies
- Better for specific sequence lengths

### vs FlashAttention (Reference)

**Reference FlashAttention**: Production-grade implementation
- Supports FP16/BF16
- Multiple attention variants (causal, etc.)
- Highly tuned for all architectures

**Our Implementation**: Educational/demonstrative
- Shows core algorithmic ideas
- Simpler codebase for learning
- Validates understanding of concepts

## Future Optimizations

### Near-term
- [ ] FP16/BF16 support (2-4x speedup potential)
- [ ] Better tiling strategies for different sequence lengths
- [ ] Causal masking support
- [ ] Multi-query attention (MQA) and Grouped-query attention (GQA)

### Advanced
- [ ] Split-K optimization for very long sequences
- [ ] Persistent kernels to reduce launch overhead
- [ ] Cooperative groups for flexible parallelism
- [ ] CUTLASS templates for portable performance

## Lessons Learned

1. **Memory is the Bottleneck**: Compute is cheap, memory access is expensive
2. **Profile First**: Don't optimize blindly - measure everything
3. **Start Simple**: Get correctness first, then optimize
4. **Know Your Hardware**: Architecture details matter significantly
5. **Test Thoroughly**: Numerical bugs are subtle and dangerous

## References

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS: CUDA Templates for Linear Algebra Subroutines](https://github.com/NVIDIA/cutlass)

