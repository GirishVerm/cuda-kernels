#include "attention_kernels.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

constexpr int BLOCK_M = 64;  // Query tile size
constexpr int BLOCK_N = 64;  // Key/Value tile size
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;

/*
 * FlashAttention-style Forward Kernel
 * 
 * Key Optimizations:
 * 1. IO-aware design: Minimizes HBM reads/writes
 * 2. Online softmax: Avoids materializing full attention matrix
 * 3. Tiling with incremental updates to output
 * 4. Fused operations: Combines softmax + matmul
 * 5. Uses high-bandwidth shared memory efficiently
 * 
 * Algorithm:
 * - Process attention in tiles without materializing full N×N matrix
 * - Maintain running statistics (max, sum) for numerical stability
 * - Incrementally update output as we process K,V tiles
 * - Memory complexity: O(N) instead of O(N²)
 * 
 * Reference: https://arxiv.org/abs/2205.14135
 */

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void attention_flash_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    float* __restrict__ logsumexp,  // For backward pass
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    // Shared memory for Q, K, V tiles
    __shared__ float Q_smem[BLOCK_M][32];  // Assuming head_dim <= 32 for simplicity
    __shared__ float K_smem[BLOCK_N][32];
    __shared__ float V_smem[BLOCK_N][32];
    __shared__ float S_smem[BLOCK_M][BLOCK_N];  // Attention scores tile
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len);
    const int num_q = q_end - q_start;
    
    if (num_q <= 0) return;
    
    const int batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const float* Q_bh = Q + batch_head_offset;
    const float* K_bh = K + batch_head_offset;
    const float* V_bh = V + batch_head_offset;
    float* output_bh = output + batch_head_offset;
    float* logsumexp_bh = logsumexp + (batch_idx * num_heads + head_idx) * seq_len;
    
    // Per-thread accumulators
    float out_acc[4] = {0.0f};  // Assuming head_dim = 64, each thread handles 4 elements
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    
    // Process each query in this block
    const int q_local = tid / (head_dim / 4);  // Which query this thread works on
    const int d_local = (tid % (head_dim / 4)) * 4;  // Which dimension
    
    if (q_local >= num_q) return;
    const int q_idx = q_start + q_local;
    
    // Load Q tile into shared memory
    if (tid < num_q * head_dim) {
        const int q_offset = tid / head_dim;
        const int d = tid % head_dim;
        if (q_start + q_offset < seq_len && d < head_dim) {
            Q_smem[q_offset][d] = Q_bh[(q_start + q_offset) * head_dim + d];
        }
    }
    __syncthreads();
    
    // Iterate over K,V tiles
    for (int k_block = 0; k_block < (seq_len + BLOCK_N - 1) / BLOCK_N; k_block++) {
        const int k_start = k_block * BLOCK_N;
        const int k_end = min(k_start + BLOCK_N, seq_len);
        const int num_k = k_end - k_start;
        
        if (num_k <= 0) break;
        
        // Load K and V tiles cooperatively
        if (tid < num_k * head_dim) {
            const int k_offset = tid / head_dim;
            const int d = tid % head_dim;
            if (k_start + k_offset < seq_len && d < head_dim) {
                K_smem[k_offset][d] = K_bh[(k_start + k_offset) * head_dim + d];
                V_smem[k_offset][d] = V_bh[(k_start + k_offset) * head_dim + d];
            }
        }
        __syncthreads();
        
        // Compute Q @ K^T for this tile
        if (q_local < num_q) {
            for (int k_local = 0; k_local < num_k; k_local++) {
                float score = 0.0f;
                
                // Vectorized dot product
                #pragma unroll
                for (int d = 0; d < head_dim; d++) {
                    score += Q_smem[q_local][d] * K_smem[k_local][d];
                }
                
                score *= scale;
                S_smem[q_local][k_local] = score;
                
                // Track max for numerical stability
                max_score = fmaxf(max_score, score);
            }
        }
        __syncthreads();
        
        // Compute new max across all previous tiles
        float new_max = warpReduceMax(max_score);
        if (lane_id == 0) {
            // Broadcast within warp
            max_score = new_max;
        }
        max_score = __shfl_sync(0xffffffff, max_score, 0);
        
        // Online softmax update
        // Rescale previous outputs and sum_exp with new max
        float exp_diff = expf(max_score - new_max);
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            out_acc[i] *= exp_diff;
        }
        sum_exp *= exp_diff;
        max_score = new_max;
        
        // Compute softmax for current tile and update output
        if (q_local < num_q) {
            for (int k_local = 0; k_local < num_k; k_local++) {
                float score = S_smem[q_local][k_local];
                float exp_score = expf(score - max_score);
                sum_exp += exp_score;
                
                // Accumulate V weighted by attention
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (d_local + i < head_dim) {
                        out_acc[i] += exp_score * V_smem[k_local][d_local + i];
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // Final normalization
    if (q_local < num_q) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (d_local + i < head_dim) {
                out_acc[i] /= sum_exp;
                output_bh[q_idx * head_dim + d_local + i] = out_acc[i];
            }
        }
        
        // Store logsumexp for backward pass
        if (d_local == 0) {
            logsumexp_bh[q_idx] = logf(sum_exp) + max_score;
        }
    }
}

/*
 * FlashAttention Backward Kernel (Simplified)
 * 
 * In FlashAttention, backward pass recomputes attention scores on-the-fly
 * instead of storing them, trading compute for memory
 */
__global__ void attention_flash_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ logsumexp,
    float* __restrict__ grad_Q,
    float* __restrict__ grad_K,
    float* __restrict__ grad_V,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    // Simplified backward implementation
    // Full implementation would use recomputation and tiling
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * num_heads * seq_len * head_dim;
    
    if (idx < total_elements) {
        // Placeholder: In production, implement tiled backward pass
        // with attention recomputation similar to forward pass
    }
}

torch::Tensor attention_flash_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale
) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);
    
    auto output = torch::empty_like(Q);
    auto logsumexp = torch::empty({batch_size, num_heads, seq_len}, Q.options());
    
    const int num_q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
    
    dim3 grid(num_q_blocks, num_heads, batch_size);
    
    // Each block handles BLOCK_M queries
    // Threads per block should handle all queries and dimensions
    const int threads_per_block = min(256, BLOCK_M * head_dim / 4);
    dim3 block(threads_per_block);
    
    attention_flash_forward_kernel<<<grid, block>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        logsumexp.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        scale
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return output;
}

std::vector<torch::Tensor> attention_flash_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor logsumexp,
    float scale
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    CHECK_INPUT(logsumexp);
    
    auto grad_Q = torch::zeros_like(Q);
    auto grad_K = torch::zeros_like(K);
    auto grad_V = torch::zeros_like(V);
    
    // For demonstration, returning zero gradients
    // Full implementation would call backward kernel
    
    return {grad_Q, grad_K, grad_V};
}

