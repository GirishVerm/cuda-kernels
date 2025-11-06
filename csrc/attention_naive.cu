#include "attention_kernels.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_SIZE = 1024;

// Warp-level reduction for sum
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level reduction for max
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

/*
 * Naive Attention Forward Kernel
 * 
 * Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
 * 
 * This is the baseline implementation that prioritizes correctness over performance.
 * It materializes the full attention matrix in global memory.
 * 
 * Grid: (num_heads, batch_size)
 * Block: (min(seq_len, MAX_BLOCK_SIZE))
 * 
 * Memory complexity: O(seq_len^2) per attention head
 */
__global__ void attention_naive_forward_kernel(
    const float* __restrict__ Q,    // [batch, num_heads, seq_len, head_dim]
    const float* __restrict__ K,    // [batch, num_heads, seq_len, head_dim]
    const float* __restrict__ V,    // [batch, num_heads, seq_len, head_dim]
    float* __restrict__ output,     // [batch, num_heads, seq_len, head_dim]
    float* __restrict__ attention,  // [batch, num_heads, seq_len, seq_len]
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Base pointers for this batch and head
    const int batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const float* Q_bh = Q + batch_head_offset;
    const float* K_bh = K + batch_head_offset;
    const float* V_bh = V + batch_head_offset;
    
    const int attention_offset = (batch_idx * num_heads + head_idx) * seq_len * seq_len;
    float* attention_bh = attention + attention_offset;
    float* output_bh = output + batch_head_offset;
    
    // Each thread processes one query position
    for (int q_idx = tid; q_idx < seq_len; q_idx += blockDim.x) {
        // Step 1: Compute Q @ K^T for this query
        float qk_scores[1024];  // Assuming max seq_len of 1024 for simplicity
        float max_score = -INFINITY;
        
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            float score = 0.0f;
            // Dot product: Q[q_idx] · K[k_idx]
            for (int d = 0; d < head_dim; d++) {
                score += Q_bh[q_idx * head_dim + d] * K_bh[k_idx * head_dim + d];
            }
            score *= scale;
            qk_scores[k_idx] = score;
            max_score = fmaxf(max_score, score);
        }
        
        // Step 2: Compute softmax (numerically stable)
        float sum_exp = 0.0f;
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            float exp_val = expf(qk_scores[k_idx] - max_score);
            qk_scores[k_idx] = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            qk_scores[k_idx] /= sum_exp;
            // Store attention weights for backward pass
            attention_bh[q_idx * seq_len + k_idx] = qk_scores[k_idx];
        }
        
        // Step 3: Compute weighted sum of values
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int v_idx = 0; v_idx < seq_len; v_idx++) {
                sum += qk_scores[v_idx] * V_bh[v_idx * head_dim + d];
            }
            output_bh[q_idx * head_dim + d] = sum;
        }
    }
}

/*
 * Naive Attention Backward Kernel
 * 
 * Computes gradients for Q, K, V given gradient of output
 * Uses chain rule through softmax and matrix multiplications
 */
__global__ void attention_naive_backward_kernel(
    const float* __restrict__ grad_output,  // [batch, num_heads, seq_len, head_dim]
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ attention,    // [batch, num_heads, seq_len, seq_len]
    float* __restrict__ grad_Q,
    float* __restrict__ grad_K,
    float* __restrict__ grad_V,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const int batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const float* grad_out_bh = grad_output + batch_head_offset;
    const float* V_bh = V + batch_head_offset;
    const float* K_bh = K + batch_head_offset;
    const float* Q_bh = Q + batch_head_offset;
    
    const int attention_offset = (batch_idx * num_heads + head_idx) * seq_len * seq_len;
    const float* attention_bh = attention + attention_offset;
    
    float* grad_Q_bh = grad_Q + batch_head_offset;
    float* grad_K_bh = grad_K + batch_head_offset;
    float* grad_V_bh = grad_V + batch_head_offset;
    
    // Each thread processes multiple positions
    for (int q_idx = tid; q_idx < seq_len; q_idx += blockDim.x) {
        // Gradient through attention @ V
        float grad_attention[1024];
        
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            float grad_att = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                grad_att += grad_out_bh[q_idx * head_dim + d] * V_bh[k_idx * head_dim + d];
            }
            grad_attention[k_idx] = grad_att;
        }
        
        // Gradient through softmax
        float sum_grad_att = 0.0f;
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            sum_grad_att += grad_attention[k_idx] * attention_bh[q_idx * seq_len + k_idx];
        }
        
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            float att_weight = attention_bh[q_idx * seq_len + k_idx];
            float grad_qk = att_weight * (grad_attention[k_idx] - sum_grad_att);
            grad_qk *= scale;
            
            // Gradient w.r.t Q
            for (int d = 0; d < head_dim; d++) {
                atomicAdd(&grad_Q_bh[q_idx * head_dim + d], grad_qk * K_bh[k_idx * head_dim + d]);
            }
            
            // Gradient w.r.t K
            for (int d = 0; d < head_dim; d++) {
                atomicAdd(&grad_K_bh[k_idx * head_dim + d], grad_qk * Q_bh[q_idx * head_dim + d]);
            }
        }
        
        // Gradient w.r.t V
        for (int v_idx = 0; v_idx < seq_len; v_idx++) {
            float att_weight = attention_bh[q_idx * seq_len + v_idx];
            for (int d = 0; d < head_dim; d++) {
                atomicAdd(&grad_V_bh[v_idx * head_dim + d], 
                         att_weight * grad_out_bh[q_idx * head_dim + d]);
            }
        }
    }
}

// Host function for forward pass
torch::Tensor attention_naive_forward_cuda(
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
    auto attention = torch::empty({batch_size, num_heads, seq_len, seq_len}, Q.options());
    
    dim3 grid(num_heads, batch_size);
    dim3 block(std::min(seq_len, MAX_BLOCK_SIZE));
    
    attention_naive_forward_kernel<<<grid, block>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        attention.data_ptr<float>(),
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

// Host function for backward pass
std::vector<torch::Tensor> attention_naive_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor attention,
    float scale
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    CHECK_INPUT(attention);
    
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);
    
    auto grad_Q = torch::zeros_like(Q);
    auto grad_K = torch::zeros_like(K);
    auto grad_V = torch::zeros_like(V);
    
    dim3 grid(num_heads, batch_size);
    dim3 block(std::min(seq_len, MAX_BLOCK_SIZE));
    
    attention_naive_backward_kernel<<<grid, block>>>(
        grad_output.data_ptr<float>(),
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        attention.data_ptr<float>(),
        grad_Q.data_ptr<float>(),
        grad_K.data_ptr<float>(),
        grad_V.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        scale
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return {grad_Q, grad_K, grad_V};
}

