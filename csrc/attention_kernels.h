#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Forward declarations for attention kernel implementations

// Naive attention: Direct implementation for baseline
torch::Tensor attention_naive_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale
);

std::vector<torch::Tensor> attention_naive_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor output,
    float scale
);

// Tiled attention: Shared memory optimization
torch::Tensor attention_tiled_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale
);

std::vector<torch::Tensor> attention_tiled_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor output,
    float scale
);

// FlashAttention-style: IO-aware implementation
torch::Tensor attention_flash_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale
);

std::vector<torch::Tensor> attention_flash_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor logsumexp,
    float scale
);

// Utility macros for CUDA error checking
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while(0)

