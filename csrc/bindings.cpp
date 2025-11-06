#include <torch/extension.h>
#include "attention_kernels.h"

/*
 * PyTorch C++ Extension Bindings
 * 
 * Exposes CUDA kernels as Python-callable functions
 * Includes input validation and shape checking
 */

// ============================================================================
// Naive Attention Wrappers
// ============================================================================

torch::Tensor attention_naive_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    c10::optional<float> scale_opt
) {
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D (batch, num_heads, seq_len, head_dim)");
    TORCH_CHECK(K.dim() == 4, "K must be 4D");
    TORCH_CHECK(V.dim() == 4, "V must be 4D");
    TORCH_CHECK(Q.size(0) == K.size(0) && K.size(0) == V.size(0), "Batch size mismatch");
    TORCH_CHECK(Q.size(1) == K.size(1) && K.size(1) == V.size(1), "Num heads mismatch");
    TORCH_CHECK(Q.size(2) == K.size(2) && K.size(2) == V.size(2), "Seq len mismatch");
    TORCH_CHECK(Q.size(3) == K.size(3), "Head dim mismatch for Q and K");
    TORCH_CHECK(K.size(3) == V.size(3), "Head dim mismatch for K and V");
    
    const int head_dim = Q.size(3);
    const float scale = scale_opt.has_value() ? scale_opt.value() : 1.0f / sqrtf(head_dim);
    
    return attention_naive_forward_cuda(Q, K, V, scale);
}

std::vector<torch::Tensor> attention_naive_backward(
    torch::Tensor grad_output,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor attention,
    c10::optional<float> scale_opt
) {
    const int head_dim = Q.size(3);
    const float scale = scale_opt.has_value() ? scale_opt.value() : 1.0f / sqrtf(head_dim);
    
    return attention_naive_backward_cuda(grad_output, Q, K, V, attention, scale);
}

// ============================================================================
// Tiled Attention Wrappers
// ============================================================================

torch::Tensor attention_tiled_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    c10::optional<float> scale_opt
) {
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D (batch, num_heads, seq_len, head_dim)");
    TORCH_CHECK(K.dim() == 4, "K must be 4D");
    TORCH_CHECK(V.dim() == 4, "V must be 4D");
    TORCH_CHECK(Q.size(0) == K.size(0) && K.size(0) == V.size(0), "Batch size mismatch");
    TORCH_CHECK(Q.size(1) == K.size(1) && K.size(1) == V.size(1), "Num heads mismatch");
    TORCH_CHECK(Q.size(2) == K.size(2) && K.size(2) == V.size(2), "Seq len mismatch");
    TORCH_CHECK(Q.size(3) == K.size(3), "Head dim mismatch for Q and K");
    
    const int head_dim = Q.size(3);
    const float scale = scale_opt.has_value() ? scale_opt.value() : 1.0f / sqrtf(head_dim);
    
    return attention_tiled_forward_cuda(Q, K, V, scale);
}

std::vector<torch::Tensor> attention_tiled_backward(
    torch::Tensor grad_output,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor output,
    c10::optional<float> scale_opt
) {
    const int head_dim = Q.size(3);
    const float scale = scale_opt.has_value() ? scale_opt.value() : 1.0f / sqrtf(head_dim);
    
    return attention_tiled_backward_cuda(grad_output, Q, K, V, output, scale);
}

// ============================================================================
// FlashAttention Wrappers
// ============================================================================

torch::Tensor attention_flash_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    c10::optional<float> scale_opt
) {
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D (batch, num_heads, seq_len, head_dim)");
    TORCH_CHECK(K.dim() == 4, "K must be 4D");
    TORCH_CHECK(V.dim() == 4, "V must be 4D");
    TORCH_CHECK(Q.size(0) == K.size(0) && K.size(0) == V.size(0), "Batch size mismatch");
    TORCH_CHECK(Q.size(1) == K.size(1) && K.size(1) == V.size(1), "Num heads mismatch");
    TORCH_CHECK(Q.size(2) == K.size(2) && K.size(2) == V.size(2), "Seq len mismatch");
    TORCH_CHECK(Q.size(3) == K.size(3), "Head dim mismatch for Q and K");
    
    const int head_dim = Q.size(3);
    const float scale = scale_opt.has_value() ? scale_opt.value() : 1.0f / sqrtf(head_dim);
    
    return attention_flash_forward_cuda(Q, K, V, scale);
}

std::vector<torch::Tensor> attention_flash_backward(
    torch::Tensor grad_output,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor logsumexp,
    c10::optional<float> scale_opt
) {
    const int head_dim = Q.size(3);
    const float scale = scale_opt.has_value() ? scale_opt.value() : 1.0f / sqrtf(head_dim);
    
    return attention_flash_backward_cuda(grad_output, Q, K, V, logsumexp, scale);
}

// ============================================================================
// PyTorch Module Registration
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Custom CUDA kernels for optimized attention mechanisms";
    
    // Naive attention
    m.def("naive_forward", &attention_naive_forward, 
          "Naive attention forward pass",
          py::arg("Q"), py::arg("K"), py::arg("V"), 
          py::arg("scale") = py::none());
    
    m.def("naive_backward", &attention_naive_backward,
          "Naive attention backward pass",
          py::arg("grad_output"), py::arg("Q"), py::arg("K"), 
          py::arg("V"), py::arg("attention"),
          py::arg("scale") = py::none());
    
    // Tiled attention
    m.def("tiled_forward", &attention_tiled_forward,
          "Tiled attention forward pass with shared memory optimization",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("scale") = py::none());
    
    m.def("tiled_backward", &attention_tiled_backward,
          "Tiled attention backward pass",
          py::arg("grad_output"), py::arg("Q"), py::arg("K"),
          py::arg("V"), py::arg("output"),
          py::arg("scale") = py::none());
    
    // FlashAttention
    m.def("flash_forward", &attention_flash_forward,
          "FlashAttention forward pass with IO-aware design",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("scale") = py::none());
    
    m.def("flash_backward", &attention_flash_backward,
          "FlashAttention backward pass with recomputation",
          py::arg("grad_output"), py::arg("Q"), py::arg("K"),
          py::arg("V"), py::arg("logsumexp"),
          py::arg("scale") = py::none());
}

