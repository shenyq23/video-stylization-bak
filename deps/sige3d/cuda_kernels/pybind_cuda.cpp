#include <torch/extension.h>

torch::Tensor gather2d_cuda(
    const torch::Tensor& x,
    int bSizeH,
    int bSizeW,
    const torch::Tensor& activeIndices);

torch::Tensor scatter2d_cuda(
    const torch::Tensor& x,
    const torch::Tensor& y,
    int offsetH,
    int offsetW,
    int strideH,
    int strideW,
    const torch::Tensor& activeIndices,
    const torch::optional<torch::Tensor>& residual);

torch::Tensor gather3d_cuda(
    const torch::Tensor& x,
    int bSizeH,
    int bSizeW,
    const torch::Tensor& activeIndices,
    const torch::optional<torch::Tensor>& gamma,
    const torch::optional<torch::Tensor>& bias,
    double eps,
    const std::string& activationName);

torch::Tensor scatter3d_cuda(
    const torch::Tensor& x,
    const torch::Tensor& y,
    int offsetH,
    int offsetW,
    int strideH,
    int strideW,
    const torch::Tensor& activeIndices,
    const torch::optional<torch::Tensor>& residual);

torch::Tensor scatter_with_block_residual3d_cuda(
    const torch::Tensor& x0,
    const torch::Tensor& y0,
    const torch::Tensor& x1,
    const torch::Tensor& y1,
    int offsetH,
    int offsetW,
    int strideH,
    int strideW,
    const torch::Tensor& activeIndices0,
    const torch::Tensor& activeIndices1);

torch::Tensor scatter_gather3d_cuda(
    const torch::Tensor& x,
    const torch::Tensor& y,
    int bSizeH,
    int bSizeW,
    const torch::Tensor& activeIndices,
    const torch::Tensor& scatterMap,
    const torch::optional<torch::Tensor>& gamma,
    const torch::optional<torch::Tensor>& bias,
    double eps,
    const std::string& activationName);

torch::Tensor get_scatter_map_cuda(
    int H,
    int W,
    int bSizeH,
    int bSizeW,
    int kSizeH,
    int kSizeW,
    int offsetH,
    int offsetW,
    int strideH,
    int strideW,
    const torch::Tensor& activeIndices);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SIGE3D fused CUDA kernels (2D+3D gather/scatter/scatter_gather)";

    m.def("gather2d", &gather2d_cuda, "Gather2d (CUDA)");
    m.def("scatter2d", &scatter2d_cuda, "Scatter2d (CUDA)");

    m.def(
        "gather3d",
        &gather3d_cuda,
        "Gather3d (CUDA), optional fused RMSNorm+activation when gamma/bias are provided");
    m.def("scatter3d", &scatter3d_cuda, "Scatter3d (CUDA)");
    m.def(
        "scatter_with_block_residual3d",
        &scatter_with_block_residual3d_cuda,
        "ScatterWithBlockResidual3d (CUDA)");
    m.def(
        "scatter_gather3d",
        &scatter_gather3d_cuda,
        "ScatterGather3d (CUDA), optional fused RMSNorm+activation when gamma/bias are provided");
    m.def("get_scatter_map", &get_scatter_map_cuda, "Get scatter map (CUDA)");
}

