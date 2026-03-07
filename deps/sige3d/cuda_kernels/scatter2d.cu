#include "sige3d_kernels_common.cuh"

using namespace sige3d_kernels;

namespace {

template <typename scalar_t>
__global__ void scatter2d_kernel(
    int64_t total,
    int num_active,
    int B,
    int C,
    int H,
    int W,
    int R,
    int S,
    int offsetH,
    int offsetW,
    int strideH,
    int strideW,
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int* __restrict__ active_indices,
    const scalar_t* __restrict__ residual,
    int residualB,
    int residualC,
    int residualH,
    int residualW) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }

    int64_t t = index;
    const int intra_bw = static_cast<int>(t % S);
    t /= S;
    const int intra_bh = static_cast<int>(t % R);
    t /= R;
    const int cc = static_cast<int>(t % C);
    t /= C;
    const int ib = static_cast<int>(t % num_active);
    const int bb = static_cast<int>(t / num_active);

    const int bi_h = (offsetH + active_indices[ib * 2]) / strideH;
    const int hh = bi_h + intra_bh;
    if (hh < 0 || hh >= H) {
        return;
    }
    const int bi_w = (offsetW + active_indices[ib * 2 + 1]) / strideW;
    const int ww = bi_w + intra_bw;
    if (ww < 0 || ww >= W) {
        return;
    }

    const int64_t p = (static_cast<int64_t>(bb) * C * H * W) + (static_cast<int64_t>(cc) * H * W) +
        (static_cast<int64_t>(hh) * W) + ww;

    float z = static_cast<float>(x[index]);
    z += load_broadcast_4d(residual, residualB, residualC, residualH, residualW, bb, cc, hh, ww);
    y[p] = static_cast<scalar_t>(z);
}

} // namespace

torch::Tensor scatter2d_cuda(
    const torch::Tensor& x,
    const torch::Tensor& y,
    int offsetH,
    int offsetW,
    int strideH,
    int strideW,
    const torch::Tensor& activeIndices,
    const torch::optional<torch::Tensor>& residual) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    CHECK_INPUT(activeIndices);
    TORCH_CHECK(activeIndices.scalar_type() == at::kInt, "activeIndices must be int32");
    TORCH_CHECK(x.dim() == 4, "scatter2d expects x with shape [B*numActive,C,R,S]");
    TORCH_CHECK(y.dim() == 4, "scatter2d expects y with shape [B,C,H,W]");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "x/y must have the same dtype");

    const int num_active = static_cast<int>(activeIndices.size(0));
    const int C = static_cast<int>(x.size(1));
    const int R = static_cast<int>(x.size(2));
    const int S = static_cast<int>(x.size(3));
    const int B = static_cast<int>(y.size(0));
    const int H = static_cast<int>(y.size(2));
    const int W = static_cast<int>(y.size(3));

    int residualB = 0, residualC = 0, residualH = 0, residualW = 0;
    if (residual.has_value()) {
        CHECK_INPUT(residual.value());
        TORCH_CHECK(residual.value().dim() == 4, "residual must have dim=4");
        TORCH_CHECK(residual.value().scalar_type() == y.scalar_type(), "residual dtype must match y");
        residualB = static_cast<int>(residual.value().size(0));
        residualC = static_cast<int>(residual.value().size(1));
        residualH = static_cast<int>(residual.value().size(2));
        residualW = static_cast<int>(residual.value().size(3));
    }

    const int64_t total = x.numel();
    if (total == 0) {
        return y;
    }
    const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads), 1);
    const auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "scatter2d_cuda", [&] {
        const scalar_t* residual_data = residual.has_value() ? residual.value().data_ptr<scalar_t>() : nullptr;
        scatter2d_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
            total,
            num_active,
            B,
            C,
            H,
            W,
            R,
            S,
            offsetH,
            offsetW,
            strideH,
            strideW,
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            activeIndices.data_ptr<int>(),
            residual_data,
            residualB,
            residualC,
            residualH,
            residualW);
    });

    return y;
}
