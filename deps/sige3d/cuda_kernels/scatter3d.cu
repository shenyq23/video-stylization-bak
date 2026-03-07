#include "sige3d_kernels_common.cuh"

using namespace sige3d_kernels;

namespace {

template <typename scalar_t>
__global__ void scatter3d_kernel(
    int64_t total,
    int num_active,
    int B,
    int C,
    int T,
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
    int residualT,
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
    const int tt = static_cast<int>(t % T);
    t /= T;
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

    const int64_t p = (((static_cast<int64_t>(bb) * C + cc) * T + tt) * H + hh) * W + ww;

    float z = static_cast<float>(x[index]);
    z += load_broadcast_5d(residual, residualB, residualC, residualT, residualH, residualW, bb, cc, tt, hh, ww);
    y[p] = static_cast<scalar_t>(z);
}

template <typename scalar_t>
__global__ void calibrate_residual3d_kernel(
    int64_t total,
    int num_active,
    int B,
    int C,
    int T,
    int H,
    int W,
    int R,
    int S,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y_residual,
    scalar_t* __restrict__ output,
    const int* __restrict__ active_indices) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }

    int64_t t = index;
    const int intra_bw = static_cast<int>(t % S);
    t /= S;
    const int intra_bh = static_cast<int>(t % R);
    t /= R;
    const int tt = static_cast<int>(t % T);
    t /= T;
    const int cc = static_cast<int>(t % C);
    t /= C;
    const int ib = static_cast<int>(t % num_active);
    const int bb = static_cast<int>(t / num_active);

    const int bi_h = active_indices[ib * 2];
    const int hh = bi_h + intra_bh;
    if (hh < 0 || hh >= H) {
        return;
    }
    const int bi_w = active_indices[ib * 2 + 1];
    const int ww = bi_w + intra_bw;
    if (ww < 0 || ww >= W) {
        return;
    }

    const int64_t p = (((static_cast<int64_t>(bb) * C + cc) * T + tt) * H + hh) * W + ww;
    const float delta = static_cast<float>(x[index]) - static_cast<float>(y_residual[p]);
    output[p] = static_cast<scalar_t>(static_cast<float>(output[p]) + delta);
}

} // namespace

torch::Tensor scatter3d_cuda(
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
    TORCH_CHECK(x.dim() == 5, "scatter3d expects x with shape [B*numActive,C,T,R,S]");
    TORCH_CHECK(y.dim() == 5, "scatter3d expects y with shape [B,C,T,H,W]");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "x/y must have the same dtype");

    const int num_active = static_cast<int>(activeIndices.size(0));
    const int C = static_cast<int>(x.size(1));
    const int T = static_cast<int>(x.size(2));
    const int R = static_cast<int>(x.size(3));
    const int S = static_cast<int>(x.size(4));
    const int B = static_cast<int>(y.size(0));
    const int H = static_cast<int>(y.size(3));
    const int W = static_cast<int>(y.size(4));

    int residualB = 0, residualC = 0, residualT = 0, residualH = 0, residualW = 0;
    if (residual.has_value()) {
        CHECK_INPUT(residual.value());
        TORCH_CHECK(residual.value().dim() == 5, "residual must have dim=5");
        TORCH_CHECK(residual.value().scalar_type() == y.scalar_type(), "residual dtype must match y");
        residualB = static_cast<int>(residual.value().size(0));
        residualC = static_cast<int>(residual.value().size(1));
        residualT = static_cast<int>(residual.value().size(2));
        residualH = static_cast<int>(residual.value().size(3));
        residualW = static_cast<int>(residual.value().size(4));
    }

    const int64_t total = x.numel();
    if (total == 0) {
        return y;
    }
    const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads), 1);
    const auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "scatter3d_cuda", [&] {
        const scalar_t* residual_data = residual.has_value() ? residual.value().data_ptr<scalar_t>() : nullptr;
        scatter3d_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
            total,
            num_active,
            B,
            C,
            T,
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
            residualT,
            residualH,
            residualW);
    });

    return y;
}

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
    const torch::Tensor& activeIndices1) {
    auto out = scatter3d_cuda(x0, y0, offsetH, offsetW, strideH, strideW, activeIndices0, y1);

    CHECK_INPUT(x1);
    CHECK_INPUT(y1);
    CHECK_INPUT(activeIndices1);
    TORCH_CHECK(activeIndices1.scalar_type() == at::kInt, "activeIndices1 must be int32");
    TORCH_CHECK(x1.scalar_type() == y1.scalar_type(), "x1/y1 must have the same dtype");
    TORCH_CHECK(x1.dim() == 5, "x1 must have dim=5");
    TORCH_CHECK(y1.dim() == 5, "y1 must have dim=5");

    const int num_active = static_cast<int>(activeIndices1.size(0));
    const int C = static_cast<int>(x1.size(1));
    const int T = static_cast<int>(x1.size(2));
    const int R = static_cast<int>(x1.size(3));
    const int S = static_cast<int>(x1.size(4));
    const int B = static_cast<int>(y1.size(0));
    const int H = static_cast<int>(y1.size(3));
    const int W = static_cast<int>(y1.size(4));

    const int64_t total = x1.numel();
    if (total == 0) {
        return out;
    }

    const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads), 1);
    const auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x1.scalar_type(), "scatter_with_block_residual3d_cuda", [&] {
        calibrate_residual3d_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
            total,
            num_active,
            B,
            C,
            T,
            H,
            W,
            R,
            S,
            x1.data_ptr<scalar_t>(),
            y1.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            activeIndices1.data_ptr<int>());
    });

    return out;
}
