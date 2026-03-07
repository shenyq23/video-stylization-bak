#include "sige3d_kernels_common.cuh"

using namespace sige3d_kernels;

namespace {

template <typename scalar_t>
__global__ void scatter_gather3d_rmsnorm_kernel(
    int64_t locations,
    int num_active,
    int B,
    int C,
    int T,
    int H,
    int W,
    int Rx,
    int Sx,
    int Ro,
    int So,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ output,
    const int* __restrict__ active_indices,
    const int* __restrict__ scatter_map,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ bias,
    float eps,
    ActivationType activation_type) {
    const int64_t loc = static_cast<int64_t>(blockIdx.x);
    if (loc >= locations) {
        return;
    }

    int64_t t = loc;
    const int intra_bw = static_cast<int>(t % So);
    t /= So;
    const int intra_bh = static_cast<int>(t % Ro);
    t /= Ro;
    const int tt = static_cast<int>(t % T);
    t /= T;
    const int ib = static_cast<int>(t % num_active);
    const int bb = static_cast<int>(t / num_active);

    const int bi_h = active_indices[ib * 2];
    const int hh = bi_h + intra_bh;
    const int bi_w = active_indices[ib * 2 + 1];
    const int ww = bi_w + intra_bw;

    if (hh < 0 || hh >= H || ww < 0 || ww >= W) {
        for (int cc = threadIdx.x; cc < C; cc += blockDim.x) {
            const int64_t out_index =
                (((((static_cast<int64_t>(bb) * num_active + ib) * C + cc) * T + tt) * Ro + intra_bh) * So) +
                intra_bw;
            output[out_index] = static_cast<scalar_t>(0);
        }
        return;
    }

    const int scatter_map_index = (hh * W + ww) * 3;
    const int bx = scatter_map[scatter_map_index];
    const int hx = scatter_map[scatter_map_index + 1];
    const int wx = scatter_map[scatter_map_index + 2];

    float sumsq = 0.0f;
    for (int cc = threadIdx.x; cc < C; cc += blockDim.x) {
        float v = 0.0f;
        if (bx >= 0) {
            const int64_t xp =
                (((static_cast<int64_t>(bb) * num_active + bx) * C + cc) * T + tt) * Rx * Sx +
                static_cast<int64_t>(hx) * Sx + wx;
            v = static_cast<float>(x[xp]);
        } else {
            const int64_t yp = (((static_cast<int64_t>(bb) * C + cc) * T + tt) * H + hh) * W + ww;
            v = static_cast<float>(y[yp]);
        }
        sumsq += v * v;
    }
    sumsq = block_reduce_sum(sumsq);

    const float sqrt_c = sqrtf(static_cast<float>(C));
    float denom = sqrtf(sumsq);
    if (denom < eps) {
        denom = eps;
    }
    const float inv = sqrt_c / denom;

    for (int cc = threadIdx.x; cc < C; cc += blockDim.x) {
        float v = 0.0f;
        if (bx >= 0) {
            const int64_t xp =
                (((static_cast<int64_t>(bb) * num_active + bx) * C + cc) * T + tt) * Rx * Sx +
                static_cast<int64_t>(hx) * Sx + wx;
            v = static_cast<float>(x[xp]);
        } else {
            const int64_t yp = (((static_cast<int64_t>(bb) * C + cc) * T + tt) * H + hh) * W + ww;
            v = static_cast<float>(y[yp]);
        }

        v *= inv;
        v *= static_cast<float>(gamma[cc]);
        if (bias != nullptr) {
            v += static_cast<float>(bias[cc]);
        }
        v = apply_activation(activation_type, v);

        const int64_t out_index =
            (((((static_cast<int64_t>(bb) * num_active + ib) * C + cc) * T + tt) * Ro + intra_bh) * So) +
            intra_bw;
        output[out_index] = static_cast<scalar_t>(v);
    }
}

__global__ void get_scatter_map_kernel(
    int64_t total,
    int H,
    int W,
    int R,
    int S,
    int offsetH,
    int offsetW,
    int strideH,
    int strideW,
    int* __restrict__ output,
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
    const int ib = static_cast<int>(t);

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

    const int p = 3 * (hh * W + ww);
    output[p] = ib;
    output[p + 1] = intra_bh;
    output[p + 2] = intra_bw;
}

} // namespace

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
    const std::string& activationName) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    CHECK_INPUT(activeIndices);
    CHECK_INPUT(scatterMap);
    TORCH_CHECK(activeIndices.scalar_type() == at::kInt, "activeIndices must be int32");
    TORCH_CHECK(scatterMap.scalar_type() == at::kInt, "scatterMap must be int32");
    TORCH_CHECK(x.dim() == 5, "scatter_gather3d expects x with shape [B*numActive,C,T,Rx,Sx]");
    TORCH_CHECK(y.dim() == 5, "scatter_gather3d expects y with shape [B,C,T,H,W]");
    TORCH_CHECK(y.size(1) == x.size(1), "x/y must have the same channel dim");
    TORCH_CHECK(y.size(2) == x.size(2), "x/y must have the same T dim");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "x/y must have the same dtype");

    const int Ro = bSizeH;
    const int So = bSizeW;
    const int Rx = static_cast<int>(x.size(3));
    const int Sx = static_cast<int>(x.size(4));
    const int B = static_cast<int>(y.size(0));
    const int C = static_cast<int>(y.size(1));
    const int T = static_cast<int>(y.size(2));
    const int H = static_cast<int>(y.size(3));
    const int W = static_cast<int>(y.size(4));
    const int num_active = static_cast<int>(activeIndices.size(0));

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device()).requires_grad(false);
    auto output = torch::empty({B * num_active, C, T, Ro, So}, options);

    const ActivationType activation_type = get_activation_type(activationName);

    const int64_t total = output.numel();
    if (total == 0) {
        return output;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "scatter_gather3d_cuda", [&] {
        CHECK_INPUT(gamma.value());
        TORCH_CHECK(gamma.value().numel() >= C, "gamma must have at least C elements");
        TORCH_CHECK(gamma.value().scalar_type() == x.scalar_type(), "gamma dtype must match x");
        const scalar_t* gamma_ptr = gamma.value().data_ptr<scalar_t>();

        const scalar_t* bias_ptr = nullptr;
        if (bias.has_value()) {
            CHECK_INPUT(bias.value());
            TORCH_CHECK(bias.value().numel() >= C, "bias must have at least C elements");
            TORCH_CHECK(bias.value().scalar_type() == x.scalar_type(), "bias dtype must match x");
            bias_ptr = bias.value().data_ptr<scalar_t>();
        }

        const int64_t locations = static_cast<int64_t>(B) * num_active * T * Ro * So;
        const dim3 blocks(static_cast<unsigned int>(locations), 1);
        const auto stream = c10::cuda::getCurrentCUDAStream();
        scatter_gather3d_rmsnorm_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
            locations,
            num_active,
            B,
            C,
            T,
            H,
            W,
            Rx,
            Sx,
            Ro,
            So,
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            activeIndices.data_ptr<int>(),
            scatterMap.data_ptr<int>(),
            gamma_ptr,
            bias_ptr,
            static_cast<float>(eps),
            activation_type);
    });

    return output;
}

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
    const torch::Tensor& activeIndices) {
    CHECK_INPUT(activeIndices);
    TORCH_CHECK(activeIndices.scalar_type() == at::kInt, "activeIndices must be int32");

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(activeIndices.device()).requires_grad(false);
    auto scatter_map = torch::full({H, W, 3}, -1, options);

    const int R = (bSizeH - kSizeH) / strideH + 1;
    const int S = (bSizeW - kSizeW) / strideW + 1;
    const int num_active = static_cast<int>(activeIndices.size(0));
    const int64_t total = static_cast<int64_t>(num_active) * R * S;
    if (total == 0) {
        return scatter_map;
    }
    const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads), 1);
    const auto stream = c10::cuda::getCurrentCUDAStream();

    get_scatter_map_kernel<<<blocks, kThreads, 0, stream>>>(
        total,
        H,
        W,
        R,
        S,
        offsetH,
        offsetW,
        strideH,
        strideW,
        scatter_map.data_ptr<int>(),
        activeIndices.data_ptr<int>());

    return scatter_map;
}
