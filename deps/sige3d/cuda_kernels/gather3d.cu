#include "sige3d_kernels_common.cuh"

using namespace sige3d_kernels;

namespace {

template <typename scalar_t>
__global__ void gather3d_plain_kernel(
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
        output[index] = static_cast<scalar_t>(0);
        return;
    }
    const int bi_w = active_indices[ib * 2 + 1];
    const int ww = bi_w + intra_bw;
    if (ww < 0 || ww >= W) {
        output[index] = static_cast<scalar_t>(0);
        return;
    }

    const int64_t p = (((static_cast<int64_t>(bb) * C + cc) * T + tt) * H + hh) * W + ww;
    output[index] = x[p];
}

template <typename scalar_t>
__global__ void gather3d_rmsnorm_kernel(
    int64_t locations,
    int num_active,
    int B,
    int C,
    int T,
    int H,
    int W,
    int R,
    int S,
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ output,
    const int* __restrict__ active_indices,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ bias,
    float eps,
    ActivationType activation_type) {
    const int64_t loc = static_cast<int64_t>(blockIdx.x);
    if (loc >= locations) {
        return;
    }

    int64_t t = loc;
    const int intra_bw = static_cast<int>(t % S);
    t /= S;
    const int intra_bh = static_cast<int>(t % R);
    t /= R;
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
            const int64_t out_index = (((((static_cast<int64_t>(bb) * num_active + ib) * C + cc) * T + tt) * R +
                                        intra_bh) *
                                           S) +
                intra_bw;
            output[out_index] = static_cast<scalar_t>(0);
        }
        return;
    }

    float sumsq = 0.0f;
    for (int cc = threadIdx.x; cc < C; cc += blockDim.x) {
        const int64_t p = (((static_cast<int64_t>(bb) * C + cc) * T + tt) * H + hh) * W + ww;
        const float v = static_cast<float>(x[p]);
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
        const int64_t p = (((static_cast<int64_t>(bb) * C + cc) * T + tt) * H + hh) * W + ww;
        float v = static_cast<float>(x[p]);
        v *= inv;
        v *= static_cast<float>(gamma[cc]);
        if (bias != nullptr) {
            v += static_cast<float>(bias[cc]);
        }
        v = apply_activation(activation_type, v);

        const int64_t out_index = (((((static_cast<int64_t>(bb) * num_active + ib) * C + cc) * T + tt) * R +
                                    intra_bh) *
                                       S) +
            intra_bw;
        output[out_index] = static_cast<scalar_t>(v);
    }
}

} // namespace

torch::Tensor gather3d_cuda(
    const torch::Tensor& x,
    int bSizeH,
    int bSizeW,
    const torch::Tensor& activeIndices,
    const torch::optional<torch::Tensor>& gamma,
    const torch::optional<torch::Tensor>& bias,
    double eps,
    const std::string& activationName) {
    CHECK_INPUT(x);
    CHECK_INPUT(activeIndices);
    TORCH_CHECK(activeIndices.scalar_type() == at::kInt, "activeIndices must be int32");
    TORCH_CHECK(x.dim() == 5, "gather3d expects x with shape [B,C,T,H,W]");

    const int B = static_cast<int>(x.size(0));
    const int C = static_cast<int>(x.size(1));
    const int T = static_cast<int>(x.size(2));
    const int H = static_cast<int>(x.size(3));
    const int W = static_cast<int>(x.size(4));
    const int R = bSizeH;
    const int S = bSizeW;
    const int num_active = static_cast<int>(activeIndices.size(0));

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device()).requires_grad(false);
    auto output = torch::empty({B * num_active, C, T, R, S}, options);

    const int64_t total = output.numel();
    if (total == 0) {
        return output;
    }

    const bool do_rmsnorm = gamma.has_value();
    const ActivationType activation_type = get_activation_type(activationName);
    const auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "gather3d_cuda", [&] {
        if (!do_rmsnorm) {
            const dim3 blocks(static_cast<unsigned int>((total + kThreads - 1) / kThreads), 1);
            gather3d_plain_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
                total,
                num_active,
                B,
                C,
                T,
                H,
                W,
                R,
                S,
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                activeIndices.data_ptr<int>());
            return;
        }

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

        const int64_t locations = static_cast<int64_t>(B) * num_active * T * R * S;
        const dim3 blocks(static_cast<unsigned int>(locations), 1);
        
        int threads = kThreads;              // 默认
        // if (C <= 96) threads = 128;
        // else if (C <= 192) threads = 256;   // 可再试 192
        // else threads = 512;                 // 384 -> 256 通常最好


        gather3d_rmsnorm_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            locations,
            num_active,
            B,
            C,
            T,
            H,
            W,
            R,
            S,
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            activeIndices.data_ptr<int>(),
            gamma_ptr,
            bias_ptr,
            static_cast<float>(eps),
            activation_type);
    });

    return output;
}
