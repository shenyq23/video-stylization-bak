#pragma once

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <string>

#include <c10/cuda/CUDAStream.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

namespace sige3d_kernels {

constexpr int kThreads = 256;

enum class ActivationType : int {
    Identity = 0,
    Silu = 1,
};

inline ActivationType get_activation_type(const std::string& activation_name) {
    if (activation_name == "identity") {
        return ActivationType::Identity;
    }
    if (activation_name == "silu" || activation_name == "swish") {
        return ActivationType::Silu;
    }
    TORCH_CHECK(false, "Unknown activation: ", activation_name, " (expected: identity/silu/swish)");
}

__device__ __forceinline__ float apply_activation(ActivationType activation_type, float x) {
    if (activation_type == ActivationType::Identity) {
        return x;
    }
    if (activation_type == ActivationType::Silu) {
        return x / (1.0f + expf(-x));
    }
    return x;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    float sum = 0.0f;
    if (wid == 0) {
        const int num_warps = (blockDim.x + 31) / 32;
        sum = (lane < num_warps) ? shared[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            shared[0] = sum;
        }
    }
    __syncthreads();
    return shared[0];
}


template <typename scalar_t>
__device__ __forceinline__ float load_broadcast_4d(
    const scalar_t* ptr,
    int B,
    int C,
    int H,
    int W,
    int b,
    int c,
    int h,
    int w) {
    if (ptr == nullptr) {
        return 0.0f;
    }
    int p = 0;
    if (W > 1) {
        p = w;
    }
    if (H > 1) {
        p += h * W;
    }
    if (C > 1) {
        p += c * H * W;
    }
    if (B > 1) {
        p += b * C * H * W;
    }
    return static_cast<float>(ptr[p]);
}

template <typename scalar_t>
__device__ __forceinline__ float load_broadcast_5d(
    const scalar_t* ptr,
    int B,
    int C,
    int T,
    int H,
    int W,
    int b,
    int c,
    int t,
    int h,
    int w) {
    if (ptr == nullptr) {
        return 0.0f;
    }
    int p = 0;
    if (W > 1) {
        p = w;
    }
    if (H > 1) {
        p += h * W;
    }
    if (T > 1) {
        p += t * H * W;
    }
    if (C > 1) {
        p += c * T * H * W;
    }
    if (B > 1) {
        p += b * C * T * H * W;
    }
    return static_cast<float>(ptr[p]);
}

} // namespace sige3d_kernels
