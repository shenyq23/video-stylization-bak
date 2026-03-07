#include "sige3d_kernels_common.cuh"
using namespace sige3d_kernels;

namespace {

template <typename scalar_t, typename index_t, typename p_index_t>
__global__ void gather2d_cuda_kernel(
    index_t total,
    int numActive,
    int B, int C, int H, int W,
    int R, int S,
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ output,
    const int* __restrict__ activeIndices
) {
    index_t index = static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x)
                  + static_cast<index_t>(threadIdx.x);
    if (index >= total) return;

    index_t t = index;
    const int intraBw = static_cast<int>(t % S); 
    t /= S;
    const int intraBh = static_cast<int>(t % R); 
    t /= R;
    const int cc      = static_cast<int>(t % C); 
    t /= C;
    const int ib      = static_cast<int>(t % numActive);
    const int bb      = static_cast<int>(t / numActive);

    const int biH = activeIndices[(ib << 1)];
    const int hh  = biH + intraBh;
    if (hh < 0 || hh >= H) {
        output[index] = static_cast<scalar_t>(0);
        return;
    }

    const int biW = activeIndices[(ib << 1) | 1];
    const int ww  = biW + intraBw;
    if (ww < 0 || ww >= W) {
        output[index] = static_cast<scalar_t>(0);
        return;
    }

    const p_index_t p =
        static_cast<p_index_t>(bb) * static_cast<p_index_t>(C) * static_cast<p_index_t>(H) * static_cast<p_index_t>(W) +
        static_cast<p_index_t>(cc) * static_cast<p_index_t>(H) * static_cast<p_index_t>(W) +
        static_cast<p_index_t>(hh) * static_cast<p_index_t>(W) +
        static_cast<p_index_t>(ww);

    output[index] = x[p];
}

} // anonymous namespace

torch::Tensor gather2d_cuda(
    const torch::Tensor& x,
    int bSizeH,
    int bSizeW,
    const torch::Tensor& activeIndices
) {
    // CHECK_INPUT(x);
    // TORCH_CHECK(x.dim() == 4, "gather2d_cuda: x must be 4D [B,C,H,W]");
    // CHECK_INPUT(activeIndices);
    // TORCH_CHECK(activeIndices.scalar_type() == torch::kInt32, "gather2d_cuda: activeIndices must be int32");
    // TORCH_CHECK(activeIndices.dim() == 2 && activeIndices.size(1) == 2, "gather2d_cuda: activeIndices must be [N,2]");

    const int R = bSizeH;
    const int S = bSizeW;

    const int numActive = static_cast<int>(activeIndices.size(0));
    const int B = static_cast<int>(x.size(0));
    const int C = static_cast<int>(x.size(1));
    const int H = static_cast<int>(x.size(2));
    const int W = static_cast<int>(x.size(3));

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device()).requires_grad(false);
    auto output = torch::empty({B * numActive, C, R, S}, options);
    if (numActive == 0 || output.numel() == 0) {
        return output;
    }

    const int* activeIndicesData = activeIndices.data_ptr<int>();
    const auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,   // FP16
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "gather2d_cuda",
        [&] {
            const scalar_t* xData = x.data_ptr<scalar_t>();
            scalar_t* outData = output.data_ptr<scalar_t>();

            const int64_t total = output.numel();
            const int threads = kThreads;
            const bool can_use_int32_p = (static_cast<int64_t>(B) * C * H * W) <= static_cast<int64_t>(std::numeric_limits<int>::max());

            if (total <= static_cast<int64_t>(std::numeric_limits<int>::max())) {
                const int total32 = static_cast<int>(total);
                const dim3 blocks((total32 + threads - 1) / threads);
                if (can_use_int32_p) {
                    gather2d_cuda_kernel<scalar_t, int, int><<<blocks, threads, 0, stream>>>(
                        total32, numActive,
                        B, C, H, W,
                        R, S,
                        xData, outData,
                        activeIndicesData);
                } else {
                    gather2d_cuda_kernel<scalar_t, int, int64_t><<<blocks, threads, 0, stream>>>(
                        total32, numActive,
                        B, C, H, W,
                        R, S,
                        xData, outData,
                        activeIndicesData);
                }
            } else {
                const dim3 blocks(static_cast<unsigned int>((total + threads - 1) / threads));
                if (can_use_int32_p) {
                    gather2d_cuda_kernel<scalar_t, int64_t, int><<<blocks, threads, 0, stream>>>(
                        total, numActive,
                        B, C, H, W,
                        R, S,
                        xData, outData,
                        activeIndicesData
                    );
                } else {
                    gather2d_cuda_kernel<scalar_t, int64_t, int64_t><<<blocks, threads, 0, stream>>>(
                        total, numActive,
                        B, C, H, W,
                        R, S,
                        xData, outData,
                        activeIndicesData
                    );
                }
            }
        }
    );

    return output;
}




// #include "sige3d_kernels_common.cuh"
// using namespace sige3d_kernels;

// __global__ void gather_cuda_kernel(
//         int total, int numActive,
//         int B, int C, int H, int W,
//         int R, int S,
//         const float *__restrict__ x,
//         float *__restrict__ output,
//         const int *activeIndices) {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index >= total) // 如果我没有对应的元素，就下班。
//         return;
//     int t = index;
//     int intraBw = t % S;
//     t /= S;
//     int intraBh = t % R;
//     t /= R;
//     int cc = t % C;
//     t /= C;
//     int ib = t % numActive, bb = t / numActive;

//     // 把第 ib 个激活 block 的左上角坐标 (biH, biW)，加上 block 内偏移 (intraBh, intraBw)，
//     // 映射回原始特征图 (H, W) 上的真实像素位置，并做越界保护。

//     // activeIndices 里存的就是每个 active block 在原图上的左上角（top-left）坐标。
//     int biH = activeIndices[ib << 1];
//     int hh = biH + intraBh; // 就是在把 block 内坐标映射回原图的真实坐标。
//     if (hh < 0 || hh >= H) {
//         output[index] = 0;
//         return;
//     }
//     int biW = activeIndices[ib << 1 | 1];
//     int ww = biW + intraBw;
//     if (ww < 0 || ww >= W) {
//         output[index] = 0;
//         return;
//     }

//     auto p = bb * C * H * W + cc * H * W + hh * W + ww;
//     output[index] = x[p];
// }


// // gather_cuda 是一个 CUDA kernel 的“胶水层（glue code / launcher）”，
// // 负责把 PyTorch 世界的 Tensor，翻译成 CUDA 世界的指针 + 整数参数。
// torch::Tensor gather2d_cuda(
//         const torch::Tensor &x,
//         int bSizeH, int bSizeW,
//         const torch::Tensor &activeIndices) {
//     const int R = bSizeH, S = bSizeW;
//     const int numActive = activeIndices.size(0);

//     // 我接下来要创建一个新 Tensor，它的数据类型和 x 一样，放在和 x 一样的设备上，而且它只是个普通结果，不参与反向传播。
//     auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device()).requires_grad(false);
//     auto xData = x.data_ptr<float>();

//     const auto activeIndicesData = activeIndices.data_ptr<int>();

//     const int B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
//     auto output = torch::empty({B * numActive, C, R, S}, options);
//     auto outputData = output.data_ptr<float>();


//     // 这就是“一共要算多少个 output 元素”
//     const int total = output.numel();   // 783 * 16 * 6 * 6 = 451008
    
//     const int threads = kThreads;
//     // 上采样
//     const dim3 blocks((total + threads - 1) / threads, 1);
//     gather_cuda_kernel<<<blocks, threads>>>(
//             total, numActive,
//             B, C, H, W, R, S,
//             xData, outputData, activeIndicesData);

//     return output;
// }


/*
在 C 语言直觉下: 
✔️ 4 重循环
✔️ 每次算 1 个 output 元素
✔️ index 是线性展开后的编号

for (int bb = 0; bb < B * numActive; bb++) {
    for (int cc = 0; cc < C; cc++) {
        for (int hh = 0; hh < R; hh++) {
            for (int ww = 0; ww < S; ww++) {

                int index = ((bb * C + cc) * R + hh) * S + ww;

                output[index] = ...; // gather + scale + shift + activation
            }
        }
    }
}

CUDA 不是不循环，而是：
把「for 循环的每一次迭代，交给一个线程来做」

CPU C: 串行
CUDA: 并行
*/
