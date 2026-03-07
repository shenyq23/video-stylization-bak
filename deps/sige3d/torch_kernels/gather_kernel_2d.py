import torch

from .backend import use_cuda_kernels


def _gather2d_torch(
    x: torch.Tensor,
    b_size_h: int,
    b_size_w: int,
    active_indices: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference for sige/cuda/gather_kernel.cu."""

    # print("*" * 40)
    # print("Use Pytorch!!!")
    # print("*" * 40)

    # 输入 x 没有 padding
    b, c, h, w = x.shape
    num_active = active_indices.size(0)
    r, s = int(b_size_h), int(b_size_w)

    output = torch.zeros((b, num_active, c, r, s), dtype=x.dtype, device=x.device)
    if num_active == 0:
        return output.view(b * num_active, c, r, s)

    for ib, (bi_h, bi_w) in enumerate(active_indices.tolist()):
        h0 = max(bi_h, 0)
        h1 = min(bi_h + r, h)
        w0 = max(bi_w, 0)
        w1 = min(bi_w + s, w)
        if h0 >= h1 or w0 >= w1:
            continue
        dh0 = h0 - bi_h
        dh1 = dh0 + (h1 - h0)
        dw0 = w0 - bi_w
        dw1 = dw0 + (w1 - w0)

        block = x[:, :, h0:h1, w0:w1]
        
        # output.shape == [1, 785, 16, 6, 6]
        # block.shape  == [1, 16, 5, 5]
        # output 左上角那一整圈（第 0 行、第 0 列）确实没有任何 x 的元素填进去
        # 全是 0
        output[:, ib, :, dh0:dh1, dw0:dw1] = block

    # gather 的输出就是后续 conv2d / conv3d 的输入 block
    return output.view(b * num_active, c, r, s)


def gather2d(
    x: torch.Tensor,
    b_size_h: int,
    b_size_w: int,
    active_indices: torch.Tensor,
) -> torch.Tensor:

    # torch.cuda.synchronize()
    # start = torch.cuda.Event(enable_timing=True)
    # end   = torch.cuda.Event(enable_timing=True)
    # start.record()
    
    if use_cuda_kernels() and x.is_cuda:
        try:
            from ._sige_cuda import get_sige3d_cuda_ext

            ext = get_sige3d_cuda_ext()
            # if active_indices.device != x.device or active_indices.dtype != torch.int32:
                # active_indices = active_indices.to(device=x.device, dtype=torch.int32)

            # torch.cuda.synchronize()
            # start = torch.cuda.Event(enable_timing=True)
            # end   = torch.cuda.Event(enable_timing=True)
            # start.record()
            # print(x.shape)

            res = ext.gather2d(x, int(b_size_h), int(b_size_w), active_indices)

            # end.record()
            # torch.cuda.synchronize()
            # print(f"gather2d time: {start.elapsed_time(end):.2f} ms")   # ms

            return res
        except Exception:
            raise

    return _gather2d_torch(x, b_size_h, b_size_w, active_indices)

'''
output block (6×6):

      0   1   2   3   4   5
    ┌─────────────────────┐
0   │ 0   0   0   0   0   0 │  ← padding 行
1   │ 0  x00 x01 x02 x03 x04│
2   │ 0  x10 x11 x12 x13 x14│
3   │ 0  x20 x21 x22 x23 x24│
4   │ 0  x30 x31 x32 x33 x34│
5   │ 0  x40 x41 x42 x43 x44│
    └─────────────────────┘
    
'''
