import torch

from .backend import use_cuda_kernels


def _scatter2d_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    offset_h: int,
    offset_w: int,
    stride_h: int,
    stride_w: int,
    active_indices: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """PyTorch reference for sige/cuda/scatter_kernel.cu."""
    b, c, h, w = y.shape
    num_active = active_indices.size(0)
    r, s = x.size(2), x.size(3)

    # output = y.clone()
    # 注意：不需要clone，直接修改就行，因为这一个chunk只依赖上一个chunk
    output = y

    if num_active == 0:
        return output

    x_blocks = x.reshape(b, num_active, c, r, s)
    residual_y = residual.expand_as(y) if residual is not None else None

    for ib, (ai_h, ai_w) in enumerate(active_indices.tolist()):
        bi_h = (offset_h + ai_h) // stride_h
        bi_w = (offset_w + ai_w) // stride_w
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

        block = x_blocks[:, ib, :, dh0:dh1, dw0:dw1]
        if residual_y is not None:
            block = block + residual_y[:, :, h0:h1, w0:w1]
        output[:, :, h0:h1, w0:w1] = block

    return output


def scatter2d(
    x: torch.Tensor,
    y: torch.Tensor,
    offset_h: int,
    offset_w: int,
    stride_h: int,
    stride_w: int,
    active_indices: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    # torch.cuda.synchronize()
    # start = torch.cuda.Event(enable_timing=True)
    # end   = torch.cuda.Event(enable_timing=True)
    # start.record()

    if use_cuda_kernels() and x.is_cuda and y.is_cuda:
        try:
            from ._sige_cuda import get_sige3d_cuda_ext

            ext = get_sige3d_cuda_ext()
            if active_indices.device != x.device or active_indices.dtype != torch.int32:
                active_indices = active_indices.to(device=x.device, dtype=torch.int32)
            res = residual
            if res is not None and res.device != x.device:
                res = res.to(device=x.device)
            res = ext.scatter2d(
                x,
                y,
                int(offset_h),
                int(offset_w),
                int(stride_h),
                int(stride_w),
                active_indices.contiguous(),
                None if res is None else res.contiguous(),
            )

            # end.record()
            # torch.cuda.synchronize()
            # print(f"scatter2d time: {start.elapsed_time(end):.2f} ms")   # ms

            return res
        except Exception:
            raise
      
    return _scatter2d_torch(
        x,
        y,
        offset_h,
        offset_w,
        stride_h,
        stride_w,
        active_indices,
        residual=residual
    )
