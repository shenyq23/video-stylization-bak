from __future__ import annotations

import torch

from .backend import use_cuda_kernels


def _scatter3d_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    offset_h: int,
    offset_w: int,
    stride_h: int,
    stride_w: int,
    active_indices: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    A 3D (T,H,W) extension of `sige/cuda/scatter_kernel.py` (torch reference).

    - `x`: blocks, shape [B*num_active, C, T, r, s]
    - `y`: baseline full tensor, shape [B, C, T, H, W]
    - returns: full tensor with sparse updates written in, shape [B, C, T, H, W]
    """
    b, c, t, h, w = y.shape
    num_active = int(active_indices.size(0))
    r, s = int(x.size(-2)), int(x.size(-1))

    # output = y.clone()
    # 注意：不需要clone，直接修改就行，因为这一个chunk只依赖上一个chunk
    output = y
    if num_active == 0:
        return output

    # [B*num_active, C, T, r, s] -> [B, num_active, C, T, r, s]
    x_blocks = x.view(b, num_active, c, t, r, s)
    residual_y = residual if residual is None else residual.expand_as(y)

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

        block = x_blocks[:, ib, :, :, dh0:dh1, dw0:dw1]
        if residual_y is not None:
            block = block + residual_y[:, :, :, h0:h1, w0:w1]
        output[:, :, :, h0:h1, w0:w1] = block

    return output


def scatter3d(
    x: torch.Tensor,
    y: torch.Tensor,
    offset_h: int,
    offset_w: int,
    stride_h: int,
    stride_w: int,
    active_indices: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    if use_cuda_kernels() and x.is_cuda and y.is_cuda:
        try:
            from ._sige_cuda import get_sige3d_cuda_ext

            ext = get_sige3d_cuda_ext()
            if active_indices.device != x.device or active_indices.dtype != torch.int32:
                active_indices = active_indices.to(device=x.device, dtype=torch.int32)
            res = residual
            if res is not None and res.device != x.device:
                res = res.to(device=x.device)
            return ext.scatter3d(
                x,
                y,
                int(offset_h),
                int(offset_w),
                int(stride_h),
                int(stride_w),
                active_indices.contiguous(),
                None if res is None else res.contiguous(),
            )
        except Exception:
            raise
    return _scatter3d_torch(x, y, offset_h, offset_w, stride_h, stride_w, active_indices, residual=residual)


def _scatter_with_block_residual3d_torch(
    x0: torch.Tensor,
    y0: torch.Tensor,
    x1: torch.Tensor,
    y1: torch.Tensor,
    offset_h: int,
    offset_w: int,
    stride_h: int,
    stride_w: int,
    active_indices0: torch.Tensor,
    active_indices1: torch.Tensor,
) -> torch.Tensor:
    """
    3D extension of `scatter_with_block_residual` in `sige/cuda/scatter_kernel.py`.

    Shapes:
      - x0/x1: [B*num_active?, C, T, r, s] sparse blocks
      - y0/y1: [B, C, T, H, W] baseline full tensors
    """
    output = _scatter3d_torch(
        x0,
        y0,
        offset_h,
        offset_w,
        stride_h,
        stride_w,
        active_indices0,
        y1,  # scatter x0 + y1
    )

    b, c, t, h, w = y1.shape
    num_active = int(active_indices1.size(0))
    if num_active == 0:
        return output

    r, s = int(x1.size(-2)), int(x1.size(-1))
    x1_blocks = x1.view(b, num_active, c, t, r, s)

    for ib, (bi_h, bi_w) in enumerate(active_indices1.tolist()):
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

        output[:, :, :, h0:h1, w0:w1] += x1_blocks[:, ib, :, :, dh0:dh1, dw0:dw1] - y1[:, :, :, h0:h1, w0:w1]

    return output


def scatter_with_block_residual3d(
    x0: torch.Tensor,
    y0: torch.Tensor,
    x1: torch.Tensor,
    y1: torch.Tensor,
    offset_h: int,
    offset_w: int,
    stride_h: int,
    stride_w: int,
    active_indices0: torch.Tensor,
    active_indices1: torch.Tensor,
) -> torch.Tensor:
    if use_cuda_kernels() and x0.is_cuda and y0.is_cuda:
        try:
            from ._sige_cuda import get_sige3d_cuda_ext

            ext = get_sige3d_cuda_ext()
            if active_indices0.device != x0.device or active_indices0.dtype != torch.int32:
                active_indices0 = active_indices0.to(device=x0.device, dtype=torch.int32)
            if active_indices1.device != x0.device or active_indices1.dtype != torch.int32:
                active_indices1 = active_indices1.to(device=x0.device, dtype=torch.int32)
            return ext.scatter_with_block_residual3d(
                x0,
                y0,
                x1,
                y1,
                int(offset_h),
                int(offset_w),
                int(stride_h),
                int(stride_w),
                active_indices0.contiguous(),
                active_indices1.contiguous(),
            )
        except Exception as e:
            raise RuntimeError("SIGE CUDA failed") from e

    return _scatter_with_block_residual3d_torch(
        x0,
        y0,
        x1,
        y1,
        offset_h,
        offset_w,
        stride_h,
        stride_w,
        active_indices0,
        active_indices1,
    )
