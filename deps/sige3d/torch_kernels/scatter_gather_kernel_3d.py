from __future__ import annotations

import torch

from ..activation import activation
from .backend import use_cuda_kernels


# fast
def _get_scatter_map_torch(
    h, w,
    b_size_h, b_size_w,
    k_size_h, k_size_w,
    offset_h, offset_w,
    stride_h, stride_w,
    active_indices: torch.Tensor,
):
    device = active_indices.device
    scatter_map = torch.full((h, w, 3), -1, dtype=torch.int32, device=device)

    # r,s: 每个 block 会覆盖的输出点数（沿 H/W 的次数）
    r = (b_size_h - k_size_h) // stride_h + 1
    s = (b_size_w - k_size_w) // stride_w + 1

    if active_indices.numel() == 0:
        return scatter_map

    if active_indices.is_cuda and use_cuda_kernels():
        try:
            from ..cuda_kernels import get_extension

            ext = get_extension()
            return ext.get_scatter_map(
                int(h),
                int(w),
                int(b_size_h),
                int(b_size_w),
                int(k_size_h),
                int(k_size_w),
                int(offset_h),
                int(offset_w),
                int(stride_h),
                int(stride_w),
                active_indices.to(dtype=torch.int32).contiguous(),
            )
        except Exception:
            pass

    # [N]
    ai_h = active_indices[:, 0].to(torch.int64)
    ai_w = active_indices[:, 1].to(torch.int64)

    # [N] 计算 block 左上角对应的输出基准位置
    bi_h = (offset_h + ai_h) // stride_h
    bi_w = (offset_w + ai_w) // stride_w

    N = active_indices.shape[0]
    ib = torch.arange(N, device=device, dtype=torch.int64)  # [N]

    # intra offsets
    intra_bh = torch.arange(r, device=device, dtype=torch.int64)  # [r]
    intra_bw = torch.arange(s, device=device, dtype=torch.int64)  # [s]

    # 生成所有覆盖位置：
    # hh: [N, r, 1] + [1, r, 1] -> [N, r, 1]
    hh = bi_h[:, None, None] + intra_bh[None, :, None]  # [N, r, 1]
    ww = bi_w[:, None, None] + intra_bw[None, None, :]  # [N, 1, s]

    # broadcast 到 [N, r, s]
    hh = hh.expand(N, r, s)
    ww = ww.expand(N, r, s)

    # 边界过滤
    valid = (hh >= 0) & (hh < h) & (ww >= 0) & (ww < w)
    hh = hh[valid]
    ww = ww[valid]

    # 同步生成对应的写入值
    # block_id：每个 (r,s) 都对应同一个 ib
    block_id = ib[:, None, None].expand(N, r, s)[valid]
    intra_h  = intra_bh[None, :, None].expand(N, r, s)[valid]
    intra_w  = intra_bw[None, None, :].expand(N, r, s)[valid]

    # 写入（一次性）
    scatter_map[hh, ww, 0] = block_id.to(torch.int32)
    scatter_map[hh, ww, 1] = intra_h.to(torch.int32)
    scatter_map[hh, ww, 2] = intra_w.to(torch.int32)

    return scatter_map



# def _get_scatter_map_torch(
#     h: int,
#     w: int,
#     b_size_h: int,
#     b_size_w: int,
#     k_size_h: int,
#     k_size_w: int,
#     offset_h: int,
#     offset_w: int,
#     stride_h: int,
#     stride_w: int,
#     active_indices: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     3D extension of get_scatter_map in `sige/cuda/scatter_gather_kernel.py`.
#     Still a pure spatial (H,W) map: [H, W, 3] -> (block_id, intra_h, intra_w).
#     """
#     scatter_map = torch.full((h, w, 3), -1, dtype=torch.int32, device=active_indices.device)
#     r = (b_size_h - k_size_h) // stride_h + 1
#     s = (b_size_w - k_size_w) // stride_w + 1
#     for ib, (ai_h, ai_w) in enumerate(active_indices.tolist()):
#         bi_h = (offset_h + ai_h) // stride_h
#         bi_w = (offset_w + ai_w) // stride_w
#         for intra_bh in range(r):
#             hh = bi_h + intra_bh
#             if hh < 0 or hh >= h:
#                 continue
#             for intra_bw in range(s):
#                 ww = bi_w + intra_bw
#                 if ww < 0 or ww >= w:
#                     continue
#                 scatter_map[hh, ww, 0] = ib
#                 scatter_map[hh, ww, 1] = intra_bh
#                 scatter_map[hh, ww, 2] = intra_bw
#     return scatter_map



def _extract_rms_norm_params(rms_norm_fn, x: torch.Tensor):
    if rms_norm_fn is None:
        return None, None, None
    mod = getattr(rms_norm_fn, "__self__", None)
    if mod is None:
        return None, None, None
    gamma = getattr(mod, "gamma", None)
    if not torch.is_tensor(gamma):
        return None, None, None
    eps = float(getattr(mod, "eps", 1e-6))
    bias = getattr(mod, "bias", None)
    gamma_t = gamma.to(device=x.device, dtype=x.dtype).contiguous().view(-1)
    bias_t = bias.to(device=x.device, dtype=x.dtype).contiguous().view(-1) if torch.is_tensor(bias) else None
    return gamma_t, bias_t, eps


def get_scatter_map(
    h: int,
    w: int,
    b_size_h: int,
    b_size_w: int,
    k_size_h: int,
    k_size_w: int,
    offset_h: int,
    offset_w: int,
    stride_h: int,
    stride_w: int,
    active_indices: torch.Tensor,
) -> torch.Tensor:
    if use_cuda_kernels() and active_indices.is_cuda:
        try:
            from ._sige_cuda import get_sige3d_cuda_ext

            ext = get_sige3d_cuda_ext()
            if active_indices.dtype != torch.int32:
                active_indices = active_indices.to(dtype=torch.int32)
            return ext.get_scatter_map(
                int(h),
                int(w),
                int(b_size_h),
                int(b_size_w),
                int(k_size_h),
                int(k_size_w),
                int(offset_h),
                int(offset_w),
                int(stride_h),
                int(stride_w),
                active_indices.contiguous(),
            )
        except Exception:
            pass
    return _get_scatter_map_torch(
        h,
        w,
        b_size_h,
        b_size_w,
        k_size_h,
        k_size_w,
        offset_h,
        offset_w,
        stride_h,
        stride_w,
        active_indices,
    )


def _scatter_gather3d_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    b_size_h: int,
    b_size_w: int,
    active_indices: torch.Tensor,
    scatter_map: torch.Tensor,
    activation_name: str = "identity",
    rms_norm_fn = None,   # ✅ 新增：可传 RMS_norm forward
) -> torch.Tensor:
    """
    3D (T,H,W) extension of `sige/cuda/scatter_gather_kernel.py` (torch reference).

    - `x`: updated conv outputs as blocks, shape [B*num_active, C, T, rx, sx]
    - `y`: baseline full tensor, shape [B, C, T, H, W]
    - returns: gathered blocks for the next conv, shape [B*num_active, C, T, b_size_h, b_size_w]
    """
    b, c, t, h, w = y.shape
    num_active = int(active_indices.size(0))
    ro, so = int(b_size_h), int(b_size_w)
    rx, sx = int(x.size(-2)), int(x.size(-1))

    # output blocks: [B, num_active, C, T, ro, so] -> flatten to [B*num_active, C, T, ro, so]
    output = torch.zeros((b, num_active, c, t, ro, so), dtype=x.dtype, device=x.device)
    if num_active == 0:
        return output.view(b * num_active, c, t, ro, so)

    x_blocks = x.reshape(b, num_active, c, t, rx, sx)

    device = x.device
    if scatter_map.device != device:
        scatter_map = scatter_map.to(device=device)
    if active_indices.device != device:
        active_indices = active_indices.to(device=device)
    ai = active_indices.to(dtype=torch.int64)
    bi_h = ai[:, 0]
    bi_w = ai[:, 1]

    intra_bh = torch.arange(ro, device=device, dtype=torch.int64)
    intra_bw = torch.arange(so, device=device, dtype=torch.int64)

    hh = bi_h[:, None, None] + intra_bh[None, :, None]  # [N, ro, 1]
    ww = bi_w[:, None, None] + intra_bw[None, None, :]  # [N, 1, so]
    hh = hh.expand(num_active, ro, so)
    ww = ww.expand(num_active, ro, so)

    valid = (hh >= 0) & (hh < h) & (ww >= 0) & (ww < w)
    hh_safe = hh.clamp(0, h - 1)
    ww_safe = ww.clamp(0, w - 1)

    scatter_vals = scatter_map[hh_safe, ww_safe]
    bx = scatter_vals[..., 0].to(torch.int64)
    hx = scatter_vals[..., 1].to(torch.int64)
    wx = scatter_vals[..., 2].to(torch.int64)

    use_x = bx >= 0
    bx_safe = torch.where(use_x, bx, torch.zeros_like(bx))
    hx_safe = torch.where(use_x, hx, torch.zeros_like(hx))
    wx_safe = torch.where(use_x, wx, torch.zeros_like(wx))

    # Gather from x blocks.
    x_flat = x_blocks.permute(0, 2, 3, 1, 4, 5).reshape(b, c, t, num_active * rx * sx)
    idx_x = (bx_safe * (rx * sx) + hx_safe * sx + wx_safe).reshape(1, 1, 1, -1)
    idx_x = idx_x.expand(b, c, t, -1)
    z_x = torch.take_along_dim(x_flat, idx_x, dim=3)
    z_x = z_x.reshape(b, c, t, num_active, ro, so).permute(0, 3, 1, 2, 4, 5)

    # Gather from baseline y.
    y_flat = y.reshape(b, c, t, h * w)
    idx_y = (hh_safe * w + ww_safe).reshape(1, 1, 1, -1)
    idx_y = idx_y.expand(b, c, t, -1)
    z_y = torch.take_along_dim(y_flat, idx_y, dim=3)
    z_y = z_y.reshape(b, c, t, num_active, ro, so).permute(0, 3, 1, 2, 4, 5)

    use_x_mask = use_x[None, :, None, None, :, :]
    z = torch.where(use_x_mask, z_x, z_y)

    if rms_norm_fn is not None:
        z_reshape = z.permute(0, 2, 3, 1, 4, 5).reshape(b, c, t, num_active * ro, so)
        z_reshape = rms_norm_fn(z_reshape)
        z_reshape = activation(z_reshape, activation_name)
        z = z_reshape.reshape(b, c, t, num_active, ro, so).permute(0, 3, 1, 2, 4, 5)

    valid_mask = valid[None, :, None, None, :, :]
    z = torch.where(valid_mask, z, torch.zeros_like(z))

    return z.reshape(b * num_active, c, t, ro, so)






def scatter_gather3d(
    x: torch.Tensor,
    y: torch.Tensor,
    b_size_h: int,
    b_size_w: int,
    active_indices: torch.Tensor,
    scatter_map: torch.Tensor,
    activation_name: str = "identity",
    rms_norm_fn=None,
) -> torch.Tensor:
    if use_cuda_kernels() and x.is_cuda and y.is_cuda:
        act = (activation_name or "identity").strip().lower()
        if act not in {"identity", "silu", "swish"}:
            return _scatter_gather3d_torch(
                x,
                y,
                b_size_h,
                b_size_w,
                active_indices,
                scatter_map,
                activation_name,
                rms_norm_fn=rms_norm_fn,
            )
        try:
            from ._sige_cuda import get_sige3d_cuda_ext

            ext = get_sige3d_cuda_ext()
            if active_indices.device != x.device or active_indices.dtype != torch.int32:
                active_indices = active_indices.to(device=x.device, dtype=torch.int32)
            if scatter_map.device != x.device or scatter_map.dtype != torch.int32:
                scatter_map = scatter_map.to(device=x.device, dtype=torch.int32)

            gamma = None
            bias = None
            eps = 1e-6
            if rms_norm_fn is not None:
                gamma, bias, eps = _extract_rms_norm_params(rms_norm_fn, x)
                if gamma is None:
                    return _scatter_gather3d_torch(
                        x,
                        y,
                        b_size_h,
                        b_size_w,
                        active_indices,
                        scatter_map,
                        activation_name,
                        rms_norm_fn=rms_norm_fn,
                    )

            return ext.scatter_gather3d(
                x,
                y,
                int(b_size_h),
                int(b_size_w),
                active_indices.contiguous(),
                scatter_map.contiguous(),
                gamma,
                bias,
                float(eps),
                act,
            )
        except Exception as e:
            raise RuntimeError("SIGE CUDA failed") from e

    return _scatter_gather3d_torch(
        x,
        y,
        b_size_h,
        b_size_w,
        active_indices,
        scatter_map,
        activation_name,
        rms_norm_fn=rms_norm_fn,
    )
