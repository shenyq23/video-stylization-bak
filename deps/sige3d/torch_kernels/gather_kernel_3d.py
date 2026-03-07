from __future__ import annotations

import torch

from ..activation import activation
from .backend import use_cuda_kernels


def _gather3d_torch(
    x: torch.Tensor,
    b_size_h: int,
    b_size_w: int,
    active_indices: torch.Tensor,
    activation_name: str = "identity",
    rms_norm_fn = None,             # ✅ 新增：可传 RMS_norm forward
    is_cache_gather: bool = False,  # 不需要norm和激活
) -> torch.Tensor:
    """
    A 3D (T,H,W) extension of `sige/cuda/gather_kernel.py` (torch reference).

    - Input:  `x` with shape [B, C, T, H, W] (NO spatial padding applied)
    - Output: gathered blocks with shape [B*num_active, C, T, b_size_h, b_size_w]

    Note:
      - We only gather on spatial (H,W) using `active_indices`.
      - Temporal causal context is handled by the causal Conv3d module via per-layer caches.
    """
    b, c, t, h, w = x.shape
    num_active = int(active_indices.size(0))
    r, s = int(b_size_h), int(b_size_w)

    # output blocks: [B, num_active, C, T, r, s] -> flatten to [B*num_active, C, T, r, s]
    output = torch.zeros((b, num_active, c, t, r, s), dtype=x.dtype, device=x.device)
    if num_active == 0:
        return output.view(b * num_active, c, t, r, s)

    for ib, (bi_h, bi_w) in enumerate(active_indices.tolist()):
        h0 = max(bi_h, 0)
        h1 = min(bi_h + r, h)
        w0 = max(bi_w, 0)
        w1 = min(bi_w + s, w)
        if h0 >= h1 or w0 >= w1:
            continue

        # Where the valid region lands inside the (r,s) output block.
        dh0 = h0 - bi_h
        dh1 = dh0 + (h1 - h0)
        dw0 = w0 - bi_w
        dw1 = dw0 + (w1 - w0)

        block = x[:, :, :, h0:h1, w0:w1]

        # 先做 RMS_Norm
        if rms_norm_fn is not None and not is_cache_gather:
            block = rms_norm_fn(block)
            block = activation(block, activation_name)

        output[:, ib, :, :, dh0:dh1, dw0:dw1] = block

    return output.view(b * num_active, c, t, r, s)


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


def gather3d(
    x: torch.Tensor,
    b_size_h: int,
    b_size_w: int,
    active_indices: torch.Tensor,
    activation_name: str = "identity",
    rms_norm_fn=None,
    is_cache_gather: bool = False,
) -> torch.Tensor:
    # torch.cuda.synchronize()
    # start = torch.cuda.Event(enable_timing=True)
    # end   = torch.cuda.Event(enable_timing=True)
    # start.record()


    if use_cuda_kernels() and x.is_cuda:
        act = (activation_name or "identity").strip().lower()
        if act not in {"identity", "silu", "swish"}:
            return _gather3d_torch(
                  x, b_size_h, b_size_w, active_indices, activation_name, rms_norm_fn=rms_norm_fn, is_cache_gather=is_cache_gather
            )
        try:
            from ._sige_cuda import get_sige3d_cuda_ext

            ext = get_sige3d_cuda_ext()
            if active_indices.device != x.device or active_indices.dtype != torch.int32:
                active_indices = active_indices.to(device=x.device, dtype=torch.int32)

            gamma = None
            bias = None
            eps = 1e-6
            if rms_norm_fn is not None and not is_cache_gather:
                gamma, bias, eps = _extract_rms_norm_params(rms_norm_fn, x)
                if gamma is None:
                    return _gather3d_torch(
                        x,
                        b_size_h,
                        b_size_w,
                        active_indices,
                        activation_name,
                        rms_norm_fn=rms_norm_fn,
                        is_cache_gather=is_cache_gather,
                    )
            return ext.gather3d(
                x,
                int(b_size_h),
                int(b_size_w),
                active_indices.contiguous(),
                gamma,
                bias,
                float(eps),
                act,
            )

            # end.record()
            # torch.cuda.synchronize()
            # print(f"gather3d time: {start.elapsed_time(end):.2f} ms")   # ms
            # return res
        except Exception:
            raise

    return _gather3d_torch(
        x,
        b_size_h,
        b_size_w,
        active_indices,
        activation_name,
        rms_norm_fn=rms_norm_fn,
        is_cache_gather=is_cache_gather
    )
