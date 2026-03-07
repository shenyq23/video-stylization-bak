from __future__ import annotations

import atexit
import os
import sys
import threading
from collections import Counter

import torch
import torch.nn.functional as F


_FLOW_CACHE_SHAPE_CALL_COUNTS: Counter[tuple[int, ...]] = Counter()
_FLOW_CACHE_SHAPE_UNIQUE_SLOT_COUNTS: Counter[tuple[int, ...]] = Counter()
_FLOW_CACHE_SLOT_TO_SHAPE: dict[tuple[int, str], tuple[int, ...]] = {}
_FLOW_CACHE_STATS_LOCK = threading.Lock()


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _infer_owner_slot_name(owner: object, cache: torch.Tensor) -> str | None:
    owner_dict = getattr(owner, "__dict__", None)
    if not isinstance(owner_dict, dict):
        return None
    for name, value in owner_dict.items():
        if value is cache:
            return str(name)
    return None


def _record_flow_cache_shape(cache: torch.Tensor, *, owner: object | None = None) -> None:
    shape = tuple(int(s) for s in cache.shape)
    with _FLOW_CACHE_STATS_LOCK:
        _FLOW_CACHE_SHAPE_CALL_COUNTS[shape] += 1

        if owner is None:
            return
        slot_name = _infer_owner_slot_name(owner, cache)
        if slot_name is None:
            return

        slot_key = (id(owner), slot_name)
        prev_shape = _FLOW_CACHE_SLOT_TO_SHAPE.get(slot_key)
        if prev_shape == shape:
            return

        if prev_shape is not None:
            _FLOW_CACHE_SHAPE_UNIQUE_SLOT_COUNTS[prev_shape] -= 1
            if _FLOW_CACHE_SHAPE_UNIQUE_SLOT_COUNTS[prev_shape] <= 0:
                del _FLOW_CACHE_SHAPE_UNIQUE_SLOT_COUNTS[prev_shape]

        _FLOW_CACHE_SLOT_TO_SHAPE[slot_key] = shape
        _FLOW_CACHE_SHAPE_UNIQUE_SLOT_COUNTS[shape] += 1


def get_flow_cache_shape_counts(*, unique_slots: bool = True) -> dict[tuple[int, ...], int]:
    """
    Return cache shape counts recorded in this process.

    - unique_slots=True: count unique cache "slots" (e.g. module.original_outputs),
      which approximates "how many caches of each shape exist".
    - unique_slots=False: count how many times each shape was passed into the warp fns.
    """
    with _FLOW_CACHE_STATS_LOCK:
        src = _FLOW_CACHE_SHAPE_UNIQUE_SLOT_COUNTS if unique_slots else _FLOW_CACHE_SHAPE_CALL_COUNTS
        return dict(src)


def reset_flow_cache_shape_counts() -> None:
    with _FLOW_CACHE_STATS_LOCK:
        _FLOW_CACHE_SHAPE_CALL_COUNTS.clear()
        _FLOW_CACHE_SHAPE_UNIQUE_SLOT_COUNTS.clear()
        _FLOW_CACHE_SLOT_TO_SHAPE.clear()


def print_flow_cache_shape_stats(*, unique_slots: bool = True, file=None) -> None:
    if file is None:
        file = sys.stderr
    counts = get_flow_cache_shape_counts(unique_slots=unique_slots)
    if not counts:
        print("[flow_cache_utils] no cache shapes recorded", file=file)
        return
    title = "unique cache slots by shape" if unique_slots else "warp calls by shape"
    print(f"[flow_cache_utils] {title}:", file=file)
    for shape, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {shape}: {count}", file=file)


if _env_truthy(os.getenv("FLOW_CACHE_SHAPE_STATS")):
    atexit.register(print_flow_cache_shape_stats)
    # atexit.register(lambda: print_flow_cache_shape_stats(unique_slots=False))



def _as_flow_hw2(flow: torch.Tensor) -> torch.Tensor:
    """Accept (H,W,2), (2,H,W), (1,2,H,W) and return contiguous (H,W,2)."""
    if flow.dim() == 4:
        if int(flow.size(0)) != 1 or int(flow.size(1)) != 2:
            raise ValueError(f"Unsupported flow shape: {tuple(flow.shape)} (expected (1,2,H,W))")
        flow = flow[0]  # (2,H,W)

    if flow.dim() == 3 and int(flow.size(-1)) == 2:
        return flow.contiguous()

    if flow.dim() == 3 and int(flow.size(0)) == 2:
        return flow.permute(1, 2, 0).contiguous()

    raise ValueError(f"Unsupported flow shape: {tuple(flow.shape)} (expected (H,W,2) or (2,H,W))")

def normalize_flow(flow: torch.Tensor, h: int, w: int, *, device: torch.device) -> torch.Tensor:
    """
    Normalize flow to shape (H, W, 2) on `device`.

    If input resolution differs from (h,w), bilinearly resize and scale the
    displacement so it stays in pixel units at the target resolution.
    """
    if not torch.is_tensor(flow):
        flow = torch.as_tensor(flow)

    flow = _as_flow_hw2(flow).to(device=device, dtype=torch.float32)

    h0, w0 = int(flow.size(0)), int(flow.size(1))
    if (h0, w0) == (h, w):
        return flow

    flow_chw = flow.permute(2, 0, 1).unsqueeze(0)  # (1,2,H,W)
    flow_rs = F.interpolate(flow_chw, size=(h, w), mode="bilinear", align_corners=False)

    # Keep displacement in pixel units after resizing.
    #
    # Intuition: if a 480p image moves +10px, then at 240p it should be +5px.
    # So we scale by the resolution ratio (target / source).
    flow_rs[:, 0].mul_(float(w) / float(max(w0, 1)))
    flow_rs[:, 1].mul_(float(h) / float(max(h0, 1)))

    return flow_rs[0].permute(1, 2, 0).contiguous()




def forward_warp_cache_5d(cache: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Backward-warp cached features by bilinear sampling (grid_sample).

    - cache: (B, C, T, H, W)
    - flow:  (H, W, 2) / (2, H, W) / (1, 2, H, W), (dx, dy) in pixel units
    - output: same shape, output[..., y, x] samples input at (x+dx, y+dy)
    """
    # _record_flow_cache_shape(cache, owner=sys._getframe(1).f_locals.get("self"))  # noqa: SLF001

    if cache.dim() != 5:
        raise ValueError(f"cache must be 5D (B,C,T,H,W); got {tuple(cache.shape)}")
    b, c, t, h, w = cache.shape

    flow = normalize_flow(flow, int(h), int(w), device=cache.device)

    # Base grid in pixel coords: (x,y) with x in [0,w-1], y in [0,h-1].
    y = torch.arange(h, device=cache.device, dtype=torch.float32).view(h, 1)
    x = torch.arange(w, device=cache.device, dtype=torch.float32).view(1, w)
    base_x = x.expand(h, w)
    base_y = y.expand(h, w)

    sample_x = base_x + flow[..., 0]
    sample_y = base_y + flow[..., 1]

    # Normalize to [-1, 1] for grid_sample (align_corners=True).
    w_denom = float(max(w - 1, 1))
    h_denom = float(max(h - 1, 1))
    x_grid = 2.0 * sample_x / w_denom - 1.0
    y_grid = 2.0 * sample_y / h_denom - 1.0

    # w_denom = float(max(w, 1))
    # h_denom = float(max(h, 1))
    # x_grid = (2.0 * sample_x + 1.0) / w_denom - 1.0
    # y_grid = (2.0 * sample_y + 1.0) / h_denom - 1.0
    
    grid = torch.stack([x_grid, y_grid], dim=-1)  # (H,W,2)

    # Warp each time slice independently with the same spatial grid.
    # cache_2d = cache.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
    cache_2d = cache.permute(0, 2, 1, 3, 4).view(b * t, c, h, w)


    # === FP16 version ===
    # print("*" * 40)
    # print("Using FP16!!!")
    # print("*" * 40)

    cache_2d_fp = cache_2d.to(dtype=torch.float16)
    grid_bt = grid.unsqueeze(0).expand(b * t, h, w, 2).to(dtype=cache_2d_fp.dtype)
   
    # torch.cuda.synchronize()
    # start = torch.cuda.Event(enable_timing=True)
    # end   = torch.cuda.Event(enable_timing=True)
    # start.record()

    warped = F.grid_sample(
        cache_2d_fp,
        grid_bt,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    # end.record()
    # torch.cuda.synchronize()
    # print(cache_2d_fp.shape)
    # print(f"flow fp16: {start.elapsed_time(end):.2f} ms")   # ms

    # warped = warped.to(dtype=cache_2d.dtype)

    # return warped.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).to(torch.float8_e4m3fn)
    return warped.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)


    # torch.cuda.synchronize()
    # start = torch.cuda.Event(enable_timing=True)
    # end   = torch.cuda.Event(enable_timing=True)
    # start.record()

    grid_bt = grid.unsqueeze(0).expand(b * t, h, w, 2).to(dtype=torch.float32)
    cache_2d_f = cache_2d.to(dtype=torch.float32)

    warped_f = F.grid_sample(
        cache_2d_f,
        grid_bt,
        mode="bilinear",
        padding_mode="zeros", # 零填充
        align_corners=True,
    )

    # end.record()
    # torch.cuda.synchronize()
    # print(f"flow float32: {start.elapsed_time(end):.2f} ms")   # ms

    warped = warped_f.to(dtype=cache_2d.dtype)
    return warped.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)



# @torch.compile(mode='reduce-overhead')
def forward_warp_cache_4d(cache: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Backward-warp cached features by bilinear sampling (grid_sample).

    - cache: (B, C, H, W)
    - flow:  (H, W, 2) / (2, H, W) / (1, 2, H, W), (dx, dy) in pixel units
    - output: same shape, output[..., y, x] samples input at (x+dx, y+dy)
    """
    # _record_flow_cache_shape(cache, owner=sys._getframe(1).f_locals.get("self"))  # noqa: SLF001

    if cache.dim() != 4:
        raise ValueError(f"cache must be 4D (B,C,H,W); got {tuple(cache.shape)}")
    b, c, h, w = cache.shape

    flow = normalize_flow(flow, int(h), int(w), device=cache.device)

    # Base grid in pixel coords: (x,y) with x in [0,w-1], y in [0,h-1].
    y = torch.arange(h, device=cache.device, dtype=torch.float32).view(h, 1)
    x = torch.arange(w, device=cache.device, dtype=torch.float32).view(1, w)
    base_x = x.expand(h, w)
    base_y = y.expand(h, w)

    sample_x = base_x + flow[..., 0]
    sample_y = base_y + flow[..., 1]

    # Normalize to [-1, 1] for grid_sample (align_corners=True).
    w_denom = float(max(w - 1, 1))
    h_denom = float(max(h - 1, 1))
    x_grid = 2.0 * sample_x / w_denom - 1.0
    y_grid = 2.0 * sample_y / h_denom - 1.0

    # w_denom = float(max(w, 1))
    # h_denom = float(max(h, 1))
    # x_grid = (2.0 * sample_x + 1.0) / w_denom - 1.0
    # y_grid = (2.0 * sample_y + 1.0) / h_denom - 1.0

    grid = torch.stack([x_grid, y_grid], dim=-1)  # (H,W,2)

    # # Expand to batch.
    grid_b = grid.unsqueeze(0).expand(b, h, w, 2).to(dtype=cache.dtype)

    warped = F.grid_sample(
        cache,
        grid_b,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return warped


    grid_b = grid.unsqueeze(0).expand(b, h, w, 2).to(dtype=torch.float32)
    cache_f = cache.to(dtype=torch.float32)
    warped_f = F.grid_sample(
        cache_f,
        grid_b,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped_f.to(dtype=cache.dtype)


# def forward_warp_cache_5d(cache: torch.Tensor, flow: torch.Tensor, align_corners: bool = False) -> torch.Tensor:
#     """
#     Warp cache (B,C,T,H,W) with 5D grid_sample.
#     flow is (H,W,2)/(2,H,W)/(1,2,H,W) in pixel units, applied to every T slice.
#     output[..., t, y, x] samples input at (t, y+dy, x+dx).
#     """
#     if cache.dim() != 5:
#         raise ValueError(f"cache must be 5D (B,C,T,H,W); got {tuple(cache.shape)}")
#     b, c, t, h, w = cache.shape

#     # 你的 normalize_flow：最后确保得到 (H,W,2)，且 dx,dy 仍是“像素单位”
#     flow_hw2 = normalize_flow(flow, int(h), int(w), device=cache.device)  # (H,W,2), float32/...
#     flow_hw2 = flow_hw2.to(torch.float32)

#     # base grid in pixel coords
#     yy = torch.arange(h, device=cache.device, dtype=torch.float32).view(h, 1).expand(h, w)
#     xx = torch.arange(w, device=cache.device, dtype=torch.float32).view(1, w).expand(h, w)

#     sample_x = xx + flow_hw2[..., 0]
#     sample_y = yy + flow_hw2[..., 1]

#     # ===== 归一化：必须和 align_corners 对齐 =====
#     # 你现在用的是 align_corners=True，所以应该用 (w-1)/(h-1) 的版本（更标准）
#     if align_corners:
#         w_denom = float(max(w - 1, 1))
#         h_denom = float(max(h - 1, 1))
#         x_grid = 2.0 * sample_x / w_denom - 1.0
#         y_grid = 2.0 * sample_y / h_denom - 1.0
#     else:
#         w_denom = float(max(w, 1))
#         h_denom = float(max(h, 1))
#         x_grid = (2.0 * sample_x + 1.0) / w_denom - 1.0
#         y_grid = (2.0 * sample_y + 1.0) / h_denom - 1.0

#     # z 维 identity：t_idx -> [-1,1]
#     tt = torch.arange(t, device=cache.device, dtype=torch.float32)  # (T,)
#     if align_corners:
#         t_denom = float(max(t - 1, 1))
#         z_grid_t = 2.0 * tt / t_denom - 1.0  # (T,)
#     else:
#         t_denom = float(max(t, 1))
#         z_grid_t = (2.0 * tt + 1.0) / t_denom - 1.0

#     # 扩维到 (T,H,W)
#     z_grid = z_grid_t.view(t, 1, 1).expand(t, h, w)
#     x_grid_t = x_grid.unsqueeze(0).expand(t, h, w)
#     y_grid_t = y_grid.unsqueeze(0).expand(t, h, w)

#     # stack成 (T,H,W,3) 注意顺序是 (x,y,z)
#     grid_thw3 = torch.stack([x_grid_t, y_grid_t, z_grid], dim=-1)  # (T,H,W,3)

#     # 扩 batch: (B,T,H,W,3)
#     grid_bthw3 = grid_thw3.unsqueeze(0).expand(b, t, h, w, 3).to(dtype=cache.dtype)



#     grid_bthw3 = grid_thw3.unsqueeze(0).expand(b, t, h, w, 3).to(dtype=torch.float32)
#     cache = cache.to(dtype=torch.float32)

#     # torch.cuda.synchronize()
#     # start = torch.cuda.Event(enable_timing=True)
#     # end   = torch.cuda.Event(enable_timing=True)
#     # start.record()

#     warped = F.grid_sample(
#         cache,
#         grid_bthw3,
#         mode="bilinear",
#         padding_mode="zeros",
#         align_corners=align_corners,
#     )

#     # end.record()
#     # torch.cuda.synchronize()
#     # print(f"flow float32: {start.elapsed_time(end):.2f} ms")   # ms

#     warped = warped.to(dtype=torch.bfloat16)

#     return warped
