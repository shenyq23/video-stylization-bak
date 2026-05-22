from __future__ import annotations


import torch
from typing import Dict, Optional, Tuple, Union

import numpy as np
from torch.nn import functional as F


GATHER_BLOCK_OCC_KEY = "__gather_block_occ__"
GATHER_BLOCK_TOPK_KEY = "__gather_block_topk__"


def compute_cdf_ratio(
    occ_map: torch.Tensor,
    coverage_rho: float = 0.7,
    r_min: float = 0.08,
    r_max: float = 0.30,
) -> float:
    """MotionFlow paper §3.4 CDF-based adaptive sparsity ratio.

    Sort tokens by motion magnitude descending, find the smallest k such that
    the cumulative motion covers `coverage_rho` of total. Return k / N clamped
    to [r_min, r_max].
    """
    flat = occ_map.detach().float().abs().flatten()
    n = flat.numel()
    if n == 0:
        return r_max
    total = flat.sum()
    if not torch.isfinite(total) or total.item() <= 0.0:
        return r_max
    sorted_vals, _ = torch.sort(flat, descending=True)
    cumsum = torch.cumsum(sorted_vals, dim=0)
    threshold = float(coverage_rho) * float(total.item())
    # smallest k such that cumsum[k-1] >= threshold; searchsorted returns first
    # index where cumsum >= threshold; add 1 to convert from index to count.
    idx = int(torch.searchsorted(cumsum, torch.tensor(threshold, device=cumsum.device)).item())
    k = min(max(idx + 1, 1), n)
    ratio = k / float(n)
    return float(max(r_min, min(r_max, ratio)))


def build_gather_block_masks(
    raw_occ_map: torch.Tensor,
    top_k_percentage: float,
    *,
    adaptive: bool = False,
    cdf_coverage: float = 0.7,
    r_min: float = 0.08,
    r_max: float = 0.30,
) -> Dict:
    if adaptive:
        # Squeeze any leading 1-dims to get a 2D HxW map for CDF computation.
        occ_for_cdf = raw_occ_map
        while occ_for_cdf.dim() > 2 and occ_for_cdf.size(0) == 1:
            occ_for_cdf = occ_for_cdf.squeeze(0)
        top_k_percentage = compute_cdf_ratio(
            occ_for_cdf, coverage_rho=cdf_coverage, r_min=r_min, r_max=r_max
        )
    return {
        GATHER_BLOCK_OCC_KEY: raw_occ_map,
        GATHER_BLOCK_TOPK_KEY: float(top_k_percentage),
    }


def resolve_mask_for_res(masks: Dict, res: Tuple[int, int]) -> tuple[torch.Tensor, Optional[float]]:
    mask = masks.get(res, None)
    if mask is not None:
        return mask, None

    raw_occ_map = masks.get(GATHER_BLOCK_OCC_KEY, None)
    if raw_occ_map is None:
        raise KeyError(f"Mask for resolution {res} is missing.")

    if not torch.is_tensor(raw_occ_map):
        raise TypeError(f"{GATHER_BLOCK_OCC_KEY} must be a torch.Tensor.")

    if raw_occ_map.dim() == 2:
        occ_4d = raw_occ_map.unsqueeze(0).unsqueeze(0)
    elif raw_occ_map.dim() == 3 and raw_occ_map.size(0) == 1:
        occ_4d = raw_occ_map.unsqueeze(0)
    elif raw_occ_map.dim() == 4:
        occ_4d = raw_occ_map
    else:
        raise ValueError(
            f"Unsupported occlusion map shape for gather_block: {tuple(raw_occ_map.shape)}"
        )

    if occ_4d.size(0) != 1 or occ_4d.size(1) != 1:
        raise ValueError(
            f"Expected occlusion map with shape [1,1,H,W] (or squeezable), got {tuple(occ_4d.shape)}"
        )

    occ_4d = occ_4d.to(torch.float32)
    h, w = int(res[0]), int(res[1])
    if occ_4d.size(2) != h or occ_4d.size(3) != w:
        occ_4d = F.interpolate(occ_4d, size=(h, w), mode="bilinear", align_corners=False)

    top_k_percentage = float(masks.get(GATHER_BLOCK_TOPK_KEY, 0.0))
    return occ_4d[0, 0], top_k_percentage


def reduce_mask(
    mask: torch.Tensor,
    block_size: Optional[Union[int, Tuple[int, int]]],
    stride: Optional[Union[int, Tuple[int, int]]],
    padding: Optional[Union[int, Tuple[int, int]]],
    top_k_percentage: Optional[float] = None,
    verbose: bool = True,
) -> Optional[torch.Tensor]:
    if block_size is None or stride is None or padding is None:
        return None
    else:
        if isinstance(block_size, int):
            block_size = (block_size, block_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(stride, int):
            stride = (stride, stride)
        H, W = mask.shape

        # Max Pooling only supports float tensor
        mask = mask.view(1, 1, H, W).to(torch.float32)
        # 补充的是 0（零元素）
        mask = F.pad(mask, (padding[1], block_size[1], padding[0], block_size[0]))
        
        if top_k_percentage is None:
            # 默认逻辑：二值掩码下，只要 block 内有一个活跃像素就选中该 block
            mask_pooled = F.max_pool2d(mask, block_size, stride)[0, 0] > 0.0
            active_indices = torch.nonzero(mask_pooled)
            total = mask_pooled.numel()
        else:
            # gather_block 逻辑：按 block 内残差和排序，固定选 top-k 个 block
            block_area = int(block_size[0]) * int(block_size[1])
            block_scores = F.avg_pool2d(mask, block_size, stride)[0, 0] * float(block_area)
            flat_scores = block_scores.reshape(-1)
            total = flat_scores.numel()

            block_cnt = max(1, int(total * top_k_percentage))

            topk_idx = torch.topk(flat_scores, k=block_cnt, largest=True, sorted=False).indices
            width = block_scores.size(1)
            idx_h = torch.div(topk_idx, width, rounding_mode="floor")
            idx_w = topk_idx % width
            active_indices = torch.stack([idx_h, idx_w], dim=1)

        if active_indices.numel() > 0:
            active_indices[:, 0] = stride[0] * active_indices[:, 0] - padding[0]
            active_indices[:, 1] = stride[1] * active_indices[:, 1] - padding[1]
        # if verbose:
            # num_active = active_indices.shape[0]
            # print("Block Sparsity: %d/%d=%.2f%%" % (num_active, total, 100 * num_active / total))
        return active_indices.to(torch.int32).contiguous()


# 参数 mask 可以是 torch.Tensor，也可以是 numpy.ndarray
def dilate_mask(
    mask: Union[torch.Tensor, np.ndarray], dilation: Union[int, Tuple[int, int]]  # [C, H, W] or [H, W]
) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    if dilation[0] <= 0 and dilation[1] <= 0:
        return mask

    if isinstance(mask, torch.Tensor):
        ret = mask.clone()
    else:
        assert isinstance(mask, np.ndarray)
        ret = mask.copy()

    if len(ret.shape) == 2:
        for i in range(1, dilation[0] + 1):
            ret[:-i] |= mask[i:]
            ret[i:] |= mask[:-i]
        for i in range(1, dilation[1] + 1):
            ret[:, :-i] |= mask[:, i:]
            ret[:, i:] |= mask[:, :-i]
    elif len(ret.shape) == 3:
        for i in range(1, dilation + 1):
            ret[:, :-i] |= mask[:, i:]
            ret[:, i:] |= mask[:, :-i]
        for i in range(1, dilation[1] + 1):
            ret[:, :, :-i] |= mask[:, :, i:]
            ret[:, :, i:] |= mask[:, :, :-i]
    else:
        raise NotImplementedError("Unknown mask dimension [%d]!!!" % mask.dim())
    return ret


def compute_difference_mask(tensor1: torch.Tensor, tensor2: torch.Tensor, eps: float = 2e-2) -> torch.Tensor:
    difference = torch.abs(tensor1 - tensor2)
    mask = difference > eps
    if mask.dim() == 2:  # [H, W]
        return mask
    elif mask.dim() == 3:  # [C, H, W]
        return torch.any(mask, 0)
    elif mask.dim() == 4:  # [B, C, H, W]
        assert mask.shape[0] == 1
        return torch.any(mask[0], 0)
    else:
        raise NotImplementedError("Unknown mask dimension [%d]!!!" % mask.dim())


def downsample_mask(
    mask: torch.Tensor,
    min_res: Union[int, Tuple[int, int]] = 4,
    dilation: Union[int, Tuple[int, int]] = 1,
    threshold: float = 0.3,
    eps: float = 1e-3,
) -> Dict[Tuple[int, int], torch.Tensor]:
    assert mask.dim() == 2
    H, W = mask.shape
    if isinstance(min_res, int):
        min_h = min_res
        min_w = min_res
    else:
        min_h, min_w = min_res
    h = H
    w = W

    masks = {}
    interpolated_mask = mask.view(1, 1, H, W).float()
    while True:
        t = min(threshold, interpolated_mask.max() - eps)
        # t = threshold
        sparsity_mask = interpolated_mask[0, 0] > t
        sparsity_mask = dilate_mask(sparsity_mask, dilation)
        masks[(h, w)] = sparsity_mask
        h //= 2
        w //= 2
        if h < min_h and w < min_w:
            break

        interpolated_mask = F.interpolate(interpolated_mask, (h, w), mode="bilinear", align_corners=False)
    return masks



# def compute_sdedit_masks(
#     init_img: torch.Tensor,
#     edited_img: torch.Tensor,
#     *,
#     min_res: Tuple[int, int] = (4, 4),
# ) -> tuple[Dict[Tuple[int, int], torch.Tensor], Dict[Tuple[int, int], torch.Tensor], torch.Tensor]:
#     """
#     Replicates `stable-diffusion/runners/sdedit_runner.py` mask logic.

#     Inputs:
#       - init_img / edited_img: [1, 3, H, W] in [-1, 1]

#     Returns:
#       - masks_enc: dict[(h,w)] -> bool mask, encoder setting
#       - masks_dec: dict[(h,w)] -> bool mask, decoder setting
#       - diff_mask: [H, W] bool
#     """
#     diff_mask = compute_difference_mask(init_img, edited_img)  # [H,W]

#     # Encoder masks
#     diff_enc = dilate_mask(diff_mask, 5)
#     masks_enc = downsample_mask(diff_enc, min_res=(4, 4), dilation=1)

#     # Decoder masks
#     diff_dec = dilate_mask(diff_mask, 40)
#     masks_dec = downsample_mask(diff_dec, min_res=min_res, dilation=0)

#     return masks_enc, masks_dec, diff_mask
