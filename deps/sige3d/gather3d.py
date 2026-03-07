from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn

# import sys
# print("\n".join(sys.path))

from utils.vae_utils.mask_utils import reduce_mask, resolve_mask_for_res
from .base import SIGEModule3d
from .torch_kernels import gather3d


class Gather3d(SIGEModule3d):
    def __init__(
        self,
        conv: nn.Conv3d,
        block_size: Union[int, Tuple[int, int]],
        offset: Optional[Union[int, Tuple[int, int]]] = None,
        activation_name: str = "identity",
        activation_first: bool = False,
        verbose: bool = True,
        rms_norm: Optional[nn.Module] = None,   # ✅ 新增
    ):
        super().__init__()

        if isinstance(block_size, int):
            block_size = (block_size, block_size)

        kernel_size = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size,) * 3
        stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride,) * 3

        # Only spatial (H,W) participates in block partitioning.
        k_h, k_w = int(kernel_size[1]), int(kernel_size[2])
        s_h, s_w = int(stride[1]), int(stride[2])

        n0 = max(block_size[0] - k_h, 0) // s_h
        n1 = max(block_size[1] - k_w, 0) // s_w
        b0 = n0 * s_h + k_h
        b1 = n1 * s_w + k_w
        if (b0, b1) != block_size:
            warnings.warn(f"Change the block size from {block_size} to {(b0, b1)}")

        self.model_stride = (s_h, s_w)
        self.kernel_size = (k_h, k_w)
        self.block_size = (b0, b1)
        self.block_stride = ((n0 + 1) * s_h, (n1 + 1) * s_w)

        if offset is None:
            spatial_padding = getattr(conv, "spatial_padding", None)
            if spatial_padding is None:
                pad = conv.padding if isinstance(conv.padding, tuple) else (conv.padding,) * 3
                spatial_padding = (int(pad[1]), int(pad[2]))
            self.offset = (int(spatial_padding[0]), int(spatial_padding[1]))
        else:
            if isinstance(offset, int):
                offset = (offset, offset)
            self.offset = (int(offset[0]), int(offset[1]))

        self.activation_name = activation_name
        self.activation_first = activation_first
        self.verbose = verbose

        self.input_res: Optional[Tuple[int, int]] = None
        self.active_indices: Optional[torch.Tensor] = None

        # rms_norm: Optional[nn.Module]
        if rms_norm is not None:
            self.rms_norm_fn = rms_norm.forward   # ✅ 只存函数
        else:
            self.rms_norm_fn = None


    def forward(
        self,
        x: torch.Tensor,
        is_cache_gather: Optional[bool] = False,
        test_gather3d = False,
    ) -> torch.Tensor:
        b, c, t, _, _ = x.shape

        if self.mode == "profile":
            if self.active_indices is None:
                raise RuntimeError("Active indices are not set for profile mode.")
            output = torch.full(
                (b * self.active_indices.size(0), c, t, *self.block_size),
                fill_value=x[0, 0, 0, 0, 0],
                dtype=x.dtype,
                device=x.device,
            )
            return output

        if self.mode == "full":
            self.input_res = (int(x.size(3)), int(x.size(4)))
            return x

        if self.mode == "nocache":
            self.input_res = (int(x.size(3)), int(x.size(4)))
            return x

        if self.mode == "sparse":
            if self.active_indices is None:
                raise RuntimeError("Active indices are not set for sparse mode.")
            
            assert self.active_indices.numel() != 0
            # torch.cuda.synchronize()
            # start = torch.cuda.Event(enable_timing=True)
            # end   = torch.cuda.Event(enable_timing=True)
            # start.record()
        
            # print("gather3d is_contiguous:", x.is_contiguous())
            x = x.contiguous()
            self.active_indices = self.active_indices.contiguous()
            
            # end.record()
            # torch.cuda.synchronize()
            # print(f"gather3d contiguous time: {start.elapsed_time(end):.2f} ms")   # ms

            # if test_gather3d:
                # print(x.shape)
                # print(self.active_indices.size(0))
                # print(self.block_size[0], self.block_size[1])
                # torch.cuda.synchronize()
                # start = torch.cuda.Event(enable_timing=True)
                # end   = torch.cuda.Event(enable_timing=True)
                # start.record()

            res = gather3d(
                # x.contiguous(),
                x,
                self.block_size[0],
                self.block_size[1],
                # self.active_indices.contiguous(),
                self.active_indices,
                self.activation_name,
                rms_norm_fn=self.rms_norm_fn,    # ✅ 改名，传函数
                is_cache_gather=is_cache_gather, # 不需要norm和激活
            )

            # if test_gather3d:
                # end.record()
                # torch.cuda.synchronize()
                # print(f"gather3d time: {start.elapsed_time(end):.2f} ms")   # ms

            return res
        raise NotImplementedError(f"Unknown mode: {self.mode}")

    def set_mask(self, masks: Dict, cache: Dict, timestamp: int):
        if self.timestamp == timestamp:
            return
        if self.mode not in {"sparse", "profile"}:
            return
        super().set_mask(masks, cache, timestamp)
        if self.input_res is None:
            raise RuntimeError("Input resolution is not set before set_mask(). Run one full forward first.")

        res = (int(self.input_res[0]), int(self.input_res[1]))
        mask, top_k_percentage = resolve_mask_for_res(masks, res)
        self.mask = mask

        top_k_tag = -1.0 if top_k_percentage is None else float(top_k_percentage)
        key = ("active_indices_3d", *res, *self.block_size, *self.block_stride, *self.offset, top_k_tag)
        active_indices = cache.get(key, None)
        if active_indices is None:
            active_indices = reduce_mask(
                mask,
                self.block_size,
                self.block_stride,
                self.offset,
                top_k_percentage=top_k_percentage,
                verbose=self.verbose,
            )
            cache[key] = active_indices
        self.active_indices = active_indices
