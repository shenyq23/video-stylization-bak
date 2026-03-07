
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from utils.vae_utils.flow_cache_utils import forward_warp_cache_5d

from .base import SIGEModule3d, SIGEModuleWrapper
from .gather3d import Gather3d
from .torch_kernels import get_scatter_map, scatter3d, scatter_gather3d


class ScatterGather3d(SIGEModule3d):
    def __init__(
        self,
        gather: Gather3d,
        activation_name: str = "identity",
        activation_first: bool = False,
        rms_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.gather = SIGEModuleWrapper(gather)
        self.activation_name = activation_name
        self.activation_first = activation_first

        self.scatter_map: torch.Tensor | None = None
        self.output_res = None
        self.original_outputs = None

        if rms_norm is not None:
            self.rms_norm_fn = rms_norm.forward
        else:
            self.rms_norm_fn = None

    def flow_cache(self, flow):
        if self.original_outputs is None:
            return
        self.original_outputs = forward_warp_cache_5d(self.original_outputs, flow).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_indices = self.gather.module.active_indices
        block_size = self.gather.module.block_size

        if self.mode == "profile":
            if active_indices is None:
                raise RuntimeError("Active indices are not set for profile mode.")
            num_active = int(active_indices.size(0))
            if num_active <= 0:
                raise RuntimeError("Active indices are empty for profile mode.")
            b = int(x.size(0)) // num_active
            t = int(x.size(2))
            _, c, _, _, _ = x.shape
            return torch.full(
                (b * num_active, c, t, *block_size),
                fill_value=x[0, 0, 0, 0, 0],
                dtype=x.dtype,
                device=x.device,
            )

        if self.mode in {"full", "nocache"}:
            output = x
            self.output_res = output.shape[2:]  # (T,H,W)
            h, w = int(output.size(-2)), int(output.size(-1))
            if self.mode == "full" and self._cache_allowed(h, w):
                self.original_outputs = output.contiguous()
                # pass
            else:
                self.original_outputs = None
            return output

        if self.mode == "sparse":
            if self.scatter_map is None:
                raise RuntimeError("scatter_map is not set. Call set_masks() first.")
            if self.original_outputs is None:
                raise RuntimeError(
                    "ScatterGather3d.original_outputs is None in sparse mode. "
                    "Run a cache-building full pass first, or route this resolution to nocache mode."
                )
            if active_indices is None:
                raise RuntimeError("Active indices are not set for sparse mode.")

            output = scatter_gather3d(
                x.contiguous(),
                self.original_outputs.contiguous(),
                block_size[0],
                block_size[1],
                active_indices.contiguous(),
                self.scatter_map.contiguous(),
                self.activation_name,
                rms_norm_fn=self.rms_norm_fn,
            )

            # Update cached original_outputs in-place.
            scatter3d(
                x.contiguous(),
                self.original_outputs.contiguous(),
                self.gather.module.offset[0],
                self.gather.module.offset[1],
                self.gather.module.model_stride[0],
                self.gather.module.model_stride[1],
                active_indices.contiguous(),
                None,
            )
            return output

        raise NotImplementedError(f"Unknown mode: {self.mode}")

    def set_mask(self, masks: Dict, cache: Dict, timestamp: int):
        if self.timestamp == timestamp:
            return
        if self.mode not in {"sparse", "profile"}:
            return
        super().set_mask(masks, cache, timestamp)

        self.gather.module.set_mask(masks, cache, timestamp)

        mask = self.gather.module.mask
        if mask is None:
            raise RuntimeError("Gather3d.mask is not set.")
        h, w = int(mask.size(0)), int(mask.size(1))
        block_size = self.gather.module.block_size
        kernel_size = self.gather.module.kernel_size
        offset = self.gather.module.offset
        stride = self.gather.module.model_stride

        key = ("scatter_map_3d", h, w, *block_size, *kernel_size, *offset, *stride)
        scatter_map = cache.get(key, None)
        if scatter_map is None:
            scatter_map = get_scatter_map(
                h,
                w,
                block_size[0],
                block_size[1],
                kernel_size[0],
                kernel_size[1],
                offset[0],
                offset[1],
                stride[0],
                stride[1],
                self.gather.module.active_indices,
            )
            cache[key] = scatter_map
        self.scatter_map = scatter_map

