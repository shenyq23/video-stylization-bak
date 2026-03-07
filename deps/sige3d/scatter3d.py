from __future__ import annotations

from typing import Optional

import torch

from utils.vae_utils.flow_cache_utils import forward_warp_cache_5d

from .base import SIGEModule3d, SIGEModuleWrapper
from .gather3d import Gather3d
from .torch_kernels import scatter3d, scatter_with_block_residual3d


class Scatter3d(SIGEModule3d):
    def __init__(self, gather: Gather3d):
        super().__init__()
        self.gather = SIGEModuleWrapper(gather)
        self.output_res = None
        # [B, C, T, H, W]
        self.original_outputs = None

    # flow: (H, W, 2), all (dx, dy) backward flow
    def flow_cache(self, flow):
        if self.original_outputs is None:
            return
        self.original_outputs = forward_warp_cache_5d(self.original_outputs, flow).contiguous()

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.check_dtype(x, residual)
        self.check_dim(x, residual)

        if self.mode == "profile":
            _, c, t, _, _ = x.shape
            if self.output_res is None:
                raise RuntimeError("Output resolution is not set for profile mode. Run one full forward first.")
            active_indices = self.gather.module.active_indices
            if active_indices is None:
                raise RuntimeError("Active indices are not set for profile mode.")
            num_active = int(active_indices.size(0))
            if num_active <= 0:
                raise RuntimeError("Active indices are empty for profile mode.")
            b = int(x.size(0)) // num_active
            output = torch.full(
                (b, c, t, *self.output_res[1:]),
                fill_value=x[0, 0, 0, 0, 0],
                dtype=x.dtype,
                device=x.device,
            )
            if residual is not None:
                output = output + residual
            return output

        if self.mode in {"full", "nocache"}:
            output = x if residual is None else x + residual
            self.output_res = output.shape[2:]  # (T,H,W)
            h, w = int(output.size(-2)), int(output.size(-1))
            if self.mode == "full" and self._cache_allowed(h, w):
                self.original_outputs = output.contiguous()
                self.original_outputs = output.contiguous()
                # pass
            else:
                self.original_outputs = None
            return output

        if self.mode == "sparse":
            active_indices = self.gather.module.active_indices
            if active_indices is None:
                raise RuntimeError("Active indices are not set for sparse mode.")
            if self.original_outputs is None:
                raise RuntimeError(
                    "Scatter3d.original_outputs is None in sparse mode. "
                    "Run a cache-building full pass first, or route this resolution to nocache mode."
                )

            offset_h, offset_w = self.gather.module.offset
            stride_h, stride_w = self.gather.module.model_stride

            output = scatter3d(
                x.contiguous(),
                self.original_outputs.contiguous(),
                offset_h,
                offset_w,
                stride_h,
                stride_w,
                active_indices.contiguous(),
                None if residual is None else residual.contiguous(),
            )
            return output

        raise NotImplementedError(f"Unknown mode: {self.mode}")


class ScatterWithBlockResidual3d(SIGEModule3d):
    def __init__(self, main_gather: Gather3d, shortcut_gather: Gather3d):
        super().__init__()
        self.main_gather = SIGEModuleWrapper(main_gather)
        self.shortcut_gather = SIGEModuleWrapper(shortcut_gather)
        self.output_res = None
        self.original_outputs = None
        self.original_residuals = None

    def clear_cache(self):
        self.original_outputs = None
        self.original_residuals = None

    def flow_cache(self, flow):
        if self.original_outputs is None or self.original_residuals is None:
            return
        self.original_outputs = forward_warp_cache_5d(self.original_outputs, flow).contiguous()
        self.original_residuals = forward_warp_cache_5d(self.original_residuals, flow).contiguous()

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        self.check_dtype(x, residual)
        self.check_dim(x, residual)

        if self.mode == "profile":
            _, c, t, _, _ = x.shape
            if self.output_res is None:
                raise RuntimeError("Output resolution is not set for profile mode. Run one full forward first.")
            active_indices = self.main_gather.module.active_indices
            if active_indices is None:
                raise RuntimeError("Active indices are not set for profile mode.")
            num_active = int(active_indices.size(0))
            if num_active <= 0:
                raise RuntimeError("Active indices are empty for profile mode.")
            b = int(x.size(0)) // num_active
            return torch.full(
                (b, c, t, *self.output_res[1:]),
                fill_value=x[0, 0, 0, 0, 0] + residual[0, 0, 0, 0, 0],
                dtype=x.dtype,
                device=x.device,
            )

        if self.mode in {"full", "nocache"}:
            output = x + residual
            self.output_res = output.shape[2:]  # (T,H,W)
            h, w = int(output.size(-2)), int(output.size(-1))
            if self.mode == "full" and self._cache_allowed(h, w):
                self.original_outputs = output.contiguous()
                self.original_residuals = residual.contiguous()
                # pass
            else:
                self.original_outputs = None
                self.original_residuals = None
            return output

        if self.mode == "sparse":
            if self.original_outputs is None or self.original_residuals is None:
                raise RuntimeError(
                    "ScatterWithBlockResidual3d cache is missing in sparse mode. "
                    "Run a cache-building full pass first, or route this resolution to nocache mode."
                )

            offset_h, offset_w = self.main_gather.module.offset
            stride_h, stride_w = self.main_gather.module.model_stride

            output = scatter_with_block_residual3d(
                x.contiguous(),
                self.original_outputs.contiguous(),
                residual.contiguous(),
                self.original_residuals.contiguous(),
                offset_h,
                offset_w,
                stride_h,
                stride_w,
                self.main_gather.module.active_indices.contiguous(),
                self.shortcut_gather.module.active_indices.contiguous(),
            )

            # Update cached residuals in-place.
            scatter3d(
                residual.contiguous(),
                self.original_residuals.contiguous(),
                self.shortcut_gather.module.offset[0],
                self.shortcut_gather.module.offset[1],
                self.shortcut_gather.module.model_stride[0],
                self.shortcut_gather.module.model_stride[1],
                self.shortcut_gather.module.active_indices.contiguous(),
                None,
            )

            return output

        raise NotImplementedError(f"Unknown mode: {self.mode}")

