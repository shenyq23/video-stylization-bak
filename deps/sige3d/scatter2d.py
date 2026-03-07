from typing import Optional

import torch

from utils.vae_utils.flow_cache_utils import forward_warp_cache_4d

from .base import SIGEModule3d, SIGEModuleWrapper
from .gather2d import Gather2d
from .torch_kernels import scatter2d


class Scatter2d(SIGEModule3d):
    def __init__(self, gather: Gather2d, backend: Optional[str] = None):
        """
        backend:
          - None: follow `gather.backend`
          - "torch": use `sige.nn.torch_kernels`
          - "ext"/"cuda": use compiled extension (`sige.cpu` / `sige.cuda` / `sige.mps`)
        """
        super(Scatter2d, self).__init__()
        self.gather = SIGEModuleWrapper(gather)

        # self.backend = backend if backend is not None else getattr(gather, "backend", "torch")
        # self.load_runtime_with_backend("scatter2d", backend=self.backend)
        self.output_res = None
        self.original_outputs = None

    def flow_cache(self, flow):
        if self.original_outputs is None:
            return
        self.original_outputs = forward_warp_cache_4d(self.original_outputs, flow).contiguous()

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.check_dtype(x, residual)
        self.check_dim(x, residual)

        if self.mode == "profile":
            _, c, _, _ = x.shape
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
                (b, c, *self.output_res),
                fill_value=x[0, 0, 0, 0],
                dtype=x.dtype,
                device=x.device,
            )
            if residual is not None:
                output = output + residual
            return output

        if self.mode in {"full", "nocache"}:
            output = x if residual is None else x + residual
            self.output_res = output.shape[2:]
            h, w = int(output.size(-2)), int(output.size(-1))
            if self.mode == "full" and self._cache_allowed(h, w):
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
                    "Scatter2d.original_outputs is None in sparse mode. "
                    "Run a cache-building full pass first, or route this resolution to nocache mode."
                )

            offset = self.gather.module.offset
            stride = self.gather.module.model_stride

            output = scatter2d(
                x.contiguous(),
                self.original_outputs.contiguous(),
                offset[0],
                offset[1],
                stride[0],
                stride[1],
                active_indices.contiguous(),
                None if residual is None else residual.contiguous(),
            )
            return output

        raise NotImplementedError(f"Unknown mode: {self.mode}")

