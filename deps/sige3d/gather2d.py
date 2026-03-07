import warnings
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
import numpy as np

from utils.vae_utils.mask_utils import reduce_mask, resolve_mask_for_res
from .base import SIGEModule3d
from .activation import activation
from .torch_kernels import gather2d


class Gather2d(SIGEModule3d):
    """
    backend:
      - "torch": use PyTorch reference kernels in `sige/cuda/*_kernel.py` via `sige.nn.torch_kernels`
      - "ext"/"cuda": use compiled extension (`sige.cpu` / `sige.cuda` / `sige.mps`)
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        block_size: Union[int, Tuple[int, int]],
        offset: Optional[Union[int, Tuple[int, int]]] = None,
        activation_name: str = "identity",
        verbose: bool = True,
    ):
        super(Gather2d, self).__init__()
        if isinstance(block_size, int):
            block_size = (block_size, block_size)

        # 下取整
        n0 = max(block_size[0] - conv.kernel_size[0], 0) // conv.stride[0]
        n1 = max(block_size[1] - conv.kernel_size[1], 0) // conv.stride[1]
        b0 = n0 * conv.stride[0] + conv.kernel_size[0]
        b1 = n1 * conv.stride[1] + conv.kernel_size[1]
        if (b0, b1) != block_size:
            warnings.warn("Change the block size from (%d, %d) to (%d, %d)" % (*block_size, b0, b1))

        self.model_stride = conv.stride
        self.kernel_size = conv.kernel_size

        self.block_size = (b0, b1)

        # block_stride的含义
        # 当你在输入上移动一个 block，到下一个 block 的起始位置时，
        # 在原始特征图上，要跨过多少像素，才能刚好对应“下一个输出位置块”
        self.block_stride = ((n0 + 1) * conv.stride[0], (n1 + 1) * conv.stride[1])


        if offset is None:
            self.offset = conv.padding
        else:
            if isinstance(offset, int):
                offset = (offset, offset)
            self.offset = offset
        self.activation_name = activation_name
        self.verbose = verbose

        # self.backend = backend
        # self.load_runtime_with_backend("gather2d", backend=self.backend)

        self.input_res: Optional[Tuple[int, int]] = None
        self.active_indices: Optional[torch.Tensor] = None

        self.time_list = []

    def forward(self, x: torch.Tensor, time_list: list) -> torch.Tensor:
        b, c, h, w = x.shape

        if self.mode == "profile":
            output = torch.full(
                (b * self.active_indices.size(0), c, *self.block_size),
                fill_value=x[0, 0, 0, 0],
                dtype=x.dtype,
                device=x.device,
            )  # create a dummy gather output depending on the input for profiling

        elif self.mode in ["full", "nocache"]:
            self.input_res = (int(x.size(2)), int(x.size(3)))
            output = x
            
        elif self.mode == "sparse":
            # device = x.device.type
            # runtime = self.runtime[device]
            # assert runtime is not None
            # active_indices = self.active_indices
            # assert active_indices is not None


            # torch.cuda.synchronize()
            # start = torch.cuda.Event(enable_timing=True)
            # end   = torch.cuda.Event(enable_timing=True)
            # start.record()

            # print("gather2d is_contiguous:", x.is_contiguous())
            # x = x.contiguous()
            # print("active_indices is_contiguous:", self.active_indices.is_contiguous())
            # self.active_indices = self.active_indices.contiguous()
            
            # end.record()
            # torch.cuda.synchronize()
            # print(f"gather2d time1111111: {start.elapsed_time(end):.2f} ms")   # ms
           
            # print("*" * 40)
            # torch.cuda.synchronize()
            # start = torch.cuda.Event(enable_timing=True)
            # end   = torch.cuda.Event(enable_timing=True)
            # start.record()

            output = gather2d(
                # x.contiguous(),
                # x[[0]],
                x,
                self.block_size[0],
                self.block_size[1],
                # self.active_indices.contiguous(),
                self.active_indices,
            )

            # end.record()
            # torch.cuda.synchronize()
            # time = start.elapsed_time(end)
            # time_list.append(time)
            # print(f"gather2d time1111111: {time:.2f} ms")   # ms
            # if len(time_list) % 6 == 0:
                # print(f"{np.mean(time_list):.2f}")
            # print("*" * 40)
            
            
        else:
            raise NotImplementedError("Unknown mode: [%s]!!!" % self.mode)
        return output

    def set_mask(self, masks: Dict, cache: Dict, timestamp: int):
        if self.timestamp != timestamp:
            super(Gather2d, self).set_mask(masks, cache, timestamp)
            assert self.input_res is not None
            if self.mode not in {"sparse", "profile"}:
                return
            res = (int(self.input_res[0]), int(self.input_res[1]))
            mask, top_k_percentage = resolve_mask_for_res(masks, res)
            self.mask = mask
            top_k_tag = -1.0 if top_k_percentage is None else float(top_k_percentage)
            key = ("active_indices", *res, *self.block_size, *self.block_stride, *self.offset, top_k_tag)
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
                # if self.backend.lower() in {"torch", "pytorch"} and active_indices is not None:
                    # active_indices = active_indices.detach().cpu()
                cache[key] = active_indices
            self.active_indices = active_indices
