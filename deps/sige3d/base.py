from __future__ import annotations
import importlib
import os

import inspect
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F

# from utils.vae_utils.flow_cache_utils import forward_warp_cache_4d, forward_warp_cache_5d


class SIGEModule3d(nn.Module):
    def __init__(self, call_super: bool = True):
        if call_super:
            super().__init__()
        # self.devices: List[str] = ["cpu", "cuda", "mps"]
        self.devices: List[str] = ["cuda"]
        self.supported_dtypes = [torch.float32, torch.float16, torch.bfloat16]
        self.mode: str = "full"
        self.runtime: Dict = {}
        self.mask: Optional[torch.Tensor] = None
        self.timestamp: Optional[int] = None
        self.cache_id: int = 0
        self.sparse_update: bool = False
        # If set, only cache tensors with spatial resolution >= this (H, W).
        # Smaller resolutions should not keep persistent caches.
        self.cache_min_res: Optional[Tuple[int, int]] = None
        # If set, only cache tensors with spatial resolution <= this (H, W).
        # Larger resolutions should not keep persistent caches.
        self.cache_max_res: Optional[Tuple[int, int]] = None
        # Last seen spatial resolution (H, W) for full/nocache execution.
        self.spatial_res: Optional[Tuple[int, int]] = None

    def set_mask(self, masks: Dict, cache: Dict, timestamp: int):
        self.timestamp = timestamp

    def set_mode(self, mode: str):
        self.mode = mode

    def set_cache_id(self, cache_id: int):
        self.cache_id = cache_id

    def clear_cache(self):
        pass

    def clear_stream_cache(self):
        pass

    def load_runtime_with_backend(self, function_name: str, runtime_dict: Dict = None, backend: str = "ext"):
        """
        backend:
          - "torch": use pure PyTorch reference kernels from `sige.nn.torch_kernels`
          - "ext"/"cuda": use compiled extension modules (`sige.cpu` / `sige.cuda` / `sige.mps`)
          - "auto": prefer extensions, fall back to torch when missing
        """
        if runtime_dict is None:
            runtime_dict = self.runtime
        backend = (backend or "ext").lower()

        if backend in {"torch", "pytorch"}:
            torch_kernels = importlib.import_module("sige3d.torch_kernels")
            try:
                runtime = getattr(torch_kernels, function_name)
            except AttributeError as e:
                raise AttributeError(f"torch_kernels has no function [{function_name}]") from e
            for device in self.devices:
                runtime_dict[device] = runtime
            return runtime_dict


        # "ext" / "cuda" / "native": prefer compiled extension modules (sige.cpu / sige.cuda / sige.mps)
        for device in self.devices:
            name = "sige.%s" % device
            try:
                module = importlib.import_module(name)
                runtime = getattr(module, function_name)
                runtime_dict[device] = runtime
                if device == "mps":
                    os.environ["SIGE_METAL_LIB_PATH"] = os.path.abspath(
                        os.path.join(os.path.dirname(module.__file__), "..", "sige.metallib")
                    )
            except (ModuleNotFoundError, AttributeError):
                runtime_dict[device] = torch_runtime if torch_runtime is not None else None
        return runtime_dict

    # def set_sparse_update(self, sparse_update: bool):
        # self.sparse_update = sparse_update
        
    def check_dtype(self, *args: Optional[torch.Tensor]):
        for x in args:
            if x is None:
                continue
            if x.dtype not in self.supported_dtypes:
                raise NotImplementedError(
                    f"[{self.__class__.__name__}] does not support dtype [{x.dtype}]. "
                    f"Supported: {self.supported_dtypes}"
                )

    def check_dim(self, *args: Optional[torch.Tensor]):
        for x in args:
            if x is None:
                continue
            if x.dim() != 5 and x.dim() != 4:
                raise NotImplementedError(
                    f"[{self.__class__.__name__}] does not support input with dim [{x.dim()}]."
                )

    def _cache_allowed(self, h: int, w: int) -> bool:
        # min_res = self.cache_min_res
        # if min_res is not None:
        #     min_h, min_w = int(min_res[0]), int(min_res[1])
        #     if int(h) < min_h or int(w) < min_w:
        #         return False

        max_res = self.cache_max_res
        if max_res is not None:
            max_h, max_w = int(max_res[0]), int(max_res[1])
            if int(h) > max_h or int(w) > max_w:
                return False

        return True

    def _known_spatial_res(self) -> Optional[Tuple[int, int]]:
        """
        Best-effort spatial resolution (H, W) inference for mode routing.
        Prefers module-specific cached resolutions (e.g. Gather.input_res, Scatter.output_res),
        falling back to `self.spatial_res` when available.
        """
        in_res = getattr(self, "input_res", None)
        if in_res is not None:
            try:
                return (int(in_res[0]), int(in_res[1]))
            except Exception:
                pass

        out_res = getattr(self, "output_res", None)
        if out_res is not None:
            try:
                if len(out_res) == 2:
                    return (int(out_res[0]), int(out_res[1]))
                if len(out_res) == 3:
                    return (int(out_res[1]), int(out_res[2]))
            except Exception:
                pass

        if self.spatial_res is not None:
            return (int(self.spatial_res[0]), int(self.spatial_res[1]))
        return None


class SIGEModuleWrapper:
    def __init__(self, module: SIGEModule3d):
        self.module = module


class SIGEModel3d(nn.Module):
    def __init__(self, call_super: bool = True):
        if call_super:
            super().__init__()
        self.mode: str = "full"
        self.timestamp: int = 0

    # def set_cache_min_res(self, min_res: Optional[Tuple[int, int]]) -> None:
    #     for module in self.modules():
    #         if isinstance(module, SIGEModule3d):
    #             module.cache_min_res = min_res

    def set_cache_max_res(self, max_res: Optional[Tuple[int, int]]) -> None:
        for module in self.modules():
            if isinstance(module, SIGEModule3d):
                module.cache_max_res = max_res
           
    def set_masks(self, masks: Dict):
        self.timestamp += 1
        cache: Dict = {}
        for module in self.modules():
            if isinstance(module, SIGEModule3d) and module.mode in {"sparse", "profile"}:
                module.set_mask(masks, cache, self.timestamp)
           
    def set_mode(self, mode: str):
        self.mode = mode
        for name, module in self.named_modules():
            if isinstance(module, SIGEModule3d):
                target = mode
                if mode == "sparse" and (
                    module.cache_min_res is not None or module.cache_max_res is not None
                ):
                    res = module._known_spatial_res()
                
                    name = name.split('.')

                    if res is not None and len(name) >= 3 and name[0] == "downsamples" and name[2] == "gather2d":
                        res = (res[0] // 2, res[1] // 2)

                    # if module.__class__.__name__ == "Resample":
                        # pass

                    # _cache_allowed: honor cache_min_res / cache_max_res bounds
                    if res is not None and not module._cache_allowed(res[0], res[1]):
                        target = "nocache"
                module.set_mode(target)

    def set_sparse_update(self, sparse_update: bool):
        for module in self.modules():
            if isinstance(module, SIGEModule3d):
                module.set_sparse_update(sparse_update)

    # def flow_cache(self, flow):
    #     if flow is None:
    #         return

    #     cache_entries_4d: list[tuple[SIGEModule3d, str, torch.Tensor]] = []
    #     cache_entries_5d: list[tuple[SIGEModule3d, str, torch.Tensor]] = []
    #     fallback_modules: list[SIGEModule3d] = []

    #     for module in self.modules():
    #         if not (isinstance(module, SIGEModule3d) and getattr(module, "mode", None) == "sparse" and hasattr(module, "flow_cache")):
    #             continue

    #         supported = True
    #         local_4d: list[tuple[SIGEModule3d, str, torch.Tensor]] = []
    #         local_5d: list[tuple[SIGEModule3d, str, torch.Tensor]] = []
    #         for attr in ("original_outputs", "original_residuals"):
    #             cache = getattr(module, attr, None)
    #             if cache is None or not torch.is_tensor(cache):
    #                 continue
    #             if cache.dim() == 4:
    #                 local_4d.append((module, attr, cache))
    #             elif cache.dim() == 5:
    #                 local_5d.append((module, attr, cache))
    #             else:
    #                 supported = False
    #                 break

    #         if not supported:
    #             fallback_modules.append(module)
    #             continue

    #         cache_entries_4d.extend(local_4d)
    #         cache_entries_5d.extend(local_5d)

    #     # Fallback for unknown module-specific cache behavior.
    #     for module in fallback_modules:
    #         module.flow_cache(flow)

    #     if not cache_entries_4d and not cache_entries_5d:
    #         return

    #     # 根据环境变量或当前设备的可用显存，自动推断一个“本批次最多允许使用的字节数预算”。
    #     def _infer_batch_bytes(device: torch.device) -> int:
    #         env = os.getenv("SIGE_FLOW_CACHE_BATCH_BYTES")
    #         if env:
    #             try:
    #                 value = int(env)
    #             except ValueError:
    #                 value = 0
    #             if value > 0:
    #                 return value

    #         if device.type == "cuda" and torch.cuda.is_available():
    #             free_bytes, _ = torch.cuda.mem_get_info(device)
    #             budget = max(int(free_bytes * 0.20), 1 * 1024 * 1024)
    #             return min(budget, int(free_bytes * 0.45))

    #         return 256 * 1024 * 1024

    #     def _iter_batches(entries: list[tuple[SIGEModule3d, str, torch.Tensor]], max_bytes: int):
    #         batch: list[tuple[SIGEModule3d, str, torch.Tensor]] = []
    #         batch_bytes = 0
    #         for item in entries:
    #             cache = item[2]
    #             item_bytes = int(cache.numel()) * int(cache.element_size())
    #             # 如果当前 batch 里已经有东西了，并且再把当前这个 item 放进去就会超过 max_bytes，
    #             # 那就先把当前 batch 交出去处理。
    #             if batch and batch_bytes + item_bytes > max_bytes:
    #                 yield batch
    #                 batch = []
    #                 batch_bytes = 0
    #             batch.append(item)
    #             batch_bytes += item_bytes
    #         if batch:
    #             yield batch


    #     # 4D caches: (B,C,H,W) -> concat on B.
    #     groups_4d: dict[tuple[torch.dtype, torch.device, int, int, int], list[tuple[SIGEModule3d, str, torch.Tensor]]] = {}
    #     for module, attr, cache in cache_entries_4d:
    #         _, c, h, w = cache.shape
    #         key = (cache.dtype, cache.device, int(c), int(h), int(w))
    #         groups_4d.setdefault(key, []).append((module, attr, cache))

    #     for (dtype, device, c, h, w), entries in groups_4d.items():
    #         max_bytes = _infer_batch_bytes(device)
    #         for batch in _iter_batches(entries, max_bytes):
    #             if len(batch) == 1:
    #                 module, attr, cache = batch[0]
    #                 setattr(module, attr, forward_warp_cache_4d(cache, flow).contiguous())
    #                 continue
                
    #             # setattr(module, attr, value)
    #             # 等价于：module.attr = value
    #             b_sizes = [int(cache.shape[0]) for _, _, cache in batch]
    #             batched = torch.cat([cache for _, _, cache in batch], dim=0)
    #             warped = forward_warp_cache_4d(batched, flow).contiguous()
    #             for (module, attr, _), chunk in zip(batch, warped.split(b_sizes, dim=0)):
    #                 setattr(module, attr, chunk)

    #     # 5D caches: (B,C,T,H,W) -> concat on T (same B,C,H,W), split back by T.
    #     groups_5d: dict[
    #         tuple[torch.dtype, torch.device, int, int, int, int, int],
    #         list[tuple[SIGEModule3d, str, torch.Tensor]],
    #     ] = {}
    #     for module, attr, cache in cache_entries_5d:
    #         b, c, t, h, w = cache.shape
    #         key = (cache.dtype, cache.device, int(b), int(c), int(t), int(h), int(w))
    #         groups_5d.setdefault(key, []).append((module, attr, cache))

    #     for (dtype, device, b, c, t, h, w), entries in groups_5d.items():
    #         max_bytes = _infer_batch_bytes(device)
    #         for batch in _iter_batches(entries, max_bytes):
    #             if len(batch) == 1:
    #                 module, attr, cache = batch[0]
    #                 setattr(module, attr, forward_warp_cache_5d(cache, flow).contiguous())
    #                 continue

    #             t_sizes = [int(cache.shape[2]) for _, _, cache in batch]
    #             batched = torch.cat([cache for _, _, cache in batch], dim=2)
    #             warped = forward_warp_cache_5d(batched, flow).contiguous()
    #             for (module, attr, _), chunk in zip(batch, warped.split(t_sizes, dim=2)):
    #                 setattr(module, attr, chunk)

    def flow_cache(self, flow):
        for name, module in self.named_modules():
            if isinstance(module, SIGEModule3d) and hasattr(module, "flow_cache"):
                # print(f"{cnt}: [flow_cache] {name} | {module.__class__.__name__} | {module}")
                module.flow_cache(flow)
        return



class SIGECausalConv3d(nn.Conv3d, SIGEModule3d):
    """
    full: pad(T causal) + pad(HW) -> conv3d(padding=0)
    sparse/profile: pad(T causal only) -> conv3d(padding=0)
    """

    def __init__(self, *args, **kwargs):
        nn.Conv3d.__init__(self, *args, **kwargs)
        SIGEModule3d.__init__(self, call_super=False)

        # 原始 conv3d 的 padding (T,H,W)
        p_t, p_h, p_w = self.padding
        
        # 供 Gather3d 推断 spatial offset（等价于原始 padding(H,W)）
        self.spatial_padding = (int(p_h), int(p_w))

        # spatial pad for F.pad order: (Wl, Wr, Hl, Hr, Tl, Tr)
        # mode = full的时候才需要
        self._spatial_pad = (p_w, p_w, p_h, p_h, 0, 0)

        # causal temporal pad: only pad "past" (left) side
        self._temporal_pad = (0, 0, 0, 0, 2 * p_t, 0)

        # 关闭 conv3d 内部 padding（我们统一用 F.pad）
        self.padding = (0, 0, 0)

    def _apply_temporal_pad(self, x: torch.Tensor, cache_x: torch.Tensor = None) -> torch.Tensor:
        # temporal pad is (Wl,Wr,Hl,Hr,Tl,Tr) but only Tl/Tr are non-zero
        pad = list(self._temporal_pad)

        if cache_x is not None and pad[4] > 0:
            cache_x = cache_x.to(x.device)
            if cache_x.shape[:2] + cache_x.shape[3:] != x.shape[:2] + x.shape[3:]:
                pass
            x = torch.cat([cache_x, x], dim=2)  # dim=2 is T
            # cache 覆盖掉一部分“过去pad需求”
            pad[4] -= cache_x.shape[2]

        return F.pad(x, pad)

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor = None) -> torch.Tensor:
        if self.mode in ["full", "nocache"]:
            self.spatial_res = (int(x.size(3)), int(x.size(4)))
            # full：整图需要 HW padding + 因果 T padding
            x = self._apply_temporal_pad(x, cache_x)
            x = F.pad(x, self._spatial_pad)

            return super(SIGECausalConv3d, self).forward(x)  # self.padding=0

        elif self.mode in ["sparse", "profile"]:
            # sparse：假设 gather 已经提供了 HW halo，所以只做 T 因果 padding
            x = self._apply_temporal_pad(x, cache_x)

            return F.conv3d(    # pylint: disable=not-callable
                x, self.weight, self.bias,
                self.stride, (0, 0, 0),
                self.dilation, self.groups
            )

        else:
            raise NotImplementedError(f"Unknown mode: {self.mode}")


class SIGEConv2d(nn.Conv2d, SIGEModule3d):
    def __init__(self, *args, **kwargs):
        nn.Conv2d.__init__(self, *args, **kwargs)
        SIGEModule3d.__init__(self, call_super=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode in ["full", "nocache"]:
            self.spatial_res = (int(x.size(2)), int(x.size(3)))
            output = super(SIGEConv2d, self).forward(x)
        elif self.mode in ["sparse", "profile"]:
            output = F.conv2d(x, self.weight, self.bias, self.stride, (0, 0), self.dilation, self.groups) # pylint: disable=not-callable
        else:
            raise NotImplementedError("Unknown mode: %s" % self.mode)
        return output
