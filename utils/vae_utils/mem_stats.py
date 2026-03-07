from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn


def format_bytes(num_bytes: int) -> str:
    num_bytes = int(num_bytes)
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    unit = units[0]
    for u in units[1:]:
        if size < 1024.0:
            break
        size /= 1024.0
        unit = u
    if unit == "B":     # 说明：字节数 < 1024，没有做过除法
        return f"{num_bytes} {unit}"
    return f"{size:.2f} {unit}"


def _tensor_storage_nbytes(t: torch.Tensor, seen: set[tuple[str, int]]) -> int:
    if t is None:
        return 0
    if not torch.is_tensor(t):
        return 0
    try:
        storage = t.untyped_storage()
        key = (str(t.device), int(storage.data_ptr()))
        if key in seen:
            return 0
        seen.add(key)
        return int(storage.nbytes())
    except Exception:
        return int(t.numel() * t.element_size())


def obj_nbytes(obj: Any, *, seen: set[tuple[str, int]] | None = None) -> int:
    if obj is None:
        return 0
    if seen is None:
        seen = set()

    if torch.is_tensor(obj):
        return _tensor_storage_nbytes(obj, seen)    # 返回的是字节数

    if isinstance(obj, dict):
        return sum(obj_nbytes(v, seen=seen) for v in obj.values())

    if isinstance(obj, (list, tuple, set)):
        return sum(obj_nbytes(v, seen=seen) for v in obj)

    return 0


def feat_map_nbytes(model: Any) -> int:
    return obj_nbytes(getattr(model, "_enc_feat_map", None)) + \
           obj_nbytes(getattr(model, "_dec_feat_map", None))

def collect_scatter_cache_modules(model: nn.Module) -> list[nn.Module]:
    modules: list[nn.Module] = []
    for m in model.modules():
        if hasattr(m, "original_outputs") or hasattr(m, "original_residuals"):
            modules.append(m)
    return modules


def scatter_cache_nbytes(modules: Iterable[Any]) -> int:
    seen: set[tuple[str, int]] = set()
    total = 0
    for m in modules:
        total += obj_nbytes(getattr(m, "original_outputs", None), seen=seen)
        total += obj_nbytes(getattr(m, "original_residuals", None), seen=seen)
    return int(total)
