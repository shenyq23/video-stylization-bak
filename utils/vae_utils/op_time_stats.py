from __future__ import annotations

import functools
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, Type

import torch


@dataclass
class _OpStats:
    total_s: float = 0.0
    calls: int = 0


_STATS: Dict[str, _OpStats] = defaultdict(_OpStats)
_PATCHED: set[Tuple[Type[Any], str]] = set()
_NESTING: int = 0


def _has_cuda_tensor(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    for v in args:
        if isinstance(v, torch.Tensor) and v.is_cuda:
            return True
    for v in kwargs.values():
        if isinstance(v, torch.Tensor) and v.is_cuda:
            return True
    return False


def _time_call_s(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Tuple[float, Any]:
    use_cuda_sync = torch.cuda.is_available() and _has_cuda_tensor(args, kwargs)
    if use_cuda_sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    if use_cuda_sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0), out


def _wrap_module_forward(cls: Type[Any], op_key: str) -> None:
    global _PATCHED
    key = (cls, "forward")
    if key in _PATCHED:
        return

    orig_forward = cls.forward

    @functools.wraps(orig_forward)
    def wrapped_forward(self, *args, **kwargs):
        global _NESTING

        if getattr(self, "mode", None) != "sparse":
            return orig_forward(self, *args, **kwargs)

        if _NESTING > 0:
            _NESTING += 1
            try:
                return orig_forward(self, *args, **kwargs)
            finally:
                _NESTING -= 1

        _NESTING += 1
        try:
            elapsed_s, out = _time_call_s(orig_forward, (self, *args), kwargs)
            s = _STATS[op_key]
            s.total_s += float(elapsed_s)
            s.calls += 1
            return out
        finally:
            _NESTING -= 1

    cls.forward = wrapped_forward  # type: ignore[assignment]
    _PATCHED.add(key)


def install_sige_op_time_stats() -> None:
    """
    Monkey-patch SIGE gather/scatter/scatter_gather module forwards to accumulate total time.
    This does NOT change any model logic and works for both torch and extension backends.
    """

    from deps.sige3d.gather2d import Gather2d
    from deps.sige3d.gather3d import Gather3d
    from deps.sige3d.scatter2d import Scatter2d
    from deps.sige3d.scatter3d import Scatter3d, ScatterWithBlockResidual3d
    from deps.sige3d.scatter_gather3d import ScatterGather3d

    _wrap_module_forward(Gather2d, "gather2d")
    _wrap_module_forward(Gather3d, "gather3d")

    _wrap_module_forward(Scatter2d, "scatter2d")
    _wrap_module_forward(Scatter3d, "scatter3d")
    
    _wrap_module_forward(ScatterWithBlockResidual3d, "scatter3d")

    _wrap_module_forward(ScatterGather3d, "scatter_gather3d")


def get_sige_op_time_stats() -> Dict[str, Tuple[float, int]]:
    return {k: (v.total_s, v.calls) for k, v in _STATS.items()}


def print_sige_op_time_stats(prefix: str = "[SIGE op time]") -> None:
    stats = get_sige_op_time_stats()
    keys = ("gather2d", "gather3d", "scatter2d","scatter3d", "scatter_gather3d")
    for k in keys:
    # for k in stats.keys():
        total_s, calls = stats.get(k, (0.0, 0))
        avg = total_s * 1000 / calls if calls > 0 else 0.0
        print(f"{prefix} {k}: {total_s:.2f}s (calls={calls}) | every time: {avg:.2f}ms")