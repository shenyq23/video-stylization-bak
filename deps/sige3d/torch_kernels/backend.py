from __future__ import annotations

import os
from typing import Literal

KernelBackend = Literal["pytorch", "cuda"]

_KERNEL_BACKEND: KernelBackend = "pytorch"


def _normalize_backend(value: str) -> KernelBackend:
    v = (value or "").strip().lower()
    if v in {"pytorch", "torch", "pt"}:
        return "pytorch"
    if v in {"cuda", "ext"}:
        return "cuda"
    raise ValueError(f"Unknown kernel backend: {value!r} (expected: 'PyTorch' or 'CUDA')")


def set_kernel_backend(backend: str) -> None:
    """
    Set SIGE3D kernel backend.

    - "pytorch": always use the torch vectorized implementations.
    - "cuda": use the custom CUDA extension when inputs are CUDA (falls back to torch on failure).
    """
    global _KERNEL_BACKEND  # noqa: PLW0603
    _KERNEL_BACKEND = _normalize_backend(backend)


def get_kernel_backend() -> KernelBackend:
    """Return current SIGE3D kernel backend."""
    return _KERNEL_BACKEND


def use_cuda_kernels() -> bool:
    """Whether CUDA extension kernels are enabled by config."""
    return _KERNEL_BACKEND == "cuda"


# Init from env (optional); CLI can override via `set_kernel_backend()`.
_env_backend = os.environ.get("SIGE3D_KERNEL_BACKEND")
if _env_backend:
    try:
        set_kernel_backend(_env_backend)
    except Exception:
        # Keep default "pytorch" on invalid env values.
        _KERNEL_BACKEND = "pytorch"

