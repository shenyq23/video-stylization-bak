from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType


@lru_cache(maxsize=1)
def _load_sige3d_cuda_ext() -> ModuleType:
    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot load SIGE3D CUDA kernels.")

    this_dir = Path(__file__).resolve().parent
    src_dir = (this_dir.parent / "cuda_kernels").resolve()
    build_dir = src_dir / "build"
    os.makedirs(build_dir, exist_ok=True)

    sources = [
        str(src_dir / "pybind_cuda.cpp"),
        str(src_dir / "gather2d.cu"),
        str(src_dir / "scatter2d.cu"),
        str(src_dir / "gather3d.cu"),
        str(src_dir / "scatter3d.cu"),
        str(src_dir / "scatter_gather3d.cu"),
    ]

    return load(
        name="sige3d_cuda_ext",
        sources=sources,
        extra_include_paths=[str(src_dir)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=bool(int(os.environ.get("SIGE3D_CUDA_BUILD_VERBOSE", "0"))),
        build_directory=str(build_dir),
    )


def get_sige3d_cuda_ext() -> ModuleType:
    return _load_sige3d_cuda_ext()
