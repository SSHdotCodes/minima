from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path

import torch

from .reference import reference_linear


@lru_cache(maxsize=1)
def _cpu_extension():
    if os.environ.get("MINIMA_DISABLE_EXT") == "1":
        return None
    try:
        from torch.utils.cpp_extension import load

        source = Path(__file__).resolve().parent.parent / "csrc" / "ternary_cpu.cpp"
        flags = ["-O3", "-DNDEBUG"]
        if os.name != "nt":
            flags += ["-ffast-math"]
        return load(name="minima_ternary_cpu_v1", sources=[str(source)], extra_cflags=flags, verbose=False)
    except Exception as exc:  # pragma: no cover - compiler availability is environment-specific
        warnings.warn(f"Minima CPU extension unavailable; using reference path: {exc}", RuntimeWarning)
        return None


def kernel_status() -> dict[str, bool]:
    return {
        "cpu_extension": _cpu_extension() is not None,
        "cuda_triton": bool(torch.cuda.is_available() and _has_triton()),
    }


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def ternary_linear(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, in_features: int,
                   group_size: int, bias: torch.Tensor | None = None) -> torch.Tensor:
    original_shape = x.shape
    flat = x.reshape(-1, original_shape[-1])
    if flat.device.type == "cpu":
        extension = _cpu_extension()
        if extension is not None and flat.dtype == torch.float32:
            out = extension.i2s_linear(flat.contiguous(), packed.contiguous(), scale.float().contiguous(),
                                       in_features, group_size)
            if bias is not None:
                out.add_(bias.float())
            return out.reshape(*original_shape[:-1], packed.shape[0])
    elif flat.device.type == "cuda" and _has_triton():
        from .triton_kernels import triton_i2s_linear

        out = triton_i2s_linear(flat, packed, scale, in_features, group_size)
        if bias is not None:
            out = out + bias
        return out.reshape(*original_shape[:-1], packed.shape[0])
    return reference_linear(x, packed, scale, in_features, group_size, bias)

