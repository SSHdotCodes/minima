"""Ternary quantization and packed storage formats.

The logical values are {-1, 0, +1}, i.e. log2(3) = 1.585 bits of information.
The hot inference representation follows BitNet's I2_S convention and uses two
physical bits per trit so SIMD kernels can unpack four weights with shifts/masks.
For checkpoints where scale density is more valuable, radix-3 storage packs five
trits per byte and is expanded to I2_S once when the model is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantizedWeight:
    packed: torch.Tensor
    scale: torch.Tensor
    shape: tuple[int, int]
    group_size: int


def _validate_matrix(weight: torch.Tensor, group_size: int) -> tuple[int, int, int]:
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D weight, got shape={tuple(weight.shape)}")
    if group_size <= 0 or group_size % 4:
        raise ValueError("group_size must be a positive multiple of four")
    rows, cols = weight.shape
    groups = (cols + group_size - 1) // group_size
    return rows, cols, groups


def ternary_values(weight: torch.Tensor, group_size: int = 128,
                   scale: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return int8 trits and per-row/per-group absmean scales.

    Padding is appended on the input-feature dimension and is always encoded as
    zero. Scale computation excludes that padding.
    """
    rows, cols, groups = _validate_matrix(weight, group_size)
    padded_cols = groups * group_size
    work = weight.detach().to(torch.float32)
    if padded_cols != cols:
        work = torch.nn.functional.pad(work, (0, padded_cols - cols))
    view = work.view(rows, groups, group_size)
    valid = torch.ones((1, groups, group_size), dtype=torch.float32, device=work.device)
    if padded_cols != cols:
        valid[..., -(padded_cols - cols) :] = 0
    denom = valid.sum(dim=-1).clamp_min(1)
    absolute = view.abs()
    if scale is None:
        scale = (absolute * valid).sum(dim=-1) / denom
        scale = scale.clamp_min(torch.finfo(torch.float32).eps)
        # Lloyd-Max updates minimize per-group squared reconstruction error for a
        # symmetric three-level codebook {-scale, 0, +scale}. Three iterations are
        # enough for stable partitions at these small group sizes.
        for _ in range(3):
            nonzero = (absolute >= scale.unsqueeze(-1) * 0.5) & valid.bool()
            scale = (absolute * nonzero).sum(dim=-1) / nonzero.sum(dim=-1).clamp_min(1)
            scale = scale.clamp_min(torch.finfo(torch.float32).eps)
    else:
        if tuple(scale.shape) != (rows, groups):
            raise ValueError(f"scale must have shape {(rows, groups)}, got {tuple(scale.shape)}")
        scale = scale.detach().to(device=work.device, dtype=torch.float32).clamp_min(
            torch.finfo(torch.float32).eps,
        )
        nonzero = (absolute >= scale.unsqueeze(-1) * 0.5) & valid.bool()
    trits = (view.sign() * nonzero).to(torch.int8)
    trits = trits * valid.to(torch.int8)
    return trits.view(rows, padded_cols), scale


def pack_i2s(trits: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Pack {-1,0,+1} into SIMD-friendly I2_S bytes.

    Within every group a byte contains the same offset from each quarter. This
    layout lets AVX2/NEON kernels decode four vector lanes using shifts only.
    """
    if trits.ndim != 2:
        raise ValueError("trits must be a 2D tensor")
    rows, cols = trits.shape
    if cols % group_size or group_size % 4:
        raise ValueError("padded input width must be divisible by group_size, itself a multiple of four")
    codes = trits.to(torch.int16) + 1
    if bool(((codes < 0) | (codes > 2)).any()):
        raise ValueError("trits may only contain -1, 0, or +1")
    groups = cols // group_size
    quarter = group_size // 4
    codes = codes.view(rows, groups, 4, quarter).to(torch.uint8)
    packed = codes[:, :, 0]
    packed = packed | (codes[:, :, 1] << 2)
    packed = packed | (codes[:, :, 2] << 4)
    packed = packed | (codes[:, :, 3] << 6)
    return packed.contiguous()


def unpack_i2s(packed: torch.Tensor, cols: int, group_size: int = 128) -> torch.Tensor:
    """Unpack I2_S bytes to an int8 matrix and remove right padding."""
    if packed.ndim != 3:
        raise ValueError("packed weights must have shape [rows, groups, group_size/4]")
    rows, groups, quarter = packed.shape
    if quarter * 4 != group_size:
        raise ValueError("packed shape and group_size disagree")
    codes = torch.stack(tuple((packed >> shift) & 0x03 for shift in (0, 2, 4, 6)), dim=2)
    values = codes.to(torch.int8).sub_(1).reshape(rows, groups * group_size)
    return values[:, :cols].contiguous()


def pack_base3(trits: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Pack five contiguous {-1,0,+1} values into each radix-3 byte.

    Groups remain independently addressable. If a group is not divisible by
    five, its final byte is padded with logical zeros (base-3 digit one).
    """
    if trits.ndim != 2:
        raise ValueError("trits must be a 2D tensor")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    rows, cols = trits.shape
    if cols % group_size:
        raise ValueError("padded input width must be divisible by group_size")
    codes = trits.to(torch.int16) + 1
    if bool(((codes < 0) | (codes > 2)).any()):
        raise ValueError("trits may only contain -1, 0, or +1")
    groups = cols // group_size
    bytes_per_group = (group_size + 4) // 5
    padded_group = bytes_per_group * 5
    codes = codes.view(rows, groups, group_size)
    if padded_group != group_size:
        codes = torch.nn.functional.pad(codes, (0, padded_group - group_size), value=1)
    powers = torch.tensor((1, 3, 9, 27, 81), dtype=torch.int16, device=codes.device)
    return (
        (codes.view(rows, groups, bytes_per_group, 5) * powers)
        .sum(dim=-1)
        .to(torch.uint8)
        .contiguous()
    )


def unpack_base3(packed: torch.Tensor, cols: int, group_size: int = 128) -> torch.Tensor:
    """Unpack radix-3 bytes to an int8 matrix and remove right padding."""
    if packed.ndim != 3:
        raise ValueError("packed weights must have shape [rows, groups, ceil(group_size/5)]")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    rows, groups, bytes_per_group = packed.shape
    if bytes_per_group != (group_size + 4) // 5:
        raise ValueError("packed shape and group_size disagree")
    if bool((packed > 242).any()):
        raise ValueError("radix-3 bytes must be in the range 0..242")
    work = packed.to(torch.int16)
    digits = torch.stack(tuple((work // power) % 3 for power in (1, 3, 9, 27, 81)), dim=-1)
    values = digits.to(torch.int8).sub_(1).view(rows, groups, bytes_per_group * 5)
    return values[:, :, :group_size].reshape(rows, groups * group_size)[:, :cols].contiguous()


def i2s_to_base3(packed: torch.Tensor, group_size: int) -> torch.Tensor:
    """Transcode a runtime I2_S tensor into compact checkpoint storage."""
    cols = packed.shape[1] * group_size
    return pack_base3(unpack_i2s(packed, cols, group_size), group_size)


def base3_to_i2s(packed: torch.Tensor, group_size: int) -> torch.Tensor:
    """Expand compact checkpoint storage into the SIMD-friendly runtime form."""
    cols = packed.shape[1] * group_size
    return pack_i2s(unpack_base3(packed, cols, group_size), group_size)


def quantize_ternary(weight: torch.Tensor, group_size: int = 128,
                     scale: torch.Tensor | None = None) -> QuantizedWeight:
    trits, scale = ternary_values(weight, group_size, scale)
    return QuantizedWeight(
        packed=pack_i2s(trits, group_size).cpu(),
        scale=scale.to(torch.float16).cpu(),
        shape=(weight.shape[0], weight.shape[1]),
        group_size=group_size,
    )


def dequantize(qweight: QuantizedWeight, *, device: torch.device | str | None = None,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    rows, cols = qweight.shape
    trits = unpack_i2s(qweight.packed, cols, qweight.group_size).to(device=device, dtype=dtype)
    groups = (cols + qweight.group_size - 1) // qweight.group_size
    padded = groups * qweight.group_size
    if padded != cols:
        trits = torch.nn.functional.pad(trits, (0, padded - cols))
    values = trits.view(rows, groups, qweight.group_size)
    scale = qweight.scale.to(device=device, dtype=dtype).unsqueeze(-1)
    return (values * scale).view(rows, padded)[:, :cols].contiguous()


def fake_quantize_weight(weight: torch.Tensor, group_size: int = 128,
                         scale: torch.Tensor | None = None) -> torch.Tensor:
    """Straight-through ternary fake quantization for latent weights and scales."""
    rows, cols, groups = _validate_matrix(weight, group_size)
    padded_cols = groups * group_size
    work = weight if cols == padded_cols else torch.nn.functional.pad(weight, (0, padded_cols - cols))
    view = work.view(rows, groups, group_size)
    absolute = view.detach().abs()
    if scale is None:
        active_scale = absolute.mean(dim=-1, keepdim=True).clamp_min(1e-8)
        for _ in range(3):
            nonzero = absolute >= active_scale * 0.5
            active_scale = ((absolute * nonzero).sum(dim=-1, keepdim=True) /
                            nonzero.sum(dim=-1, keepdim=True).clamp_min(1)).clamp_min(1e-8)
    else:
        if tuple(scale.shape) != (rows, groups):
            raise ValueError(f"scale must have shape {(rows, groups)}, got {tuple(scale.shape)}")
        active_scale = scale.to(dtype=view.dtype).clamp_min(1e-8).unsqueeze(-1)
        nonzero = absolute >= active_scale.detach() * 0.5
    code = view.detach().sign() * nonzero
    target = code * active_scale
    # Identity STE for the latent weight plus the true derivative for a learned
    # group scale. The second term is zero in the forward pass.
    quant = view + (target - view).detach()
    if scale is not None:
        quant = quant + target - target.detach()
    return quant.view(rows, padded_cols)[:, :cols]


def fake_quantize_activation(x: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Per-token grouped signed-int8 STE quantization (W1.58A8)."""
    cols = x.shape[-1]
    groups = (cols + group_size - 1) // group_size
    padded_cols = groups * group_size
    work = x if cols == padded_cols else torch.nn.functional.pad(x, (0, padded_cols - cols))
    view = work.view(*work.shape[:-1], groups, group_size)
    scale = view.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    quant = torch.round(view / scale).clamp(-127, 127) * scale
    quant = view + (quant - view).detach()
    return quant.view(*work.shape)[:, :cols] if work.ndim == 2 else quant.view(*work.shape)[..., :cols]
