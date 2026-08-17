from __future__ import annotations

import torch

from minima.quantization import QuantizedWeight, dequantize


def reference_linear(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, in_features: int,
                     group_size: int, bias: torch.Tensor | None = None) -> torch.Tensor:
    qweight = QuantizedWeight(packed, scale, (packed.shape[0], in_features), group_size)
    weight = dequantize(qweight, device=x.device, dtype=x.dtype)
    cols = x.shape[-1]
    groups = (cols + group_size - 1) // group_size
    padded_cols = groups * group_size
    work = x if cols == padded_cols else torch.nn.functional.pad(x, (0, padded_cols - cols))
    view = work.view(*work.shape[:-1], groups, group_size)
    xscale = view.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    quant = torch.round(view / xscale).clamp(-127, 127) * xscale
    quant = quant.view(*work.shape)[..., :cols]
    return torch.nn.functional.linear(quant, weight, None if bias is None else bias.to(x.dtype))
