from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import ternary_linear
from .quantization import (
    QuantizedWeight,
    dequantize,
    fake_quantize_activation,
    fake_quantize_weight,
    quantize_ternary,
)


def _dynamic_int8_linear(weight: torch.Tensor, bias: torch.Tensor | None) -> nn.Module:
    """Pack an effective FP32 matrix for PyTorch's optimized dynamic INT8 GEMM."""
    requested_engine = os.environ.get("MINIMA_QUANTIZED_ENGINE")
    if requested_engine:
        if requested_engine not in torch.backends.quantized.supported_engines:
            raise RuntimeError(f"unsupported quantized engine: {requested_engine}")
        torch.backends.quantized.engine = requested_engine
    elif torch.backends.quantized.engine == "none":
        for engine in ("x86", "fbgemm", "qnnpack"):
            if engine in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = engine
                break
    if torch.backends.quantized.engine == "none":
        raise RuntimeError("this PyTorch build has no dynamic quantized CPU engine")
    weight = weight.float().contiguous()
    scales = weight.abs().amax(dim=1).clamp_min(1.0e-8) / 127.0
    zeros = torch.zeros(weight.shape[0], dtype=torch.long)
    qweight = torch.quantize_per_channel(weight, scales, zeros, axis=0, dtype=torch.qint8)
    linear = torch.ao.nn.quantized.dynamic.Linear(
        weight.shape[1], weight.shape[0], bias_=bias is not None,
    )
    linear.set_weight_bias(qweight, None if bias is None else bias.float())
    return linear.eval()


def _factor_residual(weight: torch.Tensor, qweight: QuantizedWeight,
                     rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomized low-rank factors for W - dequant(W), computed where W lives."""
    residual = weight.detach().float() - dequantize(qweight, device=weight.device)
    rank = min(rank, min(residual.shape))
    q = min(min(residual.shape), rank + 8)
    u, s, v = torch.svd_lowrank(residual, q=q, niter=2)
    root = s[:rank].sqrt()
    a = (u[:, :rank] * root).to(torch.float16).cpu()
    b = (root[:, None] * v[:, :rank].t()).to(torch.float16).cpu()
    return a, b


class TernaryLinear(nn.Module):
    """Trainable W1.58A8 layer with FP master weights and an STE forward."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, group_size: int = 128,
                 recovery_rank: int = 0, activation_quant: bool = True, *, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.activation_quant = activation_quant
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        self.recovery_rank = recovery_rank
        if recovery_rank:
            self.recovery_a = nn.Parameter(torch.zeros(out_features, recovery_rank, device=device, dtype=dtype))
            self.recovery_b = nn.Parameter(torch.empty(recovery_rank, in_features, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.recovery_b, a=math.sqrt(5))
        else:
            self.register_parameter("recovery_a", None)
            self.register_parameter("recovery_b", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_float(cls, module: nn.Linear, group_size: int = 128, recovery_rank: int = 0,
                   activation_quant: bool = True) -> "TernaryLinear":
        result = cls(module.in_features, module.out_features, module.bias is not None, group_size,
                     recovery_rank, activation_quant, device=module.weight.device, dtype=module.weight.dtype)
        result.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            result.bias.data.copy_(module.bias.data)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qx = fake_quantize_activation(x, self.group_size) if self.activation_quant else x
        output = F.linear(qx, fake_quantize_weight(self.weight, self.group_size), self.bias)
        if self.recovery_rank:
            output = output + F.linear(F.linear(x, self.recovery_b), self.recovery_a)
        return output


class TernaryEmbedding(nn.Embedding):
    def __init__(self, *args, group_size: int = 128, recovery_rank: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_size = group_size
        self.recovery_rank = recovery_rank
        if recovery_rank:
            self.recovery_a = nn.Parameter(torch.zeros(self.num_embeddings, recovery_rank,
                                                       device=self.weight.device, dtype=self.weight.dtype))
            self.recovery_b = nn.Parameter(torch.empty(recovery_rank, self.embedding_dim,
                                                       device=self.weight.device, dtype=self.weight.dtype))
            nn.init.kaiming_uniform_(self.recovery_b, a=math.sqrt(5))
        else:
            self.register_parameter("recovery_a", None)
            self.register_parameter("recovery_b", None)

    @classmethod
    def from_float(cls, module: nn.Embedding, group_size: int = 128,
                   recovery_rank: int = 0) -> "TernaryEmbedding":
        result = cls(module.num_embeddings, module.embedding_dim, padding_idx=module.padding_idx,
                     max_norm=module.max_norm, norm_type=module.norm_type,
                     scale_grad_by_freq=module.scale_grad_by_freq, sparse=module.sparse,
                     _weight=module.weight.detach().clone(), group_size=group_size,
                     recovery_rank=recovery_rank, device=module.weight.device, dtype=module.weight.dtype)
        return result

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        weight = fake_quantize_weight(self.weight, self.group_size)
        output = F.embedding(input_ids, weight, self.padding_idx, self.max_norm, self.norm_type,
                             self.scale_grad_by_freq, self.sparse)
        if self.recovery_rank:
            output = output + F.embedding(input_ids, self.recovery_a) @ self.recovery_b
        return output

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = fake_quantize_weight(self.weight, self.group_size)
        output = F.linear(fake_quantize_activation(hidden, self.group_size), weight)
        if self.recovery_rank:
            output = output + F.linear(F.linear(hidden, self.recovery_b), self.recovery_a)
        return output


class PackedTernaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, group_size: int, packed_weight: torch.Tensor,
                 weight_scale: torch.Tensor, bias: torch.Tensor | None = None,
                 recovery_a: torch.Tensor | None = None, recovery_b: torch.Tensor | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer("packed_weight", packed_weight.to(torch.uint8).contiguous())
        self.register_buffer("weight_scale", weight_scale.contiguous())
        self.register_buffer("bias", bias.contiguous() if bias is not None else None)
        self.register_parameter("recovery_a", nn.Parameter(recovery_a.contiguous()) if recovery_a is not None else None)
        self.register_parameter("recovery_b", nn.Parameter(recovery_b.contiguous()) if recovery_b is not None else None)
        self.register_buffer("_training_weight", None, persistent=False)
        object.__setattr__(self, "_cpu_int8", None)

    @property
    def recovery_rank(self) -> int:
        return 0 if self.recovery_a is None else self.recovery_a.shape[1]

    @classmethod
    def from_float(cls, module: nn.Linear | TernaryLinear, group_size: int | None = None,
                   recovery_rank: int = 0) -> "PackedTernaryLinear":
        group_size = group_size or getattr(module, "group_size", 128)
        qweight = quantize_ternary(module.weight, group_size)
        recovery_a = recovery_b = None
        if isinstance(module, TernaryLinear) and module.recovery_rank:
            recovery_a = module.recovery_a.detach().to(torch.float16).cpu()
            recovery_b = module.recovery_b.detach().to(torch.float16).cpu()
        elif recovery_rank:
            recovery_a, recovery_b = _factor_residual(module.weight, qweight, recovery_rank)
        bias = None if module.bias is None else module.bias.detach().to(torch.float16).cpu()
        return cls(module.in_features, module.out_features, group_size, qweight.packed, qweight.scale, bias,
                   recovery_a, recovery_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused_recovery = False
        if self.training and torch.is_grad_enabled():
            if (self._training_weight is None or self._training_weight.device != x.device or
                    self._training_weight.dtype != x.dtype):
                qweight = QuantizedWeight(self.packed_weight, self.weight_scale,
                                          (self.out_features, self.in_features), self.group_size)
                self._training_weight = dequantize(qweight, device=x.device, dtype=x.dtype)
            qx = fake_quantize_activation(x, self.group_size)
            output = F.linear(qx, self._training_weight, None if self.bias is None else self.bias.to(x.dtype))
        elif x.device.type == "cpu" and x.dtype == torch.float32 and self._cpu_int8 is not None:
            output = self._cpu_int8(x)
            fused_recovery = True
        else:
            output = ternary_linear(x, self.packed_weight, self.weight_scale, self.in_features,
                                    self.group_size, self.bias)
        if self.recovery_a is not None and not fused_recovery:
            output = output + F.linear(F.linear(x, self.recovery_b.to(x.dtype)), self.recovery_a.to(x.dtype))
        return output

    def optimize_cpu(self, *, release_source: bool = True):
        """Fuse ternary + recovery weights into a fast per-channel INT8 CPU GEMM."""
        if self._cpu_int8 is None:
            effective = self.effective_cpu_weight()
            object.__setattr__(self, "_cpu_int8", _dynamic_int8_linear(effective, self.bias))
        if release_source:
            self.release_cpu_source()
        return self

    def effective_cpu_weight(self) -> torch.Tensor:
        if not self.packed_weight.numel():
            raise RuntimeError("packed source was released before CPU optimization")
        qweight = QuantizedWeight(self.packed_weight, self.weight_scale,
                                  (self.out_features, self.in_features), self.group_size)
        effective = dequantize(qweight, device="cpu", dtype=torch.float32)
        if self.recovery_a is not None:
            effective.addmm_(self.recovery_a.float(), self.recovery_b.float())
        return effective

    def release_cpu_source(self):
        self.packed_weight = torch.empty(0, dtype=torch.uint8)
        self.weight_scale = torch.empty(0, dtype=torch.float32)
        self.bias = None
        self.recovery_a = None
        self.recovery_b = None


class PackedTernaryEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, group_size: int, packed_weight: torch.Tensor,
                 weight_scale: torch.Tensor, padding_idx: int | None = None,
                 recovery_a: torch.Tensor | None = None, recovery_b: torch.Tensor | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_size = group_size
        self.padding_idx = padding_idx
        self.register_buffer("packed_weight", packed_weight.to(torch.uint8).contiguous())
        self.register_buffer("weight_scale", weight_scale.contiguous())
        self.register_parameter("recovery_a", nn.Parameter(recovery_a.contiguous()) if recovery_a is not None else None)
        self.register_parameter("recovery_b", nn.Parameter(recovery_b.contiguous()) if recovery_b is not None else None)
        self.register_buffer("_training_weight", None, persistent=False)
        object.__setattr__(self, "_cpu_int8_project", None)
        self._cpu_fast_project = False

    @property
    def recovery_rank(self) -> int:
        return 0 if self.recovery_a is None else self.recovery_a.shape[1]

    @classmethod
    def from_float(cls, module: nn.Embedding | TernaryEmbedding,
                   group_size: int | None = None, recovery_rank: int = 0) -> "PackedTernaryEmbedding":
        group_size = group_size or getattr(module, "group_size", 128)
        qweight = quantize_ternary(module.weight, group_size)
        recovery_a = recovery_b = None
        if isinstance(module, TernaryEmbedding) and module.recovery_rank:
            recovery_a = module.recovery_a.detach().to(torch.float16).cpu()
            recovery_b = module.recovery_b.detach().to(torch.float16).cpu()
        elif recovery_rank:
            recovery_a, recovery_b = _factor_residual(module.weight, qweight, recovery_rank)
        return cls(module.num_embeddings, module.embedding_dim, group_size, qweight.packed, qweight.scale,
                   module.padding_idx, recovery_a, recovery_b)

    def _rows(self, rows: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        flat = rows.reshape(-1)
        packed = self.packed_weight.index_select(0, flat)
        scale = self.weight_scale.index_select(0, flat)
        qweight = QuantizedWeight(packed, scale, (flat.numel(), self.embedding_dim), self.group_size)
        values = dequantize(qweight, device=rows.device, dtype=dtype)
        return values.view(*rows.shape, self.embedding_dim)

    def selected_rows(self, rows: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        output = self._rows(rows, dtype)
        if self.recovery_a is not None:
            output = output + F.embedding(rows, self.recovery_a.to(dtype)) @ self.recovery_b.to(dtype)
        return output


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        dtype = torch.get_default_dtype() if self.weight_scale.device.type == "cpu" else torch.float16
        return self.selected_rows(input_ids, dtype)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        fused_recovery = False
        if self.training and torch.is_grad_enabled():
            if (self._training_weight is None or self._training_weight.device != hidden.device or
                    self._training_weight.dtype != hidden.dtype):
                qweight = QuantizedWeight(self.packed_weight, self.weight_scale,
                                          (self.num_embeddings, self.embedding_dim), self.group_size)
                self._training_weight = dequantize(qweight, device=hidden.device, dtype=hidden.dtype)
            output = F.linear(fake_quantize_activation(hidden, self.group_size), self._training_weight)
        elif hidden.device.type == "cpu" and hidden.dtype == torch.float32 and self._cpu_fast_project:
            if self._cpu_int8_project is None:
                qweight = QuantizedWeight(self.packed_weight, self.weight_scale,
                                          (self.num_embeddings, self.embedding_dim), self.group_size)
                effective = dequantize(qweight, device="cpu", dtype=torch.float32)
                if self.recovery_a is not None:
                    effective.addmm_(self.recovery_a.float(), self.recovery_b.float())
                object.__setattr__(self, "_cpu_int8_project", _dynamic_int8_linear(effective, None))
            output = self._cpu_int8_project(hidden)
            fused_recovery = True
        else:
            output = ternary_linear(hidden, self.packed_weight, self.weight_scale, self.embedding_dim,
                                    self.group_size)
        if self.recovery_a is not None and not fused_recovery:
            output = output + F.linear(F.linear(hidden, self.recovery_b.to(hidden.dtype)),
                                       self.recovery_a.to(hidden.dtype))
        return output


def optimize_cpu_model(model: nn.Module) -> nn.Module:
    """Optimize each packed projection for the selected dynamic INT8 engine."""
    for module in list(model.modules()):
        if isinstance(module, PackedTernaryLinear):
            module.optimize_cpu(release_source=True)
        elif isinstance(module, PackedTernaryEmbedding):
            module._cpu_fast_project = True
    return model
