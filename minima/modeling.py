from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

from .loading import build_lfm_encoder
from .modules import (
    PackedTernaryEmbedding,
    PackedTernaryLinear,
    TernaryEmbedding,
    TernaryLinear,
    optimize_cpu_model,
)
from .quantization import (
    QuantizedWeight,
    base3_rowwise_to_i2s,
    base3_to_i2s,
    dequantize,
    i2s_to_base3,
    i2s_to_base3_rowwise,
)
from .spellcheck import patch_tied_vocab_projection

FORMAT_VERSION = 1
STORAGE_FORMATS = {"i2_s", "base3", "base3_rowwise"}
SCALE_STORAGE_FORMATS = {"fp16", "uint8_rowwise"}


def _storage_format(metadata: dict[str, Any]) -> str:
    value = metadata.get("storage_format", "i2_s")
    if value not in STORAGE_FORMATS:
        raise ValueError(f"unsupported Minima storage format {value!r}")
    return value


def _scale_storage_format(metadata: dict[str, Any]) -> str:
    value = metadata.get("scale_storage", "fp16")
    if value not in SCALE_STORAGE_FORMATS:
        raise ValueError(f"unsupported Minima scale storage format {value!r}")
    return value


def _descriptor_padded_cols(descriptor: dict[str, Any]) -> int:
    cols = descriptor.get("in_features", descriptor.get("embedding_dim"))
    group_size = descriptor["group_size"]
    return ((cols + group_size - 1) // group_size) * group_size


def _transcode_packed_state(tensors: dict[str, torch.Tensor], metadata: dict[str, Any], *,
                            loading: bool) -> dict[str, torch.Tensor]:
    """Transcode compact checkpoint tensors without changing runtime modules."""
    if _storage_format(metadata) == "i2_s":
        return tensors
    for path, descriptor in metadata["modules"].items():
        key = f"{path}.packed_weight"
        if key in tensors:
            group_size = descriptor["group_size"]
            if _storage_format(metadata) == "base3_rowwise":
                tensors[key] = (
                    base3_rowwise_to_i2s(
                        tensors[key], _descriptor_padded_cols(descriptor), group_size,
                    )
                    if loading
                    else i2s_to_base3_rowwise(tensors[key], group_size)
                )
            else:
                convert = base3_to_i2s if loading else i2s_to_base3
                tensors[key] = convert(tensors[key], group_size)
    return tensors


def _transcode_scale_state(tensors: dict[str, torch.Tensor], metadata: dict[str, Any], *,
                           loading: bool) -> dict[str, torch.Tensor]:
    """Compress FP16 group scales to per-row affine uint8 checkpoint tensors."""
    if _scale_storage_format(metadata) == "fp16":
        return tensors
    for path in metadata["modules"]:
        key = f"{path}.weight_scale"
        quantized_key = f"{path}.weight_scale_q"
        minimum_key = f"{path}.weight_scale_min"
        step_key = f"{path}.weight_scale_step"
        if loading:
            if quantized_key not in tensors:
                continue
            quantized = tensors.pop(quantized_key).float()
            minimum = tensors.pop(minimum_key).float().unsqueeze(-1)
            step = tensors.pop(step_key).float().unsqueeze(-1)
            tensors[key] = (minimum + quantized * step).half().contiguous()
        elif key in tensors:
            scale = tensors.pop(key).float()
            minimum = scale.amin(dim=-1).half()
            maximum = scale.amax(dim=-1)
            step = ((maximum - minimum.float()) / 255.0).half()
            denominator = step.float().unsqueeze(-1)
            quantized = torch.where(
                denominator > 0,
                torch.round((scale - minimum.float().unsqueeze(-1)) / denominator),
                torch.zeros_like(scale),
            ).clamp_(0, 255).to(torch.uint8)
            tensors[quantized_key] = quantized.contiguous()
            tensors[minimum_key] = minimum.contiguous()
            tensors[step_key] = step.contiguous()
    return tensors


def _parent_and_name(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def prepare_qat(model: nn.Module, group_size: int = 128, recovery_rank: int = 8,
                include_embeddings: bool = True) -> nn.Module:
    """Replace dense weights with trainable STE ternary modules in-place."""
    shared: dict[int, tuple[nn.Parameter, nn.Parameter]] = {}
    candidates = list(model.named_modules())
    for path, module in candidates:
        if not path or isinstance(module, (TernaryLinear, TernaryEmbedding, PackedTernaryLinear,
                                           PackedTernaryEmbedding)):
            continue
        replacement: nn.Module | None = None
        if isinstance(module, nn.Linear):
            replacement = TernaryLinear.from_float(module, group_size, recovery_rank)
        elif include_embeddings and isinstance(module, nn.Embedding):
            replacement = TernaryEmbedding.from_float(module, group_size, recovery_rank)
        if replacement is None:
            continue
        pointer = module.weight.data_ptr()
        if pointer in shared:
            replacement.weight, replacement.log_scale = shared[pointer]
        else:
            shared[pointer] = (replacement.weight, replacement.log_scale)
        parent, name = _parent_and_name(model, path)
        setattr(parent, name, replacement)
    return model


def unpack_for_qat(model: nn.Module) -> nn.Module:
    """Expand a strict packed artifact into FP32 latent weights for task QAT.

    The expanded tensors are optimizer state, not part of the exported model.
    Repacking after tuning writes only I2_S trits and FP16 group scales.
    """
    candidates = list(model.named_modules())
    for path, module in candidates:
        if not path or not isinstance(module, (PackedTernaryLinear, PackedTernaryEmbedding)):
            continue
        if module.recovery_rank:
            raise ValueError("full-weight QAT expects a strict artifact without recovery adapters")
        if isinstance(module, PackedTernaryLinear):
            qweight = QuantizedWeight(
                module.packed_weight,
                module.weight_scale,
                (module.out_features, module.in_features),
                module.group_size,
            )
            weight = dequantize(qweight, device=module.packed_weight.device, dtype=torch.float32)
            replacement = TernaryLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                group_size=module.group_size,
                recovery_rank=0,
                device=weight.device,
                dtype=torch.float32,
            )
            replacement.weight.data.copy_(weight)
            replacement.log_scale.data.copy_(module.weight_scale.float().log())
            if module.bias is not None:
                replacement.bias.data.copy_(module.bias.float())
        else:
            qweight = QuantizedWeight(
                module.packed_weight,
                module.weight_scale,
                (module.num_embeddings, module.embedding_dim),
                module.group_size,
            )
            weight = dequantize(qweight, device=module.packed_weight.device, dtype=torch.float32)
            replacement = TernaryEmbedding(
                module.num_embeddings,
                module.embedding_dim,
                padding_idx=module.padding_idx,
                _weight=weight,
                group_size=module.group_size,
                recovery_rank=0,
                device=weight.device,
                dtype=torch.float32,
            )
            replacement.log_scale.data.copy_(module.weight_scale.float().log())
        parent, name = _parent_and_name(model, path)
        setattr(parent, name, replacement)
    return model


def pack_model(model: nn.Module, group_size: int = 128, recovery_rank: int = 0,
               include_embeddings: bool = True) -> tuple[nn.Module, dict[str, dict[str, Any]]]:
    """Replace dense/QAT modules with packed inference modules in-place."""
    descriptors: dict[str, dict[str, Any]] = {}
    shared: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    candidates = list(model.named_modules())
    for path, module in candidates:
        if not path or isinstance(module, (PackedTernaryLinear, PackedTernaryEmbedding)):
            continue
        replacement: nn.Module | None = None
        if isinstance(module, (nn.Linear, TernaryLinear)):
            replacement = PackedTernaryLinear.from_float(module, group_size, recovery_rank)
            kind = "linear"
            descriptor = {
                "kind": kind,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "group_size": group_size,
                "bias": module.bias is not None,
                "recovery_rank": replacement.recovery_rank,
            }
        elif include_embeddings and isinstance(module, (nn.Embedding, TernaryEmbedding)):
            replacement = PackedTernaryEmbedding.from_float(module, group_size, recovery_rank)
            kind = "embedding"
            descriptor = {
                "kind": kind,
                "num_embeddings": module.num_embeddings,
                "embedding_dim": module.embedding_dim,
                "padding_idx": module.padding_idx,
                "group_size": group_size,
                "recovery_rank": replacement.recovery_rank,
            }
        else:
            continue
        pointer = module.weight.data_ptr()
        if pointer in shared:
            replacement.packed_weight = shared[pointer][0]
            replacement.weight_scale = shared[pointer][1]
            descriptor["shared_weight"] = True
        else:
            shared[pointer] = (replacement.packed_weight, replacement.weight_scale)
        parent, name = _parent_and_name(model, path)
        setattr(parent, name, replacement)
        descriptors[path] = descriptor
    return model, descriptors


def _empty_packed(descriptor: dict[str, Any]) -> nn.Module:
    group_size = descriptor["group_size"]
    if descriptor["kind"] == "linear":
        rows, cols = descriptor["out_features"], descriptor["in_features"]
        groups = (cols + group_size - 1) // group_size
        packed = torch.empty((rows, groups, group_size // 4), dtype=torch.uint8)
        scale = torch.empty((rows, groups), dtype=torch.float16)
        bias = torch.empty(rows, dtype=torch.float16) if descriptor["bias"] else None
        rank = descriptor.get("recovery_rank", 0)
        recovery_a = torch.empty((rows, rank), dtype=torch.float16) if rank else None
        recovery_b = torch.empty((rank, cols), dtype=torch.float16) if rank else None
        return PackedTernaryLinear(cols, rows, group_size, packed, scale, bias, recovery_a, recovery_b)
    rows, cols = descriptor["num_embeddings"], descriptor["embedding_dim"]
    groups = (cols + group_size - 1) // group_size
    packed = torch.empty((rows, groups, group_size // 4), dtype=torch.uint8)
    scale = torch.empty((rows, groups), dtype=torch.float16)
    rank = descriptor.get("recovery_rank", 0)
    recovery_a = torch.empty((rows, rank), dtype=torch.float16) if rank else None
    recovery_b = torch.empty((rank, cols), dtype=torch.float16) if rank else None
    return PackedTernaryEmbedding(rows, cols, group_size, packed, scale, descriptor.get("padding_idx"),
                                  recovery_a, recovery_b)


def _materialize_derived_buffers(model: nn.Module):
    """Rebuild nonpersistent RoPE buffers that are absent from safetensors."""
    for module in model.modules():
        inv_freq = getattr(module, "inv_freq", None)
        if isinstance(inv_freq, torch.Tensor) and inv_freq.device.type == "meta":
            compute = getattr(module, "compute_default_rope_parameters", None)
            if compute is None:
                raise RuntimeError(f"cannot materialize meta buffer for {type(module).__name__}")
            value, scaling = compute(module.config, device="cpu")
            module.inv_freq = nn.Buffer(value, persistent=False)
            module.original_inv_freq = nn.Buffer(value.clone(), persistent=False)
            module.attention_scaling = scaling


class MinimaModel(nn.Module):
    """Thin wrapper around an LFM2.5 model whose matrix weights are packed ternary."""

    def __init__(self, model: nn.Module, metadata: dict[str, Any]):
        super().__init__()
        self.model = model
        self.minima_metadata = metadata

    @property
    def config(self):
        return self.model.config

    @classmethod
    def from_model(cls, model: nn.Module, *, base_model: str, model_kind: str = "encoder",
                   group_size: int = 128, recovery_rank: int = 0,
                   include_embeddings: bool = True, storage_format: str = "i2_s",
                   scale_storage: str = "fp16") -> "MinimaModel":
        if storage_format not in STORAGE_FORMATS:
            raise ValueError(f"unsupported Minima storage format {storage_format!r}")
        if scale_storage not in SCALE_STORAGE_FORMATS:
            raise ValueError(f"unsupported Minima scale storage format {scale_storage!r}")
        model, descriptors = pack_model(model, group_size, recovery_rank, include_embeddings)
        if model_kind == "spellchecker":
            patch_tied_vocab_projection(model)
        metadata = {
            "format_version": FORMAT_VERSION,
            "format": "i2_s",
            "storage_format": storage_format,
            "scale_storage": scale_storage,
            "logical_weight_bits": 1.585,
            "physical_weight_bits": (
                2 if storage_format == "i2_s" else (
                    1.6 if storage_format == "base3_rowwise"
                    else 8 * ((group_size + 4) // 5) / group_size
                )
            ),
            "runtime_weight_bits": 2,
            "activation_bits": 8,
            "base_model": base_model,
            "base_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
            "model_kind": model_kind,
            "max_context": 8192,
            "modules": descriptors,
        }
        return cls(model, metadata)

    @classmethod
    def from_pretrained(cls, model_id_or_path: str | Path, *, device: str | torch.device = "cpu",
                        revision: str | None = None, token: str | bool | None = None) -> "MinimaModel":
        source = Path(model_id_or_path)
        if not source.exists():
            source = Path(snapshot_download(str(model_id_or_path), revision=revision, token=token))
        metadata = json.loads((source / "minima_config.json").read_text())
        if metadata["format_version"] != FORMAT_VERSION:
            raise ValueError(f"unsupported Minima format version {metadata['format_version']}")

        base_model = metadata["base_model"]
        config = AutoConfig.from_pretrained(
            base_model,
            revision=metadata.get("base_revision"),
            trust_remote_code=True,
            token=token,
        )
        with torch.device("meta"):
            if metadata["model_kind"] == "masked_lm":
                model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)
            elif metadata["model_kind"] == "encoder":
                model = build_lfm_encoder(config)
            else:
                model = AutoModel.from_config(config, trust_remote_code=True)
        for path, descriptor in metadata["modules"].items():
            parent, name = _parent_and_name(model, path)
            setattr(parent, name, _empty_packed(descriptor))
        state = load_file(str(source / "model.safetensors"), device="cpu")
        state = _transcode_packed_state(state, metadata, loading=True)
        state = _transcode_scale_state(state, metadata, loading=True)
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(f"artifact state mismatch: missing={missing}, unexpected={unexpected}")
        if metadata["model_kind"] == "spellchecker":
            patch_tied_vocab_projection(model)
        _materialize_derived_buffers(model)
        remaining_meta = [name for name, value in list(model.named_parameters()) + list(model.named_buffers())
                          if value.device.type == "meta"]
        if remaining_meta:
            raise RuntimeError(f"unmaterialized artifact tensors: {remaining_meta}")
        target_device = torch.device(device)
        if target_device.type == "cuda":
            # Full-weight QAT checkpoints intentionally retain FP32 latent
            # non-matrix parameters while training. The packed input embedding
            # is decoded to FP16 at runtime, so leaving norms and convolutions
            # in FP32 promotes the residual stream and makes attention queries
            # disagree with the FP16 attention mask. Keep every floating
            # runtime tensor aligned with the fused Triton output dtype.
            model.to(device=target_device, dtype=torch.float16)
        else:
            model.to(target_device)
        if target_device.type == "cpu":
            # The upstream artifact keeps non-matrix tensors in FP16. Promoting
            # the small residual set avoids mixed-dtype CPU conv/linear errors;
            # packed uint8 weights are unchanged.
            model.float()
            if os.environ.get("MINIMA_CPU_BACKEND", "dynamic_int8").lower() != "i2s":
                optimize_cpu_model(model)
        model.eval()
        return cls(model, metadata)

    def save_pretrained(self, output_dir: str | Path, tokenizer=None, *, safe_serialization: bool = True):
        if not safe_serialization:
            raise ValueError("Minima artifacts require safe_serialization=True")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "minima_config.json").write_text(json.dumps(self.minima_metadata, indent=2) + "\n")
        tensors = {
            name: (
                value.detach().to(device="cpu", dtype=torch.float16).contiguous().clone()
                if name.endswith(".weight_scale")
                else value.detach().cpu().contiguous().clone()
            )
            for name, value in self.model.state_dict().items()
        }
        tensors = _transcode_scale_state(tensors, self.minima_metadata, loading=False)
        tensors = _transcode_packed_state(tensors, self.minima_metadata, loading=False)
        storage = _storage_format(self.minima_metadata)
        save_file(
            tensors,
            str(output / "model.safetensors"),
            metadata={"format": f"minima-{storage}-v1"},
        )
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.minima_metadata["base_model"], trust_remote_code=True)
        tokenizer.save_pretrained(output)
        try:
            license_path = hf_hub_download(self.minima_metadata["base_model"], "LICENSE")
            shutil.copyfile(license_path, output / "LICENSE")
        except Exception as exc:
            raise RuntimeError("the upstream model LICENSE is required in every derived artifact") from exc
        readme = output / "README.md"
        if not readme.exists():
            readme.write_text(
                "---\nlicense: other\nlicense_name: lfm-open-license-v1.0\n"
                "license_link: https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M/blob/main/LICENSE\n"
                "library_name: minima-lfm\n"
                f"base_model: {self.minima_metadata['base_model']}\n---\n\n"
                "# Minima W1.58A8 artifact\n\n"
                "This repository stores a packed ternary model for "
                "[SSHDotCodes/minima](https://github.com/SSHDotCodes/minima). "
                "Install that package and load it with `MinimaModel.from_pretrained(...)`.\n\n"
                "Matrix weights use logical {-1, 0, +1} values with compact checkpoint storage "
                "and the I2_S runtime format. "
                "See `minima_config.json` for the exact group size, recovery rank, and context limit.\n"
            )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def prepare_qat(self):
        self.model = unpack_for_qat(self.model)
        return self

    def correct(self, *args, **kwargs):
        if not hasattr(self.model, "correct"):
            raise AttributeError("the wrapped model does not implement correct()")
        return self.model.correct(*args, **kwargs)
