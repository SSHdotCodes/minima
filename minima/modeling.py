from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

from .modules import PackedTernaryEmbedding, PackedTernaryLinear, TernaryEmbedding, TernaryLinear

FORMAT_VERSION = 1


def _parent_and_name(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def prepare_qat(model: nn.Module, group_size: int = 128, recovery_rank: int = 8,
                include_embeddings: bool = True) -> nn.Module:
    """Replace dense weights with trainable STE ternary modules in-place."""
    shared: dict[int, nn.Parameter] = {}
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
            replacement.weight = shared[pointer]
        else:
            shared[pointer] = replacement.weight
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
            replacement = PackedTernaryEmbedding.from_float(module, group_size)
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
                   include_embeddings: bool = True) -> "MinimaModel":
        model, descriptors = pack_model(model, group_size, recovery_rank, include_embeddings)
        metadata = {
            "format_version": FORMAT_VERSION,
            "format": "i2_s",
            "logical_weight_bits": 1.585,
            "physical_weight_bits": 2,
            "activation_bits": 8,
            "base_model": base_model,
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
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True, token=token)
        factory = AutoModelForMaskedLM if metadata["model_kind"] == "masked_lm" else AutoModel
        with torch.device("meta"):
            model = factory.from_config(config, trust_remote_code=True)
        for path, descriptor in metadata["modules"].items():
            parent, name = _parent_and_name(model, path)
            setattr(parent, name, _empty_packed(descriptor))
        state = load_file(str(source / "model.safetensors"), device="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(f"artifact state mismatch: missing={missing}, unexpected={unexpected}")
        model.to(device).eval()
        return cls(model, metadata)

    def save_pretrained(self, output_dir: str | Path, tokenizer=None, *, safe_serialization: bool = True):
        if not safe_serialization:
            raise ValueError("Minima artifacts require safe_serialization=True")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "minima_config.json").write_text(json.dumps(self.minima_metadata, indent=2) + "\n")
        tensors = {name: value.detach().cpu().contiguous().clone()
                   for name, value in self.model.state_dict().items()}
        save_file(tensors, str(output / "model.safetensors"), metadata={"format": "minima-i2s-v1"})
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.minima_metadata["base_model"], trust_remote_code=True)
        tokenizer.save_pretrained(output)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
