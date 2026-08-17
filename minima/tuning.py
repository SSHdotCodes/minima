from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

from .modeling import MinimaModel
from .modules import PackedTernaryEmbedding, PackedTernaryLinear


def enable_recovery_training(model: nn.Module, fp32_master: bool = True,
                             include_non_matrix: bool = True) -> int:
    """Enable recovery adapters and the small non-matrix parameter set."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    count = 0
    enabled: set[int] = set()

    def activate(parameter):
        nonlocal count
        if id(parameter) in enabled:
            return
        if fp32_master and parameter.is_floating_point():
            parameter.data = parameter.data.float()
        parameter.requires_grad_(True)
        enabled.add(id(parameter))
        count += parameter.numel()

    for name, parameter in model.named_parameters():
        if name.endswith("recovery_a") or name.endswith("recovery_b"):
            activate(parameter)
    if include_non_matrix:
        for module in model.modules():
            if isinstance(module, (PackedTernaryLinear, PackedTernaryEmbedding)):
                continue
            for parameter in module.parameters(recurse=False):
                if parameter.is_floating_point():
                    activate(parameter)
    if not count:
        raise ValueError("this artifact has no recovery adapters; use the quality profile or run QAT")
    return count


def cast_recovery_parameters(model: nn.Module, dtype: torch.dtype = torch.float16):
    for name, parameter in model.named_parameters():
        if name.endswith("recovery_a") or name.endswith("recovery_b"):
            parameter.data = parameter.data.to(dtype)


class MinimaForSequenceClassification(nn.Module):
    def __init__(self, encoder: MinimaModel, num_labels: int, pooling: str = "mean", dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.num_labels = num_labels
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, num_labels)
        self.config = encoder.config
        self.config.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs).last_hidden_state
        if self.pooling == "first":
            pooled = hidden[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = torch.nn.functional.mse_loss(logits.squeeze(-1), labels.float())
            else:
                loss = torch.nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def save_adapter(self, output_dir: str | Path, base_model: str):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        state = {name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()
                 if "recovery_" in name or name.startswith("classifier.")}
        save_file(state, str(output / "adapter.safetensors"))
        (output / "adapter_config.json").write_text(json.dumps({
            "base_model": base_model,
            "task": "sequence_classification",
            "num_labels": self.num_labels,
            "pooling": self.pooling,
        }, indent=2) + "\n")

    def save_full_tuned(self, output_dir: str | Path, tokenizer=None):
        """Repack full-weight QAT state without retaining latent FP32 weights."""
        metadata = self.encoder.minima_metadata
        descriptors = list(metadata["modules"].values())
        if not descriptors:
            raise ValueError("encoder metadata contains no packed modules")
        group_size = descriptors[0]["group_size"]
        packed = MinimaModel.from_model(
            self.encoder.model.eval(),
            base_model=metadata["base_model"],
            model_kind=metadata["model_kind"],
            group_size=group_size,
            recovery_rank=0,
            storage_format=metadata.get("storage_format", "i2_s"),
        )
        packed.minima_metadata["base_revision"] = metadata.get("base_revision")
        packed.save_pretrained(output_dir, tokenizer)
        output = Path(output_dir)
        save_file({
            "classifier.weight": self.classifier.weight.detach().cpu().contiguous(),
            "classifier.bias": self.classifier.bias.detach().cpu().contiguous(),
        }, str(output / "classifier.safetensors"))
        (output / "task_config.json").write_text(json.dumps({
            "task": "sequence_classification",
            "num_labels": self.num_labels,
            "pooling": self.pooling,
            "tuning": "full_ternary_qat",
        }, indent=2) + "\n")


class MinimaForTokenClassification(nn.Module):
    def __init__(self, encoder: MinimaModel, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, num_labels)
        self.config = encoder.config
        self.config.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs).last_hidden_state
        logits = self.classifier(self.dropout(hidden))
        loss = None if labels is None else torch.nn.functional.cross_entropy(
            logits.view(-1, self.num_labels), labels.view(-1), ignore_index=-100,
        )
        return TokenClassifierOutput(loss=loss, logits=logits)
