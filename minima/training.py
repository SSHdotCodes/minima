from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .loading import load_lfm_encoder
from .modeling import MinimaModel, prepare_qat
from .modules import PackedTernaryEmbedding, TernaryEmbedding, TernaryLinear
from .quantization import fake_quantize_weight
from .tuning import cast_recovery_parameters


@dataclass
class DistillationConfig:
    model: str = "LiquidAI/LFM2.5-Encoder-350M"
    student_model: str | None = None
    dataset: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str | None = "sample-10BT"
    split: str = "train"
    text_column: str = "text"
    output_dir: str = "artifacts/minima"
    output_repo: str | None = None
    sequence_length: int = 512
    batch_size: int = 4
    gradient_accumulation: int = 8
    steps: int = 4000
    learning_rate: float = 5e-5
    warmup_steps: int = 200
    group_size: int = 128
    recovery_rank: int = 0
    teacher_topk: int = 32
    hidden_loss_weight: float = 2.0
    layerwise_hidden_loss_weight: float = 0.0
    pooled_loss_weight: float = 0.0
    distill_loss_weight: float = 1.0
    mlm_loss_weight: float = 1.0
    temperature: float = 2.0
    seed: int = 42
    log_every: int = 10
    train_ternary_weights: bool = False
    activation_warmup_steps: int = 0
    weight_warmup_steps: int = 0
    max_weight_file_mb: float = 0.0


def _mask_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer,
                 probability: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, probability, device=labels.device)
    special = torch.tensor(
        [tokenizer.get_special_tokens_mask(row, already_has_special_tokens=True) for row in labels.tolist()],
        dtype=torch.bool,
        device=labels.device,
    )
    probability_matrix.masked_fill_(special | attention_mask.eq(0), 0.0)
    masked = torch.bernoulli(probability_matrix).bool()
    # Guarantee at least one target for tiny/degenerate batches.
    if not masked.any():
        first = (attention_mask.bool() & ~special).nonzero(as_tuple=False)[0]
        masked[first[0], first[1]] = True
    labels[~masked] = -100
    corrupted = input_ids.clone()
    replace = torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device)).bool() & masked
    corrupted[replace] = tokenizer.mask_token_id
    randomize = torch.bernoulli(torch.full(labels.shape, 0.5, device=labels.device)).bool() & masked & ~replace
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=labels.device)
    corrupted[randomize] = random_words[randomize]
    return corrupted, labels


def _selected_student_rows(embedding: TernaryEmbedding | PackedTernaryEmbedding,
                           ids: torch.Tensor) -> torch.Tensor:
    if isinstance(embedding, PackedTernaryEmbedding):
        return embedding.selected_rows(ids, torch.float32)
    rows = embedding.weight.index_select(0, ids.reshape(-1)).view(*ids.shape, embedding.embedding_dim)
    selected_scale = embedding.log_scale.index_select(0, ids.reshape(-1)).exp()
    rows = fake_quantize_weight(
        rows.reshape(-1, embedding.embedding_dim), embedding.group_size, selected_scale,
    ).view_as(rows)
    if embedding.recovery_rank:
        a = embedding.recovery_a.index_select(0, ids.reshape(-1)).view(*ids.shape, embedding.recovery_rank)
        rows = rows + a @ embedding.recovery_b
    return rows


def _candidate_loss(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor, labels: torch.Tensor,
    student_embedding: TernaryEmbedding | PackedTernaryEmbedding, teacher_weight: torch.Tensor, topk: int,
                    temperature: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    positions = labels.ne(-100)
    targets = labels[positions]
    sh = student_hidden[positions]
    th = teacher_hidden[positions]
    with torch.no_grad():
        teacher_logits = F.linear(th, teacher_weight)
        top_values, top_ids = teacher_logits.topk(topk, dim=-1)
        present = top_ids.eq(targets[:, None])
        candidate_ids = top_ids.clone()
        missing = ~present.any(dim=-1)
        candidate_ids[missing, -1] = targets[missing]
        target_index = present.float().argmax(dim=-1)
        target_index[missing] = topk - 1
        teacher_selected = teacher_logits.gather(-1, candidate_ids)
        teacher_prob = F.softmax(teacher_selected / temperature, dim=-1)
        teacher_top1 = teacher_logits.argmax(dim=-1).eq(targets).float().mean().item()
        del teacher_logits

    student_rows = _selected_student_rows(student_embedding, candidate_ids)
    student_logits = torch.einsum("ph,pch->pc", sh, student_rows)
    mlm_loss = F.cross_entropy(student_logits, target_index)
    distill_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1), teacher_prob,
        reduction="batchmean",
    ) * (temperature * temperature)
    return mlm_loss, distill_loss, teacher_top1


def _learning_rate(step: int, config: DistillationConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def _masked_cosine_loss(student: torch.Tensor, teacher: torch.Tensor,
                        attention_mask: torch.Tensor) -> torch.Tensor:
    loss = 1.0 - F.cosine_similarity(student.float(), teacher.float(), dim=-1)
    mask = attention_mask.to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def _pooled_cosine_loss(student: torch.Tensor, teacher: torch.Tensor,
                        attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    denominator = mask.sum(dim=1).clamp_min(1)
    student_pooled = (student.float() * mask).sum(dim=1) / denominator
    teacher_pooled = (teacher.float() * mask).sum(dim=1) / denominator
    return (1.0 - F.cosine_similarity(student_pooled, teacher_pooled, dim=-1)).mean()


def _set_activation_quant(model: torch.nn.Module, enabled: bool):
    for module in model.modules():
        if isinstance(module, TernaryLinear):
            module.activation_quant = enabled


def _set_weight_quant_strength(model: torch.nn.Module, strength: float):
    for module in model.modules():
        if isinstance(module, (TernaryLinear, TernaryEmbedding)):
            module.weight_quant_strength = strength


def distill(config: DistillationConfig) -> dict:
    from datasets import load_dataset

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("distillation requires a CUDA GPU")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    teacher = load_lfm_encoder(config.model, torch_dtype=torch.bfloat16,
                               attn_implementation="sdpa").to(device).eval()
    packed_student = None
    if config.student_model:
        if not config.train_ternary_weights or config.recovery_rank:
            raise ValueError("strict checkpoint continuation requires full-weight QAT and recovery_rank=0")
        resumed = MinimaModel.from_pretrained(config.student_model, device=device)
        if any(descriptor.get("recovery_rank", 0)
               for descriptor in resumed.minima_metadata["modules"].values()):
            raise ValueError("cannot continue strict QAT from an artifact with recovery adapters")
        student = resumed.prepare_qat().model.float()
    else:
        student_source = load_lfm_encoder(
            config.model,
            torch_dtype=torch.float32 if config.train_ternary_weights else torch.float16,
            attn_implementation="sdpa",
        ).to(device)
        if config.train_ternary_weights:
            student = prepare_qat(
                student_source, config.group_size, config.recovery_rank, include_embeddings=True,
            )
        else:
            packed_student = MinimaModel.from_model(
                student_source, base_model=config.model, model_kind="encoder",
                group_size=config.group_size, recovery_rank=config.recovery_rank,
            )
            student = packed_student.model.to(device)
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(config.train_ternary_weights or "recovery_" in name)
        if parameter.requires_grad and not config.train_ternary_weights:
            parameter.data = parameter.data.float()
    trainable_items = [(name, parameter) for name, parameter in student.named_parameters()
                       if parameter.requires_grad]
    trainable = [parameter for _, parameter in trainable_items]
    decay = [parameter for name, parameter in trainable_items
             if parameter.ndim >= 2 and not name.endswith("log_scale")]
    no_decay = [parameter for name, parameter in trainable_items
                if parameter.ndim < 2 or name.endswith("log_scale")]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
    )

    dataset = load_dataset(config.dataset, config.dataset_config, split=config.split, streaming=True)
    dataset = dataset.shuffle(seed=config.seed, buffer_size=10_000)

    def collate(examples):
        texts = [str(example[config.text_column]) for example in examples]
        encoded = tokenizer(texts, max_length=config.sequence_length, truncation=True, padding="max_length",
                            return_tensors="pt")
        return encoded["input_ids"], encoded["attention_mask"]

    loader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=collate, num_workers=2,
                        pin_memory=True)
    iterator = iter(loader)
    student_embedding = student.get_input_embeddings()
    if not isinstance(student_embedding, (TernaryEmbedding, PackedTernaryEmbedding)):
        raise TypeError("QAT preparation did not replace the input embedding")
    teacher_weight = teacher.get_input_embeddings().weight
    started = time.time()
    history: list[dict] = []
    optimizer.zero_grad(set_to_none=True)
    student.train()
    if config.train_ternary_weights and config.activation_warmup_steps:
        _set_activation_quant(student, False)

    for step in range(config.steps):
        if config.train_ternary_weights and config.weight_warmup_steps:
            strength = min(1.0, (step + 1) / config.weight_warmup_steps)
            _set_weight_quant_strength(student, strength)
        if config.train_ternary_weights and step == config.activation_warmup_steps:
            _set_activation_quant(student, True)
        totals = {
            "loss": 0.0,
            "hidden": 0.0,
            "layerwise_hidden": 0.0,
            "pooled": 0.0,
            "mlm": 0.0,
            "distill": 0.0,
            "teacher_acc": 0.0,
        }
        for _ in range(config.gradient_accumulation):
            try:
                input_ids, attention_mask = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                input_ids, attention_mask = next(iterator)
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            corrupted, labels = _mask_tokens(input_ids, attention_mask, tokenizer)
            return_hidden_states = config.layerwise_hidden_loss_weight > 0
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_output = teacher(
                    input_ids=corrupted,
                    attention_mask=attention_mask,
                    output_hidden_states=return_hidden_states,
                )
                teacher_hidden = teacher_output.last_hidden_state
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student_output = student(
                    input_ids=corrupted,
                    attention_mask=attention_mask,
                    output_hidden_states=return_hidden_states,
                )
                student_hidden = student_output.last_hidden_state
                hidden_loss = _masked_cosine_loss(student_hidden, teacher_hidden, attention_mask)
                pooled_loss = _pooled_cosine_loss(student_hidden, teacher_hidden, attention_mask)
                if return_hidden_states:
                    teacher_states = teacher_output.hidden_states
                    student_states = student_output.hidden_states
                    if len(student_states) != len(teacher_states):
                        raise RuntimeError(
                            f"hidden-state count mismatch: student={len(student_states)}, "
                            f"teacher={len(teacher_states)}",
                        )
                    # The final state already has its own stronger objective.
                    internal_pairs = list(zip(student_states[:-1], teacher_states[:-1]))
                    layerwise_hidden_loss = torch.stack([
                        _masked_cosine_loss(student_state, teacher_state, attention_mask)
                        for student_state, teacher_state in internal_pairs
                    ]).mean()
                else:
                    layerwise_hidden_loss = hidden_loss.new_zeros(())
                mlm_loss, distill_loss, teacher_acc = _candidate_loss(
                    student_hidden, teacher_hidden, labels, student_embedding, teacher_weight,
                    config.teacher_topk, config.temperature,
                )
                loss = (
                    config.hidden_loss_weight * hidden_loss
                    + config.layerwise_hidden_loss_weight * layerwise_hidden_loss
                    + config.pooled_loss_weight * pooled_loss
                    + config.mlm_loss_weight * mlm_loss
                    + config.distill_loss_weight * distill_loss
                ) / config.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite distillation loss at step {step + 1}")
            loss.backward()
            totals["loss"] += loss.item()
            totals["hidden"] += hidden_loss.item() / config.gradient_accumulation
            totals["layerwise_hidden"] += layerwise_hidden_loss.item() / config.gradient_accumulation
            totals["pooled"] += pooled_loss.item() / config.gradient_accumulation
            totals["mlm"] += mlm_loss.item() / config.gradient_accumulation
            totals["distill"] += distill_loss.item() / config.gradient_accumulation
            totals["teacher_acc"] += teacher_acc / config.gradient_accumulation

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        lr = _learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % config.log_every == 0 or step + 1 == config.steps:
            record = {"step": step + 1, "lr": lr, "elapsed_s": time.time() - started, **totals}
            history.append(record)
            print(json.dumps(record), flush=True)

    output = Path(config.output_dir)
    if packed_student is None:
        model = MinimaModel.from_model(student.eval(), base_model=config.model, model_kind="encoder",
                                       group_size=config.group_size, recovery_rank=config.recovery_rank)
    else:
        cast_recovery_parameters(student)
        packed_student.model = student.eval()
        model = packed_student
    model.save_pretrained(output, tokenizer)
    weight_file_bytes = (output / "model.safetensors").stat().st_size
    if config.max_weight_file_mb and weight_file_bytes > config.max_weight_file_mb * 1_000_000:
        raise RuntimeError(
            f"exported weight file is {weight_file_bytes / 1_000_000:.2f} MB, "
            f"above the {config.max_weight_file_mb:.2f} MB limit",
        )
    report = {
        "config": config.__dict__,
        "duration_seconds": time.time() - started,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "weight_file_bytes": weight_file_bytes,
        "history": history,
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if config.output_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(config.output_repo, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=config.output_repo, folder_path=output, repo_type="model",
                          commit_message="Upload distilled Minima W1.58A8 encoder")
    return report
