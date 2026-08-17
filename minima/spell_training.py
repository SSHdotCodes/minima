from __future__ import annotations

import json
import random
import re
import string
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from .modeling import MinimaModel
from .modules import PackedTernaryEmbedding
from .tuning import cast_recovery_parameters, enable_recovery_training


@dataclass
class SpellDistillationConfig:
    model: str = "LiquidAI/LFM2.5-Encoder-350M-Spellchecker"
    output_dir: str = "/tmp/minima-spellcheck"
    output_repo: str = "ProCreations/minima-spellcheck"
    dataset: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    sequence_length: int = 128
    batch_size: int = 2
    gradient_accumulation: int = 4
    steps: int = 1000
    learning_rate: float = 5e-5
    group_size: int = 32
    recovery_rank: int = 128
    topk: int = 16
    eval_batches: int = 8
    seed: int = 77
    log_every: int = 10


def _space_punctuation(text: str) -> str:
    return " ".join(re.findall(r"\w+(?:['’]\w+)?|[^\w\s]", text, flags=re.UNICODE))


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 3:
        return word
    operation = rng.choice(("drop", "swap", "repeat", "replace"))
    index = rng.randrange(1, len(word) - 1)
    if operation == "drop":
        return word[:index] + word[index + 1 :]
    if operation == "swap":
        return word[:index] + word[index + 1] + word[index] + word[index + 2 :]
    if operation == "repeat":
        return word[:index] + word[index] + word[index:]
    return word[:index] + rng.choice(string.ascii_lowercase) + word[index + 1 :]


def corrupt_text(text: str, rng: random.Random) -> str:
    words = _space_punctuation(text[:1500]).split()
    if len(words) < 4:
        return " ".join(words)
    edits = max(1, min(4, len(words) // 12))
    for _ in range(edits):
        index = rng.randrange(len(words))
        operation = rng.choice(("typo", "case", "duplicate", "delete"))
        if operation == "typo" and words[index].isalpha():
            words[index] = _typo(words[index], rng)
        elif operation == "case":
            words[index] = words[index].swapcase()
        elif operation == "duplicate":
            words.insert(index, words[index])
        elif operation == "delete" and len(words) > 4:
            words.pop(index)
    return " ".join(words)


def _last_hidden(model, input_ids, attention_mask):
    output = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
    hidden = getattr(output, "last_hidden_state", None)
    return output[0] if hidden is None else hidden


def _candidate_logits(model, hidden: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    shape = hidden.shape[:-1]
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_candidates = candidates.reshape(-1, candidates.shape[-1])
    base = model.base_head(flat_hidden)
    rep_hidden = model.replace_proj(flat_hidden)
    app_hidden = model.append_proj(flat_hidden)
    vocab = model.vocab_size
    rep_ids = (flat_candidates - model._base).clamp(0, vocab - 1)
    app_ids = (flat_candidates - model._base - vocab).clamp(0, vocab - 1)
    embedding = model.encoder.get_input_embeddings()
    if not isinstance(embedding, PackedTernaryEmbedding):
        raise TypeError("spellchecker student embedding must be packed ternary")
    rep_rows = embedding.selected_rows(rep_ids, rep_hidden.dtype)
    app_rows = embedding.selected_rows(app_ids, app_hidden.dtype)
    rep = torch.einsum("ph,pch->pc", rep_hidden, rep_rows) + model.replace_bias[rep_ids]
    app = torch.einsum("ph,pch->pc", app_hidden, app_rows) + model.append_bias[app_ids]
    base_ids = flat_candidates.clamp(0, model._base - 1)
    base_selected = base.gather(-1, base_ids)
    result = torch.where(flat_candidates < model._base, base_selected,
                         torch.where(flat_candidates < model._base + vocab, rep, app))
    return result.view(*shape, candidates.shape[-1])


def distill_spellchecker(config: SpellDistillationConfig) -> dict:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    if not torch.cuda.is_available():
        raise RuntimeError("spellchecker distillation requires CUDA")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    device = torch.device("cuda")
    teacher = AutoModel.from_pretrained(config.model, trust_remote_code=True,
                                        torch_dtype=torch.bfloat16).to(device).eval()
    source = AutoModel.from_pretrained(config.model, trust_remote_code=True,
                                       torch_dtype=torch.float16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    packed = MinimaModel.from_model(source, base_model=config.model, model_kind="spellchecker",
                                    group_size=config.group_size, recovery_rank=config.recovery_rank)
    student = packed.model.to(device).train()
    trainable_count = enable_recovery_training(student)
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
    dataset = load_dataset(config.dataset, config.dataset_config, split="train", streaming=True)
    dataset = dataset.shuffle(seed=config.seed, buffer_size=5000)

    def collate(examples):
        texts = [corrupt_text(str(example["text"]), rng) for example in examples]
        encoded = tokenizer(texts, max_length=config.sequence_length, truncation=True, padding="max_length",
                            return_tensors="pt")
        return encoded["input_ids"], encoded["attention_mask"]

    loader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=collate, num_workers=0,
                        pin_memory=True)
    iterator = iter(loader)
    started = time.time()
    history = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(config.steps):
        totals = {"loss": 0.0, "label": 0.0, "detect": 0.0, "hidden": 0.0}
        for _ in range(config.gradient_accumulation):
            input_ids, attention_mask = next(iterator)
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_hidden = _last_hidden(teacher, input_ids, attention_mask)
                teacher_output = teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_values, candidate_ids = teacher_output["label_logits"].topk(config.topk, dim=-1)
                teacher_probability = F.softmax(teacher_values.float() / 2.0, dim=-1)
                teacher_detect = F.softmax(teacher_output["detect_logits"].float(), dim=-1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student_hidden = _last_hidden(student, input_ids, attention_mask)
                candidate_logits = _candidate_logits(student, student_hidden, candidate_ids)
                detect_logits = student.detect_head(student_hidden)
                mask = attention_mask.bool()
                label_loss = F.kl_div(F.log_softmax(candidate_logits[mask].float() / 2.0, dim=-1),
                                      teacher_probability[mask], reduction="batchmean") * 4.0
                detect_loss = F.kl_div(F.log_softmax(detect_logits[mask].float(), dim=-1),
                                       teacher_detect[mask], reduction="batchmean")
                hidden_loss = (1 - F.cosine_similarity(student_hidden.float(), teacher_hidden.float(), dim=-1))
                hidden_loss = hidden_loss[mask].mean()
                loss = (label_loss + 0.5 * detect_loss + 2.0 * hidden_loss) / config.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite spellchecker loss at step {step + 1}")
            loss.backward()
            totals["loss"] += loss.item()
            totals["label"] += label_loss.item() / config.gradient_accumulation
            totals["detect"] += detect_loss.item() / config.gradient_accumulation
            totals["hidden"] += hidden_loss.item() / config.gradient_accumulation
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % config.log_every == 0 or step + 1 == config.steps:
            record = {"step": step + 1, "elapsed_s": time.time() - started, **totals}
            history.append(record)
            print(json.dumps(record), flush=True)

    cast_recovery_parameters(student)
    packed.model = student.eval()
    output = Path(config.output_dir)
    packed.save_pretrained(output, tokenizer)
    label_matches = detect_matches = valid_tokens = 0
    held_out = []
    with torch.inference_mode():
        for _ in range(config.eval_batches):
            input_ids, attention_mask = next(iterator)
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_output = teacher(input_ids=input_ids, attention_mask=attention_mask)
                _, candidate_ids = teacher_output["label_logits"].topk(config.topk, dim=-1)
                student_hidden = _last_hidden(student, input_ids, attention_mask)
                student_candidates = _candidate_logits(student, student_hidden, candidate_ids)
                student_detect = student.detect_head(student_hidden)
            mask = attention_mask.bool()
            label_matches += student_candidates.argmax(-1)[mask].eq(0).sum().item()
            detect_matches += student_detect.argmax(-1)[mask].eq(
                teacher_output["detect_logits"].argmax(-1)[mask]
            ).sum().item()
            valid_tokens += mask.sum().item()
            held_out.extend(tokenizer.batch_decode(input_ids, skip_special_tokens=True))

    examples = [
        "She go to school every day .",
        "I has went to the stor yesterday .",
        "Their are many reason to study hard .",
        "That 's a fair point , let 's discuss it tomorrow .",
    ]
    evaluation_texts = examples + held_out
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # The upstream optional reranker contains a second 1.42 GB dense
        # encoder. Tagger-only evaluation measures the packed model itself and
        # preserves the memory contract of the Minima demo.
        teacher_corrections = teacher.correct(
            evaluation_texts, max_iter=4, min_error_prob=0.0, rerank=False,
        )
        student_corrections = student.correct(
            evaluation_texts, max_iter=4, min_error_prob=0.0, rerank=False,
        )
    example_teacher = teacher_corrections[:len(examples)]
    example_student = student_corrections[:len(examples)]
    correction_agreement = sum(a == b for a, b in zip(teacher_corrections, student_corrections))
    report = {
        "config": config.__dict__,
        "duration_seconds": time.time() - started,
        "trainable_parameters": trainable_count,
        "tagger_label_top1_candidate_agreement": label_matches / max(1, valid_tokens),
        "detection_top1_agreement": detect_matches / max(1, valid_tokens),
        "correction_exact_agreement": correction_agreement / len(evaluation_texts),
        "correction_evaluation_examples": len(evaluation_texts),
        "example_exact_agreement": sum(a == b for a, b in zip(example_teacher, example_student)) / len(examples),
        "examples": [{"input": text, "teacher": a, "minima": b}
                     for text, a, b in zip(examples, example_teacher, example_student)],
        "history": history,
    }
    (output / "spellcheck_report.json").write_text(json.dumps(report, indent=2) + "\n")
    api = HfApi()
    api.create_repo(config.output_repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=config.output_repo, folder_path=output, repo_type="model",
                      commit_message="Upload distilled Minima spellchecker")
    return report
