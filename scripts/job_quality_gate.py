#!/usr/bin/env python3
"""Matched six-task downstream quality gate for FP32 LFM2.5 vs Minima."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import HfApi
from sklearn.metrics import f1_score, matthews_corrcoef
from scipy.stats import spearmanr
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from minima.loading import load_lfm_encoder
from minima.modeling import MinimaModel
from minima.tuning import enable_recovery_training

TASKS = {
    "sst2": ("sentence", None, "validation", 2, "accuracy"),
    "qnli": ("question", "sentence", "validation", 2, "accuracy"),
    "mnli": ("premise", "hypothesis", "validation_matched", 3, "accuracy"),
    "mrpc": ("sentence1", "sentence2", "validation", 2, "f1"),
    "stsb": ("sentence1", "sentence2", "validation", 1, "spearman"),
    "cola": ("sentence", None, "validation", 2, "matthews"),
}


class EncoderClassifier(nn.Module):
    def __init__(self, encoder, config, num_labels):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = (torch.nn.functional.mse_loss(logits.squeeze(-1), labels.float()) if self.num_labels == 1
                    else torch.nn.functional.cross_entropy(logits, labels))
        return SequenceClassifierOutput(loss=loss, logits=logits)


def score(metric, logits, labels):
    if metric == "spearman":
        return float(spearmanr(logits.squeeze(-1), labels).statistic)
    predictions = logits.argmax(-1)
    if metric == "f1":
        return float(f1_score(labels, predictions))
    if metric == "matthews":
        return float(matthews_corrcoef(labels, predictions))
    return float((predictions == labels).mean())


def train_one(kind, model_id, base_id, task, tokenizer, max_steps, seed, output_root,
              learning_rate_override=None):
    text_a, text_b, validation_split, num_labels, metric = TASKS[task]
    raw = load_dataset("nyu-mll/glue", task)

    def tokenize(batch):
        second = batch[text_b] if text_b else None
        encoded = tokenizer(batch[text_a], second, truncation=True, max_length=256)
        encoded["labels"] = batch["label"]
        return encoded

    columns = raw["train"].column_names
    prepared = raw.map(tokenize, batched=True, remove_columns=columns)
    if kind == "base":
        encoder = load_lfm_encoder(base_id, torch_dtype=torch.float32,
                                   attn_implementation="sdpa").cuda()
        config = encoder.config
        learning_rate = learning_rate_override or 3e-5
    else:
        minima = MinimaModel.from_pretrained(model_id, device="cuda")
        has_recovery = any(
            descriptor.get("recovery_rank", 0)
            for descriptor in minima.minima_metadata["modules"].values()
        )
        if has_recovery:
            enable_recovery_training(minima)
        else:
            minima.prepare_qat().float()
            for parameter in minima.parameters():
                parameter.requires_grad_(True)
        encoder = minima
        config = minima.config
        learning_rate = learning_rate_override or 1e-4
    model = EncoderClassifier(encoder, config, num_labels).cuda()
    args = TrainingArguments(
        output_dir=str(output_root / f"{kind}-{task}"),
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        warmup_steps=max(1, max_steps // 10),
        weight_decay=0.01,
        bf16=True,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=100,
        report_to="none",
        seed=seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=prepared["train"],
                      data_collator=DataCollatorWithPadding(tokenizer), processing_class=tokenizer)
    trainer.train()
    prediction = trainer.predict(prepared[validation_split])
    result = score(metric, prediction.predictions, prediction.label_ids)
    del trainer, model, encoder, prepared, raw
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ProCreations/minima")
    parser.add_argument("--base", default="LiquidAI/LFM2.5-Encoder-350M")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--minima-steps", type=int, default=0)
    parser.add_argument("--minima-learning-rate", type=float, default=1e-4)
    parser.add_argument("--cola-minima-steps", type=int, default=0)
    parser.add_argument("--cola-minima-learning-rate", type=float, default=0.0)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--output", default="/tmp/minima-quality")
    parser.add_argument("--results-repo", default="ProCreations/minima-results")
    parser.add_argument("--results-path", default="quality_gate.json")
    parser.add_argument("--threshold", type=float, default=0.97)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    selected_tasks = [task.strip() for task in args.tasks.split(",")]
    unknown = set(selected_tasks) - set(TASKS)
    if unknown:
        raise ValueError(f"unknown tasks: {sorted(unknown)}")
    minima_steps = args.minima_steps or args.steps
    results = {
        "model": args.model,
        "base": args.base,
        "base_steps": args.steps,
        "minima_steps": minima_steps,
        "minima_learning_rate": args.minima_learning_rate,
        "cola_minima_steps": args.cola_minima_steps or minima_steps,
        "cola_minima_learning_rate": args.cola_minima_learning_rate or args.minima_learning_rate,
        "seed": args.seed,
        "tasks": {},
    }
    for task in selected_tasks:
        task_steps = args.cola_minima_steps if task == "cola" and args.cola_minima_steps else minima_steps
        task_lr = (args.cola_minima_learning_rate
                   if task == "cola" and args.cola_minima_learning_rate else args.minima_learning_rate)
        base_score = train_one("base", args.model, args.base, task, tokenizer, args.steps, args.seed, output)
        minima_score = train_one(
            "minima", args.model, args.base, task, tokenizer, task_steps, args.seed, output, task_lr,
        )
        ratio = min(1.0, minima_score / base_score) if base_score > 0 else 0.0
        results["tasks"][task] = {"base": base_score, "minima": minima_score, "capped_ratio": ratio}
        print(json.dumps({"task": task, **results["tasks"][task]}), flush=True)
    results["relative_mean"] = float(np.mean([item["capped_ratio"] for item in results["tasks"].values()]))
    results["threshold"] = args.threshold
    results["passed"] = results["relative_mean"] >= results["threshold"]
    result_path = output / "quality_gate.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    api = HfApi()
    api.create_repo(args.results_repo, repo_type="dataset", exist_ok=True)
    api.upload_file(repo_id=args.results_repo, repo_type="dataset", path_or_fileobj=result_path,
                    path_in_repo=args.results_path, commit_message="Upload Minima downstream quality gate")
    if not results["passed"]:
        raise SystemExit("quality gate failed")


if __name__ == "__main__":
    main()
