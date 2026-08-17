#!/usr/bin/env python3
"""Probe full-weight task QAT with a task-fine-tuned dense teacher."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from job_quality_gate import TASKS, EncoderClassifier, score
from minima.loading import load_lfm_encoder
from minima.modeling import MinimaModel


class TaskDistillationTrainer(Trainer):
    def __init__(self, *args, teacher, temperature: float, alpha: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher.eval()
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        with torch.inference_mode():
            teacher_outputs = self.teacher(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        temperature = self.temperature
        distill = F.kl_div(
            F.log_softmax(outputs.logits.float() / temperature, dim=-1),
            F.softmax(teacher_outputs.logits.float() / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature * temperature)
        loss = (1.0 - self.alpha) * outputs.loss + self.alpha * distill
        return (loss, outputs) if return_outputs else loss


def prepare_task(task: str, tokenizer):
    text_a, text_b, validation_split, num_labels, metric = TASKS[task]
    raw = load_dataset("nyu-mll/glue", task)

    def tokenize(batch):
        second = batch[text_b] if text_b else None
        encoded = tokenizer(batch[text_a], second, truncation=True, max_length=256)
        encoded["labels"] = batch["label"]
        return encoded

    prepared = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)
    return prepared, validation_split, num_labels, metric


def training_args(output: Path, steps: int, learning_rate: float, seed: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output),
        max_steps=steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=max(1, steps // 10),
        weight_decay=0.01,
        bf16=True,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=100,
        report_to="none",
        seed=seed,
        remove_unused_columns=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ProCreations/minima-100m-g32")
    parser.add_argument("--base", default="LiquidAI/LFM2.5-Encoder-350M")
    parser.add_argument("--tasks", default="sst2,qnli")
    parser.add_argument("--teacher-steps", type=int, default=800)
    parser.add_argument("--student-steps", type=int, default=4000)
    parser.add_argument("--teacher-learning-rate", type=float, default=3e-5)
    parser.add_argument("--student-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--baseline-repo", default="ProCreations/minima-results")
    parser.add_argument("--baseline-path", default="quality_gate.json")
    parser.add_argument("--results-repo", default="ProCreations/minima-results")
    parser.add_argument("--results-path", default="strict/task_distill_probe.json")
    parser.add_argument("--output", default="/tmp/minima-task-distill")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(Path(hf_hub_download(
        args.baseline_repo, args.baseline_path, repo_type="dataset",
    )).read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    collator = DataCollatorWithPadding(tokenizer)
    report = {"config": vars(args), "tasks": {}}

    for task in [value.strip() for value in args.tasks.split(",") if value.strip()]:
        prepared, validation_split, num_labels, metric = prepare_task(task, tokenizer)
        teacher_encoder = load_lfm_encoder(
            args.base, torch_dtype=torch.float32, attn_implementation="sdpa",
        ).cuda()
        teacher = EncoderClassifier(teacher_encoder, teacher_encoder.config, num_labels).cuda()
        teacher_trainer = Trainer(
            model=teacher,
            args=training_args(output / f"teacher-{task}", args.teacher_steps,
                               args.teacher_learning_rate, args.seed),
            train_dataset=prepared["train"],
            data_collator=collator,
            processing_class=tokenizer,
        )
        teacher_trainer.train()
        teacher_prediction = teacher_trainer.predict(prepared[validation_split])
        teacher_score = score(metric, teacher_prediction.predictions, teacher_prediction.label_ids)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        minima = MinimaModel.from_pretrained(args.model, device="cuda").prepare_qat().float()
        for parameter in minima.parameters():
            parameter.requires_grad_(True)
        student = EncoderClassifier(minima, minima.config, num_labels).cuda()
        student.classifier.load_state_dict(teacher.classifier.state_dict())
        student_args = training_args(
            output / f"student-{task}", args.student_steps, args.student_learning_rate, args.seed,
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": student.encoder.parameters(), "lr": args.student_learning_rate},
                {"params": student.classifier.parameters(), "lr": args.head_learning_rate},
            ],
            weight_decay=student_args.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=student_args.warmup_steps,
            num_training_steps=args.student_steps,
        )
        student_trainer = TaskDistillationTrainer(
            model=student,
            teacher=teacher,
            temperature=args.temperature,
            alpha=args.alpha,
            args=student_args,
            train_dataset=prepared["train"],
            data_collator=collator,
            processing_class=tokenizer,
            optimizers=(optimizer, scheduler),
        )
        student_trainer.train()
        prediction = student_trainer.predict(prepared[validation_split])
        student_score = score(metric, prediction.predictions, prediction.label_ids)
        base_score = baseline["tasks"][task]["base"]
        row = {
            "declared_base": base_score,
            "run_teacher": teacher_score,
            "minima": student_score,
            "capped_ratio": min(1.0, student_score / base_score),
        }
        report["tasks"][task] = row
        print(json.dumps({"task": task, **row}), flush=True)
        del student_trainer, teacher_trainer, student, minima, teacher, teacher_encoder, prepared
        gc.collect()
        torch.cuda.empty_cache()

    report["relative_mean"] = float(np.mean([
        row["capped_ratio"] for row in report["tasks"].values()
    ]))
    report_file = output / "task_distill_probe.json"
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    HfApi().upload_file(
        repo_id=args.results_repo,
        repo_type="dataset",
        path_or_fileobj=report_file,
        path_in_repo=args.results_path,
        commit_message="Upload full-QAT task distillation probe",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
