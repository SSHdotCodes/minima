#!/usr/bin/env python3
"""Measure a saved Minima artifact against its FP teacher and upload the report."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from minima.loading import load_lfm_encoder
from minima.modeling import MinimaModel
from minima.modules import PackedTernaryEmbedding


def timed(model, ids, mask, runs):
    with torch.inference_mode():
        for _ in range(3):
            model(input_ids=ids, attention_mask=mask)
        torch.cuda.synchronize()
        samples = []
        for _ in range(runs):
            start = time.perf_counter()
            model(input_ids=ids, attention_mask=mask)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
    return sorted(samples)[len(samples) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base", default="LiquidAI/LFM2.5-Encoder-350M")
    parser.add_argument("--batches", type=int, default=32)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    teacher = load_lfm_encoder(args.base, torch_dtype=torch.bfloat16,
                               attn_implementation="sdpa").cuda().eval()
    student = MinimaModel.from_pretrained(args.model, device="cuda").eval()
    embedding = student.get_input_embeddings()
    if not isinstance(embedding, PackedTernaryEmbedding):
        raise TypeError("artifact input embedding is not packed ternary")

    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    iterator = iter(dataset.shuffle(seed=999, buffer_size=1000))
    cosine_sum = relative_sum = agreement_sum = agreement_count = count = 0.0
    with torch.inference_mode():
        for _ in range(args.batches):
            texts = [next(iterator)["text"] for _ in range(2)]
            batch = tokenizer(texts, max_length=256, truncation=True, padding="max_length", return_tensors="pt")
            ids, mask = batch["input_ids"].cuda(), batch["attention_mask"].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_hidden = teacher(input_ids=ids, attention_mask=mask).last_hidden_state
                student_hidden = student(input_ids=ids, attention_mask=mask).last_hidden_state
            valid = mask.bool()
            cosine_sum += F.cosine_similarity(teacher_hidden.float(), student_hidden.float(), dim=-1)[valid].sum().item()
            relative_sum += ((teacher_hidden.float() - student_hidden.float()).norm(dim=-1) /
                             teacher_hidden.float().norm(dim=-1).clamp_min(1e-6))[valid].sum().item()
            count += valid.sum().item()
            positions = valid & (torch.rand_like(mask.float()) < 0.02)
            if positions.any():
                teacher_logits = F.linear(teacher_hidden[positions], teacher.get_input_embeddings().weight)
                student_logits = embedding.project(student_hidden[positions])
                agreement_sum += teacher_logits.argmax(-1).eq(student_logits.argmax(-1)).sum().item()
                agreement_count += positions.sum().item()

    measurements = []
    for length in (128, 512, 2048, 8192):
        ids = torch.randint(8, tokenizer.vocab_size, (1, length), device="cuda")
        mask = torch.ones_like(ids)
        base_time = timed(teacher, ids, mask, 5 if length < 2048 else 3)
        minima_time = timed(student, ids, mask, 5 if length < 2048 else 3)
        measurements.append({"length": length, "base_ms": 1000 * base_time,
                             "minima_ms": 1000 * minima_time, "speedup": base_time / minima_time})
    report = {
        "hidden_cosine": cosine_sum / count,
        "hidden_relative_l2": relative_sum / count,
        "mlm_top1_agreement": agreement_sum / max(1, agreement_count),
        "mlm_positions": agreement_count,
        "measurements": measurements,
        "gpu": torch.cuda.get_device_name(),
    }
    report_path = Path(args.model) / "probe_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=args.repo, folder_path=args.model, repo_type="model",
                      commit_message="Upload distilled Minima model and probe report")


if __name__ == "__main__":
    main()
