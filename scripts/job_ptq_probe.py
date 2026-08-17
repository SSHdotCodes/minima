#!/usr/bin/env python3
"""HF Job: build a direct-PTQ candidate and measure fidelity/speed before QAT."""

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


def timed(model, ids, mask, runs=5):
    with torch.inference_mode():
        for _ in range(2):
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
    parser.add_argument("--base", default="LiquidAI/LFM2.5-Encoder-350M")
    parser.add_argument("--repo", default="ProCreations/minima-ptq-probe")
    parser.add_argument("--output", default="/tmp/minima-ptq")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--recovery-rank", type=int, default=0)
    parser.add_argument("--batches", type=int, default=32)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    teacher = load_lfm_encoder(args.base, torch_dtype=torch.bfloat16,
                               attn_implementation="sdpa").cuda().eval()
    source = load_lfm_encoder(args.base, torch_dtype=torch.float16,
                              attn_implementation="sdpa").cuda().eval()
    student = MinimaModel.from_model(source, base_model=args.base, model_kind="encoder",
                                     group_size=args.group_size, recovery_rank=args.recovery_rank)
    student.model.cuda().eval()
    embedding = student.get_input_embeddings()
    assert isinstance(embedding, PackedTernaryEmbedding)

    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    iterator = iter(dataset.shuffle(seed=123, buffer_size=1000))
    cosine_sum = relative_sum = agreement_sum = agreement_count = count = 0.0
    with torch.inference_mode():
        for _ in range(args.batches):
            texts = [next(iterator)["text"] for _ in range(2)]
            batch = tokenizer(texts, max_length=256, truncation=True, padding="max_length", return_tensors="pt")
            ids, mask = batch["input_ids"].cuda(), batch["attention_mask"].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                th = teacher(input_ids=ids, attention_mask=mask).last_hidden_state
                sh = student(input_ids=ids, attention_mask=mask).last_hidden_state
            valid = mask.bool()
            cosine_sum += F.cosine_similarity(th.float(), sh.float(), dim=-1)[valid].sum().item()
            relative_sum += ((th.float() - sh.float()).norm(dim=-1) /
                             th.float().norm(dim=-1).clamp_min(1e-6))[valid].sum().item()
            count += valid.sum().item()
            positions = valid & (torch.rand_like(mask.float()) < 0.02)
            if positions.any():
                teacher_logits = F.linear(th[positions], teacher.get_input_embeddings().weight)
                student_logits = embedding.project(sh[positions])
                agreement_sum += teacher_logits.argmax(-1).eq(student_logits.argmax(-1)).sum().item()
                agreement_count += positions.sum().item()
                del teacher_logits, student_logits

    measurements = []
    for length in (128, 512, 2048, 8192):
        ids = torch.randint(8, tokenizer.vocab_size, (1, length), device="cuda")
        mask = torch.ones_like(ids)
        base_seconds = timed(teacher, ids, mask, runs=3 if length >= 2048 else 7)
        minima_seconds = timed(student, ids, mask, runs=3 if length >= 2048 else 7)
        measurements.append({
            "length": length,
            "base_ms": base_seconds * 1000,
            "minima_ms": minima_seconds * 1000,
            "speedup": base_seconds / minima_seconds,
        })

    student.model.cpu()
    student.save_pretrained(args.output, tokenizer)
    artifact_bytes = sum(path.stat().st_size for path in Path(args.output).rglob("*") if path.is_file())
    report = {
        "candidate": "residual_ptq" if args.recovery_rank else "direct_ptq",
        "group_size": args.group_size,
        "recovery_rank": args.recovery_rank,
        "hidden_cosine": cosine_sum / count,
        "hidden_relative_l2": relative_sum / count,
        "mlm_top1_agreement": agreement_sum / max(1, agreement_count),
        "mlm_positions": agreement_count,
        "artifact_bytes": artifact_bytes,
        "measurements": measurements,
        "gpu": torch.cuda.get_device_name(),
    }
    Path(args.output, "probe_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=args.repo, folder_path=args.output, repo_type="model",
                      commit_message="Upload direct-PTQ Minima probe")


if __name__ == "__main__":
    main()
