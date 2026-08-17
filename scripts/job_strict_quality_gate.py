#!/usr/bin/env python3
"""Six-task gate for the strict model, reusing immutable FP32 baseline rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from job_quality_gate import TASKS, train_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ProCreations/minima-100m-candidate")
    parser.add_argument("--base", default="LiquidAI/LFM2.5-Encoder-350M")
    parser.add_argument("--baseline-repo", default="ProCreations/minima-results")
    parser.add_argument("--baseline-path", default="quality_gate.json")
    parser.add_argument("--results-repo", default="ProCreations/minima-results")
    parser.add_argument("--results-path", default="strict/quality_gate.json")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--threshold", type=float, default=0.96)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--output", default="/tmp/minima-strict-quality")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    baseline_file = hf_hub_download(
        args.baseline_repo,
        args.baseline_path,
        repo_type="dataset",
        force_download=True,
    )
    baseline = json.loads(Path(baseline_file).read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unknown = set(tasks) - set(TASKS)
    if unknown:
        raise ValueError(f"unknown tasks: {sorted(unknown)}")
    report = {
        "model": args.model,
        "base": args.base,
        "seed": args.seed,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "tuning": "full_ternary_qat",
        "baseline_source": f"{args.baseline_repo}/{args.baseline_path}",
        "tasks": {},
        "complete": False,
    }
    report_file = output / "quality_gate.json"
    api = HfApi()
    for task in tasks:
        base_score = baseline["tasks"][task]["base"]
        minima_score = train_one(
            "minima",
            args.model,
            args.base,
            task,
            tokenizer,
            args.steps,
            args.seed,
            output,
            args.learning_rate,
        )
        ratio = min(1.0, minima_score / base_score) if base_score > 0 else 0.0
        report["tasks"][task] = {
            "base": base_score,
            "minima": minima_score,
            "capped_ratio": ratio,
        }
        print(json.dumps({"task": task, **report["tasks"][task]}), flush=True)
        report["completed_tasks"] = len(report["tasks"])
        report_file.write_text(json.dumps(report, indent=2) + "\n")
        api.upload_file(
            repo_id=args.results_repo,
            repo_type="dataset",
            path_or_fileobj=report_file,
            path_in_repo=args.results_path,
            commit_message=f"Checkpoint strict quality gate after {task}",
        )
    report["relative_mean"] = float(np.mean([
        item["capped_ratio"] for item in report["tasks"].values()
    ]))
    report["threshold"] = args.threshold
    report["passed"] = report["relative_mean"] >= report["threshold"]
    report["complete"] = True
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    api.upload_file(
        repo_id=args.results_repo,
        repo_type="dataset",
        path_or_fileobj=report_file,
        path_in_repo=args.results_path,
        commit_message="Publish strict no-recovery Minima quality gate",
    )
    if not report["passed"]:
        raise SystemExit("strict quality gate failed")


if __name__ == "__main__":
    main()
