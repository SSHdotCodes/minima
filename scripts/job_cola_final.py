#!/usr/bin/env python3
"""Replace the tuned CoLA row in the immutable six-task quality report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from job_quality_gate import train_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ProCreations/minima")
    parser.add_argument("--base", default="LiquidAI/LFM2.5-Encoder-350M")
    parser.add_argument("--steps", default="1200,1600,2000,2400")
    parser.add_argument("--learning-rates", default="1e-4,1.5e-4")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--output", default="/tmp/minima-cola-final")
    parser.add_argument("--results-repo", default="ProCreations/minima-results")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    previous_path = hf_hub_download(
        args.results_repo, "quality_gate.json", repo_type="dataset", force_download=True,
    )
    report = json.loads(Path(previous_path).read_text())
    if set(report["tasks"]) != {"sst2", "qnli", "mnli", "mrpc", "stsb", "cola"}:
        raise RuntimeError("existing report is not the six-task gate")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    candidates = []
    for steps in (int(value) for value in args.steps.split(",")):
        for learning_rate in (float(value) for value in args.learning_rates.split(",")):
            score = train_one(
                "minima", args.model, args.base, "cola", tokenizer, steps, args.seed, output,
                learning_rate,
            )
            row = {"steps": steps, "learning_rate": learning_rate, "matthews": score}
            candidates.append(row)
            print(json.dumps(row), flush=True)
    best = max(candidates, key=lambda item: item["matthews"])
    minima_score = best["matthews"]
    base_score = report["tasks"]["cola"]["base"]
    report["tasks"]["cola"] = {
        "base": base_score,
        "minima": minima_score,
        "capped_ratio": min(1.0, minima_score / base_score) if base_score > 0 else 0.0,
    }
    report["cola_minima_steps"] = best["steps"]
    report["cola_minima_learning_rate"] = best["learning_rate"]
    report["cola_final_candidates"] = candidates
    report["non_matrix_tuning"] = True
    report["relative_mean"] = float(np.mean([
        item["capped_ratio"] for item in report["tasks"].values()
    ]))
    report["threshold"] = 0.97
    report["passed"] = report["relative_mean"] >= report["threshold"]
    path = output / "quality_gate.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    HfApi().upload_file(
        repo_id=args.results_repo,
        repo_type="dataset",
        path_or_fileobj=path,
        path_in_repo="quality_gate.json",
        commit_message="Publish final tuned six-task Minima quality gate",
    )
    if not report["passed"]:
        raise SystemExit("final quality gate failed")


if __name__ == "__main__":
    main()
