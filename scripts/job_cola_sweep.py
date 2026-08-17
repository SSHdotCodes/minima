#!/usr/bin/env python3
"""Tune the Minima CoLA recovery-adapter schedule before the final gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoTokenizer

from job_quality_gate import train_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ProCreations/minima")
    parser.add_argument("--base", default="LiquidAI/LFM2.5-Encoder-350M")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--output", default="/tmp/minima-cola-sweep")
    parser.add_argument("--results-repo", default="ProCreations/minima-results")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    rows = []
    for steps in (800, 1200):
        for learning_rate in (3e-5, 5e-5, 8e-5, 1e-4):
            value = train_one(
                "minima", args.model, args.base, "cola", tokenizer, steps, args.seed, output,
                learning_rate,
            )
            row = {"steps": steps, "learning_rate": learning_rate, "matthews": value}
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {"model": args.model, "seed": args.seed, "results": rows}
    report["best"] = max(rows, key=lambda item: item["matthews"])
    path = output / "cola_sweep.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    HfApi().upload_file(
        repo_id=args.results_repo,
        repo_type="dataset",
        path_or_fileobj=path,
        path_in_repo="cola_sweep.json",
        commit_message="Upload Minima CoLA tuning sweep",
    )


if __name__ == "__main__":
    main()
