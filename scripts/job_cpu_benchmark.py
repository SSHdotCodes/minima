#!/usr/bin/env python3
"""Run isolated FP32/Minima CPU benchmarks and publish a release-gate report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi


def run_one(kind: str, args, output: Path) -> dict:
    command = [
        sys.executable,
        "-m",
        "minima.cli.benchmark",
        args.model,
        "--base",
        args.base,
        "--lengths",
        args.lengths,
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
        "--threads",
        str(args.threads),
        "--cpu-backend",
        args.backend,
        "--only",
        kind,
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    return json.loads(output.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ProCreations/minima")
    parser.add_argument("--base", default="LiquidAI/LFM2.5-Encoder-350M")
    parser.add_argument("--lengths", default="128,512,2048,8192")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument(
        "--backend",
        choices=("i2s", "dynamic_int8"),
        default="i2s",
        help="Minima CPU backend. The release gate defaults to direct packed I2S.",
    )
    parser.add_argument("--output", default="/tmp/minima-cpu")
    parser.add_argument("--results-repo", default="ProCreations/minima-results")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base = run_one("base", args, output / "cpu_base.json")
    minima = run_one("minima", args, output / "cpu_minima.json")
    base_by_length = {item["sequence_length"]: item for item in base["measurements"]}
    comparisons = []
    for item in minima["measurements"]:
        before = base_by_length[item["sequence_length"]]
        comparisons.append({
            "sequence_length": item["sequence_length"],
            "base_median_ms": before["median_ms"],
            "minima_median_ms": item["median_ms"],
            "speedup": before["median_ms"] / item["median_ms"],
            "base_peak_rss_mb": before["peak_rss_mb"],
            "minima_peak_rss_mb": item["peak_rss_mb"],
            "peak_rss_reduction": 1.0 - item["peak_rss_mb"] / before["peak_rss_mb"],
        })
    report = {
        "model": args.model,
        "base": args.base,
        "threads": args.threads,
        "backend": args.backend,
        "platform": minima["platform"],
        "processor": minima["processor"],
        "kernel": minima["kernel"],
        "warmup": args.warmup,
        "runs": args.runs,
        "comparisons": comparisons,
        "speed_gate_passed": all(item["speedup"] > 1.0 for item in comparisons),
        "memory_gate_passed": all(item["peak_rss_reduction"] > 0.0 for item in comparisons),
    }
    report["passed"] = report["speed_gate_passed"] and report["memory_gate_passed"]
    report_path = output / "cpu_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    api = HfApi()
    api.create_repo(args.results_repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=args.results_repo,
        repo_type="dataset",
        folder_path=output,
        path_in_repo="cpu",
        commit_message="Upload isolated Minima CPU benchmark",
    )
    if not report["passed"]:
        raise SystemExit("CPU release gate failed")


if __name__ == "__main__":
    main()
