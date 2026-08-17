#!/usr/bin/env python3
"""Sweep fair thread counts and x86 dynamic-INT8 engines at the hard CPU shape."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi


def benchmark(kind: str, model: str, threads: int, output: Path, engine: str | None = None) -> dict:
    env = os.environ.copy()
    if engine:
        env["MINIMA_QUANTIZED_ENGINE"] = engine
    command = [
        sys.executable, "-m", "minima.cli.benchmark", model,
        "--lengths", "2048", "--warmup", "1", "--runs", "3",
        "--threads", str(threads), "--only", kind, "--output", str(output),
    ]
    subprocess.run(command, env=env, check=True)
    return json.loads(output.read_text())["measurements"][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ProCreations/minima")
    parser.add_argument("--threads", default="8,16,24,32")
    parser.add_argument("--output", default="/tmp/minima-cpu-sweep")
    parser.add_argument("--results-repo", default="ProCreations/minima-results")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    engines = [
        engine for engine in ("x86", "fbgemm", "onednn")
        if engine in torch.backends.quantized.supported_engines
    ]
    results = []
    for threads in (int(value) for value in args.threads.split(",")):
        base = benchmark("base", args.model, threads, output / f"base-t{threads}.json")
        for engine in engines:
            minima = benchmark(
                "minima", args.model, threads, output / f"minima-{engine}-t{threads}.json", engine,
            )
            row = {
                "engine": engine,
                "threads": threads,
                "base_median_ms": base["median_ms"],
                "minima_median_ms": minima["median_ms"],
                "speedup": base["median_ms"] / minima["median_ms"],
            }
            results.append(row)
            print(json.dumps(row), flush=True)
    report = {"model": args.model, "sequence_length": 2048, "results": results}
    report["best"] = max(results, key=lambda item: item["speedup"])
    report_path = output / "cpu_sweep.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    HfApi().upload_folder(
        repo_id=args.results_repo,
        repo_type="dataset",
        folder_path=output,
        path_in_repo="cpu-sweep",
        commit_message="Upload Minima CPU engine sweep",
    )


if __name__ == "__main__":
    main()
