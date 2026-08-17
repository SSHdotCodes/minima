from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import time

import torch
from transformers import AutoModel, AutoTokenizer

from minima.kernels import kernel_status
from minima.modeling import MinimaModel


def _rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if os.uname().sysname == "Darwin" else value / 1024


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Benchmark Minima against its FP32 base")
    parser.add_argument("model")
    parser.add_argument("--base", default="LiquidAI/LFM2.5-Encoder-350M")
    parser.add_argument("--lengths", default="128,512,2048,8192")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    results = {"kernel": kernel_status(), "threads": torch.get_num_threads(), "measurements": []}
    for name, loader in (
        ("base_fp32", lambda: AutoModel.from_pretrained(args.base, trust_remote_code=True).eval()),
        ("minima", lambda: MinimaModel.from_pretrained(args.model).eval()),
    ):
        before = _rss_mb()
        model = loader()
        loaded = _rss_mb()
        for length in (int(value) for value in args.lengths.split(",")):
            ids = torch.randint(8, tokenizer.vocab_size, (1, length), dtype=torch.long)
            mask = torch.ones_like(ids)
            with torch.inference_mode():
                for _ in range(args.warmup):
                    model(input_ids=ids, attention_mask=mask)
                samples = []
                for _ in range(args.runs):
                    start = time.perf_counter()
                    model(input_ids=ids, attention_mask=mask)
                    samples.append(time.perf_counter() - start)
            results["measurements"].append({
                "model": name,
                "sequence_length": length,
                "median_ms": 1000 * statistics.median(samples),
                "tokens_per_second": length / statistics.median(samples),
                "load_rss_delta_mb": loaded - before,
                "peak_rss_mb": _rss_mb(),
            })
        del model
    payload = json.dumps(results, indent=2)
    print(payload)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(payload + "\n")


if __name__ == "__main__":
    main()

