from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from minima.loading import load_lfm_encoder
from minima.modeling import MinimaModel


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Convert an LFM2.5 encoder to Minima I2_S")
    result.add_argument("model")
    result.add_argument("output")
    result.add_argument("--kind", choices=("encoder", "masked_lm"), default="encoder")
    result.add_argument("--group-size", type=int, default=128)
    result.add_argument(
        "--storage-format", choices=("i2_s", "base3", "base3_rowwise"), default="i2_s",
    )
    result.add_argument("--scale-storage", choices=("fp16", "uint8_rowwise"), default="fp16")
    result.add_argument("--recovery-rank", type=int, default=0)
    result.add_argument("--device", default="cpu")
    result.add_argument("--exclude-embeddings", action="store_true")
    return result


def main(argv: list[str] | None = None):
    args = parser().parse_args(argv)
    if args.kind == "masked_lm":
        model = AutoModelForMaskedLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch.float32,
        ).to(args.device)
    else:
        model = load_lfm_encoder(args.model, torch_dtype=torch.float32).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    minima = MinimaModel.from_model(model, base_model=args.model, model_kind=args.kind,
                                    group_size=args.group_size, recovery_rank=args.recovery_rank,
                                    include_embeddings=not args.exclude_embeddings,
                                    storage_format=args.storage_format,
                                    scale_storage=args.scale_storage)
    minima.save_pretrained(args.output, tokenizer)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
