#!/usr/bin/env python3
"""Launch a Minima HF Job with a conservative worst-case credit reservation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token

CAP_USD = 100.0
LEDGER = Path(__file__).resolve().parent.parent / "hf_jobs_ledger.json"


def parse_duration(value: str) -> float:
    units = {"s": 1 / 60, "m": 1, "h": 60, "d": 1440}
    try:
        return float(value[:-1]) * units[value[-1].lower()]
    except (KeyError, ValueError):
        return float(value) / 60


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flavor", required=True)
    parser.add_argument("--timeout", required=True)
    parser.add_argument("--image", default="pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel")
    parser.add_argument("--name", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("a command is required after --")
    command = args.command[1:] if args.command[0] == "--" else args.command
    api = HfApi()
    hardware = {item.name: item for item in api.list_jobs_hardware()}
    if args.flavor not in hardware:
        raise SystemExit(f"unknown flavor: {args.flavor}")
    minutes = parse_duration(args.timeout)
    reservation = hardware[args.flavor].unit_cost_usd * minutes
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"cap_usd": CAP_USD, "jobs": []}
    settled_ids = set(ledger.get("settled_job_ids", []))
    reserved = ledger.get("settled_usd", 0.0) + sum(
        item["worst_case_usd"] for item in ledger["jobs"] if item["id"] not in settled_ids
    )
    if reserved + reservation > CAP_USD:
        raise SystemExit(f"refusing launch: ${reserved + reservation:.2f} would exceed ${CAP_USD:.2f} cap")
    token = get_token()
    if not token:
        raise SystemExit("Hugging Face authentication is required")
    job = api.run_job(
        image=args.image,
        command=command,
        flavor=args.flavor,
        timeout=args.timeout,
        secrets={"HF_TOKEN": token},
        labels={"project": "minima", "name": args.name},
    )
    ledger["jobs"].append({
        "id": job.id,
        "name": args.name,
        "flavor": args.flavor,
        "timeout": args.timeout,
        "worst_case_usd": round(reservation, 4),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "url": f"https://huggingface.co/jobs/ProCreations/{job.id}",
    })
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(ledger["jobs"][-1], indent=2))


if __name__ == "__main__":
    main()
