#!/usr/bin/env python3
"""Per-category decode ladder with speculative-acceptance receipts.

Ported from the GLM-5.3-Flash 2x-Spark kit's decode fixture, with its two
important protocol choices kept:

  * `min_tokens` + `ignore_eos` pin every run to exactly the same output
    length. A run that stops early is not comparable to one that does not.
  * the decode rate excludes TTFT -- (emitted - 1) / (last - first content
    token). Counting SSE chunks under-reports badly here, because one chunk
    can carry a whole accepted speculative block.

The category split is the point. This checkpoint ships two drafters and they
fail differently: acceptance is high on structured/code output and low on
prose, so a single blended number hides which drafter is better for a given
workload. Acceptance itself is not on the OpenAI path, so it is read from the
engine's `Decode batch` log lines for each arm's window.

    BENCH_BASE_URL=http://10.0.0.1:8888 python3 bench/decode.py
    python3 bench/decode.py --concurrency 1,2,4,8 --tokens 256 --out-name dflash-b8

Run it from a worker, never the head.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# One prompt per category, each written to run long enough to fill the token
# cap without hitting a natural stop. Ordered roughly by how predictable the
# output is, which is the axis speculative acceptance moves along.
PROMPTS = {
    "counting": "Count upward from 1, numbers only, separated by single spaces. Do not stop.",
    "structured": "Emit a markdown table of 40 rows with columns id, name, role, team, location. Fill every cell with plausible values. Table only.",
    "json": "Emit a single JSON array of 30 objects, each with keys id, title, tags (array of 3 strings), score (float), active (bool). JSON only, no prose.",
    "code": "Write a complete Python module implementing an LRU cache with a doubly linked list and dict: get, put, delete, iteration, resize, clear, invariant validation, full docstrings and type hints. Then ten usage examples. Code only.",
    "math": "Compute the first 60 Fibonacci numbers, showing each as 'F(n) = value' on its own line. Then state the ratio of consecutive terms for the last five.",
    "prose": "Write a precise technical essay explaining how memory bandwidth, arithmetic intensity and batching interact in modern LLM inference. Continue until cut off.",
}


def nonce(seed: str, *parts) -> str:
    return hashlib.sha256((seed + ":" + ":".join(map(str, parts))).encode()).hexdigest()[:16]


def median(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(vals) if vals else None


def run_arm(model: str, category: str, concurrency: int, tokens: int,
            seed: str, timeout: float) -> dict:
    """One (category, concurrency) cell: `concurrency` streams in parallel."""
    prompt = PROMPTS[category]

    def one(index: int) -> dict:
        # A distinct nonce per stream keeps the prefix cache out of the result.
        text = f"Benchmark {nonce(seed, category, concurrency, index)}.\n{prompt}"
        return common.stream_chat(model, [{"role": "user", "content": text}],
                                  max_tokens=tokens, timeout=timeout,
                                  pin_length=True)

    start_line = common.log_line_count()
    wall0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        runs = list(pool.map(one, range(concurrency)))
    wall = time.perf_counter() - wall0
    spec = common.decode_stats_since(start_line)

    good = [r for r in runs if not r["error"]]
    emitted = sum(r["completion_tokens"] for r in good)
    short = [r["completion_tokens"] for r in good if r["completion_tokens"] != tokens]
    return {
        "category": category,
        "concurrency": concurrency,
        "tokens_requested": tokens,
        "ok": len(good),
        "failed": [r["error"] for r in runs if r["error"]],
        "length_mismatches": short,          # pinned length not honoured
        "aggregate_tok_s": (emitted / wall) if wall > 0 else None,
        "per_stream_tok_s_median": median([r["decode_tok_s"] for r in good]),
        "ttft_s_median": median([r["ttft_s"] for r in good]),
        "wall_s": wall,
        "nan_free": all(r["nan_free"] for r in good),
        "spec": spec,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--categories", default=",".join(PROMPTS))
    ap.add_argument("--seed", default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                    help="use the SAME seed across arms being compared")
    ap.add_argument("--out-name", default="decode")
    ap.add_argument("--allow-busy", action="store_true")
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    unknown = [c for c in cats if c not in PROMPTS]
    if unknown:
        ap.error(f"unknown categories {unknown}; choose from {list(PROMPTS)}")

    status, _ = common.get("/health")
    if status != 200:
        print(f"STOP: GET /health -> {status}", flush=True)
        return 1
    model, max_len = common.served_model()
    print(f"model={model} max_model_len={max_len} seed={args.seed}", flush=True)
    if not args.allow_busy:
        print(f"idle check: {json.dumps(common.require_idle())}", flush=True)

    # One throwaway request so the first arm is not paying JIT/warmup costs.
    print("warmup...", flush=True)
    common.stream_chat(model, [{"role": "user", "content": "Reply with OK."}],
                       max_tokens=8, timeout=180)

    timeout = max(300.0, args.tokens * 2.0)
    cells: list[dict] = []
    for concurrency in levels:
        for category in cats:
            cell = run_arm(model, category, concurrency, args.tokens,
                           args.seed, timeout)
            cells.append(cell)
            spec = cell["spec"] or {}
            accept = spec.get("accept_len_median")
            note = ""
            if cell["failed"]:
                note = f"  FAILED x{len(cell['failed'])}"
            elif cell["length_mismatches"]:
                note = f"  short: {cell['length_mismatches']}"
            print(f"  C{concurrency:<2} {category:<11} "
                  f"agg {cell['aggregate_tok_s'] or 0:7.2f}  "
                  f"per-stream {cell['per_stream_tok_s_median'] or 0:6.2f}  "
                  f"TTFT {cell['ttft_s_median'] or 0:5.2f}s  "
                  f"accept {accept if accept is not None else '?'}"
                  f"{note}", flush=True)

    print("\naggregate tok/s by category and concurrency:")
    header = "| category | " + " | ".join(f"C{c}" for c in levels) + " |"
    print(header)
    print("|---" * (len(levels) + 1) + "|")
    for category in cats:
        row = [f"{next((c['aggregate_tok_s'] for c in cells if c['category'] == category and c['concurrency'] == lvl), 0) or 0:.1f}"
               for lvl in levels]
        print(f"| {category} | " + " | ".join(row) + " |")

    print("\nmedian accepted draft tokens per step (the drafter's real score):")
    for category in cats:
        vals = [(c["concurrency"], (c["spec"] or {}).get("accept_len_median"))
                for c in cells if c["category"] == category]
        pretty = ", ".join(f"C{k}={v}" for k, v in vals if v is not None)
        print(f"  {category:<11} {pretty or 'unavailable (engine log unreadable)'}")

    path = common.write_receipt(args.out_name, {
        "served_model": model,
        "max_model_len": max_len,
        "seed": args.seed,
        "tokens": args.tokens,
        "concurrency": levels,
        "categories": cats,
        "cells": cells,
    })
    print(f"\nreceipt: {path}", flush=True)
    bad = [c for c in cells if c["failed"] or c["length_mismatches"] or not c["nan_free"]]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
