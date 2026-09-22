#!/usr/bin/env python3
"""Cold-prefill ladder. Ported from the GLM-5.3-Flash 2x-Spark kit.

The protocol is the part worth keeping: temperature 0, thinking off, a unique
salt on every cold request so the radix cache cannot serve it, one request at a
time, prompt length verified through /tokenize rather than assumed, and TTFT
taken from the first content token. Prefill rate is usage.prompt_tokens / TTFT.

It also runs one prefix-cache follow-up on the first rung (same long prompt,
plus an assistant turn and a short user turn). That row should be dramatically
faster; if it is not, the prefix cache is not doing its job, and the cold rungs
above it are the honest numbers either way.

    BENCH_BASE_URL=http://10.0.0.1:8888 python3 bench/cold_prefill.py
    python3 bench/cold_prefill.py --rungs 8000,16000,32000 --out-name chunk4096

Run it from a worker. Every rung is one cold prefill of that length, so the
long ones cost minutes: budget roughly prompt_tokens / 1000 seconds each.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# Defaults stop at 128k: enough to show the shape of the curve without a
# half-hour run. The KV pool has held a full 1M context since
# SWA_FULL_TOKENS_RATIO went to 0.03, so higher rungs are reachable -- pass
# them explicitly and expect ~1 s per 1000 tokens.
DEFAULT_RUNGS = [8_000, 16_000, 32_000, 64_000, 128_000]


def timeout_for(target: int) -> float:
    # ~1000 tok/s observed, with a floor for the short rungs and generous slack.
    return max(180.0, target / 1000.0 * 4.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rungs", help="comma-separated prompt-token targets")
    ap.add_argument("--out-name", default="cold-prefill",
                    help="receipt filename prefix, e.g. a config label")
    ap.add_argument("--no-cache-followup", action="store_true")
    ap.add_argument("--allow-busy", action="store_true",
                    help="skip the idle check (results become uninterpretable)")
    args = ap.parse_args()

    rungs = ([int(x) for x in args.rungs.split(",") if x.strip()]
             if args.rungs else list(DEFAULT_RUNGS))

    status, _ = common.get("/health")
    if status != 200:
        print(f"STOP: GET /health -> {status}", flush=True)
        return 1
    model, max_len = common.served_model()
    print(f"model={model} max_model_len={max_len}", flush=True)
    if not args.allow_busy:
        print(f"idle check: {json.dumps(common.require_idle())}", flush=True)
    if not common.metrics_available():
        print("note: /metrics is 404 (ENABLE_METRICS=0); cache-hit counters unavailable,"
              " cached_tokens needs ENABLE_CACHE_REPORT=1", flush=True)

    over = [r for r in rungs if max_len and r > max_len]
    if over:
        print(f"STOP: rungs {over} exceed max_model_len={max_len}", flush=True)
        return 1

    results: list[dict] = []
    first_prompt = None
    first_answer = None

    for index, target in enumerate(rungs):
        salt = common.unique_salt()
        timeout = timeout_for(target)
        print(f"\n=== cold {target:,} tokens (timeout {timeout:.0f}s) ===", flush=True)
        n, tokenized, prompt, rel = common.calibrate(target, salt, model,
                                                     timeout=min(timeout, 180))
        print(f"  calibrated: tokenize={tokenized:,} target={target:,} "
              f"filler={n:,} off_by={rel:.3%}", flush=True)
        if rel > 0.02:
            print("  WARNING: prompt length still >2% off target", flush=True)

        messages = [{"role": "user", "content": prompt}]
        record = common.stream_chat(model, messages, max_tokens=8, timeout=timeout)
        record.update({"target": target, "tokenize_est": tokenized,
                       "filler": n, "kind": "cold"})
        results.append(record)

        if record["error"]:
            print(f"  FAILED: {record['error']}", flush=True)
        else:
            print(f"  prompt_tokens={record['prompt_tokens']:,} "
                  f"TTFT={record['ttft_s']:.2f}s "
                  f"prefill={record['prefill_tok_s']:.0f} tok/s "
                  f"cached={record['cached_tokens']}", flush=True)

        if index == 0 and not args.no_cache_followup and not record["error"]:
            first_prompt, first_answer = prompt, record["text_head"]
            print("\n=== prefix-cache follow-up on the first rung ===", flush=True)
            follow = common.stream_chat(model, [
                {"role": "user", "content": first_prompt},
                {"role": "assistant", "content": first_answer},
                {"role": "user", "content": "Confirm with OK."},
            ], max_tokens=8, timeout=timeout)
            follow.update({"target": target, "kind": "prefix_cache_followup"})
            results.append(follow)
            if follow["error"]:
                print(f"  FAILED: {follow['error']}", flush=True)
            else:
                print(f"  prompt_tokens={follow['prompt_tokens']:,} "
                      f"TTFT={follow['ttft_s']:.2f}s "
                      f"cached={follow['cached_tokens']}", flush=True)

    ok = [r for r in results if r["kind"] == "cold" and not r["error"]]
    print("\n| prompt tokens | TTFT (s) | prefill tok/s |")
    print("|---|---:|---:|")
    for r in ok:
        print(f"| {r['prompt_tokens']:,} | {r['ttft_s']:.2f} | {r['prefill_tok_s']:.0f} |")

    path = common.write_receipt(args.out_name, {
        "served_model": model,
        "max_model_len": max_len,
        "rungs": rungs,
        "results": results,
        "metrics_available": common.metrics_available(),
    })
    print(f"\nreceipt: {path}", flush=True)
    return 0 if len(ok) == len([r for r in results if r["kind"] == "cold"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
