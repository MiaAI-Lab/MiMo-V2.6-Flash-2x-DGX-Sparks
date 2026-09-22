#!/usr/bin/env python3
"""Shared plumbing for the bench harnesses.

Adapted from the GLM-5.3-Flash 2x-Spark kit's cold-prefill harness. Two
differences from that original matter:

  * that kit runs vLLM, so its counters are `vllm:*` on /metrics. This is
    SGLang, where /metrics only exists with --enable-metrics (ENABLE_METRICS=1)
    and per-request cached-token reporting only with --enable-cache-report
    (ENABLE_CACHE_REPORT=1). Everything here degrades to "unknown" rather than
    failing when they are off.
  * acceptance length for speculative decoding is not on the OpenAI path at
    all, so it is read from the engine's own `Decode batch` log lines for the
    window a run covers. That is the number that separates EAGLE from DFlash,
    so it is worth the container read.

Run these from a worker, not the head: the head runs the HTTP server, the
tokenizer, the detokenizer and the NFS export, and it is the rank the other
waits for.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE = os.environ.get("BENCH_BASE_URL", "http://127.0.0.1:8888").rstrip("/")
API_KEY = os.environ.get("BENCH_API_KEY", "").strip()
HEAD_CTN = os.environ.get("HEAD_CTN", "mimo26-tp2-head")
RESULTS = Path(os.environ.get("BENCH_RESULTS_DIR", "bench/results"))

FILLER = "the "
TASK = "Reply with OK."


def headers() -> dict:
    out = {"Content-Type": "application/json"}
    if API_KEY:
        out["Authorization"] = "Bearer " + API_KEY
    return out


def get(path: str, timeout: float = 15.0):
    req = urllib.request.Request(BASE + path, headers=headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def post(path: str, body: dict, timeout: float):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), headers=headers(), method="POST"
    )
    return urllib.request.urlopen(req, timeout=timeout)


def served_model() -> tuple[str, int | None]:
    status, raw = get("/v1/models")
    if status != 200:
        raise SystemExit(f"GET /v1/models -> {status}")
    entry = json.loads(raw)["data"][0]
    return entry["id"], entry.get("max_model_len")


def require_idle() -> dict:
    """A benchmark against a busy server measures the other traffic.

    SGLang's /v1/loads carries the running and waiting counts without needing
    --enable-metrics, so this works on a default boot.
    """
    status, raw = get("/v1/loads")
    if status != 200:
        raise SystemExit(f"cannot verify idle state: GET /v1/loads -> {status}")
    rows = (json.loads(raw) or {}).get("loads") or []
    if not rows:
        raise SystemExit("cannot verify idle state: /v1/loads returned no rows")
    snap = rows[0]
    running = snap.get("num_running_reqs") or 0
    waiting = snap.get("num_waiting_reqs") or 0
    if running or waiting:
        raise SystemExit(f"server not idle: running={running} waiting={waiting}")
    return snap


def metrics_available() -> bool:
    return get("/metrics")[0] == 200


def tokenize(text: str, model: str, timeout: float = 180.0) -> int:
    with post("/tokenize", {"model": model, "prompt": text}, timeout=timeout) as resp:
        return int(json.loads(resp.read().decode())["count"])


def unique_salt() -> str:
    """Every cold request gets its own prefix so the radix cache cannot serve
    it. Without this a 'cold' ladder silently measures prefix-cache hits."""
    return f"COLD salt={uuid.uuid4()} pad={secrets.token_hex(24)}"


def build_prompt(n_filler: int, salt: str) -> str:
    return f"{salt}\n{FILLER * n_filler}\n{TASK}"


def calibrate(target: int, salt: str, model: str, timeout: float = 180.0):
    """Choose a filler count whose tokenized length is within ~0.5% of target."""
    overhead = tokenize(build_prompt(0, salt), model, timeout)
    n = max(target - overhead, 1)
    got = tokenize(build_prompt(n, salt), model, timeout)
    for _ in range(2):
        if target and abs(got - target) / target <= 0.005:
            break
        per = (got - overhead) / n if n else 1.0
        n = max(int(round((target - overhead) / (per if per > 0 else 1.0))), 1)
        got = tokenize(build_prompt(n, salt), model, timeout)
    if target and abs(got - target) / target > 0.005:
        n = max(n + (target - got), 1)
        got = tokenize(build_prompt(n, salt), model, timeout)
    rel = abs(got - target) / target if target else 0.0
    return n, got, build_prompt(n, salt), rel


def stream_chat(model: str, messages: list[dict], *, max_tokens: int, timeout: float,
                pin_length: bool = False, thinking: bool = False) -> dict:
    """One streamed chat completion, timed.

    `pin_length` adds min_tokens + ignore_eos so every run emits exactly
    max_tokens. Without it a run that stops early is not comparable to one
    that does not, which is the trap in naive decode benchmarks.

    Decode rate excludes TTFT: it is (emitted - 1) / (last - first content
    token). Counting SSE chunks instead would under-report by the accepted
    block length under speculative decoding.
    """
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }
    if pin_length:
        body.update({"min_tokens": max_tokens, "ignore_eos": True, "stop": []})

    t0 = time.perf_counter()
    first = last = None
    text: list[str] = []
    usage = None
    finish = None
    status = None
    error = None
    try:
        with post("/v1/chat/completions", body, timeout) as resp:
            status = resp.status
            buf = b""
            while True:
                piece = resp.read(4096)
                if not piece:
                    break
                buf += piece
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        continue
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("error"):
                        raise RuntimeError(f"server error in stream: {obj['error']}")
                    if obj.get("usage"):
                        usage = obj["usage"]
                    for choice in obj.get("choices") or []:
                        content = (choice.get("delta") or {}).get("content") or ""
                        if content:
                            now = time.perf_counter()
                            if first is None:
                                first = now
                            last = now
                            text.append(content)
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
    except Exception as exc:  # a failed rung must not lose the rungs before it
        error = f"{type(exc).__name__}: {exc}"
    t1 = time.perf_counter()

    body_text = "".join(text)
    prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
    completion_tokens = int((usage or {}).get("completion_tokens") or 0)
    ttft = None if first is None else first - t0
    decode_window = None if (first is None or last is None or last <= first) else last - first
    return {
        "http": status,
        "error": error,
        "finish_reason": finish,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
        "ttft_s": ttft,
        "wall_s": t1 - t0,
        "decode_window_s": decode_window,
        "prefill_tok_s": (prompt_tokens / ttft) if (ttft and prompt_tokens) else None,
        "decode_tok_s": ((completion_tokens - 1) / decode_window) if (decode_window and completion_tokens > 1) else None,
        "nan_free": "nan" not in body_text.lower(),
        "text_head": body_text[:160],
    }


_DECODE_LINE = re.compile(
    r"accept len: (?P<accept>[\d.]+), accept rate: (?P<rate>[\d.]+), "
    r"cuda graph: (?P<graph>\w+), gen throughput \(token/s\): (?P<tps>[\d.]+)"
)


def log_line_count() -> int | None:
    """Current engine log length, so a run can read only its own lines."""
    try:
        out = subprocess.run(["docker", "logs", HEAD_CTN], capture_output=True,
                             text=True, timeout=120)
        return len((out.stdout + out.stderr).splitlines())
    except Exception:
        return None


def decode_stats_since(start_line: int | None) -> dict:
    """Acceptance and engine-reported throughput from `Decode batch` lines.

    Speculative acceptance is not exposed on the OpenAI path, and /metrics
    needs --enable-metrics, so the engine log is the only source on a default
    boot. Returns empty when the container is not reachable.
    """
    if start_line is None:
        return {}
    try:
        out = subprocess.run(["docker", "logs", HEAD_CTN], capture_output=True,
                             text=True, timeout=120)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    lines = (out.stdout + out.stderr).splitlines()[start_line:]
    accept, rate, tps, graph = [], [], [], set()
    for line in lines:
        m = _DECODE_LINE.search(line)
        if not m:
            continue
        accept.append(float(m["accept"]))
        rate.append(float(m["rate"]))
        tps.append(float(m["tps"]))
        graph.add(m["graph"])
    if not accept:
        return {"decode_batches": 0}
    mid = sorted(accept)[len(accept) // 2]
    return {
        "decode_batches": len(accept),
        "accept_len_median": mid,
        "accept_len_min": min(accept),
        "accept_len_max": max(accept),
        "accept_rate_median": sorted(rate)[len(rate) // 2],
        "engine_tok_s_median": sorted(tps)[len(tps) // 2],
        "cuda_graph": sorted(graph),
    }


def write_receipt(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = RESULTS / f"{name}-{stamp}.json"
    payload = dict(payload)
    payload["written"] = stamp
    payload["base_url"] = BASE
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path
