<h1 align="center">MiMo-V2.6-Flash for 2x DGX Sparks</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Serve [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)
— 309B total, 15B active, MXFP4 experts, 1M context, text/image/video/audio —
with SGLang on **two** DGX Sparks (GB10, SM121, 121.7 GiB unified memory each)
over ConnectX-7 RoCE, behind one OpenAI-compatible endpoint on the head.

## Status

Honest state of this kit. Nothing below is projected — every number is from a
log line in `logs/`, and anything unmeasured says so.

| | |
|---|---|
| Boots and serves | yes, TP=2 over RoCE, `/health` 200, text + image + video + audio |
| Weight load | 96 s per rank, 82.7 GiB resident (measured both ranks) |
| Decode (sparkDash, thinking off) | **DFlash** code 69.8 / prose 25.8 at 1 stream (190.3 / 103.0 agg at 8). **EAGLE MTP** code 46.0 / prose 35.0 (138.7 / 119.9 at 8) |
| Prefill | peak **2,558 tok/s** at 8K, 1,227 at 128K, 764 at 256K (sparkDash). Chunk size not yet laddered |
| Context | 1,048,576 configured, and the pool now holds it — see below |
| **KV cache pool** | **2,878,144 tokens** with EAGLE MTP at 0.963 (19.46 GB per rank) — **2.74x a full 1M context**, or 8 concurrent requests of ~360k each. DFlash gives 1,664,704: its drafter keeps a second pool sized to the pin |
| Speculative decoding | **EAGLE MTP by default** (prose 35.0 tok/s, +36% over DFlash, and a much larger KV pool). Switch to DFlash for code, where it wins by +52% |
| CUDA graphs | decode capture failed under EAGLE multi-layer; being retried under DFlash, which upstream runs graphs with |
| Not measured | concurrency scaling, long-context prefill, needle tests, DFlash acceptance on this fleet |

Four node reboots and two OOM kills went into getting this far; "Why the first
boots crashed" is the useful part of this README if you are porting something
else to a Spark.

## Requirements

When updating an existing installation, rebuild the image on both ranks with
`./start.sh build` before serving. The [image corrections](patches/README.md#mimo-vision-windows-and-key-zero-bias)
require the patched engine image and matching Python hook, including DFlash's
image-forwarding fix.

- 2x DGX Spark (GB10), one GPU each, linked over ConnectX-7. A single Spark
  cannot serve this model at all — 161 GiB of weights on a 121.7 GiB box.
- ~161 GiB of free disk on the head for the checkpoint, plus Docker.
- Passwordless ssh from the head to the worker.
- `BASE_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20261003-65f75914` (the Dockerfile pins this nightly's source hashes; the oldest engine with the packed MXFP4 experts is lmsysorg/sglang:nightly-dev-cu13-20260921-0f6761b5):
  MiMo-V2.6's MXFP4 experts need
  [sgl-project/sglang#40448](https://github.com/sgl-project/sglang/pull/40448).
  Older engines expand the experts to FP8 and take the node down.

The checkpoint is ~161 GiB on disk (`total_size` 172,923,364,096), 64 expert shards
(`model_pp0_ep0` … `ep63`), 256 routed experts, 15B activated, 1M context,
text / image / video / audio, 3-layer MTP. The routed experts are stored **MXFP4**
(U8 nibble pairs plus E8M0 block-32 scales, 149.8 GiB for all 256); attention and the
dense MLP are FP8 e4m3 with block-128 `weight_scale_inv`; embeddings, `o_proj`, gates
and the encoders are BF16.

| Layout | Resident weights per GPU | Fits 121.7 GiB? |
|---|---:|:---|
| **2 ranks (TP=2)** | **~82.8 GiB** (74.9 experts + 6.1 sharded + 1.8 encoders, DFlash drafter) | **yes**, ~12 GiB left for the drafter and KV at `MEM_FRACTION_STATIC=0.85` |

Those numbers only hold while the engine keeps the experts in their stored 4-bit
layout. `boot.py` measures them from the safetensors headers and prints them as
`preflight: …` before SGLang starts.


## Divisibility

`config.json` has 64 query heads, 4 global KV heads, 8 SWA KV heads and 256
experts. The safetensors index metadata records `"tp_size": 4` for the fused
`qkv_proj`. The pinned engine reshards that onto TP=2 itself: it merges two
checkpoint shards per rank, then de-interleaves `[q0 k0 v0 q1 k1 v1]` into
`[q0 q1 | k0 k1 | v0 v1]` with a dequant/requant of the block scales
(`Resolved deferred qkv scale_inv … de-interleaved QKV` in the log). The MTP
draft loader (`mimo_v2_nextn.py`) does not use that path yet, so
`patches/sitecustomize.py` adds it for the draft through an import hook; without
it every rank fails right after the 82 GiB target load with
`qkv_proj scale_inv model.mtp_block.self_attn.qkv_proj.weight_scale_inv: shape
mismatch (116, 32) vs (58, 32)`. See `patches/README.md`.

## Topology

```
spark1  10.0.0.1  rank 0   API :8888, NFSv4 export of the checkpoint
spark2  10.0.0.2  rank 1   mounts spark1 over ConnectX (10.0.22.1)
```

Workers do not keep a copy of the checkpoint. `share` exports spark1's tree
and mounts it on the worker. `serve` runs that check first and will not start a
rank until the worker can read the head's `model.safetensors.index.json` (same
byte size) and all 64 expert shards.

## Quick start

```bash
cd ~/NewModels/MiMo-2.6-Flash
./start.sh doctor            # image label, engine + memory preflight, ssh/IB
./start.sh download          # ~171 GiB, once, on the head (see below)
./start.sh build             # FROM $BASE_IMAGE, then docker save | load to the worker
./start.sh share
./start.sh serve
```

### Get the checkpoint

`./start.sh download` fetches the pinned revision to `$MODEL_DIR`
(`~/NewModels/MiMo-V2.6-Flash-RL` by default) on **the head only** — workers
read it over NFS and never keep a copy.

```bash
df -h ~/NewModels          # need ~171 GiB free
./start.sh download        # 30-90 min on a fast link; resumable
```

It uses `hf download` if present, falls back to `huggingface-cli`, then to
`huggingface_hub.snapshot_download`, so any one of those is enough:

```bash
pip install -U "huggingface_hub[cli]"
```

The checkpoint is **public**, so no token is needed. If you are behind a proxy
or want a mirror, set `HF_ENDPOINT` / `HF_HUB_ENABLE_HF_TRANSFER=1` in your
shell before running it.

What lands, and why all of it matters:

| | |
|---|---|
| `model_pp0_ep0 … ep63` | 64 expert shards, ~150 GiB, the MXFP4 routed experts |
| `model_mtp.safetensors` | 1.2 GiB, the 3-layer MTP head — the **EAGLE MTP** drafter |
| `dflash/` | 2.9 GiB separate Qwen3 draft model — the **DFlash** drafter |
| `audio_tokenizer/`, `preprocessor_config.json` | audio and vision preprocessing |
| `config.json` | must contain `vision_config` **and** `audio_config` |

**A partial download is the failure mode to watch for.** It leaves valid-looking
JSON beside missing weights, and the engine then dies mid-load. `download` and
`share` both verify the result: the five files `boot.py` requires, plus every
shard the index references. If anything is missing you get the count and the
first few names rather than a confusing load error. To re-check at any time:

```bash
./start.sh doctor          # image label, engine + memory preflight, ssh/IB, shards
```

Re-running `download` after an interruption resumes rather than restarting, and
the revision is pinned (`HF_REVISION`) so a re-run cannot silently pull a
different checkpoint. Already have it elsewhere? Point `MODEL_DIR` at it in
`.env` and skip this step — `share` and `serve` validate whatever is there.

### Pick a drafter

Two speculative-decoding drafters ship with the checkpoint, and the right one
depends on the workload. Each is three variables and one restart — the third is
**not optional**, see [Speculative decoding](#speculative-decoding):

```bash
# EAGLE MTP (the default) — prose / chat / long-form, and the larger KV pool.
# 35.0 tok/s prose, 46.0 code. Decode graphs must be off.
sed -i 's/^SPEC_ALGO=.*/SPEC_ALGO=EAGLE/;  s/^SPEC_DRAFT_TOKENS=.*/SPEC_DRAFT_TOKENS=4/;  s/^CUDA_GRAPH_MAX_BS_DECODE=.*/CUDA_GRAPH_MAX_BS_DECODE=0/;  s/^MAX_TOTAL_TOKENS=.*/MAX_TOTAL_TOKENS=2743808/' .env && ./start.sh restart

# DFlash draft model — code / structured / agents. 69.8 tok/s code, 25.8 prose.
sed -i 's/^SPEC_ALGO=.*/SPEC_ALGO=DFLASH/; s/^SPEC_DRAFT_TOKENS=.*/SPEC_DRAFT_TOKENS=8/; s/^CUDA_GRAPH_MAX_BS_DECODE=.*/CUDA_GRAPH_MAX_BS_DECODE=8/; s/^MAX_TOTAL_TOKENS=.*/MAX_TOTAL_TOKENS=2299968/' .env && ./start.sh restart
```

**The pin differs by drafter**, and that is deliberate. EAGLE MTP's draft KV
pools are ~0.08 GB each; DFlash's drafter keeps a second pool sized to the
*same* `max_total_num_tokens`, which cost 7.94 GB at a 1.66M pool. So the same
memory buys EAGLE a far larger context. Under EAGLE the pin, not the budget, was
what bound the pool — 2,299,968 tokens took 15.55 GB and left 10.10 GB free —
so `MAX_TOTAL_TOKENS=2743808` (+3 GiB at 7,262 B/token, measured) reaches for
that headroom. Expect it to land near 2.74M or to clamp a little short; either
way `max_total_num_tokens=` in the boot log is the answer.

Check which is live:

```bash
grep -o "'speculative_algorithm': '[A-Z]*'" logs/mimo26.log | tail -1
```

First run copies `.env.example` → `.env`. Edit IPs there. `./start.sh status`, `logs`, `smoke`, `stop`, `restart`.
`stop --unmount` drops the worker NFS volumes; the head checkpoint stays.
Changing `BASE_IMAGE` in the profile is enough: the next `serve` rebuilds and
resends the image.

`stop` removes the head container first (it is the HTTP endpoint, so dropping
it stops new requests being admitted), then the worker, and then **waits for
the driver to release the GPU on every node**: `docker rm -f` returns as soon as the container record is gone,
but the CUDA context and the ~82 GiB behind it take seconds more. It exits 2 —
and says not to serve — if any node never clears, because that is the
signature of a wedged driver rather than a slow teardown.

`restart` is `stop` then `serve` with that wait in between, which is what
makes it safe to chain: `cmd_serve` refuses to launch while `nvidia-smi` still
lists a compute app, and SGLang sizes its KV pool from `MemAvailable`, so a
restart that does not wait either refuses or boots against a budget that is
still draining. It declines `--unmount` (serve would come back with no worker
volume) and forwards `--no-wait` / `--wait-timeout N`.

```bash
curl http://10.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"MiMo-v2.6-Flash","messages":[{"role":"user","content":"What is 19 + 23?"}],
       "max_tokens":32,"temperature":1.0,"top_p":0.95,
       "chat_template_kwargs":{"enable_thinking":false}}'
```

## Defaults

Everything below ships in `.env.example`, which `start.sh` copies to `.env` on
first run. These are the values this kit was measured at, not upstream's — the
divergences and their reasons are in "Launch flags" below.

| | default | why |
|---|---|---|
| `TP_SIZE` / `EP_SIZE` / `NNODES` | 2 / 2 / 2 | one GPU per Spark; a single Spark cannot hold the model |
| `DP_SIZE` | 1 | one GPU per node, so the DP flags stay off |
| `CONTEXT_LENGTH` | 1,048,576 | the model maximum, and the pool now holds it |
| `MAX_TOTAL_TOKENS` | 2,743,808 | a ceiling above the budget, so the budget binds. Lands at **2,476,864** measured under EAGLE. Drop to ~2,299,968 for DFlash, whose drafter needs its own pool |
| `MEM_FRACTION_STATIC` | 0.93 | aggressive on purpose: 7.84 GB of slack, ~5.5 GB free after capture |
| `MEM_GUARD_GIB` | 4 | EAGLE at 0.93 ends the boot with ~9 GB free, so the safer floor fits. Lower it to 2 only if a config leaves under ~6 GB |
| `SWA_FULL_TOKENS_RATIO` | 0.03 | the upstream Flash value. 0.3 (V2.5-era) cost 10x more for the sliding-window pool |
| `MAX_RUNNING_REQUESTS` | 8 | also the decode graph tiers, so every reachable batch replays a graph |
| `CHUNKED_PREFILL_SIZE` | 2048 | the prefill transient lives in the slack; **do not raise this at 0.93** |
| `PAGE_SIZE` | 64 | upstream uses 1; this is a fragmentation knob, not a capacity one |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | nothing smaller exists |
| `CUDA_GRAPH_MAX_BS_DECODE` | 0 | **required** by the default EAGLE MTP drafter — capture crashes under it. Set to 8 with DFlash, where it works |
| `CUDA_GRAPH_PREFILL` | 0 | prefill capture off, as upstream also does |
| `SPEC_ALGO` / `SPEC_DRAFT_TOKENS` | `EAGLE` / 4 | better on prose (+36%) and buys a far larger KV pool. `DFLASH` / 8 is the upstream default and wins code by +52% |
| `SPEC_STEPS` / `SPEC_TOPK` / `ENABLE_MTP` | 3 / 1 / 1 | EAGLE only; DFlash forces its own and ignores these |
| `ATTENTION_BACKEND` / `MM_ATTENTION_BACKEND` | `triton` / `triton_attn` | `fa4` is not an SM121 path |
| `MOE_RUNNER_BACKEND` | `flashinfer_mxfp4` | consumes the packed MXFP4 experts; `auto` would fail on them |
| `MOE_A2A_BACKEND` / `MOE_DENSE_TP_SIZE` | `none` / 1 | `deepep` needs NVSHMEM across the fabric, untested on two Sparks |
| `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE` | 0 | byte-identical greedy output, ~0.3 ms/step |
| `WEIGHT_LOADER_DROP_CACHE` | 1 | page cache and the KV budget are the same pool here |
| `ENABLE_METRICS` / `ENABLE_CACHE_REPORT` | 1 / 1 | what `bench/` reads |
| `OFFLOAD_MODE` | `off` | `ssd` spills HiCache L3; `ram`/`cpu` are refused |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:False` | `True` produced NaN logits on this GB10 stack |

The cluster half of the profile (`HEAD_IP`, `WORKER_IPS`, `WORKER_HOSTS`,
`WORKER_USER`, the `NCCL_*` and `NFS_*` block, `MODEL_DIR`) is site-specific —
edit it for your fleet. The container, volume and worker-directory names still
carry a `tp2` suffix from when this kit had two profiles; they are only
identifiers.

## Launch flags

The reference recipe is SGLang's
[MiMo-V2.6 cookbook](https://docs.sglang.io/cookbook/autoregressive/Xiaomi/MiMo-V2.6)
(Flash on 4x B300, single node, from
[sgl-project/sglang#40448](https://github.com/sgl-project/sglang/pull/40448)):
`--tp 4 --ep 4 --dp 1 --pp-size 1 --moe-runner-backend deep_gemm
--moe-a2a-backend deepep --deepep-mode auto --moe-dense-tp-size 1
--enable-dp-lm-head --attention-backend fa4 --mm-attention-backend fa4
--mm-enable-dp-encoder --context-length 1048576 --mem-fraction-static 0.6
--swa-full-tokens-ratio 0.03 --chunked-prefill-size 49152 --max-prefill-tokens
65536 --page-size 1 --speculative-algorithm DFLASH
--speculative-num-draft-tokens 8 --reasoning-parser mimo --tool-call-parser
mimo --enable-cache-report`, with running requests and the decode CUDA-graph
batch size both at 64 and **prefill CUDA graphs disabled**.

**This kit was originally built against the V2.5 recipe, which the V2.6
cookbook explicitly tells you not to carry over** ("Do not add
`--enable-dp-attention` or copy EAGLE-specific settings from the V2.5
cookbook"). Two settings came from there and were wrong for V2.6: the EAGLE
drafter, and `--swa-full-tokens-ratio 0.3` where Flash wants **0.03** — a 10x
over-allocation of the expensive sliding-window pool (see "KV geometry").
Both were corrected on 2026-09-22.

What still differs here, and why. Each Spark is one GPU, so `DP_SIZE=1` and the
DP flags stay off, and `--ep` equals `--tp` at 2. `fa4` is not an SM121 path,
so attention stays Triton. `deep_gemm`/`deepep` for the MoE need NVSHMEM across
the fabric and are untested on two Sparks, so the runner is
`flashinfer_mxfp4` with `--moe-a2a-backend none`. `--mem-fraction-static 0.85`
against the cookbook's 0.6 is forced by unified memory: host RAM *is* GPU
memory here, so a large reserve outside the pool starves KV rather than
protecting it. `--chunked-prefill-size 2048` against 49152, and
`--page-size 64` against 1, are memory-bound choices on a 121.7 GiB box and
have not been laddered.

MoE on this fleet:

- `MOE_RUNNER_BACKEND=flashinfer_mxfp4`: the experts stay packed and run on
  FlashInfer's SM120 cutlass MXFP8×MXFP4 kernel, the same path the DeepSeek-V4.1
  profile uses on these Sparks. `marlin` (W4A16) is the alternative. `auto`
  keeps the memory but hands packed tensors to the FP8 Triton kernels.
- `MOE_A2A_BACKEND=none`: plain expert parallel with all-reduce. The cookbook's
  `deepep` needs NVSHMEM across the fabric and is untested on two Sparks.
- `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0`: byte-identical greedy output,
  ~0.3 ms per step (DeepSeek-V4.1 profile finding).
- `WEIGHT_LOADER_DROP_CACHE=1` → `--weight-loader-drop-cache-after-load`: each
  shard's page cache is released once its tensors are on the GPU. On unified
  memory that cache and the KV budget come from the same pool.

## Weight loading

Measured on the head (GB10, NVMe, one 2.3 GiB expert shard of 1128 tensors):

| path | read | copies to GPU | per shard | 82 GiB rank |
|---|---:|---:|---:|---:|
| mmap, copy straight from the view (SGLang default) | 0.4 s | 17.7 s | ~18 s | 525 s (measured) |
| `--weight-loader-disable-mmap` (`read()` + parse) | 3.1 s | 1.35 s | ~4.5 s | 131 s (measured), crashed both nodes |
| mmap, `clone()` one tensor, then copy (the patch) | 0.4 s | 1.15 s + 0.26 s | ~1.9 s | 96 s (measured), 82.7 GB resident |

The CUDA pageable copy faults the mmap'd file pages 4 KiB at a time, about
15 ms per 4 MB tensor. A `clone()` reads the same tensor through the kernel's
readahead at 2 GiB/s and the copy then runs at 9 GiB/s. `read()` gets the
speed too, but every in-flight shard becomes anonymous memory twice over
(the bytes plus the parsed copies): 25 GiB for `ep0`, and ~7 GiB per rank
stayed resident after the load. `patches/sitecustomize.py` therefore keeps
the default mmap loader and clones one tensor at a time on the way to the
GPU, skipping routed experts that belong to the other EP rank (the model
skips those without touching the data, so each rank reads its ~82 GiB and
not the full 161). The same hook makes each EAGLE draft pass read only
`model_mtp.safetensors` instead of every shard. `WEIGHT_LOADER_DROP_CACHE=1`
(`--weight-loader-drop-cache-after-load`) releases each shard's page cache
once it is consumed. `fastsafetensors` is not in the image; a saved per-rank
`sharded_state` would be faster still but needs 82 GiB of free disk on every
worker.

Memory at TP=2: 82 GiB resident (82.7 measured on the worker), leaving
~30 GiB on the head and ~32 GiB on the worker. `MEM_FRACTION_STATIC=0.85`
keeps ~17 GiB of that outside the pool for prefill activations and the
encoders and leaves ~11 GiB on the head for the EAGLE draft and KV; at 0.80
the reserve is 24 GiB and the head has nothing left. fp8 KV at 262144 tokens measured 5.06 GB (SWA pool) plus three 0.09 GB draft
pools. `MAX_TOTAL_TOKENS=983616` scales that mix to 20 GB. Context is the
model limit `CONTEXT_LENGTH=1048576`, `MAX_RUNNING_REQUESTS=8`,
`CHUNKED_PREFILL_SIZE=2048`.
`MEM_GUARD_GIB=4` is the userspace OOM guard: `boot.py` kills SGLang when
`MemAvailable` drops under 4 GiB, which ends in an exited container instead
of a rebooted Spark.

## CUDA graphs

Capture is the big decode lever: with speculation the engine runs several model
passes per step across 48 layers, and uncaptured that is thousands of eager
launches per token — which is why the EAGLE measurement was only **19–28
tok/s** with `cuda graph: False` on every `Decode batch` line.

`CUDA_GRAPH_MAX_BS_DECODE=8` **failed under EAGLE + `--enable-multi-layer-eagle`**
(receipt: `logs/mimo26.crash-cudagraph-20260922-1036.log`) and is being retried
under DFlash, which uses a different draft runner (`dflash_worker_v2`) and which
the upstream cookbook does run decode graphs with. The EAGLE MTP failure below is
kept because it is a real incompatibility, not a memory problem, and because
the diagnosis generalises.

`CUDA_GRAPH_PREFILL` stays 0 either way — that also matches upstream, which
disables prefill CUDA graphs.

### The EAGLE MTP failure

The **target verify** graph captures fine:

```
Capture target verify CUDA graph begin. backend=full, num_tokens_per_req=4,
  bs=[1, 2, 3, 4, 5, 6, 7, 8], avail mem=19.67 GB
Capture target verify CUDA graph end. elapsed=4.35 s, mem usage=0.76 GB, avail mem=18.90 GB
```

The **EAGLE draft extend** graph then dies, with 18.10 GB still free — this is
not a memory failure:

```
SGLANG_ENABLE_SINGLE_CG_DRAFT is on but TritonAttnBackend cannot refresh its
  draft-extend metadata for a combined graph; falling back to per-step draft graphs
Capture draft extend CUDA graph begin. step=2, avail mem=18.10 GB
  → multi_layer_eagle_draft_extend_cuda_graph_runner.capture_one_shape
  → maybe_flashinfer_autotune_speculative_draft → mimo_v2_nextn.forward
  → triton.compiler.errors.CompilationError at offs_kv_loc = tl.load(kv_indices + ...)
    AttributeError("'NoneType' object has no attribute 'type'")
```

`ATTENTION_BACKEND=triton` and multi-layer-EAGLE draft-extend capture do not
compose in this engine: the Triton extend kernel is compiled against a dummy
batch whose `kv_indices` is `None`. Two untested candidates, in order of
preference, both one flag:

1. `--speculative-draft-attention-backend flashinfer` — takes the draft off
   the Triton extend kernel and leaves the target and the MoE autotune alone
   (`flashinfer`, `fa3`, `fa4`, `triton` are the draft choices).
2. `--disable-flashinfer-autotune` — skips exactly the call that raised
   (`flashinfer_autotune.py` returns early on this flag), at the cost of MoE
   autotuning everywhere.

Do not re-enable capture without one of them, and expect to need a boot per
candidate.

**What the attempt did establish.** Capture is cheap (0.76 GB for 8 tiers) and
comes out of the ~17 GiB kept *outside* the pool at 0.85, not out of the KV
pool — the same reserve the prefill transient uses, which is why the 8192 →
2048 chunk change helps it. And the KV pin is silently clamped, as suspected:
`MAX_TOTAL_TOKENS=983616` asked for ~20 GB and the pool came back with
**385,984 full-attention tokens and 115,776 SWA tokens** (SWAKVPool 7.46 GB
plus three 0.14 GB draft pools), `Memory pool end. avail mem=16.91–17.19 GB`.
So the effective full-attention capacity is ~386k tokens, well under
`CONTEXT_LENGTH=1048576`: a genuine 1M-token prompt cannot be served at this
memory fraction, whatever the context flag says.

**The decode max-bs flag does not select a phase.** SGLang's per-phase
defaults are `decode=full` and `prefill=breakable`
(`model_executor/cuda_graph_config.py`), and `--disable-cuda-graph` is what
had been switching *both* off. Dropping it therefore enables prefill BCG
capture too, which is untested here and wants the same reserve memory as the
prefill activation transient. `boot.py` passes
`--disable-prefill-cuda-graph` unless `CUDA_GRAPH_PREFILL=1`, so decode and
prefill move one at a time.

Capture memory comes out of the ~17 GiB `MEM_FRACTION_STATIC=0.85` keeps
*outside* the pool, not out of the KV pool — the same reserve the prefill
transient uses, which is why the 8192 → 2048 chunk change helps it. Read
`Capture cuda graph end … mem usage=` and the `Memory pool end. avail mem=`
line from the boot log before raising either.

Two things to check on the first graph-enabled boot:

- **Acceptance.** Per-position acceptance pinned at exactly 1.00 at pos0 is a
  known CUDA-graph bug, not a win; healthy is a monotone decay. Compare
  `accept len` against the 2.0–2.6 pre-graph baseline above.
- **The KV pin.** `MAX_TOTAL_TOKENS=983616` asks for ~20 GB of the ~11–13.7
  GiB the budget actually leaves for draft + KV, so it is already being
  clamped down; the boot log's pool line is the authority, not the profile.

This is the full omni checkpoint: text, image, video, and audio. The server
always starts with `--enable-multimodal`. The image installs `ffmpeg` and
refuses to build unless `torchcodec` and `torchaudio` import, because audio
goes through torchcodec and video needs `ffprobe`. Workers must see
`preprocessor_config.json`, `audio_tokenizer/model.safetensors`, and
`model_mtp.safetensors` on the head export, not only the language shards.

The publisher's sampling is temperature 1.0 and top_p 0.95. Thinking is on
unless `enable_thinking` is false. Parsers are `mimo` for tools and reasoning.
Vision attention on GB10 is `triton_attn` (`MM_ATTENTION_BACKEND`).

## KV geometry

Why the context limit is not the context you get. `config.json`'s
`hybrid_layer_pattern` gives **9 full-attention layers and 39 sliding-window
layers** (window 128) out of 48. Per token per rank at TP=2 with fp8 KV:

| | layers | KV heads/rank | bytes/token |
|---|---:|---:|---:|
| full attention | 9 | 2 | 5,760 |
| sliding window | 39 | 4 | **49,920** |

An SWA token costs **8.7x** a full-attention token, because it is 39 layers at
4 KV heads against 9 at 2. `MAX_TOTAL_TOKENS` is the pin in *full-attention*
tokens and the SWA pool is derived as `SWA_FULL_TOKENS_RATIO x` that, so the
ratio dominates the bill:

| `SWA_FULL_TOKENS_RATIO` | bytes per pin token | KV for a full 1M context |
|---|---:|---:|
| 0.30 (this kit until 2026-09-22) | 20,736 | 20.25 GiB |
| **0.03** (upstream Flash recipe) | **7,258** | **7.09 GiB** |

At 0.3 the pool clamped to 359,616 tokens against a 1,048,576 context, so a
full-context request could not be served. 0.03 is what the cookbook ships and
brings a genuine 1M context inside the measured budget.

### The pool, as measured

Both drafters, same fraction-and-pin arithmetic, very different results. EAGLE
MTP numbers are from the 2026-09-22 10:43 boot at
`MEM_FRACTION_STATIC=0.963` / `MAX_TOTAL_TOKENS=3127872`; DFlash from the 09:13
boot at 0.93 / 2299968.

```
EAGLE MTP   max_total_num_tokens=2878144   available_gpu_mem=5.72 GB
  Full KV  2,878,144 tokens   K 9.26 + V 6.18 GB
  SWA  KV     86,336 tokens   K 2.41 + V 1.61 GB
  SWAKVPool 19.46 GB  +  3 draft pools at 0.10 GB each
```

| | EAGLE MTP | DFlash |
|---|---:|---:|
| target KV | **19.46 GB** | 11.25 GB |
| drafter KV | 0.10 GB ×3 | **7.94 GB** |
| graph capture | 0 (graphs off) | 1.71 GB |
| **pool capacity** | **2,878,144 tokens** | 1,664,704 |
| vs a full 1M context | **2.74x** | 1.59x |
| free at end | 5.72 GB | 5.52 GB |

**EAGLE MTP buys 1.73x the context for the same memory.** The reason is the
drafter, not the target: DFlash keeps a second KV pool sized to the *same*
`max_total_num_tokens` — 7.94 GB of it — while EAGLE's three MTP draft pools
are 0.10 GB each. EAGLE spends more on draft *weights* (4.95 GB for three
runners against DFlash's 2.9 GB model) and far less on draft *KV*, and KV is
what scales with the pin.

`/v1/loads` reports the live split and needs no extra flags:
`weight_gb 82.984`, `kv_cache_gb 19.457`, `graph_gb 0.0`,
`token_capacity 2878144`.

### How far the memory fraction gets you

Measured on EAGLE MTP, same pin-above-budget approach each time:

| `MEM_FRACTION_STATIC` | pool | free at end | gain |
|---|---:|---:|---:|
| 0.85 | 359,616 | 20.49 GB | — |
| 0.93 | 2,476,864 | 9.02 GB | +588% |
| 0.95 | 2,762,624 | 6.60 GB | +11.5% |
| 0.963 | 2,878,144 | 5.72 GB | **+4.2%** |

The return collapses. 0.95 → 0.963 gave up 1.46 GiB of slack for 0.79 GB of
pool, because the three MTP draft runners' weights come out of the same budget.
**0.95 is the sensible stopping point** — 96% of the pool with a full extra
GiB of headroom, and the value the sibling DeepSeek-V4.1 kit runs in production
on this fleet. `.env.example` ships the more conservative 0.93.

Every rung also **clamped ~8% short of its pin**, so SGLang keeps a
proportional margin when sizing the pool and `MAX_TOTAL_TOKENS` cannot reach
past it. Raising the pin alone, once it is above the budget, does nothing.

**The pin is not the gate.** SGLang clamps `MAX_TOTAL_TOKENS` to whatever is
left after the weights inside `MEM_FRACTION_STATIC`, silently. Read
`max_total_num_tokens=` from the boot log; the env value is a ceiling, not a
promise. `boot.py`'s `preflight:` line consistently over-states the budget by
roughly 2x against what the engine then allocates, so trust the engine's line.

## Speculative decoding

The checkpoint ships **two** drafters — an **EAGLE MTP** head and a **DFlash
draft model** — and `SPEC_ALGO` picks between them. They are different things:
EAGLE MTP drafts from the target's own hidden states through the bundled 3-layer
MTP head, while DFlash is a separate 5-layer Qwen3-architecture model that
drafts a whole block at once. **Which is faster depends on what the model is
writing:**

| | best at | weak at | weights | verify window |
|---|---|---|---|---|
| `DFLASH`<br>*DFlash draft model* | **code 69.8 tok/s**, structured, JSON, counting | prose 25.8 (−26% vs EAGLE) | `dflash/`, 2.9 GiB, separate 5-layer Qwen3 model | 8 (its trained `block_size`) |
| `EAGLE`<br>*EAGLE MTP* (default) | **prose 35.0 tok/s**, narrative | code 46.0 (−34% vs DFlash), and **requires `CUDA_GRAPH_MAX_BS_DECODE=0` or the boot fails** | `model_mtp.safetensors`, 1.2 GiB, 3-layer MTP head | 4 |
| `off` | — | — | — | — |

The mechanism is acceptance, not raw speed. A drafter proposes `SPEC_DRAFT_TOKENS`
tokens per step and the target verifies them; rejected drafts are wasted compute.
DFlash proposes 8 and lands nearly all of them on predictable text, which is a
large win — but on prose it still proposes 8 and lands ~1, so the wasted drafts
make it *slower* than EAGLE's narrower 4-token window. EAGLE has a lower ceiling
everywhere and a much flatter profile.

There is **no per-request switch** and no dual-drafter mode: the draft model is
loaded once at startup (`SamplingParams` carries nothing speculative), so
picking one is a boot-time decision. Switching is one line and one boot.

### Booting each

**DFlash draft model** — code, structured output, agent work. Decode graphs work here.

```bash
sed -i 's/^SPEC_ALGO=.*/SPEC_ALGO=DFLASH/; s/^SPEC_DRAFT_TOKENS=.*/SPEC_DRAFT_TOKENS=8/; s/^CUDA_GRAPH_MAX_BS_DECODE=.*/CUDA_GRAPH_MAX_BS_DECODE=8/; s/^MAX_TOTAL_TOKENS=.*/MAX_TOTAL_TOKENS=2299968/' .env
./start.sh restart
```

**EAGLE MTP** — prose, chat, long-form writing.

> ### ⚠ `CUDA_GRAPH_MAX_BS_DECODE=0` is NOT optional under EAGLE MTP
>
> It is the **third** variable in the command below, and leaving it at 8 does
> not degrade performance — **it fails the boot outright**. Decode graph
> capture dies in `MultiLayerEagleMultiStepDraftExtendCudaGraphRunner`: the
> Triton extend-attention kernel cannot compile against a dummy batch whose
> `kv_indices` is `None`, the scheduler raises, and the server never becomes
> healthy. It is not a memory failure — 18.10 GB was free when it happened.
>
> Receipt: `logs/mimo26.crash-cudagraph-20260922-1036.log`. Full diagnosis and
> the four candidate fixes are in [The EAGLE MTP failure](#the-eagle-mtp-failure).
>
> **So switching to EAGLE means giving up decode CUDA graphs.** That is a real
> cost, and it partly offsets EAGLE's prose advantage.

```bash
sed -i 's/^SPEC_ALGO=.*/SPEC_ALGO=EAGLE/; s/^SPEC_DRAFT_TOKENS=.*/SPEC_DRAFT_TOKENS=4/; s/^CUDA_GRAPH_MAX_BS_DECODE=.*/CUDA_GRAPH_MAX_BS_DECODE=0/; s/^MAX_TOTAL_TOKENS=.*/MAX_TOTAL_TOKENS=2743808/' .env
./start.sh restart
```

Expect `cuda graph: False` on every `Decode batch` line afterwards — that is
correct for EAGLE, not a fault. `accept len` is now out of 4, not 8.

**No speculation**, for a clean baseline:

```bash
sed -i 's/^SPEC_ALGO=.*/SPEC_ALGO=off/' .env && ./start.sh restart
```

Confirm which one came up, and how well it is drafting, from the engine log —
`accept len` is the drafter's real score, out of `SPEC_DRAFT_TOKENS`:

```bash
grep -o "'speculative_algorithm': '[A-Z]*'" logs/mimo26.log | tail -1
grep -E "Decode batch" logs/mimo26.log | tail -3   # accept len: N of the window
```

**One caveat on the prose/code split.** The direction is well supported — it
follows from the acceptance mechanics, and the 2x-Spark vLLM recipe for this
same checkpoint measures DFlash at 6.85/7 accepted on structured text, 5.23/7
on code and 1.23/7 on prose. But that is a different engine, and **this kit has
no per-category measurement of its own yet**: EAGLE was measured at accept len
2.0–2.6 of 4 and ~19–28 tok/s on one stream, DFlash at 2.2–3.0 of 8 on boot
smoke prompts, and neither is a category breakdown. `bench/decode.py` exists to
settle it — run it once per drafter with the same `--seed` and compare.

Two notes on the CUDA-graph interaction, since it constrains the choice:
decode graph capture **works under DFlash** and **fails under EAGLE with
`--enable-multi-layer-eagle`** (see "CUDA graphs"), and DFlash's drafter keeps
its **own KV pool sized to `MAX_TOTAL_TOKENS`** (7.88 GB alongside the target's
11.17 GB at a 1,653,056-token pool), so at a fixed memory fraction DFlash buys
roughly half the pool EAGLE would.

### Performance — DFlash draft model

Measured with [sparkDash](https://github.com/MiaAI-Lab/sparkDash), thinking off,
`SPEC_ALGO=DFLASH`, `SPEC_DRAFT_TOKENS=8`, decode CUDA graphs on.
**Workload dominates every other variable here** — same engine, same config,
same boot:



**Code**

| streams | aggregate tok/s | per stream | TTFT |
|---|---:|---:|---:|
| ×1 | **69.8** | 69.8 | 349 ms |
| ×2 | 97.8 | 48.9 | 383 ms |
| ×4 | 120.5 | 33.9 | 610 ms |
| ×8 | **190.3** | 26.3 | 530 ms |

**Prose**

| streams | aggregate tok/s | per stream | TTFT |
|---|---:|---:|---:|
| ×1 | 25.8 | 25.8 | 289 ms |
| ×2 | 42.9 | 21.6 | 486 ms |
| ×4 | 68.9 | 17.9 | 403 ms |
| ×8 | **103.0** | 13.9 | 456 ms |

**Code is 2.71x prose at one stream** (69.8 vs 25.8), narrowing to 1.85x at
eight (190.3 vs 103.0). That is the drafter's acceptance rate showing up as
throughput: DFlash proposes 8 tokens per step either way, and on code most of
them survive verification while on prose most are thrown away. Put another way,
**eight concurrent code streams each decode about as fast as one prose stream**
(26.3 vs 25.8 per stream).

Two second-order things the pair of sweeps shows:

- **Code scales worse in aggregate** — 2.73x from ×1 to ×8, against prose's
  3.99x — precisely because it starts so much higher and is already closer to
  the memory-bandwidth ceiling at low concurrency. Prose has more idle capacity
  to fill, so concurrency flatters it.
- **TTFT is broadly flat for both** (code 349 → 530 ms, prose 289 → 456 ms) and
  noisy between rungs, so read it as a band rather than a curve.

### Performance — EAGLE MTP

Same harness and conditions, `SPEC_ALGO=EAGLE`, `SPEC_DRAFT_TOKENS=4`, decode
CUDA graphs **off** (mandatory under EAGLE MTP), and no server-side repetition
penalty.

> A `repetition_penalty` of 1.05 was briefly a server-side default here and has
> been removed. A three-request A/B suggested it cost ~12% on EAGLE, but the
> DFlash sweeps measured before and after removal came back **flat** (−0.5% to
> −7%, i.e. noise), so that 12% is not established and should not be quoted.
> What is certain is that no number on this page carries the penalty.



**Prose**

| streams | aggregate tok/s | per stream | TTFT |
|---|---:|---:|---:|
| ×1 | **35.0** | 35.0 | 332 ms |
| ×2 | 49.0 | 25.5 | 553 ms |
| ×4 | 75.7 | 20.3 | 620 ms |
| ×8 | **119.9** | 15.5 | 665 ms |

**Code**

| streams | aggregate tok/s | per stream | TTFT |
|---|---:|---:|---:|
| ×1 | **46.0** | 46.0 | 380 ms |
| ×2 | 68.6 | 34.6 | 605 ms |
| ×4 | 97.3 | 25.3 | 744 ms |
| ×8 | **138.7** | 18.1 | 833 ms |

EAGLE is faster on code than on prose too (1.31x at one stream), just far less
dramatically than DFlash's 2.71x — a 4-token window cannot exploit predictable
text the way an 8-token one can.

### EAGLE MTP vs DFlash — the full matrix

All four sweeps are penalty-free and otherwise identically configured, so these
are directly comparable. Aggregate tok/s, winner in bold:

| | EAGLE MTP | DFlash | winner |
|---|---:|---:|---|
| **code** ×1 | 46.0 | **69.8** | DFlash **+52%** |
| **code** ×8 | 138.7 | **190.3** | DFlash +37% |
| **prose** ×1 | **35.0** | 25.8 | EAGLE **+36%** |
| **prose** ×8 | **119.9** | 103.0 | EAGLE +16% |

Per concurrency:

| streams | code: DFlash vs EAGLE | prose: EAGLE vs DFlash |
|---|---:|---:|
| ×1 | +52% | +36% |
| ×2 | +43% | +14% |
| ×4 | +24% | +10% |
| ×8 | +37% | +16% |

**Each drafter wins its own workload at every concurrency, and the gap is
largest at one stream.** Running the wrong one costs real throughput: DFlash on
prose gives up **26%**, EAGLE on code gives up **34%**. There is no
per-request switch (see below), so this is a boot-time decision that should
follow your actual traffic.

At one stream the best available on this fleet is **69.8 tok/s on code
(DFlash)** and **35.0 tok/s on prose (EAGLE MTP)** — no single configuration
reaches both.

Three things worth drawing out:

- **EAGLE takes prose without CUDA graphs**, which DFlash has and EAGLE cannot
  use. The drafter's fit to the workload outweighs graph capture. Acceptance is
  the whole story: EAGLE lands 2.6–2.7 of 4 (~0.55 rate) against DFlash's
  2.2–3.0 of 8 (~0.3) on prose. **A narrower window that lands beats a wider
  one that does not** — DFlash spends 8 draft slots to accept what EAGLE
  accepts out of 4. On code the acceptance ordering reverses and so does the
  result.
- **Aggregate scaling ×1 → ×8** runs opposite to single-stream strength, as
  expected: DFlash code 2.73x, EAGLE code 3.02x, EAGLE prose 3.43x, DFlash
  prose 3.99x. Whichever configuration starts lowest has the most idle capacity
  for concurrency to fill, so the gaps narrow under load without ever closing.
- TTFT does not separate them (EAGLE 332–833 ms, DFlash 289–610 ms across both
  workloads) and is noisy between rungs; it is not a reason to pick either.

The workload split in the comparison table at the top of this section is now
measured on both sides rather than inferred.

Prefill is unaffected by the drafter and is in "Performance — prefill" below.
EAGLE MTP's prose sweep, and the head-to-head, are in the next section.

For reference, and *not* comparable to the tables above: this kit's own
`bench/decode.py` measured DFlash at C1 on 128-token runs as counting 73.1
aggregate tok/s, prose 28.3, code 26.7, with accepted-draft medians of 3.27 of
8 on prose and 4.65 on code. It excludes TTFT from the decode rate and pins
output length, and 128 tokens is short enough that warmup still shows — which
is why its code figure disagrees with the 69.8 above. Treat sparkDash as the
published number and `bench/` as the A/B instrument.

SGLang carries DFLASH natively (`srt/models/dflash.py`, `srt/speculative/dflash_worker_v2.py`)
and `mimo_v2.py` already implements the aux-hidden-state capture the drafter
needs (`AuxHiddenStatePacker`, `layers_to_capture`), so no target patch is
required — unlike vLLM, where that marker had to be added by hand. The
drafter's `dflash_config.target_layer_ids` is `[0, 11, 23, 35, 47]` against
`num_target_layers: 48`, which matches this checkpoint exactly.

DFLASH costs **+1.37 GiB per rank** over EAGLE (82.79 vs 81.42 GiB resident),
which `boot.py` now counts: `drafter_files()` measures whichever drafter
`SPEC_ALGO` will actually load, so the preflight is not under-estimating the
rank by the difference.

DFLASH forces `speculative_num_steps` and `speculative_eagle_topk` to 1 itself
and warns if they are set otherwise, so `boot.py` emits neither under DFLASH;
`SPEC_STEPS` / `SPEC_TOPK` / `ENABLE_MTP` are read only by EAGLE.

`SPEC_DRAFT_TOKENS` becomes `--speculative-num-draft-tokens`, which SGLang
documents `--speculative-dflash-block-size` as an alias of for DFLASH.

The published `dflash/config.json` has a trailing comma before its closing
brace, so it is not valid JSON and transformers refuses it. The checkpoint is
mounted read-only and shared to the worker over NFS, so `boot.py` does not edit
it: under `SPEC_ALGO=DFLASH` it writes a re-serialized `config.json` into
`/state/dflash` and symlinks the weights beside it, then points
`--speculative-draft-model-path` there. If the trailing-comma repair does not
yield valid JSON it refuses rather than guessing.

## Performance — prefill

Measured with [sparkDash](https://github.com/MiaAI-Lab/sparkDash) on the
shipped profile (`CHUNKED_PREFILL_SIZE=2048`, `MEM_FRACTION_STATIC=0.93`,
decode graphs on). Speculation does not touch prefill, so these hold for either
drafter — and unlike the decode tables, the removed `repetition_penalty` does
not affect them: it acts on sampling, and prefill samples one token.

| context | prompt tokens | TTFT | prefill tok/s |
|---|---:|---:|---:|
| 4K | 4,130 | 2.04 s | 2,021.7 |
| 8K | 8,226 | 3.22 s | **2,558.1** |
| 16K | 16,421 | 6.43 s | 2,552.5 |
| 32K | 32,805 | 14.16 s | 2,316.9 |
| 64K | 65,569 | 36.78 s | 1,783.0 |
| 128K | 131,104 | 106.84 s | 1,227.1 |
| 256K | 262,179 | 343.11 s | 764.1 |

The curve has two regimes, and both are expected:

- **4K is slower than 8K** (2,022 vs 2,558) because fixed per-request cost —
  tokenize, template, prefix-cache hash, queue, first-chunk ramp — amortizes
  poorly at short lengths. That is the TTFT a fresh chat session feels.
- **Past 16K it decays with length**, to 30% of peak at 256K. Each 2048-token
  chunk attends over the whole prefix so far, so per-chunk cost grows as the
  prompt does. A 256K prompt is 128 chunks, each more expensive than the last.

`CHUNKED_PREFILL_SIZE` has **not been laddered on this engine** — 2048 was
adopted from the GLM-5.3 kit, where it beat 1024, 3584 and 4096 on every rung.
Upstream's B300 recipe uses 49152, which this box has nowhere near the slack
for at 0.93, but 4096 is an untested rung worth one boot. `bench/cold_prefill.py`
runs this ladder with a unique salt per request, so the prefix cache cannot
flatter the result.

For a like-for-like reference point: the 2x-Spark **vLLM** recipe for this same
checkpoint reports 1,697.9 tok/s at 7.9K and 656.4 at 248K, so this SGLang
profile is running **1.5x faster in the 8K–64K band** and 1.16x at 256K. Not a
controlled comparison — different engine, different MoE runner (marlin vs
`flashinfer_mxfp4`), 300K max-model-len against 1M — but it is the same model on
the same two boxes.

```bash
# image
curl http://10.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"MiMo-v2.6-Flash","max_tokens":128,"temperature":1.0,"top_p":0.95,
       "messages":[{"role":"user","content":[
         {"type":"image_url","image_url":{"url":"https://example.com/photo.jpg"}},
         {"type":"text","text":"What is in this image?"}]}]}'

# video (ffprobe) and audio (torchcodec) use the same content list
# with type video_url / audio_url
```

Set `API_KEY` in the profile env to require a bearer token (`state-tpN/api-key`).

## SSD offload

In `.env`:

```bash
OFFLOAD_MODE=ssd
SSD_DIR=$HOME/mimo26-ssd-tp2          # head NVMe
WORKER_SSD_DIR=/home/zurih/mimo26-tp2/ssd
HICACHE_SIZE_GB=4                     # unified-memory staging, per rank
```

Each rank gets its own subdirectory so file-backend buckets do not collide.
This does not move expert weights off the GPU. It only spills hierarchical KV
to disk. Leave it `off` until a long-context run actually needs the extra tier.
`OFFLOAD_MODE=ram` and `cpu` are refused: host RAM is GPU memory on a Spark.

## Kernels

GB10 defaults: Triton attention, Triton multimodal attention, FlashInfer MXFP4
MoE, FP8 E4M3 KV. Override with `ATTENTION_BACKEND`, `MM_ATTENTION_BACKEND`,
`MOE_RUNNER_BACKEND`, `KV_CACHE_DTYPE`.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` is set because expandable
segments produced NaN logits on this same GB10 stack with another SGLang model.

### Long-context attention on GB10

The 9 full-attention layers read the whole prefix every step, so they dominate
long prompts. Three image-level patches cover them; switch them off in `.env` to
fall back to the stock kernels:

| Switch (default) | What it does |
|---|---|
| `SGLANG_MIMO_CUDA_VERIFY=1` | grouped-head split-KV kernel for EAGLE/DFlash target verify (`patches/mimo_cuda_verify.py`) |
| `SGLANG_MIMO_PREFILL_IMPL=flashinfer` | FlashInfer FA2 paged prefill for prefixes ≥ `SGLANG_MIMO_PREFILL_MIN_PREFIX` (16384); needs `patches/flashinfer-fp8-qk192-vo128.patch` ([flashinfer#5085](https://github.com/flashinfer-ai/flashinfer/pull/5085)) |
| `SGLANG_MIMO_PREFILL_GROUPED=1` | grouped Triton prefill kernel, the fallback for any batch FlashInfer refuses (`patches/mimo_prefill_grouped.py`) |

Head logs print `MIMO_CUDA_VERIFY_DISPATCH` once per captured verify batch size
and `MIMO_PREFILL_GROUPED_DISPATCH flashinfer-fa2` at the first long prefill.
`MIMO_PREFILL_FLASHINFER_FALLBACK` names any batch that went to the Triton fallback.

Measured on two GB10s, TP=2, FP8 KV, one request, 384–512 output tokens, thinking
off, greedy:

| Prompt | Stock decode | Patched, EAGLE | Patched, DFlash (code / JSON / prose) | Stock → patched cold TTFT |
|---|---:|---:|---:|---:|
| 128k | 17.1 tok/s | 35.5 tok/s | 48.7 / 68.6 / 33.2 tok/s | 87 s → 68 s |
| 512k | 6.6 tok/s | 29.0 tok/s | 39.7 / 56.6 / 33.0 tok/s | 945 s → 615 s |

With thinking on, T=1.0 and top_p=0.95, DFlash decodes at 42.6 / 67.4 / 32.2 tok/s at
128k (code / JSON / prose). EAGLE gets 37.4 / 43.4 / 30.8 tok/s on the same prompts.

Notes from these runs:

- DFlash beats EAGLE on long prompts even for prose. With DFlash, `MAX_TOTAL_TOKENS=2299968`
  tripped the memory guard during graph capture once the verify workspace was in;
  `1500000` boots and still holds two 600k sessions.
- Several sessions at once stay bandwidth-bound on the MoE weights, not on attention:
  - 4–6 concurrent agents get about 15–19 tok/s each.
  - MoE takes about 60% of GPU time at 4 concurrent requests.
- A new request waits behind a long prompt that is already prefilling, because
  SGLang chunks one request at a time.
- We tried several other options and none improved this workload:
  - `--schedule-policy shortest-prefill-first`,
  - `--prefill-decode-interval`,
  - DFlash with 4 draft tokens,
  - 2 NCCL channels,
  - HiCache at a 1.5M pool, which did not fit in memory.
- Quality: a synthetic suite (24 needles at 128k/512k, 10 checked prompts, logprob
  comparison) is unchanged. The new prefill kernels are closer to FP32 than the
  stock kernel: 0.2–0.4% relative L2 error, against 4%.

## Benchmarks

Ported from the GLM-5.3-Flash 2x-Spark kit, adapted for SGLang. **Run them
from a worker, never the head** — the head runs the HTTP server, tokenizer,
detokenizer and NFS export, and is the rank the other waits for.

```bash
BENCH_BASE_URL=http://10.0.0.1:8888 python3 bench/cold_prefill.py
BENCH_BASE_URL=http://10.0.0.1:8888 python3 bench/decode.py --concurrency 1,2,4,8
```

Receipts land in `bench/results/` (gitignored). Both refuse to run against a
busy server (`/v1/loads`), and both take `--out-name` so an A/B writes two
labelled receipts.

What the protocol buys, and why these details are not optional:

- **`cold_prefill.py`** puts a unique salt on every request, so the radix
  cache cannot serve it. Without that a "cold" ladder silently measures
  prefix-cache hits. Prompt length is verified through `/tokenize` to within
  0.5% rather than assumed, and TTFT is the first *content* token. It also runs
  one prefix-cache follow-up row, which should be far faster than its cold twin.
- **`decode.py`** pins output length with `min_tokens` + `ignore_eos`, because
  a run that stops early is not comparable to one that does not. The decode
  rate excludes TTFT — `(emitted - 1) / (last - first content token)`. Counting
  SSE chunks would under-report badly, since one chunk can carry a whole
  accepted speculative block.
- **Acceptance is per category, and that is the point.** This checkpoint ships
  two drafters that fail differently — acceptance is high on structured and
  code output, low on prose — so a blended number hides which drafter suits a
  workload. Acceptance is not on the OpenAI path, so it is read from the
  engine's `Decode batch` lines for each arm's window.

`ENABLE_METRICS=1` and `ENABLE_CACHE_REPORT=1` (both default on) give
`/metrics` and `prompt_tokens_details.cached_tokens`; the harnesses degrade to
"unknown" rather than failing when they are off.

## What is in the box

```
start.sh              doctor / build / share / download / serve / restart / stop / status / logs / smoke
stop.sh               teardown; waits for the driver to release the GPU on every node
boot.py               in-container entrypoint: engine + memory preflight, arg planner,
                      DFlash config repair, userspace OOM guard, smoke
Dockerfile            FROM $BASE_IMAGE + ffmpeg, torchcodec/torchaudio import gate
.env.example          the profile; copied to .env on first run
patches/
  sitecustomize.py    import hooks: per-tensor weight materialization and
                      MTP fused-qkv resharding
files/                NFSv4 export helpers, NCCL setup
scripts/remote.py     ssh helper for the worker
tests/                arg planner + memory preflight, stop/restart guard rails,
                      in-image patch checks
```

## Attribution

- Decode figures come from [sparkDash](https://github.com/MiaAI-Lab/sparkDash),
  which drives the OpenAI endpoint at a fixed stream count and reports
  aggregate and per-stream rates with TTFT.
- Serving engine: [SGLang](https://github.com/sgl-project/sglang). MiMo-V2.6
  MXFP4 expert support is
  [#40448](https://github.com/sgl-project/sglang/pull/40448); the reference
  launch settings come from SGLang's
  [MiMo-V2.6 cookbook](https://docs.sglang.io/cookbook/autoregressive/Xiaomi/MiMo-V2.6).
- Model weights are Xiaomi's and are not redistributed here. Pinned revision
  `3b38d063180c3e4aed9691fdc735f3d10b266ee4`.
- Several Spark-specific settings were carried over from sibling kits on this
  fleet rather than rediscovered: the NCCL connection-buffer trim and
  `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0` from a DeepSeek-V4.1 profile, and
  the GB10 UVM/swap hygiene notes from a GLM-5.3 one.

## License

Copyright (C) 2026 Mia'a AI Lab.

This repository is licensed under the **GNU Affero General Public License v3.0**
([LICENSE](LICENSE)). If you run a modified version as a network service, the
AGPL requires you to offer its source to the users of that service. Model
weights are Xiaomi's and are not part of this repository.

## Tests

```bash
python3 -m unittest tests.test_boot_args -v     # arg planner + memory preflight, no torch needed
bash tests/test_stop_restart.sh                 # stop/restart guard rails; safe against a live fleet
docker run --rm --entrypoint python3 -v "$PWD:/w:ro" -w /w \
  lmsysorg/sglang:nightly-dev-cu13-20260921-0f6761b5 tests/test_engine_patches.py   # MTP qkv patch, CPU
```

Measured loads at TP=2 (`Load weight end` lines):

| loader | target model | resident | each EAGLE draft |
|---|---:|---:|---:|
| default mmap (first boot, worker) | 525 s | 82.74 GB | crashed on the fused qkv |
| `--weight-loader-disable-mmap` (second boot) | 131 s | 89.79 GB | crashed the nodes re-reading 65 shards |
| mmap + per-tensor `clone()` patch (third boot, head) | 96 s | 82.71 GB | 0.5 to 0.7 s, 1 file |

With the pool sized at `MAX_TOTAL_TOKENS=262144`: full-attention KV 1.4 GB,
SWA pool 78,592 tokens at 3.65 GB, three draft pools of 0.1 GB, and
`Memory pool end. avail mem=20.49 GB` on the head (21.84 GB on the worker).
