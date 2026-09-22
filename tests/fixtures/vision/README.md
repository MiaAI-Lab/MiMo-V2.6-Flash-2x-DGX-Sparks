# MiMo vision reproduction

Checkpoint: `XiaomiMiMo/MiMo-V2.6-Flash-RL`, revision
`3b38d063180c3e4aed9691fdc735f3d10b266ee4`.
Engine: SGLang `0f6761b54facebb47f2068f87ecccd8f14da3a0e`, image
`lmsysorg/sglang:nightly-dev-cu13-20260921-0f6761b5`.
Hardware: two NVIDIA DGX Sparks, GB10/SM121, TP2/EP2.

## Attention definition

The [checkpoint reference](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/3b38d063180c3e4aed9691fdc735f3d10b266ee4/modeling_mimo_v2.py#L712-L723)
adds a per-head bias to the score for the first real key, after QK scaling.
The inclusive local-window mask still applies to that key. Its actual value
contributes to the output when it is inside the window. This differs from
adding a value-less extra logit to the softmax denominator.

The missing Triton arguments are tracked in
[SGLang issue 37983](https://github.com/sgl-project/sglang/issues/37983).
[PR 38142](https://github.com/sgl-project/sglang/pull/38142) supplies useful
window traversal and masked-row handling, but its value-less sink definition
differs from this checkpoint. [vLLM PR 49815](https://github.com/vllm-project/vllm/pull/49815)
also uses the real-key-zero bias for MiMo vision.

## Numerical checks

The saved [first](kernel-gb10-a.json) and [second](kernel-gb10-b.json) GB10
results each contain 12 passing cases against FP32 PyTorch SDPA, with BF16
Q/K/V, 16 query heads, 4 KV heads and head dimension 64. They include short
and variable sequences, strided tensors, window boundaries, fully masked
intermediate tiles, and unchanged full/causal kernel paths.

Maximum absolute error was 0.01414 and maximum RMSE was 0.00109. The original
full/causal paths were bitwise equal. Deliberately using full attention without
sinks, or a value-less sink, differs by more than 1 in representative cases.

These GPU receipts were collected during deployment qualification. The PR's
build step produces the same engine files:

```text
61ac3d0e04fb61ed9bb17867de66cff80d6323cc3ffc0e11b56f22ff3b3147eb  prefill_attention.py
f096a0701217acf0027fa53fb6302b07770cf8e5603aecc1ea93c80a44b12cf1  vision.py
12e02fe45c0ab307cb8055fdcdebac754e02b448cb360a81274f4f393f3447fd  mimo_v2.py
```

With a rebuilt recipe image, run on an idle GB10 without model weights:

```bash
docker run --rm --gpus all --entrypoint python3 \
  -v "$PWD:/w:ro" -w /w "$IMAGE" tests/check_vision_kernel.py
```

The image keeps the original kernel under `/opt/mimo26/` for the bitwise
comparison. `--original-kernel PATH` and `--output PATH` override the input
and optional JSON output locations. The GPU check uses synthetic tensors.

CPU checks:

```bash
python3 -m unittest discover -s tests -p test_vision_binding.py -v
docker run --rm --entrypoint python3 -v "$PWD:/w:ro" -w /w \
  "$IMAGE" tests/test_engine_patches.py
docker run --rm --entrypoint python3 -v "$PWD:/w:ro" -w /w \
  "$IMAGE" tests/check_dflash_multimodal.py
```

## Exact image requests

The PNGs are synthetic test cards. Run against an already loaded endpoint:

```bash
python3 tests/reproduce_vision_cards.py \
  --base-url http://127.0.0.1:8888 --model MiMo-v2.6-Flash
```

Use the actual served model name. `BENCH_API_KEY` is optional for endpoints
that require authentication. The script sends each image separately with
temperature 0, thinking disabled, streaming enabled and 256 output tokens.
It checks the returned JSON and normal completion.

| Saved image | Expected | Before correction | After correction |
| --- | --- | --- | --- |
| `card-fhd.png`, 1920×1080 | 873, green | Four objects with ID 7 and different colors | Exact, 2091 prompt / 22 output tokens |
| `card-4k.png`, 3840×2160 | 651, yellow | Invented cards 1 through 18; output hit the 256-token limit | Exact, 8211 prompt / 22 output tokens |

These attention checks used speculation off. DFlash exposes a separate
dispatch bug: auxiliary-state capture bypasses image embedding altogether.
With the attention correction installed but the dispatch bug present, the
same FHD request invented cards 1 through 19 and hit its 256-token limit.

After both corrections, DFlash returned the exact FHD and 4K answers with
normal stops. Repeated images and a follow-up question also passed. Saved
[DFlash results](dflash-results.json) include the actual responses, token
counts and elapsed times. That run used TP2/EP2, FP8 target/draft KV,
8 draft tokens, decode graphs for batches 1/2/4, chunked prefill 2048,
temperature 0 and thinking disabled. No broad quality claim follows from
these bounded checks.

`check_dflash_multimodal.py` reproduces the original bypass, then exercises
the installed forward method and shared helper without model weights or a
GPU. It checks image/text execution, auxiliary-state preservation, prefill
chunks, decode/verify, no-new-image batches and graph-buffer identity. The
image keeps the original model at `/opt/mimo26/original_mimo_v2.py` for this
comparison; `--original-model` and `--engine-root` override those paths.

The post-fix requests used the exact saved image bytes and prompt. The local
deployment also has a separate multi-image splitting patch; it leaves these
single-image inputs unchanged. That patch is outside this reproduction.
These checks demonstrate the reported failure, not a general vision score.

```text
a2fc2de361b55c115c3221be62cdce40bff82e03449cf374512c8d48d77ac3be  card-fhd.png
c7505038aaeb198a550260fe2596e259d751d6061a0af5ff95199d5a807bea7c  card-4k.png
```
