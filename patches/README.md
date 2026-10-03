# patches/

This directory is bind-mounted into every rank at `/opt/mimo26/patches` and
put first on `PYTHONPATH` by `boot.py`, so `sitecustomize.py` runs in every
Python process of the container. Python-level fixes use that import hook;
engine source fixes are applied during the image build.
Keep the hook free of sglang/torch imports at module level: register an import
hook and patch when SGLang imports the module itself, so no CUDA context exists
before the scheduler forks.

## MiMo vision windows and key-zero bias

The pinned Triton vision backend drops `window_size` and `s_aux`. MiMo's
vision reference applies an inclusive local window and a learned per-head
bias to the first real key, with that key still subject to the window.
Ignoring those settings changes image features and can produce wrong answers.

`vision-attention.patch` adds those optional operations to SGLang's existing
Triton kernel. The Docker build checks the two original source hashes and
applies it without fuzz. `sitecustomize.py` enables the new semantics only
for MiMo vision's Triton backend. Other backends and default kernel callers
retain their behavior. The patch is tied to the pinned engine; a different
base image must be checked before updating the hashes.

After updating this recipe, stop the ranks and run `./start.sh build` before
serving. That command rebuilds and transfers the image to every worker.
Updating only the bind-mounted Python hook is insufficient; it reports a
rebuild error when used with an old kernel.

Numerical checks and exact synthetic-image reproductions are documented in
[tests/fixtures/vision/README.md](../tests/fixtures/vision/README.md).

## DFlash image forwarding

MiMo's auxiliary-state capture branch calls the language model directly,
skipping `general_mm_embed_routine`. DFlash enables that branch, so image
placeholders reach the language model without their visual embeddings.

`dflash-multimodal.patch` selects multimodal or plain execution first, then
unpacks the auxiliary states. It follows the Qwen/GLM pattern in the pinned
engine and keeps the shared embedding, cache and chunk handling unchanged.
The Docker build checks the original model source hash before patching it.

The CPU regression runs the actual forward and shared helper with CPU tensor
fixtures, covering image chunks, decode/verify, no-new-image batches, graph buffers
and auxiliary-state identity:

```bash
docker run --rm --entrypoint python3 -v "$PWD:/w:ro" -w /w \
  "$IMAGE" tests/check_dflash_multimodal.py
```

## sitecustomize.py: MTP draft fused qkv at TP != checkpoint TP

The pinned nightly reshards the TP=4-interleaved fused `qkv_proj` onto TP=2
for the target model (`load_mimo_v2_qkv_proj_weight` merges shards, then
`_resolve_deferred_qkv_scale_inv` de-interleaves and re-quantizes the block
scales; log line `Resolved deferred qkv scale_inv … de-interleaved QKV`).
The MTP draft loader in `mimo_v2_nextn.py` does not pass the deferred dict,
so every rank failed with

```
ValueError: qkv_proj scale_inv model.mtp_block.self_attn.qkv_proj.weight_scale_inv:
shape mismatch (116, 32) vs (58, 32) due to block quantization ceiling; pass deferred_scale_inv dict
```

after the 82 GiB target load had completed. The patch routes the draft's
qkv loads through a deferred dict, picks the global/SWA geometry from the
fused row count (the MTP blocks use the SWA geometry, 14848 rows, which the
`layers.N.`-based helper cannot see), and resolves the scales after
`MiMoV2MTP.load_weights`. It logs
`mimo26 patch (MTP fused qkv at TP != checkpoint TP): installed` and becomes
a no-op once upstream passes the dict itself.

Test (inside the engine image, CPU only):

```bash
docker run --rm --entrypoint python3 -v "$PWD:/w:ro" -w /w \
  lmsysorg/sglang:nightly-dev-cu13-20260921-0f6761b5 tests/test_engine_patches.py
```

## EAGLE draft-extend CUDA graphs with sliding-window attention

`eagle-swa-draft-extend-cudagraph.patch` backports the two buffer/metadata
changes from [SGLang PR #29858](https://github.com/sgl-project/sglang/pull/29858)
to the pinned engine. Draft-extend fills the persistent SWA prefix indices
and offsets before capture and replay, and its metadata exposes those buffers.
The pinned translator API is used; non-SWA metadata retains None window fields.
The Docker build checks the original SHA256 and applies the patch without fuzz.

Run `./start.sh build` to rebuild and transfer the image before serving.
The CPU regression executes the installed methods with a stand-in for the GPU
window builder, covering SWA/non-SWA, capture warmup, zero-length padded prefixes,
replay refresh, address stability and different per-request draft token counts:

```bash
docker run --rm --entrypoint python3 -v "$PWD:/w:ro" -w /w \
  "$IMAGE" tests/check_eagle_swa_draft_extend.py
```

This CPU check does not validate the Triton window kernel or GPU capture.
After deployment, check capture on both ranks and smoke, then compare output
quality, speculative acceptance and throughput against eager on the same inputs.
Acceptance need not decrease monotonically across successive log entries.

## SWA prefix window locks

`swa-prefix-window-lock.patch` backports [PR #41580](https://github.com/sgl-project/sglang/pull/41580) for [issue #41579](https://github.com/sgl-project/sglang/issues/41579). A device SWA lock splits a live, unlocked ancestor so it protects only the remaining page-aligned window. Locked, tombstoned, write-through-pending nodes and host locks preserve their behavior. Full-cache locks and lock-receipt identity are retained. Set `SGLANG_SWA_LOCK_WINDOW_ONLY=0` to disable the optimization. The Docker build verifies both original source hashes and applies with zero fuzz. This fixes admission starvation; it is not a claim of faster token generation.

## GB10 MiMo prefill tile

`gb10-mimo-extend-tile.patch` selects a 32x64 tile with 4 warps for SM121, MiMo TP=2, QK192/V128 and at least 512 extend tokens. It covers BF16 Q/K/V with BF16 or E4M3 KV cache, causal full attention and SWA128/sinks. Custom masks, score modifiers, other shapes/architectures and short draft/verify steps keep the original path. The build checks the original kernel hash. `SGLANG_GB10_MIMO_EXTEND_TUNING=0` disables the selection; start.sh forwards both this switch and `SGLANG_SWA_LOCK_WINDOW_ONLY`. Local workspace benchmark artifacts document numerical tolerance, tested shapes and timings.
