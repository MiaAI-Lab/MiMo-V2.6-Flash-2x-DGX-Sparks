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
