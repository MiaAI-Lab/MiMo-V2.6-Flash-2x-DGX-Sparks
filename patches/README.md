# patches/

This directory is bind-mounted into every rank at `/opt/mimo26/patches` and
put first on `PYTHONPATH` by `boot.py`, so `sitecustomize.py` runs in every
Python process of the container. It is for hot fixes to the engine that must
not wait for an image rebuild. Keep it free of sglang/torch imports at
module level: register an import hook and patch when SGLang imports the
module itself, so no CUDA context exists before the scheduler forks.

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
