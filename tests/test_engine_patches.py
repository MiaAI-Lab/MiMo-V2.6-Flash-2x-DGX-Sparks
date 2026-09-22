"""CPU checks of patches/sitecustomize.py inside the engine image:
  - the loader hook installs and its helpers pick the right shards/experts,
  - the MTP draft's fused qkv lands as [q0 q1 | k0 k1 | v0 v1] on TP=2 from a
    TP=4-interleaved checkpoint.
Run:
  docker run --rm --entrypoint python3 -v "$PWD:/w:ro" -w /w \
    lmsysorg/sglang:nightly-dev-cu13-20260921-0f6761b5 tests/test_engine_patches.py
"""
import sys
import types
from pathlib import Path

# Ubuntu ships a stdlib sitecustomize that `site` already imported, so load
# ours by path (at runtime PYTHONPATH puts patches/ ahead of the stdlib).
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "mimo26_sitecustomize", Path(__file__).resolve().parents[1] / "patches" / "sitecustomize.py")
patch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patch)  # registers the import hook

# ─── loader helpers ──────────────────────────────────────────────────────────
assert patch.wanted_tensor("model.layers.3.self_attn.qkv_proj.weight", 1, 2, 256)
assert patch.wanted_tensor("model.layers.3.mlp.experts.5.down_proj.weight", 0, 2, 256)
assert not patch.wanted_tensor("model.layers.3.mlp.experts.5.down_proj.weight", 1, 2, 256)
assert patch.wanted_tensor("model.layers.3.mlp.experts.200.up_proj.weight_scale", 1, 2, 256)
assert not patch.wanted_tensor("model.layers.3.mlp.experts.200.up_proj.weight_scale", 0, 2, 256)
assert patch.wanted_tensor("model.layers.3.mlp.experts.200.up_proj.weight", 0, 1, 256)
files = ["/m/model_mtp.safetensors", "/m/model_pp0_ep0_shard0.safetensors", "/m/model_pp0_ep1_shard0.safetensors"]
index = {"model.mtp.layers.0.mtp_block.x": "model_mtp.safetensors",
         "model.mtp.layers.1.mtp_block.x": "model_mtp.safetensors",
         "model.embed_tokens.weight": "model_pp0_ep0_shard0.safetensors",
         "model.layers.1.mlp.experts.4.up_proj.weight": "model_pp0_ep1_shard0.safetensors"}
assert patch.draft_shard_files(files, index, 1) == ["/m/model_mtp.safetensors"]
assert patch.draft_shard_files(files, index, 7) == files, "unknown draft idx keeps every file"
assert patch.draft_shard_files(files, {}, 0) == files, "no index keeps every file"
print("loader helpers: ok")

import torch  # noqa: E402
import sglang.srt.model_loader.loader as loader  # noqa: E402
import sglang.srt.models.mimo_v2 as mimo  # noqa: E402
import sglang.srt.models.mimo_v2_nextn as nextn  # noqa: E402

assert getattr(loader.DefaultModelLoader, "_mimo26_materialize", False), "loader patch did not install"
print("loader hook: installed")

# ─── loader hook end to end: the real DefaultModelLoader on a 2-file checkpoint ──
import json  # noqa: E402
import tempfile  # noqa: E402

import safetensors.torch  # noqa: E402
import sglang.srt.runtime_context as runtime_context  # noqa: E402
from sglang.srt.configs.load_config import LoadConfig  # noqa: E402

tmp = tempfile.mkdtemp()
shard = {
    "model.embed_tokens.weight": torch.zeros(2, 8, dtype=torch.bfloat16),
    "model.layers.0.mlp.experts.0.up_proj.weight": torch.zeros(4, 8, dtype=torch.uint8),
    "model.layers.0.mlp.experts.200.up_proj.weight": torch.zeros(4, 16, dtype=torch.uint8),
}
mtp = {
    "model.mtp.layers.0.mtp_block.w": torch.zeros(2, 2),
    "model.mtp.layers.1.mtp_block.w": torch.ones(3, 3),
}
safetensors.torch.save_file(shard, f"{tmp}/model_pp0_ep0_shard0.safetensors")
safetensors.torch.save_file(mtp, f"{tmp}/model_mtp.safetensors")
with open(f"{tmp}/model.safetensors.index.json", "w") as handle:
    json.dump({"metadata": {}, "weight_map": {
        **{k: "model_pp0_ep0_shard0.safetensors" for k in shard},
        **{k: "model_mtp.safetensors" for k in mtp}}}, handle)

# No published runtime context here: stand in for the server-args and model
# bags the loader reads its flags from.
_flags = dict(weight_loader_disable_mmap=False, weight_loader_prefetch_checkpoints=False,
              weight_loader_prefetch_num_threads=1, weight_loader_drop_cache_after_load=True)
loader.get_server_args = lambda: types.SimpleNamespace(**_flags)
loader.get_model = lambda: types.SimpleNamespace(model_checksum=None, **_flags)
runtime_context.get_parallel = lambda: types.SimpleNamespace(moe_ep_rank=0, moe_ep_size=2)
model_config = types.SimpleNamespace(hf_config=types.SimpleNamespace(
    architectures=["MiMoV2ForCausalLM"], n_routed_experts=256))
cloned = []
_clone = torch.Tensor.clone
torch.Tensor.clone = lambda self, *a, **k: (cloned.append(tuple(self.shape)), _clone(self, *a, **k))[1]

def run_loader(draft_idx):
    cfg = LoadConfig()
    cfg.draft_model_idx = draft_idx
    ldr = loader.DefaultModelLoader(cfg)
    source = loader.DefaultModelLoader.Source(
        model_or_path=tmp, revision=None, prefix="", fall_back_to_pt=True,
        allow_patterns_overrides=None, model_config=model_config)
    return dict(ldr._get_weights_iterator(source))

target = run_loader(None)
assert set(target) == set(shard) | set(mtp), sorted(target)
assert (4, 8) in cloned and (2, 8) in cloned, cloned          # local expert + embeddings cloned
assert (4, 16) not in cloned, cloned                          # expert 200 belongs to EP rank 1
print("target pass: every file, local tensors materialized, foreign expert left as a view")

cloned.clear()
draft = run_loader(1)
assert list(draft) == ["model.mtp.layers.0.mtp_block.w"], list(draft)   # idx 1 remapped to layer 0
assert draft["model.mtp.layers.0.mtp_block.w"].shape == (3, 3)
assert cloned == [(3, 3)], cloned
print("draft pass: only model_mtp.safetensors read, only its own layer kept")
torch.Tensor.clone = _clone

# ─── MTP fused qkv ───────────────────────────────────────────────────────────
assert getattr(nextn, "_mimo26_resolve", None) is not None, "patch did not install"

CFG = types.SimpleNamespace(
    num_attention_heads=64, num_key_value_heads=4, head_dim=192, v_head_dim=128,
    swa_num_attention_heads=64, swa_num_key_value_heads=8, swa_head_dim=192,
    swa_v_head_dim=128, attention_projection_layout="fused_qkv",
)
CKPT_TP, TP, COLS, BLOCK = 4, 2, 256, 128
W = "model.mtp_block.self_attn.qkv_proj.weight"
S = W + "_scale_inv"


def quantize_padded(shard):
    """Per-shard block-128 scales with the rows padded, as the checkpoint stores them."""
    n, k = shard.shape
    n_pad = -(-n // BLOCK) * BLOCK
    padded = torch.zeros(n_pad, k)
    padded[:n] = shard
    view = padded.view(n_pad // BLOCK, BLOCK, k // BLOCK, BLOCK)
    amax = view.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    finfo = torch.finfo(torch.float8_e4m3fn)
    scale = amax / finfo.max
    q = (view / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn).reshape(n_pad, k)[:n]
    return q, scale.reshape(n_pad // BLOCK, k // BLOCK)


def dequant(w, s):
    n, k = w.shape
    expanded = s.float().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[:n, :k]
    return w.float() * expanded


def run(label, q_per, k_per, v_per):
    shards, scales, logical = [], [], []
    for r in range(CKPT_TP):
        q = torch.full((q_per, COLS), float(r + 1))
        k = torch.full((k_per, COLS), float(10 + r))
        v = torch.full((v_per, COLS), float(20 + r))
        w, s = quantize_padded(torch.cat([q, k, v], 0))
        shards.append(w)
        scales.append(s)
        logical.append((q, k, v))
    fused_w, fused_s = torch.cat(shards, 0), torch.cat(scales, 0)
    per_rank_rows = (q_per + k_per + v_per) * CKPT_TP // TP
    for rank in range(TP):
        mimo.get_parallel = lambda rank=rank: types.SimpleNamespace(attn_tp_size=TP, attn_tp_rank=rank)
        wp = torch.nn.Parameter(torch.empty(per_rank_rows, COLS, dtype=torch.float8_e4m3fn), requires_grad=False)
        sp = torch.nn.Parameter(torch.empty(-(-per_rank_rows // BLOCK), COLS // BLOCK), requires_grad=False)
        params = {W: wp, S: sp}
        # Exactly what MiMoV2MTP.load_weights does: no deferred dict passed.
        nextn.load_mimo_v2_qkv_proj_weight(W, wp, fused_w, expected_fused_tp_size=CKPT_TP)
        nextn.load_mimo_v2_qkv_proj_weight(S, sp, fused_s, expected_fused_tp_size=CKPT_TP)
        nextn._mimo26_resolve(params, CFG)
        restored = dequant(wp.data, sp.data)
        a, b = logical[rank * 2], logical[rank * 2 + 1]
        expected = torch.cat([a[0], b[0], a[1], b[1], a[2], b[2]], 0)
        err = (restored - expected).abs().max().item()
        assert err < 0.5, (label, rank, err)
        naive = torch.cat([dequant(shards[i], scales[i]) for i in (rank * 2, rank * 2 + 1)], 0)
        assert not torch.allclose(naive, expected), "test could not tell a missing de-interleave"
    print(f"{label}: ok")


run("global attention (4 kv heads)", 16 * 192, 1 * 192, 1 * 128)
run("sliding-window attention (8 kv heads, the MTP geometry)", 16 * 192, 2 * 192, 2 * 128)
print("ok")


print("ok")
