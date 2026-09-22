"""Engine hot fixes for MiMo-V2.6 on DGX Spark (SGLang nightly 2026-09-21).

boot.py puts this directory first on PYTHONPATH for every rank. This file
only registers an import hook: nothing from sglang or torch is imported
until SGLang imports the target module itself, so no CUDA context exists
before the scheduler processes fork. Each patch logs one line, and each is
a no-op once upstream carries the fix.

1. model_loader.loader (DefaultModelLoader)
   a. Materialize one tensor at a time. The default loader hands the model
      zero-copy views of the mmap'd shard, and the CUDA pageable copy then
      faults the file pages 4 KiB at a time: 17.7 s for a 2.3 GiB expert
      shard on GB10, 525 s for the 82 GiB rank. A clone() first reads the
      tensor through the kernel's readahead at ~2 GiB/s and the copy to the
      GPU runs at ~9 GiB/s. --weight-loader-disable-mmap gets the same speed
      but turns every in-flight shard into anonymous memory (2x the file:
      25 GiB for the 12.5 GiB ep0 shard) and hung both Sparks. Routed
      experts that belong to another EP rank are not cloned: the model's
      weight loader skips them without touching the data.
   b. Draft passes read only the MTP file. The multi-layer EAGLE worker
      loads one draft model per speculative step, and each load iterates
      over every shard of the checkpoint (65 files, 161 GiB) to keep the 48
      tensors of model.mtp.layers.<step>. The draft's embed_tokens/lm_head
      are replaced by the target's (init_lm_head), so the other files carry
      nothing the draft keeps.

2. models.mimo_v2_nextn (MiMoV2MTP)
   The target model reshards the TP=4-interleaved fused qkv_proj onto TP=2
   through a deferred scale dict (_resolve_deferred_qkv_scale_inv). The
   draft loader calls load_mimo_v2_qkv_proj_weight without that dict and
   dies with "qkv_proj scale_inv model.mtp_block.self_attn.qkv_proj.
   weight_scale_inv: shape mismatch (116, 32) vs (58, 32)". The MTP blocks
   use the SWA geometry, which the layers.N.-keyed helper cannot see, so
   the geometry is picked from the fused row count.
"""
from __future__ import annotations

import importlib.abc
import inspect
import json
import os
import re
import sys

TAG = "mimo26 patch"
EXPERT_RE = re.compile(r"\.mlp\.experts\.(\d+)\.")
MTP_RE = re.compile(r"^model\.mtp\.layers\.(\d+)\.")


# ─── helpers (pure, unit-tested on the host) ─────────────────────────────────

def wanted_tensor(name, ep_rank, ep_size, n_experts):
    """False for a routed expert that another EP rank stores."""
    match = EXPERT_RE.search(name)
    if match is None or ep_size <= 1 or not n_experts:
        return True
    per_rank = n_experts // ep_size
    expert = int(match.group(1))
    return ep_rank * per_rank <= expert < (ep_rank + 1) * per_rank


def draft_shard_files(files, index_map, draft_idx):
    """Files that hold model.mtp.layers.<draft_idx>.* tensors, or all files
    when the index does not list any (other checkpoint layouts)."""
    prefix = f"model.mtp.layers.{draft_idx}."
    keep = {index_map[name] for name in index_map if name.startswith(prefix)}
    if not keep:
        return list(files)
    kept = [f for f in files if os.path.basename(f) in keep]
    return kept or list(files)


def _index_map(folder):
    path = os.path.join(folder, "model.safetensors.index.json")
    try:
        with open(path) as handle:
            return json.load(handle)["weight_map"]
    except (OSError, ValueError, KeyError):
        return {}


# ─── 1. loader ───────────────────────────────────────────────────────────────

def install_loader(loader):
    cls = loader.DefaultModelLoader
    if getattr(cls, "_mimo26_materialize", False):
        return "already installed"

    original_prepare = cls._prepare_weights
    original_iterator = cls._get_weights_iterator

    def prepare_weights(self, *args, **kwargs):
        # (model_name_or_path, revision, fall_back_to_pt[, allow_patterns_overrides])
        folder, files, use_safetensors = original_prepare(self, *args, **kwargs)
        draft_idx = getattr(self.load_config, "draft_model_idx", None)
        if draft_idx is not None and use_safetensors:
            kept = draft_shard_files(files, _index_map(folder), draft_idx)
            if len(kept) < len(files):
                print(
                    f"{TAG}: draft {draft_idx} reads {len(kept)} of {len(files)} "
                    f"shard files ({', '.join(os.path.basename(f) for f in kept)})",
                    flush=True,
                )
                files = kept
        return folder, files, use_safetensors

    def get_weights_iterator(self, source):
        iterator = original_iterator(self, source)
        n_experts = 0
        ep_rank, ep_size = 0, 1
        try:
            hf_config = source.model_config.hf_config
            n_experts = int(getattr(hf_config, "n_routed_experts", 0) or 0)
            from sglang.srt.runtime_context import get_parallel

            parallel = get_parallel()
            ep_rank, ep_size = int(parallel.moe_ep_rank), int(parallel.moe_ep_size)
        except Exception:
            pass  # clone everything; still correct, only slower

        def materialized():
            for name, tensor in iterator:
                if (
                    tensor.device.type == "cpu"
                    and wanted_tensor(name, ep_rank, ep_size, n_experts)
                ):
                    tensor = tensor.clone()
                yield name, tensor

        return materialized()

    cls._prepare_weights = prepare_weights
    cls._get_weights_iterator = get_weights_iterator
    cls._mimo26_materialize = True
    return "installed (materialize per tensor, draft reads only its MTP shard)"


# ─── 2. MTP draft fused qkv ──────────────────────────────────────────────────

def _qkv_geometries(config, ckpt_tp):
    """(q, k, v) rows per checkpoint shard: global attention, then SWA."""
    out = []
    for nh, nkv, hd, vhd in (
        (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            getattr(config, "v_head_dim", config.head_dim),
        ),
        (
            config.swa_num_attention_heads,
            config.swa_num_key_value_heads,
            config.swa_head_dim,
            getattr(config, "swa_v_head_dim", config.swa_head_dim),
        ),
    ):
        out.append(
            (
                (nh // ckpt_tp) * hd,
                max(1, nkv // ckpt_tp) * hd,
                max(1, nkv // ckpt_tp) * vhd,
            )
        )
    return out


def install_nextn(nextn):
    import sglang.srt.models.mimo_v2 as mimo
    from sglang.srt.configs.model_config import get_mimo_v2_fused_qkv_expected_tp_size

    if "deferred_scale_inv" in inspect.getsource(nextn.MiMoV2MTP.load_weights):
        return "upstream handles it, nothing patched"
    if "deferred_scale_inv" not in inspect.signature(
        nextn.load_mimo_v2_qkv_proj_weight
    ).parameters:
        raise RuntimeError("engine has no deferred qkv scale_inv path")

    original_loader = nextn.load_mimo_v2_qkv_proj_weight
    original_load_weights = nextn.MiMoV2MTP.load_weights
    original_sizes = mimo._get_ckpt_qkv_shard_sizes
    pending = {}  # scale name -> checkpoint scale_inv (ckpt TP layout)
    fused_rows = {}  # scale name -> rows of the fused checkpoint weight

    def loader(name, param, loaded_weight, expected_fused_tp_size=None,
               deferred_scale_inv=None):
        if deferred_scale_inv is None:
            deferred_scale_inv = pending
        return original_loader(
            name, param, loaded_weight,
            expected_fused_tp_size=expected_fused_tp_size,
            deferred_scale_inv=deferred_scale_inv,
        )

    def shard_sizes(config, layer_name, ckpt_tp):
        sizes = original_sizes(config, layer_name, ckpt_tp)
        rows = fused_rows.get(layer_name)
        if sizes is not None or rows is None:
            return sizes
        for candidate in _qkv_geometries(config, ckpt_tp):
            if sum(candidate) * ckpt_tp == rows:
                return candidate
        raise ValueError(
            f"{layer_name}: fused qkv has {rows} rows, which is neither the "
            f"global nor the SWA geometry of this config at TP={ckpt_tp}"
        )

    def resolve(params_dict, config):
        if not pending:
            return
        ckpt_tp = get_mimo_v2_fused_qkv_expected_tp_size(config)
        tp_size = mimo.get_parallel().attn_tp_size
        for scale_name in pending:
            weight = params_dict[scale_name.replace(".weight_scale_inv", ".weight")]
            fused_rows[scale_name] = weight.shape[0] * tp_size
        try:
            mimo._resolve_deferred_qkv_scale_inv(
                params_dict, dict(pending), ckpt_tp, config=config
            )
        finally:
            pending.clear()
            fused_rows.clear()

    def load_weights(self, weights, *args, **kwargs):
        pending.clear()
        fused_rows.clear()
        result = original_load_weights(self, weights, *args, **kwargs)
        resolve(dict(self.named_parameters()), self.config)
        return result

    nextn.load_mimo_v2_qkv_proj_weight = loader
    mimo._get_ckpt_qkv_shard_sizes = shard_sizes
    nextn.MiMoV2MTP.load_weights = load_weights
    nextn._mimo26_resolve = resolve  # exercised by tests/test_engine_patches.py
    return "installed (MTP fused qkv at TP != checkpoint TP)"


# ─── import hook ─────────────────────────────────────────────────────────────

INSTALLERS = {
    "sglang.srt.model_loader.loader": install_loader,
    "sglang.srt.models.mimo_v2_nextn": install_nextn,
}


class _PatchOnImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in INSTALLERS:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                spec.loader_state = {"inner": spec.loader}
                spec.loader = self
                return spec
        return None

    def create_module(self, spec):
        return spec.loader_state["inner"].create_module(spec)

    def exec_module(self, module):
        module.__spec__.loader_state["inner"].exec_module(module)
        try:
            state = INSTALLERS[module.__name__](module)
        except Exception as exc:  # never break the import itself
            print(f"{TAG} [{module.__name__}]: NOT installed: {exc}", flush=True)
        else:
            print(f"{TAG} [{module.__name__}]: {state}", flush=True)


if not any(isinstance(f, _PatchOnImport) for f in sys.meta_path):
    sys.meta_path.insert(0, _PatchOnImport())
