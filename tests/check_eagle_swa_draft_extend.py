"""CPU regression of installed draft-extend methods; no model or GPU load.

Execute real method ASTs, replacing only the GPU window builder with a CPU
stand-in. This checks prefix lengths, replay refresh and stable metadata views;
it does not validate Triton kernels or CUDA graph capture on the GPU.
"""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=Path(
    '/sgl-workspace/sglang/python/sglang/srt/layers/attention/triton_backend.py'))
args = parser.parse_args()
tree = ast.parse(args.source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TritonAttnBackend')

def extract(name, env):
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
    scope = {'torch': torch, 'ForwardMetadata': lambda **kw: NS(**kw), **env}
    exec(compile(ast.fix_missing_locations(module), str(args.source), 'exec'), scope)
    return scope[name]

mode = NS(is_decode_or_idle=lambda: False, is_target_verify=lambda: False,
          is_draft_extend_v2=lambda: True)
checks = 0
for window in (128, None, -1, 0):
    for bs in (1, 2):
        for draft_tokens in (4, 7):
            calls = []
            backend = NS(sliding_window_size=window, num_draft_tokens=draft_tokens,
                speculative_num_steps=3, device='cpu',
                qo_indptr=torch.zeros(4, dtype=torch.int32),
                kv_indptr=torch.zeros(4, dtype=torch.int32),
                cuda_graph_kv_indices=torch.zeros(32, dtype=torch.int64),
                window_kv_indptr=torch.zeros(4, dtype=torch.int32),
                cuda_graph_window_kv_indices=torch.zeros(32, dtype=torch.int64),
                cuda_graph_window_kv_offsets=torch.zeros(3, dtype=torch.int32),
                cuda_graph_window_num_kv_splits=torch.ones(3, dtype=torch.int32),
                kv_index_translator=object(), token_to_kv_pool=object())

            def fill(n, lens, reqs, indices):
                assert n == bs and indices is backend.cuda_graph_kv_indices
                backend.kv_indptr[0] = 0
                backend.kv_indptr[1:n+1] = torch.cumsum(lens, 0)
                return backend.kv_indptr[:n+1]
            backend._fill_kv_indptr_and_indices = fill

            def update(indptr, translator, reqs, size, lens, n,
                       token_to_kv_pool, window_kv_indices):
                assert indptr is backend.window_kv_indptr
                assert translator is backend.kv_index_translator
                assert token_to_kv_pool is backend.token_to_kv_pool
                assert window_kv_indices is backend.cuda_graph_window_kv_indices
                assert size == 128 and n == bs
                calls.append(lens.clone())
                indptr[0] = 0
                indptr[1:n+1] = torch.cumsum(lens.clamp(max=size), 0)
                window_kv_indices[:n] = reqs[:n] + lens
                return indptr[:n+1], window_kv_indices, lens, (lens-size).clamp(min=0)

            build = extract('_build_cuda_graph_forward_metadata', {})
            refresh = extract('_update_draft_extend_buffers', {'update_sliding_window_buffer': update})
            reqs = torch.tensor([2, 5], dtype=torch.int64)[:bs]
            lengths = torch.tensor([300, 1], dtype=torch.int32)[:bs]
            # Capture warmup: missing extend lengths must use the whole prefix.
            qo, kv, count = refresh(backend, bs, lengths, mode, None, reqs)
            assert count == draft_tokens
            assert torch.equal(qo, torch.arange(0, (bs+1)*draft_tokens, draft_tokens))
            metadata = build(backend, bs, mode, None, None, None)
            assert metadata.max_extend_len == draft_tokens
            fields = ('window_kv_indptr', 'window_kv_indices',
                      'window_num_kv_splits', 'window_kv_offsets')
            if window != 128:
                assert not calls
                assert all(getattr(metadata, name) is None for name in fields)
            else:
                assert len(calls) == 1, 'draft-extend did not fill the SWA window buffers'
                assert torch.equal(calls[-1], lengths)
                assert metadata.window_kv_indptr.numel() == bs+1
                assert metadata.window_kv_indices is backend.cuda_graph_window_kv_indices
                assert metadata.window_num_kv_splits is backend.cuda_graph_window_num_kv_splits
                assert metadata.window_kv_offsets is backend.cuda_graph_window_kv_offsets
                pointers = [getattr(metadata, name).data_ptr() for name in fields]
            # Replay: prefix lengths exclude the newly extended tokens and clamp
            # padded rows to zero. Previously captured views must see new values.
            for delta in (0, 9):
                current = lengths + delta
                spec = NS(extend_seq_lens_tensor=torch.tensor([4, 5])[:bs])
                prefix = (current-spec.extend_seq_lens_tensor).clamp(min=0).to(torch.int32)
                refresh(backend, bs, current, mode, spec, reqs)
                if window == 128:
                    assert torch.equal(calls[-1], prefix)
                    assert torch.equal(metadata.window_kv_indices[:bs], reqs+prefix)
                    assert torch.equal(metadata.window_kv_offsets[:bs], (prefix-window).clamp(min=0))
                    assert [getattr(metadata, name).data_ptr() for name in fields] == pointers
            checks += 1
print('EAGLE_SWA_DRAFT_EXTEND_CPU_PASS', checks)
