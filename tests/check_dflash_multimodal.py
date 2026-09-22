"""CPU execution of exact installed forward and shared helper ASTs; no GPU/model load."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--engine-root', type=Path,
                    default=Path('/sgl-workspace/sglang/python/sglang/srt'))
parser.add_argument('--original-model', type=Path,
                    default=Path('/opt/mimo26/original_mimo_v2.py'))
args = parser.parse_args()
ENGINE = args.engine_root
old = args.original_model
OLD_SHA = 'a3dd3c7e9d59345c1e435cf43bf569ed1e5dd0ce75a588881283399de8e79656'
assert hashlib.sha256(old.read_bytes()).hexdigest() == OLD_SHA

def extract(path, name, cls=None, ns=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body if cls is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    fn = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
    env = {'torch': torch} | (ns or {})
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), env)
    return env[name]

checks = []
def case(path, mm, aux, mode, contains=True, graph=False, first=True, last=True):
    calls = []
    ids = torch.tensor([1, 2, 3])
    image = torch.full((3, 2), 7.)
    text = torch.full((3, 2), 2.)
    h, norm, auxiliary = object(), object(), object()
    pp = object()
    class Language:
        pp_group = NS(is_first_rank=first)
        def get_input_embeddings(self):
            return lambda tokens: text
        def __call__(self, input_ids=None, positions=None, forward_batch=None, input_embeds=None, pp_proxy_tensors=None):
            calls.append(('language', input_embeds, input_ids, positions, pp_proxy_tensors))
            return (h, norm, auxiliary) if aux else (h, norm)
    def embed(**kw):
        calls.append(('embed', kw))
        return image, {}
    helper = extract(ENGINE/'managers/mm_utils.py', 'general_mm_embed_routine', ns={
        'get_server_args': lambda: None, 'get_disagg': lambda: NS(language_only=False),
        'embed_mm_inputs': embed,
    })
    forward = extract(path, 'forward', 'MiMoV2ForCausalLM', {'general_mm_embed_routine': helper})
    model = NS(config=NS(encoder_only=False), capture_aux_hidden_states=aux,
        _is_multimodal=mm, model=Language(), pp_group=NS(is_last_rank=last), lm_head=object())
    model.logits_processor = lambda *a, **kw: (a, kw)
    batch = NS(forward_mode=NS(is_decode=lambda: mode=='decode', is_target_verify=lambda: mode=='target_verify'),
        contains_mm_inputs=lambda: contains, mm_inputs=[NS(mm_items=[])],
        extend_prefix_lens_cpu=[2048 if mode=='chunk' else 0], extend_seq_lens_cpu=[3],
        spec_algorithm=NS(is_eagle=lambda: False), input_embeds=torch.empty_like(image) if graph else None)
    result = forward(model, ids, ids, batch, pp_proxy_tensors=pp)
    expected_embed = mm and first and contains and mode not in ('decode', 'target_verify')
    actual_embed = any(c[0]=='embed' for c in calls)
    if actual_embed != expected_embed:
        return False  # Original DFLASH/MM routing must fail this exact check.
    lang = calls[-1]
    assert lang[3] is ids and lang[4] is pp
    if mm:
        assert lang[2] is None
        if first:
            assert torch.equal(lang[1], image if expected_embed else text)
            if graph: assert lang[1] is batch.input_embeds
            if expected_embed:
                assert batch.mm_inputs is None and batch.mm_input_embeds is image
        else: assert lang[1] is None
    else: assert lang[2] is ids and lang[1] is None
    if last:
        assert result[0][1] is h
        assert result[1]['hidden_states_before_norm'] is norm
        assert result[1]['aux_hidden_states'] is (auxiliary if aux else None)
    else: assert result is h
    return True

assert not case(old, True, True, 'extend')
checks.append('old DFLASH image bypass reproduced')
new = ENGINE/'models/mimo_v2.py'
for mm in (False, True):
    for aux in (False, True):
        for mode in ('extend', 'chunk', 'decode', 'target_verify'):
            assert case(new, mm, aux, mode, graph=mm)
            checks.append(f'mm={mm} aux={aux} mode={mode} graph={mm}')
for aux in (False, True):
    assert case(new, True, aux, 'chunk', contains=False)
    checks.append(f'no-new-image aux={aux}')
    assert case(new, True, aux, 'extend', first=False, last=False)
    checks.append(f'non-first/non-last PP aux={aux}')
print(json.dumps({'passed': True, 'checks': checks, 'installed_sha256': hashlib.sha256(new.read_bytes()).hexdigest()}, indent=2))
print('DISPATCH_CPU_PASS', len(checks))
