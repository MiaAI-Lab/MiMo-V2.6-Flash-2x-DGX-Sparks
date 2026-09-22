"""Arg planner and memory preflight. Does not import SGLang or torch."""
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('MODEL_PATH', '/tmp/mimo-missing')
import boot  # noqa: E402


class BootArgs(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.copy()
        os.environ['NNODES'] = '1'
        os.environ['NODE_RANK'] = '0'
        os.environ['TP_SIZE'] = '2'
        os.environ['EP_SIZE'] = '2'
        os.environ['DIST_INIT_ADDR'] = ''
        for name in ('EXTRA_SGLANG_ARGS', 'MOE_RUNNER_BACKEND', 'MOE_A2A_BACKEND',
                     'WEIGHT_LOADER_DROP_CACHE', 'CUDA_GRAPH_MAX_BS_DECODE',
                     'CUDA_GRAPH_PREFILL', 'SPEC_ALGO', 'SPEC_DRAFT_TOKENS',
                     'SPEC_STEPS', 'SPEC_TOPK', 'SPEC_DRAFT_MODEL', 'ENABLE_MTP'):
            os.environ.pop(name, None)
        self._state = boot.STATE

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        boot.STATE = self._state

    def test_off_has_no_hicache(self):
        os.environ['OFFLOAD_MODE'] = 'off'
        args, _, mode = boot.build_args()
        self.assertEqual(mode, 'off')
        self.assertNotIn('--enable-hierarchical-cache', args)
        self.assertIn('--enable-multimodal', args)
        self.assertEqual(args[args.index('--speculative-algorithm') + 1], 'EAGLE')
        self.assertEqual(args[args.index('--speculative-num-steps') + 1], '3')
        self.assertEqual(args[args.index('--speculative-eagle-topk') + 1], '1')
        self.assertEqual(args[args.index('--speculative-num-draft-tokens') + 1], '4')
        self.assertIn('--enable-multi-layer-eagle', args)
        self.assertEqual(args[args.index('--kv-cache-dtype') + 1], 'fp8_e4m3')
        self.assertEqual(args[args.index('--page-size') + 1], '64')
        self.assertEqual(args[args.index('--moe-dense-tp-size') + 1], '1')
        self.assertEqual(args[args.index('--chunked-prefill-size') + 1], '32768')
        self.assertNotIn('--enable-dp-attention', args)
        self.assertEqual(args[args.index('--tp') + 1], '2')

    def test_moe_defaults_consume_packed_mxfp4_experts(self):
        os.environ['OFFLOAD_MODE'] = 'off'
        args, _, _ = boot.build_args()
        self.assertEqual(args[args.index('--moe-runner-backend') + 1], 'flashinfer_mxfp4')
        self.assertEqual(args[args.index('--moe-a2a-backend') + 1], 'none')
        self.assertIn('--weight-loader-drop-cache-after-load', args)
        self.assertNotIn('--weight-loader-disable-mmap', args)
        self.assertNotIn('--model-loader-extra-config', args)
        self.assertNotIn('--disable-cuda-graph', args)

    def _draft_dir(self, body):
        folder = Path(tempfile.mkdtemp())
        (folder / 'config.json').write_text(body)
        (folder / 'dflash_draft_model.safetensors').write_bytes(b'\0')
        return folder

    def test_eagle_is_the_default_and_keeps_its_own_flags(self):
        os.environ['OFFLOAD_MODE'] = 'off'
        args, _, _ = boot.build_args()
        self.assertEqual(args[args.index('--speculative-algorithm') + 1], 'EAGLE')
        self.assertEqual(args[args.index('--speculative-num-draft-tokens') + 1], '4')
        self.assertEqual(args[args.index('--speculative-num-steps') + 1], '3')
        self.assertEqual(args[args.index('--speculative-eagle-topk') + 1], '1')
        self.assertIn('--enable-multi-layer-eagle', args)
        self.assertNotIn('--speculative-draft-model-path', args)

    def test_dflash_drops_the_eagle_only_flags(self):
        os.environ['OFFLOAD_MODE'] = 'off'
        os.environ['SPEC_ALGO'] = 'DFLASH'
        os.environ['SPEC_DRAFT_TOKENS'] = '7'
        folder = self._draft_dir('{"architectures": ["DFlashDraftModel"]}')
        os.environ['SPEC_DRAFT_MODEL'] = str(folder)
        args, _, _ = boot.build_args()
        self.assertEqual(args[args.index('--speculative-algorithm') + 1], 'DFLASH')
        self.assertEqual(args[args.index('--speculative-num-draft-tokens') + 1], '7')
        # A valid config needs no repair, so the checkpoint folder is used as-is.
        self.assertEqual(args[args.index('--speculative-draft-model-path') + 1], str(folder))
        for flag in ('--speculative-num-steps', '--speculative-eagle-topk',
                     '--enable-multi-layer-eagle'):
            self.assertNotIn(flag, args)

    def test_dflash_repairs_a_trailing_comma_without_touching_the_checkpoint(self):
        os.environ['OFFLOAD_MODE'] = 'off'
        os.environ['SPEC_ALGO'] = 'DFLASH'
        folder = self._draft_dir('{\n "model_type": "qwen3",\n "use_cache": true,\n\n}\n')
        os.environ['SPEC_DRAFT_MODEL'] = str(folder)
        with tempfile.TemporaryDirectory() as state:
            boot.STATE = Path(state)
            args, _, _ = boot.build_args()
            served = Path(args[args.index('--speculative-draft-model-path') + 1])
            self.assertEqual(served, Path(state) / 'dflash')
            self.assertEqual(json.loads((served / 'config.json').read_text()),
                             {'model_type': 'qwen3', 'use_cache': True})
            # weights are symlinked, never copied
            weights = served / 'dflash_draft_model.safetensors'
            self.assertTrue(weights.is_symlink())
            self.assertEqual(weights.resolve(), (folder / weights.name).resolve())
        # the published file is left exactly as it was
        self.assertIn(',\n\n}', (folder / 'config.json').read_text())

    def test_dflash_treats_an_empty_draft_model_as_unset(self):
        """start.sh forwards SPEC_DRAFT_MODEL= unconditionally; empty must fall
        back to $MODEL_PATH/dflash, not resolve to the current directory."""
        os.environ['OFFLOAD_MODE'] = 'off'
        os.environ['SPEC_ALGO'] = 'DFLASH'
        os.environ['SPEC_DRAFT_MODEL'] = ''
        with self.assertRaises(RuntimeError) as caught:
            boot.build_args()
        # boot.MODEL is /tmp/mimo-missing in this suite, so it must complain
        # about that path rather than about a bare 'config.json'.
        self.assertIn(str(boot.MODEL / 'dflash'), str(caught.exception))

    def test_dflash_without_a_draft_model_refuses(self):
        os.environ['OFFLOAD_MODE'] = 'off'
        os.environ['SPEC_ALGO'] = 'DFLASH'
        os.environ['SPEC_DRAFT_MODEL'] = '/tmp/mimo-no-such-drafter'
        with self.assertRaises(RuntimeError) as caught:
            boot.build_args()
        self.assertIn('SPEC_ALGO=EAGLE', str(caught.exception))

    def test_spec_off_emits_no_speculative_flags(self):
        os.environ['OFFLOAD_MODE'] = 'off'
        os.environ['SPEC_ALGO'] = 'off'
        args, _, _ = boot.build_args()
        self.assertFalse([a for a in args if a.startswith('--speculative')])

    def test_decode_graphs_on_leave_prefill_graphs_off(self):
        """SGLang defaults both phases on (decode=full, prefill=breakable), so
        enabling decode capture must still pin prefill off explicitly."""
        os.environ['OFFLOAD_MODE'] = 'off'
        os.environ['CUDA_GRAPH_MAX_BS_DECODE'] = '8'
        args, _, _ = boot.build_args()
        self.assertNotIn('--disable-cuda-graph', args)
        self.assertEqual(args[args.index('--cuda-graph-max-bs-decode') + 1], '8')
        self.assertIn('--disable-prefill-cuda-graph', args)

    def test_prefill_graphs_are_opt_in(self):
        os.environ['OFFLOAD_MODE'] = 'off'
        os.environ['CUDA_GRAPH_MAX_BS_DECODE'] = '8'
        os.environ['CUDA_GRAPH_PREFILL'] = '1'
        args, _, _ = boot.build_args()
        self.assertNotIn('--disable-prefill-cuda-graph', args)
        self.assertNotIn('--disable-cuda-graph', args)

    def test_unset_max_bs_still_pins_prefill_off(self):
        """Empty CUDA_GRAPH_MAX_BS_DECODE leaves SGLang's decode max_bs default
        but must not silently enable prefill capture."""
        os.environ['OFFLOAD_MODE'] = 'off'
        args, _, _ = boot.build_args()
        self.assertNotIn('--cuda-graph-max-bs-decode', args)
        self.assertIn('--disable-prefill-cuda-graph', args)

    def test_disabling_all_graphs_does_not_add_the_prefill_flag(self):
        os.environ['OFFLOAD_MODE'] = 'off'
        os.environ['CUDA_GRAPH_MAX_BS_DECODE'] = '0'
        os.environ['CUDA_GRAPH_PREFILL'] = '1'
        args, _, _ = boot.build_args()
        self.assertIn('--disable-cuda-graph', args)
        self.assertNotIn('--disable-prefill-cuda-graph', args)
        self.assertNotIn('--cuda-graph-max-bs-decode', args)

    def test_moe_overrides(self):
        os.environ['OFFLOAD_MODE'] = 'off'
        os.environ['MOE_RUNNER_BACKEND'] = 'marlin'
        os.environ['MOE_A2A_BACKEND'] = 'deepep'
        os.environ['WEIGHT_LOADER_DROP_CACHE'] = '0'
        os.environ['CUDA_GRAPH_MAX_BS_DECODE'] = '0'
        args, _, _ = boot.build_args()
        self.assertEqual(args[args.index('--moe-runner-backend') + 1], 'marlin')
        self.assertEqual(args[args.index('--moe-a2a-backend') + 1], 'deepep')
        self.assertNotIn('--weight-loader-drop-cache-after-load', args)
        self.assertIn('--disable-cuda-graph', args)
        os.environ['MOE_RUNNER_BACKEND'] = 'auto'
        args, _, _ = boot.build_args()
        self.assertNotIn('--moe-runner-backend', args)

    def test_dp_attention_follows_cookbook_when_dp_gt_1(self):
        os.environ['OFFLOAD_MODE'] = 'off'
        os.environ['DP_SIZE'] = '2'
        args, _, _ = boot.build_args()
        self.assertEqual(args[args.index('--dp') + 1], '2')
        self.assertIn('--enable-dp-attention', args)
        self.assertIn('--mm-enable-dp-encoder', args)

    def test_ssd_points_at_rank_dir(self):
        os.environ['OFFLOAD_MODE'] = 'ssd'
        os.environ['SSD_MOUNT'] = '/tmp/mimo26-ssd-test'
        os.environ['HICACHE_SIZE_GB'] = '4'
        os.environ['NODE_RANK'] = '1'
        args, env, mode = boot.build_args()
        self.assertEqual(mode, 'ssd')
        self.assertIn('--hicache-storage-backend', args)
        self.assertEqual(args[args.index('--hicache-storage-backend') + 1], 'file')
        self.assertEqual(env['SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR'], '/tmp/mimo26-ssd-test/rank1')

    def test_cpu_offload_refused(self):
        os.environ['OFFLOAD_MODE'] = 'cpu'
        with self.assertRaises(RuntimeError):
            boot.build_args()


def _write_safetensors_header(path, tensors):
    """Header-only safetensors file: the estimate never reads tensor data."""
    header = json.dumps({name: {'dtype': dtype, 'shape': shape, 'data_offsets': [0, 0]}
                         for name, (dtype, shape) in tensors.items()}).encode()
    path.write_bytes(len(header).to_bytes(8, 'little') + header)


class ResidentEstimate(unittest.TestCase):
    """Tensor sizes mirror the real checkpoint (one expert, one attention layer)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        model = Path(self.tmp.name)
        _write_safetensors_header(model / 'a.safetensors', {
            'model.layers.1.mlp.experts.0.gate_proj.weight': ('U8', [2048, 2048]),
            'model.layers.1.mlp.experts.0.gate_proj.weight_scale': ('U8', [2048, 128]),
            'model.layers.1.mlp.experts.0.down_proj.weight': ('U8', [4096, 1024]),
            'model.layers.0.self_attn.qkv_proj.weight': ('F8_E4M3', [13568, 4096]),
            'model.embed_tokens.weight': ('BF16', [152576, 4096]),
        })
        _write_safetensors_header(model / 'b.safetensors', {
            'model.visual.blocks.0.attn.qkv.weight': ('BF16', [3072, 1024]),
            'audio_encoder.layers.0.mlp.up_proj.weight': ('BF16', [4096, 1024]),
        })
        _write_safetensors_header(model / 'model_mtp.safetensors', {
            'model.mtp.layers.0.mlp.gate_proj.weight': ('F8_E4M3', [16384, 4096]),
        })
        (model / 'model.safetensors.index.json').write_text(json.dumps({
            'metadata': {'save_format': 'mxfp4', 'tp_size': 4},
            'weight_map': {
                'model.layers.1.mlp.experts.0.gate_proj.weight': 'a.safetensors',
                'model.layers.1.mlp.experts.0.gate_proj.weight_scale': 'a.safetensors',
                'model.layers.1.mlp.experts.0.down_proj.weight': 'a.safetensors',
                'model.layers.0.self_attn.qkv_proj.weight': 'a.safetensors',
                'model.embed_tokens.weight': 'a.safetensors',
                'model.visual.blocks.0.attn.qkv.weight': 'b.safetensors',
                'audio_encoder.layers.0.mlp.up_proj.weight': 'b.safetensors',
            }}))
        self.model = model

    def tearDown(self):
        self.tmp.cleanup()

    def test_split_by_kind(self):
        est = boot.resident_estimate(self.model, tp=2)
        gib = 2 ** 30
        experts = 2048 * 2048 + 2048 * 128 + 4096 * 1024
        sharded = 13568 * 4096 + 152576 * 4096 * 2 + 16384 * 4096
        replicated = 3072 * 1024 * 2 + 4096 * 1024 * 2
        self.assertEqual(est['files'], 3)
        self.assertAlmostEqual(est['experts_packed_gib'], experts / gib)
        self.assertAlmostEqual(est['experts_per_rank_gib'], experts / 2 / gib)
        self.assertAlmostEqual(est['sharded_per_rank_gib'], sharded / 2 / gib)
        self.assertAlmostEqual(est['replicated_gib'], replicated / gib)
        self.assertAlmostEqual(est['per_rank_gib'], (experts / 2 + sharded / 2 + replicated) / gib)
        self.assertAlmostEqual(est['per_rank_if_fp8_experts_gib'], est['per_rank_gib'] + experts / 2 / gib)


class EvaluateFit(unittest.TestCase):
    # The real TP=2 numbers: 75 GiB of experts per rank, 5 sharded, 2 encoders.
    EST = {
        'experts_per_rank_gib': 75.0, 'sharded_per_rank_gib': 5.0,
        'replicated_gib': 2.0, 'per_rank_gib': 82.0,
        'per_rank_if_fp8_experts_gib': 157.0,
    }

    def test_packed_experts_fit_a_spark(self):
        # head on 2026-09-22: MemAvailable 117 GiB before launch
        problems, summary = boot.evaluate_fit(self.EST, True, 121.7, 117.0, 0.85)
        self.assertEqual(problems, [])
        self.assertIn('packed MXFP4', summary)
        # (117 - 5) x 0.15 + 0.1 = 16.9 reserve; 112 - 82 - 16.9 = 13.1
        self.assertIn('17 GiB kept outside the pool', summary)
        self.assertIn('~13 GiB for the draft and KV', summary)

    def test_head_at_080_is_too_tight(self):
        # (112 - 5) x 0.20 + 0.1 = 21.5 reserve; 107 - 82 - 21.5 = 3.5 < 4
        problems, _ = boot.evaluate_fit(self.EST, True, 121.7, 112.0, 0.80)
        self.assertEqual(len(problems), 1)
        self.assertIn('Raise MEM_FRACTION_STATIC', problems[0])

    def test_fp8_expansion_is_refused(self):
        problems, summary = boot.evaluate_fit(self.EST, False, 121.7, 111.0, 0.80)
        self.assertTrue(problems)
        self.assertIn('FP8', problems[0])
        self.assertIn(boot.MIN_BASE_IMAGE, problems[0])
        self.assertIn('EXPANDED TO FP8', summary)

    def test_kv_floor(self):
        problems, _ = boot.evaluate_fit(self.EST, True, 121.7, 117.0, 0.70)
        self.assertEqual(len(problems), 1)
        self.assertIn('for the draft and KV', problems[0])

    def test_memory_already_taken(self):
        problems, _ = boot.evaluate_fit(self.EST, True, 121.7, 60.0, 0.85)
        self.assertEqual(len(problems), 2)
        self.assertTrue(any('something else is holding' in p for p in problems))


if __name__ == '__main__':
    unittest.main()
