"""Check that only MiMo's Triton backend opts into key-zero sink semantics."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'mimo_vision_patch', Path(__file__).resolve().parents[1] / 'patches/sitecustomize.py')
hooks = sys.meta_path[:]
fix = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(fix)
finally:
    sys.meta_path[:] = hooks


class VisionBinding(unittest.TestCase):
    def setUp(self):
        class TritonBackend:
            pass

        class Block:
            def __init__(self, backend):
                self.attn = types.SimpleNamespace(qkv_backend=backend)

        self.backend = TritonBackend
        self.module = types.SimpleNamespace(MiMoVisionBlock=Block)
        self.vision = types.ModuleType('sglang.srt.layers.attention.vision')
        self.vision.VisionTritonAttention = TritonBackend
        self.kernel = types.ModuleType('sglang.kernels.ops.attention.prefill_attention')
        self.kernel.context_attention_fwd = lambda q=None, key0_sink=None: None
        self.modules = patch.dict(sys.modules, {
            self.vision.__name__: self.vision, self.kernel.__name__: self.kernel})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_only_mimo_triton_instances_are_marked(self):
        fix.install_mimo_vision(self.module)
        block = self.module.MiMoVisionBlock(self.backend())
        self.assertTrue(block.attn.qkv_backend.mimo_vision_key0_sink)
        other = types.SimpleNamespace()
        self.module.MiMoVisionBlock(other)
        self.assertFalse(hasattr(other, 'mimo_vision_key0_sink'))
        self.assertFalse(hasattr(self.backend(), 'mimo_vision_key0_sink'))

    def test_stale_image_fails_with_rebuild_instruction(self):
        self.kernel.context_attention_fwd = lambda q=None: None
        fix.install_mimo_vision(self.module)
        with self.assertRaisesRegex(RuntimeError, 'rebuild with ./start.sh build'):
            self.module.MiMoVisionBlock(self.backend())

    def test_reinstall_does_not_wrap_constructor_again(self):
        fix.install_mimo_vision(self.module)
        constructor = self.module.MiMoVisionBlock.__init__
        fix.install_mimo_vision(self.module)
        self.assertIs(constructor, self.module.MiMoVisionBlock.__init__)


if __name__ == '__main__':
    unittest.main()
