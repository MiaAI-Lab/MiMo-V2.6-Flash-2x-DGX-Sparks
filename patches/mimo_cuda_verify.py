"""Unintegrated CUDA experiment using the pinned generic Triton verify kernels.

No backend guards are patched. Only target VERIFY, a nonempty full-attention
prefix, TP2 MiMo geometry and a proven uniform topk1 causal chain are accepted.
SWA, sinks, tree branches and draft/MTP must retain the existing backend.
"""
from dataclasses import dataclass, asdict
from numbers import Real

import torch
import triton

from sglang.kernels.ops.attention.verify_mla import VerifyMLA
from sglang.kernels.ops.attention.verify_splitkv import VerifySplitKV


@dataclass(frozen=True)
class LayoutReceipt:
    batch: int
    extend_length: int
    # The caller establishes this from host scheduling metadata. Do not read
    # GPU indptr values or assume that matching tensor shapes prove the mask.
    uniform_pure_causal_topk1_target_verify: bool
    all_prefixes_nonempty: bool
    # Kernel-facing flat row IDs for the actual per-layer 3-D buffers. In the
    # unified pool these can include a page multiplier; they are not raw
    # virtual IDs or page-table entries.
    physical_token_indices: bool
    page_size: int = 1


class _FixedGrouped(VerifyMLA):
    def __init__(self, batch, extend_length, splits, block_h, block_n, warps, device):
        self.fixed_splits = splits
        super().__init__(batch, 32, 192, 128, extend_length, device=device,
                         block_h=block_h, block_n=block_n, num_warps=warps,
                         kv_group_num=16)

    def _num_splits(self, bs):
        return self.fixed_splits

    def _alloc(self, max_bs):
        # The pinned parent reserves MAX_N_SPLITS=32. This experiment has a
        # fixed captured tactic, so reserve only its actual split count.
        self.max_bs = max_bs
        self.att_out = torch.empty(
            (max_bs, self.h_q, self.fixed_splits, self.l_pad, self.v_head_dim),
            dtype=torch.float32, device=self.device)
        self.att_lse = torch.empty(
            (max_bs, self.h_q, self.fixed_splits, self.l_pad),
            dtype=torch.float32, device=self.device)


class _CudaPerHead(VerifySplitKV):
    def __init__(self, batch, extend_length, splits, block_n, warps, device):
        super().__init__(batch, 32, 2, 192, 128, extend_length, device=device,
                         n_splits=splits, block_n=block_n, num_warps=warps)
        # The pinned AMD path allows tiny row tiles; the CUDA dot experiment
        # needs at least 16 rows. Fresh-draft rows remain masked to real length.
        self.l_pad = max(16, triton.next_power_of_2(extend_length))
        self._alloc(batch)


class FullVerifyExperiment:
    def __init__(self, layout, variant='grouped', splits=8, block_h=4,
                 block_n=64, warps=4, device='cuda'):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Construct and warm up before CUDA graph capture')
        if torch.cuda.get_device_capability() != (12, 1):
            raise ValueError('SM121 only')
        if not all([layout.uniform_pure_causal_topk1_target_verify,
                    layout.all_prefixes_nonempty, layout.physical_token_indices]):
            raise ValueError('A host layout receipt is required')
        # For 3-D buffers _extract_kv_strides gives page_stride=P*slot_stride,
        # tok_stride=slot_stride. Hence (id//P)*page_stride+(id%P)*tok_stride
        # equals id*slot_stride for any positive P, including page_size=256.
        if layout.page_size < 1 or layout.batch < 1 or layout.extend_length not in [3, 4, 5, 6, 8]:
            raise ValueError('Unsupported experimental layout')
        if splits not in [4, 8, 16, 32] or block_h not in [4, 8, 16] or block_n not in [32, 64, 128]:
            raise ValueError('Use explicit candidate tactics')
        if warps not in [4, 8]:
            raise ValueError('Use 4 or 8 warps')
        self.layout = layout
        self.tactic = dict(variant=variant, splits=splits, block_h=block_h,
                           block_n=block_n, warps=warps)
        if variant == 'grouped':
            self.kernel = _FixedGrouped(layout.batch, layout.extend_length, splits,
                                        block_h, block_n, warps, device)
        elif variant == 'per_head':
            self.kernel = _CudaPerHead(layout.batch, layout.extend_length, splits,
                                       block_n, warps, device)
        else:
            raise ValueError('Unknown variant')

    def receipt(self):
        return dict(layout=asdict(self.layout), tactic=self.tactic,
                    q_heads=32, kv_heads=2, qk_dim=192, v_dim=128,
                    row_pad=self.kernel.l_pad,
                    workspace_bytes=sum(t.numel()*t.element_size() for t in
                                        [self.kernel.att_out, self.kernel.att_lse]),
                    production_integrated=False)

    def _reject(self, code):
        # Debug aid: numbered guard in run() that refused the call.
        self.last_reject = code
        return False

    def run(self, q, k, v, out, k_buffer, v_buffer, qo_indptr, kv_indptr,
            kv_indices, k_scale, v_scale, sm_scale, *, sliding_window_size=-1,
            sinks=None, score_mod=None, logit_cap=0, target_verify=True, topk=1):
        # No tensor-value inspection and no buffer growth/JIT initialization in
        # this path. False means the eventual caller must use the old backend.
        if not target_verify or topk != 1 or sliding_window_size != -1:
            return self._reject(1)
        if sinks is not None or score_mod is not None or logit_cap != 0:
            return self._reject(2)
        b, l = self.layout.batch, self.layout.extend_length
        if q.shape != (b*l, 32, 192) or k.shape != (b*l, 2, 192) or v.shape != (b*l, 2, 128):
            return self._reject(3)
        if out.shape != (b*l, 32, 128) or qo_indptr.shape != (b+1,) or kv_indptr.shape != (b+1,):
            return self._reject(4)
        if k_buffer.ndim != 3 or v_buffer.ndim != 3:
            return self._reject(5)
        tensors = [q, k, v, out, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices]
        if any(not t.is_cuda or t.device != q.device for t in tensors):
            return self._reject(6)
        # SGLang graph buffers keep qo_indptr as int64; the kernels only
        # tl.load the offsets, so both widths are valid.
        if any(t.dtype not in [torch.int32, torch.int64] for t in [qo_indptr, kv_indptr]):
            return self._reject(7)
        if kv_indices.ndim != 1 or kv_indices.numel() == 0 or kv_indices.dtype not in [torch.int32, torch.int64]:
            return self._reject(8)
        if any(not t.is_contiguous() for t in [qo_indptr, kv_indptr, kv_indices]):
            return self._reject(9)
        if k_buffer.shape[0] != v_buffer.shape[0]:
            return self._reject(10)
        if k_buffer.shape[1:] != (2, 192) or v_buffer.shape[1:] != (2, 128):
            return self._reject(11)
        if any(t.dtype != torch.bfloat16 for t in [q, k, v, out]):
            return self._reject(12)
        if any(t.dtype != torch.float8_e4m3fn for t in [k_buffer, v_buffer]):
            return self._reject(13)
        if any(t.stride(-1) != 1 for t in [q, k, v, out, k_buffer, v_buffer]):
            return self._reject(14)
        if any(not isinstance(scale, Real) for scale in [k_scale, v_scale, sm_scale]):
            return self._reject(15)  # Never float(cuda_scalar) inside capture.
        self.kernel(q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr,
                    kv_indices, sm_scale, o_out=out, k_scale=k_scale, v_scale=v_scale)
        return True
