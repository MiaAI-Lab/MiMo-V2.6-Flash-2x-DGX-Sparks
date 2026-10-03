"""Grouped-head prefill attention for MiMo full-attention layers on GB10 (SM121).

The pinned extend kernel runs one program per (query head, 32-token tile), so
each FP8 prefix K/V tile is fetched once per query head: 16x for the TP2 GQA
group (32 query heads, 2 KV heads). Here one program owns G query heads that
share a KV head and BLOCK_T tokens; its query tile has BLOCK_T*G rows
(token-major, heads interleaved), so every K/V tile feeds G heads and the MMA
M dimension grows. Casts mirror the pinned CUDA path: Q->FP8 against the FP8
prefix cache, P->FP8 for the prefix P.V, BF16 for the fresh causal extend.

Scope: full causal attention only (no sinks, sliding window, custom mask,
logit cap, score mod), QK 192 / V 128, BF16 Q/K/V, FP8 e4m3 cache addressed
by flat token rows (3-D buffers), uniform heads-per-KV group.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _mimo_prefill_grouped_kernel(
    Q, K_ext, V_ext, O, K_buf, V_buf,
    qo_indptr, kv_indptr, kv_indices,
    sm_scale_log2, k_scale, v_scale,
    stride_qbs, stride_qh, stride_kbs, stride_kh, stride_vbs, stride_vh,
    stride_obs, stride_oh, stride_buf_kbs, stride_buf_kh, stride_buf_vbs, stride_buf_vh,
    HEADS_PER_KV: tl.constexpr, G: tl.constexpr, BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_N_EXT: tl.constexpr,
    D: tl.constexpr, DV: tl.constexpr, NUM_STAGES: tl.constexpr,
    QK_BF16: tl.constexpr, PV_BF16: tl.constexpr, LAZY: tl.constexpr, OFF32: tl.constexpr,
    ABLATE: tl.constexpr = 0,
):
    seq = tl.program_id(0)
    head_group = tl.program_id(1)
    block_t = tl.program_id(2)
    head0 = head_group * G
    kv_head = head0 // HEADS_PER_KV

    q_start = tl.load(qo_indptr + seq)
    ext_len = tl.load(qo_indptr + seq + 1) - q_start
    if block_t * BLOCK_T >= ext_len:
        return
    kv_start = tl.load(kv_indptr + seq)
    prefix_len = tl.load(kv_indptr + seq + 1) - kv_start

    ROWS: tl.constexpr = BLOCK_T * G
    rows = tl.arange(0, ROWS)
    tok = block_t * BLOCK_T + rows // G          # extend-relative token per row
    head = head0 + rows % G
    mask_row = tok < ext_len
    # Head dim 192 = 128 + 64 (power-of-two pieces, as the pinned kernel's DPE split).
    offs_d = tl.arange(0, 128)
    offs_dpe = 128 + tl.arange(0, 64)
    offs_dv = tl.arange(0, DV)

    q_rows = Q + (q_start + tok)[:, None] * stride_qbs + head[:, None] * stride_qh
    q = tl.load(q_rows + offs_d[None, :], mask=mask_row[:, None], other=0.0)
    qpe = tl.load(q_rows + offs_dpe[None, :], mask=mask_row[:, None], other=0.0)
    q8 = q.to(tl.float8e4nv)
    qpe8 = qpe.to(tl.float8e4nv)

    acc = tl.zeros([ROWS, DV], dtype=tl.float32)
    deno = tl.zeros([ROWS], dtype=tl.float32)
    e_max = tl.zeros([ROWS], dtype=tl.float32) - float("inf")
    prefix_scale = sm_scale_log2 * k_scale

    # Stage 1: FP8 prefix from the paged cache (no mask beyond length).
    offs_n = tl.arange(0, BLOCK_N)
    for start_n in tl.range(0, prefix_len, BLOCK_N, num_stages=NUM_STAGES):
        mask_n = (start_n + offs_n) < prefix_len
        loc = tl.load(kv_indices + kv_start + start_n + offs_n, mask=mask_n, other=0)
        if OFF32:
            # Row offsets fit int32 for caches below 2^31 bytes per buffer.
            loc = loc.to(tl.int32)
        k_cols = K_buf + loc[None, :] * stride_buf_kbs + kv_head * stride_buf_kh
        k = tl.load(k_cols + offs_d[:, None], mask=mask_n[None, :], other=0.0)
        kpe = tl.load(k_cols + offs_dpe[:, None], mask=mask_n[None, :], other=0.0)
        if ABLATE == 3:  # diagnostic: skip the QK product
            qk = tl.zeros([ROWS, BLOCK_N], dtype=tl.float32) + tl.sum(k.to(tl.float32), 0)[None, :] * 0.0 + tl.sum(kpe.to(tl.float32), 0)[None, :] * 0.0
        elif QK_BF16:
            qk = (tl.dot(q, k.to(tl.bfloat16)) + tl.dot(qpe, kpe.to(tl.bfloat16))) * prefix_scale
        else:
            qk = (tl.dot(q8, k) + tl.dot(qpe8, kpe)) * prefix_scale
        qk = tl.where(mask_n[None, :], qk, float("-inf"))
        tile_max = tl.max(qk, 1)
        if LAZY:
            # FA3-style lazy rescale: keep the running max unless some row grew by
            # more than 2^8; p stays <= 256 in fp32, so nothing overflows.
            grow = tl.max(tile_max - e_max, 0) > 8.0
            if grow:
                n_max = tl.maximum(e_max, tile_max)
                re_scale = tl.exp2(e_max - n_max)
                acc = acc * re_scale[:, None]
                deno = deno * re_scale
                e_max = n_max
            if ABLATE == 1:  # diagnostic: skip exp2
                p = qk - e_max[:, None]
            else:
                p = tl.exp2(qk - e_max[:, None])
            deno = deno + tl.sum(p, 1)
        else:
            n_max = tl.maximum(e_max, tile_max)
            re_scale = tl.exp2(e_max - n_max)
            p = tl.exp2(qk - n_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)
            acc = acc * re_scale[:, None]
            e_max = n_max
        v = tl.load(V_buf + loc[:, None] * stride_buf_vbs + kv_head * stride_buf_vh + offs_dv[None, :],
                    mask=mask_n[:, None], other=0.0)
        if ABLATE == 2:  # diagnostic: skip the PV product
            acc += tl.sum(p, 1)[:, None] * 0.0 + tl.sum(v.to(tl.float32), 0)[None, :] * 0.0
        elif PV_BF16:
            acc += tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16)) * v_scale
        else:
            acc += tl.dot(p.to(v.dtype), v) * v_scale

    # Stage 2: causal BF16 extend (keys up to this tile's last token).
    offs_e = tl.arange(0, BLOCK_N_EXT)
    ext_end = tl.minimum(ext_len, (block_t + 1) * BLOCK_T)
    for start_n in range(0, ext_end, BLOCK_N_EXT):
        key = start_n + offs_e
        mask_n = key < ext_end
        k_cols = K_ext + (q_start + key)[None, :] * stride_kbs + kv_head * stride_kh
        k = tl.load(k_cols + offs_d[:, None], mask=mask_n[None, :], other=0.0)
        kpe = tl.load(k_cols + offs_dpe[:, None], mask=mask_n[None, :], other=0.0)
        qk = (tl.dot(q, k, out_dtype=tl.float32) + tl.dot(qpe, kpe, out_dtype=tl.float32)) * sm_scale_log2
        causal = (tok[:, None] >= key[None, :]) & mask_n[None, :] & mask_row[:, None]
        qk = tl.where(causal, qk, float("-inf"))
        row_max = tl.max(qk, 1)
        n_max = tl.maximum(e_max, tl.where(row_max == float("-inf"), -1e20, row_max))
        re_scale = tl.exp2(e_max - n_max)
        p = tl.exp2(qk - n_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)
        v = tl.load(V_ext + (q_start + key)[:, None] * stride_vbs + kv_head * stride_vh + offs_dv[None, :],
                    mask=mask_n[:, None], other=0.0)
        acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v)
        e_max = n_max

    out = acc / tl.where(deno == 0.0, 1.0, deno)[:, None]
    tl.store(O + (q_start + tok)[:, None] * stride_obs + head[:, None] * stride_oh + offs_dv[None, :],
             out.to(O.dtype.element_ty), mask=mask_row[:, None])


DEFAULT_TACTIC = dict(G=4, BLOCK_T=16, BLOCK_N=64, BLOCK_N_EXT=64, num_warps=8, num_stages=3,
                      QK_BF16=True, PV_BF16=True, LAZY=True, OFF32=True)


def mimo_prefill_grouped(q, k_ext, v_ext, o, k_buf, v_buf, qo_indptr, kv_indptr, kv_indices,
                         max_len_extend, sm_scale, k_scale=1.0, v_scale=1.0, tactic=None):
    """q (T, Hq, 192) bf16, k_ext/v_ext (T, Hkv, 192/128) bf16, o (T, Hq, 128),
    k_buf/v_buf (N, Hkv, 192/128) float8_e4m3fn, indptr int32/int64, kv_indices flat token rows."""
    t = dict(DEFAULT_TACTIC, **(tactic or {}))
    hq, hkv = q.shape[1], k_ext.shape[1]
    heads_per_kv = hq // hkv
    assert hq % hkv == 0 and heads_per_kv % t['G'] == 0
    assert q.shape[2] == 192 and v_ext.shape[2] == 128 and k_buf.dtype == torch.float8_e4m3fn
    if t['OFF32']:
        assert k_buf.numel() < 2**31 and v_buf.numel() < 2**31
    log2e = 1.4426950408889634
    grid = (qo_indptr.shape[0] - 1, hq // t['G'], triton.cdiv(max_len_extend, t['BLOCK_T']))
    _mimo_prefill_grouped_kernel[grid](
        q, k_ext, v_ext, o, k_buf, v_buf, qo_indptr, kv_indptr, kv_indices,
        float(sm_scale) * log2e, float(k_scale), float(v_scale),
        q.stride(0), q.stride(1), k_ext.stride(0), k_ext.stride(1), v_ext.stride(0), v_ext.stride(1),
        o.stride(0), o.stride(1), k_buf.stride(0), k_buf.stride(1), v_buf.stride(0), v_buf.stride(1),
        HEADS_PER_KV=heads_per_kv, G=t['G'], BLOCK_T=t['BLOCK_T'], BLOCK_N=t['BLOCK_N'],
        BLOCK_N_EXT=t['BLOCK_N_EXT'], D=192, DV=128, NUM_STAGES=t['num_stages'],
        QK_BF16=t['QK_BF16'], PV_BF16=t['PV_BF16'], LAZY=t['LAZY'], OFF32=t['OFF32'], ABLATE=t.get('ABLATE', 0),
        # Pipeline only the prefix sweep (tl.range above); the causal tail stays single-stage.
        num_warps=t['num_warps'], num_stages=1,
    )
    return o
