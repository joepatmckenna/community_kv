"""Fused causal attention with top-kappa key extraction (bitonic version).

This kernel mirrors the structure of the PyTorch reference in `reference.py`:
- Online softmax for attention output
- Per-K-tile sort to produce tile-local top-kappa
- Merge tile-local top-kappa with running top-kappa via sort over 2*kappa
- Skip queries with position < kappa - 1 + num_sink_tok_to_exclude (no valid top-kappa)
- Apply positional scaling at end
- Convert raw-equivalent scores to softmax weights at end via final m, l

We use Triton's native `tl.sort` for both sorts. Since `tl.sort` returns values
only (not indices), we pack (score, index) into a single int64 via:
    packed = (score_as_uint32 << 32) | index_as_uint32
Sorting int64 descending sorts primarily by score with index as tiebreaker.
This works because softmax-style scores are always non-negative; for positive
floats, the IEEE 754 bit pattern is monotonic with float value.

UNTESTED ON GPU. The reference (`reference.py`) is verified on CPU and is the
ground truth. Run `tests/test_kernel_correctness.py` on a GPU to validate.
"""

from __future__ import annotations
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _pack_score_index(score, index):
    """Pack fp32 score and int32 index into an int64.

    Score is bitcast to int32 (preserves ordering for non-negative floats),
    promoted to int64, shifted into the high 32 bits. Index occupies the
    low 32 bits. Sorting int64 descending sorts primarily by score.
    """
    score_bits = score.to(tl.int32, bitcast=True)
    score_hi = score_bits.to(tl.int64) << 32
    index_lo = index.to(tl.int64) & 0xFFFFFFFF
    return score_hi | index_lo


@triton.jit
def _unpack_score_index(packed):
    """Unpack int64 into (score_fp32, index_int32)."""
    score_bits = (packed >> 32).to(tl.int32)
    score = score_bits.to(tl.float32, bitcast=True)
    index = (packed & 0xFFFFFFFF).to(tl.int32)
    return score, index


@triton.jit
def _attention_with_topk_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    Out_ptr,
    TopkIdx_ptr, TopkScore_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_tib, stride_tih, stride_tis, stride_tik,
    stride_tsb, stride_tsh, stride_tss, stride_tsk,
    B, H, H_kv, S, D,
    init_q_start,        # = kappa - 1 + num_sink_tok_to_exclude
    num_sink_tok_to_exclude,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    KAPPA: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """One program instance per (batch * head, query-tile)."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    h_kv = h * H_kv // H

    q_start = pid_m * BLOCK_Q
    q_offsets = q_start + tl.arange(0, BLOCK_Q)
    q_in_seq_mask = q_offsets < S
    q_topk_eligible = q_offsets >= init_q_start

    q_base = Q_ptr + b * stride_qb + h * stride_qh
    k_base = K_ptr + b * stride_kb + h_kv * stride_kh
    v_base = V_ptr + b * stride_vb + h_kv * stride_vh
    o_base = Out_ptr + b * stride_ob + h * stride_oh

    d_offsets = tl.arange(0, HEAD_DIM)
    q_ptrs = q_base + q_offsets[:, None] * stride_qs + d_offsets[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_in_seq_mask[:, None], other=0.0)

    # Online softmax accumulators
    m_i = tl.full((BLOCK_Q,), value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, HEAD_DIM), dtype=tl.float32)

    # Top-kappa running buffer, shape (BLOCK_Q, KAPPA).
    # Stored as exp(raw - m_i) — non-negative, rescaled when m_i updates.
    topk_score = tl.zeros((BLOCK_Q, KAPPA), dtype=tl.float32)
    topk_idx = tl.full((BLOCK_Q, KAPPA), value=-1, dtype=tl.int32)

    # Causal: only iterate K-tiles up to the max query position in this tile.
    max_q = q_start + BLOCK_Q - 1
    n_blocks_causal = tl.cdiv(tl.minimum(max_q + 1, S), BLOCK_K)

    kappa_range = tl.arange(0, KAPPA)

    for kn in range(0, n_blocks_causal):
        k_start = kn * BLOCK_K
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_in_seq_mask = k_offsets < S

        k_ptrs = k_base + k_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd
        v_ptrs = v_base + k_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd
        k_tile = tl.load(k_ptrs, mask=k_in_seq_mask[:, None], other=0.0)
        v_tile = tl.load(v_ptrs, mask=k_in_seq_mask[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k_tile)) * scale

        # Mask invalid score positions (causal, padded keys, padded queries)
        causal_mask = q_offsets[:, None] < k_offsets[None, :]
        s = tl.where(causal_mask, -float("inf"), s)
        s = tl.where(k_in_seq_mask[None, :], s, -float("inf"))
        s = tl.where(q_in_seq_mask[:, None], s, -float("inf"))

        # Online softmax update — uses ALL keys (including sinks) for the
        # standard attention output.
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p_attn = tl.exp(s - m_new[:, None])

        acc = acc * alpha[:, None] + tl.dot(p_attn.to(v_tile.dtype), v_tile)
        l_i = l_i * alpha + tl.sum(p_attn, axis=1)

        # ---- Top-kappa pool: same scores but with sinks masked out ----
        is_sink = k_offsets < num_sink_tok_to_exclude
        s_for_topk = tl.where(is_sink[None, :], -float("inf"), s)
        p_topk = tl.exp(s_for_topk - m_new[:, None])
        # NaN check: if both s and m_new are -inf, exp gives nan. Replace with 0.
        p_topk = tl.where(p_topk == p_topk, p_topk, 0.0)

        # Rescale running buffer to new max scale
        topk_score = topk_score * alpha[:, None]

        # Pack tile candidates and sort to get tile-local top-KAPPA
        k_idx_2d = (k_offsets[None, :] + tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.int32))
        tile_packed = _pack_score_index(p_topk, k_idx_2d)
        tile_sorted = tl.sort(tile_packed, dim=1, descending=True)

        # Slice first KAPPA: shape (BLOCK_Q, KAPPA)
        # Use gather to extract — Triton's slicing on dynamic ranges of a
        # constexpr-shaped tensor is awkward; gather is reliable.
        gather_idx = kappa_range[None, :] + tl.zeros((BLOCK_Q, KAPPA), dtype=tl.int32)
        tile_topk_packed = tl.gather(tile_sorted, gather_idx, axis=-1)

        # Pack running buffer
        running_packed = _pack_score_index(topk_score, topk_idx)

        # Concatenate (BLOCK_Q, KAPPA) + (BLOCK_Q, KAPPA) -> (BLOCK_Q, 2*KAPPA)
        # tl.join interleaves along a new last axis -> (BLOCK_Q, KAPPA, 2)
        # then reshape to (BLOCK_Q, 2*KAPPA). Order doesn't matter since we sort next.
        merged_packed = tl.reshape(tl.join(running_packed, tile_topk_packed), (BLOCK_Q, 2 * KAPPA))

        # Sort merged buffer descending and take first KAPPA
        merged_sorted = tl.sort(merged_packed, dim=1, descending=True)
        new_topk_packed = tl.gather(merged_sorted, gather_idx, axis=-1)

        topk_score, topk_idx = _unpack_score_index(new_topk_packed)

        m_i = m_new

    # Finalize attention output
    acc = acc / l_i[:, None]

    # Finalize top-kappa scores: divide by l_i
    topk_score_final = topk_score / l_i[:, None]

    invalid_slot = topk_idx < 0
    topk_score_final = tl.where(invalid_slot, 0.0, topk_score_final)

    # Store output (all queries)
    o_ptrs = o_base + q_offsets[:, None] * stride_os + d_offsets[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=q_in_seq_mask[:, None])

    # Store top-kappa: only for queries with position >= init_q_start.
    out_q_offsets = q_offsets - init_q_start
    write_mask = q_topk_eligible & q_in_seq_mask

    ti_base = TopkIdx_ptr + b * stride_tib + h * stride_tih
    ts_base = TopkScore_ptr + b * stride_tsb + h * stride_tsh
    ti_ptrs = ti_base + out_q_offsets[:, None] * stride_tis + kappa_range[None, :] * stride_tik
    ts_ptrs = ts_base + out_q_offsets[:, None] * stride_tss + kappa_range[None, :] * stride_tsk

    tl.store(ti_ptrs, topk_idx, mask=write_mask[:, None])
    tl.store(ts_ptrs, topk_score_final.to(TopkScore_ptr.dtype.element_ty), mask=write_mask[:, None])


def attention_with_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kappa: int,
    scale: float | None = None,
    num_sink_tok_to_exclude: int = 0,
    block_q: int = 128,
    block_k: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Causal attention + fused top-kappa extraction.

    - Output covers ALL query positions [0, S).
    - Top-kappa indices/scores cover query positions [init_q_start, S) where
      init_q_start = kappa - 1 + num_sink_tok_to_exclude.
    - Top-kappa pool excludes positions [0, num_sink_tok_to_exclude).
    - Top-kappa scores are post-softmax weights.

    Args:
        q: (B, H, S, D)
        k: (B, H_kv, S, D)
        v: (B, H_kv, S, D)
        kappa: number of top keys per query, must be a power of 2
        scale: softmax scale; defaults to 1/sqrt(D)
        num_sink_tok_to_exclude: leading positions excluded from top-kappa
        block_q: query tile size, must be a power of 2
        block_k: key tile size, must be a power of 2 and >= kappa

    Returns:
        out: (B, H, S, D) attention output
        topk_indices: (B, H, S - init_q_start, kappa) int32
        topk_scores: (B, H, S - init_q_start, kappa) post-softmax weights
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == k.dtype == v.dtype

    B, H, S, D = q.shape
    _, H_kv, S_kv, _ = k.shape
    assert S == S_kv
    assert H % H_kv == 0
    assert (kappa & (kappa - 1)) == 0
    assert (block_k & (block_k - 1)) == 0
    assert block_k >= kappa
    assert (D & (D - 1)) == 0

    init_q_start = kappa - 1 + num_sink_tok_to_exclude
    assert init_q_start < S, f"sequence too short for kappa={kappa}, sink={num_sink_tok_to_exclude}"

    out_topk_len = S - init_q_start

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(q)
    topk_indices = torch.empty((B, H, out_topk_len, kappa), dtype=torch.int32, device=q.device)
    topk_scores = torch.empty((B, H, out_topk_len, kappa), dtype=q.dtype, device=q.device)

    grid = (triton.cdiv(S, block_q), B * H)

    with torch.cuda.device(q.device):
        _attention_with_topk_fwd_kernel[grid](
            q, k, v,
            out,
            topk_indices, topk_scores,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            topk_indices.stride(0), topk_indices.stride(1), topk_indices.stride(2), topk_indices.stride(3),
            topk_scores.stride(0), topk_scores.stride(1), topk_scores.stride(2), topk_scores.stride(3),
            B, H, H_kv, S, D,
            init_q_start,
            num_sink_tok_to_exclude,
            scale,
            BLOCK_Q=block_q,
            BLOCK_K=block_k,
            KAPPA=kappa,
            HEAD_DIM=D,
        )

    return out, topk_indices, topk_scores
