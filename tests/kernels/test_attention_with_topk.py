"""Correctness tests for fused attention + top-kappa Triton kernel.

Tests check that the kernel matches the PyTorch reference on:
- Attention output (per-element close)
- Top-kappa indices and scores (set equality per query, ignoring order)

Run on a CUDA host with: pytest tests/kernels/test_attention_with_topk.py -v
"""

from __future__ import annotations
import pytest
import torch

from community_kv.kernels.attention_with_topk import attention_with_topk


def attention_with_topk_reference(
    q: torch.Tensor,            # (B, H, S, D)
    k: torch.Tensor,            # (B, H_kv, S, D)
    v: torch.Tensor,            # (B, H_kv, S, D)
    kappa: int,
    scale: float | None = None,
    num_sink_tok_to_exclude: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Causal attention + top-kappa key extraction (PyTorch reference).

    Semantics:
    - Output covers all query positions [0, S).
    - Top-kappa indices/scores cover positions [init_q_start, S) where
      init_q_start = kappa - 1 + num_sink_tok_to_exclude.
    - Top-kappa pool excludes [0, num_sink_tok_to_exclude).
    - Scores are post-softmax weights.

    Inefficient by design (materializes full attention matrix).

    Returns:
        output: (B, H, S, D)
        topk_indices: (B, H, S - init_q_start, kappa), int32 — key positions
        topk_scores: (B, H, S - init_q_start, kappa) — post-softmax weights
    """
    B, H_q, S, D = q.shape
    _, H_kv, _, _ = k.shape
    assert S == k.shape[2]
    assert H_q % H_kv == 0

    init_q_start = kappa - 1 + num_sink_tok_to_exclude
    assert init_q_start < S, f"sequence too short for kappa={kappa}, sink={num_sink_tok_to_exclude}"

    if scale is None:
        scale = torch.rsqrt(torch.tensor(float(D))).item()

    # GQA expansion
    if H_kv != H_q:
        repeat = H_q // H_kv
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    causal_mask = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)  # (B, H, S, S)
    output = torch.matmul(weights, v.float()).to(q.dtype)

    # Position-based eligibility mask: valid key positions are
    # [num_sink_tok_to_exclude, query_pos] (inclusive on both ends).
    q_positions = torch.arange(init_q_start, S, device=q.device).unsqueeze(-1)  # (S_elig, 1)
    k_positions = torch.arange(S, device=q.device).unsqueeze(0)                 # (1, S)
    valid_mask = (k_positions >= num_sink_tok_to_exclude) & (k_positions <= q_positions)

    eligible_weights = weights[:, :, init_q_start:, :].masked_fill(~valid_mask, -float("inf"))

    topk_scores, topk_indices = torch.topk(eligible_weights, k=kappa, dim=-1)

    return output, topk_indices.to(torch.int32), topk_scores.to(q.dtype)


def _make_qkv(B, H_q, H_kv, S, D, dtype=torch.float16, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, H_q, S, D, dtype=dtype, device=device, generator=g) * 0.1
    k = torch.randn(B, H_kv, S, D, dtype=dtype, device=device, generator=g) * 0.1
    v = torch.randn(B, H_kv, S, D, dtype=dtype, device=device, generator=g) * 0.1
    return q, k, v


def _topk_sets_match(idx_a, score_a, idx_b, score_b, score_atol=5e-3):
    """Per-query set equality, tolerant of tie-breaking differences at the cutoff.

    When both sides' disagreeing entries sit within score_atol of the top-k
    cutoff (the lower of the two kappa-th scores), they're treated as a tie
    at the boundary — the kernel and reference may legitimately pick different
    keys there due to fp16/fp32 accumulation differences.
    """
    SENTINEL = 2**30
    idx_a_s = torch.where(idx_a < 0, torch.full_like(idx_a, SENTINEL), idx_a)
    idx_b_s = torch.where(idx_b < 0, torch.full_like(idx_b, SENTINEL), idx_b)

    sorted_a, perm_a = torch.sort(idx_a_s, dim=-1)
    sorted_b, perm_b = torch.sort(idx_b_s, dim=-1)

    if torch.equal(sorted_a, sorted_b):
        score_a_sorted = torch.gather(score_a, -1, perm_a)
        score_b_sorted = torch.gather(score_b, -1, perm_b)
        if not torch.allclose(score_a_sorted.float(), score_b_sorted.float(), atol=score_atol, rtol=0):
            diff = (score_a_sorted.float() - score_b_sorted.float()).abs()
            return False, f"score mismatch, max abs diff = {diff.max().item():.4e}"
        return True, "ok"

    min_score_a = score_a.min(dim=-1, keepdim=True).values
    min_score_b = score_b.min(dim=-1, keepdim=True).values
    boundary = torch.minimum(min_score_a, min_score_b)

    idx_mismatch = sorted_a != sorted_b

    score_a_sorted = torch.gather(score_a, -1, perm_a)
    score_b_sorted = torch.gather(score_b, -1, perm_b)

    a_near_boundary = (score_a_sorted.float() - boundary.float()).abs() < score_atol
    b_near_boundary = (score_b_sorted.float() - boundary.float()).abs() < score_atol

    acceptable = a_near_boundary & b_near_boundary
    hard_mismatch = idx_mismatch & ~acceptable

    if hard_mismatch.any():
        first = hard_mismatch.nonzero()[0]
        return False, f"index mismatch at {first.tolist()}"

    return True, "ok"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestAttentionTopK:
    @pytest.mark.parametrize("S,kappa,num_sink", [
        (4, 2, 0),
        (5, 2, 1),
        (6, 3, 0),
    ])
    def test_reference(self, S, kappa, num_sink):
        """Verify the reference's top-kappa outputs on small, inspectable cases.

        Checks — independently of the kernel — that the returned
        (index, score) pairs really are the top-kappa among the eligible
        (non-sink, causal) keys.

        Layouts (Q = query row, * = eligible key, . = masked):

            (S=4, kappa=2, sink=0) init_q_start=1, eligible queries 1..3
              Q\\K  0 1 2 3
               1   * *
               2   * * *
               3   * * * *

            (S=5, kappa=2, sink=1) init_q_start=2, eligible queries 2..4
              Q\\K  0 1 2 3 4
               2   . * *
               3   . * * *
               4   . * * * *

            (S=6, kappa=3, sink=0) init_q_start=2, eligible queries 2..5
              Q\\K  0 1 2 3 4 5
               2   * * *
               3   * * * *
               4   * * * * *
               5   * * * * * *

        Args:
            S: sequence length.
            kappa: number of top keys to select per eligible query.
            num_sink: number of leading sink positions excluded from top-kappa.
        """
        B, H_q, H_kv, D = 1, 1, 1, 8
        q, k, v = _make_qkv(B, H_q, H_kv, S, D)

        _, idx, score = attention_with_topk_reference(
            q, k, v, kappa=kappa, num_sink_tok_to_exclude=num_sink,
        )

        # Recompute full post-softmax weights for comparison.
        scale = torch.rsqrt(torch.tensor(float(D))).item()
        s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
        causal = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
        weights = torch.softmax(s.masked_fill(causal, float("-inf")), dim=-1)  # (1, 1, S, S)

        init_q_start = kappa - 1 + num_sink
        for r in range(S - init_q_start):
            q_abs = init_q_start + r
            returned_idx = idx[0, 0, r].tolist()
            returned_score = score[0, 0, r].float()

            # 1. Indices are absolute key positions within [num_sink, q_abs].
            for ki in returned_idx:
                assert num_sink <= ki <= q_abs, (
                    f"q={q_abs}: key idx {ki} outside eligible range "
                    f"[{num_sink}, {q_abs}]"
                )

            # 2. Returned scores match the post-softmax weights at those indices.
            expected_scores = weights[0, 0, q_abs, returned_idx].float()
            torch.testing.assert_close(returned_score, expected_scores, atol=1e-3, rtol=0)

            # 3. The returned set is the true top-kappa among eligible keys.
            eligible = list(range(num_sink, q_abs + 1))
            eligible_w = weights[0, 0, q_abs, eligible]
            _, top_pos = torch.topk(eligible_w, k=kappa)
            expected_set = {eligible[p] for p in top_pos.tolist()}
            assert set(returned_idx) == expected_set, (
                f"q={q_abs}: returned {sorted(returned_idx)} != "
                f"expected {sorted(expected_set)}"
            )


    @pytest.mark.parametrize("S", [128, 512, 1024])
    @pytest.mark.parametrize("D", [64, 128])
    @pytest.mark.parametrize("kappa", [4, 8])
    @pytest.mark.parametrize("num_sink_tok_to_exclude", [0, 4])
    def test_kernel_matches_reference(self, S, D, kappa, num_sink_tok_to_exclude):
        B, H_q, H_kv = 1, 4, 2
        q, k, v = _make_qkv(B, H_q, H_kv, S, D)

        out_ref, idx_ref, score_ref = attention_with_topk_reference(
            q, k, v, kappa=kappa, num_sink_tok_to_exclude=num_sink_tok_to_exclude,
        )
        out_tri, idx_tri, score_tri = attention_with_topk(
            q, k, v, kappa=kappa, num_sink_tok_to_exclude=num_sink_tok_to_exclude,
        )

        torch.testing.assert_close(out_tri, out_ref, atol=5e-3, rtol=0)

        init_q_start = kappa - 1 + num_sink_tok_to_exclude
        expected_shape = (B, H_q, S - init_q_start, kappa)
        assert idx_ref.shape == expected_shape, f"ref shape {idx_ref.shape} != {expected_shape}"
        assert idx_tri.shape == expected_shape, f"triton shape {idx_tri.shape} != {expected_shape}"

        ok, msg = _topk_sets_match(idx_ref, score_ref, idx_tri, score_tri)
        assert ok, f"top-kappa mismatch: {msg}"
