"""Tests for community_kv.graph.utils (GraphAggregation + build_adjacency)."""

from __future__ import annotations
import pytest
import torch

from community_kv.graph.utils import GraphAggregation, build_adjacency


def _edges_to_dict(src, dst, weight):
    """Convert COO tensors to {(src, dst): weight} for order-independent comparison."""
    return {
        (int(s), int(d)): pytest.approx(float(w), abs=1e-6)
        for s, d, w in zip(src.tolist(), dst.tolist(), weight.tolist())
    }


def _coo_to_dense_symmetric(src, dst, weight, n):
    """Materialize a symmetric dense adjacency from upper-triangle COO.

    build_adjacency stores each undirected edge once as (min, max). The
    physical graph the COO represents is symmetric: both (i, j) and (j, i)
    have the same weight.
    """
    m = torch.zeros(n, n)
    m[src.long(), dst.long()] = weight
    m[dst.long(), src.long()] = weight
    return m


class TestGraphUtils:
    @pytest.mark.parametrize("agg,H_q,H_kv,expected", [
        (GraphAggregation.PER_QUERY_HEAD, 32, 8, 32),
        (GraphAggregation.QUERY_GROUP, 32, 8, 8),
        (GraphAggregation.LAYER_WISE, 32, 8, 1),
        (GraphAggregation.PER_QUERY_HEAD, 4, 4, 4),  # H_q == H_kv
    ])
    def test_aggregation_num_graphs_per_layer(self, agg, H_q, H_kv, expected):
        assert agg.num_graphs_per_layer(H_q, H_kv) == expected

    def test_build_adjacency_direct_only(self):
        """lam=1.0: only w1 (query->key) edges, each scaled by lam/2 = 0.5.

        Verifies the full symmetric dense adjacency matrix.

        Queries at [3, 4], kappa=2, seq_len=5:
          q=3 -> keys [0, 2] with scores [0.4, 0.6]
          q=4 -> keys [1, 2] with scores [0.3, 0.7]

        Expected dense matrix (symmetric, 5x5, scaled by lam/2=0.5):
                k=0   k=1   k=2   k=3   k=4
          i=0    0     0     0    0.20   0
          i=1    0     0     0     0    0.15
          i=2    0     0     0    0.30  0.35
          i=3   0.20   0    0.30   0     0
          i=4    0    0.15  0.35   0     0
        """
        topk_indices = torch.tensor([[0, 2], [1, 2]], dtype=torch.int32)
        topk_scores = torch.tensor([[0.4, 0.6], [0.3, 0.7]])
        src, dst, weight = build_adjacency(
            topk_indices, topk_scores,
            seq_len=5, lam=1.0, query_offset=3,
        )
        dense = _coo_to_dense_symmetric(src, dst, weight, n=5)

        expected = torch.zeros(5, 5)
        expected[0, 3] = expected[3, 0] = 0.4 * 0.5
        expected[2, 3] = expected[3, 2] = 0.6 * 0.5
        expected[1, 4] = expected[4, 1] = 0.3 * 0.5
        expected[2, 4] = expected[4, 2] = 0.7 * 0.5

        torch.testing.assert_close(dense, expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(dense, dense.T, atol=0, rtol=0)

    def test_build_adjacency_coattention_only(self):
        """lam=0.0: only w2 (co-attention) contributes; w1 entries have weight 0.

        w2 is the outer product of alpha[m] over keys in topk(m), so a pair
        (i, j) in topk produces all four entries (i,i), (i,j), (j,i), (j,j).

        Verifies the full symmetric dense adjacency matrix.

        Same inputs as direct-only. Co-attention from:
          q=3: keys [0, 2], scores [0.4, 0.6]
            (0,0)=0.16  (0,2)=(2,0)=0.24  (2,2)=0.36
          q=4: keys [1, 2], scores [0.3, 0.7]
            (1,1)=0.09  (1,2)=(2,1)=0.21  (2,2)=0.49

        (2,2) is the sum across both queries: 0.36 + 0.49 = 0.85.

        Expected dense matrix (symmetric, 5x5):
                k=0   k=1   k=2   k=3   k=4
          i=0   0.16   0    0.24   0     0
          i=1    0    0.09  0.21   0     0
          i=2   0.24  0.21  0.85   0     0
          i=3    0     0     0     0     0
          i=4    0     0     0     0     0
        """
        topk_indices = torch.tensor([[0, 2], [1, 2]], dtype=torch.int32)
        topk_scores = torch.tensor([[0.4, 0.6], [0.3, 0.7]])
        src, dst, weight = build_adjacency(
            topk_indices, topk_scores,
            seq_len=5, lam=0.0, query_offset=3,
        )
        dense = _coo_to_dense_symmetric(src, dst, weight, n=5)

        expected = torch.zeros(5, 5)
        # q=3 contributions (keys 0, 2; scores 0.4, 0.6)
        expected[0, 0] += 0.4 * 0.4
        expected[0, 2] += 0.4 * 0.6
        expected[2, 0] += 0.4 * 0.6
        expected[2, 2] += 0.6 * 0.6
        # q=4 contributions (keys 1, 2; scores 0.3, 0.7)
        expected[1, 1] += 0.3 * 0.3
        expected[1, 2] += 0.3 * 0.7
        expected[2, 1] += 0.3 * 0.7
        expected[2, 2] += 0.7 * 0.7

        torch.testing.assert_close(dense, expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(dense, dense.T, atol=0, rtol=0)

    def test_build_adjacency_mixed_with_dedup(self):
        """lam=0.5: w1 and w2 combine; overlapping edges sum; w2 includes diagonals.

        Queries at [3, 4], kappa=2, seq_len=5:
          q=3 -> keys [0, 1] with scores [0.4, 0.6]
          q=4 -> keys [1, 3] with scores [0.5, 0.5]

        w1 (scale lam/2 = 0.25), symmetrized:
          (0, 3)=(3, 0)=0.10
          (1, 3)=(3, 1)=0.15       <-- overlaps w2 (1,3)=(3,1) below
          (1, 4)=(4, 1)=0.125
          (3, 4)=(4, 3)=0.125

        w2 (scale 1-lam = 0.5), outer product per query:
          q=3 -> (0,0)=0.08, (0,1)=(1,0)=0.12, (1,1)=0.18
          q=4 -> (1,1)=0.125, (1,3)=(3,1)=0.125, (3,3)=0.125

        Dedup sums entries that share a canonical (src, dst):
          (1, 1) = 0.18 + 0.125 = 0.305       (q=3 diag + q=4 diag)
          (1, 3) = 0.15 + 0.125 = 0.275       (w1 overlap with w2)
        """
        topk_indices = torch.tensor([[0, 1], [1, 3]], dtype=torch.int32)
        topk_scores = torch.tensor([[0.4, 0.6], [0.5, 0.5]])
        src, dst, weight = build_adjacency(
            topk_indices, topk_scores,
            seq_len=5, lam=0.5, query_offset=3,
        )
        dense = _coo_to_dense_symmetric(src, dst, weight, n=5)

        expected = torch.zeros(5, 5)
        # w1 (scale lam/2 = 0.25), symmetrized
        expected[0, 3] = expected[3, 0] = 0.10
        expected[1, 3] = expected[3, 1] = 0.15       # will dedup-sum with w2 below
        expected[1, 4] = expected[4, 1] = 0.125
        expected[3, 4] = expected[4, 3] = 0.125
        # w2 (scale 1-lam = 0.5), outer product per query
        # q=3: (0,0)=0.08, (0,1)=(1,0)=0.12, (1,1)=0.18
        expected[0, 0] += 0.08
        expected[0, 1] += 0.12
        expected[1, 0] += 0.12
        expected[1, 1] += 0.18
        # q=4: (1,1)=0.125, (1,3)=(3,1)=0.125, (3,3)=0.125
        expected[1, 1] += 0.125
        expected[1, 3] += 0.125
        expected[3, 1] += 0.125
        expected[3, 3] += 0.125

        torch.testing.assert_close(dense, expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(dense, dense.T, atol=0, rtol=0)

    def test_build_adjacency_filters_invalid_indices(self):
        """Slots with index -1 produce no edges (neither w1 nor w2).

        Inputs (lam=1.0 so only w1 exercised):
          q=3 -> keys [-1, 2], scores [0.0, 1.0]   # -1 slot must be dropped
          q=4 -> keys  [1, 2], scores [0.5, 0.5]

        w1 after filter + lam/2=0.5 scale, symmetrized:
          (2, 3)=(3, 2)=0.5     (q=3 -> k=2)
          (1, 4)=(4, 1)=0.25    (q=4 -> k=1)
          (2, 4)=(4, 2)=0.25    (q=4 -> k=2)
        """
        topk_indices = torch.tensor([[-1, 2], [1, 2]], dtype=torch.int32)
        topk_scores = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
        src, dst, weight = build_adjacency(
            topk_indices, topk_scores,
            seq_len=5, lam=1.0, query_offset=3,
        )
        # Guard: dense materialization below would silently treat -1 as
        # "last row" via negative indexing. Assert it's absent first.
        assert (src >= 0).all() and (dst >= 0).all(), (
            f"-1 leaked into output: src={src}, dst={dst}"
        )

        dense = _coo_to_dense_symmetric(src, dst, weight, n=5)

        expected = torch.zeros(5, 5)
        expected[2, 3] = expected[3, 2] = 0.5
        expected[1, 4] = expected[4, 1] = 0.25
        expected[2, 4] = expected[4, 2] = 0.25

        torch.testing.assert_close(dense, expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(dense, dense.T, atol=0, rtol=0)

    def test_build_adjacency_canonicalization(self):
        """Every returned edge satisfies edge_src <= edge_dst."""
        # Mix of cases where query < key, query > key (can't happen causally, but the
        # function doesn't enforce it), and co-attention pairs.
        topk_indices = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
        topk_scores = torch.tensor([[0.4, 0.6], [0.5, 0.5]])
        src, dst, weight = build_adjacency(
            topk_indices, topk_scores,
            seq_len=5, lam=0.5, query_offset=3,
        )
        assert (src <= dst).all(), f"found non-canonical edges: src={src}, dst={dst}"
