"""Tests for community_kv.graph.partition.

Pulls invariant tests from prod's test_partition.py — shape and dense-numbering
invariants on community_ids per aggregation, num_communities matches the actual
range, COO edges match build_adjacency_batched. Empty-input case is CPU-runnable;
the live-Leiden cases require CUDA (skip without it).
"""

import pytest
import torch

from community_kv.graph.partition import (
    PartitionResult,
    build_adjacency_batched,
    dense_remap_per_graph,
    fill_isolated_vertices,
    partition,
    scatter_membership,
)
from community_kv.graph.state import GraphAggregation
from tests.conftest import LEIDEN_REQUIRED


def _coo_dict(src, dst, weight):
    return {(int(s), int(d)): float(w) for s, d, w in zip(src, dst, weight)}


def _expected_batched_inputs(topk_indices, topk_scores, aggregation, num_kv_heads, num_sink):
    """Reconstruct the (G, M, kappa) reshape that partition() does internally."""
    H_q, S_eligible, kappa = topk_indices.shape
    device = topk_indices.device
    G = aggregation.num_graphs_per_layer(H_q, num_kv_heads)
    base = torch.arange(
        kappa - 1 + num_sink,
        kappa - 1 + num_sink + S_eligible,
        device=device,
        dtype=torch.int64,
    )
    if aggregation == GraphAggregation.PER_QUERY_HEAD:
        return topk_indices, topk_scores, base.unsqueeze(0).expand(G, -1)
    if aggregation == GraphAggregation.QUERY_GROUP:
        heads_per_group = H_q // G
        bi = topk_indices.view(G, heads_per_group, S_eligible, kappa).reshape(
            G,
            heads_per_group * S_eligible,
            kappa,
        )
        bs = topk_scores.view(G, heads_per_group, S_eligible, kappa).reshape(
            G,
            heads_per_group * S_eligible,
            kappa,
        )
        bp = base.repeat(heads_per_group).unsqueeze(0).expand(G, -1)
        return bi, bs, bp
    if aggregation == GraphAggregation.LAYER_WISE:
        return (
            topk_indices.reshape(1, H_q * S_eligible, kappa),
            topk_scores.reshape(1, H_q * S_eligible, kappa),
            base.repeat(H_q).unsqueeze(0),
        )
    raise AssertionError


class TestPartition:
    def test_empty_topk_returns_singletons(self):
        """When kappa is 0 there are no edges; every position becomes its
        own singleton community."""
        H_q, S_eligible, kappa = 4, 6, 0
        topk_indices = torch.zeros(H_q, S_eligible, kappa, dtype=torch.int32)
        topk_scores = torch.zeros(H_q, S_eligible, kappa)
        result = partition(
            topk_indices,
            topk_scores,
            aggregation=GraphAggregation.PER_QUERY_HEAD,
            num_kv_heads=H_q,
            prefill_seq_len=20,
            num_sink=0,
            lam=0.5,
            leiden_resolution=1.0,
            leiden_max_iter=2,
        )
        assert isinstance(result, PartitionResult)
        assert result.edge_src.numel() == 0
        # Each vertex is a singleton -> num_communities == prefill_seq_len.
        assert (result.num_communities == 20).all()
        # Modularity is 0 with no edges.
        assert result.modularity == 0.0


class TestDenseRemapPerGraph:
    """The vectorised version must match the original Python-loop semantics:
    each row's labels become densely numbered in [0, num_unique[g]) preserving
    relative-position ordering by ``torch.unique`` (sorted ascending)."""

    def _loop_reference(self, membership: torch.Tensor) -> torch.Tensor:
        """Old per-row loop using torch.unique — kept here to pin behavior."""
        out = torch.empty_like(membership, dtype=torch.int32)
        for g in range(membership.shape[0]):
            _, inverse = torch.unique(membership[g], return_inverse=True)
            out[g] = inverse.to(torch.int32)
        return out

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_matches_loop_reference(self, seed):
        torch.manual_seed(seed)
        # Mixed range, including duplicates and -1 (sentinel) values.
        G, S = 4, 32
        membership = torch.randint(0, 8, (G, S), dtype=torch.int32)
        # Inject some -1 sentinels.
        mask = torch.rand(G, S) < 0.1
        membership[mask] = -1
        expected = self._loop_reference(membership)
        actual = dense_remap_per_graph(membership)
        assert torch.equal(actual, expected)

    def test_singleton_per_row(self):
        membership = torch.tensor([[5, 5, 5], [3, 3, 3]], dtype=torch.int32)
        out = dense_remap_per_graph(membership)
        # Every row collapses to all-zeros.
        assert torch.equal(out, torch.zeros_like(out))

    def test_already_dense(self):
        membership = torch.tensor([[0, 1, 2, 0], [3, 3, 4, 4]], dtype=torch.int32)
        out = dense_remap_per_graph(membership)
        # Within each row, ordering follows torch.unique ascending: row 1's
        # {3,4} -> {0,1}.
        assert out[0].tolist() == [0, 1, 2, 0]
        assert out[1].tolist() == [0, 0, 1, 1]


class TestFillIsolatedVertices:
    def test_no_isolation_passes_through(self):
        m = torch.tensor([[0, 1, 0, 1]], dtype=torch.int32)
        out = fill_isolated_vertices(m)
        assert torch.equal(out, m)

    def test_isolated_get_unique_singleton_ids(self):
        # Row max is 2, so isolated entries become 3, 4, 5 in order.
        m = torch.tensor([[0, -1, 2, -1, -1]], dtype=torch.int32)
        out = fill_isolated_vertices(m)
        assert out[0].tolist() == [0, 3, 2, 4, 5]


class TestScatterMembership:
    def test_unmapped_vertices_stay_minus_one(self):
        vertex = torch.tensor([0, 3], dtype=torch.long)
        partition = torch.tensor([7, 9], dtype=torch.int32)
        out = scatter_membership(
            vertex, partition, G=2, prefill_seq_len=2, device=torch.device("cpu")
        )
        # G=2, S=2 -> shape (2, 2). Vertices 0 (graph 0 pos 0) and 3 (graph 1 pos 1)
        # are filled; the others stay -1.
        assert out[0].tolist() == [7, -1]
        assert out[1].tolist() == [-1, 9]


@LEIDEN_REQUIRED
class TestPartitionGPU:
    @pytest.mark.parametrize(
        "aggregation,expected_G",
        [
            (GraphAggregation.PER_QUERY_HEAD, 8),
            (GraphAggregation.QUERY_GROUP, 2),
            (GraphAggregation.LAYER_WISE, 1),
        ],
    )
    def test_shapes_and_dense_numbering(self, aggregation, expected_G, make_topk):
        H_q, num_kv_heads, S_eligible, kappa = 8, 2, 16, 4
        prefill_seq_len = 32
        num_sink = 2
        ti, ts = make_topk(H_q, S_eligible, kappa, kappa - 1 + num_sink, device="cuda")
        result = partition(
            ti,
            ts,
            aggregation=aggregation,
            num_kv_heads=num_kv_heads,
            prefill_seq_len=prefill_seq_len,
            num_sink=num_sink,
            lam=0.5,
            leiden_resolution=1.0,
            leiden_max_iter=2,
        )
        assert result.community_ids.shape == (expected_G, prefill_seq_len)
        assert result.num_communities.shape == (expected_G,)
        # Dense numbering: ids in [0, num_communities[g]).
        for g in range(expected_G):
            n = int(result.num_communities[g].item())
            assert result.community_ids[g].max().item() == n - 1
            assert result.community_ids[g].min().item() >= 0

    def test_edges_match_build_adjacency_batched(self, make_topk):
        """The COO triple in PartitionResult must equal what
        build_adjacency_batched would produce on the same reshape."""
        H_q, num_kv_heads, S_eligible, kappa = 8, 2, 12, 4
        prefill_seq_len = 32
        num_sink = 2
        ti, ts = make_topk(H_q, S_eligible, kappa, kappa - 1 + num_sink, device="cuda")
        result = partition(
            ti,
            ts,
            aggregation=GraphAggregation.QUERY_GROUP,
            num_kv_heads=num_kv_heads,
            prefill_seq_len=prefill_seq_len,
            num_sink=num_sink,
            lam=0.5,
            leiden_resolution=1.0,
            leiden_max_iter=2,
        )
        bi, bs, bp = _expected_batched_inputs(
            ti,
            ts,
            GraphAggregation.QUERY_GROUP,
            num_kv_heads,
            num_sink,
        )
        exp_src, exp_dst, exp_w = build_adjacency_batched(
            bi,
            bs,
            bp,
            seq_len=prefill_seq_len,
            lam=0.5,
        )

        # Compare as sorted edge lists (Leiden may permute edge order? — no,
        # but to be safe sort by (src,dst) on both sides).
        def _sort_edges(s, d, w):
            order = torch.argsort(s.long() * (prefill_seq_len * 16) + d.long())
            return s[order], d[order], w[order]

        rs, rd, rw = _sort_edges(result.edge_src, result.edge_dst, result.edge_weight)
        es, ed, ew = _sort_edges(exp_src, exp_dst, exp_w)
        assert torch.equal(rs, es)
        assert torch.equal(rd, ed)
        assert torch.allclose(rw, ew, atol=1e-6)


class TestBuildAdjacencyBatched:
    def test_canonical_order(self):
        """All emitted edges must be upper-triangular (src <= dst)."""
        G, M, kappa = 1, 4, 3
        topk_indices = torch.tensor(
            [[[2, 1, 0], [3, 2, 1], [4, 3, 2], [5, 4, 3]]],
            dtype=torch.int32,
        )
        topk_scores = torch.full((G, M, kappa), 0.5)
        query_positions = torch.arange(2, 6, dtype=torch.int64).unsqueeze(0)
        src, dst, _ = build_adjacency_batched(
            topk_indices,
            topk_scores,
            query_positions,
            seq_len=8,
            lam=0.5,
        )
        assert (src <= dst).all()

    def test_invalid_indices_filtered(self):
        G, M, kappa = 1, 2, 3
        topk_indices = torch.tensor([[[0, -1, 1], [-1, -1, -1]]], dtype=torch.int32)
        topk_scores = torch.tensor([[[0.5, 0.3, 0.2], [0.1, 0.1, 0.1]]])
        query_positions = torch.tensor([[2, 3]], dtype=torch.int64)
        src, dst, _ = build_adjacency_batched(
            topk_indices,
            topk_scores,
            query_positions,
            seq_len=8,
            lam=1.0,
        )
        # All -1 entries (first row second col, all of row 1) drop out.
        # Remaining: (0,2) and (1,2) from the direct edges.
        assert _coo_dict(src, dst, torch.ones_like(src, dtype=torch.float32)).keys() == {
            (0, 2),
            (1, 2),
        }

    def test_per_graph_disjoint_offsets(self):
        """Vertex IDs must be offset by g*seq_len so subgraphs don't overlap."""
        G, M, kappa, seq_len = 2, 1, 1, 4
        topk_indices = torch.tensor([[[0]], [[1]]], dtype=torch.int32)
        topk_scores = torch.full((G, M, kappa), 0.5)
        query_positions = torch.tensor([[2], [2]], dtype=torch.int64)
        src, dst, _ = build_adjacency_batched(
            topk_indices,
            topk_scores,
            query_positions,
            seq_len=seq_len,
            lam=1.0,
        )
        # Graph 0 edge: (0, 2). Graph 1 edge: (5, 6) = (4+1, 4+2).
        d = _coo_dict(src, dst, torch.ones_like(src, dtype=torch.float32))
        assert (0, 2) in d
        assert (5, 6) in d
