from __future__ import annotations

import torch

from community_kv import GraphAggregation
import community_kv.graph.partition as partition_module
from community_kv.graph.partition import (
    PartitionResult,
    _reshape_by_aggregation,
    _reshape_query_groups,
    aggregation_mappings,
    build_adjacency,
    build_member_csr,
    build_partitioned_layer,
    compute_centroids,
    leiden_max_iterations,
)


def test_leiden_max_iterations_scale_with_context_length() -> None:
    assert leiden_max_iterations(1) == 1
    assert leiden_max_iterations(65_536) == 4
    assert leiden_max_iterations(99_999) == 4
    assert leiden_max_iterations(100_000) == 5


def test_query_group_reshape_accepts_qwen14b_group_size_five() -> None:
    indices = torch.arange(5 * 3 * 2, dtype=torch.int32).view(5, 3, 2)
    scores = indices.float()

    grouped_indices, grouped_scores, positions = _reshape_query_groups(
        indices,
        scores,
        num_kv_heads=1,
        num_sink=1,
    )

    assert grouped_indices.shape == (1, 15, 2)
    torch.testing.assert_close(grouped_scores, grouped_indices.float())
    assert positions.tolist() == [[2, 3, 4] * 5]


def test_partition_reshape_supports_all_aggregation_modes() -> None:
    indices = torch.arange(4 * 3 * 2, dtype=torch.int32).view(4, 3, 2)
    scores = indices.float()
    expected = {
        GraphAggregation.PER_QUERY_HEAD: (4, 3, 2),
        GraphAggregation.QUERY_GROUP: (2, 6, 2),
        GraphAggregation.LAYER_WISE: (1, 12, 2),
    }
    for aggregation, shape in expected.items():
        grouped, grouped_scores, positions = _reshape_by_aggregation(
            indices,
            scores,
            aggregation=aggregation,
            num_kv_heads=2,
            num_sink=1,
        )
        assert grouped.shape == shape
        assert grouped_scores.shape == shape
        assert positions.shape == shape[:2]


def test_aggregation_mappings_separate_graphs_from_kv_heads() -> None:
    expected = {
        GraphAggregation.PER_QUERY_HEAD: ([0, 1, 2, 3], [0, 0, 1, 1]),
        GraphAggregation.QUERY_GROUP: ([0, 1], [0, 1]),
        GraphAggregation.LAYER_WISE: ([0, 0], [0, 1]),
    }
    for aggregation, (graphs, kv_heads) in expected.items():
        to_graph, to_kv = aggregation_mappings(
            aggregation,
            query_heads=4,
            kv_heads=2,
            device="cpu",
        )
        assert to_graph.tolist() == graphs
        assert to_kv.tolist() == kv_heads


def test_member_csr_is_recency_descending() -> None:
    ids = torch.tensor([[0, 1, 1, 0, 1, 0]], dtype=torch.int32)
    counts = torch.tensor([2], dtype=torch.int32)
    offsets, positions = build_member_csr(ids, counts, num_sink=1)

    torch.testing.assert_close(
        offsets,
        torch.tensor([[0, 2, 5]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        positions,
        torch.tensor([[5, 3, 4, 2, 1]], dtype=torch.int32),
    )


def test_centroids_exclude_sink_and_reserve_decode_headroom() -> None:
    ids = torch.tensor([[0, 1, 1, 0]], dtype=torch.int32)
    counts = torch.tensor([2], dtype=torch.int32)
    keys = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    centroids, sizes = compute_centroids(
        ids,
        counts,
        keys,
        num_sink=1,
        max_decode_tokens=3,
    )

    assert centroids.shape == (1, 5, 2)
    assert centroids.dtype == torch.float8_e4m3fn
    assert sizes.tolist() == [[1, 2, 0, 0, 0]]
    torch.testing.assert_close(
        centroids[0, :2].float(),
        torch.tensor([[6.0, 7.0], [3.0, 4.0]]),
    )


def test_centroids_follow_retrieval_graph_and_kv_mappings() -> None:
    ids = torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.int32)
    counts = torch.tensor([2, 2], dtype=torch.int32)
    keys = torch.tensor(
        [
            [[0.0], [2.0], [4.0]],
            [[0.0], [10.0], [14.0]],
        ]
    )
    centroids, sizes = compute_centroids(
        ids,
        counts,
        keys,
        retrieval_to_graph=torch.tensor([0, 1, 1, 0], dtype=torch.int32),
        retrieval_to_kv=torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        num_sink=1,
        max_decode_tokens=1,
    )
    assert sizes[:, :2].tolist() == [[1, 1], [2, 0], [2, 0], [1, 1]]
    assert centroids[:, :2, 0].float().tolist() == [
        [2.0, 4.0],
        [3.0, 0.0],
        [12.0, 0.0],
        [10.0, 14.0],
    ]


def test_partitioned_layer_keeps_one_membership_row_per_graph(
    monkeypatch,
) -> None:
    community_ids = torch.tensor(
        [
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=torch.int32,
    )
    result = PartitionResult(
        community_ids=community_ids,
        community_counts=torch.full((4,), 2, dtype=torch.int32),
        modularity=0.0,
        edge_src=torch.empty(0, dtype=torch.int32),
        edge_dst=torch.empty(0, dtype=torch.int32),
        edge_weight=torch.empty(0),
    )
    monkeypatch.setattr(
        partition_module,
        "partition_graphs",
        lambda *args, **kwargs: result,
    )
    monkeypatch.setattr(
        partition_module,
        "compute_centroids",
        lambda *args, **kwargs: (
            torch.zeros((4, 3, 1), dtype=torch.float8_e4m3fn),
            torch.ones((4, 3), dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(
        partition_module,
        "build_member_csr",
        lambda *args, **kwargs: (
            torch.zeros((4, 3), dtype=torch.int32),
            torch.zeros((4, 3), dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(
        partition_module,
        "init_modularity_state",
        lambda *args, **kwargs: (
            torch.zeros((4, 3)),
            torch.zeros(4),
        ),
    )

    layer = build_partitioned_layer(
        layer_idx=0,
        topk_indices=torch.zeros((4, 1, 2), dtype=torch.int32),
        topk_scores=torch.ones((4, 1, 2)),
        keys=torch.zeros((2, 3, 1)),
        num_sink=1,
        lam=0.5,
        leiden_resolution=1.0,
        leiden_seed=0,
        max_decode_tokens=2,
        aggregation=GraphAggregation.PER_QUERY_HEAD,
    )

    assert layer.token_communities.shape == (4, 5)
    torch.testing.assert_close(layer.token_communities[:, :3], community_ids)
    assert torch.count_nonzero(layer.token_communities[:, 3:] != -1) == 0


def test_adjacency_coalesces_undirected_edges() -> None:
    indices = torch.tensor([[[1, 2], [2, 1]]], dtype=torch.int32)
    scores = torch.ones_like(indices, dtype=torch.float32)
    query_positions = torch.tensor([[2, 3]], dtype=torch.int64)
    source, destination, weight = build_adjacency(
        indices,
        scores,
        query_positions,
        seq_len=4,
        lam=1.0,
    )

    pairs = {
        (int(src), int(dst)): float(value)
        for src, dst, value in zip(source, destination, weight)
    }
    assert pairs == {(1, 2): 0.5, (2, 2): 0.5, (1, 3): 0.5, (2, 3): 0.5}


def test_adjacency_discards_positions_outside_the_sequence() -> None:
    indices = torch.tensor([[[1, 4], [-1, 5]]], dtype=torch.int32)
    scores = torch.ones_like(indices, dtype=torch.float32)
    query_positions = torch.tensor([[2, 3]], dtype=torch.int64)

    source, destination, weight = build_adjacency(
        indices,
        scores,
        query_positions,
        seq_len=4,
        lam=0.5,
    )

    pairs = {
        (int(src), int(dst)): float(value)
        for src, dst, value in zip(source, destination, weight)
    }
    assert pairs == {(1, 1): 0.5, (1, 2): 0.25}
