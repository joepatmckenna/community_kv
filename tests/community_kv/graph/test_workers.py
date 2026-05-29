"""Tests for community_kv.graph.workers — the pure helpers used by the
prefill / decode async workers (compute_centroids, build_member_csr,
init_modularity_state). Async-worker behaviour itself is exercised
end-to-end through the attention-forward functional tests."""

import torch

from community_kv.graph.workers import (
    build_member_csr,
    compute_centroids,
    init_modularity_state,
)


class TestGraphConstructionHelpers:
    """Pure helpers run after partitioning to build per-layer state:
      * ``compute_centroids`` — mean key per community + per-head sizes.
      * ``build_member_csr`` — CSR mapping (graph, community) -> recency-
        sorted prefill positions.
      * ``init_modularity_state`` — community_weight / total_weight from a
        freshly partitioned graph's COO edges.

    All three are CPU-runnable on tiny inputs; we pin behavioural
    contracts (sink exclusion, recency ordering, weight aggregation)."""

    def test_compute_centroids_excludes_sinks_from_scatter(self):
        """Sink positions [0, num_sink) should not contribute to any centroid;
        their singleton communities stay at -inf with size 0."""
        G, S, D = 1, 6, 4
        num_sink = 2
        # 2 sinks (community 0), then 4 prefill positions in community 1.
        community_ids = torch.tensor(
            [[0, 0, 1, 1, 1, 1]],
            dtype=torch.int32,
        )
        num_communities = torch.tensor([2], dtype=torch.int32)
        keys = torch.ones(G, S, D)
        centroids, sizes = compute_centroids(
            community_ids,
            num_communities,
            keys,
            num_kv_heads=1,
            max_new_tokens=1,
            num_sink=num_sink,
        )
        # Community 0 (sinks) has size 0.
        assert sizes[0, 0].item() == 0.0
        # Community 1 has 4 positions.
        assert sizes[0, 1].item() == 4.0
        # Sink centroid stays -inf; community 1 centroid is the mean (1.0).
        assert torch.isinf(centroids[0, 0]).all()
        assert torch.allclose(centroids[0, 1], torch.ones(D))

    def test_build_member_csr_two_communities_recency_order(self):
        """G=1, S=8, num_sink=2. Communities: [_, _, 0, 1, 0, 1, 0, 1].
        After dropping sinks, community 0 has positions {2, 4, 6}
        sorted newest-first as [6, 4, 2]; community 1 has [7, 5, 3]."""
        community_ids = torch.tensor(
            [[0, 0, 0, 1, 0, 1, 0, 1]],
            dtype=torch.int32,
        )
        num_communities = torch.tensor([2], dtype=torch.int32)
        offsets, positions = build_member_csr(
            community_ids,
            num_communities,
            num_sink=2,
        )
        assert offsets[0].tolist() == [0, 3, 6]
        assert positions[0, 0:3].tolist() == [6, 4, 2]
        assert positions[0, 3:6].tolist() == [7, 5, 3]

    def test_init_modularity_state_no_edges_zero_state(self):
        G, S, max_C = 2, 4, 3
        cw, tw = init_modularity_state(
            community_ids=torch.zeros(G, S, dtype=torch.int32),
            num_communities=torch.tensor([1, 1], dtype=torch.int32),
            edge_src=torch.empty(0, dtype=torch.int32),
            edge_dst=torch.empty(0, dtype=torch.int32),
            edge_weight=torch.empty(0, dtype=torch.float32),
            seq_len=S,
            max_C=max_C,
        )
        assert cw.shape == (G, max_C)
        assert tw.shape == (G,)
        assert torch.all(cw == 0)
        assert torch.all(tw == 0)

    def test_init_modularity_state_within_community_edge(self):
        G, S, max_C = 1, 4, 2
        community_ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)
        edge_src = torch.tensor([0], dtype=torch.int32)
        edge_dst = torch.tensor([1], dtype=torch.int32)
        edge_weight = torch.tensor([3.0], dtype=torch.float32)
        cw, tw = init_modularity_state(
            community_ids=community_ids,
            num_communities=torch.tensor([2], dtype=torch.int32),
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_weight=edge_weight,
            seq_len=S,
            max_C=max_C,
        )
        assert tw[0].item() == 6.0  # 2 * w
        assert cw[0, 0].item() == 6.0  # both endpoints contribute w each
        assert cw[0, 1].item() == 0.0
