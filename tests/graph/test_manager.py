"""Tests for community_kv.graph.manager._initialize_impl.

These tests call _initialize_impl directly (bypassing the async executor in
.initialize()) and check shape invariants plus a few concrete numerical
properties:

- community_sizes[g, c] is the histogram of community_ids[g, :prefill_seq_len]
- total_weight[g] equals the sum of adjacency[g] edge weights
- centroids[ch, c] equals the mean of keys at positions assigned to c

Leiden is stochastic, so we don't check *which* communities get formed — only
that the derived state is self-consistent.
"""

from __future__ import annotations
import pytest
import torch

from community_kv.graph.manager import GraphManager
from community_kv.graph.utils import GraphAggregation


try:
    import cugraph  # noqa: F401
    import cudf  # noqa: F401
    _HAS_CUGRAPH = True
except ImportError:
    _HAS_CUGRAPH = False


# Parametrize every test across both backends. cugraph is skipped when not installed.
BACKENDS = [
    "igraph",
    pytest.param("cugraph", marks=pytest.mark.skipif(
        not _HAS_CUGRAPH, reason="cugraph/cudf not installed",
    )),
]


def _make_topk(H_q, S_eligible, kappa, init_q_start, device, seed=0):
    """Synthesize valid causal top-kappa data.

    For query at absolute pos p = init_q_start + r, we pick the kappa most
    recent valid keys [p, p-1, ..., p-kappa+1]. Since p >= kappa - 1 these
    are all non-negative. Scores are uniform random in [0.1, 0.6].
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    topk_indices = torch.zeros(H_q, S_eligible, kappa, dtype=torch.int32, device=device)
    for r in range(S_eligible):
        p = init_q_start + r
        for j in range(kappa):
            topk_indices[:, r, j] = p - j
    topk_scores = torch.rand(H_q, S_eligible, kappa, generator=gen, device=device) * 0.5 + 0.1
    return topk_indices, topk_scores


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestGraphManager:
    H_Q = 4
    H_KV = 2
    HEAD_DIM = 8
    PREFILL_SEQ_LEN = 16
    KAPPA = 2
    NUM_SINK = 0
    MAX_NEW_TOKENS = 8
    LAM = 0.5
    LAYER_IDX = 0

    def _setup(self, aggregation, leiden_backend="igraph", device="cuda"):
        manager = GraphManager(
            aggregation=aggregation,
            num_query_heads=self.H_Q,
            num_kv_heads=self.H_KV,
            num_layers=1,
            head_dim=self.HEAD_DIM,
            num_sink_tok_to_exclude=self.NUM_SINK,
            lam=self.LAM,
            token_budget=8,
            max_new_tokens=self.MAX_NEW_TOKENS,
            leiden_backend=leiden_backend,
        )
        init_q_start = self.KAPPA - 1 + self.NUM_SINK
        S_eligible = self.PREFILL_SEQ_LEN - init_q_start
        topk_indices, topk_scores = _make_topk(
            self.H_Q, S_eligible, self.KAPPA, init_q_start, device
        )
        # fp32 keys so centroid means are precise enough for tight atol.
        keys = torch.randn(self.H_KV, self.PREFILL_SEQ_LEN, self.HEAD_DIM, device=device)
        manager._initialize_impl(self.LAYER_IDX, topk_indices, topk_scores, keys)
        return manager, keys

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("aggregation,expected_G", [
        (GraphAggregation.PER_QUERY_HEAD, 4),
        (GraphAggregation.QUERY_GROUP, 2),
        (GraphAggregation.LAYER_WISE, 1),
    ])
    def test_initialize_impl_shapes(self, aggregation, expected_G, backend):
        manager, _ = self._setup(aggregation, leiden_backend=backend)
        layer = self.LAYER_IDX
        G = expected_G
        num_centroid_heads = max(G, self.H_KV)
        max_seq_len = self.PREFILL_SEQ_LEN + self.MAX_NEW_TOKENS

        assert manager.num_graphs_per_layer == G
        assert manager.seq_len[layer] == self.PREFILL_SEQ_LEN
        assert layer in manager._initialized

        assert manager.community_ids[layer].shape == (G, max_seq_len)
        assert manager.num_communities[layer].shape == (G,)
        assert manager.total_weight[layer].shape == (G,)
        assert manager.modularity[layer].shape == (G,)
        assert len(manager.adjacency[layer]) == G

        # Shapes that depend on the number of communities Leiden discovered
        max_num_communities = int(manager.num_communities[layer].max().item()) + self.MAX_NEW_TOKENS
        assert manager.centroids[layer].shape == (num_centroid_heads, max_num_communities, self.HEAD_DIM)
        assert manager.community_sizes[layer].shape == (G, max_num_communities)
        assert manager.community_weight[layer].shape == (G, max_num_communities)

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("aggregation", [
        GraphAggregation.PER_QUERY_HEAD,
        GraphAggregation.QUERY_GROUP,
        GraphAggregation.LAYER_WISE,
    ])
    def test_initialize_impl_community_sizes_histogram(self, aggregation, backend):
        """community_sizes[g, c] = count of prefill tokens assigned to community c."""
        manager, _ = self._setup(aggregation, leiden_backend=backend)
        layer = self.LAYER_IDX
        G = manager.num_graphs_per_layer
        cids = manager.community_ids[layer][:, :self.PREFILL_SEQ_LEN]  # (G, S)
        sizes = manager.community_sizes[layer]  # (G, max_num_communities)

        # Every prefill token is assigned to exactly one community per graph
        assert (sizes.sum(dim=-1) == self.PREFILL_SEQ_LEN).all()

        # Entry-by-entry histogram match
        for g in range(G):
            num_c = int(manager.num_communities[layer][g].item())
            for c in range(num_c):
                expected = int((cids[g] == c).sum().item())
                assert int(sizes[g, c].item()) == expected, (
                    f"g={g} c={c}: size={sizes[g, c].item()} != count={expected}"
                )

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("aggregation", [
        GraphAggregation.PER_QUERY_HEAD,
        GraphAggregation.QUERY_GROUP,
        GraphAggregation.LAYER_WISE,
    ])
    def test_initialize_impl_total_weight_matches_adjacency(self, aggregation, backend):
        """total_weight[g] == sum of edge weights in adjacency[g]."""
        manager, _ = self._setup(aggregation, leiden_backend=backend)
        layer = self.LAYER_IDX
        G = manager.num_graphs_per_layer
        for g in range(G):
            _, _, edge_weight = manager.adjacency[layer][g]
            expected = edge_weight.sum().to(manager.total_weight[layer].dtype)
            got = manager.total_weight[layer][g]
            torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("aggregation", [
        GraphAggregation.PER_QUERY_HEAD,
        GraphAggregation.QUERY_GROUP,
        GraphAggregation.LAYER_WISE,
    ])
    def test_initialize_impl_centroids_are_means(self, aggregation, backend):
        """centroids[ch, c] is the mean of the keys at positions assigned to c.

        Mirrors the per-centroid-head mapping from _initialize_impl:
          - G < H_kv:   community_ids expands across centroid heads (same partition)
          - G > H_kv:   keys are replicated across centroid heads
          - G == H_kv:  identity
        Empty community slots are -inf.
        """
        manager, keys = self._setup(aggregation, leiden_backend=backend)
        layer = self.LAYER_IDX
        G = manager.num_graphs_per_layer
        num_centroid_heads = max(G, self.H_KV)
        cids = manager.community_ids[layer][:, :self.PREFILL_SEQ_LEN]  # (G, S)

        if G < self.H_KV:
            cids_per_ch = cids.expand(num_centroid_heads, -1)
            keys_per_ch = keys  # (H_kv == num_centroid_heads, S, D)
        elif G > self.H_KV:
            factor = G // self.H_KV
            cids_per_ch = cids  # (G == num_centroid_heads, S)
            keys_per_ch = keys.unsqueeze(1).expand(-1, factor, -1, -1).contiguous().view(
                num_centroid_heads, self.PREFILL_SEQ_LEN, self.HEAD_DIM
            )
        else:
            cids_per_ch = cids
            keys_per_ch = keys

        num_c_max = int(manager.num_communities[layer].max().item())
        centroids = manager.centroids[layer]
        for ch in range(num_centroid_heads):
            for c in range(num_c_max):
                mask = cids_per_ch[ch] == c
                if mask.any():
                    expected = keys_per_ch[ch][mask].mean(dim=0)
                    got = centroids[ch, c]
                    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)
                else:
                    assert torch.isinf(centroids[ch, c]).all() and (centroids[ch, c] < 0).all(), (
                        f"ch={ch} c={c}: empty community should be -inf, got {centroids[ch, c]}"
                    )
