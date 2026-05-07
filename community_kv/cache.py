"""CommunityKV cache utilities."""

from __future__ import annotations
from enum import Enum


class GraphAggregation(str, Enum):
    """Strategy for aggregating per-query-head top-kappa edges into graphs.

    - PER_HEAD: one graph per attention head.
    - QUERY_GROUP: one graph per KV head, summing edges across the query heads
      that share that KV head.
    - LAYER_WISE: one graph per layer, summing edges across all heads.
    """

    PER_HEAD = "per_head"
    QUERY_GROUP = "query_group"
    LAYER_WISE = "layer_wise"

    def num_graphs_per_layer(self, num_query_heads: int, num_kv_heads: int) -> int:
        """Compute how many independent graphs this strategy produces per layer."""
        if self is GraphAggregation.PER_HEAD:
            return num_query_heads
        if self is GraphAggregation.QUERY_GROUP:
            return num_kv_heads
        if self is GraphAggregation.LAYER_WISE:
            return 1
        raise ValueError(f"Unknown aggregation: {self}")
