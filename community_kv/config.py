"""Public configuration for CommunityKV graph aggregation and execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GraphAggregation(str, Enum):
    """How query-head top-k edges are combined into layer graphs."""

    PER_QUERY_HEAD = "per_query_head"
    QUERY_GROUP = "query_group"
    LAYER_WISE = "layer_wise"

    def graph_count(self, *, query_heads: int, kv_heads: int) -> int:
        if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
            raise ValueError("query heads must divide evenly into KV-head groups")
        if self is GraphAggregation.PER_QUERY_HEAD:
            return query_heads
        if self is GraphAggregation.QUERY_GROUP:
            return kv_heads
        return 1

    def retrieval_head_count(self, *, query_heads: int, kv_heads: int) -> int:
        graph_count = self.graph_count(
            query_heads=query_heads,
            kv_heads=kv_heads,
        )
        return max(graph_count, kv_heads)


@dataclass(frozen=True, slots=True)
class CommunityKVConfig:
    """Configuration whose values are fixed for one captured request."""

    token_budget: int = 4096
    num_sink: int = 10
    lam: float = 0.5
    aggregation: GraphAggregation = GraphAggregation.PER_QUERY_HEAD
    leiden_resolution: float = 1.0
    leiden_seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.aggregation, str):
            object.__setattr__(
                self,
                "aggregation",
                GraphAggregation(self.aggregation),
            )
        if self.token_budget <= 0 or self.token_budget % 64:
            raise ValueError("token_budget must be a positive multiple of 64")
        if self.num_sink < 0 or self.num_sink + 1 >= self.token_budget:
            raise ValueError("num_sink leaves no selected-token capacity")
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError("lam must be in [0, 1]")
        if self.leiden_resolution <= 0:
            raise ValueError("leiden_resolution must be positive")


@dataclass(frozen=True, slots=True)
class PartitionConfig:
    """Dedicated-GPU prefill partition scheduling configuration."""

    devices: tuple[str, ...] = ()
    workers_per_device: int = 2

    def __post_init__(self) -> None:
        if self.workers_per_device <= 0:
            raise ValueError("workers_per_device must be positive")


__all__ = ["CommunityKVConfig", "GraphAggregation", "PartitionConfig"]
