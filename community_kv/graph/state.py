"""Graph kind enum + per-layer state and timing-record dataclasses.

These hold the in-memory shape of a layer's graph and its measurement
records. Pure-data dataclasses — no methods that drive runtime behavior
(``GraphRuntime`` does that). Split out so workers and the runtime can
import them without dragging the executor / repartition logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class GraphAggregation(str, Enum):
    """Strategy for aggregating per-query-head top-kappa edges into graphs.

    - PER_QUERY_HEAD: one graph per query head. Highest accuracy, highest memory.
    - QUERY_GROUP: one graph per KV head, summing edges across the query heads
      that share that KV head.
    - LAYER_WISE: one graph per layer, summing edges across all heads.
    """

    PER_QUERY_HEAD = "per_query_head"
    QUERY_GROUP = "query_group"
    LAYER_WISE = "layer_wise"

    def num_graphs_per_layer(self, num_query_heads: int, num_kv_heads: int) -> int:
        if self is GraphAggregation.PER_QUERY_HEAD:
            return num_query_heads
        if self is GraphAggregation.QUERY_GROUP:
            return num_kv_heads
        if self is GraphAggregation.LAYER_WISE:
            return 1
        raise ValueError(f"Unknown aggregation: {self}")


@dataclass
class LayerLog:
    """One layer's measurement record.

    CUDA events are stripped on pickling (they're not picklable) so this
    dataclass is safe to send across ranks via ``dist.gather_object``.
    Consumers should read ``attn_ms`` only after calling
    ``log.resolve(devices)``; events are unresolved until then.
    """

    fwd_device: str
    part_device: str
    prefill_seq_len: int
    ev_attn_start: torch.cuda.Event | None = None
    ev_attn_end: torch.cuda.Event | None = None
    attn_ms: float = 0.0

    def resolve(self) -> None:
        """Compute ``attn_ms`` from the event pair if both are present."""
        if self.ev_attn_start is not None and self.ev_attn_end is not None:
            self.attn_ms = self.ev_attn_start.elapsed_time(self.ev_attn_end)

    def __getstate__(self) -> dict:
        # CUDA events aren't picklable; drop them after resolving timing.
        self.resolve()
        return {
            "fwd_device": self.fwd_device,
            "part_device": self.part_device,
            "prefill_seq_len": self.prefill_seq_len,
            "ev_attn_start": None,
            "ev_attn_end": None,
            "attn_ms": self.attn_ms,
        }

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


@dataclass
class PartitionRecord:
    """One layer's async partition stats. Filled in by worker thread."""

    layer_idx: int
    graph_idx: int
    fwd_device: str
    part_device: str
    start: float
    launch: float
    end: float
    elapsed_ms: float
    kernel_ms: float
    centroid_ms: float
    csr_ms: float
    n_edges: int
    num_communities_mean: float
    num_communities_max: int
    modularity: float
    rank: int = 0


@dataclass
class LayerGraph:
    """Per-layer graph state, held on the partition device.

    One per layer_idx (per rank under TP). Populated by ``async_partition_leiden``
    right after Leiden returns. The decode-time update path mutates several
    fields in place; the periodic re-partition path swaps the entire object
    atomically via dict assignment in ``GraphRuntime.graphs``.
    """

    layer_idx: int
    aggregation: GraphAggregation
    num_kv_heads_local: int
    prefill_seq_len: int
    head_dim: int
    device: torch.device
    community_ids: torch.Tensor
    num_communities: torch.Tensor
    centroids: torch.Tensor
    community_sizes: torch.Tensor
    community_sizes_prefill: torch.Tensor
    member_offsets: torch.Tensor
    member_positions: torch.Tensor
    community_weight: torch.Tensor
    total_weight: torch.Tensor
    decode_log_position: torch.Tensor
    decode_log_community: torch.Tensor
    decode_log_size: int = 0
    prefill_edge_src: torch.Tensor | None = None
    prefill_edge_dst: torch.Tensor | None = None
    prefill_edge_weight: torch.Tensor | None = None
    decode_edge_graph: torch.Tensor | None = None
    decode_edge_src_pos: torch.Tensor | None = None
    decode_edge_dst_pos: torch.Tensor | None = None
    decode_edge_weight: torch.Tensor | None = None
    decode_edge_size: int = 0
    version: int = 0
