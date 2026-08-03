"""Top-k graph construction, Leiden partitioning, and decode-state assembly."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import torch

from community_kv.config import GraphAggregation
from community_kv.graph.leiden import run_leiden


def leiden_max_iterations(input_context_length: int) -> int:
    """Return the Leiden sweep limit for an input context length."""

    if (
        not isinstance(input_context_length, int)
        or isinstance(input_context_length, bool)
        or input_context_length <= 0
    ):
        raise ValueError("input_context_length must be a positive integer")
    return max(1, math.floor(math.log10(input_context_length)))


@dataclass(slots=True)
class PartitionResult:
    community_ids: torch.Tensor
    community_counts: torch.Tensor
    modularity: float
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_weight: torch.Tensor


@dataclass(slots=True)
class PartitionedLayer:
    """Compact tensors transferred from a partition GPU to the model GPU."""

    layer_idx: int
    prefill_seq_len: int
    aggregation: GraphAggregation
    query_heads: int
    kv_heads: int
    retrieval_to_graph: torch.Tensor
    retrieval_to_kv: torch.Tensor
    member_offsets: torch.Tensor
    member_positions: torch.Tensor
    community_counts: torch.Tensor
    centroids: torch.Tensor
    community_sizes: torch.Tensor
    community_weight: torch.Tensor
    total_weight: torch.Tensor
    token_communities: torch.Tensor

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "PartitionedLayer":
        target = torch.device(device)
        updates: dict[str, object] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, torch.Tensor):
                updates[item.name] = value.to(target, non_blocking=non_blocking)
        return replace(self, **updates)


def aggregation_mappings(
    aggregation: GraphAggregation,
    *,
    query_heads: int,
    kv_heads: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each retrieval head to its graph and KV-cache head."""

    retrieval_head_count = aggregation.retrieval_head_count(
        query_heads=query_heads,
        kv_heads=kv_heads,
    )
    retrieval_head = torch.arange(
        retrieval_head_count,
        dtype=torch.int32,
        device=device,
    )
    if aggregation is GraphAggregation.PER_QUERY_HEAD:
        group_size = query_heads // kv_heads
        return retrieval_head, retrieval_head // group_size
    if aggregation is GraphAggregation.QUERY_GROUP:
        return retrieval_head, retrieval_head
    return torch.zeros_like(retrieval_head), retrieval_head


def _reshape_by_aggregation(
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    *,
    aggregation: GraphAggregation,
    num_kv_heads: int,
    num_sink: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if topk_indices.ndim != 3 or topk_scores.shape != topk_indices.shape:
        raise ValueError("top-k tensors must have shape [query heads, rows, kappa]")
    query_heads, eligible_rows, kappa = topk_indices.shape
    if num_kv_heads <= 0 or query_heads % num_kv_heads:
        raise ValueError("query heads must divide evenly into KV-head groups")
    group_size = query_heads // num_kv_heads
    if not 1 <= group_size <= 16:
        raise ValueError("GQA group size must be in [1, 16]")
    base_positions = torch.arange(
        kappa - 1 + num_sink,
        kappa - 1 + num_sink + eligible_rows,
        dtype=torch.int64,
        device=topk_indices.device,
    )
    if aggregation is GraphAggregation.PER_QUERY_HEAD:
        return (
            topk_indices,
            topk_scores,
            base_positions.unsqueeze(0).expand(query_heads, -1),
        )
    if aggregation is GraphAggregation.QUERY_GROUP:
        return (
            topk_indices.view(
                num_kv_heads,
                group_size,
                eligible_rows,
                kappa,
            ).reshape(num_kv_heads, group_size * eligible_rows, kappa),
            topk_scores.view(
                num_kv_heads,
                group_size,
                eligible_rows,
                kappa,
            ).reshape(num_kv_heads, group_size * eligible_rows, kappa),
            base_positions.repeat(group_size)
            .unsqueeze(0)
            .expand(num_kv_heads, -1),
        )
    return (
        topk_indices.reshape(1, query_heads * eligible_rows, kappa),
        topk_scores.reshape(1, query_heads * eligible_rows, kappa),
        base_positions.repeat(query_heads).unsqueeze(0),
    )


def _reshape_query_groups(
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    *,
    num_kv_heads: int,
    num_sink: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for the original query-group partition helper."""

    return _reshape_by_aggregation(
        topk_indices,
        topk_scores,
        aggregation=GraphAggregation.QUERY_GROUP,
        num_kv_heads=num_kv_heads,
        num_sink=num_sink,
    )


def build_adjacency(
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    seq_len: int,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build and coalesce the disjoint batched CommunityKV graph."""

    graph_count, _, kappa = topk_indices.shape
    device = topk_indices.device
    total_vertices = graph_count * seq_len
    graph_offset = (
        torch.arange(graph_count, device=device, dtype=torch.int64).unsqueeze(1)
        * seq_len
    )

    source = (query_positions.long() + graph_offset).unsqueeze(-1).expand_as(
        topk_indices
    )
    destination = topk_indices.long() + graph_offset.unsqueeze(-1)
    valid = (topk_indices >= 0) & (topk_indices < seq_len)
    direct_src = source[valid]
    direct_dst = destination[valid]
    direct_weight = topk_scores.float()[valid]

    if lam < 1.0:
        row, column = torch.triu_indices(
            kappa,
            kappa,
            device=device,
        )
        co_src = topk_indices[..., row]
        co_dst = topk_indices[..., column]
        valid_co = (
            (co_src >= 0)
            & (co_src < seq_len)
            & (co_dst >= 0)
            & (co_dst < seq_len)
        )
        co_src_global = co_src.long() + graph_offset.unsqueeze(-1)
        co_dst_global = co_dst.long() + graph_offset.unsqueeze(-1)
        co_weight = (
            topk_scores[..., row].float() * topk_scores[..., column].float()
        )
        co_src_global = co_src_global[valid_co]
        co_dst_global = co_dst_global[valid_co]
        co_weight = co_weight[valid_co]
    else:
        co_src_global = torch.empty(0, dtype=torch.int64, device=device)
        co_dst_global = torch.empty(0, dtype=torch.int64, device=device)
        co_weight = torch.empty(0, dtype=torch.float32, device=device)

    all_src = torch.cat(
        (
            torch.minimum(direct_src, direct_dst),
            torch.minimum(co_src_global, co_dst_global),
        )
    )
    all_dst = torch.cat(
        (
            torch.maximum(direct_src, direct_dst),
            torch.maximum(co_src_global, co_dst_global),
        )
    )
    all_weight = torch.cat(
        (direct_weight * (lam / 2.0), co_weight * (1.0 - lam))
    )
    keys = all_src * total_vertices + all_dst
    unique_keys, inverse = keys.unique(return_inverse=True)
    edge_weight = torch.zeros(
        unique_keys.shape[0],
        dtype=torch.float32,
        device=device,
    )
    edge_weight.scatter_add_(0, inverse, all_weight)
    return (
        (unique_keys // total_vertices).to(torch.int32),
        (unique_keys % total_vertices).to(torch.int32),
        edge_weight,
    )


def _scatter_membership(
    vertex: torch.Tensor,
    labels: torch.Tensor,
    *,
    graph_count: int,
    seq_len: int,
) -> torch.Tensor:
    membership = torch.full(
        (graph_count * seq_len,),
        -1,
        dtype=torch.int32,
        device=vertex.device,
    )
    membership[vertex] = labels
    return membership.view(graph_count, seq_len)


def _fill_isolated_vertices(membership: torch.Tensor) -> torch.Tensor:
    missing = membership == -1
    next_ids = membership.amax(dim=1) + 1
    offsets = missing.to(torch.int32).cumsum(dim=1) - 1
    return torch.where(missing, next_ids.unsqueeze(1) + offsets, membership)


def _dense_remap(membership: torch.Tensor) -> torch.Tensor:
    sorted_values, order = membership.sort(dim=-1)
    transitions = torch.ones_like(sorted_values, dtype=torch.int32)
    if sorted_values.shape[-1] > 1:
        transitions[:, 1:] = (
            sorted_values[:, 1:] != sorted_values[:, :-1]
        ).to(torch.int32)
    sorted_dense = transitions.cumsum(dim=-1) - 1
    dense = torch.empty_like(membership, dtype=torch.int32)
    dense.scatter_(1, order, sorted_dense.to(torch.int32))
    return dense


def partition_graphs(
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    *,
    aggregation: GraphAggregation,
    num_kv_heads: int,
    prefill_seq_len: int,
    num_sink: int,
    lam: float,
    leiden_resolution: float,
    leiden_seed: int,
) -> PartitionResult:
    aggregation = GraphAggregation(aggregation)
    grouped_indices, grouped_scores, query_positions = _reshape_by_aggregation(
        topk_indices,
        topk_scores,
        aggregation=aggregation,
        num_kv_heads=num_kv_heads,
        num_sink=num_sink,
    )
    graph_count = grouped_indices.shape[0]
    edge_src, edge_dst, edge_weight = build_adjacency(
        grouped_indices,
        grouped_scores,
        query_positions,
        seq_len=prefill_seq_len,
        lam=lam,
    )
    if edge_src.numel():
        # Native Leiden allocates outside PyTorch's caching allocator.
        torch.cuda.empty_cache()
        vertex, labels, modularity = run_leiden(
            edge_src,
            edge_dst,
            edge_weight,
            G=graph_count,
            seq_len=prefill_seq_len,
            resolution=leiden_resolution,
            max_inner_iter=leiden_max_iterations(prefill_seq_len),
            seed=leiden_seed,
        )
        membership = _scatter_membership(
            vertex,
            labels,
            graph_count=graph_count,
            seq_len=prefill_seq_len,
        )
    else:
        modularity = 0.0
        membership = torch.full(
            (graph_count, prefill_seq_len),
            -1,
            dtype=torch.int32,
            device=topk_indices.device,
        )
    community_ids = _dense_remap(_fill_isolated_vertices(membership))
    community_counts = (community_ids.amax(dim=1) + 1).to(torch.int32)
    return PartitionResult(
        community_ids=community_ids,
        community_counts=community_counts,
        modularity=modularity,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_weight=edge_weight,
    )


def partition_query_groups(
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    *,
    num_kv_heads: int,
    prefill_seq_len: int,
    num_sink: int,
    lam: float,
    leiden_resolution: float,
    leiden_seed: int,
) -> PartitionResult:
    """Compatibility wrapper for query-group graph construction."""

    return partition_graphs(
        topk_indices,
        topk_scores,
        aggregation=GraphAggregation.QUERY_GROUP,
        num_kv_heads=num_kv_heads,
        prefill_seq_len=prefill_seq_len,
        num_sink=num_sink,
        lam=lam,
        leiden_resolution=leiden_resolution,
        leiden_seed=leiden_seed,
    )


def compute_centroids(
    community_ids: torch.Tensor,
    community_counts: torch.Tensor,
    keys: torch.Tensor,
    *,
    retrieval_to_graph: torch.Tensor | None = None,
    retrieval_to_kv: torch.Tensor | None = None,
    num_sink: int,
    max_decode_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one FP8 centroid table per retrieval head."""

    graph_count, seq_len = community_ids.shape
    if keys.ndim != 3 or keys.shape[1] != seq_len:
        raise ValueError("keys must have shape [KV heads, sequence, head dim]")
    if retrieval_to_graph is None or retrieval_to_kv is None:
        if keys.shape[0] != graph_count:
            raise ValueError("aggregation mappings are required when graph and KV counts differ")
        retrieval_to_graph = torch.arange(
            graph_count,
            dtype=torch.int32,
            device=community_ids.device,
        )
        retrieval_to_kv = retrieval_to_graph
    if (
        retrieval_to_graph.ndim != 1
        or retrieval_to_kv.shape != retrieval_to_graph.shape
        or retrieval_to_graph.dtype != torch.int32
        or retrieval_to_kv.dtype != torch.int32
    ):
        raise TypeError("aggregation mappings must be matching int32 vectors")
    if (
        retrieval_to_graph.device != community_ids.device
        or retrieval_to_kv.device != community_ids.device
    ):
        raise ValueError("aggregation mappings must share the partition device")
    if retrieval_to_graph.numel() == 0:
        raise ValueError("at least one retrieval head is required")
    if (
        int(retrieval_to_graph.min()) < 0
        or int(retrieval_to_graph.max()) >= graph_count
        or int(retrieval_to_kv.min()) < 0
        or int(retrieval_to_kv.max()) >= keys.shape[0]
    ):
        raise ValueError("aggregation mapping index is out of bounds")
    retrieval_head_count = retrieval_to_graph.numel()
    active_capacity = int(community_counts.max().item())
    capacity = active_capacity + max_decode_tokens
    head_dim = keys.shape[-1]
    centroids = torch.zeros(
        (retrieval_head_count, capacity, head_dim),
        dtype=torch.float32,
        device=keys.device,
    )
    sizes = torch.zeros(
        (retrieval_head_count, capacity),
        dtype=torch.int32,
        device=keys.device,
    )
    ids = community_ids[retrieval_to_graph.long(), num_sink:].long()
    source = keys[retrieval_to_kv.long(), num_sink:].float()
    centroids.scatter_add_(
        1,
        ids.unsqueeze(-1).expand(-1, -1, head_dim),
        source,
    )
    sizes.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.int32))
    centroids[:, :active_capacity].div_(
        sizes[:, :active_capacity].float().clamp_min(1).unsqueeze(-1)
    )
    limit = torch.finfo(torch.float8_e4m3fn).max
    centroids.clamp_(min=-limit, max=limit)
    return centroids.to(torch.float8_e4m3fn), sizes


def build_member_csr(
    community_ids: torch.Tensor,
    community_counts: torch.Tensor,
    *,
    num_sink: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build recency-descending immutable prefill membership."""

    graph_count, seq_len = community_ids.shape
    max_communities = int(community_counts.max().item())
    ids = community_ids[:, num_sink:]
    positions = torch.arange(
        num_sink,
        seq_len,
        dtype=torch.int64,
        device=community_ids.device,
    ).unsqueeze(0).expand(graph_count, -1)
    sort_key = ids.long() * seq_len + (seq_len - 1 - positions)
    member_positions = torch.gather(
        positions,
        1,
        sort_key.argsort(dim=1),
    ).to(torch.int32)
    counts = torch.zeros(
        (graph_count, max_communities),
        dtype=torch.int64,
        device=community_ids.device,
    )
    counts.scatter_add_(1, ids.long(), torch.ones_like(ids, dtype=torch.int64))
    offsets = torch.zeros(
        (graph_count, max_communities + 1),
        dtype=torch.int32,
        device=community_ids.device,
    )
    offsets[:, 1:] = counts.cumsum(dim=1).to(torch.int32)
    return offsets, member_positions


def init_modularity_state(
    result: PartitionResult,
    *,
    community_capacity: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    graph_count = result.community_ids.shape[0]
    community_weight = torch.zeros(
        (graph_count, community_capacity),
        dtype=torch.float32,
        device=result.community_ids.device,
    )
    total_weight = torch.zeros(
        (graph_count,),
        dtype=torch.float32,
        device=result.community_ids.device,
    )
    if not result.edge_src.numel():
        return community_weight, total_weight

    source = result.edge_src.long()
    destination = result.edge_dst.long()
    graph = source // seq_len
    source_community = result.community_ids[graph, source % seq_len].long()
    destination_community = result.community_ids[
        graph, destination % seq_len
    ].long()
    weight = result.edge_weight.float()
    flat = community_weight.view(-1)
    flat.scatter_add_(
        0,
        graph * community_capacity + source_community,
        weight,
    )
    flat.scatter_add_(
        0,
        graph * community_capacity + destination_community,
        weight,
    )
    total_weight.scatter_add_(0, graph, 2.0 * weight)
    return community_weight, total_weight


def build_partitioned_layer(
    *,
    layer_idx: int,
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    keys: torch.Tensor,
    num_sink: int,
    lam: float,
    leiden_resolution: float,
    leiden_seed: int,
    max_decode_tokens: int,
    aggregation: GraphAggregation = GraphAggregation.PER_QUERY_HEAD,
) -> PartitionedLayer:
    """Execute the full prefill partition and construct required decode state."""

    if max_decode_tokens <= 0:
        raise ValueError("max_decode_tokens must be positive")
    num_kv_heads, prefill_seq_len, _ = keys.shape
    query_heads = topk_indices.shape[0]
    aggregation = GraphAggregation(aggregation)
    retrieval_to_graph, retrieval_to_kv = aggregation_mappings(
        aggregation,
        query_heads=query_heads,
        kv_heads=num_kv_heads,
        device=keys.device,
    )
    result = partition_graphs(
        topk_indices,
        topk_scores,
        aggregation=aggregation,
        num_kv_heads=num_kv_heads,
        prefill_seq_len=prefill_seq_len,
        num_sink=num_sink,
        lam=lam,
        leiden_resolution=leiden_resolution,
        leiden_seed=leiden_seed,
    )
    centroids, community_sizes = compute_centroids(
        result.community_ids,
        result.community_counts,
        keys,
        retrieval_to_graph=retrieval_to_graph,
        retrieval_to_kv=retrieval_to_kv,
        num_sink=num_sink,
        max_decode_tokens=max_decode_tokens,
    )
    member_offsets, member_positions = build_member_csr(
        result.community_ids,
        result.community_counts,
        num_sink=num_sink,
    )
    community_capacity = centroids.shape[1]
    community_weight, total_weight = init_modularity_state(
        result,
        community_capacity=community_capacity,
        seq_len=prefill_seq_len,
    )
    token_communities = torch.full(
        (result.community_ids.shape[0], prefill_seq_len + max_decode_tokens),
        -1,
        dtype=torch.int32,
        device=keys.device,
    )
    token_communities[:, :prefill_seq_len].copy_(result.community_ids)
    return PartitionedLayer(
        layer_idx=layer_idx,
        prefill_seq_len=prefill_seq_len,
        aggregation=aggregation,
        query_heads=query_heads,
        kv_heads=num_kv_heads,
        retrieval_to_graph=retrieval_to_graph,
        retrieval_to_kv=retrieval_to_kv,
        member_offsets=member_offsets,
        member_positions=member_positions,
        community_counts=result.community_counts,
        centroids=centroids,
        community_sizes=community_sizes,
        community_weight=community_weight,
        total_weight=total_weight,
        token_communities=token_communities,
    )


__all__ = [
    "PartitionResult",
    "PartitionedLayer",
    "_reshape_by_aggregation",
    "aggregation_mappings",
    "build_adjacency",
    "build_member_csr",
    "build_partitioned_layer",
    "compute_centroids",
    "init_modularity_state",
    "leiden_max_iterations",
    "partition_graphs",
    "partition_query_groups",
]
