"""Self-contained topk -> Leiden partition pipeline.

Takes the FA fused-topk output for one layer, builds the per-graph
adjacency, runs Leiden on the batched edge list, and returns dense
per-graph community IDs along with the COO edge list and modularity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from community_kv.graph._leiden import run_leiden
from community_kv.graph.state import GraphAggregation


def build_adjacency_batched(
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    query_positions: torch.Tensor,
    seq_len: int,
    lam: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched adjacency build for G subgraphs at once.

    Vertex IDs are offset by ``g * seq_len`` so the G subgraphs are disjoint.
    Returns the upper-triangular COO triple (src <= dst).
    """
    G, M, kappa = topk_indices.shape
    device = topk_indices.device
    total_vertices = G * seq_len

    g_offset = torch.arange(G, device=device, dtype=torch.int64).unsqueeze(1) * seq_len

    w1_src = (query_positions.long() + g_offset).unsqueeze(-1).expand_as(topk_indices)
    w1_dst = topk_indices.long() + g_offset.unsqueeze(-1)
    w1_weight = topk_scores.float()

    valid1 = topk_indices >= 0
    w1_src = w1_src[valid1]
    w1_dst = w1_dst[valid1]
    w1_weight = w1_weight[valid1]

    if lam < 1.0:
        grid_i, grid_j = torch.meshgrid(
            torch.arange(kappa, device=device),
            torch.arange(kappa, device=device),
            indexing="ij",
        )
        upper_mask = grid_i <= grid_j
        col_i = grid_i[upper_mask]
        col_j = grid_j[upper_mask]

        w2_node_i = topk_indices[..., col_i]
        w2_node_j = topk_indices[..., col_j]
        w2_node_i_off = w2_node_i.long() + g_offset.unsqueeze(-1)
        w2_node_j_off = w2_node_j.long() + g_offset.unsqueeze(-1)
        w2_weight_all = topk_scores[..., col_i].float() * topk_scores[..., col_j].float()

        valid2 = (w2_node_i >= 0) & (w2_node_j >= 0)
        w2_src_v = w2_node_i_off[valid2]
        w2_dst_v = w2_node_j_off[valid2]
        w2_weight = w2_weight_all[valid2]
        w2_canonical_src = torch.minimum(w2_src_v, w2_dst_v)
        w2_canonical_dst = torch.maximum(w2_src_v, w2_dst_v)
    else:
        w2_canonical_src = torch.empty(0, dtype=torch.long, device=device)
        w2_canonical_dst = torch.empty(0, dtype=torch.long, device=device)
        w2_weight = torch.empty(0, dtype=torch.float32, device=device)

    w1_canonical_src = torch.minimum(w1_src, w1_dst)
    w1_canonical_dst = torch.maximum(w1_src, w1_dst)

    all_src = torch.cat([w1_canonical_src, w2_canonical_src])
    all_dst = torch.cat([w1_canonical_dst, w2_canonical_dst])
    all_weight = torch.cat([w1_weight * (lam / 2.0), w2_weight * (1.0 - lam)])

    edge_key = all_src * total_vertices + all_dst
    unique_keys, inverse = edge_key.unique(return_inverse=True)
    edge_weight = torch.zeros(unique_keys.shape[0], dtype=torch.float32, device=device)
    edge_weight.scatter_add_(0, inverse, all_weight)

    edge_src = (unique_keys // total_vertices).to(torch.int32)
    edge_dst = (unique_keys % total_vertices).to(torch.int32)

    return edge_src, edge_dst, edge_weight


@dataclass
class PartitionResult:
    community_ids: torch.Tensor
    num_communities: torch.Tensor
    modularity: float
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_weight: torch.Tensor


def _reshape_by_aggregation(
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    aggregation: GraphAggregation,
    num_kv_heads: int,
    num_sink: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    H_q, S_eligible, kappa = topk_indices.shape
    device = topk_indices.device
    G = aggregation.num_graphs_per_layer(H_q, num_kv_heads)
    base_positions = torch.arange(
        kappa - 1 + num_sink,
        kappa - 1 + num_sink + S_eligible,
        device=device,
        dtype=torch.int64,
    )

    if aggregation == GraphAggregation.PER_QUERY_HEAD:
        return topk_indices, topk_scores, base_positions.unsqueeze(0).expand(G, -1)
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
        bp = base_positions.repeat(heads_per_group).unsqueeze(0).expand(G, -1)
        return bi, bs, bp
    if aggregation == GraphAggregation.LAYER_WISE:
        return (
            topk_indices.reshape(1, H_q * S_eligible, kappa),
            topk_scores.reshape(1, H_q * S_eligible, kappa),
            base_positions.repeat(H_q).unsqueeze(0),
        )
    raise ValueError(f"Unknown aggregation: {aggregation}")


def scatter_membership(
    vertex: torch.Tensor,
    partition: torch.Tensor,
    G: int,
    prefill_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    membership_flat = torch.full(
        (G * prefill_seq_len,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    membership_flat[vertex] = partition
    return membership_flat.view(G, prefill_seq_len)


def fill_isolated_vertices(membership_per_graph: torch.Tensor) -> torch.Tensor:
    missing = membership_per_graph == -1
    per_graph_max = membership_per_graph.amax(dim=1)
    next_ids = per_graph_max + 1
    missing_int = missing.to(torch.int32)
    offset_within = missing_int.cumsum(dim=1).to(torch.int32) - 1
    new_ids = next_ids.unsqueeze(1) + offset_within
    return torch.where(missing, new_ids, membership_per_graph)


def dense_remap_per_graph(membership_filled: torch.Tensor) -> torch.Tensor:
    """Remap each row's labels to the dense range ``[0, num_unique[g])``.

    Vectorised: sort each row, detect transitions in the sorted view, cumsum
    the transition mask to assign dense ids, then scatter back to the original
    positions. No Python loop over graphs.
    """
    sorted_vals, sort_idx = membership_filled.sort(dim=-1)
    is_new = torch.ones_like(sorted_vals, dtype=torch.int32)
    if sorted_vals.shape[-1] > 1:
        is_new[..., 1:] = (sorted_vals[..., 1:] != sorted_vals[..., :-1]).to(torch.int32)
    sorted_dense = is_new.cumsum(dim=-1) - 1
    out = torch.empty_like(membership_filled, dtype=torch.int32)
    out.scatter_(-1, sort_idx, sorted_dense.to(torch.int32))
    return out


def partition(
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    *,
    aggregation: GraphAggregation,
    num_kv_heads: int,
    prefill_seq_len: int,
    num_sink: int,
    lam: float,
    leiden_resolution: float,
    leiden_max_iter: int,
) -> PartitionResult:
    """Run the topk -> Leiden -> dense per-graph community ids pipeline."""
    assert topk_indices.ndim == 3, f"expected (H_q, S_eligible, kappa), got {topk_indices.shape}"
    H_q = topk_indices.shape[0]
    device = topk_indices.device
    G = aggregation.num_graphs_per_layer(H_q, num_kv_heads)

    batched_indices, batched_scores, batched_q_positions = _reshape_by_aggregation(
        topk_indices,
        topk_scores,
        aggregation,
        num_kv_heads,
        num_sink,
    )

    edge_src, edge_dst, edge_weight = build_adjacency_batched(
        batched_indices,
        batched_scores,
        batched_q_positions,
        seq_len=prefill_seq_len,
        lam=lam,
    )

    if edge_src.numel() > 0:
        vertex, leiden_partition, modularity = run_leiden(
            edge_src,
            edge_dst,
            edge_weight,
            G=G,
            seq_len=prefill_seq_len,
            resolution=leiden_resolution,
        )
        membership_per_graph = scatter_membership(
            vertex,
            leiden_partition,
            G,
            prefill_seq_len,
            device,
        )
    else:
        modularity = 0.0
        membership_per_graph = torch.full(
            (G, prefill_seq_len),
            -1,
            dtype=torch.int32,
            device=device,
        )

    membership_filled = fill_isolated_vertices(membership_per_graph)
    community_ids = dense_remap_per_graph(membership_filled)
    num_communities = (community_ids.max(dim=-1).values + 1).to(torch.int32)

    return PartitionResult(
        community_ids=community_ids,
        num_communities=num_communities,
        modularity=modularity,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_weight=edge_weight,
    )
