"""Graph utilities: aggregation strategies and adjacency construction."""

from __future__ import annotations

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
        """Compute how many independent graphs this strategy produces per layer."""
        if self is GraphAggregation.PER_QUERY_HEAD:
            return num_query_heads
        if self is GraphAggregation.QUERY_GROUP:
            return num_kv_heads
        if self is GraphAggregation.LAYER_WISE:
            return 1
        raise ValueError(f"Unknown aggregation: {self}")


def build_adjacency(
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    seq_len: int,
    num_sink_tok_to_exclude: int = 0,
    lam: float = 0.5,
    query_offset: int | None = None,
    query_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build sparse symmetric adjacency from top-kappa data in COO format.

    Let alpha_tilde[m, k] = alpha[m, k] * 1(k in topk(m)) be the top-kappa-
    masked attention matrix. The adjacency is:

        w = (lam/2) * (alpha_tilde + alpha_tilde^T) + (1 - lam) * alpha_tilde^T @ alpha_tilde

    - Direct term (w1): symmetrized masked attention.
        w1[i, j] = alpha_tilde[i, j] + alpha_tilde[j, i]
    - Co-attention term (w2): Gram matrix of masked attention (includes diagonal).
        w2[i, j] = sum_m alpha[m, i] * alpha[m, j] * 1(i in topk(m)) * 1(j in topk(m))

    Output is the upper triangle (src <= dst); the implied dense matrix is symmetric.

    Args:
        topk_indices: (S_eligible, kappa) int32 — key indices per eligible query
        topk_scores: (S_eligible, kappa) float — post-softmax attention scores
        seq_len: total sequence length S
        num_sink_tok_to_exclude: number of sink tokens
        lam: mixing parameter in [0, 1]
        query_offset: starting query position (default: kappa - 1 + num_sink_tok_to_exclude)
        query_positions: (S_eligible,) explicit query positions per row.
            Overrides query_offset if provided.

    Returns:
        edge_src: (E,) int64 — source node indices (edge_src <= edge_dst)
        edge_dst: (E,) int64 — destination node indices
        edge_weight: (E,) float32 — edge weights
    """
    S_elig, kappa = topk_indices.shape
    device = topk_indices.device

    # Query positions
    if query_positions is not None:
        pass  # use as-is
    elif query_offset is not None:
        query_positions = torch.arange(query_offset, query_offset + S_elig, device=device)
    else:
        init_q_start = kappa - 1 + num_sink_tok_to_exclude
        query_positions = torch.arange(init_q_start, init_q_start + S_elig, device=device)

    # --- Direct attention graph w1 ---
    w1_src = query_positions.unsqueeze(1).expand_as(topk_indices).reshape(-1)
    w1_dst = topk_indices.reshape(-1).long()
    w1_weight = topk_scores.reshape(-1).float()

    # Filter invalid entries
    valid = w1_dst >= 0
    w1_src = w1_src[valid]
    w1_dst = w1_dst[valid]
    w1_weight = w1_weight[valid]

    # --- Co-attention graph w2 ---
    if lam < 1.0:
        grid_i, grid_j = torch.meshgrid(
            torch.arange(kappa, device=device),
            torch.arange(kappa, device=device),
            indexing="ij",
        )
        upper_mask = grid_i <= grid_j
        col_i = grid_i[upper_mask]
        col_j = grid_j[upper_mask]

        w2_node_i = topk_indices[:, col_i]
        w2_node_j = topk_indices[:, col_j]
        w2_pair_weight = topk_scores[:, col_i].float() * topk_scores[:, col_j].float()

        w2_src_all = w2_node_i.reshape(-1).long()
        w2_dst_all = w2_node_j.reshape(-1).long()
        w2_weight_all = w2_pair_weight.reshape(-1)

        valid2 = (w2_src_all >= 0) & (w2_dst_all >= 0)
        w2_src_all = w2_src_all[valid2]
        w2_dst_all = w2_dst_all[valid2]
        w2_weight_all = w2_weight_all[valid2]

        # Canonicalize: ensure src <= dst
        swap = w2_src_all > w2_dst_all
        w2_src_all[swap], w2_dst_all[swap] = w2_dst_all[swap].clone(), w2_src_all[swap].clone()
    else:
        w2_src_all = torch.empty(0, dtype=torch.long, device=device)
        w2_dst_all = torch.empty(0, dtype=torch.long, device=device)
        w2_weight_all = torch.empty(0, dtype=torch.float32, device=device)

    # --- Combine: w = (lam/2)(alpha_tilde + alpha_tilde^T) + (1-lam)*alpha_tilde^T@alpha_tilde ---
    w1_canonical_src = torch.minimum(w1_src, w1_dst)
    w1_canonical_dst = torch.maximum(w1_src, w1_dst)

    all_src = torch.cat([w1_canonical_src, w2_src_all])
    all_dst = torch.cat([w1_canonical_dst, w2_dst_all])
    all_weight = torch.cat([
        w1_weight * (lam / 2.0),
        w2_weight_all * (1.0 - lam),
    ])

    # Aggregate duplicate edges by summing weights
    edge_key = all_src * seq_len + all_dst
    unique_keys, inverse = edge_key.unique(return_inverse=True)
    edge_weight = torch.zeros(unique_keys.shape[0], dtype=torch.float32, device=device)
    edge_weight.scatter_add_(0, inverse, all_weight)

    edge_src = unique_keys // seq_len
    edge_dst = unique_keys % seq_len

    return edge_src, edge_dst, edge_weight
