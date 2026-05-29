"""Async workers that mutate ``GraphRuntime`` state.

Three entry points, each runs on a thread pool worker:
  * ``async_partition_leiden`` — initial prefill-time partition; populates
    ``graph_runtime.graphs[layer_idx]`` with a fresh ``LayerGraph``.
  * ``decode_step_update`` — synchronous (called from the attention forward,
    not the executor) per-step graph update on a single ``LayerGraph``.
  * ``async_repartition_leiden`` — periodic decode-time repartition; returns
    a new ``LayerGraph`` that the main thread swaps in atomically.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from community_kv.graph._leiden import run_leiden
from community_kv.graph.partition import (
    dense_remap_per_graph,
    fill_isolated_vertices,
    partition,
    scatter_membership,
)
from community_kv.graph.state import GraphAggregation, LayerGraph, PartitionRecord

if TYPE_CHECKING:
    from community_kv.graph.runtime import GraphRuntime


def compute_centroids(
    community_ids: torch.Tensor,
    num_communities: torch.Tensor,
    keys: torch.Tensor,
    num_kv_heads: int,
    max_new_tokens: int,
    num_sink: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean key per community (fp32) + community size per centroid head.

    Sinks ([0, num_sink)) are excluded from the scatter; their singleton
    communities stay at -inf / size 0 so the decode-time nan_to_num naturally
    excludes them. Headroom is ``max_new_tokens - 1`` (the first generated
    token comes from prefill logits and writes no KV).
    """
    G, S = community_ids.shape
    H_kv = keys.shape[0]
    D = keys.shape[-1]
    assert H_kv == num_kv_heads, f"H_kv mismatch: keys={H_kv} num_kv_heads={num_kv_heads}"
    num_centroid_heads = max(G, num_kv_heads)
    num_communities_max = int(num_communities.max().item())
    decode_headroom = max(max_new_tokens - 1, 0)
    max_C = num_communities_max + decode_headroom

    centroids = torch.full(
        (num_centroid_heads, max_C, D),
        float("-inf"),
        dtype=torch.float32,
        device=keys.device,
    )
    centroids[:, :num_communities_max].zero_()
    sizes = torch.zeros(
        (num_centroid_heads, max_C),
        dtype=torch.float32,
        device=keys.device,
    )

    I_full = community_ids.long()
    I = I_full[:, num_sink:]
    K = keys[:, num_sink:].to(torch.float32)
    S_eff = S - num_sink
    if G < num_kv_heads:
        I = I.expand(num_centroid_heads, -1)
    elif G > num_kv_heads:
        factor = G // num_kv_heads
        K = (
            K.unsqueeze(1)
            .expand(-1, factor, -1, -1)
            .contiguous()
            .view(
                num_centroid_heads,
                S_eff,
                D,
            )
        )

    I_expanded = I.unsqueeze(-1).expand(num_centroid_heads, S_eff, D)
    centroids.scatter_add_(1, I_expanded, K)
    sizes.scatter_add_(
        1, I, torch.ones((num_centroid_heads, S_eff), dtype=torch.float32, device=keys.device)
    )
    centroids[:, :num_communities_max] /= sizes[:, :num_communities_max].unsqueeze(-1).clamp(min=1)
    empty_in_prefill = (sizes[:, :num_communities_max] == 0).unsqueeze(-1)
    centroids[:, :num_communities_max] = torch.where(
        empty_in_prefill,
        torch.full_like(centroids[:, :num_communities_max], float("-inf")),
        centroids[:, :num_communities_max],
    )
    return centroids, sizes


def build_member_csr(
    community_ids_prefill: torch.Tensor,
    num_communities: torch.Tensor,
    num_sink: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CSR map from (graph, community) -> prefill positions, recency-sorted."""
    G, S = community_ids_prefill.shape
    device = community_ids_prefill.device
    max_C = int(num_communities.max().item())

    cids = community_ids_prefill[:, num_sink:]
    S_eff = S - num_sink
    pos = torch.arange(num_sink, S, device=device, dtype=torch.long)
    pos_g = pos.unsqueeze(0).expand(G, -1)
    key = cids.long() * S + (S - 1 - pos_g)
    sort_idx = key.argsort(dim=-1)
    member_positions = torch.gather(pos_g, 1, sort_idx).to(torch.int32)

    counts = torch.zeros((G, max_C), dtype=torch.long, device=device)
    counts.scatter_add_(
        1,
        cids.long(),
        torch.ones((G, S_eff), dtype=torch.long, device=device),
    )
    member_offsets = torch.zeros((G, max_C + 1), dtype=torch.int32, device=device)
    member_offsets[:, 1:] = counts.cumsum(dim=-1).to(torch.int32)
    return member_offsets, member_positions


def init_modularity_state(
    community_ids: torch.Tensor,
    num_communities: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_weight: torch.Tensor,
    seq_len: int,
    max_C: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (community_weight, total_weight) from a freshly-partitioned graph."""
    G = community_ids.shape[0]
    device = community_ids.device
    community_weight = torch.zeros((G, max_C), dtype=torch.float32, device=device)
    total_weight = torch.zeros((G,), dtype=torch.float32, device=device)
    if edge_src.numel() == 0:
        return community_weight, total_weight

    src_g = edge_src.long() // seq_len
    src_local = edge_src.long() % seq_len
    dst_local = edge_dst.long() % seq_len
    src_comm = community_ids[src_g, src_local].long()
    dst_comm = community_ids[src_g, dst_local].long()
    w = edge_weight.float()

    flat_idx = src_g * max_C + src_comm
    community_weight.view(-1).scatter_add_(0, flat_idx, w)
    flat_idx = src_g * max_C + dst_comm
    community_weight.view(-1).scatter_add_(0, flat_idx, w)

    total_weight.scatter_add_(0, src_g, 2.0 * w)
    return community_weight, total_weight


def _reencode_edges(
    src: torch.Tensor,
    dst: torch.Tensor,
    old_seq_len: int,
    new_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-encode global vertex IDs after extending the per-graph seq_len.

    Each global ID was ``g * old_seq_len + local``; after the extension it
    must become ``g * new_seq_len + local``. Returns int32 tensors.
    """
    src_l = src.long()
    dst_l = dst.long()
    return (
        ((src_l // old_seq_len) * new_seq_len + (src_l % old_seq_len)).to(torch.int32),
        ((dst_l // old_seq_len) * new_seq_len + (dst_l % old_seq_len)).to(torch.int32),
    )


def async_partition_leiden(
    graph_runtime: "GraphRuntime",
    layer_idx: int,
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    keys: torch.Tensor,
    H_kv: int,
    S_k: int,
    num_sink: int,
    lam: float,
    leiden_resolution: float,
    leiden_max_iter: int,
    max_new_tokens: int,
    aggregation: GraphAggregation,
    fwd_device: torch.device,
    part_device: torch.device,
    attn_done_event: torch.cuda.Event,
) -> PartitionRecord:
    t_submit = time.perf_counter()
    attn_done_event.synchronize()
    sem = graph_runtime.gpu_semaphores.get(str(part_device))
    if sem is not None:
        sem.acquire()
    try:
        with torch.cuda.device(part_device):
            t_launch = time.perf_counter()
            result = partition(
                topk_indices,
                topk_scores,
                aggregation=aggregation,
                num_kv_heads=H_kv,
                prefill_seq_len=S_k,
                num_sink=num_sink,
                lam=lam,
                leiden_resolution=leiden_resolution,
                leiden_max_iter=leiden_max_iter,
            )
            torch.cuda.synchronize(part_device)
            t_partition_done = time.perf_counter()

            centroids, community_sizes = compute_centroids(
                result.community_ids,
                result.num_communities,
                keys,
                num_kv_heads=H_kv,
                max_new_tokens=max_new_tokens,
                num_sink=num_sink,
            )
            torch.cuda.synchronize(part_device)
            t_centroid_done = time.perf_counter()

            member_offsets, member_positions = build_member_csr(
                result.community_ids,
                result.num_communities,
                num_sink=num_sink,
            )
            G = result.community_ids.shape[0]
            decode_headroom = max(max_new_tokens - 1, 0)
            community_ids_full = torch.full(
                (G, S_k + decode_headroom),
                -1,
                dtype=result.community_ids.dtype,
                device=result.community_ids.device,
            )
            community_ids_full[:, :S_k] = result.community_ids

            max_C = centroids.shape[1]
            community_weight, total_weight = init_modularity_state(
                result.community_ids,
                result.num_communities,
                result.edge_src,
                result.edge_dst,
                result.edge_weight,
                seq_len=S_k,
                max_C=max_C,
            )

            decode_headroom_log = max(max_new_tokens - 1, 0)
            decode_log_position = torch.full(
                (decode_headroom_log,),
                -1,
                dtype=torch.int32,
                device=part_device,
            )
            decode_log_community = torch.full(
                (G, decode_headroom_log),
                -1,
                dtype=torch.int32,
                device=part_device,
            )
            kappa = topk_indices.shape[-1]
            edges_per_step = G * (kappa + (kappa * (kappa - 1)) // 2)
            decode_edge_capacity = edges_per_step * decode_headroom_log + 1
            decode_edge_graph = torch.zeros(
                (decode_edge_capacity,),
                dtype=torch.int32,
                device=part_device,
            )
            decode_edge_src_pos = torch.zeros(
                (decode_edge_capacity,),
                dtype=torch.int32,
                device=part_device,
            )
            decode_edge_dst_pos = torch.zeros(
                (decode_edge_capacity,),
                dtype=torch.int32,
                device=part_device,
            )
            decode_edge_weight = torch.zeros(
                (decode_edge_capacity,),
                dtype=torch.float32,
                device=part_device,
            )

            torch.cuda.synchronize(part_device)
            t_end = time.perf_counter()
    finally:
        if sem is not None:
            sem.release()

    graph_runtime.graphs[layer_idx] = LayerGraph(
        layer_idx=layer_idx,
        aggregation=aggregation,
        num_kv_heads_local=H_kv,
        prefill_seq_len=S_k,
        head_dim=keys.shape[-1],
        device=keys.device,
        community_ids=community_ids_full,
        num_communities=result.num_communities,
        centroids=centroids,
        community_sizes=community_sizes,
        community_sizes_prefill=community_sizes.clone(),
        member_offsets=member_offsets,
        member_positions=member_positions,
        community_weight=community_weight,
        total_weight=total_weight,
        decode_log_position=decode_log_position,
        decode_log_community=decode_log_community,
        decode_log_size=0,
        prefill_edge_src=result.edge_src.to(torch.int32),
        prefill_edge_dst=result.edge_dst.to(torch.int32),
        prefill_edge_weight=result.edge_weight.to(torch.float32),
        decode_edge_graph=decode_edge_graph,
        decode_edge_src_pos=decode_edge_src_pos,
        decode_edge_dst_pos=decode_edge_dst_pos,
        decode_edge_weight=decode_edge_weight,
        decode_edge_size=0,
        version=0,
    )

    return PartitionRecord(
        layer_idx=layer_idx,
        graph_idx=0,
        fwd_device=str(fwd_device),
        part_device=str(part_device),
        start=t_submit,
        launch=t_launch,
        end=t_end,
        elapsed_ms=(t_end - t_submit) * 1000.0,
        kernel_ms=(t_partition_done - t_launch) * 1000.0,
        centroid_ms=(t_centroid_done - t_partition_done) * 1000.0,
        csr_ms=(t_end - t_centroid_done) * 1000.0,
        n_edges=int(result.edge_src.numel()),
        num_communities_mean=float(result.num_communities.float().mean().item()),
        num_communities_max=int(result.num_communities.max().item()),
        modularity=result.modularity,
    )


def decode_step_update(
    graph: LayerGraph,
    topk_local_scores: torch.Tensor,
    topk_local_idx: torch.Tensor,
    retrieved: torch.Tensor,
    new_key: torch.Tensor,
    current_pos: int,
    H_q: int,
    H_kv: int,
    G: int,
    lam: float,
    kappa: int,
) -> None:
    """Per-decode-step graph update: edges + greedy assignment + state mutation."""
    device = new_key.device
    max_C = graph.community_weight.shape[1]
    num_centroid_heads = graph.centroids.shape[0]

    topk_global = torch.gather(retrieved, 1, topk_local_idx.long())

    heads_per_group = H_q // G
    rep_heads = torch.arange(0, H_q, heads_per_group, device=device)
    g_topk_idx = topk_global[rep_heads].long().clamp(min=-1, max=current_pos)
    g_topk_scores = topk_local_scores[rep_heads].float()

    valid = g_topk_idx >= 0
    w1 = g_topk_scores * (lam / 2.0) * valid.float()
    node_degrees = w1.sum(dim=1)

    if lam < 1.0:
        outer = g_topk_scores.unsqueeze(2) * g_topk_scores.unsqueeze(1)
        valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)
        upper = torch.triu(
            torch.ones(kappa, kappa, dtype=torch.bool, device=device),
            diagonal=1,
        )
        w2_outer = outer * (1.0 - lam) * valid_pair.float() * upper.float()
        w2_sum = w2_outer.flatten(1).sum(dim=1)
    else:
        w2_outer = torch.zeros(G, kappa, kappa, dtype=torch.float32, device=device)
        w2_sum = torch.zeros(G, dtype=torch.float32, device=device)

    new_edge_totals = node_degrees + w2_sum

    safe_topk_idx = g_topk_idx.clamp(min=0)
    neighbor_comms = torch.gather(graph.community_ids, 1, safe_topk_idx).long()
    neighbor_valid = valid & (neighbor_comms >= 0)
    w1_masked = w1 * neighbor_valid.float()
    safe_neighbor_comms = neighbor_comms.clamp(min=0)
    w_ic = torch.zeros(G, max_C, dtype=torch.float32, device=device)
    w_ic.scatter_add_(1, safe_neighbor_comms, w1_masked)

    two_m = (2.0 * graph.total_weight).clamp(min=1.0)
    delta_q = w_ic - node_degrees.unsqueeze(1) * graph.community_weight / two_m.unsqueeze(1)
    delta_q = torch.where(w_ic > 0, delta_q, torch.full_like(delta_q, float("-inf")))
    best_delta_q, best_comm = delta_q.max(dim=1)
    join_mask = best_delta_q > 0
    new_singleton_id = graph.num_communities.long()
    assigned_comm = torch.where(join_mask, best_comm, new_singleton_id)

    ar_g = torch.arange(G, device=device)
    graph.community_ids[:, current_pos] = assigned_comm.to(graph.community_ids.dtype)

    if num_centroid_heads == G:
        sizes_assigned = assigned_comm
        ch_index = ar_g
        kv_h_index = ar_g if H_kv == G else ar_g // (G // H_kv)
    elif num_centroid_heads > G:
        sizes_assigned = assigned_comm.expand(num_centroid_heads)
        ch_index = torch.arange(num_centroid_heads, device=device)
        kv_h_index = ch_index.clamp(max=H_kv - 1)
    else:
        sizes_assigned = assigned_comm[:num_centroid_heads]
        ch_index = torch.arange(num_centroid_heads, device=device)
        kv_h_index = ch_index

    graph.community_sizes.scatter_add_(
        1,
        sizes_assigned.unsqueeze(1),
        torch.ones(num_centroid_heads, 1, dtype=graph.community_sizes.dtype, device=device),
    )
    graph.community_weight.scatter_add_(1, safe_neighbor_comms, w1_masked)
    graph.community_weight.scatter_add_(
        1,
        assigned_comm.unsqueeze(1),
        node_degrees.unsqueeze(1),
    )
    graph.num_communities.add_((~join_mask).to(graph.num_communities.dtype))
    graph.total_weight.add_(new_edge_totals)

    new_size = graph.community_sizes[ch_index, sizes_assigned]
    old_size = (new_size - 1).clamp(min=0)
    old_centroid = graph.centroids[ch_index, sizes_assigned]
    new_key_per_ch = new_key[kv_h_index]
    old_size_f = old_size.to(graph.centroids.dtype).unsqueeze(1)
    new_size_f = new_size.to(graph.centroids.dtype).unsqueeze(1).clamp(min=1)
    is_join = (old_size > 0).unsqueeze(1)
    averaged = (old_centroid * old_size_f + new_key_per_ch) / new_size_f
    graph.centroids[ch_index, sizes_assigned] = torch.where(
        is_join,
        averaged,
        new_key_per_ch,
    )

    log_idx = graph.decode_log_size
    if log_idx < graph.decode_log_position.shape[0]:
        graph.decode_log_position[log_idx] = current_pos
        graph.decode_log_community[:, log_idx] = assigned_comm.to(torch.int32)
        graph.decode_log_size = log_idx + 1

    edge_size_now = graph.decode_edge_size
    cap = graph.decode_edge_graph.shape[0]
    n_direct = G * kappa
    if edge_size_now + n_direct <= cap:
        gs = ar_g.unsqueeze(1).expand(G, kappa)
        graph_flat = gs.to(torch.int32).flatten()
        src_pos_flat = torch.full(
            (n_direct,),
            current_pos,
            dtype=torch.int32,
            device=device,
        )
        dst_pos_flat = g_topk_idx.clamp(min=0).to(torch.int32).flatten()
        w_flat = (w1 * valid.float()).flatten()
        sl = slice(edge_size_now, edge_size_now + n_direct)
        graph.decode_edge_graph[sl] = graph_flat
        graph.decode_edge_src_pos[sl] = src_pos_flat
        graph.decode_edge_dst_pos[sl] = dst_pos_flat
        graph.decode_edge_weight[sl] = w_flat
        edge_size_now += n_direct

    if lam < 1.0:
        triu_i, triu_j = torch.triu_indices(kappa, kappa, offset=1, device=device)
        n_pairs = triu_i.shape[0]
        n_pair_edges = G * n_pairs
        if edge_size_now + n_pair_edges <= cap and n_pair_edges > 0:
            src_pos = g_topk_idx.clamp(min=0)[:, triu_i]
            dst_pos = g_topk_idx.clamp(min=0)[:, triu_j]
            valid_pair_flat = (valid[:, triu_i] & valid[:, triu_j]).float()
            pair_weights = w2_outer[:, triu_i, triu_j] * valid_pair_flat
            gs = ar_g.unsqueeze(1).expand(G, n_pairs)
            sl = slice(edge_size_now, edge_size_now + n_pair_edges)
            graph.decode_edge_graph[sl] = gs.to(torch.int32).flatten()
            graph.decode_edge_src_pos[sl] = src_pos.to(torch.int32).flatten()
            graph.decode_edge_dst_pos[sl] = dst_pos.to(torch.int32).flatten()
            graph.decode_edge_weight[sl] = pair_weights.flatten()
            edge_size_now += n_pair_edges

    graph.decode_edge_size = edge_size_now


def async_repartition_leiden(
    layer_idx: int,
    src_graph: LayerGraph,
    snap_log_size: int,
    snap_edge_size: int,
    keys_snap: torch.Tensor,
    leiden_resolution: float,
    leiden_max_iter: int,
    num_sink: int,
    max_new_tokens: int,
    aggregation: GraphAggregation,
    kappa: int,
    part_device: torch.device,
) -> tuple[LayerGraph, dict]:
    """Re-run Leiden on prefill + decode-log edges; build a fresh LayerGraph
    that the main thread atomically swaps into ``graph_runtime.graphs`` at a safe point."""
    t_start = time.perf_counter()
    G = src_graph.community_ids.shape[0]
    H_kv = keys_snap.shape[0]
    old_prefill_S = src_graph.prefill_seq_len
    new_prefill_S = old_prefill_S + snap_log_size

    with torch.cuda.device(part_device):
        prefill_src_new, prefill_dst_new = _reencode_edges(
            src_graph.prefill_edge_src,
            src_graph.prefill_edge_dst,
            old_prefill_S,
            new_prefill_S,
        )
        if snap_edge_size > 0:
            g_vec = src_graph.decode_edge_graph[:snap_edge_size].long()
            sp = src_graph.decode_edge_src_pos[:snap_edge_size].long()
            dp = src_graph.decode_edge_dst_pos[:snap_edge_size].long()
            dw = src_graph.decode_edge_weight[:snap_edge_size]
            sp_canon = torch.minimum(sp, dp)
            dp_canon = torch.maximum(sp, dp)
            decode_src_new = (g_vec * new_prefill_S + sp_canon).to(torch.int32)
            decode_dst_new = (g_vec * new_prefill_S + dp_canon).to(torch.int32)
            all_src = torch.cat([prefill_src_new, decode_src_new])
            all_dst = torch.cat([prefill_dst_new, decode_dst_new])
            all_w = torch.cat([src_graph.prefill_edge_weight, dw])
        else:
            all_src = prefill_src_new
            all_dst = prefill_dst_new
            all_w = src_graph.prefill_edge_weight.clone()

        if all_src.numel() > 0:
            vertex, leiden_partition, modularity = run_leiden(
                all_src,
                all_dst,
                all_w,
                G=G,
                seq_len=new_prefill_S,
                resolution=leiden_resolution,
            )
            membership_per_graph = scatter_membership(
                vertex,
                leiden_partition,
                G,
                new_prefill_S,
                part_device,
            )
        else:
            modularity = 0.0
            membership_per_graph = torch.full(
                (G, new_prefill_S),
                -1,
                dtype=torch.int32,
                device=part_device,
            )

        membership_filled = fill_isolated_vertices(membership_per_graph)
        community_ids_new = dense_remap_per_graph(membership_filled)
        num_communities_new = (community_ids_new.max(dim=-1).values + 1).to(torch.int32)

        keys_for_build = keys_snap[:, :new_prefill_S, :].contiguous()
        centroids_new, sizes_new = compute_centroids(
            community_ids_new,
            num_communities_new,
            keys_for_build,
            num_kv_heads=H_kv,
            max_new_tokens=max_new_tokens,
            num_sink=num_sink,
        )
        member_offsets_new, member_positions_new = build_member_csr(
            community_ids_new,
            num_communities_new,
            num_sink=num_sink,
        )

        total_len = src_graph.community_ids.shape[1]
        community_ids_full = torch.full(
            (G, total_len),
            -1,
            dtype=community_ids_new.dtype,
            device=part_device,
        )
        community_ids_full[:, :new_prefill_S] = community_ids_new

        max_C = centroids_new.shape[1]
        community_weight_new, total_weight_new = init_modularity_state(
            community_ids_new,
            num_communities_new,
            all_src,
            all_dst,
            all_w,
            seq_len=new_prefill_S,
            max_C=max_C,
        )

        log_capacity = src_graph.decode_log_position.shape[0]
        decode_log_position = torch.full(
            (log_capacity,),
            -1,
            dtype=torch.int32,
            device=part_device,
        )
        decode_log_community = torch.full(
            (G, log_capacity),
            -1,
            dtype=torch.int32,
            device=part_device,
        )
        edges_per_step = G * (kappa + (kappa * (kappa - 1)) // 2)
        decode_edge_capacity = edges_per_step * log_capacity + 1
        decode_edge_graph = torch.zeros(
            (decode_edge_capacity,),
            dtype=torch.int32,
            device=part_device,
        )
        decode_edge_src_pos = torch.zeros(
            (decode_edge_capacity,),
            dtype=torch.int32,
            device=part_device,
        )
        decode_edge_dst_pos = torch.zeros(
            (decode_edge_capacity,),
            dtype=torch.int32,
            device=part_device,
        )
        decode_edge_weight = torch.zeros(
            (decode_edge_capacity,),
            dtype=torch.float32,
            device=part_device,
        )

        torch.cuda.synchronize(part_device)

    new_graph = LayerGraph(
        layer_idx=layer_idx,
        aggregation=aggregation,
        num_kv_heads_local=H_kv,
        prefill_seq_len=new_prefill_S,
        head_dim=src_graph.head_dim,
        device=part_device,
        community_ids=community_ids_full,
        num_communities=num_communities_new,
        centroids=centroids_new,
        community_sizes=sizes_new,
        community_sizes_prefill=sizes_new.clone(),
        member_offsets=member_offsets_new,
        member_positions=member_positions_new,
        community_weight=community_weight_new,
        total_weight=total_weight_new,
        decode_log_position=decode_log_position,
        decode_log_community=decode_log_community,
        decode_log_size=0,
        prefill_edge_src=all_src,
        prefill_edge_dst=all_dst,
        prefill_edge_weight=all_w,
        decode_edge_graph=decode_edge_graph,
        decode_edge_src_pos=decode_edge_src_pos,
        decode_edge_dst_pos=decode_edge_dst_pos,
        decode_edge_weight=decode_edge_weight,
        decode_edge_size=0,
        version=src_graph.version + 1,
    )

    t_end = time.perf_counter()
    record = {
        "layer_idx": layer_idx,
        "version": new_graph.version,
        "wall_ms": (t_end - t_start) * 1000.0,
        "modularity": float(modularity),
        "num_communities_max": int(num_communities_new.max().item()),
        "num_communities_mean": float(num_communities_new.float().mean().item()),
        "n_edges": int(all_src.numel()),
        "snap_log_size": snap_log_size,
        "new_prefill_seq_len": new_prefill_S,
    }
    return new_graph, record
