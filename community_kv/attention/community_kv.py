"""CommunityKV attention impl + registration.

The forward dispatches on prefill vs decode and delegates to the
``_prefill`` / ``_decode`` helpers. Both paths use the patched
flash-attention forward (top-K probe enabled — see
:mod:`community_kv.attention.fused_attn_fwd_topk`) and read/write
per-sample state through the ``GraphRuntime`` instance held on
``CommunityKVAttention.graph_runtime``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from community_kv.attention.fused_attn_fwd_topk import attn_forward_topk
from community_kv.graph.runtime import GraphRuntime, cuda_event_pair
from community_kv.graph.state import LayerLog
from community_kv.graph.workers import async_partition_leiden, decode_step_update

COMMUNITY_KV_ATTN_IMPL = "COMMUNITY_KV_ATTN"


# --------------------------------------------------------------------------- #
# Forward dispatcher
# --------------------------------------------------------------------------- #


def community_kv_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    *,
    graph_runtime: GraphRuntime,
    **kwargs,
):
    """One impl, two paths gated on S_q. Prefill builds the layer graph
    asynchronously; decode retrieves community members, runs FA over the
    retrieved subset, and updates the graph in-line.
    """
    cfg = graph_runtime.config
    S_q = query.shape[2]
    S_k = key.shape[2]
    if S_q > 1 and S_k >= cfg["kappa"] + cfg["num_sink"]:
        return _prefill(module, query, key, value, scaling, graph_runtime=graph_runtime)
    return _decode(module, query, key, value, scaling, graph_runtime=graph_runtime)


# --------------------------------------------------------------------------- #
# Prefill path
# --------------------------------------------------------------------------- #


def _prefill(module, query, key, value, scaling, *, graph_runtime: GraphRuntime):
    """Patched FA forward + topk probe; partition fires async on a worker thread."""
    cfg = graph_runtime.config
    kappa = cfg["kappa"]
    num_sink = cfg["num_sink"]
    lam = cfg["lam"]
    leiden_max_iter = cfg["leiden_max_iter"]
    max_new_tokens = cfg["max_new_tokens"]
    layer_idx = module.layer_idx
    leiden_resolution = graph_runtime.resolutions.get(layer_idx, cfg["leiden_resolution"])

    H_kv = key.shape[1]
    S_k = key.shape[2]

    q_bshd = query.transpose(1, 2).contiguous()
    k_bshd = key.transpose(1, 2).contiguous()
    v_bshd = value.transpose(1, 2).contiguous()

    fwd_device = key.device
    with cuda_event_pair(fwd_device) as (ev_attn_start, ev_attn_end):
        fa_out = attn_forward_topk(
            q_bshd,
            k_bshd,
            v_bshd,
            softmax_scale=scaling,
            topk_K=kappa,
            exclude_sink_tokens=num_sink,
        )
        attn_output = fa_out[0]
        topk_scores_full = fa_out[4]
        topk_indices_full = fa_out[5]

    with torch.cuda.device(fwd_device):
        ev_attn_done = torch.cuda.Event(enable_timing=False)
        ti_full = topk_indices_full[0]
        ts_full = topk_scores_full[0]
        ev_attn_done.record()

    assert graph_runtime.executor is not None, "GraphRuntime.executor must be set before forward"
    fut = graph_runtime.executor.submit(
        async_partition_leiden,
        graph_runtime,
        layer_idx,
        ti_full,
        ts_full,
        key[0],
        H_kv,
        S_k,
        num_sink,
        lam,
        leiden_resolution,
        leiden_max_iter,
        max_new_tokens,
        graph_runtime.aggregation,
        fwd_device,
        fwd_device,
        ev_attn_done,
    )
    graph_runtime.futures.append(fut)

    graph_runtime.layer_log[layer_idx] = LayerLog(
        fwd_device=str(fwd_device),
        part_device=str(fwd_device),
        prefill_seq_len=S_k,
        ev_attn_start=ev_attn_start,
        ev_attn_end=ev_attn_end,
    )
    return attn_output, None


# --------------------------------------------------------------------------- #
# Decode path
# --------------------------------------------------------------------------- #


def _decode(module, query, key, value, scaling, *, graph_runtime: GraphRuntime):
    """Per-q-head retrieval over community centroids -> FA over the retrieved
    subset -> in-line graph update."""
    cfg = graph_runtime.config
    kappa = cfg["kappa"]
    num_sink = cfg["num_sink"]
    lam = cfg["lam"]
    token_budget = cfg["token_budget"]
    layer_idx = module.layer_idx

    B, H_q, S_q, D = query.shape
    H_kv = key.shape[1]
    S_k = key.shape[2]

    graph = graph_runtime.graphs.get(layer_idx)
    assert graph is not None, f"decode called before prefill populated layer {layer_idx}"

    if layer_idx in graph_runtime.repartition_trigger_pending:
        graph_runtime.repartition_key_snapshots[layer_idx] = key[0].detach().clone()

    G_graph = graph.member_offsets.shape[0]
    current_pos = S_k - 1
    log_size = int(graph.decode_log_size)
    # Cap the gather at the actual cache size: anything past S_k is either
    # unwritten decode-buffer slots or pure padding. With this cap, when
    # ``token_budget >= S_k`` we attend densely over the full cache; when
    # it's smaller, sparse retrieval kicks in unchanged.
    effective_budget = min(token_budget, S_k)
    retrieve_budget = max(effective_budget - num_sink - 1 - log_size, 0)
    post_sink_pool = effective_budget - num_sink

    retrieved, topk_comm, valid_mask, cumsum, K = _retrieve(
        graph,
        query,
        layer_idx,
        current_pos,
        num_sink,
        retrieve_budget,
        log_size,
        graph_runtime=graph_runtime,
    )
    _record_retrieval_stats(
        K,
        valid_mask,
        cumsum,
        retrieve_budget,
        H_q,
        graph_runtime=graph_runtime,
        layer_idx=layer_idx,
    )

    h_to_kv = torch.arange(H_q, device=query.device, dtype=torch.long) * H_kv // H_q
    hkv_b = h_to_kv.unsqueeze(1).expand(H_q, retrieved.shape[-1])
    gathered_K = key[0][hkv_b, retrieved]
    gathered_V = value[0][hkv_b, retrieved]

    q_bshd = query.transpose(1, 2).contiguous()
    k_bshd = gathered_K.unsqueeze(0).transpose(1, 2).contiguous()
    v_bshd = gathered_V.unsqueeze(0).transpose(1, 2).contiguous()

    # The patched FA-topK kernel only dispatches ``topk_K in {2,4,8,16,32}``
    # and silently skips the q-row when ``post_sink_pool < topk_K``.
    # When the cache is too small for an in-kernel top-K of size kappa,
    # run the same kernel with ``return_topk=False`` (plain FA, no probe)
    # and synthesize the graph-update inputs in Python from the
    # post-sink positions of the gather. Cheap because post_sink_pool is
    # small precisely when this branch fires.
    use_kernel_topk = post_sink_pool >= kappa
    fa_out = attn_forward_topk(
        q_bshd,
        k_bshd,
        v_bshd,
        softmax_scale=scaling,
        topk_K=kappa,
        exclude_sink_tokens=num_sink,
        return_topk=use_kernel_topk,
    )
    attn_output = fa_out[0]

    with cuda_event_pair(query.device) as update_pair:
        if use_kernel_topk:
            topk_local_scores = fa_out[4][0, :, 0, :].float()
            topk_local_idx = fa_out[5][0, :, 0, :]
            n_top_tokens = kappa
        else:
            # All post-sink positions become the "top tokens" for this step.
            # Indices into ``retrieved``: [num_sink, num_sink + post_sink_pool).
            n_top_tokens = post_sink_pool
            topk_local_idx = (
                torch.arange(
                    num_sink, num_sink + n_top_tokens, dtype=torch.int32, device=query.device
                )
                .unsqueeze(0)
                .expand(H_q, n_top_tokens)
                .contiguous()
            )
            # Scores: q · k_post_sink^T * scaling. Q is (1, H_q, 1, D);
            # gathered_K is (H_q, S_k, D). Compute in fp32 to match the kernel path.
            q_score = query[0, :, 0, :].float()  # (H_q, D)
            k_post = gathered_K[:, num_sink : num_sink + n_top_tokens, :].float()
            topk_local_scores = (q_score.unsqueeze(1) @ k_post.transpose(-1, -2)).squeeze(
                1
            ) * scaling
        decode_step_update(
            graph=graph,
            topk_local_scores=topk_local_scores,
            topk_local_idx=topk_local_idx,
            retrieved=retrieved,
            new_key=key[0, :, current_pos, :],
            current_pos=current_pos,
            H_q=H_q,
            H_kv=H_kv,
            G=G_graph,
            lam=lam,
            kappa=n_top_tokens,
        )
    graph_runtime.decode_update_events.setdefault(layer_idx, []).append(update_pair)

    return attn_output, None


def _retrieve(
    graph,
    query: torch.Tensor,
    layer_idx: int,
    current_pos: int,
    num_sink: int,
    retrieve_budget: int,
    log_size: int,
    *,
    graph_runtime: GraphRuntime,
):
    """Score communities, gather slots up to ``retrieve_budget``, splice in
    sinks + decode-log hits + the current position. Records the timing event.

    Returns ``(retrieved, topk_comm, valid_mask, cumsum, K)``: the gather-ready
    indices of shape (H_q, token_budget), plus the topk-community ids, valid
    mask, K, and cumulative-size tensor used by ``_record_retrieval_stats``.
    """
    cfg = graph_runtime.config
    token_budget = cfg["token_budget"]

    B, H_q, S_q, D = query.shape
    G_graph = graph.member_offsets.shape[0]
    num_centroid_heads = graph.centroids.shape[0]

    with cuda_event_pair(query.device) as retrieve_pair:
        head_map = torch.arange(H_q, device=query.device, dtype=torch.long) // (
            H_q // num_centroid_heads
        )
        head_centroids = graph.centroids[head_map]
        q_hsd = query[0]
        scores = torch.bmm(
            q_hsd.to(torch.float32),
            head_centroids.transpose(1, 2),
        ).squeeze(1)
        # A priori we know which centroid slots are invalid for retrieval:
        #   * Sink-singleton slots — sinks were excluded from the centroid
        #     scatter, so their slots have ``community_sizes_prefill == 0``
        #     and centroids = -inf in every dim.
        #   * Decode-headroom slots (the trailing ``max_new_tokens-1``
        #     slots reserved for new singletons created during decode) —
        #     same: prefill never wrote them, so size is 0.
        # Mask both groups by setting their scores to -inf BEFORE topk so
        # the kernel never picks them (would OOB on member_offsets[g, c]
        # since member_offsets only allocates ``max_C_prefill+1`` slots).
        sizes_per_q = graph.community_sizes_prefill[head_map]  # (H_q, max_C)
        scores = torch.where(
            sizes_per_q > 0,
            scores,
            torch.full_like(scores, float("-inf")),
        )
        # Catch any NaN that might leak through (e.g., 0/0 in a
        # degenerate centroid average).
        scores = scores.nan_to_num(
            nan=float("-inf"),
            posinf=float("-inf"),
        )

        max_C = scores.shape[-1]
        K = min(retrieve_budget, max_C) if retrieve_budget > 0 else 0
        graph_map = head_map * G_graph // num_centroid_heads

        if K > 0:
            _, topk_comm = torch.topk(scores, k=K, dim=-1)
            topk_comm = topk_comm.to(torch.long)

            sizes_per_q = graph.community_sizes_prefill[head_map].to(torch.long)
            sizes_topk = torch.gather(sizes_per_q, 1, topk_comm)
            cumsum = sizes_topk.cumsum(dim=-1)
            cumsum_excl = cumsum - sizes_topk

            out_idx = torch.arange(retrieve_budget, device=query.device, dtype=torch.long)
            out_idx_b = out_idx.unsqueeze(0).expand(H_q, retrieve_budget).contiguous()
            bucket_idx = torch.searchsorted(cumsum, out_idx_b, right=True)
            valid_mask = bucket_idx < K
            bucket_idx_safe = bucket_idx.clamp(max=K - 1)
            within = out_idx_b - torch.gather(cumsum_excl, 1, bucket_idx_safe)
            comm_id = torch.gather(topk_comm, 1, bucket_idx_safe)

            gmap_b = graph_map.unsqueeze(1).expand(H_q, retrieve_budget)
            offsets_for_q = graph.member_offsets.long()[gmap_b, comm_id]
            abs_pos = offsets_for_q + within
            retrieved_ret = graph.member_positions.long()[gmap_b, abs_pos]
            retrieved_ret = torch.where(
                valid_mask,
                retrieved_ret,
                torch.full_like(retrieved_ret, -1),
            )
        else:
            retrieved_ret = torch.empty((H_q, 0), dtype=torch.long, device=query.device)
            cumsum = torch.zeros((H_q, 1), dtype=torch.long, device=query.device)
            valid_mask = torch.zeros((H_q, 0), dtype=torch.bool, device=query.device)
            topk_comm = torch.empty((H_q, 0), dtype=torch.long, device=query.device)

        if log_size > 0:
            decode_comms_h = graph.decode_log_community[graph_map].long()[:, :log_size]
            decode_pos_h = (
                graph.decode_log_position[:log_size].long().unsqueeze(0).expand(H_q, log_size)
            )
            if K > 0:
                hit = (decode_comms_h.unsqueeze(-1) == topk_comm.unsqueeze(1)).any(-1)
                decode_to_include = torch.where(
                    hit,
                    decode_pos_h,
                    torch.full_like(decode_pos_h, -1),
                )
            else:
                decode_to_include = decode_pos_h
        else:
            decode_to_include = torch.empty(
                (H_q, 0),
                dtype=torch.long,
                device=query.device,
            )

        sink_positions = (
            torch.arange(
                num_sink,
                device=query.device,
                dtype=torch.long,
            )
            .unsqueeze(0)
            .expand(H_q, num_sink)
        )
        current_positions = torch.full(
            (H_q, 1),
            current_pos,
            dtype=torch.long,
            device=query.device,
        )
        retrieved = torch.cat(
            [sink_positions, decode_to_include, retrieved_ret, current_positions],
            dim=-1,
        )
        retrieved = torch.where(
            retrieved >= 0,
            retrieved,
            torch.full_like(retrieved, current_pos),
        )

    graph_runtime.decode_retrieve_events.setdefault(layer_idx, []).append(retrieve_pair)
    return retrieved, topk_comm, valid_mask, cumsum, K


def _record_retrieval_stats(
    K, valid_mask, cumsum, retrieve_budget, H_q, *, graph_runtime, layer_idx
):
    """Append per-call (N_communities_used, total_retrieved) into
    ``graph_runtime.decode_retrieve_n[layer_idx]``."""
    if K > 0:
        budget_t = torch.full(
            (H_q, 1),
            retrieve_budget,
            device=cumsum.device,
            dtype=cumsum.dtype,
        )
        boundary = torch.searchsorted(cumsum, budget_t).squeeze(-1)
        N = (boundary + 1).clamp(max=K)
        total_retrieved = valid_mask.sum(dim=-1, dtype=torch.long)
    else:
        device = cumsum.device
        N = torch.zeros((H_q,), dtype=torch.long, device=device)
        total_retrieved = torch.zeros((H_q,), dtype=torch.long, device=device)
    per_call = torch.stack([N, total_retrieved], dim=-1)
    graph_runtime.decode_retrieve_n.setdefault(layer_idx, []).append(per_call)


# --------------------------------------------------------------------------- #
# OOP handle + HF registration
# --------------------------------------------------------------------------- #


class CommunityKVAttention:
    """OOP handle around the CommunityKV attention impl.

    Each instance is bound to a ``GraphRuntime`` (per-sample mutable state).
    The forward delegates to the free function.
    """

    IMPL_NAME = COMMUNITY_KV_ATTN_IMPL

    def __init__(self, *, graph_runtime: GraphRuntime):
        self.graph_runtime = graph_runtime

    def forward(
        self, module, query, key, value, attention_mask, scaling, dropout: float = 0.0, **kwargs
    ):
        return community_kv_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            dropout=dropout,
            graph_runtime=self.graph_runtime,
            **kwargs,
        )

    def register(
        self,
        *,
        kappa: int,
        num_sink: int,
        lam: float,
        leiden_resolution: float,
        leiden_max_iter: int,
        max_new_tokens: int,
        token_budget: int,
    ) -> str:
        """Populate config in self.graph_runtime and register self.forward with HF.

        Returns the name to set on ``model.config._attn_implementation``.
        """
        cfg = self.graph_runtime.config
        cfg["kappa"] = kappa
        cfg["num_sink"] = num_sink
        cfg["lam"] = lam
        cfg["leiden_resolution"] = leiden_resolution
        cfg["leiden_max_iter"] = leiden_max_iter
        cfg["max_new_tokens"] = max_new_tokens
        cfg["token_budget"] = token_budget
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS[self.IMPL_NAME] = self.forward
        return self.IMPL_NAME
