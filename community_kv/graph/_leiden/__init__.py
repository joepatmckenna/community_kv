"""Batched Leiden — pure-CUDA implementation.

Multi-level orchestration (Louvain hierarchy) lives here in Python. Each
level: single-level Leiden on the current COO -> dense per-graph relabel
-> aggregate COO. The chain of relabelings is composed back to original
vertices to produce the final (vertex, partition) output.

The compiled extension lives next to this file as
``_community_kv_leiden.cpython-*.so`` and is built by the package's
top-level ``setup.py``.
"""

from __future__ import annotations

import functools
import os

import torch


@functools.cache
def _load_module():
    """Lazy import of the compiled extension, memoized after first call.
    Keeps the package importable on machines without the .so built."""
    from . import _community_kv_leiden as _m

    return _m


def _densify_per_graph(
    labels: torch.Tensor,
    G: int,
    seq_len: int,
) -> tuple[torch.Tensor, int]:
    """Per-graph dense relabel."""
    labels_2d = labels.view(G, seq_len)
    inverses: list[torch.Tensor] = []
    n_per_graph: list[int] = []
    for g in range(G):
        u, inv = torch.unique(labels_2d[g], return_inverse=True)
        inverses.append(inv.to(torch.int32))
        n_per_graph.append(u.numel())
    new_seq_len = max(n_per_graph)
    dense = torch.empty(G * seq_len, dtype=torch.int32, device=labels.device)
    for g in range(G):
        offset = g * new_seq_len
        dense[g * seq_len : (g + 1) * seq_len] = inverses[g] + offset
    return dense, int(new_seq_len)


def run_leiden(
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    G: int,
    seq_len: int,
    max_level: int = 6,
    resolution: float = 1.0,
    theta: float = 0.01,
    max_inner_iter: int = 16,
    seed: int = 0,
    use_boltzmann: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Run multi-level Leiden over G vertex-disjoint subgraphs.

    Refinement target selection defaults to greedy best-gain. Set
    ``use_boltzmann=True`` (or env ``COMMUNITY_KV_LEIDEN_BOLTZMANN=1``) to
    sample targets from the Boltzmann distribution at temperature ``theta``
    instead — ``theta`` is ignored in best-gain mode.
    """
    _module = _load_module()
    edge_src = edge_src.contiguous()
    edge_dst = edge_dst.contiguous()
    edge_weight = edge_weight.contiguous()
    V_orig = G * seq_len
    device = edge_src.device

    theta_env = os.environ.get("COMMUNITY_KV_LEIDEN_THETA")
    if theta_env is not None:
        theta = float(theta_env)
    max_iter_env = os.environ.get("COMMUNITY_KV_LEIDEN_MAX_INNER_ITER")
    if max_iter_env is not None:
        max_inner_iter = int(max_iter_env)
    boltzmann_env = os.environ.get("COMMUNITY_KV_LEIDEN_BOLTZMANN")
    if boltzmann_env is not None:
        use_boltzmann = boltzmann_env not in ("", "0", "false", "False")

    labels0, _ = _module.batched_leiden(
        edge_src,
        edge_dst,
        edge_weight,
        G,
        seq_len,
        max_level,
        float(resolution),
        float(theta),
        max_inner_iter,
        seed,
        use_boltzmann,
    )
    is_isolated = labels0 == -1
    arange_v = torch.arange(V_orig, dtype=torch.int32, device=device)
    cur_labels = torch.where(is_isolated, arange_v, labels0)

    dense0, cur_seq_len = _densify_per_graph(cur_labels, G, seq_len)
    chain = dense0

    cur_src = edge_src
    cur_dst = edge_dst
    cur_w = edge_weight
    prev_seq_len = seq_len
    dense_for_agg = dense0
    V_super_total = G * cur_seq_len

    for level in range(1, max_level):
        if cur_seq_len >= prev_seq_len:
            break
        cur_src, cur_dst, cur_w = _module.aggregate_coo(
            cur_src,
            cur_dst,
            cur_w,
            dense_for_agg,
            V_super_total,
        )
        if cur_src.numel() == 0:
            break
        labels_k, _ = _module.batched_leiden(
            cur_src,
            cur_dst,
            cur_w,
            G,
            cur_seq_len,
            max_level,
            float(resolution),
            float(theta),
            max_inner_iter,
            seed + level,
            use_boltzmann,
        )
        arange_super = torch.arange(G * cur_seq_len, dtype=torch.int32, device=device)
        labels_k_filled = torch.where(labels_k == -1, arange_super, labels_k)
        dense_k, next_seq_len = _densify_per_graph(labels_k_filled, G, cur_seq_len)
        chain = dense_k[chain.long()]
        prev_seq_len = cur_seq_len
        cur_seq_len = next_seq_len
        dense_for_agg = dense_k
        V_super_total = G * cur_seq_len

    last_modularity = _compute_disjoint_modularity(
        edge_src,
        edge_dst,
        edge_weight,
        chain,
        float(resolution),
    )

    keep = ~is_isolated
    vertex = torch.nonzero(keep, as_tuple=False).squeeze(-1).long()
    partition = chain[keep].to(torch.int32)
    return vertex, partition, last_modularity


def _compute_disjoint_modularity(
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_weight: torch.Tensor,
    labels: torch.Tensor,
    resolution: float,
) -> float:
    """Newman modularity on the disjoint union of all G subgraphs."""
    if edge_src.numel() == 0:
        return 0.0

    device = edge_src.device
    V = labels.numel()
    src_l = edge_src.long()
    dst_l = edge_dst.long()
    w64 = edge_weight.to(torch.float64)

    k_v = torch.zeros(V, dtype=torch.float64, device=device)
    k_v.scatter_add_(0, src_l, w64)
    k_v.scatter_add_(0, dst_l, w64)
    two_m = float(k_v.sum().item())
    if two_m <= 0.0:
        return 0.0

    labels_l = labels.long()
    same = labels_l[src_l] == labels_l[dst_l]
    upper_in = float(w64[same].sum().item())
    in_sum_total = 2.0 * upper_in

    _, inverse = torch.unique(labels_l, return_inverse=True)
    n_comm = int(inverse.max().item()) + 1
    sigma_tot = torch.zeros(n_comm, dtype=torch.float64, device=device)
    sigma_tot.scatter_add_(0, inverse, k_v)
    tot_sq_sum = float((sigma_tot * sigma_tot).sum().item())

    return in_sum_total / two_m - resolution * tot_sq_sum / (two_m * two_m)
