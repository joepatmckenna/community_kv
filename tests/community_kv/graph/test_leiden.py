"""Tests for community_kv.graph._leiden — the batched multi-level Leiden.

Two layers:
  * CPU helper unit tests (``TestDensifyPerGraph``,
    ``TestComputeDisjointModularity``) — the pure-Python pieces of the
    multi-level orchestration, runnable without the ``.so``.
  * ``TestRunLeidenFunctional`` (GPU) — end-to-end ``run_leiden`` on
    planted graphs whose correct partition is *unique* and therefore
    robust to the Boltzmann sampling: disjoint-component recovery,
    vertex-disjoint subgraph isolation, resolution control,
    reproducibility, and modularity-report consistency.

``partition()``'s wrapping of ``run_leiden`` is separately covered by
``test_partition.py::TestPartitionGPU``.
"""

import inspect
import math

import torch

from community_kv.graph._leiden import (
    _compute_disjoint_modularity,
    _densify_per_graph,
    run_leiden,
)
from tests.conftest import LEIDEN_REQUIRED


class TestDensifyPerGraph:
    def test_identity_labels_unchanged(self):
        # G=1, seq_len=4, labels already dense [0..3] -> same labels, same seq_len.
        labels = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        dense, new_seq_len = _densify_per_graph(labels, G=1, seq_len=4)
        assert new_seq_len == 4
        assert dense.tolist() == [0, 1, 2, 3]

    def test_collapses_unused_labels(self):
        # G=1, seq_len=4, labels {5, 7, 5, 7} -> dense {0, 1, 0, 1}, new_seq_len=2.
        labels = torch.tensor([5, 7, 5, 7], dtype=torch.int32)
        dense, new_seq_len = _densify_per_graph(labels, G=1, seq_len=4)
        assert new_seq_len == 2
        # torch.unique sorts ascending: 5 -> 0, 7 -> 1.
        assert dense.tolist() == [0, 1, 0, 1]

    def test_per_graph_offsets(self):
        # G=2, seq_len=3. Graph 0: {2, 2, 5} -> {0, 0, 1}. Graph 1: {9, 9, 9} -> {0, 0, 0}.
        # new_seq_len = max(2, 1) = 2; graph 1 entries get +1*2=+2 offset.
        labels = torch.tensor([2, 2, 5, 9, 9, 9], dtype=torch.int32)
        dense, new_seq_len = _densify_per_graph(labels, G=2, seq_len=3)
        assert new_seq_len == 2
        # Graph 0 dense: [0, 0, 1]. Graph 1 dense: [0, 0, 0] + offset 2 = [2, 2, 2].
        assert dense[:3].tolist() == [0, 0, 1]
        assert dense[3:].tolist() == [2, 2, 2]

    def test_singleton_per_graph(self):
        # G=3, seq_len=1, all distinct labels -> each densifies to 0.
        labels = torch.tensor([7, 9, 5], dtype=torch.int32)
        dense, new_seq_len = _densify_per_graph(labels, G=3, seq_len=1)
        assert new_seq_len == 1
        # Per-graph dense + offset: 0, 1, 2.
        assert dense.tolist() == [0, 1, 2]


class TestComputeDisjointModularity:
    def test_no_edges_zero(self):
        empty = torch.empty(0, dtype=torch.int32)
        labels = torch.tensor([0, 0, 0], dtype=torch.int32)
        assert (
            _compute_disjoint_modularity(empty, empty, empty.float(), labels, resolution=1.0) == 0.0
        )

    def test_single_triangle_one_community(self):
        # K3 with all weights 1, all in one community. Q = 1 - resolution.
        edge_src = torch.tensor([0, 0, 1], dtype=torch.int32)
        edge_dst = torch.tensor([1, 2, 2], dtype=torch.int32)
        edge_weight = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        labels = torch.tensor([0, 0, 0], dtype=torch.int32)
        # k_v = [2, 2, 2], two_m = 6, in_sum_total = 6 (all 3 edges in-community, x2).
        # sigma_tot[0] = 6, tot_sq_sum = 36.
        # Q = 6/6 - 1.0 * 36/36 = 0.0.
        q = _compute_disjoint_modularity(edge_src, edge_dst, edge_weight, labels, resolution=1.0)
        assert math.isclose(q, 0.0, abs_tol=1e-6)

    def test_two_communities_no_cross_edges(self):
        # Two disjoint K2s, one community each: Q = 1 - resolution * (1/2).
        # Edges: (0,1) and (2,3); labels [A,A,B,B].
        edge_src = torch.tensor([0, 2], dtype=torch.int32)
        edge_dst = torch.tensor([1, 3], dtype=torch.int32)
        edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float32)
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        # k_v = [1, 1, 1, 1], two_m = 4. in_sum_total = 2*2 = 4.
        # sigma_tot per community = [2, 2], tot_sq_sum = 8.
        # Q = 4/4 - 1.0 * 8 / 16 = 1.0 - 0.5 = 0.5.
        q = _compute_disjoint_modularity(edge_src, edge_dst, edge_weight, labels, resolution=1.0)
        assert math.isclose(q, 0.5, abs_tol=1e-6)

    def test_resolution_scales_penalty(self):
        # Same two-community graph; resolution=0.5 halves the penalty term.
        # Q = 1.0 - 0.5 * 0.5 = 0.75.
        edge_src = torch.tensor([0, 2], dtype=torch.int32)
        edge_dst = torch.tensor([1, 3], dtype=torch.int32)
        edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float32)
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        q = _compute_disjoint_modularity(edge_src, edge_dst, edge_weight, labels, resolution=0.5)
        assert math.isclose(q, 0.75, abs_tol=1e-6)

    def test_zero_weight_edges_yield_zero(self):
        # All weights zero -> two_m == 0 short-circuit.
        edge_src = torch.tensor([0, 1], dtype=torch.int32)
        edge_dst = torch.tensor([1, 2], dtype=torch.int32)
        edge_weight = torch.zeros(2, dtype=torch.float32)
        labels = torch.tensor([0, 0, 0], dtype=torch.int32)
        assert (
            _compute_disjoint_modularity(edge_src, edge_dst, edge_weight, labels, resolution=1.0)
            == 0.0
        )


# --------------------------------------------------------------------------- #
# run_leiden functional helpers
# --------------------------------------------------------------------------- #


def _clique(base: int, n: int) -> list[tuple[int, int]]:
    """Upper-triangular edge list of a clique on vertices ``[base, base+n)``."""
    return [(base + i, base + j) for i in range(n) for j in range(i + 1, n)]


def _run(edges, G, seq_len, *, resolution=1.0, seed=0, weight=1.0, use_boltzmann=False):
    """Build a CUDA COO from an upper-triangular edge list and run Leiden.

    Returns everything on CPU for assertion convenience:
    ``(src, dst, w, vertex, partition, modularity)``.
    """
    if edges:
        src = torch.tensor([a for a, _ in edges], dtype=torch.int32, device="cuda")
        dst = torch.tensor([b for _, b in edges], dtype=torch.int32, device="cuda")
        w = torch.full((len(edges),), float(weight), dtype=torch.float32, device="cuda")
    else:
        src = torch.empty(0, dtype=torch.int32, device="cuda")
        dst = torch.empty(0, dtype=torch.int32, device="cuda")
        w = torch.empty(0, dtype=torch.float32, device="cuda")
    vertex, partition, modularity = run_leiden(
        src,
        dst,
        w,
        G=G,
        seq_len=seq_len,
        resolution=resolution,
        seed=seed,
        use_boltzmann=use_boltzmann,
    )
    return (
        src.cpu(),
        dst.cpu(),
        w.cpu(),
        vertex.cpu(),
        partition.cpu(),
        float(modularity),
    )


def _group_by_community(vertex, partition) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for v, p in zip(vertex.tolist(), partition.tolist()):
        groups.setdefault(p, []).append(v)
    return groups


def _assert_valid_partition(vertex, partition, G, seq_len):
    """Pin the invariants every ``run_leiden`` output must satisfy and return
    the community groups: unique in-range vertices, and — the property
    community_kv leans on hardest — no community straddles two subgraphs."""
    V = G * seq_len
    vlist = vertex.tolist()
    assert len(vlist) == len(set(vlist)), "duplicate vertices in output"
    assert all(0 <= x < V for x in vlist), "vertex index out of range"
    groups = _group_by_community(vertex, partition)
    for label, verts in groups.items():
        subgraphs = {v // seq_len for v in verts}
        assert len(subgraphs) == 1, f"community {label} spans subgraphs {subgraphs}"
    return groups


def _assert_communities_connected(groups, src, dst):
    """Leiden's defining guarantee over Louvain: every community induces a
    connected subgraph over the input edges."""
    adj: dict[int, set[int]] = {}
    for a, b in zip(src.tolist(), dst.tolist()):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    for label, verts in groups.items():
        vset = set(verts)
        start = next(iter(vset))
        seen = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for nb in adj.get(u, ()):
                if nb in vset and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        assert (
            seen == vset
        ), f"community {label} is disconnected: {sorted(vset)} but BFS reached {sorted(seen)}"


def _newman_modularity(src, dst, w, labels, resolution):
    """Independent Newman modularity reference (CPU, float64) over the
    disjoint union of subgraphs. ``labels`` is a community id per vertex.

    A separate implementation of the same standard formula the kernel
    reports — a regression where the reported number drifts from the
    partition it was computed on shows up as a mismatch here.
    """
    V = labels.numel()
    src_l, dst_l = src.long(), dst.long()
    w64 = w.double()
    deg = torch.zeros(V, dtype=torch.float64)
    deg.scatter_add_(0, src_l, w64)
    deg.scatter_add_(0, dst_l, w64)
    two_m = deg.sum().item()
    if two_m <= 0.0:
        return 0.0
    same = labels[src_l] == labels[dst_l]
    in_total = 2.0 * w64[same].sum().item()
    _, inv = torch.unique(labels, return_inverse=True)
    sigma = torch.zeros(int(inv.max()) + 1, dtype=torch.float64)
    sigma.scatter_add_(0, inv, deg)
    return in_total / two_m - resolution * (sigma * sigma).sum().item() / (two_m * two_m)


def _full_labels(vertex, partition, V):
    """Lift the (non-isolated) returned partition to a full per-vertex label
    array; isolated vertices become distinct singletons (offset past the
    real labels so they can never alias a real community)."""
    offset = int(partition.max()) + 1 if partition.numel() else 0
    labels = torch.arange(V, dtype=torch.int64) + offset
    labels[vertex.long()] = partition.long()
    return labels


def _planted_partition(C, n, p_in, p_out, *, seed=0):
    """Deterministic planted-partition (stochastic block model) graph: C
    communities of n vertices, intra-community edge prob ``p_in``, inter
    ``p_out``. Returns (upper-triangular edge list, V)."""
    g = torch.Generator().manual_seed(seed)
    V = C * n
    comm = [i // n for i in range(V)]
    edges = []
    for i in range(V):
        for j in range(i + 1, V):
            p = p_in if comm[i] == comm[j] else p_out
            if torch.rand((), generator=g).item() < p:
                edges.append((i, j))
    return edges, V


# --------------------------------------------------------------------------- #
# run_leiden end-to-end (GPU + compiled extension)
# --------------------------------------------------------------------------- #


def test_run_leiden_defaults_to_best_gain():
    """Refinement target selection must default to greedy best-gain
    (Boltzmann OFF) — the higher-modularity option on ambiguous graphs.
    CPU-only signature check (no .so needed)."""
    assert inspect.signature(run_leiden).parameters["use_boltzmann"].default is False


@LEIDEN_REQUIRED
class TestRunLeidenFunctional:
    """End-to-end correctness of the batched multi-level Leiden kernel on
    planted graphs with a unique optimum (sampling-robust).

    The cases mirror what community_kv depends on:
      * disjoint structure is recovered and communities stay connected;
      * vertex-disjoint subgraphs are clustered independently — no
        community ever spans two subgraphs;
      * ``resolution`` monotonically controls granularity (the knob the
        per-layer resolutions.json tuning relies on);
      * the reported modularity equals an independent Newman-Q of the
        returned partition, and beats the trivial all-singletons partition;
      * a fixed seed is reproducible; degenerate inputs are handled.
    """

    def test_empty_graph_returns_no_communities(self):
        _, _, _, vertex, partition, mod = _run([], G=1, seq_len=4)
        assert vertex.numel() == 0
        assert partition.numel() == 0
        assert mod == 0.0

    def test_isolated_vertices_excluded(self):
        """A single triangle in a seq_len=6 graph: the three edge-bearing
        vertices form one community; the three degree-0 vertices are
        dropped from the output (Leiden returns only non-isolated vertices)."""
        src, dst, _, vertex, partition, _ = _run(_clique(0, 3), G=1, seq_len=6)
        groups = _assert_valid_partition(vertex, partition, G=1, seq_len=6)
        assert set(vertex.tolist()) == {0, 1, 2}, "edge-bearing vertices must all appear"
        assert len(groups) == 1, "a triangle is a single community"
        _assert_communities_connected(groups, src, dst)

    def test_disjoint_cliques_recovered(self):
        """Three size-4 cliques with no inter-clique edges (seq_len=16,
        4 isolated tail vertices). The unique modularity optimum is the
        three cliques — robust to sampling because merging disconnected
        components or splitting a clique both lower modularity."""
        edges = _clique(0, 4) + _clique(4, 4) + _clique(8, 4)
        src, dst, w, vertex, partition, mod = _run(edges, G=1, seq_len=16)
        groups = _assert_valid_partition(vertex, partition, G=1, seq_len=16)
        assert set(vertex.tolist()) == set(range(12)), "tail vertices 12..15 are isolated"
        assert len(groups) == 3, f"expected 3 cliques, got {len(groups)}"
        assert {frozenset(v) for v in groups.values()} == {
            frozenset(range(0, 4)),
            frozenset(range(4, 8)),
            frozenset(range(8, 12)),
        }
        _assert_communities_connected(groups, src, dst)

    def test_vertex_disjoint_subgraphs_clustered_independently(self):
        """The community_kv-critical property: G subgraphs packed into one
        COO are partitioned by their *own* internal structure, never
        bleeding across the ``g*seq_len`` boundary.

        Subgraph 0: one 5-clique -> 1 community.
        Subgraph 1: two 3-cliques -> 2 communities.
        Subgraph 2: one 5-clique -> 1 community.
        """
        G, seq_len = 3, 8
        edges = (
            _clique(0, 5)  # subgraph 0
            + _clique(seq_len, 3)
            + _clique(seq_len + 3, 3)  # subgraph 1: two cliques
            + _clique(2 * seq_len, 5)  # subgraph 2
        )
        src, dst, w, vertex, partition, mod = _run(edges, G=G, seq_len=seq_len)
        groups = _assert_valid_partition(vertex, partition, G=G, seq_len=seq_len)
        _assert_communities_connected(groups, src, dst)
        # Per-subgraph community counts prove independent clustering.
        per_subgraph: dict[int, int] = {}
        for verts in groups.values():
            per_subgraph[verts[0] // seq_len] = per_subgraph.get(verts[0] // seq_len, 0) + 1
        assert per_subgraph == {0: 1, 1: 2, 2: 1}, per_subgraph
        assert len(groups) == 4

    def test_modularity_report_matches_reference_and_beats_singletons(self):
        """The reported modularity equals an independent Newman-Q of the
        returned partition, and exceeds the all-singletons partition —
        a degenerate kernel that returned trivial partitions would fail."""
        edges = _clique(0, 4) + _clique(4, 4) + _clique(8, 4)
        src, dst, w, vertex, partition, mod = _run(edges, G=1, seq_len=16, resolution=1.0)
        V = 16
        labels = _full_labels(vertex, partition, V)
        ref = _newman_modularity(src, dst, w, labels, resolution=1.0)
        assert math.isclose(mod, ref, abs_tol=1e-5), f"reported {mod} vs reference {ref}"
        singletons = torch.arange(V, dtype=torch.int64)
        q_singletons = _newman_modularity(src, dst, w, singletons, resolution=1.0)
        assert mod > q_singletons, f"Leiden {mod} should beat all-singletons {q_singletons}"

    def test_resolution_controls_granularity(self):
        """Granularity is monotone non-decreasing in resolution — the
        contract the per-layer resolutions.json tuning depends on. A single
        6-clique stays whole at low resolution and fragments at high
        resolution (clique split threshold gamma > n/(n-1) = 1.2)."""
        edges = _clique(0, 6)
        n_comms = []
        for res in (0.5, 1.0, 3.0):
            _, _, _, vertex, partition, _ = _run(edges, G=1, seq_len=6, resolution=res)
            n_comms.append(len(_group_by_community(vertex, partition)))
        assert n_comms == sorted(n_comms), f"not monotone in resolution: {n_comms}"
        assert n_comms[0] == 1, f"clique should stay whole at low resolution, got {n_comms[0]}"
        assert n_comms[-1] > n_comms[0], f"high resolution must fragment: {n_comms}"

    def test_reproducible_with_fixed_seed(self):
        """Same inputs + same seed -> bit-identical partition and modularity."""
        edges = _clique(0, 4) + _clique(4, 4) + _clique(8, 4)
        _, _, _, v1, p1, m1 = _run(edges, G=1, seq_len=16, seed=7)
        _, _, _, v2, p2, m2 = _run(edges, G=1, seq_len=16, seed=7)
        assert torch.equal(v1, v2)
        assert torch.equal(p1, p2)
        assert m1 == m2

    def test_scale_disjoint_cliques_across_subgraphs(self):
        """Regression guard at realistic width: 8 subgraphs, each with three
        size-6 cliques. Exercises G-batching + multi-level aggregation and
        pins disjointness, connectivity, and the 24-community recovery."""
        G, seq_len, n_cliques, csize = 8, 128, 3, 6
        edges: list[tuple[int, int]] = []
        for g in range(G):
            base = g * seq_len
            for c in range(n_cliques):
                edges += _clique(base + c * csize, csize)
        src, dst, w, vertex, partition, mod = _run(edges, G=G, seq_len=seq_len)
        groups = _assert_valid_partition(vertex, partition, G=G, seq_len=seq_len)
        _assert_communities_connected(groups, src, dst)
        assert len(groups) == G * n_cliques, f"expected {G * n_cliques}, got {len(groups)}"
        labels = _full_labels(vertex, partition, G * seq_len)
        ref = _newman_modularity(src, dst, w, labels, resolution=1.0)
        assert math.isclose(mod, ref, abs_tol=1e-5), f"reported {mod} vs reference {ref}"

    # ---- merge / anti-fragmentation invariants -------------------------- #

    def test_single_edge_merges_into_one_community(self):
        """A single edge's two endpoints must end up in ONE community —
        its modularity optimum (Q=0 vs Q=-0.5 for two singletons). A
        symmetric connected pair must never be left split."""
        src, dst, w, vertex, partition, mod = _run([(0, 1)], G=1, seq_len=2)
        groups = _assert_valid_partition(vertex, partition, G=1, seq_len=2)
        assert set(vertex.tolist()) == {0, 1}
        assert len(groups) == 1, "a single edge must merge its two endpoints"
        assert math.isclose(mod, 0.0, abs_tol=1e-6)

    def test_path_merges_into_one_community(self):
        """A 3-vertex path's modularity optimum is a single community (Q=0);
        it must not fragment into singletons (Q=-0.375)."""
        src, dst, w, vertex, partition, mod = _run([(0, 1), (1, 2)], G=1, seq_len=3)
        groups = _assert_valid_partition(vertex, partition, G=1, seq_len=3)
        _assert_communities_connected(groups, src, dst)
        assert len(groups) == 1, "a 3-path's optimum is one community"

    def test_planted_partition_recovered_not_fragmented(self):
        """A dense stochastic-block-model graph (4 communities, strong intra /
        weak inter edges) must be recovered to ~C communities at high
        modularity, not split into many low-modularity pieces. Guards
        community-detection quality on a realistic connected graph."""
        C, n = 4, 20
        edges, V = _planted_partition(C, n, p_in=0.5, p_out=0.02, seed=0)
        src, dst, w, vertex, partition, mod = _run(edges, G=1, seq_len=V)
        groups = _assert_valid_partition(vertex, partition, G=1, seq_len=V)
        _assert_communities_connected(groups, src, dst)
        assert len(groups) <= 2 * C, f"over-fragmented: {len(groups)} communities (planted {C})"
        assert mod > 0.55, f"modularity too low ({mod:.4f}) — structure not recovered"

    def test_boltzmann_flag_both_modes_recover_disjoint_cliques(self):
        """Both refinement modes (default best-gain and opt-in Boltzmann) must
        produce a valid partition; on unambiguous structure (disjoint cliques)
        they recover the same 3 communities. Exercises the use_boltzmann path
        end-to-end so the flag plumbing can't silently rot."""
        edges = _clique(0, 4) + _clique(4, 4) + _clique(8, 4)
        for ub in (False, True):
            src, dst, w, vertex, partition, mod = _run(edges, G=1, seq_len=16, use_boltzmann=ub)
            groups = _assert_valid_partition(vertex, partition, G=1, seq_len=16)
            _assert_communities_connected(groups, src, dst)
            assert len(groups) == 3, f"use_boltzmann={ub}: expected 3, got {len(groups)}"
