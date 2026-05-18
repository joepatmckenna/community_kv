"""Graph manager for CommunityKV.

Per-layer state (allocated at initialize time):
- community_ids[layer]: (num_graphs, max_seq_len) — community ID per token, -1 = unassigned
- centroids[layer]: (num_centroid_heads, max_num_communities, head_dim) — mean key per community, -inf = unused
- num_communities[layer]: (num_graphs,) — active community count per graph
- community_sizes[layer]: (num_graphs, max_num_communities) — token count per community
- adjacency[layer]: list of (src, dst, weight) COO tuples per graph
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import torch
import igraph as ig

try:
    import cugraph
    import cudf
    import cupy
    from torch.utils.dlpack import to_dlpack, from_dlpack
    _HAS_CUGRAPH = True
except ImportError:
    _HAS_CUGRAPH = False

from community_kv.graph.utils import GraphAggregation, build_adjacency


class GraphManager:
    """Manages per-layer graph state for CommunityKV decode."""

    def __init__(
        self,
        *,
        num_query_heads: int,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        token_budget: int,
        max_new_tokens: int,
        aggregation: GraphAggregation = GraphAggregation.QUERY_GROUP,
        num_sink_tok_to_exclude: int = 10,
        lam: float = 0.5,
        leiden_backend: str = "igraph",
    ):
        assert leiden_backend in ("igraph", "cugraph"), (
            f"leiden_backend must be 'igraph' or 'cugraph', got {leiden_backend!r}"
        )
        self.aggregation = aggregation
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.num_sink_tok_to_exclude = num_sink_tok_to_exclude
        self.lam = lam
        self.token_budget = token_budget
        self.max_new_tokens = max_new_tokens
        self.leiden_backend = leiden_backend
        self.num_graphs_per_layer = aggregation.num_graphs_per_layer(num_query_heads, num_kv_heads)

        # Per-layer state (allocated at initialize time)
        self.community_ids: dict[int, torch.Tensor] = {}
        self.centroids: dict[int, torch.Tensor] = {}
        self.num_communities: dict[int, torch.Tensor] = {}
        self.community_sizes: dict[int, torch.Tensor] = {}
        self.community_weight: dict[int, torch.Tensor] = {}
        self.seq_len: dict[int, int] = {}
        self.modularity: dict[int, torch.Tensor] = {}
        self.total_weight: dict[int, torch.Tensor] = {}
        # Per-layer, per-graph COO adjacency (for incremental updates)
        self.adjacency: dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        self._initialized: set[int] = set()

        # Async initialization
        self._executor = ThreadPoolExecutor(max_workers=4)
        # Separate pool for per-graph Leiden calls inside _initialize_impl. Kept
        # separate from _executor to avoid deadlock: _executor workers block on
        # leiden futures, so those must run on a different pool.
        self._leiden_executor = ThreadPoolExecutor(max_workers=8)
        self._ready_events: dict[int, threading.Event] = {}
        self._init_errors: dict[int, Exception] = {}
        # CUDA event recorded on the worker's stream after init writes finish.
        # retrieve() waits on it so the main thread's reads are ordered after
        # the worker's writes on the same device.
        self._completion_events: dict[int, torch.cuda.Event] = {}

    def initialize(
        self,
        layer_idx: int,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor,
        keys: torch.Tensor,
    ):
        """Submit async graph initialization for a layer.

        Returns immediately. Decode will block in retrieve() until done.

        Args:
            layer_idx: current layer index
            topk_indices: (H_q, S_eligible, kappa)
            topk_scores: (H_q, S_eligible, kappa)
            keys: (num_kv_heads, prefill_seq_len, head_dim)
        """
        event = threading.Event()
        self._ready_events[layer_idx] = event
        # Record a CUDA event on the producer's current stream so the worker
        # can wait for the producing kernels (attention_with_topk) to complete
        # before reading topk_indices/topk_scores/keys on its own stream.
        with torch.cuda.device(keys.device):
            producer_event = torch.cuda.Event()
            producer_event.record()
        self._executor.submit(
            self._do_initialize, layer_idx, topk_indices, topk_scores, keys, event, producer_event,
        )

    def _do_initialize(
        self,
        layer_idx: int,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor,
        keys: torch.Tensor,
        event: threading.Event,
        producer_event: torch.cuda.Event,
    ):
        """Actual initialization logic. Runs in thread pool."""
        try:
            with torch.cuda.device(keys.device):
                # Order this thread's stream after the producer's stream.
                producer_event.wait()
                self._initialize_impl(layer_idx, topk_indices, topk_scores, keys)
                # Record an event so retrieve() can order the main thread's
                # reads after the writes done here.
                completion = torch.cuda.Event()
                completion.record()
                self._completion_events[layer_idx] = completion
        except Exception as e:
            self._init_errors[layer_idx] = e
        finally:
            event.set()

    def _initialize_impl(
        self,
        layer_idx: int,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor,
        keys: torch.Tensor,
    ):
        """Initialize graph state for a layer after prefill.

        Builds adjacency per graph, runs leiden, computes centroids.

        Args:
            layer_idx: current layer index
            topk_indices: (H_q, S_eligible, kappa)
            topk_scores: (H_q, S_eligible, kappa)
            keys: (num_kv_heads, prefill_seq_len, head_dim)
        """
        assert topk_indices.ndim == 3
        num_kv_heads, prefill_seq_len, head_dim = keys.shape
        device = keys.device
        num_centroid_heads = max(self.num_graphs_per_layer, num_kv_heads)
        H_q = topk_indices.shape[0]
        kappa = topk_indices.shape[2]

        self.seq_len[layer_idx] = prefill_seq_len

        # Allocate community_ids: (num_graphs, max_seq_len), -1 = unassigned
        max_seq_len = prefill_seq_len + self.max_new_tokens
        self.community_ids[layer_idx] = torch.full(
            (self.num_graphs_per_layer, max_seq_len), -1, dtype=torch.int32, device=device
        )

        adjacency_list = []
        heads_per_group = H_q // self.num_graphs_per_layer
        # Per-graph filtered edge tensors (kept on GPU; move to CPU only for igraph).
        filtered_edges: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        # --- Phase A: build adjacency on GPU for every graph ---
        for g in range(self.num_graphs_per_layer):
            # Get this graph's topk data
            if self.aggregation == GraphAggregation.PER_QUERY_HEAD:
                g_indices = topk_indices[g]  # (S_elig, kappa)
                g_scores = topk_scores[g]
                q_positions = None  # use default
            elif self.aggregation == GraphAggregation.QUERY_GROUP:
                start = g * heads_per_group
                end = start + heads_per_group
                g_indices = topk_indices[start:end].reshape(-1, kappa)  # (heads_per_group * S_elig, kappa)
                g_scores = topk_scores[start:end].reshape(-1, kappa)
                # Tile query positions: each head has the same positions
                S_elig = topk_indices.shape[1]
                base_positions = torch.arange(
                    kappa - 1 + self.num_sink_tok_to_exclude,
                    kappa - 1 + self.num_sink_tok_to_exclude + S_elig,
                    device=device,
                )
                q_positions = base_positions.repeat(heads_per_group)  # (heads_per_group * S_elig,)
            else:  # LAYER_WISE
                g_indices = topk_indices.reshape(-1, kappa)  # (H_q * S_elig, kappa)
                g_scores = topk_indices.reshape(-1, kappa)
                S_elig = topk_indices.shape[1]
                base_positions = torch.arange(
                    kappa - 1 + self.num_sink_tok_to_exclude,
                    kappa - 1 + self.num_sink_tok_to_exclude + S_elig,
                    device=device,
                )
                q_positions = base_positions.repeat(H_q)

            # Build adjacency on GPU
            edge_src, edge_dst, edge_weight = build_adjacency(
                g_indices, g_scores,
                seq_len=prefill_seq_len,
                num_sink_tok_to_exclude=self.num_sink_tok_to_exclude,
                lam=self.lam,
                query_positions=q_positions,
            )
            adjacency_list.append((edge_src, edge_dst, edge_weight))

            # Filter edges that exceed valid node range
            valid_edges = (edge_src < prefill_seq_len) & (edge_dst < prefill_seq_len)
            filtered_edges.append((
                edge_src[valid_edges],
                edge_dst[valid_edges],
                edge_weight[valid_edges],
            ))

        # --- Phase B: run Leiden per graph ---
        n_iterations = max(1, int(torch.tensor(float(prefill_seq_len)).log10().item()))
        modularity_values: list[float] = [0.0] * self.num_graphs_per_layer

        if self.leiden_backend == "igraph":
            # CPU Leiden, parallelized across graphs via thread pool.
            def _run_igraph(src_cpu, dst_cpu, weight_cpu):
                graph = ig.Graph(
                    n=prefill_seq_len,
                    edges=list(zip(src_cpu, dst_cpu)),
                    directed=False,
                )
                graph.es["weight"] = weight_cpu
                partition = graph.community_leiden(
                    objective_function="modularity",
                    weights="weight",
                    resolution=1.0,
                    n_iterations=n_iterations,
                )
                return partition.membership, graph.modularity(partition.membership, weights="weight")

            # Move edges to CPU, submit in parallel.
            cpu_inputs = [
                (s.cpu().tolist(), d.cpu().tolist(), w.cpu().tolist())
                for s, d, w in filtered_edges
            ]
            futures = [self._leiden_executor.submit(_run_igraph, *ci) for ci in cpu_inputs]
            for g, f in enumerate(futures):
                membership, modularity = f.result()
                self.community_ids[layer_idx][g, :prefill_seq_len] = torch.tensor(
                    membership, dtype=torch.int32, device=device,
                )
                modularity_values[g] = modularity

        else:  # cugraph
            # GPU Leiden — edges stay on device. Each task runs on its own
            # CUDA stream so cugraph calls can overlap if the GPU has spare
            # throughput (small graphs); for compute-bound graphs streams just
            # serialize at the hardware level.
            assert _HAS_CUGRAPH, (
                "leiden_backend='cugraph' requires cugraph/cudf/cupy; "
                "install with `pip install '.[cugraph]' --extra-index-url=https://pypi.nvidia.com`"
            )

            def _torch_to_cupy(t: torch.Tensor):
                return cupy.from_dlpack(to_dlpack(t.contiguous()))

            def _run_cugraph(src, dst, weight, stream):
                with torch.cuda.stream(stream):
                    df = cudf.DataFrame({
                        "src": _torch_to_cupy(src.int()),
                        "dst": _torch_to_cupy(dst.int()),
                        "weight": _torch_to_cupy(weight.float()),
                    })
                    G = cugraph.Graph(directed=False)
                    G.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight")
                    parts, modularity = cugraph.leiden(G, max_iter=n_iterations, resolution=1.0)
                    parts_sorted = parts.sort_values("vertex")
                    vertex = from_dlpack(parts_sorted["vertex"].to_dlpack()).long()
                    partition = from_dlpack(parts_sorted["partition"].to_dlpack()).to(torch.int32)
                return vertex, partition, float(modularity)

            streams = [torch.cuda.Stream(device=device) for _ in filtered_edges]
            futures = [
                self._leiden_executor.submit(_run_cugraph, s, d, w, stream)
                for (s, d, w), stream in zip(filtered_edges, streams)
            ]
            for g, (f, stream) in enumerate(zip(futures, streams)):
                vertex, partition, modularity = f.result()
                stream.synchronize()

                membership = torch.full(
                    (prefill_seq_len,), -1, dtype=torch.int32, device=device,
                )
                membership[vertex] = partition
                # Give each isolated vertex its own singleton community id.
                missing = membership == -1
                if missing.any():
                    next_id = int(partition.max().item()) + 1 if partition.numel() > 0 else 0
                    num_missing = int(missing.sum().item())
                    singleton_ids = torch.arange(
                        next_id, next_id + num_missing, dtype=torch.int32, device=device,
                    )
                    membership[missing] = singleton_ids

                self.community_ids[layer_idx][g, :prefill_seq_len] = membership
                modularity_values[g] = modularity

        # Store adjacency for incremental updates
        self.adjacency[layer_idx] = adjacency_list

        # --- Derive num_communities, centroids, sizes from partition ---
        self.num_communities[layer_idx] = self.community_ids[layer_idx].max(dim=-1).values + 1  # (G,)
        num_communities = int(self.num_communities[layer_idx].max().item())
        max_num_communities = num_communities + self.max_new_tokens

        # Centroids
        centroids = torch.full(
            (num_centroid_heads, max_num_communities, head_dim),
            float("-inf"), dtype=keys.dtype, device=device,
        )

        I = self.community_ids[layer_idx][:, :prefill_seq_len].long()  # (num_graphs, prefill_seq_len)
        K = keys  # (num_kv_heads, prefill_seq_len, head_dim)

        if self.num_graphs_per_layer < num_kv_heads:
            I = I.expand(num_centroid_heads, -1)
        elif self.num_graphs_per_layer > num_kv_heads:
            factor = self.num_graphs_per_layer // num_kv_heads
            K = K.unsqueeze(1).expand(-1, factor, -1, -1).contiguous().view(
                num_centroid_heads, prefill_seq_len, head_dim
            )

        I_expanded = I.unsqueeze(-1).expand(num_centroid_heads, prefill_seq_len, head_dim)
        centroids[:, :num_communities].zero_()
        centroids.scatter_add_(1, I_expanded, K)

        # Sizes for division
        sizes = torch.zeros(num_centroid_heads, max_num_communities, dtype=torch.float32, device=device)
        sizes.scatter_add_(1, I, torch.ones(num_centroid_heads, prefill_seq_len, dtype=torch.float32, device=device))
        centroids /= sizes.unsqueeze(-1).clamp(min=1)

        # Mark empty as -inf
        empty_mask = (sizes == 0).unsqueeze(-1).expand(num_centroid_heads, max_num_communities, head_dim)
        centroids.masked_fill_(empty_mask, float("-inf"))

        self.centroids[layer_idx] = centroids

        # Community sizes (per graph)
        community_sizes = torch.zeros(
            self.num_graphs_per_layer, max_num_communities, dtype=torch.long, device=device
        )
        I_graph = self.community_ids[layer_idx][:, :prefill_seq_len].long()
        community_sizes.scatter_add_(
            1, I_graph, torch.ones(self.num_graphs_per_layer, prefill_seq_len, dtype=torch.long, device=device)
        )
        self.community_sizes[layer_idx] = community_sizes

        # Compute total_weight and community_weight from adjacency
        # total_weight[g] = sum of all edge weights for graph g
        # community_weight[g, c] = sum of weighted degrees of nodes in community c
        self.community_weight[layer_idx] = torch.zeros(
            self.num_graphs_per_layer, max_num_communities, dtype=torch.float32, device=device
        )
        self.total_weight[layer_idx] = torch.zeros(self.num_graphs_per_layer, dtype=torch.float32, device=device)

        for g in range(self.num_graphs_per_layer):
            edge_src, edge_dst, edge_weight = adjacency_list[g]
            if edge_weight.numel() == 0:
                continue

            # Total weight = sum of all edge weights (undirected, so each edge counted once)
            self.total_weight[layer_idx][g] = edge_weight.sum()

            # Node degrees: degree[node] = sum of weights of edges incident to node
            # Since edges are stored as upper triangle (src <= dst), each edge contributes to both endpoints
            node_degree = torch.zeros(prefill_seq_len, dtype=torch.float32, device=device)
            node_degree.scatter_add_(0, edge_src, edge_weight)
            node_degree.scatter_add_(0, edge_dst, edge_weight)

            # Community weight: scatter node degrees into communities
            membership = self.community_ids[layer_idx][g, :prefill_seq_len].long()
            self.community_weight[layer_idx][g].scatter_add_(0, membership, node_degree)

        self.modularity[layer_idx] = torch.tensor(modularity_values, dtype=torch.float32, device=device)

        self._initialized.add(layer_idx)

    def retrieve(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve indices of tokens to attend to.

        Blocks until initialize for this layer is complete.

        Args:
            layer_idx: current layer index
            query: (H_q, 1, D) decode query
            key: (H_kv, S, D) full key cache
            value: (H_kv, S, D) full value cache

        Returns:
            retrieved_indices: (H_kv, token_budget) position indices per KV head
        """
        # Block until initialization is done for this layer
        event = self._ready_events.get(layer_idx)
        if event is not None:
            event.wait()
            # Re-raise any error from the init thread
            if layer_idx in self._init_errors:
                raise self._init_errors[layer_idx]
            # Order this thread's reads after the worker's writes on the same
            # device. event.set() only gives CPU-level happens-before; without
            # this CUDA wait the main thread can read uninitialized state.
            completion = self._completion_events.get(layer_idx)
            if completion is not None:
                with torch.cuda.device(key.device):
                    completion.wait()

        H_q = query.shape[0]
        H_kv = key.shape[0]
        S = key.shape[1]
        D = key.shape[2]
        device = key.device

        budget = min(self.token_budget, S)
        centroids = self.centroids[layer_idx]  # (num_centroid_heads, max_num_communities, D)
        num_centroid_heads = centroids.shape[0]

        # Map each query head to its centroid set
        head_map = torch.arange(H_q, device=device) // (H_q // num_centroid_heads)

        # Gather centroids for each query head
        head_centroids = centroids[head_map]  # (H_q, max_num_communities, D)

        # Score query against centroids: (H_q, 1, D) @ (H_q, D, C) -> (H_q, C)
        scores = torch.bmm(query, head_centroids.transpose(1, 2)).squeeze(1)  # (H_q, C)
        # Mask NaN scores from -inf centroids (inf * 0 = nan in dot product)
        scores = scores.nan_to_num(nan=float("-inf"))
        # Mask sink communities (first num_sink_tok_to_exclude communities are singletons for sink tokens)
        scores[:, :self.num_sink_tok_to_exclude] = float("-inf")

        # Sort communities by score, get sorted community sizes, cumsum
        _, sorted_comm_idx = scores.sort(dim=-1, descending=True)  # (H_q, C)

        # Community sizes per centroid head: (num_centroid_heads, C)
        C = centroids.shape[1]
        sizes = self.community_sizes[layer_idx]  # (G, max_num_communities)
        # Expand to num_centroid_heads
        if self.num_graphs_per_layer < num_centroid_heads:
            head_sizes = sizes.expand(num_centroid_heads, -1)
        elif self.num_graphs_per_layer > num_centroid_heads:
            heads_per_group = self.num_graphs_per_layer // num_centroid_heads
            head_sizes = sizes[::heads_per_group]  # take first of each group
        else:
            head_sizes = sizes
        # Map to per-query-head: (H_q, max_num_communities)
        head_sizes = head_sizes[head_map][:, :C]

        # Gather sizes in score-sorted order and cumsum
        sorted_sizes = head_sizes.gather(1, sorted_comm_idx)  # (H_q, C)
        cumulative_sizes = sorted_sizes.cumsum(dim=-1)  # (H_q, C)

        # Find communities that fit entirely within budget
        available_budget = budget - self.num_sink_tok_to_exclude - 1  # reserve for sinks + current token
        fits = cumulative_sizes <= available_budget  # (H_q, C) bool
        num_full = fits.sum(dim=-1)  # (H_q,) — number of full communities per head

        # Use first query head per KV group as representative
        heads_per_group = H_q // H_kv
        rep_heads = torch.arange(0, H_q, heads_per_group, device=device)  # (H_kv,)
        rep_num_full = num_full[rep_heads]  # (H_kv,)
        rep_sorted_comm_idx = sorted_comm_idx[rep_heads]  # (H_kv, C)

        # Get community_ids expanded to num_centroid_heads, then select per KV head
        community_ids = self.community_ids[layer_idx][:, :S]  # (G, S)
        if self.num_graphs_per_layer < num_centroid_heads:
            cids = community_ids.expand(num_centroid_heads, -1)
        else:
            cids = community_ids
        rep_head_map = head_map[rep_heads]  # (H_kv,)
        kv_cids = cids[rep_head_map]  # (H_kv, S)

        # Build mask of selected full communities per KV head
        # comm_selected[kv, c] = True if community rank c is within num_full for that head
        comm_range = torch.arange(C, device=device).unsqueeze(0)  # (1, C)
        comm_selected = comm_range < rep_num_full.unsqueeze(1)  # (H_kv, C)

        # Get the actual community IDs that are selected (pad unselected with -2 to avoid matching -1)
        selected_comm_ids = rep_sorted_comm_idx.masked_fill(~comm_selected, -2)  # (H_kv, C)

        # Position mask: position is selected if its community is in the full set
        # kv_cids: (H_kv, S), selected_comm_ids: (H_kv, C)
        full_pos_mask = (kv_cids.unsqueeze(2) == selected_comm_ids.unsqueeze(1)).any(dim=2)  # (H_kv, S)

        # Boundary community: the community at rank num_full (first one that doesn't fit)
        # Clamp to valid range for gather
        boundary_rank = rep_num_full.clamp(max=C - 1)  # (H_kv,)
        boundary_comm_id = rep_sorted_comm_idx.gather(1, boundary_rank.unsqueeze(1)).squeeze(1)  # (H_kv,)

        # Remaining budget after full communities
        # cumsum at num_full-1 gives tokens used by full communities
        used_idx = (rep_num_full - 1).clamp(min=0).unsqueeze(1)  # (H_kv, 1)
        rep_cumulative = cumulative_sizes[rep_heads]  # (H_kv, C)
        used = rep_cumulative.gather(1, used_idx).squeeze(1)  # (H_kv,)
        used = used * (rep_num_full > 0).long()  # zero if no full communities
        remaining = (available_budget - used).clamp(min=0)  # (H_kv,)

        # Boundary positions mask: positions in the boundary community
        boundary_mask = kv_cids == boundary_comm_id.unsqueeze(1)  # (H_kv, S)

        # For boundary: we want the most recent `remaining` positions.
        # Assign a priority: position index (higher = more recent), -1 for non-boundary
        boundary_priority = torch.where(boundary_mask, torch.arange(S, device=device).unsqueeze(0), -1)  # (H_kv, S)
        # Sort descending to get most recent first, take `remaining` per head
        # Use topk with max possible remaining
        max_remaining = int(remaining.max().item()) if remaining.max() > 0 else 0
        if max_remaining > 0:
            # Get top positions from boundary per head
            topk_boundary, _ = boundary_priority.topk(min(max_remaining, S), dim=-1)  # (H_kv, max_remaining)
            # Mask out positions beyond each head's remaining budget
            remain_mask = torch.arange(topk_boundary.shape[1], device=device).unsqueeze(0) < remaining.unsqueeze(1)
            # Convert back to position mask
            boundary_selected = torch.zeros(H_kv, S, dtype=torch.bool, device=device)
            valid_positions = topk_boundary.clamp(min=0) * remain_mask.long()
            # Scatter True at selected boundary positions
            boundary_selected.scatter_(1, valid_positions, remain_mask)
        else:
            boundary_selected = torch.zeros(H_kv, S, dtype=torch.bool, device=device)

        # Combine: full communities + boundary partial + sinks + current token
        pos_mask = full_pos_mask | boundary_selected
        pos_mask[:, :self.num_sink_tok_to_exclude] = True
        pos_mask[:, S - 1] = True

        # Truncate to budget: argsort descending on mask, take first `budget`
        pos_indices = pos_mask.float().argsort(dim=-1, descending=True)[:, :budget]
        pos_indices = pos_indices.sort(dim=-1).values  # (H_kv, budget)
        pos_indices = pos_indices.clamp(0, S - 1)  # safety clamp

        return pos_indices

    def update(
        self,
        layer_idx: int,
        topk_indices_global: torch.Tensor,
        topk_scores: torch.Tensor,
        attn_weights: torch.Tensor,
        retrieved_indices: torch.Tensor,
        keys: torch.Tensor,
    ):
        """Update graph state: add edges, assign community, update centroid.

        Uses build_adjacency for edge construction (same logic as prefill).

        Args:
            layer_idx: current layer index
            topk_indices_global: (H_q, kappa) global position indices of top-k keys
            topk_scores: (H_q, kappa) attention scores for top-k keys
            attn_weights: (H_q, 1, S_retrieved) full attention weights over retrieved set
            retrieved_indices: (H_kv, S_retrieved) global positions of retrieved tokens
            keys: (H_kv, S, D) full key cache
        """
        S_cur = self.seq_len[layer_idx]
        new_pos = S_cur
        G = self.num_graphs_per_layer
        H_q = topk_indices_global.shape[0]
        kappa = topk_indices_global.shape[1]
        H_kv = keys.shape[0]
        D = keys.shape[2]
        device = keys.device
        heads_per_group = H_q // G
        num_centroid_heads = self.centroids[layer_idx].shape[0]

        # Aggregate topk to per-graph: use first head in each group
        rep_heads = torch.arange(0, H_q, heads_per_group, device=device)
        g_topk_idx = topk_indices_global[rep_heads].long()       # (G, kappa)
        g_topk_scores = topk_scores[rep_heads].float()    # (G, kappa)

        # Clamp topk indices to valid range for community_ids lookup
        max_pos = self.community_ids[layer_idx].shape[1] - 1
        g_topk_idx = g_topk_idx.clamp(-1, max_pos)

        # New token's key
        new_key = keys[:, new_pos, :]  # (H_kv, D)

        # Per-graph: build edges, compute ΔQ, assign community
        assigned_comms = torch.zeros(G, dtype=torch.long, device=device)
        join_mask = torch.zeros(G, dtype=torch.bool, device=device)
        node_degrees = torch.zeros(G, dtype=torch.float32, device=device)
        new_edge_totals = torch.zeros(G, dtype=torch.float32, device=device)

        for g in range(G):
            # Build edges using build_adjacency (1 query at position new_pos)
            edge_src, edge_dst, edge_weight = build_adjacency(
                topk_indices=g_topk_idx[g].unsqueeze(0),   # (1, kappa)
                topk_scores=g_topk_scores[g].unsqueeze(0), # (1, kappa)
                seq_len=S_cur + 1,
                num_sink_tok_to_exclude=self.num_sink_tok_to_exclude,
                lam=self.lam,
                query_offset=new_pos,
            )

            if edge_weight.numel() == 0:
                # No valid edges — stay singleton
                new_comm_id = int(self.num_communities[layer_idx][g].item())
                assigned_comms[g] = new_comm_id
                continue

            # Total new edge weight
            new_edge_totals[g] = edge_weight.sum()

            # Node degree of new_pos: sum of weights of edges incident to new_pos
            incident_mask = (edge_src == new_pos) | (edge_dst == new_pos)
            node_degrees[g] = edge_weight[incident_mask].sum()

            # Neighbor positions (nodes connected to new_pos)
            neighbors = torch.where(edge_src == new_pos, edge_dst, edge_src)
            neighbor_weights = edge_weight[incident_mask]
            neighbor_positions = neighbors[incident_mask]

            # Look up neighbor communities
            neighbor_comms = self.community_ids[layer_idx][g, neighbor_positions]  # (num_neighbors,)

            # Compute ΔQ for each neighbor's community
            total_w = self.total_weight[layer_idx][g].item()
            two_m = max(2.0 * total_w, 1.0)
            w_i = node_degrees[g].item()

            # Group edge weights by community
            unique_comms = neighbor_comms.unique()
            unique_comms = unique_comms[unique_comms >= 0]

            best_delta_q = 0.0
            best_comm = -1

            for c in unique_comms:
                c_val = c.item()
                w_ic = neighbor_weights[neighbor_comms == c].sum().item()
                w_C = self.community_weight[layer_idx][g, c_val].item()
                delta_q = w_ic - w_i * w_C / two_m
                if delta_q > best_delta_q:
                    best_delta_q = delta_q
                    best_comm = c_val

            if best_comm >= 0:
                assigned_comms[g] = best_comm
                join_mask[g] = True
            else:
                new_comm_id = int(self.num_communities[layer_idx][g].item())
                assigned_comms[g] = new_comm_id

            # Update community_weight for neighbors (their degree increased)
            for idx in range(neighbor_positions.shape[0]):
                j_pos = neighbor_positions[idx].item()
                j_comm = self.community_ids[layer_idx][g, j_pos].item()
                if j_comm >= 0:
                    self.community_weight[layer_idx][g, j_comm] += neighbor_weights[idx].item()

        # --- Apply assignments ---
        for g in range(G):
            comm = assigned_comms[g].item()
            self.community_ids[layer_idx][g, new_pos] = comm
            self.community_sizes[layer_idx][g, comm] += 1

            # Community weight: new node's degree
            self.community_weight[layer_idx][g, comm] += node_degrees[g].item()

        # Increment num_communities for singletons
        self.num_communities[layer_idx] = torch.where(
            join_mask, self.num_communities[layer_idx], self.num_communities[layer_idx] + 1
        )

        # Update total weight
        self.total_weight[layer_idx] = self.total_weight[layer_idx] + new_edge_totals

        # --- Update centroids ---
        for g in range(G):
            comm = assigned_comms[g].item()
            if self.num_graphs_per_layer == num_centroid_heads:
                ch = g
                kv_h = g if H_kv == G else g // (G // H_kv)
            elif self.num_graphs_per_layer < num_centroid_heads:
                kv_h = min(g, H_kv - 1)
                ch = kv_h
            else:
                ch = g
                kv_h = g // (G // H_kv)

            if join_mask[g]:
                old_size = self.community_sizes[layer_idx][g, comm].item() - 1
                new_size = old_size + 1
                old_centroid = self.centroids[layer_idx][ch, comm]
                self.centroids[layer_idx][ch, comm] = (old_centroid * old_size + new_key[kv_h]) / new_size
            else:
                self.centroids[layer_idx][ch, comm] = new_key[kv_h]

        # Update seq_len
        self.seq_len[layer_idx] = S_cur + 1

    def reset(self):
        """Clear all per-layer state for a new sequence."""
        self.community_ids.clear()
        self.centroids.clear()
        self.num_communities.clear()
        self.community_sizes.clear()
        self.community_weight.clear()
        self.seq_len.clear()
        self.modularity.clear()
        self.total_weight.clear()
        self.adjacency.clear()
        self._initialized.clear()
        self._ready_events.clear()
        self._init_errors.clear()
        self._completion_events.clear()

    def shutdown(self):
        self._executor.shutdown(wait=True)
        self._leiden_executor.shutdown(wait=True)
