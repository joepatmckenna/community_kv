"""Typed state for immutable prefill CSR and mutable decode pages."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _require_int32(name: str, value: torch.Tensor) -> None:
    if value.dtype != torch.int32:
        raise TypeError(f"{name} must use torch.int32")


@dataclass(slots=True)
class PrefillCSR:
    """Immutable community-to-prefill-position index for one layer."""

    member_offsets: torch.Tensor
    member_positions: torch.Tensor
    community_counts: torch.Tensor

    def validate(self) -> None:
        _require_int32("member_offsets", self.member_offsets)
        _require_int32("member_positions", self.member_positions)
        _require_int32("community_counts", self.community_counts)
        if self.member_offsets.ndim != 2 or self.member_positions.ndim != 2:
            raise ValueError("member offsets and positions must be rank two")
        if self.community_counts.shape != (self.member_offsets.shape[0],):
            raise ValueError("community_counts must have one entry per graph")
        if self.member_positions.shape[0] != self.member_offsets.shape[0]:
            raise ValueError("member_positions must have one row per graph")
        offsets = self.member_offsets.to(device="cpu", dtype=torch.int64)
        counts = self.community_counts.to(device="cpu", dtype=torch.int64)
        for graph in range(offsets.shape[0]):
            count = int(counts[graph])
            if count < 0 or count + 1 > offsets.shape[1]:
                raise ValueError("community count exceeds offset capacity")
            row = offsets[graph, : count + 1]
            if int(row[0]) != 0 or torch.any(row[1:] < row[:-1]):
                raise ValueError("active CSR offsets must start at zero and be monotonic")
            if int(row[-1]) > self.member_positions.shape[1]:
                raise ValueError("CSR offsets exceed member-position capacity")

    @property
    def graph_count(self) -> int:
        return self.member_offsets.shape[0]

    @property
    def community_capacity(self) -> int:
        return self.member_offsets.shape[1] - 1

    def members(self, graph: int, community: int) -> torch.Tensor:
        if graph < 0 or graph >= self.graph_count:
            raise IndexError("graph is outside the CSR")
        count = int(self.community_counts[graph])
        if community < 0 or community >= count:
            raise IndexError("community is outside the active CSR")
        start = int(self.member_offsets[graph, community])
        end = int(self.member_offsets[graph, community + 1])
        return self.member_positions[graph, start:end]


@dataclass(slots=True)
class DecodeDeltaChunks:
    """Single-writer geometric chunks for newly assigned token positions."""

    positions: torch.Tensor
    chunk_start: torch.Tensor
    chunk_capacity: torch.Tensor
    chunk_size: torch.Tensor
    chunk_next: torch.Tensor
    chunk_previous: torch.Tensor
    community_head: torch.Tensor
    community_tail: torch.Tensor
    community_sizes: torch.Tensor
    next_free_chunk: torch.Tensor
    next_free_position: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        graph_count: int,
        community_capacity: int,
        max_decode_tokens: int,
        device: torch.device | str = "cpu",
    ) -> "DecodeDeltaChunks":
        if min(graph_count, community_capacity, max_decode_tokens) <= 0:
            raise ValueError("all capacities must be positive")
        chunk_shape = (graph_count, max_decode_tokens)
        table_shape = (graph_count, community_capacity)
        return cls(
            positions=torch.full(
                (graph_count, 2 * max_decode_tokens),
                -1,
                dtype=torch.int32,
                device=device,
            ),
            chunk_start=torch.full(
                chunk_shape, -1, dtype=torch.int32, device=device
            ),
            chunk_capacity=torch.zeros(
                chunk_shape, dtype=torch.int32, device=device
            ),
            chunk_size=torch.zeros(
                chunk_shape, dtype=torch.int32, device=device
            ),
            chunk_next=torch.full(
                chunk_shape, -1, dtype=torch.int32, device=device
            ),
            chunk_previous=torch.full(
                chunk_shape, -1, dtype=torch.int32, device=device
            ),
            community_head=torch.full(
                table_shape, -1, dtype=torch.int32, device=device
            ),
            community_tail=torch.full(
                table_shape, -1, dtype=torch.int32, device=device
            ),
            community_sizes=torch.zeros(
                table_shape, dtype=torch.int32, device=device
            ),
            next_free_chunk=torch.zeros(
                (graph_count,), dtype=torch.int32, device=device
            ),
            next_free_position=torch.zeros(
                (graph_count,), dtype=torch.int32, device=device
            ),
        )

    @property
    def graph_count(self) -> int:
        return self.positions.shape[0]

    @property
    def position_capacity(self) -> int:
        return self.positions.shape[1]

    @property
    def chunk_capacity_per_graph(self) -> int:
        return self.chunk_start.shape[1]

    @property
    def community_capacity(self) -> int:
        return self.community_head.shape[1]

    def validate(self) -> None:
        for name, value in (
            ("positions", self.positions),
            ("chunk_start", self.chunk_start),
            ("chunk_capacity", self.chunk_capacity),
            ("chunk_size", self.chunk_size),
            ("chunk_next", self.chunk_next),
            ("chunk_previous", self.chunk_previous),
            ("community_head", self.community_head),
            ("community_tail", self.community_tail),
            ("community_sizes", self.community_sizes),
            ("next_free_chunk", self.next_free_chunk),
            ("next_free_position", self.next_free_position),
        ):
            _require_int32(name, value)
        if self.positions.ndim != 2 or self.chunk_start.ndim != 2:
            raise ValueError("delta position and chunk pools must be rank two")
        graph_count = self.positions.shape[0]
        chunk_shape = self.chunk_start.shape
        if chunk_shape[0] != graph_count or min(self.positions.shape[1:]) <= 0:
            raise ValueError("delta pool capacities must be positive")
        for value in (
            self.chunk_capacity,
            self.chunk_size,
            self.chunk_next,
            self.chunk_previous,
        ):
            if value.shape != chunk_shape:
                raise ValueError("chunk metadata shapes must match")
        if self.community_head.ndim != 2:
            raise ValueError("community chunk tables must be rank two")
        if self.community_head.shape != self.community_tail.shape:
            raise ValueError("community head and tail tables must match")
        if self.community_head.shape != self.community_sizes.shape:
            raise ValueError("community chunk tables and sizes must match")
        if self.community_head.shape[0] != graph_count:
            raise ValueError("community chunk tables must match graph count")
        if self.next_free_chunk.shape != (graph_count,):
            raise ValueError("next_free_chunk must have one entry per graph")
        if self.next_free_position.shape != (graph_count,):
            raise ValueError("next_free_position must have one entry per graph")

    def append(self, graph: int, community: int, position: int) -> None:
        """Append one position; callers guarantee one writer per graph."""

        if graph < 0 or graph >= self.graph_count:
            raise IndexError("graph is outside the delta store")
        if community < 0 or community >= self.community_capacity:
            raise IndexError("community is outside the delta store")
        tail = int(self.community_tail[graph, community])
        tail_size = int(self.chunk_size[graph, tail]) if tail >= 0 else 0
        tail_capacity = (
            int(self.chunk_capacity[graph, tail]) if tail >= 0 else 0
        )
        if tail < 0 or tail_size == tail_capacity:
            chunk = int(self.next_free_chunk[graph])
            capacity = 1 if tail < 0 else 2 * tail_capacity
            start = int(self.next_free_position[graph])
            if chunk >= self.chunk_capacity_per_graph:
                raise RuntimeError("decode-delta chunk capacity exhausted")
            if start + capacity > self.position_capacity:
                raise RuntimeError("decode-delta position capacity exhausted")
            self.next_free_chunk[graph] = chunk + 1
            self.next_free_position[graph] = start + capacity
            self.chunk_start[graph, chunk] = start
            self.chunk_capacity[graph, chunk] = capacity
            self.chunk_previous[graph, chunk] = tail
            if tail < 0:
                self.community_head[graph, community] = chunk
            else:
                self.chunk_next[graph, tail] = chunk
            self.community_tail[graph, community] = chunk
            tail = chunk
            tail_size = 0
        start = int(self.chunk_start[graph, tail])
        self.positions[graph, start + tail_size] = position
        self.chunk_size[graph, tail] = tail_size + 1
        self.community_sizes[graph, community] += 1

    def members(self, graph: int, community: int) -> torch.Tensor:
        size = int(self.community_sizes[graph, community])
        if size == 0:
            return torch.empty(
                (0,), dtype=torch.int32, device=self.positions.device
            )
        result = torch.empty(
            (size,), dtype=torch.int32, device=self.positions.device
        )
        chunk = int(self.community_head[graph, community])
        cursor = 0
        while chunk >= 0 and cursor < size:
            count = int(self.chunk_size[graph, chunk])
            start = int(self.chunk_start[graph, chunk])
            result[cursor : cursor + count] = self.positions[
                graph, start : start + count
            ]
            cursor += count
            chunk = int(self.chunk_next[graph, chunk])
        if cursor != size:
            raise RuntimeError("decode-delta chunk chain ended early")
        return result


@dataclass(slots=True)
class MutableGraphState:
    """Hot graph and centroid state mutated once per layer and decode token."""

    centroids: torch.Tensor
    community_sizes: torch.Tensor
    community_counts: torch.Tensor
    community_weight: torch.Tensor
    total_weight: torch.Tensor
    token_communities: torch.Tensor
    deltas: DecodeDeltaChunks
    retrieval_to_graph: torch.Tensor
    retrieval_to_kv: torch.Tensor

    @property
    def graph_count(self) -> int:
        return self.total_weight.shape[0]

    @property
    def retrieval_head_count(self) -> int:
        return self.centroids.shape[0]

    @property
    def community_capacity(self) -> int:
        return self.centroids.shape[1]

    @property
    def head_dim(self) -> int:
        return self.centroids.shape[2]

    def validate(self) -> None:
        if self.centroids.ndim != 3:
            raise ValueError("centroids must have shape [graphs, communities, dim]")
        if self.centroids.dtype != torch.float8_e4m3fn:
            raise TypeError("centroids must use FP8 E4M3")
        retrieval_head_count, community_capacity, _ = self.centroids.shape
        graph_count = self.graph_count
        if self.community_sizes.shape != (retrieval_head_count, community_capacity):
            raise ValueError("community_sizes must match centroid rows")
        _require_int32("community_sizes", self.community_sizes)
        if self.community_counts.shape != (retrieval_head_count,):
            raise ValueError("community_counts must have one entry per retrieval head")
        _require_int32("community_counts", self.community_counts)
        if self.community_weight.shape != (graph_count, community_capacity):
            raise ValueError("community_weight must have one row per graph")
        if self.community_weight.dtype != torch.float32:
            raise TypeError("community_weight must use float32")
        if self.total_weight.shape != (graph_count,):
            raise ValueError("total_weight must have one entry per graph")
        if self.total_weight.dtype != torch.float32:
            raise TypeError("total_weight must use float32")
        if self.token_communities.ndim != 2:
            raise ValueError("token_communities must be rank two")
        if self.token_communities.shape[0] != graph_count:
            raise ValueError("token_communities must have one row per graph")
        _require_int32("token_communities", self.token_communities)
        self.deltas.validate()
        if self.deltas.graph_count != graph_count:
            raise ValueError("delta pages must match graph count")
        if self.deltas.community_capacity != community_capacity:
            raise ValueError("delta pages must match community capacity")
        for name, mapping in (
            ("retrieval_to_graph", self.retrieval_to_graph),
            ("retrieval_to_kv", self.retrieval_to_kv),
        ):
            _require_int32(name, mapping)
            if mapping.shape != (retrieval_head_count,):
                raise ValueError(f"{name} must have one entry per retrieval head")
        if (
            int(self.retrieval_to_graph.min()) < 0
            or int(self.retrieval_to_graph.max()) >= graph_count
            or int(self.retrieval_to_kv.min()) < 0
        ):
            raise ValueError("retrieval mapping index is out of bounds")
        device = self.centroids.device
        if any(
            value.device != device
            for value in (
                self.community_sizes,
                self.community_counts,
                self.community_weight,
                self.total_weight,
                self.token_communities,
                self.retrieval_to_graph,
                self.retrieval_to_kv,
                self.deltas.positions,
                self.deltas.chunk_start,
                self.deltas.chunk_capacity,
                self.deltas.chunk_size,
                self.deltas.chunk_next,
                self.deltas.chunk_previous,
                self.deltas.community_head,
                self.deltas.community_tail,
                self.deltas.community_sizes,
                self.deltas.next_free_chunk,
                self.deltas.next_free_position,
            )
        ):
            raise ValueError("all mutable graph tensors must share one device")
