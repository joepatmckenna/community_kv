"""Grouped centroid scoring and exact weighted descriptor selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only development hosts.
    triton = None
    tl = None

from community_kv.attention.kernels.packed_segments import (
    ATTENTION_TILE,
    PackedSegments,
)

HEAD_DIM = 128
HISTOGRAM_BINS = 256
SELECTOR_PARTS = 16
SCORE_BLOCK_N = 64
CLUSTER_SELECTOR_MAX_COMMUNITIES = 65_536


if triton is not None:

    @triton.jit
    def _grouped_centroid_score_kernel(
        query,
        centroids,
        community_sizes,
        community_counts,
        scores,
        community_capacity,
        stride_q_h,
        stride_q_d,
        stride_c_h,
        stride_c_n,
        stride_c_d,
        stride_sizes_h,
        stride_sizes_n,
        stride_scores_h,
        stride_scores_n,
        HEAD_DIM_CONST: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        FP8_DOT: tl.constexpr,
    ):
        kv_head = tl.program_id(0)
        community_start = tl.program_id(1) * BLOCK_N
        query_lane = tl.arange(0, BLOCK_M)
        community = community_start + tl.arange(0, BLOCK_N)
        dim = tl.arange(0, BLOCK_D)

        query_mask = query_lane < GROUP_SIZE
        active_communities = tl.load(community_counts + kv_head)
        community_mask = (community < active_communities) & (
            community < community_capacity
        )
        dim_mask = dim < HEAD_DIM_CONST
        query_head = kv_head * GROUP_SIZE + query_lane

        q = tl.load(
            query + query_head[:, None] * stride_q_h + dim[None, :] * stride_q_d,
            mask=query_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        c = tl.load(
            centroids
            + kv_head * stride_c_h
            + community[:, None] * stride_c_n
            + dim[None, :] * stride_c_d,
            mask=community_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        if FP8_DOT:
            score_matrix = tl.dot(q.to(tl.float8e4nv), tl.trans(c))
        else:
            score_matrix = tl.dot(q, tl.trans(c.to(tl.bfloat16)))

        score_matrix = tl.where(
            query_mask[:, None],
            score_matrix,
            -float("inf"),
        )
        reduced = tl.max(score_matrix, axis=0)
        size = tl.load(
            community_sizes + kv_head * stride_sizes_h + community * stride_sizes_n,
            mask=community_mask,
            other=0,
        )
        reduced = tl.where(
            community_mask & (size > 0),
            reduced,
            -float("inf"),
        )
        tl.store(
            scores + kv_head * stride_scores_h + community * stride_scores_n,
            reduced,
            mask=community < community_capacity,
        )

    @triton.jit
    def _ragged_grouped_centroid_score_kernel(
        query,
        centroids,
        community_sizes,
        community_counts,
        score_schedule,
        scores,
        community_capacity,
        stride_q_h,
        stride_q_d,
        stride_c_h,
        stride_c_n,
        stride_c_d,
        stride_sizes_h,
        stride_sizes_n,
        stride_scores_h,
        stride_scores_n,
        HEAD_DIM_CONST: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        FP8_DOT: tl.constexpr,
    ):
        encoded_block = tl.load(score_schedule + tl.program_id(0))
        kv_head = encoded_block >> 24
        community_start = (encoded_block & 0xFFFFFF) * BLOCK_N
        query_lane = tl.arange(0, BLOCK_M)
        community = community_start + tl.arange(0, BLOCK_N)
        dim = tl.arange(0, BLOCK_D)

        query_mask = query_lane < GROUP_SIZE
        active_communities = tl.load(community_counts + kv_head)
        community_mask = (community < active_communities) & (
            community < community_capacity
        )
        dim_mask = dim < HEAD_DIM_CONST
        query_head = kv_head * GROUP_SIZE + query_lane

        q = tl.load(
            query + query_head[:, None] * stride_q_h + dim[None, :] * stride_q_d,
            mask=query_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        c = tl.load(
            centroids
            + kv_head * stride_c_h
            + community[:, None] * stride_c_n
            + dim[None, :] * stride_c_d,
            mask=community_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        if FP8_DOT:
            score_matrix = tl.dot(q.to(tl.float8e4nv), tl.trans(c))
        else:
            score_matrix = tl.dot(q, tl.trans(c.to(tl.bfloat16)))

        score_matrix = tl.where(
            query_mask[:, None],
            score_matrix,
            -float("inf"),
        )
        reduced = tl.max(score_matrix, axis=0)
        size = tl.load(
            community_sizes + kv_head * stride_sizes_h + community * stride_sizes_n,
            mask=community_mask,
            other=0,
        )
        reduced = tl.where(
            community_mask & (size > 0),
            reduced,
            -float("inf"),
        )
        tl.store(
            scores + kv_head * stride_scores_h + community * stride_scores_n,
            reduced,
            mask=community < community_capacity,
        )


@dataclass(slots=True)
class CentroidSelectionWorkspace:
    """Preallocated scoring and descriptor-selection outputs."""

    scores: torch.Tensor
    score_schedule: torch.Tensor | None
    native_histogram: torch.Tensor
    native_threshold: torch.Tensor
    native_part_state: torch.Tensor
    communities: torch.Tensor
    cumulative_ends: torch.Tensor
    descriptor_counts: torch.Tensor
    tile_descriptors: torch.Tensor
    token_budget: int
    num_sink: int
    use_cluster_selector: bool

    @classmethod
    def allocate(
        cls,
        *,
        kv_heads: int,
        community_capacity: int,
        token_budget: int,
        num_sink: int,
        score_community_counts: Sequence[int] | None = None,
        score_headroom: int = 0,
        device: torch.device | str,
    ) -> "CentroidSelectionWorkspace":
        if triton is None:
            raise RuntimeError("Triton is unavailable")
        if kv_heads <= 0 or community_capacity <= 0:
            raise ValueError("head and community capacities must be positive")
        if token_budget <= 0 or token_budget % ATTENTION_TILE:
            raise ValueError("token budget must be a positive multiple of 64")
        if num_sink < 0 or num_sink + 1 >= token_budget:
            raise ValueError("num_sink leaves no selected-token capacity")
        if score_headroom < 0:
            raise ValueError("score_headroom must be nonnegative")
        score_schedule = None
        score_coverage = None
        if score_community_counts is not None:
            if len(score_community_counts) != kv_heads:
                raise ValueError("score counts must contain one value per KV head")
            encoded_blocks = []
            score_coverage = []
            for kv_head, count in enumerate(score_community_counts):
                if count <= 0 or count > community_capacity:
                    raise ValueError("score counts must fit the community capacity")
                covered = min(community_capacity, count + score_headroom)
                score_coverage.append(covered)
                block_count = triton.cdiv(covered, SCORE_BLOCK_N)
                encoded_blocks.extend(
                    (kv_head << 24) | block for block in range(block_count)
                )
            score_schedule = torch.tensor(
                encoded_blocks,
                dtype=torch.int32,
                device=device,
            )
        use_cluster_selector = (
            sum(score_coverage) <= kv_heads * CLUSTER_SELECTOR_MAX_COMMUNITIES
            if score_coverage is not None
            else community_capacity <= CLUSTER_SELECTOR_MAX_COMMUNITIES
        )
        selected_budget = token_budget - num_sink - 1
        tile_count = token_budget // ATTENTION_TILE
        return cls(
            scores=torch.empty(
                (kv_heads, community_capacity),
                dtype=torch.float16,
                device=device,
            ),
            score_schedule=score_schedule,
            native_histogram=torch.empty(
                (kv_heads, SELECTOR_PARTS, HISTOGRAM_BINS),
                dtype=torch.int32,
                device=device,
            ),
            native_threshold=torch.full(
                (kv_heads, 4),
                -1,
                dtype=torch.int32,
                device=device,
            ),
            native_part_state=torch.empty(
                (kv_heads, SELECTOR_PARTS, 8),
                dtype=torch.int32,
                device=device,
            ),
            communities=torch.full(
                (kv_heads, selected_budget),
                -1,
                dtype=torch.int32,
                device=device,
            ),
            cumulative_ends=torch.zeros(
                (kv_heads, selected_budget),
                dtype=torch.int32,
                device=device,
            ),
            descriptor_counts=torch.zeros(
                (kv_heads,),
                dtype=torch.int32,
                device=device,
            ),
            tile_descriptors=torch.zeros(
                (kv_heads, tile_count),
                dtype=torch.int32,
                device=device,
            ),
            token_budget=token_budget,
            num_sink=num_sink,
            use_cluster_selector=use_cluster_selector,
        )

    @property
    def selected_budget(self) -> int:
        return self.token_budget - self.num_sink - 1

    @property
    def community_capacity(self) -> int:
        return self.scores.shape[1]

    @property
    def kv_heads(self) -> int:
        return self.scores.shape[0]

    def packed_segments(self) -> PackedSegments:
        return PackedSegments(
            communities=self.communities,
            cumulative_ends=self.cumulative_ends,
            descriptor_counts=self.descriptor_counts,
            tile_descriptors=self.tile_descriptors,
            token_budget=self.token_budget,
            num_sink=self.num_sink,
        )


def score_centroids(
    *,
    query: torch.Tensor,
    centroids: torch.Tensor,
    community_sizes: torch.Tensor,
    community_counts: torch.Tensor,
    workspace: CentroidSelectionWorkspace,
) -> torch.Tensor:
    """Score each contiguous query group against its centroid table."""

    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if (
        query.ndim != 2
        or query.shape[1] != HEAD_DIM
        or query.shape[0] % workspace.kv_heads
    ):
        raise ValueError("query heads must divide into [KV head, group, 128]")
    group_size = query.shape[0] // workspace.kv_heads
    if not 1 <= group_size <= 16:
        raise ValueError("GQA group size must be in [1, 16]")
    if centroids.shape != (
        workspace.kv_heads,
        workspace.community_capacity,
        HEAD_DIM,
    ):
        raise ValueError("centroids have the wrong fixed capacity")
    if community_sizes.shape != centroids.shape[:2]:
        raise ValueError("community_sizes must match centroid rows")
    if community_counts.shape != (workspace.kv_heads,):
        raise ValueError("community_counts must have one entry per KV head")
    if query.dtype != torch.bfloat16:
        raise TypeError("query must use BF16")
    if centroids.dtype != torch.float8_e4m3fn:
        raise TypeError("centroids must use FP8 E4M3")
    if community_sizes.dtype != torch.int32:
        raise TypeError("community_sizes must use int32")
    if community_counts.dtype != torch.int32:
        raise TypeError("community_counts must use int32")
    if any(
        value.device != query.device
        for value in (
            centroids,
            community_sizes,
            community_counts,
            workspace.scores,
        )
    ):
        raise ValueError("all score tensors must share one device")
    if (
        workspace.score_schedule is not None
        and workspace.score_schedule.device != query.device
    ):
        raise ValueError("the score schedule must share the query device")

    score_kernel = _grouped_centroid_score_kernel
    grid = (
        (workspace.kv_heads, triton.cdiv(workspace.community_capacity, SCORE_BLOCK_N))
        if workspace.score_schedule is None
        else (workspace.score_schedule.numel(),)
    )
    if workspace.score_schedule is None:
        score_kernel[grid](
            query,
            centroids,
            community_sizes,
            community_counts,
            workspace.scores,
            workspace.community_capacity,
            query.stride(0),
            query.stride(1),
            centroids.stride(0),
            centroids.stride(1),
            centroids.stride(2),
            community_sizes.stride(0),
            community_sizes.stride(1),
            workspace.scores.stride(0),
            workspace.scores.stride(1),
            HEAD_DIM_CONST=HEAD_DIM,
            GROUP_SIZE=group_size,
            BLOCK_M=16,
            BLOCK_N=SCORE_BLOCK_N,
            BLOCK_D=HEAD_DIM,
            FP8_DOT=False,
            num_warps=4,
            num_stages=3,
        )
        return workspace.scores

    _ragged_grouped_centroid_score_kernel[grid](
        query,
        centroids,
        community_sizes,
        community_counts,
        workspace.score_schedule,
        workspace.scores,
        workspace.community_capacity,
        query.stride(0),
        query.stride(1),
        centroids.stride(0),
        centroids.stride(1),
        centroids.stride(2),
        community_sizes.stride(0),
        community_sizes.stride(1),
        workspace.scores.stride(0),
        workspace.scores.stride(1),
        HEAD_DIM_CONST=HEAD_DIM,
        GROUP_SIZE=group_size,
        BLOCK_M=16,
        BLOCK_N=SCORE_BLOCK_N,
        BLOCK_D=HEAD_DIM,
        FP8_DOT=False,
        num_warps=4,
        num_stages=3,
    )
    return workspace.scores


def select_weighted_descriptors_native(
    *,
    community_sizes: torch.Tensor,
    community_counts: torch.Tensor,
    workspace: CentroidSelectionWorkspace,
) -> PackedSegments:
    """Run the multi-CTA Hopper descriptor selector."""

    if community_sizes.shape != workspace.scores.shape:
        raise ValueError("community_sizes must match the score capacity")
    if community_sizes.dtype != torch.int32:
        raise TypeError("community_sizes must use int32")
    if community_counts.shape != (workspace.kv_heads,):
        raise ValueError("community_counts must have one entry per KV head")
    if community_counts.dtype != torch.int32:
        raise TypeError("community_counts must use int32")
    extension = import_module("community_kv.attention.kernels._selection_native")
    extension.select_weighted_descriptors(
        workspace.scores,
        community_sizes,
        community_counts,
        workspace.native_histogram,
        workspace.native_threshold,
        workspace.native_part_state,
        workspace.communities,
        workspace.cumulative_ends,
        workspace.descriptor_counts,
        workspace.tile_descriptors,
        workspace.num_sink,
        False,
    )
    return workspace.packed_segments()


def select_weighted_descriptors_cluster(
    *,
    community_sizes: torch.Tensor,
    community_counts: torch.Tensor,
    workspace: CentroidSelectionWorkspace,
) -> PackedSegments:
    """Run the single-launch eight-CTA Hopper cluster selector."""

    if community_sizes.shape != workspace.scores.shape:
        raise ValueError("community_sizes must match the score capacity")
    if community_sizes.dtype != torch.int32:
        raise TypeError("community_sizes must use int32")
    if community_counts.shape != (workspace.kv_heads,):
        raise ValueError("community_counts must have one entry per KV head")
    if community_counts.dtype != torch.int32:
        raise TypeError("community_counts must use int32")
    extension = import_module("community_kv.attention.kernels._selection_native")
    extension.select_weighted_descriptors(
        workspace.scores,
        community_sizes,
        community_counts,
        workspace.native_histogram,
        workspace.native_threshold,
        workspace.native_part_state,
        workspace.communities,
        workspace.cumulative_ends,
        workspace.descriptor_counts,
        workspace.tile_descriptors,
        workspace.num_sink,
        True,
    )
    return workspace.packed_segments()


def select_weighted_descriptors_auto(
    *,
    community_sizes: torch.Tensor,
    community_counts: torch.Tensor,
    workspace: CentroidSelectionWorkspace,
) -> PackedSegments:
    """Dispatch to the fastest exact selector for the fixed state capacity."""

    if workspace.use_cluster_selector:
        return select_weighted_descriptors_cluster(
            community_sizes=community_sizes,
            community_counts=community_counts,
            workspace=workspace,
        )
    return select_weighted_descriptors_native(
        community_sizes=community_sizes,
        community_counts=community_counts,
        workspace=workspace,
    )


def score_and_select(
    *,
    query: torch.Tensor,
    centroids: torch.Tensor,
    community_sizes: torch.Tensor,
    community_counts: torch.Tensor,
    workspace: CentroidSelectionWorkspace,
) -> PackedSegments:
    score_centroids(
        query=query,
        centroids=centroids,
        community_sizes=community_sizes,
        community_counts=community_counts,
        workspace=workspace,
    )
    return select_weighted_descriptors_auto(
        community_sizes=community_sizes,
        community_counts=community_counts,
        workspace=workspace,
    )


__all__ = [
    "CentroidSelectionWorkspace",
    "score_and_select",
    "score_centroids",
    "select_weighted_descriptors_auto",
    "select_weighted_descriptors_cluster",
    "select_weighted_descriptors_native",
]
