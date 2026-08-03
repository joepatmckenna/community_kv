"""Allocation-free graph-selective decode layer composition."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from community_kv.graph.state import MutableGraphState, PrefillCSR
from community_kv.attention.kernels.centroid_selection import (
    CentroidSelectionWorkspace,
    score_and_select,
)
from community_kv.graph.kernels.online_update import OnlineUpdateWorkspace
from community_kv.attention.kernels.segmented_attention import (
    TritonSegmentedAttentionWorkspace,
    triton_segmented_attention,
)


@dataclass(slots=True)
class DecodeLayerWorkspace:
    selection: CentroidSelectionWorkspace
    attention: TritonSegmentedAttentionWorkspace
    update: OnlineUpdateWorkspace

    @classmethod
    def allocate(
        cls,
        *,
        retrieval_head_count: int,
        graph_count: int,
        group_size: int = 4,
        community_capacity: int,
        token_budget: int,
        num_sink: int,
        dtype: torch.dtype,
        device: torch.device | str,
        score_community_counts: Sequence[int] | None = None,
        score_headroom: int = 0,
    ) -> "DecodeLayerWorkspace":
        return cls(
            selection=CentroidSelectionWorkspace.allocate(
                kv_heads=retrieval_head_count,
                community_capacity=community_capacity,
                token_budget=token_budget,
                num_sink=num_sink,
                score_community_counts=score_community_counts,
                score_headroom=score_headroom,
                device=device,
            ),
            attention=TritonSegmentedAttentionWorkspace.allocate(
                kv_heads=retrieval_head_count,
                group_size=group_size,
                token_budget=token_budget,
                dtype=dtype,
                device=device,
            ),
            update=OnlineUpdateWorkspace.allocate(
                graph_count=graph_count,
                device=device,
            ),
        )


def decode_layer(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    csr: PrefillCSR,
    state: MutableGraphState,
    sink_positions: torch.Tensor,
    current_position: torch.Tensor,
    softmax_scale: float,
    lam: float,
    workspace: DecodeLayerWorkspace,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run one complete CKV-specific layer transition."""

    segments = score_and_select(
        query=query,
        centroids=state.centroids,
        community_sizes=state.community_sizes,
        community_counts=state.community_counts,
        workspace=workspace.selection,
    )
    output, lse, topk_scores, topk_positions = triton_segmented_attention(
        query=query,
        key=key,
        value=value,
        csr=csr,
        deltas=state.deltas,
        segments=segments,
        sink_positions=sink_positions,
        current_position=current_position,
        softmax_scale=softmax_scale,
        workspace=workspace.attention,
        update_state=state,
        update_workspace=workspace.update,
        update_lam=lam,
    )
    assigned = workspace.update.assigned_communities
    return output, lse, topk_scores, topk_positions, assigned


__all__ = [
    "DecodeLayerWorkspace",
    "decode_layer",
]
