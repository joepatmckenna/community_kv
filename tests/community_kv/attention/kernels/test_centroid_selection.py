from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from community_kv.attention.kernels import centroid_selection


@dataclass(frozen=True, slots=True)
class _SelectionDescriptor:
    community_id: int
    take_count: int
    score_key: int


@dataclass(frozen=True, slots=True)
class _CommunitySelection:
    descriptors: tuple[_SelectionDescriptor, ...]
    token_count: int
    threshold_key: int | None


def _ordered_fp16_keys(scores: torch.Tensor) -> torch.Tensor:
    values = scores.detach().to(device="cpu", dtype=torch.float32).numpy()
    bits = values.astype(np.float16).view(np.uint16)
    flip = np.where((bits & np.uint16(0x8000)) != 0, 0xFFFF, 0x8000).astype(
        np.uint16
    )
    return torch.from_numpy(np.bitwise_xor(bits, flip).astype(np.int32))


def _weighted_prefix_select(
    scores: torch.Tensor,
    community_sizes: torch.Tensor,
    token_budget: int,
    *,
    active_communities: int | None = None,
) -> _CommunitySelection:
    active = scores.numel() if active_communities is None else active_communities
    sizes = community_sizes.detach().to(device="cpu", dtype=torch.int64)
    keys = _ordered_fp16_keys(scores[:active])
    ranked = sorted(
        (int(keys[index]), index, int(sizes[index]))
        for index in range(active)
        if int(sizes[index]) > 0
    )
    ranked.sort(key=lambda item: (-item[0], item[1]))

    remaining = token_budget
    descriptors: list[_SelectionDescriptor] = []
    for key, community_id, size in ranked:
        if remaining == 0:
            break
        take = min(size, remaining)
        descriptors.append(_SelectionDescriptor(community_id, take, key))
        remaining -= take
    threshold = descriptors[-1].score_key if descriptors else None
    return _CommunitySelection(
        tuple(descriptors),
        token_budget - remaining,
        threshold,
    )


def test_ordered_fp16_keys_preserve_numeric_order() -> None:
    scores = torch.tensor(
        [-float("inf"), -2.0, -0.0, 0.0, 1.0, float("inf")],
        dtype=torch.float32,
    )
    keys = _ordered_fp16_keys(scores)
    assert torch.all(keys[1:] >= keys[:-1])


def test_weighted_prefix_truncates_last_community() -> None:
    result = _weighted_prefix_select(
        torch.tensor([0.1, 0.9, 0.5]),
        torch.tensor([4, 7, 5], dtype=torch.int32),
        10,
    )
    assert [(item.community_id, item.take_count) for item in result.descriptors] == [
        (1, 7),
        (2, 3),
    ]
    assert result.token_count == 10


def test_weighted_prefix_uses_community_id_for_fp16_ties() -> None:
    scores = torch.tensor([1.0001, 1.0002, 1.0001])
    assert torch.unique(scores.to(torch.float16)).numel() == 1
    result = _weighted_prefix_select(
        scores,
        torch.tensor([1, 1, 1], dtype=torch.int32),
        2,
    )
    assert [item.community_id for item in result.descriptors] == [0, 1]


def test_weighted_prefix_ignores_zero_size_and_inactive_capacity() -> None:
    result = _weighted_prefix_select(
        torch.tensor([2.0, 3.0, 100.0, 200.0]),
        torch.tensor([2, 0, 9, 9], dtype=torch.int32),
        4,
        active_communities=2,
    )
    assert [(item.community_id, item.take_count) for item in result.descriptors] == [
        (0, 2)
    ]


def _workspace(monkeypatch) -> centroid_selection.CentroidSelectionWorkspace:
    monkeypatch.setattr(
        centroid_selection,
        "triton",
        SimpleNamespace(cdiv=lambda value, block: (value + block - 1) // block),
    )
    return centroid_selection.CentroidSelectionWorkspace.allocate(
        kv_heads=2,
        community_capacity=32,
        token_budget=128,
        num_sink=10,
        score_community_counts=(8, 12),
        score_headroom=2,
        device="cpu",
    )


def test_workspace_allocation_builds_fixed_descriptor_contract(monkeypatch) -> None:
    workspace = _workspace(monkeypatch)

    assert workspace.scores.shape == (2, 32)
    assert workspace.score_schedule.tolist() == [0, 1 << 24]
    assert workspace.native_histogram.shape == (2, 16, 256)
    assert workspace.selected_budget == 117
    assert workspace.packed_segments().tile_descriptors.shape == (2, 2)


def test_workspace_rejects_invalid_specialization(monkeypatch) -> None:
    monkeypatch.setattr(
        centroid_selection,
        "triton",
        SimpleNamespace(cdiv=lambda value, block: (value + block - 1) // block),
    )
    with pytest.raises(ValueError, match="multiple of 64"):
        centroid_selection.CentroidSelectionWorkspace.allocate(
            kv_heads=1,
            community_capacity=8,
            token_budget=96,
            num_sink=1,
            device="cpu",
        )


def test_native_selector_forwards_preallocated_outputs(monkeypatch) -> None:
    workspace = _workspace(monkeypatch)
    calls = {}

    class _Extension:
        @staticmethod
        def select_weighted_descriptors(*args):
            calls["args"] = args

    monkeypatch.setattr(
        centroid_selection,
        "import_module",
        lambda name: _Extension,
    )
    sizes = torch.ones_like(workspace.scores, dtype=torch.int32)
    counts = torch.tensor([8, 12], dtype=torch.int32)

    packed = centroid_selection.select_weighted_descriptors_native(
        community_sizes=sizes,
        community_counts=counts,
        workspace=workspace,
    )

    assert packed.communities is workspace.communities
    assert calls["args"][0] is workspace.scores
    assert calls["args"][-1] is False


def test_auto_selector_honors_workspace_dispatch(monkeypatch) -> None:
    workspace = _workspace(monkeypatch)
    sizes = torch.ones_like(workspace.scores, dtype=torch.int32)
    counts = torch.tensor([8, 12], dtype=torch.int32)
    monkeypatch.setattr(
        centroid_selection,
        "select_weighted_descriptors_cluster",
        lambda **kwargs: "cluster",
    )
    monkeypatch.setattr(
        centroid_selection,
        "select_weighted_descriptors_native",
        lambda **kwargs: "native",
    )

    workspace.use_cluster_selector = True
    assert centroid_selection.select_weighted_descriptors_auto(
        community_sizes=sizes,
        community_counts=counts,
        workspace=workspace,
    ) == "cluster"
    workspace.use_cluster_selector = False
    assert centroid_selection.select_weighted_descriptors_auto(
        community_sizes=sizes,
        community_counts=counts,
        workspace=workspace,
    ) == "native"


def test_score_centroids_validates_query_geometry_before_launch(monkeypatch) -> None:
    workspace = _workspace(monkeypatch)
    with pytest.raises(ValueError, match="query heads"):
        centroid_selection.score_centroids(
            query=torch.empty((3, 64), dtype=torch.bfloat16),
            centroids=torch.empty((2, 32, 128), dtype=torch.float8_e4m3fn),
            community_sizes=torch.ones((2, 32), dtype=torch.int32),
            community_counts=torch.tensor([8, 12], dtype=torch.int32),
            workspace=workspace,
        )


def test_score_centroids_requires_fp8_storage(monkeypatch) -> None:
    workspace = _workspace(monkeypatch)
    with pytest.raises(TypeError, match="FP8 E4M3"):
        centroid_selection.score_centroids(
            query=torch.empty((8, 128), dtype=torch.bfloat16),
            centroids=torch.empty((2, 32, 128), dtype=torch.bfloat16),
            community_sizes=torch.ones((2, 32), dtype=torch.int32),
            community_counts=torch.tensor([8, 12], dtype=torch.int32),
            workspace=workspace,
        )
