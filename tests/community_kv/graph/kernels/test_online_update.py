from __future__ import annotations

import pytest
import torch

from community_kv.graph.kernels.online_update import OnlineUpdateWorkspace


def test_workspace_allocates_one_result_and_overflow_flag_per_graph() -> None:
    workspace = OnlineUpdateWorkspace.allocate(graph_count=3, device="cpu")

    assert workspace.assigned_communities.shape == (3,)
    assert workspace.assigned_communities.dtype == torch.int32
    assert workspace.overflow.tolist() == [0, 0, 0]


def test_workspace_requires_a_positive_graph_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        OnlineUpdateWorkspace.allocate(graph_count=0, device="cpu")
