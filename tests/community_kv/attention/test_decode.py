from __future__ import annotations

from types import SimpleNamespace

import torch

from community_kv.attention import decode


def test_workspace_allocation_composes_kernel_workspaces(monkeypatch) -> None:
    calls = []

    def allocate(name):
        def inner(**kwargs):
            calls.append((name, kwargs))
            return name

        return inner

    monkeypatch.setattr(
        decode.CentroidSelectionWorkspace,
        "allocate",
        allocate("selection"),
    )
    monkeypatch.setattr(
        decode.TritonSegmentedAttentionWorkspace,
        "allocate",
        allocate("attention"),
    )
    monkeypatch.setattr(
        decode.OnlineUpdateWorkspace,
        "allocate",
        allocate("update"),
    )

    workspace = decode.DecodeLayerWorkspace.allocate(
        retrieval_head_count=2,
        graph_count=2,
        group_size=5,
        community_capacity=32,
        token_budget=128,
        num_sink=10,
        dtype=torch.bfloat16,
        device="cpu",
    )

    assert (workspace.selection, workspace.attention, workspace.update) == (
        "selection",
        "attention",
        "update",
    )
    assert [name for name, _ in calls] == ["selection", "attention", "update"]
    assert calls[1][1]["group_size"] == 5
    assert calls[2][1]["graph_count"] == 2


def test_decode_layer_composes_selection_attention_and_update(monkeypatch) -> None:
    segments = object()
    assigned = torch.tensor([3], dtype=torch.int32)
    workspace = SimpleNamespace(
        selection=object(),
        attention=object(),
        update=SimpleNamespace(assigned_communities=assigned),
    )
    tensors = tuple(torch.tensor([float(index)]) for index in range(4))
    captured = {}

    monkeypatch.setattr(
        decode,
        "score_and_select",
        lambda **kwargs: captured.setdefault("selection", kwargs) and segments,
    )

    def fake_attention(**kwargs):
        captured["attention"] = kwargs
        return tensors

    monkeypatch.setattr(decode, "triton_segmented_attention", fake_attention)
    state = SimpleNamespace(
        centroids=torch.empty(0),
        community_sizes=torch.empty(0),
        community_counts=torch.empty(0),
        deltas=object(),
    )

    result = decode.decode_layer(
        query=torch.empty(0),
        key=torch.empty(0),
        value=torch.empty(0),
        csr=object(),
        state=state,
        sink_positions=torch.empty(0),
        current_position=torch.empty(0),
        softmax_scale=0.25,
        lam=0.5,
        workspace=workspace,
    )

    assert result == (*tensors, assigned)
    assert captured["attention"]["segments"] is segments
    assert captured["attention"]["update_state"] is state
    assert captured["attention"]["update_workspace"] is workspace.update
