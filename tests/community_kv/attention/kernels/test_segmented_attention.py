from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from community_kv.attention.kernels import segmented_attention


def test_workspace_allocation_has_fixed_kernel_shapes(monkeypatch) -> None:
    monkeypatch.setattr(segmented_attention, "triton", SimpleNamespace())
    workspace = segmented_attention.TritonSegmentedAttentionWorkspace.allocate(
        kv_heads=2,
        group_size=5,
        token_budget=128,
        dtype=torch.bfloat16,
        device="cpu",
    )

    assert workspace.partial_max.shape == (10, 2)
    assert workspace.partial_acc.shape == (10, 2, 128)
    assert workspace.partial_topk_packed.shape == (2, 2, 8)
    assert workspace.output.shape == (10, 128)
    assert workspace.topk_positions.dtype == torch.int32


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kv_heads": 0}, "kv_heads"),
        ({"group_size": 17}, "group_size"),
        ({"token_budget": 96}, "multiple of 64"),
        ({"dtype": torch.float32}, "BF16"),
    ],
)
def test_workspace_rejects_unsupported_specializations(
    monkeypatch,
    kwargs,
    message,
) -> None:
    monkeypatch.setattr(segmented_attention, "triton", SimpleNamespace())
    arguments = {
        "kv_heads": 1,
        "group_size": 4,
        "token_budget": 128,
        "dtype": torch.bfloat16,
        "device": "cpu",
    }
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        segmented_attention.TritonSegmentedAttentionWorkspace.allocate(**arguments)


def test_attention_validates_query_shape_before_kernel_launch(monkeypatch) -> None:
    monkeypatch.setattr(segmented_attention, "triton", SimpleNamespace())
    workspace = segmented_attention.TritonSegmentedAttentionWorkspace.allocate(
        kv_heads=1,
        token_budget=64,
        dtype=torch.bfloat16,
        device="cpu",
    )

    with pytest.raises(ValueError, match="heads per retrieval head"):
        segmented_attention.triton_segmented_attention(
            query=torch.empty((3, 128), dtype=torch.bfloat16),
            key=torch.empty((1, 8, 128), dtype=torch.bfloat16),
            value=torch.empty((1, 8, 128), dtype=torch.bfloat16),
            csr=None,
            deltas=None,
            segments=None,
            sink_positions=torch.empty(0, dtype=torch.int32),
            current_position=torch.zeros(1, dtype=torch.int32),
            softmax_scale=1.0,
            workspace=workspace,
            update_state=None,
            update_workspace=None,
            update_lam=0.5,
        )
