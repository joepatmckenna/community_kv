from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from community_kv.attention import adapter


class _Event:
    def __init__(self, *, enable_timing: bool) -> None:
        assert enable_timing is False
        self.recorded = False

    def record(self) -> None:
        self.recorded = True


class _Runtime:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_sink=2)
        self.submitted = None
        self.ready = False
        self.decode_runtime = SimpleNamespace(decode=self._decode)

    def submit_prefill_layer(self, **kwargs) -> None:
        self.submitted = kwargs

    def ensure_ready(self) -> None:
        self.ready = True

    def _decode(self, **kwargs) -> torch.Tensor:
        self.decode_kwargs = kwargs
        return torch.tensor([7.0])


def test_register_installs_hugging_face_attention_function(monkeypatch) -> None:
    import transformers.modeling_utils as modeling_utils

    registry = {}
    monkeypatch.setattr(modeling_utils, "ALL_ATTENTION_FUNCTIONS", registry)
    attention = adapter.CommunityKVAttention(_Runtime())

    assert attention.register() == adapter.ATTENTION_IMPLEMENTATION
    assert registry[adapter.ATTENTION_IMPLEMENTATION] == attention.forward


def test_prefill_runs_flash_attention_and_submits_partition(monkeypatch) -> None:
    runtime = _Runtime()
    attention = adapter.CommunityKVAttention(runtime)
    query = torch.randn(1, 4, 12, 128)
    key = torch.randn(1, 1, 14, 128)
    value = torch.randn_like(key)
    output = torch.randn(1, 12, 4, 128)
    scores = torch.randn(1, 4, 12, 8)
    indices = torch.zeros_like(scores, dtype=torch.int32)
    calls = {}

    def fake_prefill(q, k, v, **kwargs):
        calls.update(q=q, k=k, v=v, **kwargs)
        return output, None, None, None, scores, indices

    monkeypatch.setattr(adapter, "prefill_attention_topk", fake_prefill)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", _Event)

    actual, weights = attention.forward(
        SimpleNamespace(layer_idx=3),
        query,
        key,
        value,
        None,
        0.125,
    )

    assert actual is output
    assert weights is None
    assert calls["q"].shape == (1, 12, 4, 128)
    assert calls["k"].shape == (1, 12, 1, 128)
    assert calls["softmax_scale"] == 0.125
    assert "kappa" not in calls
    assert runtime.submitted["layer_idx"] == 3
    torch.testing.assert_close(runtime.submitted["topk_indices"], indices[0])
    assert runtime.submitted["completion_event"].recorded


def test_decode_waits_for_partitions_and_dispatches() -> None:
    runtime = _Runtime()
    attention = adapter.CommunityKVAttention(runtime)
    query = torch.randn(1, 4, 1, 128)
    key = torch.randn(1, 1, 8, 128)

    output, weights = attention.forward(
        SimpleNamespace(layer_idx=2),
        query,
        key,
        key,
        None,
        0.5,
    )

    assert runtime.ready
    assert output.tolist() == [7.0]
    assert weights is None
    assert runtime.decode_kwargs["layer_idx"] == 2


def test_prefill_rejects_sequences_too_short_for_graph() -> None:
    runtime = _Runtime()
    attention = adapter.CommunityKVAttention(runtime)
    query = torch.randn(1, 4, 9, 128)
    key = torch.randn(1, 1, 9, 128)

    with pytest.raises(ValueError, match="too short"):
        attention.forward(
            SimpleNamespace(layer_idx=0),
            query,
            key,
            key,
            None,
            1.0,
        )
