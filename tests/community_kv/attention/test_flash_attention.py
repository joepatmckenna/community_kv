from __future__ import annotations

import torch

from community_kv.attention import flash_attention


def test_prefill_attention_passes_flash_attention_contract(monkeypatch) -> None:
    captured_args = []
    captured_kwargs = {}

    def forward(*args, **kwargs):
        captured_args.extend(args)
        captured_kwargs.update(kwargs)
        return "output", "lse", "topk"

    monkeypatch.setattr(flash_attention, "_load_flash_attention", lambda: forward)
    query = torch.randn(1, 4, 128, 2).transpose(-1, -2)
    key = torch.randn(1, 4, 128, 1).transpose(-1, -2)
    value = torch.empty((1, 4, 1, 128))

    result = flash_attention.prefill_attention_topk(
        query,
        key,
        value,
        softmax_scale=0.125,
        num_sink=10,
    )

    assert result == ("output", "lse", "topk")
    assert len(captured_args) == 3
    assert captured_args[0].is_contiguous()
    assert captured_args[1].is_contiguous()
    assert captured_kwargs == {
        "softmax_scale": 0.125,
        "is_causal": True,
        "is_rotary_interleaved": True,
        "num_splits": 1,
        "pack_gqa": True,
        "exclude_sink_tokens": 10,
    }
