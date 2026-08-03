"""Binding for the patched FlashAttention prefill kernel."""

from __future__ import annotations

import functools
from typing import Any

import torch


@functools.cache
def _load_flash_attention() -> Any:
    try:
        from . import _C  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "CommunityKV requires its patched FlashAttention extension. "
            "Reinstall CommunityKV in a supported CUDA build environment."
        ) from error
    return torch.ops.flash_attn_3.fwd


def prefill_attention_topk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    num_sink: int = 10,
) -> tuple[Any, ...]:
    """Run causal PackGQA FlashAttention and retain row-wise top-8 scores."""

    if query.stride(-1) != 1:
        query = query.contiguous()
    if key.stride(-1) != 1:
        key = key.contiguous()
    if value.stride(-1) != 1 and value.stride(-3) != 1:
        value = value.contiguous()

    out, softmax_lse, *rest = _load_flash_attention()(
        query,
        key,
        value,
        softmax_scale=softmax_scale,
        is_causal=True,
        is_rotary_interleaved=True,
        num_splits=1,
        pack_gqa=True,
        exclude_sink_tokens=num_sink,
    )
    return out, softmax_lse, *rest


__all__ = ["prefill_attention_topk"]
