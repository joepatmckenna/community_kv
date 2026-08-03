"""Hugging Face attention adapter for CommunityKV."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from community_kv.attention.flash_attention import prefill_attention_topk

if TYPE_CHECKING:
    from community_kv.runtime import CommunityKVRuntime


ATTENTION_IMPLEMENTATION = "community_kv"
_TOPK_WIDTH = 8


class CommunityKVAttention:
    """Hugging Face attention adapter for CommunityKV."""

    IMPL_NAME = ATTENTION_IMPLEMENTATION

    def __init__(self, runtime: CommunityKVRuntime) -> None:
        self.runtime = runtime

    def register(self) -> str:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS[self.IMPL_NAME] = self.forward
        return self.IMPL_NAME

    def forward(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        del attention_mask, dropout, kwargs
        layer_idx = int(module.layer_idx)
        if query.shape[2] > 1:
            sequence_length = query.shape[2]
            if sequence_length < _TOPK_WIDTH + self.runtime.config.num_sink:
                raise ValueError("CommunityKV prefill is too short to build a graph")
            key = key[:, :, :sequence_length, :]
            value = value[:, :, :sequence_length, :]
            result = prefill_attention_topk(
                query.transpose(1, 2).contiguous(),
                key.transpose(1, 2).contiguous(),
                value.transpose(1, 2).contiguous(),
                softmax_scale=float(scaling),
                num_sink=self.runtime.config.num_sink,
            )
            with torch.cuda.device(key.device):
                completion = torch.cuda.Event(enable_timing=False)
                completion.record()
            self.runtime.submit_prefill_layer(
                layer_idx=layer_idx,
                topk_indices=result[5][0],
                topk_scores=result[4][0],
                keys=key[0],
                completion_event=completion,
            )
            return result[0], None

        self.runtime.ensure_ready()
        return (
            self.runtime.decode_runtime.decode(
                layer_idx=layer_idx,
                query=query,
                key=key,
                value=value,
                softmax_scale=float(scaling),
            ),
            None,
        )


HuggingFaceAttention = CommunityKVAttention


__all__ = [
    "ATTENTION_IMPLEMENTATION",
    "CommunityKVAttention",
    "HuggingFaceAttention",
]
