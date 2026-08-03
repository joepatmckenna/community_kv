"""Fixed-address BF16 Hugging Face cache for prefill and CUDA-graph decode."""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache, StaticLayer


class StaticCacheLayer(StaticLayer):
    """Static backing storage with host prefill and device decode cursors."""

    def __init__(self, max_cache_len: int) -> None:
        super().__init__(max_cache_len=max_cache_len)
        self.valid_length = 0
        self.device_updates = False
        self.cumulative_length = torch.zeros(1, dtype=torch.int32)

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        if key_states.dtype != torch.bfloat16 or value_states.dtype != torch.bfloat16:
            raise TypeError("CommunityKV cache requires BF16 K/V")
        super().lazy_initialization(key_states, value_states)
        self.valid_length = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        if self.device_updates:
            length = key_states.shape[-2]
            positions = (
                torch.arange(length, device=self.device) + self.cumulative_length
            )
            self.cumulative_length.add_(length)
            self.keys.index_copy_(2, positions, key_states)
            self.values.index_copy_(2, positions, value_states)
            return self.keys, self.values

        start = self.valid_length
        end = start + key_states.shape[-2]
        if end > self.max_cache_len:
            raise ValueError(f"KV cache capacity exceeded: {end} > {self.max_cache_len}")
        self.keys[:, :, start:end, :].copy_(key_states)
        self.values[:, :, start:end, :].copy_(value_states)
        self.valid_length = end
        self.cumulative_length.fill_(end)
        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        if self.device_updates:
            return self.max_cache_len, 0
        return min(self.valid_length + query_length, self.max_cache_len), 0

    def get_seq_length(self) -> int | torch.Tensor:
        if self.device_updates and self.is_initialized:
            return self.cumulative_length
        return self.valid_length if self.is_initialized else 0

    def enable_device_updates(self) -> None:
        if not self.is_initialized:
            raise RuntimeError("cache layer must be initialized before graph decode")
        with torch.inference_mode():
            self.cumulative_length.fill_(self.valid_length)
        self.device_updates = True

    def sync_valid_length(self) -> None:
        if self.is_initialized:
            self.valid_length = int(self.cumulative_length.item())

    def reset(self) -> None:
        if self.is_initialized:
            with torch.inference_mode():
                self.keys.zero_()
                self.values.zero_()
                self.cumulative_length.zero_()
        self.valid_length = 0
        self.device_updates = False


class StaticCache(Cache):
    """One fixed-capacity cache layer per transformer layer."""

    def __init__(
        self,
        *,
        num_hidden_layers: int,
        max_cache_len: int,
    ) -> None:
        super().__init__(
            layers=[
                StaticCacheLayer(max_cache_len=max_cache_len)
                for _ in range(num_hidden_layers)
            ]
        )

    def layer(self, layer_idx: int) -> StaticCacheLayer:
        layer = self.layers[layer_idx]
        if not isinstance(layer, StaticCacheLayer):
            raise TypeError(f"unexpected cache layer: {type(layer).__name__}")
        return layer

    def valid_length(self, layer_idx: int) -> int:
        return self.layer(layer_idx).valid_length

    def enable_device_updates(self) -> None:
        for layer_idx in range(len(self.layers)):
            self.layer(layer_idx).enable_device_updates()

    def sync_valid_lengths(self) -> None:
        for layer_idx in range(len(self.layers)):
            self.layer(layer_idx).sync_valid_length()


__all__ = ["StaticCache", "StaticCacheLayer"]
