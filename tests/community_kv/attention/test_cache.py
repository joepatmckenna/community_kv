from __future__ import annotations

import pytest
import torch

from community_kv.attention.cache import StaticCache, StaticCacheLayer


def _states(length: int, value: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.full((1, 2, length, 4), value, dtype=torch.bfloat16)
    return key, key + 1


def test_static_layer_tracks_host_and_device_updates() -> None:
    layer = StaticCacheLayer(5)
    key, value = _states(2)

    stored_key, stored_value = layer.update(key, value)
    assert stored_key.shape == stored_value.shape == (1, 2, 5, 4)
    assert layer.valid_length == 2
    assert layer.get_mask_sizes(1) == (3, 0)

    layer.enable_device_updates()
    next_key, next_value = _states(1, 4.0)
    full_key, full_value = layer.update(next_key, next_value)
    assert full_key.shape == full_value.shape == (1, 2, 5, 4)
    assert int(layer.get_seq_length().item()) == 3
    layer.sync_valid_length()
    assert layer.valid_length == 3


def test_static_layer_enforces_bf16_and_capacity() -> None:
    layer = StaticCacheLayer(2)
    invalid = torch.zeros((1, 1, 1, 4))
    with pytest.raises(TypeError, match="BF16"):
        layer.update(invalid, invalid)

    layer.update(*_states(2))
    with pytest.raises(ValueError, match="capacity exceeded"):
        layer.update(*_states(1))


def test_static_cache_controls_all_layers() -> None:
    cache = StaticCache(num_hidden_layers=2, max_cache_len=4)
    for index in range(2):
        cache.layer(index).update(*_states(1, float(index + 1)))

    cache.enable_device_updates()
    assert all(cache.layer(index).device_updates for index in range(2))
    cache.sync_valid_lengths()
    assert [cache.valid_length(index) for index in range(2)] == [1, 1]
    cache.reset()
    assert [cache.valid_length(index) for index in range(2)] == [0, 0]
