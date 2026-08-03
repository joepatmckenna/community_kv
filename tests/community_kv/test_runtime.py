from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from community_kv import GraphAggregation, StaticCache
from community_kv.runtime import (
    CommunityKVRuntime,
    _effective_token_budget,
    convert_prefill_tensors,
)


class _GenerateModel:
    def __init__(self, events: list[object]) -> None:
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            _attn_implementation="community_kv",
        )
        self.generation_config = SimpleNamespace(
            max_new_tokens=None,
            max_length=20,
            num_beams=1,
            num_return_sequences=1,
            disable_compile=None,
        )
        self.events = events
        self.call: tuple[tuple[object, ...], dict[str, object]] | None = None

    def generate(self, *args: object, **kwargs: object) -> torch.Tensor:
        self.events.append("generate")
        self.call = (args, kwargs)
        return torch.tensor([[1, 2, 3, 4]])


def _generate_runtime(events: list[object]) -> CommunityKVRuntime:
    runtime = object.__new__(CommunityKVRuntime)
    runtime.num_layers = 2
    runtime._generation_cache = None
    runtime.start_request = lambda **kwargs: events.append(("start", kwargs))
    runtime.finish_prefill = lambda: events.append("finish")
    runtime.assert_no_overflow = lambda: events.append("overflow")
    return runtime


def test_effective_token_budget_caps_short_requests_to_full_tiles() -> None:
    assert _effective_token_budget(
        configured_budget=4096,
        prefill_seq_len=3806,
        num_sink=10,
    ) == 3776
    assert _effective_token_budget(
        configured_budget=4096,
        prefill_seq_len=4095,
        num_sink=10,
    ) == 4096


def test_effective_token_budget_keeps_configured_upper_bound() -> None:
    assert _effective_token_budget(
        configured_budget=2048,
        prefill_seq_len=8192,
        num_sink=10,
    ) == 2048


def test_runtime_context_manager_closes_partition_runtime() -> None:
    closed = []
    runtime = object.__new__(CommunityKVRuntime)
    runtime.partition_runtime = SimpleNamespace(
        shutdown=lambda: closed.append(True)
    )

    with runtime as active:
        assert active is runtime

    assert closed == [True]


def test_generate_manages_cache_and_request_lifecycle() -> None:
    events: list[object] = []
    runtime = _generate_runtime(events)
    model = _GenerateModel(events)
    input_ids = torch.tensor([[1, 2, 3]])
    resolutions = [1.0, 2.0]

    output = runtime.generate(
        model,
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=4,
        resolutions=resolutions,
    )

    torch.testing.assert_close(output, torch.tensor([[1, 2, 3, 4]]))
    assert events == [
        ("start", {"resolutions": resolutions, "max_decode_tokens": 3}),
        "generate",
        "finish",
        "overflow",
    ]
    assert model.call is not None
    args, kwargs = model.call
    assert args == ()
    cache = kwargs["past_key_values"]
    assert isinstance(cache, StaticCache)
    assert cache.layer(0).max_cache_len == 7
    assert kwargs["input_ids"] is input_ids
    assert kwargs["do_sample"] is False
    assert kwargs["num_beams"] == 1
    assert kwargs["num_return_sequences"] == 1
    assert kwargs["disable_compile"] is True
    assert runtime._generation_cache is None


def test_finish_prefill_enables_generation_cache_updates() -> None:
    runtime = object.__new__(CommunityKVRuntime)
    runtime._ready = False
    runtime.num_layers = 2
    runtime.decode_runtime = SimpleNamespace(
        layers={0: object(), 1: object()},
        prepare_partitioned_layer=lambda layer: None,
    )
    runtime.partition_runtime = SimpleNamespace(wait=lambda callback: None)
    enabled = []
    runtime._generation_cache = SimpleNamespace(
        enable_device_updates=lambda: enabled.append(True)
    )

    runtime.finish_prefill()

    assert runtime._ready
    assert enabled == [True]


def test_generate_accepts_positional_inputs_and_configured_max_length() -> None:
    events: list[object] = []
    runtime = _generate_runtime(events)
    model = _GenerateModel(events)
    input_ids = torch.tensor([[1, 2, 3]])

    runtime.generate(model, input_ids)

    assert events[0] == (
        "start",
        {"resolutions": None, "max_decode_tokens": 16},
    )
    assert model.call is not None
    args, kwargs = model.call
    assert args == (input_ids,)
    cache = kwargs["past_key_values"]
    assert isinstance(cache, StaticCache)
    assert cache.layer(0).max_cache_len == 20


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"input_ids": torch.ones((2, 3), dtype=torch.long)}, "batch size 1"),
        ({"num_beams": 2}, "beam search"),
        ({"num_return_sequences": 2}, "one returned sequence"),
        ({"assistant_model": object()}, "assisted generation"),
        ({"penalty_alpha": 0.6}, "contrastive search"),
        ({"do_sample": True}, "greedy decoding"),
        ({"past_key_values": object()}, "manages past_key_values"),
    ],
)
def test_generate_rejects_unsupported_modes(
    updates: dict[str, object],
    message: str,
) -> None:
    events: list[object] = []
    runtime = _generate_runtime(events)
    model = _GenerateModel(events)
    arguments: dict[str, object] = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "max_new_tokens": 4,
    }
    arguments.update(updates)

    with pytest.raises(ValueError, match=message):
        runtime.generate(model, **arguments)

    assert events == []


def test_generate_requires_registered_attention() -> None:
    events: list[object] = []
    runtime = _generate_runtime(events)
    model = _GenerateModel(events)
    model.config._attn_implementation = "sdpa"

    with pytest.raises(ValueError, match="_attn_implementation"):
        runtime.generate(
            model,
            input_ids=torch.tensor([[1, 2, 3]]),
            max_new_tokens=4,
        )

    assert events == []


def test_convert_prefill_tensors_pads_decode_community_offsets() -> None:
    graphs = 2
    prefill_communities = 3
    community_capacity = 7
    prefill_seq_len = 8
    decode_capacity = 4
    member_offsets = torch.tensor(
        [[0, 2, 4, 6], [0, 1, 3, 6]],
        dtype=torch.int32,
    )
    member_positions = torch.tensor(
        [[2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7]],
        dtype=torch.int32,
    )
    num_communities = torch.full(
        (graphs,),
        prefill_communities,
        dtype=torch.int32,
    )
    centroids = torch.randn(
        (graphs, community_capacity, 128),
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    community_sizes = torch.zeros(
        (graphs, community_capacity),
        dtype=torch.int32,
    )
    community_sizes[:, :prefill_communities] = torch.tensor(
        [[2, 2, 2], [1, 2, 3]],
        dtype=torch.int32,
    )
    community_weight = torch.zeros(
        (graphs, community_capacity),
        dtype=torch.float32,
    )
    total_weight = torch.ones((graphs,), dtype=torch.float32)
    token_communities = torch.full(
        (graphs, prefill_seq_len + decode_capacity),
        -1,
        dtype=torch.int32,
    )
    token_communities[:, 2:prefill_seq_len] = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2]],
        dtype=torch.int32,
    )

    converted = convert_prefill_tensors(
        member_offsets=member_offsets,
        member_positions=member_positions,
        num_communities=num_communities,
        centroids=centroids,
        community_sizes=community_sizes,
        community_weight=community_weight,
        total_weight=total_weight,
        token_communities=token_communities,
        prefill_seq_len=prefill_seq_len,
        max_decode_tokens=decode_capacity,
    )

    assert converted.csr.member_offsets.shape == (
        graphs,
        community_capacity + 1,
    )
    torch.testing.assert_close(
        converted.csr.member_offsets[:, : prefill_communities + 1],
        member_offsets,
    )
    torch.testing.assert_close(
        converted.csr.member_offsets[:, prefill_communities + 1 :],
        torch.full(
            (graphs, community_capacity - prefill_communities),
            6,
            dtype=torch.int32,
        ),
    )
    assert converted.mutable.centroids.data_ptr() == centroids.data_ptr()
    assert converted.mutable.community_sizes.data_ptr() == community_sizes.data_ptr()
    assert converted.current_position.tolist() == [prefill_seq_len] * graphs
    assert converted.initial_community_counts == (3, 3)
    assert converted.mutable.deltas.position_capacity == 2 * decode_capacity
    assert converted.group_size == 4


def test_convert_prefill_tensors_supports_shared_layer_graph() -> None:
    centroids = torch.zeros((2, 5, 128), dtype=torch.float8_e4m3fn)
    converted = convert_prefill_tensors(
        member_offsets=torch.tensor([[0, 2, 4]], dtype=torch.int32),
        member_positions=torch.tensor([[4, 3, 2, 1]], dtype=torch.int32),
        num_communities=torch.tensor([2], dtype=torch.int32),
        centroids=centroids,
        community_sizes=torch.tensor(
            [[2, 2, 0, 0, 0], [2, 2, 0, 0, 0]],
            dtype=torch.int32,
        ),
        community_weight=torch.zeros((1, 5), dtype=torch.float32),
        total_weight=torch.ones(1, dtype=torch.float32),
        token_communities=torch.tensor(
            [[-1, 0, 1, 1, 0, -1, -1]],
            dtype=torch.int32,
        ),
        retrieval_to_graph=torch.tensor([0, 0], dtype=torch.int32),
        retrieval_to_kv=torch.tensor([0, 1], dtype=torch.int32),
        prefill_seq_len=5,
        max_decode_tokens=2,
        group_size=2,
        aggregation=GraphAggregation.LAYER_WISE,
        query_heads=4,
        kv_heads=2,
    )

    assert converted.mutable.graph_count == 1
    assert converted.mutable.retrieval_head_count == 2
    assert converted.mutable.community_counts.tolist() == [2, 2]
    assert converted.current_position.tolist() == [5]
    assert converted.aggregation is GraphAggregation.LAYER_WISE
