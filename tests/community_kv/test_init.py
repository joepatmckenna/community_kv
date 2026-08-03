import community_kv
from community_kv import (
    ATTENTION_IMPLEMENTATION,
    CommunityKVAttention,
    CommunityKVConfig,
    CommunityKVRuntime,
    GraphAggregation,
    PartitionConfig,
    StaticCache,
    partition_graphs,
    partition_query_groups,
    prefill_attention_topk,
)
from community_kv.attention import CommunityKVAttention as AttentionImplementation


def test_root_api_exposes_only_supported_operations_and_integration_types() -> None:
    assert set(community_kv.__all__) == {
        "ATTENTION_IMPLEMENTATION",
        "CommunityKVAttention",
        "CommunityKVConfig",
        "CommunityKVRuntime",
        "GraphAggregation",
        "HuggingFaceAttention",
        "PartitionConfig",
        "StaticCache",
        "StaticCacheLayer",
        "leiden_max_iterations",
        "partition_graphs",
        "partition_query_groups",
        "prefill_attention_topk",
    }
    assert CommunityKVAttention is AttentionImplementation
    assert ATTENTION_IMPLEMENTATION == "community_kv"
    assert all(
        value is not None
        for value in (
            CommunityKVConfig,
            CommunityKVRuntime,
            GraphAggregation,
            PartitionConfig,
            StaticCache,
            partition_graphs,
            partition_query_groups,
            prefill_attention_topk,
        )
    )


def test_root_api_does_not_expose_internal_graph_or_reference_helpers() -> None:
    for name in (
        "MutableGraphState",
        "PrefillCSR",
        "ordered_fp16_keys",
        "pack_segments",
        "weighted_prefix_select",
    ):
        assert not hasattr(community_kv, name)
