from __future__ import annotations

import pytest

from community_kv.config import CommunityKVConfig, GraphAggregation, PartitionConfig


def test_config_defaults_to_per_query_head() -> None:
    config = CommunityKVConfig()
    assert config.aggregation is GraphAggregation.PER_QUERY_HEAD
    assert PartitionConfig(devices=("cuda:1",)).workers_per_device == 2


@pytest.mark.parametrize("aggregation", list(GraphAggregation))
def test_config_accepts_all_graph_aggregations(
    aggregation: GraphAggregation,
) -> None:
    assert CommunityKVConfig(aggregation=aggregation).aggregation is aggregation
    assert CommunityKVConfig(aggregation=aggregation.value).aggregation is aggregation


def test_config_rejects_invalid_token_budget() -> None:
    with pytest.raises(ValueError):
        CommunityKVConfig(token_budget=1000)
