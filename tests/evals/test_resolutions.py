"""Tests for evals.resolutions — load + key construction + filtering."""

from __future__ import annotations

import json

import pytest

from evals import resolutions

KEY_QWEN_4B_SHORT = (
    "model=Qwen/Qwen3-4B|dataset=LongBench-v2:short|"
    "agg=query_group,kappa=8,num_sink=10,lam=0.5,target=16.0|"
    "sample=aaaaaaaaaaaaaaaaaaaaaaaa"
)
KEY_QWEN_8B_SHORT = (
    "model=Qwen/Qwen3-8B|dataset=LongBench-v2:short|"
    "agg=query_group,kappa=8,num_sink=10,lam=0.5,target=16.0|"
    "sample=bbbbbbbbbbbbbbbbbbbbbbbb"
)
KEY_QWEN_4B_LONG = (
    "model=Qwen/Qwen3-4B|dataset=LongBench-v2:long|"
    "agg=query_group,kappa=8,num_sink=10,lam=0.5,target=16.0|"
    "sample=cccccccccccccccccccccccc"
)


@pytest.fixture
def synthetic_table():
    return {
        KEY_QWEN_4B_SHORT: 0.31,
        KEY_QWEN_8B_SHORT: 0.45,
        KEY_QWEN_4B_LONG: 0.27,
    }


class TestMakeKey:
    def test_round_trip_with_filter(self, synthetic_table):
        key = resolutions.make_key(
            model="Qwen/Qwen3-4B",
            dataset="LongBench-v2",
            split="short",
            aggregation="query_group",
            kappa=8,
            num_sink=10,
            lam=0.5,
            target=16.0,
            sample_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        )
        assert key == KEY_QWEN_4B_SHORT
        assert synthetic_table[key] == 0.31


class TestFilterTable:
    def test_picks_only_matching_prefix(self, synthetic_table):
        out = resolutions.filter_table(
            synthetic_table,
            model="Qwen/Qwen3-4B",
            dataset_lookup_name="LongBench-v2:short",
            aggregation="query_group",
            kappa=8,
            num_sink=10,
            lam=0.5,
            target=16.0,
        )
        assert out == {"aaaaaaaaaaaaaaaaaaaaaaaa": 0.31}

    def test_returns_empty_when_no_match(self, synthetic_table):
        out = resolutions.filter_table(
            synthetic_table,
            model="Qwen/Qwen3-99B",  # not present
            dataset_lookup_name="LongBench-v2:short",
            aggregation="query_group",
            kappa=8,
            num_sink=10,
            lam=0.5,
            target=16.0,
        )
        assert out == {}

    def test_split_change_partitions_correctly(self, synthetic_table):
        out_long = resolutions.filter_table(
            synthetic_table,
            model="Qwen/Qwen3-4B",
            dataset_lookup_name="LongBench-v2:long",
            aggregation="query_group",
            kappa=8,
            num_sink=10,
            lam=0.5,
            target=16.0,
        )
        assert out_long == {"cccccccccccccccccccccccc": 0.27}


class TestLoad:
    def test_load_custom_path(self, tmp_path, synthetic_table):
        p = tmp_path / "tiny.json"
        p.write_text(json.dumps(synthetic_table))
        loaded = resolutions.load(p)
        assert loaded == synthetic_table

    def test_load_default_bundled(self):
        """The bundled file ships as a non-empty nested-per-sample-per-layer
        table with a flat top-level config."""
        loaded = resolutions.load()
        assert isinstance(loaded, dict)
        assert resolutions.is_nested(loaded)
        assert "config" in loaded
        assert "context_window" in loaded["config"]  # flat (no sub-object)
        models = [k for k in loaded if k not in ("_meta", "config")]
        assert models
        assert all(isinstance(loaded[m], dict) and loaded[m] for m in models)


class TestNestedReader:
    @pytest.fixture
    def nested_table(self):
        return {
            "_meta": {"schema": "nested-per-sample-per-layer-v1"},
            "config": {
                "context_extension_strategy": "middle_out",
                "context_window": 131072,
                "rope_factor": 4,
                "rope_original_max": 32768,
                "aggregation": "per_query_head",
                "kappa": 8,
                "num_sink": 10,
                "lam": 0.5,
                "target_avg_community_size": 16.0,
            },
            "Qwen/Qwen3-4B": {
                "LongBench-v2": {"short": {"sid1": [1.0, 2.0, None, 4.0]}},
            },
            "Qwen/Qwen3-8B": {
                "babilong": {"qa1": {"64k": {"sidB": [5.0, 6.0]}}},
            },
        }

    def test_is_nested(self, nested_table, synthetic_table):
        assert resolutions.is_nested(nested_table)
        assert not resolutions.is_nested(synthetic_table)

    def test_config_and_context_extension(self, nested_table):
        assert resolutions.tune_config(nested_table)["target_avg_community_size"] == 16.0
        ce = resolutions.context_extension(nested_table)
        assert ce["context_window"] == 131072
        assert set(ce) == {
            "context_extension_strategy",
            "context_window",
            "rope_factor",
            "rope_original_max",
        }
        assert resolutions.context_extension({}) is None
        assert resolutions.tune_config({}) is None

    def test_longbench_single_level_split(self, nested_table):
        d = resolutions.layer_resolutions(
            nested_table, model="Qwen/Qwen3-4B", lookup_name="LongBench-v2:short", sample_id="sid1"
        )
        # None entries (unfilled layers) are dropped
        assert d == {0: 1.0, 1: 2.0, 3: 4.0}

    def test_babilong_nested_task_length_split(self, nested_table):
        d = resolutions.layer_resolutions(
            nested_table, model="Qwen/Qwen3-8B", lookup_name="babilong:qa1:64k", sample_id="sidB"
        )
        assert d == {0: 5.0, 1: 6.0}

    def test_missing_returns_none(self, nested_table):
        assert (
            resolutions.layer_resolutions(
                nested_table,
                model="Qwen/Qwen3-4B",
                lookup_name="LongBench-v2:short",
                sample_id="nope",
            )
            is None
        )
        assert (
            resolutions.layer_resolutions(
                nested_table,
                model="Qwen/Qwen3-99B",
                lookup_name="LongBench-v2:short",
                sample_id="sid1",
            )
            is None
        )
