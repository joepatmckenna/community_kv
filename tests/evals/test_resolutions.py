from __future__ import annotations

import logging
from pathlib import Path

import pytest

from community_kv import GraphAggregation
from evals.resolutions import (
    RESOLUTION_SCHEMA,
    load_model_resolutions,
    load_resolution_table,
    model_resolutions,
    write_model_resolutions,
)


ROOT = Path(__file__).parents[2]


def _calibration(aggregation: GraphAggregation) -> dict:
    return {
        "dataset": "emozilla/pg19-test",
        "split": "test",
        "prompt_rows": [0, 2],
        "prompt_sha256": ["a" * 64, "b" * 64],
        "prompt_tokens": [8192, 65_536],
        "target_mean_non_sink_community_size": 16.0,
        "aggregation": "geometric_median",
        "graph_aggregation": aggregation.value,
        "prompt_selection": "first_qualifying_rows_first_tokens",
        "dataset_revision": "main",
        "model_revision": "main",
        "kappa": 8,
        "num_sink": 10,
        "lam": 0.5,
        "resolution_min": 0.001,
        "resolution_max": 5000.0,
        "tolerance": 0.05,
        "max_steps": 18,
        "leiden_max_iter": [3, 4],
        "leiden_seed": 0,
        "rope_factor": [1.0, 2.0],
        "model_profile": "model",
        "model_geometry": {
            "model_type": "test",
            "num_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "hidden_size": 1024,
            "head_dim": 128,
            "max_position_embeddings": 65_536,
        },
    }


def test_staged_resolution_table_uses_per_model_per_layer_schema() -> None:
    table = load_resolution_table(ROOT / "evals/resolutions.json")
    assert table["_meta"]["schema"] == RESOLUTION_SCHEMA
    expected_layers = {
        "Qwen/Qwen3-4B": 36,
        "Qwen/Qwen3-8B": 36,
        "Qwen/Qwen3-14B": 40,
        "meta-llama/Llama-3.1-8B-Instruct": 32,
    }
    assert {
        model: {
            aggregation: len(values["resolutions"])
            for aggregation, values in entry["aggregations"].items()
        }
        for model, entry in table["models"].items()
    } == {
        model: {
            "query_group": layers,
            "per_query_head": layers,
        }
        for model, layers in expected_layers.items()
    }


def test_load_model_resolutions_uses_packaged_table() -> None:
    assert len(
        load_model_resolutions(
            "Qwen/Qwen3-8B",
            num_layers=36,
        )
    ) == 36


@pytest.mark.parametrize(
    "aggregation",
    [GraphAggregation.PER_QUERY_HEAD, GraphAggregation.QUERY_GROUP],
)
def test_write_and_load_model_vector(
    tmp_path: Path,
    aggregation: GraphAggregation,
) -> None:
    path = tmp_path / "resolutions.json"
    write_model_resolutions(
        path,
        model="org/model",
        aggregation=aggregation,
        resolutions=[1.5, 2.5],
        calibration=_calibration(aggregation),
    )
    table = load_resolution_table(path)
    assert model_resolutions(
        table,
        "org/model",
        aggregation=aggregation,
        num_layers=2,
    ) == [1.5, 2.5]


def test_layer_count_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "resolutions.json"
    write_model_resolutions(
        path,
        model="org/model",
        aggregation=GraphAggregation.QUERY_GROUP,
        resolutions=[1.5, 2.5],
        calibration=_calibration(GraphAggregation.QUERY_GROUP),
    )
    with pytest.raises(ValueError, match="expected 3"):
        model_resolutions(
            load_resolution_table(path),
            "org/model",
            aggregation=GraphAggregation.QUERY_GROUP,
            num_layers=3,
        )


def test_prompt_metadata_lengths_must_align(tmp_path: Path) -> None:
    calibration = _calibration(GraphAggregation.PER_QUERY_HEAD)
    calibration["prompt_tokens"] = [8192]
    with pytest.raises(ValueError, match="prompt_tokens must match prompt_rows"):
        write_model_resolutions(
            tmp_path / "resolutions.json",
            model="org/model",
            aggregation=GraphAggregation.PER_QUERY_HEAD,
            resolutions=[1.5, 2.5],
            calibration=calibration,
        )


def test_rope_factor_metadata_lengths_must_align(tmp_path: Path) -> None:
    calibration = _calibration(GraphAggregation.PER_QUERY_HEAD)
    calibration["rope_factor"] = [1.0]
    with pytest.raises(ValueError, match="rope_factor must be positive or match"):
        write_model_resolutions(
            tmp_path / "resolutions.json",
            model="org/model",
            aggregation=GraphAggregation.PER_QUERY_HEAD,
            resolutions=[1.5, 2.5],
            calibration=calibration,
        )


def test_untuned_layer_wise_mode_logs_explicit_scalar_fallback(caplog) -> None:
    table = load_resolution_table(ROOT / "evals/resolutions.json")
    with caplog.at_level(logging.WARNING, logger="evals.resolutions"):
        assert model_resolutions(
            table,
            "Qwen/Qwen3-8B",
            aggregation=GraphAggregation.LAYER_WISE,
            num_layers=36,
            fallback=1.25,
        ) == [1.25] * 36
    assert (
        "No layer_wise resolutions for model 'Qwen/Qwen3-8B'; "
        "using fallback 1.25 for all 36 layers"
    ) in caplog.messages
