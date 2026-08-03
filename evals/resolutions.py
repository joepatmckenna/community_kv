"""Validated per-model CommunityKV resolution tables."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from community_kv import GraphAggregation, leiden_max_iterations


LOGGER = logging.getLogger(__name__)
RESOLUTION_SCHEMA = "community-kv-per-model-per-layer-v3"
DEFAULT_RESOLUTIONS = Path(__file__).with_name("resolutions.json")
CALIBRATION_FIELDS = {
    "aggregation",
    "dataset",
    "dataset_revision",
    "graph_aggregation",
    "kappa",
    "lam",
    "leiden_max_iter",
    "leiden_seed",
    "max_steps",
    "model_geometry",
    "model_profile",
    "model_revision",
    "num_sink",
    "prompt_rows",
    "prompt_selection",
    "prompt_sha256",
    "prompt_tokens",
    "resolution_max",
    "resolution_min",
    "rope_factor",
    "split",
    "target_mean_non_sink_community_size",
    "tolerance",
}
MODEL_GEOMETRY_FIELDS = {
    "head_dim",
    "hidden_size",
    "max_position_embeddings",
    "model_type",
    "num_attention_heads",
    "num_key_value_heads",
    "num_layers",
}


def empty_resolution_table() -> dict[str, Any]:
    return {
        "_meta": {
            "schema": RESOLUTION_SCHEMA,
            "description": (
                "Frozen Leiden resolutions per model, layer, and aggregation."
            ),
        },
        "models": {},
    }


def _positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _validate_calibration(
    model: str,
    aggregation: GraphAggregation,
    calibration: object,
) -> None:
    prefix = f"{model}/{aggregation.value}"
    if not isinstance(calibration, dict):
        raise ValueError(f"{prefix}: calibration metadata is required")
    if set(calibration) != CALIBRATION_FIELDS:
        missing = sorted(CALIBRATION_FIELDS - set(calibration))
        extra = sorted(set(calibration) - CALIBRATION_FIELDS)
        raise ValueError(
            f"{prefix}: calibration fields do not match the schema "
            f"(missing={missing}, extra={extra})"
        )
    if calibration.get("dataset") != "emozilla/pg19-test":
        raise ValueError(f"{prefix}: calibration dataset must be PG19")
    if calibration.get("split") != "test":
        raise ValueError(f"{prefix}: PG19 calibration split must be test")
    if calibration.get("aggregation") != "geometric_median":
        raise ValueError(f"{prefix}: prompt resolutions must use geometric_median")
    if calibration.get("graph_aggregation") != aggregation.value:
        raise ValueError(f"{prefix}: graph aggregation metadata does not match")
    if calibration.get("prompt_selection") not in {
        "first_qualifying_rows_first_tokens",
        "first_qualifying_rows_middle_out",
        "first_qualifying_rows_by_context_first_tokens",
        "first_qualifying_rows_by_context_middle_out",
    }:
        raise ValueError(f"{prefix}: unsupported PG19 prompt selection")
    if calibration.get("kappa") != 8:
        raise ValueError(f"{prefix}: calibration kappa must be 8")
    if (
        not isinstance(calibration.get("num_sink"), int)
        or isinstance(calibration.get("num_sink"), bool)
        or calibration["num_sink"] < 0
    ):
        raise ValueError(f"{prefix}: calibration.num_sink must be non-negative")
    lam = calibration.get("lam")
    if (
        not isinstance(lam, (int, float))
        or isinstance(lam, bool)
        or not math.isfinite(float(lam))
        or not 0 <= float(lam) <= 1
    ):
        raise ValueError(f"{prefix}: calibration.lam must be in [0, 1]")
    for key in ("dataset_revision", "model_revision"):
        if calibration.get(key) != "main":
            raise ValueError(f"{prefix}: calibration.{key} must use main HEAD")
    if (
        not isinstance(calibration.get("leiden_seed"), int)
        or isinstance(calibration.get("leiden_seed"), bool)
        or calibration["leiden_seed"] < 0
    ):
        raise ValueError(f"{prefix}: calibration.leiden_seed must be non-negative")
    for key in (
        "target_mean_non_sink_community_size",
        "resolution_min",
        "resolution_max",
        "tolerance",
        "max_steps",
    ):
        if not _positive_number(calibration.get(key)):
            raise ValueError(f"{prefix}: calibration.{key} must be positive")
    rows = calibration.get("prompt_rows")
    hashes = calibration.get("prompt_sha256")
    token_counts = calibration.get("prompt_tokens")
    max_iterations = calibration.get("leiden_max_iter")
    rope_factors = calibration.get("rope_factor")
    if (
        not isinstance(rows, list)
        or not rows
        or not all(
            isinstance(row, int) and not isinstance(row, bool) and row >= 0
            for row in rows
        )
    ):
        raise ValueError(f"{prefix}: calibration.prompt_rows must contain row indices")
    if (
        not isinstance(hashes, list)
        or len(hashes) != len(rows)
        or not all(
            isinstance(digest, str)
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest)
            for digest in hashes
        )
    ):
        raise ValueError(f"{prefix}: calibration.prompt_sha256 must match prompt_rows")
    if (
        not isinstance(token_counts, list)
        or len(token_counts) != len(rows)
        or not all(
            isinstance(tokens, int)
            and not isinstance(tokens, bool)
            and tokens > 0
            for tokens in token_counts
        )
    ):
        raise ValueError(f"{prefix}: calibration.prompt_tokens must match prompt_rows")
    expected_max_iterations = [
        leiden_max_iterations(tokens) for tokens in token_counts
    ]
    if max_iterations != expected_max_iterations:
        raise ValueError(
            f"{prefix}: calibration.leiden_max_iter must be "
            f"{expected_max_iterations}"
        )
    if _positive_number(rope_factors):
        pass
    elif (
        not isinstance(rope_factors, list)
        or len(rope_factors) != len(rows)
        or not all(_positive_number(factor) for factor in rope_factors)
    ):
        raise ValueError(
            f"{prefix}: calibration.rope_factor must be positive or match prompt_rows"
        )
    profile = calibration.get("model_profile")
    if not isinstance(profile, str) or not profile:
        raise ValueError(f"{prefix}: calibration.model_profile must be non-empty")
    geometry = calibration.get("model_geometry")
    if not isinstance(geometry, dict) or set(geometry) != MODEL_GEOMETRY_FIELDS:
        raise ValueError(f"{prefix}: calibration.model_geometry is invalid")
    if not isinstance(geometry["model_type"], str) or not geometry["model_type"]:
        raise ValueError(f"{prefix}: model geometry type must be non-empty")
    for key in MODEL_GEOMETRY_FIELDS - {"model_type"}:
        if (
            not isinstance(geometry[key], int)
            or isinstance(geometry[key], bool)
            or geometry[key] <= 0
        ):
            raise ValueError(f"{prefix}: model geometry {key} must be positive")


def validate_resolution_table(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("resolution table must be a JSON object")
    meta = payload.get("_meta")
    if not isinstance(meta, dict) or meta.get("schema") != RESOLUTION_SCHEMA:
        actual = meta.get("schema") if isinstance(meta, dict) else None
        raise ValueError(
            f"expected resolution schema {RESOLUTION_SCHEMA!r}, got {actual!r}"
        )
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError("resolution table must contain a models object")

    for model, entry in models.items():
        if not isinstance(model, str) or not model:
            raise ValueError("model identifiers must be non-empty strings")
        if not isinstance(entry, dict):
            raise ValueError(f"{model}: model entry must be an object")
        aggregations = entry.get("aggregations")
        if not isinstance(aggregations, dict) or not aggregations:
            raise ValueError(f"{model}: aggregations must be a non-empty object")
        for name, aggregation_entry in aggregations.items():
            try:
                aggregation = GraphAggregation(name)
            except ValueError as error:
                raise ValueError(f"{model}: unknown aggregation {name!r}") from error
            if not isinstance(aggregation_entry, dict):
                raise ValueError(f"{model}/{name}: entry must be an object")
            values = aggregation_entry.get("resolutions")
            if not isinstance(values, list) or not values:
                raise ValueError(f"{model}/{name}: resolutions must be non-empty")
            if not all(_positive_number(value) for value in values):
                raise ValueError(
                    f"{model}/{name}: every resolution must be finite and positive"
                )
            calibration = aggregation_entry.get("calibration")
            _validate_calibration(
                model,
                aggregation,
                calibration,
            )
            if len(values) != calibration["model_geometry"]["num_layers"]:
                raise ValueError(
                    f"{model}/{name}: resolution count does not match model geometry"
                )
    return payload


def load_resolution_table(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return validate_resolution_table(json.load(handle))


def model_resolutions(
    table: dict[str, Any],
    model: str,
    *,
    aggregation: GraphAggregation = GraphAggregation.PER_QUERY_HEAD,
    num_layers: int | None = None,
    fallback: float | None = None,
) -> list[float]:
    validate_resolution_table(table)
    aggregation = GraphAggregation(aggregation)
    try:
        values = table["models"][model]["aggregations"][aggregation.value][
            "resolutions"
        ]
    except KeyError as error:
        if fallback is None or num_layers is None:
            raise KeyError(
                f"no {aggregation.value} resolutions for model {model!r}"
            ) from error
        if not _positive_number(fallback):
            raise ValueError("fallback resolution must be finite and positive")
        LOGGER.warning(
            "No %s resolutions for model %r; using fallback %.6g for all %d layers",
            aggregation.value,
            model,
            fallback,
            num_layers,
        )
        return [float(fallback)] * num_layers
    if num_layers is not None and len(values) != num_layers:
        raise ValueError(
            f"{model}: expected {num_layers} layer resolutions, found {len(values)}"
        )
    return [float(value) for value in values]


def load_model_resolutions(
    model: str,
    *,
    aggregation: GraphAggregation = GraphAggregation.PER_QUERY_HEAD,
    num_layers: int | None = None,
    fallback: float | None = None,
    path: str | Path = DEFAULT_RESOLUTIONS,
) -> list[float]:
    """Load one model and aggregation vector from a resolution table."""

    return model_resolutions(
        load_resolution_table(path),
        model,
        aggregation=aggregation,
        num_layers=num_layers,
        fallback=fallback,
    )


def write_model_resolutions(
    path: str | Path,
    *,
    model: str,
    aggregation: GraphAggregation,
    resolutions: list[float],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        table = load_resolution_table(destination)
    else:
        table = empty_resolution_table()
    aggregation = GraphAggregation(aggregation)
    model_entry = table["models"].setdefault(model, {"aggregations": {}})
    model_entry["aggregations"][aggregation.value] = {
        "resolutions": [float(value) for value in resolutions],
        "calibration": calibration,
    }
    validate_resolution_table(table)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(table, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return table


__all__ = [
    "DEFAULT_RESOLUTIONS",
    "RESOLUTION_SCHEMA",
    "empty_resolution_table",
    "load_model_resolutions",
    "load_resolution_table",
    "model_resolutions",
    "validate_resolution_table",
    "write_model_resolutions",
]
