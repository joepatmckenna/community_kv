"""Leiden resolution overrides for eval runs.

The bundled ``resolutions.json`` is a NESTED per-sample per-layer table
(schema ``nested-per-sample-per-layer-v1``):

    {
      "_meta": { "schema", "metric", "leaf", "n_samples" },
      "config": {                            # uniform across all tune jobs (flat)
        "context_extension_strategy", "context_window", "rope_factor",
        "rope_original_max", "aggregation", "kappa", "num_sink", "lam",
        "target_avg_community_size"
      },
      "<MODEL>": {                         # e.g. "Qwen/Qwen3-8B"
        "<DATASET>": { <split-path> : { "<sample_id>": [res_l0, res_l1, ...] } }
      }
    }

``config`` (flat: context-extension + edge/aggregation params) is uniform
across every tune job, so it sits at the top level. ``<DATASET>`` is the
dataset portion of a dataset's ``lookup_name()`` ("LongBench-v2", "babilong"); ``<split-path>`` is the
remainder — one level for LongBench ("short"/"medium"/"long"), two for
BABILong ("qa1" -> "64k"). The leaf is ``{sample_id: [resolution per layer]}``:
per-sample AND per-layer, each value the resolution that hit the target avg
community size for that sample's graphs in that layer. The runner copies a
sample's per-layer map into ``graph_runtime.resolutions`` for that sample.

The older FLAT per-sample schema (``model=...|dataset=...|sample=...`` ->
scalar) is still readable via the legacy ``filter_table`` / ``lookup`` /
``make_key`` helpers below; against a nested table they simply match nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    from evals.distributed import DistContext
    from evals.datasets.dataset import Dataset

DEFAULT_PATH = Path(__file__).parent / "resolutions.json"

# Top-level keys that are NOT datasets.
_RESERVED_KEYS = {"_meta", "config"}


def load(path: str | Path | None = None) -> dict:
    """Load the resolutions table. ``None`` reads the bundled file."""
    p = Path(path) if path is not None else DEFAULT_PATH
    with open(p) as f:
        return json.load(f)


def tune_config(table: dict) -> dict | None:
    """Top-level flat config block (strategy, context_window, rope_factor,
    rope_original_max, aggregation, kappa, num_sink, lam, target_avg_community_size),
    or ``None`` for a non-nested table."""
    return table.get("config") if is_nested(table) else None


_CONTEXT_KEYS = ("context_extension_strategy", "context_window", "rope_factor", "rope_original_max")


def context_extension(table: dict) -> dict | None:
    """The context-extension subset of the flat config
    (context_extension_strategy, context_window, rope_factor,
    rope_original_max), or ``None``."""
    cfg = tune_config(table)
    if not cfg:
        return None
    return {k: cfg[k] for k in _CONTEXT_KEYS if k in cfg}


def is_nested(table: dict) -> bool:
    """True if ``table`` is the nested per-sample-per-layer schema (model-keyed:
    ``<model> -> <dataset> -> <split> -> <sample_id> -> [resolutions]``) vs the
    legacy flat schema (composite string key -> single float).

    Detected structurally by value type -- nested entries map a model to a
    ``dict``, whereas flat entries map a composite key to a ``float`` -- so it
    does not depend on a ``_meta.schema`` tag being present."""
    return any(isinstance(v, dict) for k, v in table.items() if k not in ("_meta", "config"))


def split_lookup_name(lookup_name: str) -> tuple[str, str]:
    """Split a dataset ``lookup_name()`` into (dataset, split).

    ``"LongBench-v2:short"`` -> ``("LongBench-v2", "short")``;
    ``"babilong:qa1:64k"`` -> ``("babilong", "qa1:64k")``.
    """
    dataset, _, split = lookup_name.partition(":")
    return dataset, split


def layer_resolutions(
    table: dict, *, model: str, lookup_name: str, sample_id: str
) -> dict[int, float] | None:
    """Return ``{layer_idx: resolution}`` for one sample under
    (model, dataset:split), or ``None`` if the nested table has no entry.

    The split-path nests one level for LongBench (``"short"``) and two for
    BABILong (``"qa1" -> "64k"``); the leaf is ``{sample_id: [per-layer res]}``.
    Layers whose tuned value is ``null`` (not yet filled) are omitted so the
    runtime falls back to its default for them.
    """
    if not is_nested(table):
        return None
    dataset, _, rest = lookup_name.partition(":")
    node = table.get(model, {}).get(dataset)
    for key in rest.split(":"):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    if not isinstance(node, dict):
        return None
    lst = node.get(sample_id)
    if not isinstance(lst, list):
        return None
    return {i: r for i, r in enumerate(lst) if r is not None}


def make_key(
    *,
    model: str,
    dataset: str,
    split: str,
    aggregation: str,
    kappa: int,
    num_sink: int,
    lam: float,
    target: float,
    sample_id: str,
) -> str:
    """Reconstruct the canonical lookup key."""
    cfg = f"agg={aggregation},kappa={kappa},num_sink={num_sink}," f"lam={lam},target={target}"
    return f"model={model}|dataset={dataset}:{split}|{cfg}|sample={sample_id}"


def lookup(
    table: dict[str, float],
    *,
    model: str,
    dataset: str,
    split: str,
    aggregation: str,
    kappa: int,
    num_sink: int,
    lam: float,
    target: float,
    sample_id: str,
) -> float | None:
    """Return the resolution for the given configuration + sample, or
    ``None`` if no entry matches."""
    return table.get(
        make_key(
            model=model,
            dataset=dataset,
            split=split,
            aggregation=aggregation,
            kappa=kappa,
            num_sink=num_sink,
            lam=lam,
            target=target,
            sample_id=sample_id,
        )
    )


def filter_table(
    table: dict[str, float],
    *,
    model: str,
    dataset_lookup_name: str,
    aggregation: str,
    kappa: int,
    num_sink: int,
    lam: float,
    target: float = 16.0,
) -> dict[str, float]:
    """Reduce ``table`` to a flat ``{sample_id: resolution}`` map for the
    given run config. ``dataset_lookup_name`` is the colon-form name as it
    appears in the JSON's ``dataset=`` field, e.g. ``"LongBench-v2:short"``.
    """
    cfg = f"agg={aggregation},kappa={kappa},num_sink={num_sink}," f"lam={lam},target={target}"
    prefix = f"model={model}|dataset={dataset_lookup_name}|{cfg}|sample="
    return {k[len(prefix) :]: v for k, v in table.items() if k.startswith(prefix)}


def load_for_run(
    args: "argparse.Namespace",
    *,
    dataset: "Dataset",
    dist_ctx: "DistContext",
) -> dict[str, float] | None:
    """High-level entry point used by the CLI: honor ``args.per_sample_resolutions``,
    ask the dataset for its lookup name, load + filter the table for the
    current run config, and log a one-line summary on rank 0.

    Returns ``None`` when per-sample resolutions are disabled or the
    dataset opts out of the lookup.
    """
    if args.per_sample_resolutions is None:
        return None

    lookup_name = dataset.lookup_name()
    if lookup_name is None:
        dist_ctx.print0(
            f"--per_sample_resolutions ignored: dataset {dataset.name!r} " f"has no lookup_name().",
            flush=True,
        )
        return None

    path = None if args.per_sample_resolutions == "auto" else args.per_sample_resolutions
    table = load(path)

    if is_nested(table):
        # Nested per-sample per-layer table: pass it through whole; the runner
        # installs each sample's {layer: resolution} map. Count this cell's
        # samples for the log.
        ds_key, _, rest = lookup_name.partition(":")
        node = table.get(args.model, {}).get(ds_key)
        for key in rest.split(":"):
            node = node.get(key) if isinstance(node, dict) else None
        n = len(node) if isinstance(node, dict) else 0
        dist_ctx.print0(
            f"per-sample per-layer resolutions: {n} samples for " f"{args.model} {lookup_name}",
            flush=True,
        )
        return table

    filtered = filter_table(
        table,
        model=args.model,
        dataset_lookup_name=lookup_name,
        aggregation=args.aggregation,
        kappa=args.kappa,
        num_sink=args.num_sink,
        lam=args.lam,
    )
    dist_ctx.print0(
        f"per-sample resolutions: {len(filtered)} entries matched for {lookup_name}",
        flush=True,
    )
    return filtered
