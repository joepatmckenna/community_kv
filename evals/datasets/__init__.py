"""Eval-dataset implementations.

Each dataset module ``@register_dataset("name")``-decorates a
:class:`Dataset` subclass; importing the module is what triggers
registration. This package's ``__init__`` does that import for every
known dataset, so callers only need ``from evals.datasets
import get_dataset`` to look one up by name.

To add a new dataset:
    1. Drop a module under ``evals/datasets/`` that decorates
       its class with ``@register_dataset("your-name")``.
    2. Add an import for it below so the decorator runs at package load.
"""

from evals.datasets.dataset import (
    DATASET_REGISTRY,
    Dataset,
    get_dataset,
    register_dataset,
)
from evals.datasets import babilong  # noqa: F401  registers "babilong"
from evals.datasets import longbench_v2  # noqa: F401  registers "longbench-v2"

__all__ = [
    "DATASET_REGISTRY",
    "Dataset",
    "babilong",
    "get_dataset",
    "longbench_v2",
    "register_dataset",
]
