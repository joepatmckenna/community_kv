"""``community-kv-eval`` entry point — dataset-agnostic streaming evaluation.

Argument parsing is two-pass, so each dataset can contribute its own flags:
    1. Parse ``--dataset NAME`` (unknown args ignored on this pass).
    2. Look up the named dataset class, call its ``add_args(parser)`` to
       attach dataset-specific args to the same parser.
    3. Re-parse for the full namespace; the dataset is materialized via
       ``cls.from_args(args)``.

Adding a new dataset = drop a module under
``evals/datasets/`` that ``@register_dataset("name")``-decorates a
``Dataset`` subclass, and import it from
``evals/datasets/__init__.py`` so registration runs.
"""

from __future__ import annotations

import argparse

from community_kv.graph.state import GraphAggregation
from evals import resolutions as resmod
from evals.datasets import (  # noqa: F401  side-effect: triggers dataset imports
    DATASET_REGISTRY,
    Dataset,
    get_dataset,
)
from evals.runner import EvalRunner
from evals.utils import setup_distributed

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def _add_core_args(p: argparse.ArgumentParser) -> None:
    """Args shared by every dataset (model / runtime / attention / TP)."""
    p.add_argument(
        "--dataset",
        required=True,
        help="Eval dataset name (registered via @register_dataset). Pass an "
        "unknown name to see what's registered.",
    )
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--token_budget", type=int, default=4096)
    p.add_argument("--kappa", type=int, default=8)
    p.add_argument("--num_sink", type=int, default=10)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--leiden_resolution", type=float, default=1.0)
    p.add_argument(
        "--aggregation",
        default=GraphAggregation.PER_QUERY_HEAD.value,
        choices=[a.value for a in GraphAggregation],
    )
    p.add_argument(
        "--max_partition_workers_per_gpu",
        type=int,
        default=None,
        help="Cap on concurrent Leiden partition jobs per GPU. None (default) "
        "= unlimited; lower values (e.g. 4) trade pipeline latency for memory "
        "headroom when partitioning large graphs alongside a resident model.",
    )
    p.add_argument("--tp_size", type=int, default=None)
    p.add_argument("--pp", action="store_true")
    p.add_argument("--repartition_every", type=int, default=0)
    p.add_argument(
        "--context_extension_strategy",
        default="yarn",
        choices=["yarn", "middle_out"],
        help="How to fit prompts longer than the model's native window. "
        "'yarn' rebuilds the model at the smallest power-of-2 rope_factor "
        "that fits each sample (samples are sorted ascending by length so "
        "rebuilds happen at most a handful of times). 'middle_out' builds "
        "once at the package's recommended cap (4x native_max) and asks "
        "the dataset to truncate longer prompts middle-out.",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap the number of samples evaluated. Default: full split as "
        "loaded by the dataset.",
    )
    p.add_argument(
        "--per_sample_resolutions",
        default=None,
        help="Path to a JSON file of per-sample Leiden resolutions to use "
        "instead of --leiden_resolution. Pass 'auto' to use the bundled "
        "evals/resolutions.json (tuned for avg community size 16).",
    )


def parse_args() -> tuple[argparse.Namespace, type[Dataset]]:
    """Two-stage parse: pick a dataset, let it add its own args, then parse."""
    pre = argparse.ArgumentParser(add_help=False)
    _add_core_args(pre)
    pre_args, _ = pre.parse_known_args()
    dataset_cls = get_dataset(pre_args.dataset)

    p = argparse.ArgumentParser(description="CommunityKV streaming evaluator")
    _add_core_args(p)
    dataset_cls.add_args(p)
    return p.parse_args(), dataset_cls


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    args, dataset_cls = parse_args()
    dist_ctx = setup_distributed(args)

    dataset = dataset_cls.from_args(args)

    samples = dataset.load_samples()
    if args.max_samples is not None and args.max_samples < len(samples):
        # Evenly-spaced subsample across the (length-sorted) cell so the subset
        # spans the full short->long distribution -- a head slice would bias to
        # the shortest docs and skip the long-context cases the eval exists to
        # measure. (max_samples==1 -> samples[0], unchanged for the smoke gate.)
        n = args.max_samples
        if n <= 1:
            samples = samples[:n]
        else:
            step = (len(samples) - 1) / (n - 1)
            samples = [samples[round(i * step)] for i in range(n)]

    per_sample_resolutions = resmod.load_for_run(args, dataset=dataset, dist_ctx=dist_ctx)

    runner = EvalRunner(
        args=args,
        dataset=dataset,
        dist=dist_ctx,
        per_sample_resolutions=per_sample_resolutions,
    )
    runner.run_iterate(samples)
    dist_ctx.destroy()


if __name__ == "__main__":
    main()
