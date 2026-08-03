"""Minimal installed CLI for CommunityKV evaluation."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from community_kv import GraphAggregation
from evals.datasets import Dataset, get_dataset
from evals.models import ContextStrategy, ModelProfile, resolve_model
from evals.resolutions import DEFAULT_RESOLUTIONS
from evals.runner import EvalConfig, EvalRunner


def _preparse_dataset(argv: list[str] | None) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", required=True)
    args, _ = parser.parse_known_args(argv)
    return args.dataset


def _add_core_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("longbench-v2", "babilong", "ruler"),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="supported alias or generic Hugging Face causal model ID",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--num-sink", type=int, default=10)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--context-window", type=int, default=131072)
    parser.add_argument(
        "--context-strategy",
        choices=tuple(strategy.value for strategy in ContextStrategy),
        default=ContextStrategy.TRUNCATE.value,
        help=(
            "use bounded 1x/2x/4x scaling with middle-out overflow, "
            "or extend without truncation"
        ),
    )
    parser.add_argument("--rope-factor", type=float)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--aggregation",
        choices=tuple(aggregation.value for aggregation in GraphAggregation),
        default=GraphAggregation.PER_QUERY_HEAD.value,
    )
    parser.add_argument(
        "--resolutions",
        type=Path,
        default=DEFAULT_RESOLUTIONS,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--partition-devices",
        default="",
        help="comma-separated CUDA devices used for graph partitioning",
    )


def _ruler_family(profile: ModelProfile) -> str:
    expected = profile.expected_geometry
    if expected is None:
        raise ValueError(
            "RULER uses tokenizer-specific prepared prompts and therefore "
            "requires an explicitly supported Qwen3 or Llama 3.1 model"
        )
    if expected.model_type == "qwen3":
        return "qwen3"
    if expected.model_type == "llama":
        return "llama3"
    raise ValueError(f"RULER does not support model type {expected.model_type!r}")


def parse_args(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, type[Dataset]]:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(flag in arguments for flag in ("-h", "--help")) and "--dataset" not in arguments:
        parser = argparse.ArgumentParser(description=__doc__)
        _add_core_args(parser)
        parser.parse_args(arguments)
        raise AssertionError("argparse help should exit")
    dataset_name = _preparse_dataset(arguments)
    dataset_cls = get_dataset(dataset_name)
    parser = argparse.ArgumentParser(description=__doc__)
    _add_core_args(parser)
    dataset_cls.add_args(parser)
    args = parser.parse_args(arguments)
    if args.max_new_tokens < 2:
        parser.error("--max-new-tokens must be at least 2")
    if args.token_budget <= 0 or args.token_budget % 64:
        parser.error("--token-budget must be a positive multiple of 64")
    if args.num_sink < 0 or args.num_sink + 1 >= args.token_budget:
        parser.error("--num-sink leaves no selected-token capacity")
    if not 0 <= args.lam <= 1:
        parser.error("--lam must be in [0, 1]")
    if args.context_window <= args.max_new_tokens:
        parser.error("--context-window must exceed --max-new-tokens")
    if args.rope_factor is not None and (
        not math.isfinite(args.rope_factor) or args.rope_factor < 1
    ):
        parser.error("--rope-factor must be finite and at least 1")
    if (
        args.context_strategy == ContextStrategy.TRUNCATE.value
        and args.rope_factor is not None
    ):
        parser.error("--rope-factor requires --context-strategy extend")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    profile = resolve_model(args.model)
    args.model = profile.model_id
    if dataset_name == "ruler":
        try:
            args.model_family = _ruler_family(profile)
        except ValueError as error:
            parser.error(str(error))
    return args, dataset_cls


def _partition_devices(value: str) -> tuple[str, ...]:
    return tuple(device.strip() for device in value.split(",") if device.strip())


def main(argv: list[str] | None = None) -> None:
    args, dataset_cls = parse_args(argv)
    dataset = dataset_cls.from_args(args)
    config = EvalConfig(
        model=args.model,
        output=args.output,
        resolutions=args.resolutions,
        max_new_tokens=args.max_new_tokens,
        token_budget=args.token_budget,
        num_sink=args.num_sink,
        lam=args.lam,
        context_window=args.context_window,
        context_strategy=ContextStrategy(args.context_strategy),
        rope_factor=args.rope_factor,
        max_samples=args.max_samples,
        device=args.device,
        partition_devices=_partition_devices(args.partition_devices),
        aggregation=GraphAggregation(args.aggregation),
    )
    payload = EvalRunner(config, dataset).run()
    summary = payload["summary"]
    print(
        f"complete: samples={summary['sample_count']}, "
        f"mean_score={summary['mean_score']:.6f}, output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
