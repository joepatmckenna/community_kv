from __future__ import annotations

import tomllib
from pathlib import Path

from evals.cli.main import parse_args
from evals.datasets.babilong import BabilongDataset
from evals.datasets.longbench_v2 import LongBenchV2Dataset
from evals.datasets.ruler import RulerDataset


ROOT = Path(__file__).parents[3]


def test_package_scripts_target_evals_modules() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["scripts"] == {
        "community-kv-eval": "evals.cli.main:main",
        "community-kv-eval-launch": "evals.cli.launch:main",
        "community-kv-tune-resolutions": "evals.cli.tune_resolutions:main",
    }


def test_longbench_cli_exposes_only_common_and_split_flags(tmp_path: Path) -> None:
    args, dataset_cls = parse_args(
        [
            "--dataset",
            "longbench-v2",
            "--model",
            "qwen3-8b",
            "--output",
            str(tmp_path / "result.json"),
            "--split",
            "long",
            "--max-samples",
            "3",
            "--aggregation",
            "per_query_head",
        ]
    )
    assert dataset_cls is LongBenchV2Dataset
    assert args.model == "Qwen/Qwen3-8B"
    assert args.split == "long"
    assert args.max_samples == 3
    assert args.aggregation == "per_query_head"


def test_eval_cli_defaults_to_per_query_head_and_exposes_graph_weights(
    tmp_path: Path,
) -> None:
    args, _ = parse_args(
        [
            "--dataset",
            "longbench-v2",
            "--model",
            "qwen3-8b",
            "--output",
            str(tmp_path / "result.json"),
            "--num-sink",
            "12",
            "--lam",
            "0.25",
        ]
    )
    assert args.aggregation == "per_query_head"
    assert args.num_sink == 12
    assert args.lam == 0.25
    assert args.context_strategy == "truncate"
    assert args.rope_factor is None


def test_eval_cli_can_select_context_extension(tmp_path: Path) -> None:
    args, _ = parse_args(
        [
            "--dataset",
            "ruler",
            "--model",
            "qwen3-8b",
            "--output",
            str(tmp_path / "result.json"),
            "--context-strategy",
            "extend",
            "--rope-factor",
            "4",
        ]
    )
    assert args.context_strategy == "extend"
    assert args.rope_factor == 4.0


def test_babilong_cli_fixes_public_sample_repository(tmp_path: Path) -> None:
    args, dataset_cls = parse_args(
        [
            "--dataset",
            "babilong",
            "--model",
            "qwen3-4b",
            "--output",
            str(tmp_path / "result.json"),
            "--task",
            "qa3",
            "--length",
            "64k",
        ]
    )
    dataset = dataset_cls.from_args(args)
    assert isinstance(dataset, BabilongDataset)
    assert dataset.samples_repo == "100"
    assert dataset.lookup_name() == "babilong:qa3:64k"


def test_ruler_family_is_inferred_from_supported_model(tmp_path: Path) -> None:
    args, dataset_cls = parse_args(
        [
            "--dataset",
            "ruler",
            "--model",
            "llama3.1-8b-instruct",
            "--output",
            str(tmp_path / "result.json"),
            "--task",
            "vt",
            "--length",
            "32k",
        ]
    )
    dataset = dataset_cls.from_args(args)
    assert isinstance(dataset, RulerDataset)
    assert dataset.lookup_name() == "ruler:llama3:vt:32k"
