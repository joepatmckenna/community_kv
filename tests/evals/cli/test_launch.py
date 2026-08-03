from __future__ import annotations

import sys

import pytest

from evals.cli.launch import build_launch, parse_gpus


def test_launcher_maps_first_gpu_to_model_and_rest_to_partition() -> None:
    command, environment = build_launch(
        parse_gpus("3,5,7"),
        ["--dataset", "longbench-v2", "--model", "qwen3-8b"],
    )
    assert command[:4] == [sys.executable, "-u", "-m", "evals.cli.main"]
    assert command[4:8] == [
        "--device",
        "cuda:0",
        "--partition-devices",
        "cuda:1,cuda:2",
    ]
    assert environment["CUDA_VISIBLE_DEVICES"] == "3,5,7"


def test_launcher_rejects_manual_device_placement() -> None:
    with pytest.raises(ValueError, match="launcher owns device placement"):
        build_launch(("0",), ["--device", "cuda:1"])


@pytest.mark.parametrize("value", ("", "0,0"))
def test_gpu_parser_rejects_empty_or_duplicate_assignments(value: str) -> None:
    with pytest.raises(ValueError):
        parse_gpus(value)
