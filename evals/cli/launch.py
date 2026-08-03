"""GPU-mapping launcher for the installed CommunityKV evaluation CLI."""

from __future__ import annotations

import argparse
import os
import sys


def parse_gpus(value: str) -> tuple[str, ...]:
    gpus = tuple(item.strip() for item in value.split(",") if item.strip())
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU")
    if len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must not contain duplicates")
    return gpus


def build_launch(
    gpus: tuple[str, ...],
    runner_args: list[str],
) -> tuple[list[str], dict[str, str]]:
    forbidden = {"--device", "--partition-devices"}
    used_forbidden = [
        argument.split("=", 1)[0]
        for argument in runner_args
        if argument.split("=", 1)[0] in forbidden
    ]
    if used_forbidden:
        raise ValueError(
            "the launcher owns device placement; remove "
            + ", ".join(sorted(set(used_forbidden)))
        )
    partition_devices = ",".join(
        f"cuda:{index}" for index in range(1, len(gpus))
    )
    command = [
        sys.executable,
        "-u",
        "-m",
        "evals.cli.main",
        "--device",
        "cuda:0",
        "--partition-devices",
        partition_devices,
        *runner_args,
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    return command, environment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpus",
        required=True,
        help="visible GPU IDs; first runs the model and the rest partition",
    )
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        gpus = parse_gpus(args.gpus)
        runner_args = list(args.runner_args)
        if runner_args[:1] == ["--"]:
            runner_args.pop(0)
        if not runner_args:
            parser.error("runner arguments are required after --")
        command, environment = build_launch(gpus, runner_args)
    except ValueError as error:
        parser.error(str(error))
    os.execve(sys.executable, command, environment)


if __name__ == "__main__":
    main()
