from __future__ import annotations

from concurrent.futures import Future

import torch

from community_kv.config import CommunityKVConfig, PartitionConfig
from community_kv.graph.runtime import PartitionRuntime


def _runtime(*, devices=()) -> PartitionRuntime:
    return PartitionRuntime(
        algorithm=CommunityKVConfig(),
        scheduling=PartitionConfig(
            devices=devices,
            workers_per_device=1,
        ),
        max_layers=4,
    )


def test_partition_devices_round_robin_or_fall_back_to_model_device() -> None:
    runtime = _runtime(devices=("cuda:2", "cuda:4"))
    try:
        assert runtime.partition_device(0, torch.device("cpu")) == torch.device("cuda:2")
        assert runtime.partition_device(3, torch.device("cpu")) == torch.device("cuda:4")
    finally:
        runtime.shutdown()

    fallback = _runtime()
    try:
        assert fallback.partition_device(2, torch.device("cpu")) == torch.device("cpu")
    finally:
        fallback.shutdown()


def test_submission_starts_immediately(monkeypatch) -> None:
    runtime = _runtime()
    submitted = []

    def record_submission(job) -> None:
        submitted.append(job)
        runtime._futures[job.layer_idx] = Future()

    monkeypatch.setattr(runtime, "_submit", record_submission)
    keys = torch.empty((1, 4, 128))

    try:
        for layer_idx in (2, 0):
            runtime.submit(
                layer_idx=layer_idx,
                topk_indices=torch.empty(0, dtype=torch.int32),
                topk_scores=torch.empty(0),
                keys=keys,
                completion_event=object(),
                resolution=1.0,
                max_decode_tokens=8,
            )
        assert [job.layer_idx for job in submitted] == [2, 0]
    finally:
        runtime.shutdown()


def test_wait_installs_completed_layers() -> None:
    runtime = _runtime()
    installed = []
    try:
        for layer_idx in (2, 0):
            future = Future()
            future.set_result(f"state-{layer_idx}")
            runtime._futures[layer_idx] = future

        result = runtime.wait(installed.append)

        assert set(installed) == {"state-0", "state-2"}
        assert result is None
    finally:
        runtime.shutdown()


def test_duplicate_layer_submission_is_rejected(monkeypatch) -> None:
    runtime = _runtime()

    def record_submission(job) -> None:
        runtime._futures[job.layer_idx] = Future()

    monkeypatch.setattr(runtime, "_submit", record_submission)
    keys = torch.empty((1, 4, 128))
    arguments = {
        "layer_idx": 1,
        "topk_indices": torch.empty(0, dtype=torch.int32),
        "topk_scores": torch.empty(0),
        "keys": keys,
        "completion_event": object(),
        "resolution": 1.0,
        "max_decode_tokens": 8,
    }
    try:
        runtime.submit(**arguments)
        try:
            runtime.submit(**arguments)
        except ValueError as error:
            assert "already has" in str(error)
        else:
            raise AssertionError("duplicate layer submission was accepted")
    finally:
        runtime.shutdown()
