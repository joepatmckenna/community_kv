"""Asynchronous dedicated-GPU orchestration for prefill partitioning."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import torch

from community_kv.config import CommunityKVConfig, PartitionConfig
from community_kv.graph.partition import PartitionedLayer, build_partitioned_layer


@dataclass(slots=True)
class _Job:
    layer_idx: int
    topk_indices: torch.Tensor
    topk_scores: torch.Tensor
    keys: torch.Tensor
    completion_event: torch.cuda.Event
    model_device: torch.device
    partition_device: torch.device
    resolution: float
    max_decode_tokens: int


class PartitionRuntime:
    """Own worker threads and install compact partition results on demand."""

    def __init__(
        self,
        *,
        algorithm: CommunityKVConfig,
        scheduling: PartitionConfig,
        max_layers: int,
    ) -> None:
        if max_layers <= 0:
            raise ValueError("max_layers must be positive")
        self.algorithm = algorithm
        self.scheduling = scheduling
        self._executor = ThreadPoolExecutor(
            max_workers=max(
                1,
                min(
                    max_layers,
                    max(1, len(scheduling.devices))
                    * scheduling.workers_per_device,
                ),
            ),
            thread_name_prefix="ckv-partition",
        )
        self._semaphores = {
            str(torch.device(device)): threading.Semaphore(
                scheduling.workers_per_device
            )
            for device in scheduling.devices
        }
        self._futures: dict[int, Future[PartitionedLayer]] = {}

    def reset(self) -> None:
        if self._futures:
            for future in self._futures.values():
                future.result()
        self._futures.clear()

    def partition_device(
        self,
        layer_idx: int,
        model_device: torch.device,
    ) -> torch.device:
        if not self.scheduling.devices:
            return model_device
        return torch.device(
            self.scheduling.devices[layer_idx % len(self.scheduling.devices)]
        )

    def submit(
        self,
        *,
        layer_idx: int,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor,
        keys: torch.Tensor,
        completion_event: torch.cuda.Event,
        resolution: float,
        max_decode_tokens: int,
    ) -> None:
        if layer_idx in self._futures:
            raise ValueError(f"layer {layer_idx} already has a partition job")
        model_device = keys.device
        job = _Job(
            layer_idx=layer_idx,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            keys=keys,
            completion_event=completion_event,
            model_device=model_device,
            partition_device=self.partition_device(layer_idx, model_device),
            resolution=resolution,
            max_decode_tokens=max_decode_tokens,
        )
        self._submit(job)

    def _submit(self, job: _Job) -> None:
        self._futures[job.layer_idx] = self._executor.submit(
            self._run_job,
            job,
        )

    def _run_job(
        self,
        job: _Job,
    ) -> PartitionedLayer:
        semaphore = self._semaphores.get(str(job.partition_device))
        if semaphore is not None:
            semaphore.acquire()
        try:
            cross_device = job.partition_device != job.model_device
            if cross_device:
                job.completion_event.synchronize()
            with torch.cuda.device(job.partition_device):
                stream = torch.cuda.Stream(device=job.partition_device)
                if not cross_device:
                    stream.wait_event(job.completion_event)
                with torch.cuda.stream(stream):
                    indices = job.topk_indices.to(
                        job.partition_device,
                        non_blocking=cross_device,
                    )
                    scores = job.topk_scores.to(
                        job.partition_device,
                        non_blocking=cross_device,
                    )
                    keys = job.keys.to(
                        job.partition_device,
                        non_blocking=cross_device,
                    )
                stream.synchronize()
                with torch.cuda.stream(stream):
                    state = build_partitioned_layer(
                        layer_idx=job.layer_idx,
                        topk_indices=indices,
                        topk_scores=scores,
                        keys=keys,
                        num_sink=self.algorithm.num_sink,
                        lam=self.algorithm.lam,
                        leiden_resolution=job.resolution,
                        leiden_seed=self.algorithm.leiden_seed,
                        max_decode_tokens=job.max_decode_tokens,
                        aggregation=self.algorithm.aggregation,
                    )
                stream.synchronize()
                with torch.cuda.stream(stream):
                    if cross_device:
                        state = state.to(job.model_device, non_blocking=True)
                stream.synchronize()
        finally:
            if semaphore is not None:
                semaphore.release()

        return state

    def wait(
        self,
        on_layer_ready: Callable[[PartitionedLayer], None],
    ) -> None:
        by_future = {
            future: layer_idx for layer_idx, future in self._futures.items()
        }
        for future in as_completed(by_future):
            on_layer_ready(future.result())

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


__all__ = ["PartitionRuntime"]
