"""``GraphRuntime`` — the per-sample mutable state container.

Holds scalar config (kappa, num_sink, lam, ...), the per-layer
``LayerGraph`` dict, the partition ``ThreadPoolExecutor`` and its
concurrency controls, and the decode-time repartition orchestration
state. Per-layer state shapes live in ``community_kv.graph.state``;
async workers that mutate the runtime live in
``community_kv.graph.workers``.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch

from community_kv.graph.state import GraphAggregation, LayerGraph, LayerLog
from community_kv.graph.workers import async_repartition_leiden


@contextmanager
def cuda_event_pair(device: torch.device):
    """Bracket a block of GPU work with a ``(start, end)`` CUDA-event pair.

    Yields ``(start, end)`` already-recorded for ``start`` on entry and
    records ``end`` on exit. Storage of the pair (into ``layer_log``,
    ``decode_retrieve_events``, ``decode_update_events``, ...) is the
    caller's responsibility — the context manager is intentionally
    storage-agnostic so it can serve all three sites in the attention
    forward.

    Uses ``torch.cuda.Event(enable_timing=True)`` so the events can be
    later resolved into milliseconds via ``elapsed_time``.
    """
    with torch.cuda.device(device):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield start, end
        finally:
            end.record()


@dataclass
class GraphRuntime:
    """Per-sample mutable state + scalar config for the CommunityKV runtime.

    Construct one ``GraphRuntime`` per attention instance at the entry
    point and pass it through to anything that reads or writes layer-graph
    state.
    """

    # ---- scalar config / knobs ------------------------------------------- #
    config: dict[str, Any] = field(
        default_factory=lambda: {
            "kappa": 8,
            "num_sink": 10,
            "lam": 0.5,
            "leiden_resolution": 1.0,
            "leiden_max_iter": 2,
            "max_new_tokens": 8,
            "token_budget": 4096,
        }
    )
    aggregation: GraphAggregation = GraphAggregation.PER_QUERY_HEAD
    repartition_every: int = 0

    # ---- per-sample graph state ------------------------------------------ #
    resolutions: dict[int, float] = field(default_factory=dict)
    graphs: dict[int, LayerGraph] = field(default_factory=dict)

    # ---- partition workers + concurrency control ------------------------- #
    executor: ThreadPoolExecutor | None = None
    gpu_semaphores: dict[str, threading.Semaphore] = field(default_factory=dict)
    futures: list[Future] = field(default_factory=list)

    # ---- decode-time repartition orchestration --------------------------- #
    repartition_in_flight: dict[int, Future] = field(default_factory=dict)
    repartition_trigger_pending: set[int] = field(default_factory=set)
    repartition_key_snapshots: dict[int, torch.Tensor] = field(default_factory=dict)

    # ---- perf / telemetry (CUDA event pairs, timing logs) ---------------- #
    layer_log: dict[int, LayerLog] = field(default_factory=dict)
    decode_retrieve_events: dict[int, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = field(
        default_factory=dict
    )
    decode_retrieve_n: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    decode_update_events: dict[int, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = field(
        default_factory=dict
    )
    repartition_records: list[dict] = field(default_factory=list)

    def reset_per_sample(self) -> None:
        """Clear all per-sample state. Call before each new sample."""
        self.drain_pending_repartitions()
        self.layer_log.clear()
        self.graphs.clear()
        self.decode_retrieve_events.clear()
        self.decode_retrieve_n.clear()
        self.decode_update_events.clear()
        self.repartition_records.clear()
        self.repartition_trigger_pending.clear()
        self.repartition_key_snapshots.clear()
        self.futures.clear()

    def drain_pending_repartitions(self) -> None:
        """Block until every in-flight repartition completes; do NOT swap
        results into ``graphs`` (the next sample will clear it anyway)."""
        for L, fut in list(self.repartition_in_flight.items()):
            try:
                fut.result()
            except Exception:
                pass
            del self.repartition_in_flight[L]

    # --- decode-time repartition orchestration ---------------------------- #

    def maybe_trigger_repartition(self, step_idx: int) -> None:
        """If ``step_idx`` is a multiple of ``repartition_every``, mark
        every layer's graph as eligible for snapshotting on the *next*
        attention forward. The forward writes the snapshot keys into
        ``repartition_key_snapshots``; ``dispatch_pending_repartitions``
        then submits the async Leiden jobs."""
        if self.repartition_every <= 0 or step_idx <= 0 or step_idx % self.repartition_every != 0:
            return
        self.repartition_trigger_pending.update(
            L for L in self.graphs if L not in self.repartition_in_flight
        )

    def dispatch_pending_repartitions(self) -> None:
        """Submit async-repartition jobs for layers whose key snapshots
        were captured during the most recent attention forward. Caller
        must have set ``self.executor`` before the run starts. No-op when
        no snapshots are pending."""
        if not self.repartition_key_snapshots:
            return
        cfg = self.config
        for L in list(self.repartition_key_snapshots.keys()):
            if L in self.repartition_in_flight:
                self.repartition_key_snapshots.pop(L)
                continue
            keys_snap = self.repartition_key_snapshots.pop(L)
            g_now = self.graphs[L]
            fut = self.executor.submit(
                async_repartition_leiden,
                L,
                g_now,
                int(g_now.decode_log_size),
                int(g_now.decode_edge_size),
                keys_snap,
                self.resolutions.get(L, cfg["leiden_resolution"]),
                cfg["leiden_max_iter"],
                cfg["num_sink"],
                cfg["max_new_tokens"],
                self.aggregation,
                cfg["kappa"],
                g_now.device,
            )
            self.repartition_in_flight[L] = fut
        self.repartition_trigger_pending.clear()

    def collect_completed_repartitions(self, step_idx: int) -> None:
        """Swap any freshly-completed repartition results into
        ``self.graphs`` and append the per-job record (with
        ``completed_at_step``) to ``self.repartition_records``."""
        if not self.repartition_in_flight:
            return
        for L, fut in list(self.repartition_in_flight.items()):
            if not fut.done():
                continue
            try:
                new_graph, record = fut.result()
            except Exception:
                del self.repartition_in_flight[L]
                continue
            self.graphs[L] = new_graph
            record["completed_at_step"] = step_idx
            self.repartition_records.append(record)
            del self.repartition_in_flight[L]

    def configure_workers(
        self,
        *,
        n_layers: int,
        devices: Iterable[str] = (),
        max_workers_per_gpu: int | None = None,
    ) -> None:
        """Set up the partition ``ThreadPoolExecutor`` + per-GPU semaphores
        for a fresh model. Idempotent: any existing executor is shut down
        first so device handles stay consistent across model swaps.

        ``devices`` is the set of device strings (e.g. ``"cuda:0"``) on
        which partition jobs may run; ``max_workers_per_gpu`` caps
        concurrent partition jobs per device, or ``None`` for unlimited.
        """
        if self.executor is not None:
            self.shutdown()
        self.executor = ThreadPoolExecutor(max_workers=n_layers, thread_name_prefix="part")
        self.gpu_semaphores.clear()
        if max_workers_per_gpu is not None:
            for dev in devices:
                self.gpu_semaphores[dev] = threading.Semaphore(max_workers_per_gpu)

    def shutdown(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
