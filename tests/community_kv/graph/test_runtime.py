"""Tests for community_kv.graph.runtime — the per-sample mutable state
container plus the CUDA event-pair helper.

Three groups:
  * ``TestCudaEventPair`` — the storage-agnostic timing helper.
  * ``TestGraphRuntime`` — basic config / reset / drain / cross-instance
    isolation for the dataclass.
  * ``TestRepartitionLifecycle`` — the three-step ``maybe_trigger →
    dispatch → collect`` orchestration plus draining.
  * ``TestExecutorLifecycle`` — ``configure_workers`` / ``shutdown`` for
    the partition ThreadPoolExecutor.

LayerLog/LayerGraph dataclass tests live in ``test_state.py`` (next to
their definitions).
"""

from concurrent.futures import Future

import pytest
import torch

from community_kv.graph.runtime import GraphRuntime, cuda_event_pair


class TestCudaEventPair:
    """``cuda_event_pair`` brackets GPU work with a (start, end) event
    pair. Storage is the caller's responsibility, but the manager always
    yields a fresh pair, records ``start`` on entry, and records ``end``
    on exit (even if the body raised)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA Event")
    def test_yields_event_pair_recorded_around_body(self):
        device = torch.device("cuda:0")
        with cuda_event_pair(device) as (s, e):
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        assert isinstance(s, torch.cuda.Event)
        assert isinstance(e, torch.cuda.Event)
        # Both events are queryable; elapsed_time returns a non-negative
        # number of milliseconds.
        assert s.elapsed_time(e) >= 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA Event")
    def test_records_end_event_even_on_exception(self):
        device = torch.device("cuda:0")
        captured: list = []
        with pytest.raises(RuntimeError):
            with cuda_event_pair(device) as (s, e):
                captured.extend([s, e])
                raise RuntimeError("boom")
        torch.cuda.synchronize()
        s, e = captured
        # ``end`` was recorded by the finally branch; elapsed_time
        # succeeds rather than blocking forever.
        assert s.elapsed_time(e) >= 0.0


class TestGraphRuntime:
    """Per-sample state container: defaults, reset semantics, drain on
    in-flight futures, and cross-instance isolation."""

    def test_default_config(self):
        graph_runtime = GraphRuntime()
        assert graph_runtime.config["kappa"] == 8
        assert graph_runtime.config["num_sink"] == 10
        assert graph_runtime.config["lam"] == 0.5
        assert graph_runtime.repartition_every == 0
        assert graph_runtime.executor is None

    def test_reset_per_sample_clears_all_state(self):
        graph_runtime = GraphRuntime()
        graph_runtime.layer_log[0] = "anything"
        graph_runtime.graphs[1] = "anything"
        graph_runtime.decode_retrieve_events[0] = []
        graph_runtime.decode_update_events[1] = []
        graph_runtime.repartition_records.append({"x": 1})
        graph_runtime.repartition_trigger_pending.add(7)
        graph_runtime.repartition_key_snapshots[3] = "snap"
        graph_runtime.futures.append("fut")
        graph_runtime.reset_per_sample()
        assert graph_runtime.layer_log == {}
        assert graph_runtime.graphs == {}
        assert graph_runtime.decode_retrieve_events == {}
        assert graph_runtime.decode_update_events == {}
        assert graph_runtime.repartition_records == []
        assert graph_runtime.repartition_trigger_pending == set()
        assert graph_runtime.repartition_key_snapshots == {}
        assert graph_runtime.futures == []

    def test_drain_pending_repartitions_waits_and_clears(self):
        graph_runtime = GraphRuntime()
        fut = Future()
        fut.set_result(("graph", {"meta": 1}))
        graph_runtime.repartition_in_flight[5] = fut
        graph_runtime.drain_pending_repartitions()
        assert graph_runtime.repartition_in_flight == {}

    def test_drain_swallows_worker_exceptions(self):
        """If a worker raised, ``drain_pending_repartitions`` must still
        remove the future from ``repartition_in_flight`` rather than
        propagating the exception (the next sample will clear graphs anyway)."""
        graph_runtime = GraphRuntime()
        good = Future()
        good.set_result(("graph", {"meta": 1}))
        bad = Future()
        bad.set_exception(RuntimeError("worker died"))
        graph_runtime.repartition_in_flight[1] = good
        graph_runtime.repartition_in_flight[2] = bad
        graph_runtime.drain_pending_repartitions()
        assert graph_runtime.repartition_in_flight == {}

    def test_separate_instances_dont_share_state(self):
        """Each ``GraphRuntime()`` instance is independent."""
        a = GraphRuntime()
        b = GraphRuntime()
        a.config["kappa"] = 99
        a.layer_log[1] = "from-a"
        assert b.config["kappa"] == 8
        assert b.layer_log == {}


class _FakeExecutor:
    """Captures submitted jobs without running them. ``submit(fn, *args)``
    returns a ``Future`` we never resolve unless the test calls
    ``set_result`` on it. Lets us drive the dispatch / collect lifecycle
    deterministically without spawning real threads."""

    def __init__(self):
        self.submitted: list = []

    def submit(self, fn, *args, **kwargs):
        fut: Future = Future()
        self.submitted.append((fn, args, kwargs, fut))
        return fut


class _StubGraph:
    """Minimal stand-in for LayerGraph for repartition-orchestration tests.
    The real LayerGraph carries CUDA tensors; we only need the fields
    that ``dispatch_pending_repartitions`` reads."""

    def __init__(self, *, layer_idx: int, decode_log_size: int = 0, decode_edge_size: int = 0):
        self.layer_idx = layer_idx
        self.decode_log_size = decode_log_size
        self.decode_edge_size = decode_edge_size
        self.device = torch.device("cpu")


class TestRepartitionLifecycle:
    """The three-step decode-time orchestration: ``maybe_trigger_repartition``
    flags eligible layers, ``dispatch_pending_repartitions`` submits async
    Leiden jobs once snapshots arrive, and ``collect_completed_repartitions``
    swaps results back in. Driven via a ``_FakeExecutor`` so the test
    doesn't spawn worker threads."""

    # ---- maybe_trigger_repartition ---------------------------------------- #

    def test_trigger_disabled_when_repartition_every_zero(self):
        gr = GraphRuntime()
        gr.repartition_every = 0
        gr.graphs[0] = _StubGraph(layer_idx=0)
        gr.maybe_trigger_repartition(step_idx=10)
        assert gr.repartition_trigger_pending == set()

    def test_trigger_no_op_at_step_zero(self):
        gr = GraphRuntime()
        gr.repartition_every = 5
        gr.graphs[0] = _StubGraph(layer_idx=0)
        gr.maybe_trigger_repartition(step_idx=0)
        assert gr.repartition_trigger_pending == set()

    def test_trigger_no_op_when_step_not_multiple(self):
        gr = GraphRuntime()
        gr.repartition_every = 5
        gr.graphs[0] = _StubGraph(layer_idx=0)
        gr.maybe_trigger_repartition(step_idx=3)
        assert gr.repartition_trigger_pending == set()

    def test_trigger_marks_eligible_layers_at_multiple(self):
        gr = GraphRuntime()
        gr.repartition_every = 5
        gr.graphs[0] = _StubGraph(layer_idx=0)
        gr.graphs[2] = _StubGraph(layer_idx=2)
        gr.maybe_trigger_repartition(step_idx=10)
        assert gr.repartition_trigger_pending == {0, 2}

    def test_trigger_skips_layers_already_in_flight(self):
        """A layer with a partition already running should not be re-flagged
        — the in-flight job will produce the next snapshot."""
        gr = GraphRuntime()
        gr.repartition_every = 5
        gr.graphs[0] = _StubGraph(layer_idx=0)
        gr.graphs[1] = _StubGraph(layer_idx=1)
        gr.repartition_in_flight[0] = Future()  # layer 0 is busy
        gr.maybe_trigger_repartition(step_idx=5)
        assert gr.repartition_trigger_pending == {1}

    # ---- dispatch_pending_repartitions ------------------------------------ #

    def test_dispatch_no_op_without_snapshots(self):
        gr = GraphRuntime()
        gr.executor = _FakeExecutor()
        gr.dispatch_pending_repartitions()
        assert gr.executor.submitted == []
        assert gr.repartition_in_flight == {}

    def test_dispatch_submits_jobs_for_pending_snapshots(self):
        gr = GraphRuntime()
        gr.executor = _FakeExecutor()
        gr.graphs[3] = _StubGraph(layer_idx=3, decode_log_size=2, decode_edge_size=10)
        gr.repartition_key_snapshots[3] = torch.zeros(1)
        gr.repartition_trigger_pending.add(3)
        gr.dispatch_pending_repartitions()
        assert len(gr.executor.submitted) == 1
        fn, args, _, fut = gr.executor.submitted[0]
        # First two args are layer_idx and the LayerGraph snapshot.
        assert args[0] == 3
        assert args[1] is gr.graphs[3]
        # Snapshot moved into in-flight; trigger flag cleared.
        assert gr.repartition_in_flight[3] is fut
        assert 3 not in gr.repartition_key_snapshots
        assert gr.repartition_trigger_pending == set()

    def test_dispatch_skips_layers_with_in_flight_job(self):
        """If a snapshot arrives for a layer whose previous repartition is
        still running, drop the snapshot rather than starting a second job."""
        gr = GraphRuntime()
        gr.executor = _FakeExecutor()
        gr.graphs[1] = _StubGraph(layer_idx=1)
        gr.repartition_in_flight[1] = Future()  # layer 1 busy
        gr.repartition_key_snapshots[1] = torch.zeros(1)
        gr.dispatch_pending_repartitions()
        assert gr.executor.submitted == []
        assert 1 not in gr.repartition_key_snapshots

    def test_dispatch_passes_resolution_override(self):
        """``resolutions[L]`` should override the default leiden_resolution
        when present, falling back to ``config["leiden_resolution"]``."""
        gr = GraphRuntime()
        gr.executor = _FakeExecutor()
        gr.graphs[5] = _StubGraph(layer_idx=5)
        gr.repartition_key_snapshots[5] = torch.zeros(1)
        gr.resolutions[5] = 0.7  # per-layer override
        gr.dispatch_pending_repartitions()
        _, args, _, _ = gr.executor.submitted[0]
        # Args order: (layer_idx, src_graph, snap_log_size, snap_edge_size,
        # keys_snap, leiden_resolution, ...) — leiden_resolution is index 5.
        assert args[5] == 0.7

    # ---- collect_completed_repartitions ----------------------------------- #

    def test_collect_no_op_without_inflight(self):
        gr = GraphRuntime()
        gr.collect_completed_repartitions(step_idx=42)
        assert gr.repartition_records == []

    def test_collect_skips_unfinished_futures(self):
        gr = GraphRuntime()
        gr.repartition_in_flight[0] = Future()  # never set_result
        gr.collect_completed_repartitions(step_idx=42)
        assert 0 in gr.repartition_in_flight
        assert gr.repartition_records == []

    def test_collect_swaps_completed_graph_and_records(self):
        gr = GraphRuntime()
        new_graph = _StubGraph(layer_idx=7)
        record = {"layer_idx": 7, "wall_ms": 1.0}
        fut = Future()
        fut.set_result((new_graph, record))
        gr.repartition_in_flight[7] = fut
        gr.collect_completed_repartitions(step_idx=99)
        # New graph swapped in.
        assert gr.graphs[7] is new_graph
        # In-flight cleared, record stamped with step_idx and appended.
        assert 7 not in gr.repartition_in_flight
        assert gr.repartition_records == [{"layer_idx": 7, "wall_ms": 1.0, "completed_at_step": 99}]

    def test_collect_failed_worker_drops_inflight_no_record(self):
        gr = GraphRuntime()
        fut = Future()
        fut.set_exception(RuntimeError("worker died"))
        gr.repartition_in_flight[3] = fut
        gr.collect_completed_repartitions(step_idx=99)
        # Cleared, but no record appended (no swap, no graph in self.graphs).
        assert 3 not in gr.repartition_in_flight
        assert gr.repartition_records == []


class TestExecutorLifecycle:
    """``configure_workers`` (re-)builds the partition ThreadPoolExecutor
    + per-GPU semaphores; ``shutdown`` clears it. Idempotent in both
    directions so model swaps don't leak threads or device handles."""

    def test_configure_creates_executor_with_n_workers(self):
        gr = GraphRuntime()
        gr.configure_workers(n_layers=4)
        assert gr.executor is not None
        # ThreadPoolExecutor doesn't expose max_workers as a public attr,
        # but it's stored on the private ``_max_workers``.
        assert gr.executor._max_workers == 4
        gr.shutdown()

    def test_configure_no_semaphores_when_unlimited(self):
        gr = GraphRuntime()
        gr.configure_workers(n_layers=2, devices=["cuda:0", "cuda:1"], max_workers_per_gpu=None)
        assert gr.gpu_semaphores == {}
        gr.shutdown()

    def test_configure_semaphores_per_device_when_capped(self):
        gr = GraphRuntime()
        gr.configure_workers(n_layers=2, devices=["cuda:0", "cuda:1"], max_workers_per_gpu=3)
        assert set(gr.gpu_semaphores.keys()) == {"cuda:0", "cuda:1"}
        # Drain three permits per semaphore non-blockingly to confirm capacity 3.
        for sem in gr.gpu_semaphores.values():
            for _ in range(3):
                assert sem.acquire(blocking=False)
            assert not sem.acquire(blocking=False)
        gr.shutdown()

    def test_configure_idempotent_replaces_existing_executor(self):
        """Calling ``configure_workers`` after a previous call must shut down
        the old executor before creating a new one (so device handles
        stay consistent across model swaps)."""
        gr = GraphRuntime()
        gr.configure_workers(n_layers=2)
        first = gr.executor
        gr.configure_workers(n_layers=4)
        # New executor created; old one was shut down.
        assert gr.executor is not first
        assert gr.executor._max_workers == 4
        gr.shutdown()

    def test_shutdown_no_op_when_no_executor(self):
        gr = GraphRuntime()
        gr.shutdown()
        assert gr.executor is None

    def test_shutdown_clears_executor(self):
        gr = GraphRuntime()
        gr.configure_workers(n_layers=2)
        gr.shutdown()
        assert gr.executor is None
