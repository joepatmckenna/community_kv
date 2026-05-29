"""Eval runner — orchestrates per-sample prefill+decode and the iterate loop.

The runner is dataset-agnostic: sample formatting, gold-answer extraction,
and prediction parsing all go through the
:class:`~evals.datasets.dataset.Dataset` instance the caller passes in.

Sample scheduling: samples are processed sorted by token length within
the smallest rope_factor that fits them. Under ``--context_extension_strategy=yarn``
this means the model is rebuilt at progressively larger rope factors only
when the next sample needs more context. Under ``middle_out`` the model
is built once at the cap (``DEFAULT_MAX_ROPE_FACTOR``) and oversize
prompts are middle-out-truncated before iteration.

``EvalRunner`` is the public surface; its methods access state stored on
the instance (``args``, ``tokenizer``, ``dataset``, ``dist``,
``per_sample_resolutions``, plus internally-constructed
``graph_runtime``, ``native_max``, ``model``).
Locally-scoped or stateless helpers stay as module-level free functions
so they can be unit-tested without constructing a runner.
"""

from __future__ import annotations

import argparse
import functools
import gc
import math
import time
from dataclasses import dataclass, field
from itertools import groupby
from typing import TYPE_CHECKING

import torch

from community_kv.attention import CommunityKVAttention
from community_kv.graph.runtime import GraphRuntime
from community_kv.graph.state import GraphAggregation
from evals.distributed import DistContext
from evals.datasets.dataset import Dataset
from evals.models import build_model
from evals import resolutions as resmod
from evals.utils import DEFAULT_MAX_ROPE_FACTOR, compute_rope_factor

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


# --------------------------------------------------------------------------- #
# Module-level helpers (locally-scoped or stateless)
# --------------------------------------------------------------------------- #


def _perf_phase(method):
    """Bracket a method's body with ``dist.barrier`` + ``torch.cuda.synchronize``
    on entry and exit, time it with ``time.perf_counter``, and return
    ``(result, t_start, t_end)``.

    The wrapped method must be on an object with ``self.dist`` (a
    ``DistContext``). Use one decorated method per phase you want to time;
    multi-phase callers stitch the timestamps together themselves.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        self.dist.barrier()
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        result = method(self, *args, **kwargs)
        torch.cuda.synchronize()
        self.dist.barrier()
        t_end = time.perf_counter()
        return result, t_start, t_end

    return wrapper


def _model_layer_devices(model, dist_ctx: DistContext) -> list[torch.device]:
    """Unique devices in the model's hf_device_map, in stable order."""
    if dist_ctx.is_tp:
        return [torch.device(f"cuda:{dist_ctx.local_rank}")]
    seen: list[torch.device] = []
    for _, dev in (getattr(model, "hf_device_map", None) or {}).items():
        d = torch.device(dev) if not isinstance(dev, torch.device) else dev
        if d.type == "cuda" and d not in seen:
            seen.append(d)
    if not seen:
        seen.append(next(model.parameters()).device)
    return seen


def _resolve_input_device(model, dist_ctx: DistContext) -> torch.device:
    """Pick the device that prefill input tensors must live on."""
    if dist_ctx.is_tp:
        return torch.device(f"cuda:{dist_ctx.local_rank}")
    return model.device


def _free_model(model) -> None:
    """Drop a model's GPU memory before building the next one."""
    if model is None:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _avg_event_pair_ms(events: dict) -> float:
    """Mean elapsed time across all ``(start, end)`` CUDA-event pairs in
    a per-layer dict. Synchronizes events before reading."""
    if not events:
        return 0.0
    for layer_evs in events.values():
        for _, e in layer_evs:
            e.synchronize()
    tot = sum(s.elapsed_time(e) for evs in events.values() for s, e in evs)
    n = sum(len(evs) for evs in events.values())
    return tot / max(n, 1)


def _summarize_decode_events(
    graph_runtime: GraphRuntime, dist_ctx: DistContext
) -> tuple[float, float]:
    """Return ``(retrieve_ms_per_call, update_ms_per_call)`` averaged
    across the decode steps. Only computed on rank 0."""
    if dist_ctx.rank != 0:
        return 0.0, 0.0
    return (
        _avg_event_pair_ms(graph_runtime.decode_retrieve_events),
        _avg_event_pair_ms(graph_runtime.decode_update_events),
    )


def _format_sample_line(
    *,
    sample_idx: int,
    total: int,
    sid_short: str,
    factor: float,
    metrics: dict,
    pred: str | None,
    gold: str,
    stats: "_RunStats",
) -> str:
    """Per-sample log line."""
    correct = pred == gold
    resp = " ".join(metrics["response"].split())
    if len(resp) > 60:
        resp = resp[:57] + "..."
    return (
        f"[{sample_idx:3d}/{total}] _id={sid_short} f={factor:g}x "
        f"tok={metrics['n_tokens']:5d} "
        f"pred={pred or '?'} gold={gold} "
        f"{'OK ' if correct else 'BAD'} "
        f"acc={stats.running_acc:.3f}({stats.n_correct}/{stats.n_attempted}) "
        f"prefill={metrics['prefill_ms']:5.0f}ms "
        f"decode={metrics['decode_ms']:5.0f}ms "
        f"retr={metrics['retrieve_ms_per_call']:.2f}ms/c "
        f"upd={metrics['update_ms_per_call']:.2f}ms/c "
        f"avg_total={stats.avg_total:5.0f}ms "
        f"resp={resp!r}"
    )


@dataclass
class _RunStats:
    """Running tally over the iterate loop."""

    n_correct: int = 0
    n_attempted: int = 0
    sum_prefill: float = 0.0
    sum_decode: float = 0.0
    sum_total: float = 0.0
    sum_retr: float = 0.0
    sum_upd: float = 0.0
    sum_tokens: int = 0

    def record(self, metrics: dict, *, correct: bool) -> None:
        self.n_correct += int(correct)
        self.n_attempted += 1
        self.sum_prefill += metrics["prefill_ms"]
        self.sum_decode += metrics["decode_ms"]
        self.sum_total += metrics["total_ms"]
        self.sum_retr += metrics["retrieve_ms_per_call"]
        self.sum_upd += metrics["update_ms_per_call"]
        self.sum_tokens += metrics["n_tokens"]

    @property
    def running_acc(self) -> float:
        return self.n_correct / max(self.n_attempted, 1)

    @property
    def avg_total(self) -> float:
        return self.sum_total / max(self.n_attempted, 1)

    def print_final(self, dist_ctx: DistContext) -> None:
        if dist_ctx.rank != 0:
            return
        denom = max(self.n_attempted, 1)
        print()
        print("=== Final ===")
        print(f"  accuracy: {self.n_correct}/{self.n_attempted} = {self.n_correct/denom:.4f}")
        print(f"  avg prefill   : {self.sum_prefill/denom:7.0f} ms")
        print(f"  avg decode    : {self.sum_decode/denom:7.0f} ms")
        print(f"  avg total     : {self.sum_total/denom:7.0f} ms")
        print(f"  avg retr/call : {self.sum_retr/denom:7.3f} ms")
        print(f"  avg upd/call  : {self.sum_upd/denom:7.3f} ms")
        print(f"  avg n_tokens  : {self.sum_tokens/denom:7.0f}")
        print(f"  total wall    : {self.sum_total/1000:.1f} s")


# --------------------------------------------------------------------------- #
# EvalRunner
# --------------------------------------------------------------------------- #


@dataclass
class EvalRunner:
    """Holds the per-run context (args, tokenizer, dataset, dist info,
    optional per-sample resolutions) and exposes ``run_iterate`` /
    ``run_one_sample`` as methods so callers don't have to thread the
    same six-or-seven kwargs through every helper.

    The runner constructs its own ``GraphRuntime``, registers the
    ``CommunityKVAttention`` forward against the HF attention registry,
    and looks up the model's ``native_max`` from its config — these are
    all derivable from ``args`` and shouldn't be the caller's job.

    The held ``model`` evolves across rope-factor groups during
    ``run_iterate`` and is freed at the end. ``run_one_sample`` is also
    callable directly, but the caller must have set ``self.model``,
    ``graph_runtime.executor``, etc., themselves.
    """

    args: argparse.Namespace
    dataset: Dataset
    dist: DistContext
    per_sample_resolutions: dict[str, float] | None = None
    tokenizer: "PreTrainedTokenizerBase" = field(init=False)
    graph_runtime: GraphRuntime = field(init=False)
    native_max: int = field(init=False)
    model: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from transformers import AutoConfig, AutoTokenizer

        self.graph_runtime = GraphRuntime()
        attn = CommunityKVAttention(graph_runtime=self.graph_runtime)
        attn.register(
            kappa=self.args.kappa,
            num_sink=self.args.num_sink,
            lam=self.args.lam,
            leiden_resolution=self.args.leiden_resolution,
            leiden_max_iter=2,  # overwritten per-sample to floor(log10(n_tokens))
            max_new_tokens=self.args.max_new_tokens,
            token_budget=self.args.token_budget,
        )
        self.native_max = int(AutoConfig.from_pretrained(self.args.model).max_position_embeddings)
        self.dist.print0(f"Loading {self.args.model} tokenizer ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)

    # ---- public entry points -------------------------------------------- #

    def run_iterate(self, samples: list[dict]) -> None:
        """Loop over ``samples``, rebuilding the model whenever the next
        sample needs a larger rope_factor. Prints streaming accuracy +
        timing and a single final summary across all samples."""
        plan = self._plan_run(samples)
        self.graph_runtime.aggregation = GraphAggregation(self.args.aggregation)
        self.graph_runtime.repartition_every = int(self.args.repartition_every)
        self._print_run_header(samples)

        stats = _RunStats()
        sample_idx = 0
        total = len(plan)
        for factor, group_iter in groupby(plan, key=lambda fns: fns[0]):
            group = list(group_iter)
            _free_model(self.model)
            self.dist.print0(
                f"\n--- Building model at rope_factor={factor:g} for {len(group)} samples ---",
                flush=True,
            )
            self.model = build_model(self.args, rope_factor=factor, is_tp=self.dist.is_tp)
            self.graph_runtime.configure_workers(
                n_layers=self.model.config.num_hidden_layers,
                devices={str(d) for d in _model_layer_devices(self.model, self.dist)},
                max_workers_per_gpu=self.args.max_partition_workers_per_gpu,
            )

            for _, _, sample in group:
                sample_idx += 1
                self._process_sample(sample, sample_idx, total, factor, stats)

        stats.print_final(self.dist)
        _free_model(self.model)
        self.model = None
        self.graph_runtime.shutdown()

    def run_one_sample(self, sample: dict) -> dict:
        """Prefill + decode on one sample. Returns metrics + decoded text.

        Caller must have already set ``self.model``,
        ``graph_runtime.aggregation``, ``graph_runtime.executor``,
        ``graph_runtime.gpu_semaphores``, and
        ``graph_runtime.repartition_every``.
        """
        if self.model is None:
            raise RuntimeError(
                "EvalRunner.run_one_sample requires self.model to be set; "
                "call run_iterate or build a model manually first."
            )
        self.graph_runtime.reset_per_sample()

        input_ids = self.dataset.tokenize(self.tokenizer, sample)
        n_tokens = int(input_ids.shape[-1])
        input_ids = input_ids.to(_resolve_input_device(self.model, self.dist))

        self._apply_per_sample_overrides(sample, n_tokens)

        prefill_out, t_pf_start, t_pf_end, t_part_done = self._run_prefill(input_ids)
        generated, t_dc_start, t_dc_end = self._run_decode(prefill_out)

        output = torch.cat([input_ids] + generated, dim=-1)
        n_decode_tokens = output.shape[-1] - n_tokens
        response = self.tokenizer.decode(output[0][n_tokens:], skip_special_tokens=True)

        retrieve_ms_per_call, update_ms_per_call = _summarize_decode_events(
            self.graph_runtime, self.dist
        )

        return {
            "n_tokens": n_tokens,
            "n_decode_tokens": n_decode_tokens,
            "response": response,
            "prefill_ms": (t_pf_end - t_pf_start) * 1000.0,
            "barrier_ms": (t_part_done - t_pf_end) * 1000.0,
            "decode_ms": (t_dc_end - t_dc_start) * 1000.0,
            "total_ms": (t_dc_end - t_pf_start) * 1000.0,
            "retrieve_ms_per_call": retrieve_ms_per_call,
            "update_ms_per_call": update_ms_per_call,
        }

    # ---- internal: planning, header, per-sample apply ------------------- #

    def _plan_run(self, samples: list[dict]) -> list[tuple[float, int, dict]]:
        """Return ``(rope_factor, n_prompt_tokens, sample)`` tuples sorted
        ascending by rope_factor then by length.

        Under ``yarn`` strategy: each sample's rope_factor is the smallest
        power of two that fits its prompt + ``max_new_tokens``.
        Under ``middle_out`` strategy: every sample runs at
        ``DEFAULT_MAX_ROPE_FACTOR`` and oversize prompts are truncated in
        place to fit the rope-extended window minus decoder headroom.
        """
        cap_factor = float(DEFAULT_MAX_ROPE_FACTOR)
        self.dist.print0(
            f"\nPre-tokenizing {len(samples)} samples "
            f"(strategy={self.args.context_extension_strategy}, "
            f"native_max={self.native_max}) ...",
            flush=True,
        )

        annotated: list[tuple[float, int, dict]] = []
        for s in samples:
            n = int(self.dataset.tokenize(self.tokenizer, s).shape[-1])
            if self.args.context_extension_strategy == "middle_out":
                f = cap_factor
            else:
                f = compute_rope_factor(n + self.args.max_new_tokens, self.native_max)
            annotated.append((f, n, s))

        if self.args.context_extension_strategy == "middle_out":
            max_len = int(self.native_max * cap_factor) - self.args.max_new_tokens
            truncated = sum(
                int(self.dataset.fit_sample(s, self.tokenizer, max_len)) for _, _, s in annotated
            )
            self.dist.print0(
                f"middle_out: rope_factor={cap_factor}, max_len={max_len}, "
                f"truncated {truncated}/{len(samples)} samples",
                flush=True,
            )
        else:
            counts: dict[float, int] = {}
            for f, _, _ in annotated:
                counts[f] = counts.get(f, 0) + 1
            summary = ", ".join(f"{f:g}x: {c}" for f, c in sorted(counts.items()))
            self.dist.print0(f"yarn: rope_factor distribution = {{{summary}}}", flush=True)

        annotated.sort(key=lambda fns: (fns[0], fns[1]))
        return annotated

    def _print_run_header(self, samples: list[dict]) -> None:
        """Print the two-line iterate banner."""
        if not self.per_sample_resolutions:
            res_msg = f"{self.args.leiden_resolution}"
        elif resmod.is_nested(self.per_sample_resolutions):
            res_msg = "per-sample per-layer (tuned table)"
        else:
            res_msg = f"per_sample({len(self.per_sample_resolutions)} entries)"
        self.dist.print0(
            f"\n=== {self.dataset.name} iterate (N={len(samples)}, "
            f"model={self.args.model}, agg={self.args.aggregation}, "
            f"resolution={res_msg}, "
            f"max_new_tokens={self.args.max_new_tokens}, "
            f"token_budget={self.args.token_budget}, "
            f"strategy={self.args.context_extension_strategy}) ===",
            flush=True,
        )
        self.dist.print0(
            f"  repartition_every={self.graph_runtime.repartition_every}",
            flush=True,
        )

    @_perf_phase
    def _do_prefill(self, input_ids: torch.Tensor):
        """Single-phase prefill body — wrapped by ``_perf_phase`` to add
        barrier/sync/timer scaffolding."""
        with torch.no_grad():
            return self.model(input_ids=input_ids, use_cache=True)

    @_perf_phase
    def _drain_prefill_workers(self) -> None:
        """Wait for every prefill-time async-partition worker to complete.
        Wrapped by ``_perf_phase`` for the post-prefill timestamp. A worker
        that raises (e.g. CUDA OOM partitioning a large graph) leaves its
        layer's graph unpopulated, which decode would hit as an opaque
        assertion — so surface the failure here instead of swallowing it."""
        n_failed = 0
        last_exc: Exception | None = None
        for fut in self.graph_runtime.futures:
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 - report, don't crash the run
                n_failed += 1
                last_exc = e
        if n_failed:
            self.dist.print0(
                f"  WARNING: {n_failed}/{len(self.graph_runtime.futures)} "
                f"prefill partition workers failed; last error: "
                f"{type(last_exc).__name__}: {last_exc}",
                flush=True,
            )

    def _run_prefill(self, input_ids: torch.Tensor) -> tuple[object, float, float, float]:
        """Prefill + drain async partition workers. Returns
        ``(prefill_out, t_pf_start, t_pf_end, t_part_done)``."""
        prefill_out, t_pf_start, t_pf_end = self._do_prefill(input_ids)
        _, _, t_part_done = self._drain_prefill_workers()
        return prefill_out, t_pf_start, t_pf_end, t_part_done

    @_perf_phase
    def _run_decode(self, prefill_out) -> list[torch.Tensor]:
        """Greedy decode loop with per-step repartition orchestration on
        ``self.graph_runtime``. Returns the list of generated token tensors;
        ``_perf_phase`` adds ``(t_dc_start, t_dc_end)``."""
        past = prefill_out.past_key_values
        next_id = prefill_out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated = [next_id]

        with torch.no_grad():
            for step_idx_loop in range(self.args.max_new_tokens - 1):
                step_idx = step_idx_loop + 1
                self.graph_runtime.maybe_trigger_repartition(step_idx)

                step = self.model(next_id, past_key_values=past, use_cache=True)
                past = step.past_key_values
                next_id = step.logits[:, -1, :].argmax(-1, keepdim=True)
                generated.append(next_id)

                self.graph_runtime.dispatch_pending_repartitions()
                self.graph_runtime.collect_completed_repartitions(step_idx)

        return generated

    def _apply_per_sample_overrides(self, sample: dict, n_tokens: int) -> None:
        """Update per-sample graph knobs: tuned Leiden resolutions from the
        lookup table (nested per-layer, or legacy flat scalar) and always the
        per-prompt ``leiden_max_iter = floor(log10(n_tokens))``."""
        table = self.per_sample_resolutions
        if table is not None:
            sid = self.dataset.sample_id(sample)
            if resmod.is_nested(table):
                # Per-sample, per-layer: install this sample's {layer: res} map.
                # If the sample isn't in the table we fall back to
                # config["leiden_resolution"] for every layer -- warn loudly so
                # a missing precomputed resolution is never silent.
                layers = resmod.layer_resolutions(
                    table,
                    model=self.args.model,
                    lookup_name=self.dataset.lookup_name(),
                    sample_id=sid,
                )
                if layers:
                    self.graph_runtime.resolutions = dict(layers)
                else:
                    self.graph_runtime.resolutions = {}
                    self.dist.print0(
                        f"WARNING: no precomputed per-layer resolution for "
                        f"sample {sid} ({self.args.model} "
                        f"{self.dataset.lookup_name()}) -- falling back to "
                        f"default leiden_resolution="
                        f"{self.graph_runtime.config.get('leiden_resolution')}",
                        flush=True,
                    )
            else:
                tuned = table.get(sid)
                if tuned is not None:
                    self.graph_runtime.config["leiden_resolution"] = float(tuned)
        self.graph_runtime.config["leiden_max_iter"] = max(1, int(math.log10(max(n_tokens, 10))))

    def _process_sample(
        self,
        sample: dict,
        sample_idx: int,
        total: int,
        factor: float,
        stats: _RunStats,
    ) -> None:
        """Run one sample, update ``stats``, print the per-sample line."""
        sid = self.dataset.sample_id(sample)
        sid_short = sid[:8] if sid else "?"
        try:
            metrics = self.run_one_sample(sample)
        except Exception as e:
            self.dist.print0(
                f"[{sample_idx:3d}/{total}] _id={sid_short} f={factor:g}x "
                f"ERROR: {type(e).__name__}: {e}",
                flush=True,
            )
            torch.cuda.empty_cache()
            return

        pred = self.dataset.extract_answer(metrics["response"], sample)
        gold = self.dataset.gold(sample)
        stats.record(metrics, correct=(pred == gold))
        self.dist.print0(
            _format_sample_line(
                sample_idx=sample_idx,
                total=total,
                sid_short=sid_short,
                factor=factor,
                metrics=metrics,
                pred=pred,
                gold=gold,
                stats=stats,
            ),
            flush=True,
        )
        torch.cuda.empty_cache()
