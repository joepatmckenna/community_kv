"""Tune one frozen Leiden resolution per model layer on held-out PG19 text."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import torch

from community_kv import (
    GraphAggregation,
    leiden_max_iterations,
    partition_graphs,
    prefill_attention_topk,
)
from evals.models import (
    ContextStrategy,
    HF_REVISION,
    base_model,
    load_causal_model,
    load_tokenizer,
    prepare_model,
)
from evals.resolutions import DEFAULT_RESOLUTIONS, write_model_resolutions


PG19_DATASET = "emozilla/pg19-test"
PG19_SPLIT = "test"
ATTENTION_IMPLEMENTATION = "community_kv_resolution_capture"


@dataclass(frozen=True, slots=True)
class CalibrationPrompt:
    row: int
    token_ids: tuple[int, ...]
    sha256: str


@dataclass(slots=True)
class LayerCapture:
    topk_indices: torch.Tensor
    topk_scores: torch.Tensor
    num_kv_heads: int
    sequence_length: int


@dataclass(frozen=True, slots=True)
class SearchResult:
    resolution: float
    mean_community_size: float
    evaluations: int


def _token_sha256(token_ids: Iterable[int]) -> str:
    serialized = ",".join(str(token) for token in token_ids).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()


def middle_out_calibration_prompt(
    prompt: CalibrationPrompt,
    max_tokens: int,
) -> CalibrationPrompt:
    """Retain equal-sized prefix and suffix slices of a calibration prompt."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if len(prompt.token_ids) <= max_tokens:
        return prompt
    head = max_tokens // 2
    token_ids = prompt.token_ids[:head] + prompt.token_ids[-(max_tokens - head) :]
    return CalibrationPrompt(
        row=prompt.row,
        token_ids=token_ids,
        sha256=_token_sha256(token_ids),
    )


def select_pg19_prompts(
    rows: Iterable[dict[str, Any]],
    tokenizer: Any,
    *,
    count: int,
    prompt_tokens: Iterable[int],
) -> Iterator[CalibrationPrompt]:
    """Select deterministic PG19 prompts independently at each context length."""

    token_counts = tuple(prompt_tokens)
    if (
        count <= 0
        or not token_counts
        or any(tokens <= 0 for tokens in token_counts)
    ):
        raise ValueError("count and prompt_tokens must be positive")
    if tuple(sorted(set(token_counts))) != token_counts:
        raise ValueError("prompt_tokens must be strictly increasing")

    selected: dict[int, list[CalibrationPrompt]] = {
        tokens: [] for tokens in token_counts
    }
    max_tokens = token_counts[-1]
    for row_index, row in enumerate(rows):
        text = row.get("text")
        if not isinstance(text, str):
            continue
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
            return_attention_mask=False,
        )
        token_ids = tuple(int(token) for token in encoded["input_ids"])
        for tokens in token_counts:
            prompts = selected[tokens]
            if len(prompts) == count or len(token_ids) < tokens:
                continue
            prompt_ids = token_ids[:tokens]
            prompts.append(
                CalibrationPrompt(
                    row=row_index,
                    token_ids=prompt_ids,
                    sha256=_token_sha256(prompt_ids),
                )
            )
        if all(len(prompts) == count for prompts in selected.values()):
            for tokens in token_counts:
                yield from selected[tokens]
            return

    missing = ", ".join(
        f"{tokens}: {len(prompts)}/{count}"
        for tokens, prompts in selected.items()
        if len(prompts) < count
    )
    raise RuntimeError(
        "PG19 does not contain enough qualifying prompts by token count "
        f"({missing})"
    )


def geometric_median(values: Iterable[float]) -> float:
    """Median in log-resolution space, robust to multiplicative outliers."""

    materialized = [float(value) for value in values]
    if not materialized or any(
        not math.isfinite(value) or value <= 0 for value in materialized
    ):
        raise ValueError("resolution candidates must be finite and positive")
    return math.exp(statistics.median(math.log(value) for value in materialized))


def bisect_resolution(
    evaluate: Callable[[float], float],
    *,
    target: float,
    tolerance: float,
    resolution_min: float,
    resolution_max: float,
    max_steps: int,
) -> SearchResult:
    """Log-space search for a resolution yielding the target community size."""

    if not all(
        math.isfinite(value) and value > 0
        for value in (target, tolerance, resolution_min, resolution_max)
    ):
        raise ValueError("search values must be finite and positive")
    if resolution_min >= resolution_max:
        raise ValueError("resolution_min must be below resolution_max")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    def checked_evaluate(resolution: float) -> float:
        size = float(evaluate(resolution))
        if not math.isfinite(size) or size <= 0:
            raise RuntimeError(
                f"invalid mean community size {size!r} at resolution {resolution}"
            )
        return size

    low_size = checked_evaluate(resolution_min)
    if low_size <= target:
        return SearchResult(resolution_min, low_size, 1)
    high_size = checked_evaluate(resolution_max)
    if high_size >= target:
        return SearchResult(resolution_max, high_size, 2)
    if low_size < high_size:
        raise RuntimeError("community size did not decrease with resolution")

    low = resolution_min
    high = resolution_max
    if abs(high_size - target) < abs(low_size - target):
        best_resolution, best_size = high, high_size
    else:
        best_resolution, best_size = low, low_size
    evaluations = 2
    for _ in range(max_steps):
        middle = math.sqrt(low * high)
        size = checked_evaluate(middle)
        evaluations += 1
        if abs(size - target) < abs(best_size - target):
            best_resolution = middle
            best_size = size
        if abs(size - target) <= tolerance * target:
            return SearchResult(middle, size, evaluations)
        if size > target:
            low = middle
        else:
            high = middle
    return SearchResult(best_resolution, best_size, evaluations)


def mean_non_sink_community_size(
    capture: LayerCapture,
    *,
    resolution: float,
    aggregation: GraphAggregation,
    num_sink: int,
    lam: float,
    leiden_seed: int,
) -> float:
    result = partition_graphs(
        capture.topk_indices,
        capture.topk_scores,
        aggregation=aggregation,
        num_kv_heads=capture.num_kv_heads,
        prefill_seq_len=capture.sequence_length,
        num_sink=num_sink,
        lam=lam,
        leiden_resolution=resolution,
        leiden_seed=leiden_seed,
    )
    return _mean_non_sink_community_size_from_ids(
        result.community_ids,
        num_sink=num_sink,
    )


def _mean_non_sink_community_size_from_ids(
    community_ids: torch.Tensor,
    *,
    num_sink: int,
) -> float:
    sequence_length = community_ids.shape[1]
    non_sink_ids = community_ids[:, num_sink:].long()
    present = torch.zeros(
        (non_sink_ids.shape[0], sequence_length),
        dtype=torch.bool,
        device=non_sink_ids.device,
    )
    present.scatter_(1, non_sink_ids, True)
    non_sink_counts = present.sum(dim=1).clamp_min(1)
    sizes = (sequence_length - num_sink) / non_sink_counts
    return float(sizes.mean().item())


def non_sink_community_size_distribution(
    community_ids: torch.Tensor,
    *,
    num_sink: int,
) -> dict[str, Any]:
    """Return exact community-cardinality histograms after excluding sinks."""

    if community_ids.ndim != 2:
        raise ValueError("community_ids must have shape [graphs, sequence]")
    if not 0 <= num_sink < community_ids.shape[1]:
        raise ValueError("num_sink must leave at least one non-sink token")

    pooled: dict[int, int] = {}
    graph_histograms: list[list[list[int]]] = []
    graph_mean_sizes: list[float] = []
    non_sink_ids = community_ids[:, num_sink:].to(device="cpu", dtype=torch.int64)
    for ids in non_sink_ids:
        cardinalities = torch.bincount(ids)
        cardinalities = cardinalities[cardinalities > 0]
        histogram: dict[int, int] = {}
        for size, count in zip(
            *torch.unique(cardinalities, return_counts=True),
            strict=True,
        ):
            size_int = int(size)
            count_int = int(count)
            histogram[size_int] = count_int
            pooled[size_int] = pooled.get(size_int, 0) + count_int
        graph_histograms.append(
            [[size, count] for size, count in sorted(histogram.items())]
        )
        graph_mean_sizes.append(
            float(ids.numel()) / max(1, int(cardinalities.numel()))
        )

    pooled_count = sum(pooled.values())
    singleton_count = pooled.get(1, 0)
    return {
        "graph_count": int(community_ids.shape[0]),
        "non_sink_token_count_per_graph": int(community_ids.shape[1] - num_sink),
        "non_sink_community_count": pooled_count,
        "mean_size_across_graphs": statistics.fmean(graph_mean_sizes),
        "singleton_fraction": (
            float(singleton_count) / pooled_count if pooled_count else 0.0
        ),
        "pooled_histogram": [
            [size, count] for size, count in sorted(pooled.items())
        ],
        "graph_histograms": graph_histograms,
    }


def tune_layer(
    capture: LayerCapture,
    *,
    aggregation: GraphAggregation,
    target: float,
    tolerance: float,
    resolution_min: float,
    resolution_max: float,
    max_steps: int,
    num_sink: int,
    lam: float,
    leiden_seed: int,
) -> tuple[SearchResult, dict[str, Any]]:
    """Tune one layer and measure exact cardinalities at the selected value."""

    partitions: dict[float, Any] = {}

    def evaluate(resolution: float) -> float:
        result = partition_graphs(
            capture.topk_indices,
            capture.topk_scores,
            aggregation=aggregation,
            num_kv_heads=capture.num_kv_heads,
            prefill_seq_len=capture.sequence_length,
            num_sink=num_sink,
            lam=lam,
            leiden_resolution=resolution,
            leiden_seed=leiden_seed,
        )
        partitions[resolution] = result
        return _mean_non_sink_community_size_from_ids(
            result.community_ids,
            num_sink=num_sink,
        )

    tuned = bisect_resolution(
        evaluate,
        target=target,
        tolerance=tolerance,
        resolution_min=resolution_min,
        resolution_max=resolution_max,
        max_steps=max_steps,
    )
    result = partitions[tuned.resolution]
    distribution = non_sink_community_size_distribution(
        result.community_ids,
        num_sink=num_sink,
    )
    measured = float(distribution["mean_size_across_graphs"])
    if not math.isclose(
        measured,
        tuned.mean_community_size,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise RuntimeError(
            "community-size distribution does not match the tuning metric "
            f"({measured} != {tuned.mean_community_size})"
        )
    return tuned, distribution


class ResolutionCaptureAttention:
    """FlashAttention adapter that retains only each layer's top-k graph."""

    def __init__(self, *, num_sink: int) -> None:
        self.num_sink = num_sink
        self.captures: dict[int, LayerCapture] = {}

    def register(self) -> None:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS[ATTENTION_IMPLEMENTATION] = self.forward

    def forward(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        del attention_mask, dropout, kwargs
        sequence_length = query.shape[2]
        if sequence_length <= 1:
            raise RuntimeError("resolution tuning accepts prefill passes only")
        key = key[:, :, :sequence_length, :]
        value = value[:, :, :sequence_length, :]
        result = prefill_attention_topk(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            softmax_scale=float(scaling),
            num_sink=self.num_sink,
        )
        self.captures[int(module.layer_idx)] = LayerCapture(
            topk_indices=result[5][0].detach().clone(),
            topk_scores=result[4][0].detach().clone(),
            num_kv_heads=int(key.shape[1]),
            sequence_length=sequence_length,
        )
        return result[0], None


def _load_pg19() -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit(
            "PG19 tuning requires the optional evaluation dependencies; "
            "install CommunityKV with the 'eval' extra."
        ) from error
    return load_dataset(
        PG19_DATASET,
        split=PG19_SPLIT,
        streaming=True,
        revision=HF_REVISION,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune one CommunityKV Leiden resolution per model layer using "
            "label-free held-out PG19 prompts."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="supported alias or generic Hugging Face causal model ID",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESOLUTIONS,
        help="resolution table to create or update",
    )
    parser.add_argument(
        "--community-sizes-output",
        type=Path,
        help=(
            "optional JSON output containing exact non-sink community-size "
            "histograms per prompt, layer, and graph"
        ),
    )
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        nargs="+",
        default=[2**13, 2**14, 2**15, 2**16],
        metavar="TOKENS",
    )
    parser.add_argument("--target", type=float, default=16.0)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--resolution-min", type=float, default=0.000001)
    parser.add_argument("--resolution-max", type=float, default=5000.0)
    parser.add_argument("--max-steps", type=int, default=18)
    parser.add_argument("--num-sink", type=int, default=10)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--leiden-seed", type=int, default=0)
    parser.add_argument(
        "--aggregation",
        choices=tuple(aggregation.value for aggregation in GraphAggregation),
        default=GraphAggregation.PER_QUERY_HEAD.value,
    )
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)
    positive = {
        "--num-prompts": args.num_prompts,
        "--target": args.target,
        "--tolerance": args.tolerance,
        "--resolution-min": args.resolution_min,
        "--resolution-max": args.resolution_max,
        "--max-steps": args.max_steps,
    }
    for option, value in positive.items():
        if not math.isfinite(float(value)) or value <= 0:
            parser.error(f"{option} must be positive")
    if any(tokens <= 0 for tokens in args.prompt_tokens):
        parser.error("--prompt-tokens values must be positive")
    if sorted(set(args.prompt_tokens)) != args.prompt_tokens:
        parser.error("--prompt-tokens values must be strictly increasing")
    if args.resolution_min >= args.resolution_max:
        parser.error("--resolution-min must be below --resolution-max")
    if args.num_sink < 0:
        parser.error("--num-sink must be non-negative")
    if not 0 <= args.lam <= 1:
        parser.error("--lam must be in [0, 1]")
    if args.prompt_tokens[0] <= args.num_sink + 8:
        parser.error("--prompt-tokens is too short for top-8 attention selection")
    if args.rope_factor is not None and (
        not math.isfinite(args.rope_factor) or args.rope_factor < 1
    ):
        parser.error("--rope-factor must be finite and at least 1")
    if (
        args.context_strategy == ContextStrategy.TRUNCATE.value
        and args.rope_factor is not None
    ):
        parser.error("--rope-factor requires --context-strategy extend")
    return args


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    aggregation = GraphAggregation(args.aggregation)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit(
            "resolution tuning requires a CUDA GPU and the compiled "
            "CommunityKV native extensions"
        )

    context_strategy = ContextStrategy(args.context_strategy)
    plans_by_tokens = {
        tokens: prepare_model(
            args.model,
            required_tokens=tokens,
            rope_factor=args.rope_factor,
            trust_remote_code=args.trust_remote_code,
            context_strategy=context_strategy,
        )
        for tokens in args.prompt_tokens
    }
    reference_plan = plans_by_tokens[args.prompt_tokens[-1]]
    pg19_rows = _load_pg19()
    tokenizer = load_tokenizer(
        reference_plan.profile,
        trust_remote_code=args.trust_remote_code,
    )
    prompts = list(
        select_pg19_prompts(
            pg19_rows,
            tokenizer,
            count=args.num_prompts,
            prompt_tokens=args.prompt_tokens,
        )
    )
    used_middle_out = False
    prompt_plans = []
    for prompt in prompts:
        plan = plans_by_tokens[len(prompt.token_ids)]
        max_tokens = len(prompt.token_ids)
        if context_strategy is ContextStrategy.TRUNCATE:
            max_tokens = min(max_tokens, plan.truncation_context_window)
        fitted = middle_out_calibration_prompt(prompt, max_tokens)
        used_middle_out |= len(fitted.token_ids) != len(prompt.token_ids)
        prompt_plans.append((fitted, plan))

    capture_attention = ResolutionCaptureAttention(num_sink=args.num_sink)
    capture_attention.register()
    num_layers = reference_plan.geometry.num_layers
    candidates: list[list[float]] = [[] for _ in range(num_layers)]
    distribution_payload = {
        "_meta": {
            "schema": "community-kv-community-size-distributions-v1",
            "model": reference_plan.profile.model_id,
            "aggregation": aggregation.value,
            "dataset": PG19_DATASET,
            "dataset_revision": HF_REVISION,
            "target_mean_non_sink_community_size": args.target,
            "num_sink": args.num_sink,
            "lam": args.lam,
        },
        "prompts": [
            {
                "row": prompt.row,
                "sha256": prompt.sha256,
                "tokens": len(prompt.token_ids),
                "rope_factor": plan.rope_factor,
                "layers": [],
            }
            for prompt, plan in prompt_plans
        ],
    }

    grouped_prompts: dict[float, list[tuple[int, CalibrationPrompt, Any]]] = {}
    for prompt_index, (prompt, plan) in enumerate(prompt_plans):
        grouped_prompts.setdefault(plan.rope_factor, []).append(
            (prompt_index, prompt, plan)
        )

    for rope_factor, group in grouped_prompts.items():
        plan = group[0][2]
        model = load_causal_model(
            plan,
            device=device,
            trust_remote_code=args.trust_remote_code,
        )
        model.config._attn_implementation = ATTENTION_IMPLEMENTATION
        transformer = base_model(model)
        print(
            f"loaded {plan.profile.model_id} with rope factor={rope_factor:g} "
            f"for {len(group)} prompts",
            flush=True,
        )
        for prompt_index, prompt, _ in group:
            prompt_number = prompt_index + 1
            capture_attention.captures.clear()
            input_ids = torch.tensor(
                prompt.token_ids,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            with torch.inference_mode():
                output = transformer(
                    input_ids=input_ids,
                    use_cache=False,
                    return_dict=True,
                )
            del output, input_ids
            actual_layers = set(capture_attention.captures)
            expected_layers = set(range(num_layers))
            if actual_layers != expected_layers:
                raise RuntimeError(
                    "incomplete FlashAttention capture: "
                    f"missing={sorted(expected_layers - actual_layers)}, "
                    f"extra={sorted(actual_layers - expected_layers)}"
                )

            for layer_idx in range(num_layers):
                capture = capture_attention.captures.pop(layer_idx)
                tuned, distribution = tune_layer(
                    capture,
                    aggregation=aggregation,
                    target=args.target,
                    tolerance=args.tolerance,
                    resolution_min=args.resolution_min,
                    resolution_max=args.resolution_max,
                    max_steps=args.max_steps,
                    num_sink=args.num_sink,
                    lam=args.lam,
                    leiden_seed=args.leiden_seed,
                )
                candidates[layer_idx].append(tuned.resolution)
                distribution_payload["prompts"][prompt_index]["layers"].append(
                    {
                        "layer": layer_idx,
                        "resolution": tuned.resolution,
                        "evaluations": tuned.evaluations,
                        **distribution,
                    }
                )
                if args.community_sizes_output is not None:
                    _write_json(args.community_sizes_output, distribution_payload)
                print(
                    f"prompt {prompt_number}/{len(prompts)} layer "
                    f"{layer_idx + 1}/{num_layers}: "
                    f"resolution={tuned.resolution:.8g}, "
                    f"mean_size={tuned.mean_community_size:.4f}, "
                    f"singleton_fraction={distribution['singleton_fraction']:.4f}",
                    flush=True,
                )
                del capture
            torch.cuda.empty_cache()
        del transformer, model
        capture_attention.captures.clear()
        gc.collect()
        torch.cuda.empty_cache()

    resolutions = [geometric_median(values) for values in candidates]
    calibration = {
        "dataset": PG19_DATASET,
        "split": PG19_SPLIT,
        "dataset_revision": HF_REVISION,
        "prompt_rows": [prompt.row for prompt, _ in prompt_plans],
        "prompt_sha256": [prompt.sha256 for prompt, _ in prompt_plans],
        "prompt_tokens": [len(prompt.token_ids) for prompt, _ in prompt_plans],
        "prompt_selection": (
            "first_qualifying_rows_by_context_middle_out"
            if used_middle_out
            else "first_qualifying_rows_by_context_first_tokens"
        ),
        "target_mean_non_sink_community_size": args.target,
        "aggregation": "geometric_median",
        "graph_aggregation": aggregation.value,
        "kappa": 8,
        "num_sink": args.num_sink,
        "lam": args.lam,
        "leiden_max_iter": [
            leiden_max_iterations(len(prompt.token_ids))
            for prompt, _ in prompt_plans
        ],
        "leiden_seed": args.leiden_seed,
        "resolution_min": args.resolution_min,
        "resolution_max": args.resolution_max,
        "tolerance": args.tolerance,
        "max_steps": args.max_steps,
        "rope_factor": [plan.rope_factor for _, plan in prompt_plans],
        "model_revision": HF_REVISION,
        "model_profile": reference_plan.profile.name,
        "model_geometry": {
            "model_type": reference_plan.geometry.model_type,
            "num_layers": reference_plan.geometry.num_layers,
            "num_attention_heads": reference_plan.geometry.num_attention_heads,
            "num_key_value_heads": reference_plan.geometry.num_key_value_heads,
            "hidden_size": reference_plan.geometry.hidden_size,
            "head_dim": reference_plan.geometry.head_dim,
            "max_position_embeddings": (
                reference_plan.geometry.max_position_embeddings
            ),
        },
    }
    write_model_resolutions(
        args.output,
        model=reference_plan.profile.model_id,
        aggregation=aggregation,
        resolutions=resolutions,
        calibration=calibration,
    )
    print(
        f"wrote {num_layers} resolutions for "
        f"{reference_plan.profile.model_id} to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
