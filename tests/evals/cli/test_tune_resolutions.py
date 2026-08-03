from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from community_kv import GraphAggregation
from evals.cli import tune_resolutions
from evals.cli.tune_resolutions import (
    CalibrationPrompt,
    LayerCapture,
    bisect_resolution,
    geometric_median,
    mean_non_sink_community_size,
    middle_out_calibration_prompt,
    non_sink_community_size_distribution,
    select_pg19_prompts,
    tune_layer,
)


def test_tuner_defaults_to_per_query_head() -> None:
    args = tune_resolutions._parse_args(["--model", "qwen3-8b"])
    assert args.aggregation == "per_query_head"
    assert args.context_strategy == "truncate"
    assert args.prompt_tokens == [8192, 16384, 32768, 65536]
    assert args.resolution_min == 0.000001


def test_middle_out_calibration_retains_prefix_and_suffix() -> None:
    prompt = CalibrationPrompt(
        row=3,
        token_ids=tuple(range(10)),
        sha256="original",
    )
    truncated = middle_out_calibration_prompt(prompt, 5)
    assert truncated.row == 3
    assert truncated.token_ids == (0, 1, 7, 8, 9)
    assert truncated.sha256 != prompt.sha256


class _Tokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int,
        return_attention_mask: bool,
    ) -> dict[str, list[int]]:
        assert not add_special_tokens
        assert truncation
        assert not return_attention_mask
        return {
            "input_ids": [
                sum(word.encode("utf-8")) for word in text.split()[:max_length]
            ]
        }


def test_pg19_selection_is_deterministic_and_skips_short_rows() -> None:
    rows = [
        {"text": "too short"},
        {"text": "zero one two three four five"},
        {"missing": "text"},
        {"text": "six seven eight nine ten eleven"},
    ]
    selected = list(
        select_pg19_prompts(
            rows,
            _Tokenizer(),
            count=2,
            prompt_tokens=[4, 6],
        )
    )
    assert [prompt.row for prompt in selected] == [1, 3, 1, 3]
    assert [len(prompt.token_ids) for prompt in selected] == [4, 4, 6, 6]
    assert len({prompt.sha256 for prompt in selected}) == 4


def test_prompt_token_grid_must_be_strictly_increasing() -> None:
    with pytest.raises(SystemExit):
        tune_resolutions._parse_args(
            ["--model", "qwen3-8b", "--prompt-tokens", "16384", "8192"]
        )


def test_log_bisection_finds_target() -> None:
    result = bisect_resolution(
        lambda resolution: 64.0 / resolution,
        target=8.0,
        tolerance=0.01,
        resolution_min=1.0,
        resolution_max=64.0,
        max_steps=20,
    )
    assert result.resolution == pytest.approx(8.0, rel=0.01)
    assert result.mean_community_size == pytest.approx(8.0, rel=0.01)


def test_prompt_candidates_use_geometric_median() -> None:
    assert geometric_median([1.0, 4.0]) == pytest.approx(2.0)
    assert geometric_median([1.0, 4.0, 16.0]) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        geometric_median([math.nan])


def test_mean_community_size_counts_exact_non_sink_communities(monkeypatch) -> None:
    monkeypatch.setattr(
        tune_resolutions,
        "partition_graphs",
        lambda *args, **kwargs: SimpleNamespace(
            community_ids=torch.tensor(
                [
                    [0, 1, 2, 2, 3, 3, 3, 4],
                    [0, 1, 2, 2, 2, 2, 3, 3],
                ],
                dtype=torch.int32,
            )
        ),
    )
    capture = LayerCapture(
        topk_indices=torch.empty((2, 1, 8), dtype=torch.int32),
        topk_scores=torch.empty((2, 1, 8)),
        num_kv_heads=2,
        sequence_length=8,
    )
    size = mean_non_sink_community_size(
        capture,
        resolution=1.0,
        aggregation=GraphAggregation.PER_QUERY_HEAD,
        num_sink=2,
        lam=0.5,
        leiden_seed=0,
    )
    assert size == pytest.approx(2.5)


def test_exact_non_sink_community_size_distribution() -> None:
    distribution = non_sink_community_size_distribution(
        torch.tensor(
            [
                [0, 1, 2, 2, 3, 3, 3, 4],
                [0, 1, 2, 2, 2, 2, 3, 3],
            ],
            dtype=torch.int32,
        ),
        num_sink=2,
    )
    assert distribution["graph_count"] == 2
    assert distribution["non_sink_token_count_per_graph"] == 6
    assert distribution["non_sink_community_count"] == 5
    assert distribution["mean_size_across_graphs"] == pytest.approx(2.5)
    assert distribution["singleton_fraction"] == pytest.approx(0.2)
    assert distribution["pooled_histogram"] == [
        [1, 1],
        [2, 2],
        [3, 1],
        [4, 1],
    ]
    assert distribution["graph_histograms"] == [
        [[1, 1], [2, 1], [3, 1]],
        [[2, 1], [4, 1]],
    ]


def test_tune_layer_reuses_the_partition_selected_during_search(monkeypatch) -> None:
    calls = 0

    def partition(*args, **kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            community_ids=torch.arange(4, dtype=torch.int32).unsqueeze(0)
        )

    monkeypatch.setattr(tune_resolutions, "partition_graphs", partition)
    capture = LayerCapture(
        topk_indices=torch.empty((1, 1, 8), dtype=torch.int32),
        topk_scores=torch.empty((1, 1, 8)),
        num_kv_heads=1,
        sequence_length=4,
    )
    tuned, distribution = tune_layer(
        capture,
        aggregation=GraphAggregation.PER_QUERY_HEAD,
        target=2.0,
        tolerance=0.05,
        resolution_min=0.001,
        resolution_max=100.0,
        max_steps=10,
        num_sink=0,
        lam=0.5,
        leiden_seed=0,
    )

    assert calls == 1
    assert tuned.mean_community_size == 1.0
    assert distribution["mean_size_across_graphs"] == 1.0
