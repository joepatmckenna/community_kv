from __future__ import annotations

import math

import torch

from community_kv.graph.leiden import algorithm


def test_densify_relabels_each_graph_without_cross_graph_aliases() -> None:
    labels = torch.tensor([2, 2, 5, 9, 9, 9], dtype=torch.int32)

    dense, width = algorithm._densify_per_graph(labels, G=2, seq_len=3)

    assert width == 2
    assert dense.tolist() == [0, 0, 1, 2, 2, 2]


def test_disjoint_modularity_matches_two_component_reference() -> None:
    src = torch.tensor([0, 2], dtype=torch.int32)
    dst = torch.tensor([1, 3], dtype=torch.int32)
    weight = torch.ones(2)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

    modularity = algorithm._compute_disjoint_modularity(
        src,
        dst,
        weight,
        labels,
        1.0,
    )

    assert math.isclose(modularity, 0.5, abs_tol=1e-6)


def test_run_leiden_handles_isolates_and_explicit_boltzmann_parameters(
    monkeypatch,
) -> None:
    calls = []

    class _Module:
        @staticmethod
        def batched_leiden(*args):
            calls.append(args)
            return torch.tensor([0, 0, -1, 3], dtype=torch.int32), 0.0

        @staticmethod
        def aggregate_coo(*args):
            raise AssertionError("single-level run must not aggregate")

    monkeypatch.setattr(algorithm, "_load_module", lambda: _Module)
    src = torch.tensor([0], dtype=torch.int32)
    dst = torch.tensor([1], dtype=torch.int32)
    weight = torch.ones(1)

    vertex, partition, modularity = algorithm.run_leiden(
        src,
        dst,
        weight,
        G=1,
        seq_len=4,
        max_level=1,
        theta=0.25,
        max_inner_iter=7,
        use_boltzmann=True,
    )

    assert vertex.tolist() == [0, 1, 3]
    assert partition.tolist() == [0, 0, 2]
    assert math.isfinite(modularity)
    assert calls[0][7] == 0.25
    assert calls[0][8] == 7
    assert calls[0][10] is True


def test_run_leiden_applies_environment_overrides(monkeypatch) -> None:
    calls = []

    class _Module:
        @staticmethod
        def batched_leiden(*args):
            calls.append(args)
            return torch.tensor([0, 0], dtype=torch.int32), 0.0

        @staticmethod
        def aggregate_coo(*args):
            raise AssertionError("single-level run must not aggregate")

    monkeypatch.setattr(algorithm, "_load_module", lambda: _Module)
    monkeypatch.setenv("COMMUNITY_KV_LEIDEN_THETA", "0.125")
    monkeypatch.setenv("COMMUNITY_KV_LEIDEN_MAX_INNER_ITER", "11")
    monkeypatch.setenv("COMMUNITY_KV_LEIDEN_BOLTZMANN", "1")

    algorithm.run_leiden(
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.ones(1),
        G=1,
        seq_len=2,
        max_level=1,
    )

    assert calls[0][7] == 0.125
    assert calls[0][8] == 11
    assert calls[0][10] is True
