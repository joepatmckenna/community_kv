"""Tests for the ``Dataset`` protocol + registry + LongBench-v2 wiring.

CPU-only — these tests don't touch the model, runner, or CUDA. They
verify:
  * The registry decorator wires a class in by name.
  * Duplicate registration is an error.
  * ``get_dataset`` errors helpfully.
  * Two-stage argparse: CLI's ``parse_args`` picks up the right dataset
    class and adds its dataset-specific args.
  * ``LongBenchV2Dataset`` is registered under ``longbench-v2`` and its
    ``add_args`` / ``from_args`` round-trip cleanly.
"""

from __future__ import annotations

import argparse

import pytest

from evals.datasets.dataset import (
    DATASET_REGISTRY,
    Dataset,
    get_dataset,
    register_dataset,
)
from evals.datasets.longbench_v2 import LongBenchV2Dataset


class TestRegistry:
    def test_longbench_v2_is_registered(self):
        assert DATASET_REGISTRY["longbench-v2"] is LongBenchV2Dataset
        assert LongBenchV2Dataset.name == "longbench-v2"

    def test_get_dataset_returns_class(self):
        assert get_dataset("longbench-v2") is LongBenchV2Dataset

    def test_get_dataset_unknown_raises_systemexit(self):
        with pytest.raises(SystemExit, match="unknown --dataset"):
            get_dataset("does-not-exist")

    def test_decorator_rejects_duplicate(self):
        @register_dataset("test-dup-temp")
        class _A(Dataset):
            pass

        try:
            with pytest.raises(ValueError, match="already registered"):

                @register_dataset("test-dup-temp")
                class _B(Dataset):
                    pass

        finally:
            DATASET_REGISTRY.pop("test-dup-temp", None)

    def test_decorator_sets_class_name(self):
        @register_dataset("test-name-temp")
        class _C(Dataset):
            pass

        try:
            assert _C.name == "test-name-temp"
        finally:
            DATASET_REGISTRY.pop("test-name-temp", None)


class TestLongBenchV2Dataset:
    """Behaviour of the registered dataset *without* hitting the network.
    ``load_samples`` is exercised indirectly elsewhere; here we just pin
    the surface (constructor / argparse / answer extraction / gold)."""

    def test_default_construction(self):
        d = LongBenchV2Dataset()
        assert d.split == "short"

    def test_from_args(self):
        ns = argparse.Namespace(split="medium")
        d = LongBenchV2Dataset.from_args(ns)
        assert d.split == "medium"

    def test_add_args_attaches_to_parser(self):
        p = argparse.ArgumentParser()
        LongBenchV2Dataset.add_args(p)
        ns = p.parse_args(["--split", "long"])
        assert ns.split == "long"

    def test_add_args_defaults(self):
        p = argparse.ArgumentParser()
        LongBenchV2Dataset.add_args(p)
        ns = p.parse_args([])
        assert ns.split == "short"

    def test_format_prompt_substitutes_all_fields(self):
        d = LongBenchV2Dataset()
        sample = {
            "context": "  doc text  ",
            "question": " q? ",
            "choice_A": "a-choice",
            "choice_B": "b-choice",
            "choice_C": "c-choice",
            "choice_D": "d-choice",
        }
        out = d.format_prompt(sample)
        assert "doc text" in out
        assert "q?" in out
        for letter in "ABCD":
            assert f"({letter}) {letter.lower()}-choice" in out

    def test_gold_returns_answer_field(self):
        d = LongBenchV2Dataset()
        assert d.gold({"answer": "B"}) == "B"
        assert d.gold({}) == "?"

    def test_sample_id_returns_full_id(self):
        d = LongBenchV2Dataset()
        assert d.sample_id({"_id": "abcdefghij"}) == "abcdefghij"
        assert d.sample_id({}) == "?"

    def test_lookup_name_includes_split(self):
        assert LongBenchV2Dataset(split="short").lookup_name() == "LongBench-v2:short"
        assert LongBenchV2Dataset(split="long").lookup_name() == "LongBench-v2:long"

    def test_extract_answer_matches_official_phrase(self):
        d = LongBenchV2Dataset()
        assert d.extract_answer("The correct answer is (C)") == "C"
        assert d.extract_answer("The correct answer is C") == "C"
        assert d.extract_answer("Answer: C") is None  # bare letter, not the official phrase
        assert d.extract_answer("no answer here") is None


_HAS_TRANSFORMERS = True
try:  # noqa: SIM105
    import transformers  # noqa: F401
except Exception:
    _HAS_TRANSFORMERS = False


@pytest.mark.skipif(
    not _HAS_TRANSFORMERS,
    reason="evals.main pulls in transformers at import time",
)
class TestParseArgsTwoStage:
    """The CLI's two-stage parse must pick up the right dataset class and
    attach its dataset-specific args. Run it without invoking the rest of
    ``main()``."""

    def test_dataset_required(self, monkeypatch):
        from evals import main as eval_main

        monkeypatch.setattr("sys.argv", ["community-kv-eval"])
        with pytest.raises(SystemExit):
            eval_main.parse_args()

    def test_dataset_specific_args_picked_up(self, monkeypatch):
        from evals import main as eval_main

        monkeypatch.setattr(
            "sys.argv",
            [
                "community-kv-eval",
                "--dataset",
                "longbench-v2",
                "--split",
                "long",
                "--max_samples",
                "7",
            ],
        )
        args, dataset_cls = eval_main.parse_args()
        assert dataset_cls is LongBenchV2Dataset
        assert args.dataset == "longbench-v2"
        assert args.split == "long"
        assert args.max_samples == 7

    def test_unknown_dataset_exits(self, monkeypatch):
        from evals import main as eval_main

        monkeypatch.setattr("sys.argv", ["community-kv-eval", "--dataset", "no-such"])
        with pytest.raises(SystemExit, match="unknown --dataset"):
            eval_main.parse_args()
