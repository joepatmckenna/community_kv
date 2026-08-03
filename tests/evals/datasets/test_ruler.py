import argparse
import sys
from types import SimpleNamespace

import pytest

from evals.datasets import DATASET_REGISTRY
from evals.datasets.ruler import RulerDataset


class FakeTokenizer:
    def __call__(self, text, return_tensors):
        assert return_tensors == "pt"
        return SimpleNamespace(input_ids=SimpleNamespace(shape=(1, len(text.split()))))

    def encode(self, text):
        return text.split()

    def decode(self, token_ids, *, skip_special_tokens=True):
        assert skip_special_tokens
        return " ".join(token_ids)


def test_registered():
    assert DATASET_REGISTRY["ruler"] is RulerDataset


def test_args_round_trip():
    parser = argparse.ArgumentParser()
    RulerDataset.add_args(parser)
    args = parser.parse_args(["--task", "vt", "--length", "32k"])
    args.model_family = "llama3"
    dataset = RulerDataset.from_args(
        args
    )
    assert dataset.lookup_name() == "ruler:llama3:vt:32k"


def test_rejects_unsupported_task():
    with pytest.raises(ValueError, match="unknown task"):
        RulerDataset(task="qa_1")


def test_prompt_and_answers_support_both_huggingface_schemas():
    dataset = RulerDataset()
    qwen = {"index": 3, "input": "prompt", "outputs": ["A", "B"]}
    llama = {"input": "prompt", "answers": ["A", "B"]}
    assert dataset.render_prompt(qwen) == "prompt"
    assert dataset.gold(qwen) == "A,B"
    assert dataset.gold(llama) == "A,B"
    assert dataset.sample_id(qwen) == "3"


def test_official_string_match_all_score():
    dataset = RulerDataset(task="vt")
    sample = {"outputs": ["ALPHA", "BETA", "GAMMA"]}
    assert dataset.score("alpha and gamma", sample) == pytest.approx(2 / 3)
    assert dataset.score("nothing", sample) == 0.0


def test_completion_prompt_is_not_chat_wrapped():
    dataset = RulerDataset()
    ids = dataset.tokenize(FakeTokenizer(), {"input": "one two three"})
    assert ids.shape == (1, 3)


def test_fit_sample_uses_middle_out_truncation():
    dataset = RulerDataset()
    sample = {"input": "one two three four"}
    assert dataset.fit_sample(sample, FakeTokenizer(), max_len=3)
    assert dataset.format_prompt(sample) == "one three four"


@pytest.mark.parametrize(
    "family,expected",
    [
        (
            "qwen3",
            (
                "lighteval/RULER-65536-Qwen-3-Instruct",
                None,
                "niah_multikey_2",
            ),
        ),
        (
            "llama3",
            (
                "self-long/RULER-llama3-1M",
                "niah_multikey_2_64k",
                "validation",
            ),
        ),
    ],
)
def test_huggingface_loader_contract(monkeypatch, family, expected):
    calls = []

    def load_dataset(repository, config=None, *, split):
        calls.append((repository, config, split))
        return [
            {"index": 2, "input": "longer", "outputs": ["B"], "length": 20},
            {"index": 1, "input": "shorter", "outputs": ["A"], "length": 10},
        ]

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=load_dataset),
    )
    samples = RulerDataset(model_family=family).load_samples()
    assert calls == [expected]
    assert [sample["index"] for sample in samples] == [1, 2]
