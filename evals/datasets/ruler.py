"""Hugging Face RULER dataset adapter.

RULER examples are generated for a specific tokenizer and are completion-style
prompts. The adapter therefore selects a matching pre-generated repository and
does not apply a second chat template.
"""

from __future__ import annotations

import argparse

from evals.datasets.dataset import Dataset, register_dataset

SUPPORTED_TASKS = ("niah_multikey_2", "niah_multiquery", "vt", "cwe")
SUPPORTED_FAMILIES = ("qwen3", "llama3")
SUPPORTED_LENGTHS = ("32k", "64k")

_QWEN_LENGTHS = {"32k": "32768", "64k": "65536"}
_QWEN_REPOSITORY = "lighteval/RULER-{tokens}-Qwen-3-Instruct"
_LLAMA_REPOSITORY = "self-long/RULER-llama3-1M"


@register_dataset("ruler")
class RulerDataset(Dataset):
    """One tokenizer family, task, and context length from RULER."""

    def __init__(
        self,
        *,
        model_family: str = "qwen3",
        task: str = "niah_multikey_2",
        length: str = "64k",
    ):
        if model_family not in SUPPORTED_FAMILIES:
            raise ValueError(
                f"unknown model family {model_family!r}. "
                f"Supported: {SUPPORTED_FAMILIES}."
            )
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"unknown task {task!r}. Supported: {SUPPORTED_TASKS}.")
        if length not in SUPPORTED_LENGTHS:
            raise ValueError(
                f"unknown length {length!r}. Supported: {SUPPORTED_LENGTHS}."
            )
        self.model_family = model_family
        self.task = task
        self.length = length

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--task",
            default="niah_multikey_2",
            choices=SUPPORTED_TASKS,
            help="RULER task.",
        )
        parser.add_argument(
            "--length",
            default="64k",
            choices=SUPPORTED_LENGTHS,
            help="RULER context length.",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RulerDataset":
        return cls(
            model_family=args.model_family,
            task=args.task,
            length=args.length,
        )

    def load_samples(self) -> list[dict]:
        from datasets import load_dataset

        if self.model_family == "qwen3":
            repository = _QWEN_REPOSITORY.format(tokens=_QWEN_LENGTHS[self.length])
            dataset = load_dataset(repository, split=self.task)
        else:
            dataset = load_dataset(
                _LLAMA_REPOSITORY,
                f"{self.task}_{self.length}",
                split="validation",
            )
        samples = list(dataset)
        samples.sort(key=lambda sample: int(sample.get("length", 0)))
        return samples

    def render_prompt(self, sample: dict) -> str:
        return str(sample.get("input", ""))

    def tokenize(self, tokenizer, sample: dict):
        return tokenizer(self.format_prompt(sample), return_tensors="pt").input_ids

    def answers(self, sample: dict) -> tuple[str, ...]:
        values = sample.get("outputs")
        if values is None:
            values = sample.get("answers", ())
        return tuple(str(value) for value in values)

    def gold(self, sample: dict) -> str:
        return ",".join(self.answers(sample))

    def sample_id(self, sample: dict) -> str:
        return str(sample.get("index", "?"))

    def lookup_name(self) -> str:
        return f"ruler:{self.model_family}:{self.task}:{self.length}"

    def extract_answer(self, response: str, sample: dict | None = None) -> str | None:
        prediction = response.strip()
        return prediction or None

    def score(self, response: str, sample: dict) -> float:
        """Official RULER ``string_match_all`` score for one sample."""
        references = self.answers(sample)
        if not references:
            return 0.0
        prediction = response.lower()
        return sum(reference.lower() in prediction for reference in references) / len(
            references
        )
