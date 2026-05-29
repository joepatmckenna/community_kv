"""LongBench-v2 dataset adapter.

Prompt template, truncation strategy, and answer extraction match the
official THUDM/LongBench reference (``pred.py`` + ``prompts/0shot.txt``)
so headline accuracy is comparable to the published baselines.
"""

from __future__ import annotations

import argparse
import re

from evals.datasets.dataset import Dataset, register_dataset

TEMPLATE_0SHOT = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n"
    "$DOC$\n"
    "</text>\n\n"
    "What is the correct answer to this question: $Q$\n"
    "Choices:\n"
    "(A) $C_A$\n"
    "(B) $C_B$\n"
    "(C) $C_C$\n"
    "(D) $C_D$\n\n"
    'Format your response as follows: "The correct answer is (insert answer here)".'
)


_ANSWER_RE_PARENS = re.compile(r"The correct answer is \(([A-D])\)")
_ANSWER_RE_BARE = re.compile(r"The correct answer is ([A-D])")


def load_split(split: str) -> list[dict]:
    """All samples in the named length bucket, sorted shortest-first by char count."""
    from datasets import load_dataset

    ds = load_dataset("THUDM/LongBench-v2", split="train")
    samples = [s for s in ds if s.get("length") == split]
    samples.sort(key=lambda s: len(s.get("context", "")))
    return samples


@register_dataset("longbench-v2")
class LongBenchV2Dataset(Dataset):
    """LongBench-v2 multi-choice (A/B/C/D) eval driver."""

    def __init__(self, *, split: str = "short"):
        self.split = split

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--split",
            default="short",
            choices=["short", "medium", "long"],
            help="LongBench-v2 length bucket (default: short).",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "LongBenchV2Dataset":
        return cls(split=args.split)

    def load_samples(self) -> list[dict]:
        return load_split(self.split)

    def render_prompt(self, sample: dict) -> str:
        return (
            TEMPLATE_0SHOT.replace("$DOC$", sample.get("context", "").strip())
            .replace("$Q$", sample.get("question", "").strip())
            .replace("$C_A$", sample.get("choice_A", "").strip())
            .replace("$C_B$", sample.get("choice_B", "").strip())
            .replace("$C_C$", sample.get("choice_C", "").strip())
            .replace("$C_D$", sample.get("choice_D", "").strip())
        )

    def gold(self, sample: dict) -> str:
        return sample.get("answer", "?")

    def sample_id(self, sample: dict) -> str:
        return sample.get("_id") or "?"

    def lookup_name(self) -> str:
        return f"LongBench-v2:{self.split}"

    def extract_answer(self, response: str, sample: dict | None = None) -> str | None:
        """Official LongBench-v2 extractor: match ``The correct answer is (X)``
        or ``The correct answer is X`` (after stripping asterisks). Returns
        ``None`` when neither matches. ``sample`` is unused — kept for
        interface compatibility."""
        response = response.replace("*", "")
        m = _ANSWER_RE_PARENS.search(response)
        if m:
            return m.group(1)
        m = _ANSWER_RE_BARE.search(response)
        if m:
            return m.group(1)
        return None
