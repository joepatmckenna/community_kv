"""Dataset abstraction and registry for CommunityKV evaluations.

Implementations load samples, render prompts, extract predictions, score
responses, and optionally truncate samples to a token budget. The registry
provides one stable lookup surface for evaluation drivers.
"""

from __future__ import annotations

import argparse
from typing import Callable

import torch


class Dataset:
    """Base class. Subclasses override the methods below; the ``@register_dataset``
    decorator wires each subclass into ``DATASET_REGISTRY``.

    Subclass methods must not depend on mutable module-level state.
    """

    name: str = ""  # registry key — set by the @register_dataset decorator

    # Where ``fit_sample`` caches the truncated prompt on a sample dict.
    # Standard across datasets so the cache lives in a known place.
    _PROMPT_CACHE_KEY = "_truncated_prompt"

    # ---- argparse plumbing ---------------------------------------------- #

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add dataset-specific args to ``parser``. No-op by default."""

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Dataset":
        """Construct an instance from the post-parse args namespace.
        Default: no-arg constructor."""
        return cls()

    # ---- sample I/O ----------------------------------------------------- #

    def load_samples(self) -> list[dict]:
        """Return all samples for this evaluation run. Caller iterates."""
        raise NotImplementedError

    def render_prompt(self, sample: dict) -> str:
        """Render the bare prompt for ``sample`` (no chat-template wrapping,
        no truncation cache lookup). Subclasses must implement."""
        raise NotImplementedError

    def format_prompt(self, sample: dict) -> str:
        """Return the prompt for ``sample``, honoring the truncation cache
        if ``fit_sample`` has run. Default: cache lookup → fall back to
        ``render_prompt``."""
        cached = sample.get(self._PROMPT_CACHE_KEY)
        if cached is not None:
            return cached
        return self.render_prompt(sample)

    def tokenize(self, tokenizer, sample: dict) -> torch.Tensor:
        """Render ``sample`` to input-token ids for the model.

        Default implementation wraps ``self.format_prompt(sample)`` in the
        tokenizer's chat template (``add_generation_prompt=True``,
        ``enable_thinking=False``). Override when a dataset / task / model
        combination needs a different shape — e.g. raw completion-style
        prompts, custom system messages, multi-turn formatting.
        """
        from evals.utils import tokenize_chat_prompt

        return tokenize_chat_prompt(tokenizer, self.format_prompt(sample))

    def gold(self, sample: dict) -> str:
        """Reference answer string for accuracy comparison."""
        raise NotImplementedError

    def sample_id(self, sample: dict) -> str:
        """Return a stable sample identifier, or ``"?"`` when unavailable."""
        return "?"

    def lookup_name(self) -> str | None:
        """Return a stable name for this configured benchmark."""
        return None

    def extract_answer(self, response: str, sample: dict | None = None) -> str | None:
        """Pull the prediction out of free-form model output. Return ``None``
        when nothing matches.

        Datasets whose scoring depends on the gold question (e.g. BABILong's
        question-label exclusion) accept the sample via the optional second
        arg. Datasets that don't need it ignore the arg.
        """
        raise NotImplementedError

    def score(self, response: str, sample: dict) -> float:
        """Return per-sample accuracy in ``[0, 1]``.

        Exact-match datasets use this default. Benchmarks with fractional
        metrics, such as RULER, override it.
        """
        prediction = self.extract_answer(response, sample)
        return float(prediction is not None and prediction == self.gold(sample))

    # ---- optional: truncation ------------------------------------------- #

    def fit_sample(self, sample: dict, tokenizer, max_len: int) -> bool:
        """Middle-out-truncate the rendered prompt to fit ``max_len`` tokens
        and cache the result for the next ``format_prompt`` call. Returns
        True iff truncation happened.

        Default uses the canonical BABILong / LongBench-v2 truncation
        (``evals.utils.middle_out_truncate_prompt``). Datasets that
        don't need any truncation can override to return False
        unconditionally; datasets that need a different cut strategy
        override with custom logic.
        """
        from evals.utils import middle_out_truncate_prompt, tokenize_chat_prompt

        prompt = self.render_prompt(sample)

        def final_length(text: str) -> int:
            if hasattr(tokenizer, "apply_chat_template"):
                return int(tokenize_chat_prompt(tokenizer, text).shape[-1])
            return len(tokenizer.encode(text))

        final_len = final_length(prompt)
        if final_len <= max_len:
            return False

        bare_len = len(tokenizer.encode(prompt))
        # Account for the model-specific chat wrapper, then tighten by the
        # measured residual if token-boundary merges make the estimate differ.
        bare_cap = max(1, max_len - max(final_len - bare_len, 0))
        while True:
            truncated, _ = middle_out_truncate_prompt(prompt, tokenizer, bare_cap)
            fitted_len = final_length(truncated)
            if fitted_len <= max_len:
                sample[self._PROMPT_CACHE_KEY] = truncated
                return True
            bare_cap -= max(1, fitted_len - max_len)
            if bare_cap <= 0:
                raise ValueError(
                    f"chat template overhead exceeds requested max_len={max_len}"
                )


DATASET_REGISTRY: dict[str, type[Dataset]] = {}


def register_dataset(name: str) -> Callable[[type[Dataset]], type[Dataset]]:
    """Class decorator: register ``cls`` under ``name`` in ``DATASET_REGISTRY``."""

    def decorate(cls: type[Dataset]) -> type[Dataset]:
        if name in DATASET_REGISTRY:
            raise ValueError(f"dataset {name!r} already registered")
        cls.name = name
        DATASET_REGISTRY[name] = cls
        return cls

    return decorate


def get_dataset(name: str) -> type[Dataset]:
    """Look up a dataset class, listing registered names on failure."""
    if name not in DATASET_REGISTRY:
        known = ", ".join(sorted(DATASET_REGISTRY)) or "(none registered)"
        raise SystemExit(
            f"unknown --dataset {name!r}. Known: {known}. "
            f"(Make sure the dataset's module is imported so its "
            f"@register_dataset decorator runs.)"
        )
    return DATASET_REGISTRY[name]
