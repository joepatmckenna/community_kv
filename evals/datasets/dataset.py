"""``Dataset`` abstraction + registry for community-kv-eval.

Each dataset implementation:
    * Declares its own CLI args via ``add_args(parser)``.
    * Materializes itself from parsed args via ``from_args(args)``.
    * Yields samples (free dicts — no required schema beyond the dataset's
      own use), formats prompts, extracts predictions, and (optionally)
      knows how to truncate a sample to fit a token budget.

The CLI does a two-stage parse: first ``--dataset NAME``, then the named
dataset's ``add_args`` is called on the same parser before the final
parse. That keeps dataset-specific args out of the top-level CLI surface.

Implementations register themselves via the ``@register_dataset("name")``
decorator. Lookup is by string name. New datasets only need to import
their module once for the side-effect registration.
"""

from __future__ import annotations

import argparse
from typing import Callable

import torch


class Dataset:
    """Base class. Subclasses override the methods below; the ``@register_dataset``
    decorator wires each subclass into ``DATASET_REGISTRY``.

    Subclass methods must NOT depend on module-level state — the CLI may
    instantiate multiple datasets in one process (e.g., for testing).
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
        """Stable identifier — used for resolution lookups and log lines.
        Callers may shorten for display. Falls back to '?'."""
        return "?"

    def lookup_name(self) -> str | None:
        """Name as it appears in ``evals/resolutions.json``'s ``dataset=``
        field (e.g. ``"LongBench-v2:short"``). Return ``None`` to disable
        per-sample resolution lookup for this dataset."""
        return None

    def extract_answer(self, response: str, sample: dict | None = None) -> str | None:
        """Pull the prediction out of free-form model output. Return ``None``
        when nothing matches.

        Datasets whose scoring depends on the gold question (e.g. BABILong's
        question-label exclusion) accept the sample via the optional second
        arg. Datasets that don't need it ignore the arg.
        """
        raise NotImplementedError

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
        from evals.utils import middle_out_truncate_prompt

        prompt = self.render_prompt(sample)
        truncated, did = middle_out_truncate_prompt(prompt, tokenizer, max_len)
        if did:
            sample[self._PROMPT_CACHE_KEY] = truncated
        return did


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
    """Look up a registered dataset class by name. Raises ``SystemExit`` with
    a friendly message listing what IS registered."""
    if name not in DATASET_REGISTRY:
        known = ", ".join(sorted(DATASET_REGISTRY)) or "(none registered)"
        raise SystemExit(
            f"unknown --dataset {name!r}. Known: {known}. "
            f"(Make sure the dataset's module is imported so its "
            f"@register_dataset decorator runs.)"
        )
    return DATASET_REGISTRY[name]
