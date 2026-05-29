"""Dataset-agnostic helpers used by the eval harness."""

from __future__ import annotations

import argparse
import os
import sys

import torch

from evals.distributed import DistContext

# Maximum rope_factor before we either cap (and truncate) or keep extending.
# 4 * 40960 = 163840 ~ 2**17.32 — covers Qwen3's recommended YARN window.
DEFAULT_MAX_ROPE_FACTOR = 4.0


# --------------------------------------------------------------------------- #
# Context-extension primitives
# --------------------------------------------------------------------------- #


def compute_rope_factor(
    n_tokens: int,
    native_max: int,
    *,
    max_rope_factor: float | None = None,
) -> float:
    """Smallest power-of-2 such that ``native_max * factor >= n_tokens``,
    optionally capped at ``max_rope_factor``.

    Floor 1 (we never go below the model's native context). YARN scaling
    requires factor >= 1. When ``max_rope_factor`` is set and the natural
    factor exceeds it, the cap is returned — the caller is responsible for
    truncating inputs that don't fit.
    """
    target = 1
    while target * native_max < n_tokens:
        target *= 2
    factor = float(max(target, 1))
    if max_rope_factor is not None and factor > max_rope_factor:
        return float(max_rope_factor)
    return factor


# --------------------------------------------------------------------------- #
# Sample preparation
# --------------------------------------------------------------------------- #


def tokenize_chat_prompt(tokenizer, prompt: str) -> torch.Tensor:
    """Wrap a user-message ``prompt`` in the tokenizer's chat template
    and return the token ids tensor (shape ``(1, S)``).

    Chat-template kwargs (``add_generation_prompt``, ``enable_thinking``)
    are model-family-specific; the dataset's only contribution is the
    string body via its ``format_prompt(sample)`` method, so callers
    typically write ``tokenize_chat_prompt(tok, dataset.format_prompt(s))``.
    """
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(chat, return_tensors="pt").input_ids


def middle_out_truncate_prompt(
    prompt: str,
    tokenizer,
    max_len: int,
) -> tuple[str, bool]:
    """Middle-out-truncate ``prompt`` so its tokenized length fits in
    ``max_len`` tokens.

    Mirrors the canonical truncation used by both BABILong and LongBench-v2
    (see ``THUDM/LongBench/pred.py:query_llm``): encode the bare prompt,
    slice ``[:max_len//2] + [-max_len//2:]``, decode back to a string.
    Returns ``(truncated_prompt, did_truncate)``. When the prompt fits the
    original string is returned unchanged with ``did_truncate=False``.
    """
    ids = tokenizer.encode(prompt)
    if len(ids) <= max_len:
        return prompt, False
    head = max_len // 2
    tail = max_len - head
    ids = ids[:head] + ids[-tail:]
    return tokenizer.decode(ids, skip_special_tokens=True), True


# --------------------------------------------------------------------------- #
# Distributed bootstrap
# --------------------------------------------------------------------------- #


def setup_distributed(args: argparse.Namespace) -> DistContext:
    """Re-exec under torchrun if running outside it (and ``--pp`` wasn't set),
    otherwise initialize NCCL via ``DistContext.from_env``. Validates that
    the user-requested ``--tp_size`` matches ``WORLD_SIZE``.

    May not return: when re-exec'ing, ``os.execvp`` replaces this process.
    """
    if "WORLD_SIZE" not in os.environ and not args.pp:
        nproc = args.tp_size if args.tp_size is not None else 8
        # Re-exec via torchrun's ``-m`` (not the script path): passing
        # ``sys.argv[0]`` would put the ``evals/`` dir on sys.path[0], which
        # shadows HF's top-level ``datasets`` with our ``evals.datasets``
        # subpackage. ``-m evals.main`` keeps CWD on the path instead.
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={nproc}",
            "-m",
            "evals.main",
        ] + sys.argv[1:]
        if args.tp_size is None:
            cmd.insert(5, f"--tp_size={nproc}")
        os.execvp("torchrun", cmd)

    dist_ctx = DistContext.from_env()
    if args.tp_size is not None and args.tp_size != dist_ctx.world_size:
        raise SystemExit(
            f"--tp_size={args.tp_size} but WORLD_SIZE={dist_ctx.world_size}. "
            f"Launch via: torchrun --nproc_per_node={args.tp_size} ..."
        )
    return dist_ctx
