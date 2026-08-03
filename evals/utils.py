"""Prompt tokenization and truncation helpers for evaluation datasets."""

from __future__ import annotations

import torch


def tokenize_chat_prompt(tokenizer, prompt: str) -> torch.Tensor:
    """Apply the model chat template and return token IDs with shape ``(1, S)``."""
    from evals.models import tokenize_chat_prompt as model_tokenize_chat_prompt

    return model_tokenize_chat_prompt(tokenizer, prompt)


def middle_out_truncate_prompt(
    prompt: str,
    tokenizer,
    max_len: int,
) -> tuple[str, bool]:
    """Middle-out truncate a prompt to at most ``max_len`` bare tokens."""
    ids = tokenizer.encode(prompt)
    if len(ids) <= max_len:
        return prompt, False
    head = max_len // 2
    tail = max_len - head
    return (
        tokenizer.decode(
            ids[:head] + ids[-tail:],
            skip_special_tokens=True,
        ),
        True,
    )
