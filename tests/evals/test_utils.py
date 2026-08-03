from __future__ import annotations

from types import SimpleNamespace

import torch

from evals import models, utils


class _Tokenizer:
    def encode(self, prompt: str) -> list[int]:
        return [int(value) for value in prompt.split()]

    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return " ".join(str(value) for value in ids)


def test_middle_out_truncation_preserves_head_and_tail() -> None:
    prompt = "0 1 2 3 4 5 6"

    truncated, changed = utils.middle_out_truncate_prompt(prompt, _Tokenizer(), 5)

    assert changed
    assert truncated == "0 1 4 5 6"


def test_middle_out_truncation_leaves_fitting_prompt_unchanged() -> None:
    prompt = "0 1 2"
    assert utils.middle_out_truncate_prompt(prompt, _Tokenizer(), 3) == (
        prompt,
        False,
    )


def test_chat_tokenization_delegates_to_model_policy(monkeypatch) -> None:
    expected = torch.tensor([[1, 2]])
    calls = {}

    def tokenize(tokenizer, prompt):
        calls["args"] = tokenizer, prompt
        return expected

    monkeypatch.setattr(models, "tokenize_chat_prompt", tokenize)
    tokenizer = SimpleNamespace()

    assert utils.tokenize_chat_prompt(tokenizer, "prompt") is expected
    assert calls["args"] == (tokenizer, "prompt")
