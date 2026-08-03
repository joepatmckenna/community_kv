from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

from evals.datasets.dataset import Dataset
from evals.runner import EvalConfig, EvalRunner, atomic_json_dump


ROOT = Path(__file__).parents[2]


def test_runner_uses_only_the_root_community_kv_api() -> None:
    tree = ast.parse((ROOT / "evals/runner.py").read_text())
    community_imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("community_kv")
    ]
    assert community_imports == ["community_kv"]


class _Dataset(Dataset):
    def tokenize(self, tokenizer, sample):
        length = len(self.format_prompt(sample).split())
        return torch.arange(length).view(1, -1)

    def render_prompt(self, sample):
        return sample["prompt"]

    def fit_sample(self, sample, tokenizer, max_len):
        sample[self._PROMPT_CACHE_KEY] = " ".join(
            self.render_prompt(sample).split()[:max_len]
        )
        return True

    def gold(self, sample):
        return "answer"

    def sample_id(self, sample):
        return "sample"

    def extract_answer(self, response, sample=None):
        return response


def _config(tmp_path: Path, **updates) -> EvalConfig:
    values = {
        "model": "Qwen/Qwen3-8B",
        "output": tmp_path / "results.json",
        "resolutions": tmp_path / "resolutions.json",
        "max_new_tokens": 2,
        "context_window": 22,
    }
    values.update(updates)
    return EvalConfig(**values)


def test_eval_config_validates_token_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="multiple of 64"):
        _config(tmp_path, token_budget=1000)
    with pytest.raises(ValueError, match="at least 2"):
        _config(tmp_path, max_new_tokens=1)
    with pytest.raises(ValueError, match="num_sink"):
        _config(tmp_path, num_sink=4095)
    with pytest.raises(ValueError, match="lam"):
        _config(tmp_path, lam=1.1)
    with pytest.raises(ValueError, match="context_strategy"):
        _config(tmp_path, context_strategy="truncate", rope_factor=2.0)


def test_prompt_preparation_uses_dataset_fit_policy(tmp_path: Path) -> None:
    runner = EvalRunner(_config(tmp_path), _Dataset())
    sample = {"prompt": " ".join(f"t{index}" for index in range(25))}
    ids, original_tokens, truncated = runner._prepare_input(None, sample)
    assert original_tokens == 25
    assert ids.shape == (1, 20)
    assert truncated is True


def test_prompt_preparation_accepts_native_context_cap(tmp_path: Path) -> None:
    runner = EvalRunner(
        _config(tmp_path, max_new_tokens=2, context_window=100),
        _Dataset(),
    )
    sample = {"prompt": " ".join(f"t{index}" for index in range(25))}
    ids, original_tokens, truncated = runner._prepare_input(
        None,
        sample,
        context_window=24,
    )
    assert original_tokens == 25
    assert ids.shape == (1, 22)
    assert truncated is True


def test_prompt_preparation_can_disable_middle_out_truncation(tmp_path: Path) -> None:
    runner = EvalRunner(
        _config(tmp_path, max_new_tokens=2, context_window=24),
        _Dataset(),
    )
    sample = {"prompt": " ".join(f"t{index}" for index in range(25))}
    with pytest.raises(ValueError, match="increase --context-window"):
        runner._prepare_input(
            None,
            sample,
            allow_truncation=False,
        )


def test_atomic_json_dump_replaces_complete_document(tmp_path: Path) -> None:
    path = tmp_path / "nested/results.json"
    atomic_json_dump({"status": "running"}, path)
    atomic_json_dump({"status": "complete", "samples": [1]}, path)
    assert json.loads(path.read_text()) == {
        "status": "complete",
        "samples": [1],
    }
    assert not path.with_suffix(".json.tmp").exists()


def test_sample_generation_uses_runtime_greedy_generation(tmp_path: Path) -> None:
    runner = EvalRunner(
        _config(tmp_path, device="cpu", max_new_tokens=3, context_window=24),
        _Dataset(),
    )
    calls = {}

    class Runtime:
        def generate(self, model, **kwargs):
            calls.update(model=model, **kwargs)
            return torch.cat(
                (kwargs["input_ids"], torch.tensor([[7, 8]])),
                dim=1,
            )

    class Tokenizer:
        def decode(
            self,
            token_ids,
            *,
            skip_special_tokens,
            clean_up_tokenization_spaces,
        ):
            assert token_ids == [7, 8]
            assert skip_special_tokens
            assert clean_up_tokenization_spaces is False
            return "answer"

    model = object()
    resolutions = [1.0, 2.0]
    result = runner._run_sample(
        model=model,
        tokenizer=Tokenizer(),
        runtime=Runtime(),
        resolutions=resolutions,
        sample={"prompt": " ".join(["token"] * 18)},
    )

    assert calls["model"] is model
    assert calls["resolutions"] is resolutions
    assert calls["do_sample"] is False
    assert calls["max_new_tokens"] == 3
    assert result["generated_tokens"] == 2
    assert result["response"] == "answer"
    assert result["score"] == 1.0
