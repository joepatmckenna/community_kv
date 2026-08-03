from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from evals.models.loader import (
    ContextStrategy,
    base_model,
    configure_context,
    prepare_model,
    tokenize_chat_prompt,
)
from evals.models.profiles import model_geometry, resolve_model


def _config(profile):
    expected = profile.expected_geometry
    assert expected is not None
    return SimpleNamespace(
        is_encoder_decoder=False,
        model_type=expected.model_type,
        num_hidden_layers=expected.num_layers,
        num_attention_heads=expected.num_attention_heads,
        num_key_value_heads=expected.num_key_value_heads,
        hidden_size=expected.hidden_size,
        head_dim=expected.head_dim,
        max_position_embeddings=expected.max_position_embeddings,
    )


def test_qwen_profile_applies_yarn_for_64k() -> None:
    profile = resolve_model("qwen3-8b")
    config = _config(profile)
    geometry = model_geometry(config, profile)

    factor = configure_context(config, profile, geometry, required_tokens=65536)

    assert factor == 2.0
    assert config.rope_parameters == {
        "rope_type": "yarn",
        "rope_theta": 1_000_000.0,
        "factor": 2.0,
        "original_max_position_embeddings": 32768,
    }


def test_llama_profile_rejects_untested_extension() -> None:
    profile = resolve_model("llama3.1-8b-instruct")
    config = _config(profile)
    geometry = model_geometry(config, profile)

    with pytest.raises(ValueError, match="no tested extension policy"):
        configure_context(config, profile, geometry, required_tokens=262144)


def test_truncation_strategy_scales_qwen_up_to_128k_cap(monkeypatch) -> None:
    profile = resolve_model("qwen3-8b")
    config = _config(profile)
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *args, **kwargs: config,
    )

    plan = prepare_model(
        profile,
        required_tokens=131072,
        context_strategy=ContextStrategy.TRUNCATE,
    )

    assert plan.rope_factor == 4.0
    assert plan.truncation_context_window == 2**17
    assert config.rope_parameters["factor"] == 4.0


def test_model_preparation_can_extend_context(monkeypatch) -> None:
    profile = resolve_model("qwen3-8b")
    config = _config(profile)
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *args, **kwargs: config,
    )

    plan = prepare_model(
        profile,
        required_tokens=200000,
        context_strategy=ContextStrategy.EXTEND,
    )

    assert plan.rope_factor == 8.0
    assert config.rope_parameters["factor"] == 8.0


class _ChatTokenizer:
    def __init__(self, profile) -> None:
        self._community_kv_model_profile = profile
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return "rendered"

    def __call__(self, text, *, return_tensors):
        assert text == "rendered"
        assert return_tensors == "pt"
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))


def test_chat_template_options_are_model_specific() -> None:
    qwen = _ChatTokenizer(resolve_model("qwen3-4b"))
    llama = _ChatTokenizer(resolve_model("llama3.1-8b-instruct"))

    tokenize_chat_prompt(qwen, "prompt")
    tokenize_chat_prompt(llama, "prompt")

    assert qwen.template_kwargs["enable_thinking"] is False
    assert "enable_thinking" not in llama.template_kwargs


def test_base_model_accepts_both_hugging_face_wrapping_styles() -> None:
    nested = object()
    fallback = object()
    assert base_model(SimpleNamespace(model=nested, base_model=fallback)) is nested
    assert base_model(SimpleNamespace(model=None, base_model=fallback)) is fallback
