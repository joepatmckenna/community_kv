from __future__ import annotations

from types import SimpleNamespace

import pytest

from evals.models.profiles import (
    SUPPORTED_PROFILES,
    model_geometry,
    resolve_model,
    supported_model_ids,
)


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


def test_named_profiles_cover_requested_models() -> None:
    assert supported_model_ids() == (
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "meta-llama/Llama-3.1-8B-Instruct",
    )
    for profile in SUPPORTED_PROFILES:
        geometry = model_geometry(_config(profile), profile)
        assert geometry.head_dim == 128
        assert 1 <= geometry.group_size <= 16
        if geometry.model_type == "qwen3":
            assert profile.native_context_window == 32768


@pytest.mark.parametrize(
    ("alias", "model_id"),
    (
        ("qwen3-4b", "Qwen/Qwen3-4B"),
        ("qwen3-8b", "Qwen/Qwen3-8B"),
        ("qwen3-14b", "Qwen/Qwen3-14B"),
        ("llama3.1-8b-instruct", "meta-llama/Llama-3.1-8B-Instruct"),
    ),
)
def test_model_aliases_resolve_to_canonical_ids(alias: str, model_id: str) -> None:
    assert resolve_model(alias).model_id == model_id


def test_unknown_model_uses_generic_geometry_validation() -> None:
    profile = resolve_model("organization/compatible-model")
    assert not profile.is_explicitly_supported
    geometry = model_geometry(
        SimpleNamespace(
            is_encoder_decoder=False,
            model_type="custom",
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            hidden_size=2048,
            max_position_embeddings=65536,
        ),
        profile,
    )
    assert geometry.group_size == 4


def test_generic_model_rejects_incompatible_head_dimension() -> None:
    profile = resolve_model("organization/incompatible-model")
    with pytest.raises(ValueError, match="head dimension 128"):
        model_geometry(
            SimpleNamespace(
                is_encoder_decoder=False,
                model_type="custom",
                num_hidden_layers=24,
                num_attention_heads=16,
                num_key_value_heads=4,
                hidden_size=4096,
                max_position_embeddings=65536,
            ),
            profile,
        )
