"""Model profiles and architecture compatibility checks for evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelGeometry:
    model_type: str
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    head_dim: int
    max_position_embeddings: int

    @property
    def group_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    model_id: str
    context_extension: str
    disable_thinking: bool
    native_context_window: int | None = None
    expected_geometry: ModelGeometry | None = None

    @property
    def is_explicitly_supported(self) -> bool:
        return self.expected_geometry is not None


_QWEN3_4B = ModelProfile(
    name="qwen3-4b",
    model_id="Qwen/Qwen3-4B",
    context_extension="yarn",
    disable_thinking=True,
    native_context_window=32768,
    expected_geometry=ModelGeometry(
        model_type="qwen3",
        num_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=2560,
        head_dim=128,
        max_position_embeddings=40960,
    ),
)

_QWEN3_8B = ModelProfile(
    name="qwen3-8b",
    model_id="Qwen/Qwen3-8B",
    context_extension="yarn",
    disable_thinking=True,
    native_context_window=32768,
    expected_geometry=ModelGeometry(
        model_type="qwen3",
        num_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
        head_dim=128,
        max_position_embeddings=40960,
    ),
)

_QWEN3_14B = ModelProfile(
    name="qwen3-14b",
    model_id="Qwen/Qwen3-14B",
    context_extension="yarn",
    disable_thinking=True,
    native_context_window=32768,
    expected_geometry=ModelGeometry(
        model_type="qwen3",
        num_layers=40,
        num_attention_heads=40,
        num_key_value_heads=8,
        hidden_size=5120,
        head_dim=128,
        max_position_embeddings=40960,
    ),
)

_LLAMA31_8B_INSTRUCT = ModelProfile(
    name="llama3.1-8b-instruct",
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    context_extension="native",
    disable_thinking=False,
    native_context_window=131072,
    expected_geometry=ModelGeometry(
        model_type="llama",
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
        head_dim=128,
        max_position_embeddings=131072,
    ),
)

SUPPORTED_PROFILES = (
    _QWEN3_4B,
    _QWEN3_8B,
    _QWEN3_14B,
    _LLAMA31_8B_INSTRUCT,
)

_BY_MODEL_ID = {profile.model_id.casefold(): profile for profile in SUPPORTED_PROFILES}
_ALIASES = {
    "qwen3-4b": _QWEN3_4B,
    "qwen3-8b": _QWEN3_8B,
    "qwen3-14b": _QWEN3_14B,
    "llama3.1-8b-instruct": _LLAMA31_8B_INSTRUCT,
    "llama-3.1-8b-instruct": _LLAMA31_8B_INSTRUCT,
}


def resolve_model(model: str | ModelProfile) -> ModelProfile:
    """Resolve a known alias or create a generic Hugging Face model profile."""

    if isinstance(model, ModelProfile):
        return model
    model_id = model.strip()
    if not model_id:
        raise ValueError("model identifier must be non-empty")
    normalized = model_id.casefold()
    known = _ALIASES.get(normalized) or _BY_MODEL_ID.get(normalized)
    if known is not None:
        return known
    return ModelProfile(
        name="generic",
        model_id=model_id,
        context_extension="auto",
        disable_thinking=False,
    )


def _positive_config_int(config: Any, name: str) -> int:
    value = getattr(config, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"model config {name} must be a positive integer")
    return value


def model_geometry(config: Any, profile: ModelProfile) -> ModelGeometry:
    """Validate and return the architecture fields CommunityKV depends on."""

    if bool(getattr(config, "is_encoder_decoder", False)):
        raise ValueError("CommunityKV evaluations require a decoder-only model")
    model_type = getattr(config, "model_type", None)
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("model config must define model_type")
    num_layers = _positive_config_int(config, "num_hidden_layers")
    num_attention_heads = _positive_config_int(config, "num_attention_heads")
    num_key_value_heads = _positive_config_int(config, "num_key_value_heads")
    hidden_size = _positive_config_int(config, "hidden_size")
    max_position_embeddings = _positive_config_int(
        config,
        "max_position_embeddings",
    )
    configured_head_dim = getattr(config, "head_dim", None)
    if configured_head_dim is None:
        if hidden_size % num_attention_heads:
            raise ValueError(
                "model must define head_dim when hidden size does not divide heads"
            )
        head_dim = hidden_size // num_attention_heads
    else:
        head_dim = configured_head_dim
    if not isinstance(head_dim, int) or isinstance(head_dim, bool):
        raise ValueError("model config head_dim must be an integer")
    if head_dim != 128:
        raise ValueError("CommunityKV currently requires attention head dimension 128")
    if num_attention_heads % num_key_value_heads:
        raise ValueError("query heads must divide evenly into KV-head groups")
    group_size = num_attention_heads // num_key_value_heads
    if not 1 <= group_size <= 16:
        raise ValueError("CommunityKV requires GQA group size in [1, 16]")

    geometry = ModelGeometry(
        model_type=model_type,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        hidden_size=hidden_size,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
    )
    expected = profile.expected_geometry
    if expected is not None and geometry != expected:
        mismatches = [
            f"{field}={getattr(geometry, field)!r} "
            f"(expected {getattr(expected, field)!r})"
            for field in expected.__dataclass_fields__
            if getattr(geometry, field) != getattr(expected, field)
        ]
        raise ValueError(
            f"{profile.model_id} HEAD no longer matches its supported profile: "
            + ", ".join(mismatches)
        )
    return geometry


def supported_model_ids() -> tuple[str, ...]:
    return tuple(profile.model_id for profile in SUPPORTED_PROFILES)


__all__ = [
    "ModelGeometry",
    "ModelProfile",
    "SUPPORTED_PROFILES",
    "model_geometry",
    "resolve_model",
    "supported_model_ids",
]
