"""Supported and generic Hugging Face model integration for evaluations."""

from evals.models.loader import (
    ContextStrategy,
    HF_REVISION,
    MAX_TRUNCATION_CONTEXT_WINDOW,
    MAX_TRUNCATION_ROPE_FACTOR,
    ModelPlan,
    base_model,
    configure_context,
    load_causal_model,
    load_tokenizer,
    prepare_model,
    tokenize_chat_prompt,
)
from evals.models.profiles import (
    ModelGeometry,
    ModelProfile,
    SUPPORTED_PROFILES,
    model_geometry,
    resolve_model,
    supported_model_ids,
)

__all__ = [
    "ContextStrategy",
    "HF_REVISION",
    "MAX_TRUNCATION_CONTEXT_WINDOW",
    "MAX_TRUNCATION_ROPE_FACTOR",
    "ModelGeometry",
    "ModelPlan",
    "ModelProfile",
    "SUPPORTED_PROFILES",
    "base_model",
    "configure_context",
    "load_causal_model",
    "load_tokenizer",
    "model_geometry",
    "prepare_model",
    "resolve_model",
    "supported_model_ids",
    "tokenize_chat_prompt",
]
