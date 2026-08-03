"""Generic Hugging Face loading with explicit supported-model contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import torch

from evals.models.profiles import (
    ModelGeometry,
    ModelProfile,
    model_geometry,
    resolve_model,
)


HF_REVISION = "main"
MAX_TRUNCATION_ROPE_FACTOR = 4
MAX_TRUNCATION_CONTEXT_WINDOW = 2**17


class ContextStrategy(str, Enum):
    """How evaluation inputs that exceed native model context are handled."""

    TRUNCATE = "truncate"
    EXTEND = "extend"


@dataclass(slots=True)
class ModelPlan:
    profile: ModelProfile
    geometry: ModelGeometry
    config: Any
    rope_factor: float

    @property
    def truncation_context_window(self) -> int:
        return _truncation_context_window(self.profile, self.geometry)


def _context_mode(profile: ModelProfile, geometry: ModelGeometry) -> str:
    if profile.context_extension != "auto":
        return profile.context_extension
    return "yarn" if geometry.model_type == "qwen3" else "native"


def _native_context_window(
    profile: ModelProfile,
    geometry: ModelGeometry,
) -> int:
    if profile.native_context_window is not None:
        return profile.native_context_window
    if geometry.model_type == "qwen3":
        return 2**15
    return geometry.max_position_embeddings


def _truncation_context_window(
    profile: ModelProfile,
    geometry: ModelGeometry,
) -> int:
    factor = (
        MAX_TRUNCATION_ROPE_FACTOR
        if _context_mode(profile, geometry) == "yarn"
        else 1
    )
    return min(
        _native_context_window(profile, geometry) * factor,
        MAX_TRUNCATION_CONTEXT_WINDOW,
    )


def configure_context(
    config: Any,
    profile: ModelProfile,
    geometry: ModelGeometry,
    *,
    required_tokens: int,
    rope_factor: float | None = None,
) -> float:
    """Configure context extension only when the model has a known policy."""

    if required_tokens <= 0:
        raise ValueError("required_tokens must be positive")
    if rope_factor is not None and (
        not math.isfinite(rope_factor) or rope_factor < 1
    ):
        raise ValueError("rope_factor must be finite and at least 1")

    native_context_window = _native_context_window(profile, geometry)
    factor = rope_factor
    if factor is None:
        factor = 1.0
        while native_context_window * factor < required_tokens:
            factor *= 2.0
    if native_context_window * factor < required_tokens:
        raise ValueError(
            f"{profile.model_id} needs {required_tokens} tokens but "
            f"factor {factor:g} provides only "
            f"{native_context_window * factor:g}"
        )
    if factor == 1:
        return 1.0

    mode = _context_mode(profile, geometry)
    if mode != "yarn":
        raise ValueError(
            f"{profile.model_id} supports {native_context_window} "
            "tokens natively; no tested extension policy is available"
        )
    config.rope_parameters = {
        "rope_type": "yarn",
        "rope_theta": float(getattr(config, "rope_theta", 1_000_000)),
        "factor": float(factor),
        "original_max_position_embeddings": native_context_window,
    }
    return float(factor)


def prepare_model(
    model: str | ModelProfile,
    *,
    required_tokens: int,
    rope_factor: float | None = None,
    trust_remote_code: bool = False,
    context_strategy: ContextStrategy = ContextStrategy.TRUNCATE,
) -> ModelPlan:
    from transformers import AutoConfig

    profile = resolve_model(model)
    config = AutoConfig.from_pretrained(
        profile.model_id,
        revision=HF_REVISION,
        trust_remote_code=trust_remote_code,
    )
    geometry = model_geometry(config, profile)
    if not profile.is_explicitly_supported:
        profile = replace(
            profile,
            context_extension=("yarn" if geometry.model_type == "qwen3" else "native"),
            disable_thinking=(geometry.model_type == "qwen3"),
        )
    context_strategy = ContextStrategy(context_strategy)
    if context_strategy is ContextStrategy.EXTEND:
        factor = configure_context(
            config,
            profile,
            geometry,
            required_tokens=required_tokens,
            rope_factor=rope_factor,
        )
    else:
        if rope_factor is not None:
            raise ValueError("rope_factor requires context_strategy='extend'")
        factor = configure_context(
            config,
            profile,
            geometry,
            required_tokens=min(
                required_tokens,
                _truncation_context_window(profile, geometry),
            ),
        )
    return ModelPlan(
        profile=profile,
        geometry=geometry,
        config=config,
        rope_factor=factor,
    )


def load_tokenizer(
    model: str | ModelProfile,
    *,
    trust_remote_code: bool = False,
):
    from transformers import AutoTokenizer

    profile = resolve_model(model)
    tokenizer = AutoTokenizer.from_pretrained(
        profile.model_id,
        revision=HF_REVISION,
        trust_remote_code=trust_remote_code,
    )
    tokenizer._community_kv_model_profile = profile
    return tokenizer


def load_causal_model(
    plan: ModelPlan,
    *,
    device: torch.device | str,
    trust_remote_code: bool = False,
):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        plan.profile.model_id,
        revision=HF_REVISION,
        config=plan.config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
    ).to(device)
    model.eval()
    return model


def base_model(model: Any) -> Any:
    nested = getattr(model, "model", None)
    return nested if nested is not None else model.base_model


def tokenize_chat_prompt(tokenizer: Any, prompt: str) -> torch.Tensor:
    profile = getattr(tokenizer, "_community_kv_model_profile", None)
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if profile is None or profile.disable_thinking:
        kwargs["enable_thinking"] = False
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        **kwargs,
    )
    return tokenizer(chat, return_tensors="pt").input_ids


__all__ = [
    "ContextStrategy",
    "HF_REVISION",
    "MAX_TRUNCATION_CONTEXT_WINDOW",
    "MAX_TRUNCATION_ROPE_FACTOR",
    "ModelPlan",
    "base_model",
    "configure_context",
    "load_causal_model",
    "load_tokenizer",
    "prepare_model",
    "tokenize_chat_prompt",
]
