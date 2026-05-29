"""HF model loader — registers CommunityKV attention + applies YARN rope scaling."""

from __future__ import annotations

import argparse

import torch

from community_kv.attention import COMMUNITY_KV_ATTN_IMPL


def build_model(
    args: argparse.Namespace,
    *,
    rope_factor: float,
    is_tp: bool,
):
    """Load the HF model with YARN rope scaling and the CommunityKV attention
    impl registered. Caller is responsible for calling
    ``CommunityKVAttention(...).register(...)`` BEFORE this function so the
    impl name is in the HF registry when the model loads.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(args.model)
    native_max = int(config.max_position_embeddings)
    if rope_factor > 1:
        config.rope_parameters = {
            "rope_type": "yarn",
            "rope_theta": 1000000,
            "factor": rope_factor,
            "original_max_position_embeddings": native_max,
        }
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "config": config,
        "attn_implementation": COMMUNITY_KV_ATTN_IMPL,
    }
    kwargs["tp_plan" if is_tp else "device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    return model
