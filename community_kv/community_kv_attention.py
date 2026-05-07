"""Transformers-compatible attention interface for CommunityKV.

Registers a custom attention implementation that can be activated via:

    from community_kv.community_kv_attention import register_community_kv_attention, ATTN_NAME
    register_community_kv_attention(kappa=8, sink_size=4)
    model.config._attn_implementation = ATTN_NAME
"""

from __future__ import annotations

import torch
import torch.nn as nn

from community_kv.kernels.attention_topk import attention_with_topk
from community_kv.graph import GraphManager

ATTN_NAME = "community_kv"

# Global config for the attention function (set via register_community_kv_attention)
_CONFIG = {
    "kappa": 8,
    "sink_size": 4,
    "graph_manager": None,  # type: GraphManager | None
    "previous_attn_fn": None,  # type: Callable | None
}


def community_kv_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Custom attention using the CommunityKV fused kernel.

    Uses our kernel for prefill (q_len > 1) and the previous attn impl for decode.
    """
    kappa = _CONFIG["kappa"]
    sink_size = _CONFIG["sink_size"]

    B, H, S_q, D = query.shape
    S_k = key.shape[2]

    manager: GraphManager = _CONFIG["graph_manager"]

    # Use our kernel for prefill, previous impl for decode
    if S_q > 1 and S_k >= kappa + sink_size:
        attn_output, topk_indices, topk_scores = attention_with_topk(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            kappa=kappa,
            sink_size=sink_size,
        )

        # Store graph data and launch async graph construction
        manager.initialize(
            module.layer_idx,
            topk_indices[0],   # (H_q, S_eligible, kappa)
            topk_scores[0],    # (H_q, S_eligible, kappa)
            keys=key[0],       # (H_kv, S, D)
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None
    else:
        # Decode: retrieve indices and gather KV
        retrieved_indices = manager.retrieve(
            module.layer_idx,
            query[0],   # (H_q, 1, D)
            key[0],     # (H_kv, S, D)
            value[0],   # (H_kv, S, D)
        )  # (H_kv, budget)

        H_q = query.shape[1]
        H_kv = key.shape[1]
        D = key.shape[3]
        heads_per_group = H_q // H_kv

        # Gather KV at retrieved positions per KV head
        gather_idx = retrieved_indices.unsqueeze(-1).expand(-1, -1, D)  # (H_kv, budget, D)
        retrieved_keys = key[0].gather(1, gather_idx).unsqueeze(0)      # (1, H_kv, budget, D)
        retrieved_values = value[0].gather(1, gather_idx).unsqueeze(0)  # (1, H_kv, budget, D)

        # Run attention using the previous implementation (handles GQA expansion)
        previous_attn_fn = _CONFIG["previous_attn_fn"]
        attn_output, attn_weights = previous_attn_fn(
            module, query, retrieved_keys, retrieved_values, attention_mask,
            scaling=scaling, dropout=dropout, **kwargs,
        )

        # Extract top-k from attention weights for graph update
        # attn_weights: (B, H_q, 1, budget)
        topk_scores_local, topk_indices_local = attn_weights.squeeze(2).topk(kappa, dim=-1)

        # Remap local indices to global positions
        kv_head_for_q = torch.arange(H_q, device=query.device) // heads_per_group
        retrieved_indices_per_head = retrieved_indices[kv_head_for_q]  # (H_q, budget)
        topk_indices_global = retrieved_indices_per_head.unsqueeze(0).gather(2, topk_indices_local)

        # Update graph with new edges and assign community to new token
        manager.update(
            layer_idx=module.layer_idx,
            topk_indices_global=topk_indices_global[0],  # (H_q, kappa)
            topk_scores=topk_scores_local[0],            # (H_q, kappa)
            attn_weights=attn_weights[0],                # (H_q, 1, budget)
            retrieved_indices=retrieved_indices,         # (H_kv, budget)
            keys=key[0],                                 # (H_kv, S, D)
        )

        return attn_output, None


def register_community_kv_attention(
    kappa: int = 8,
    sink_size: int = 4,
    graph_manager: GraphManager | None = None,
    previous_attn_impl: str = "eager",
):
    """Register the CommunityKV attention implementation with transformers.

    After calling this, set `model.config._attn_implementation = ATTN_NAME`
    to activate it.

    Args:
        kappa: top-k keys per query
        sink_size: number of sink tokens excluded from top-k
        graph_manager: GraphManager for async graph construction.
        previous_attn_impl: the attention implementation to delegate to during
            decode (e.g. "sdpa", "flash_attention_2", "eager").
    """
    _CONFIG["kappa"] = kappa
    _CONFIG["sink_size"] = sink_size
    _CONFIG["graph_manager"] = graph_manager

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if previous_attn_impl == "eager":
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
        _CONFIG["previous_attn_fn"] = eager_attention_forward
    else:
        _CONFIG["previous_attn_fn"] = ALL_ATTENTION_FUNCTIONS.get(previous_attn_impl)
    ALL_ATTENTION_FUNCTIONS[ATTN_NAME] = community_kv_attention_forward
