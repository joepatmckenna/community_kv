"""CommunityKV — community-graph based KV cache compression for long-context LLMs.

Public surface:
    CommunityKVAttention: HF-pluggable attention impl + registration handle.
    GraphRuntime:         per-sample mutable state (caller-constructed).

The eval harness (``EvalRunner``, dataset registry, ``community-kv-eval``
CLI) lives in the sibling ``evals`` package.

Build the Leiden CUDA extension via ``pip install -e .`` from this package's root.
"""

from community_kv.attention import (
    COMMUNITY_KV_ATTN_IMPL,
    CommunityKVAttention,
    attn_forward_topk,
    community_kv_attention_forward,
)
from community_kv.graph import (
    GraphAggregation,
    GraphRuntime,
    LayerGraph,
    LayerLog,
    PartitionRecord,
    PartitionResult,
    partition,
)

__all__ = [
    "COMMUNITY_KV_ATTN_IMPL",
    "CommunityKVAttention",
    "GraphAggregation",
    "GraphRuntime",
    "LayerGraph",
    "LayerLog",
    "PartitionRecord",
    "PartitionResult",
    "attn_forward_topk",
    "community_kv_attention_forward",
    "partition",
]
