"""CommunityKV graph package.

Re-exports the public API so existing `from community_kv.graph import ...`
imports continue to work after the split into `utils` and `manager`.
"""

from community_kv.graph.utils import GraphAggregation, build_adjacency
from community_kv.graph.manager import GraphManager

__all__ = ["GraphAggregation", "build_adjacency", "GraphManager"]
