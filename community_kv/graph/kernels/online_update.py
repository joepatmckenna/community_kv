"""Workspace for the fused online graph-state update."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class OnlineUpdateWorkspace:
    assigned_communities: torch.Tensor
    overflow: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        graph_count: int,
        device: torch.device | str,
    ) -> "OnlineUpdateWorkspace":
        if graph_count <= 0:
            raise ValueError("graph_count must be positive")
        return cls(
            assigned_communities=torch.empty(
                (graph_count,),
                dtype=torch.int32,
                device=device,
            ),
            overflow=torch.zeros(
                (graph_count,),
                dtype=torch.int32,
                device=device,
            ),
        )


__all__ = ["OnlineUpdateWorkspace"]
