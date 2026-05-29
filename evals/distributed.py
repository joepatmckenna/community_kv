"""Distributed runtime context.

PP mode: single process, world_size=1, ``DistContext()``.
TP mode: launched under torchrun, ``DistContext.from_env()`` initializes NCCL.

Construct a ``DistContext`` at the program entry point and pass it
through code paths that need rank info.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistContext:
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    is_tp: bool = False

    @classmethod
    def from_env(cls) -> "DistContext":
        ws = int(os.environ.get("WORLD_SIZE", "1"))
        if ws <= 1:
            return cls()
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return cls(world_size=ws, rank=rank, local_rank=local_rank, is_tp=True)

    def print0(self, *args, **kwargs) -> None:
        if self.rank == 0:
            print(*args, **kwargs)

    def barrier(self) -> None:
        if self.is_tp:
            dist.barrier()

    def destroy(self) -> None:
        if self.is_tp:
            dist.destroy_process_group()
