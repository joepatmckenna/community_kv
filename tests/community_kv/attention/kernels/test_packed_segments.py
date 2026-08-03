from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

import torch

from community_kv.attention.kernels.packed_segments import (
    ATTENTION_TILE,
    PackedSegments,
)


@dataclass(frozen=True, slots=True)
class _SelectionDescriptor:
    community_id: int
    take_count: int
    score_key: int


@dataclass(frozen=True, slots=True)
class _CommunitySelection:
    descriptors: tuple[_SelectionDescriptor, ...]
    token_count: int

    @property
    def community_count(self) -> int:
        return len(self.descriptors)


def _selection(
    *descriptors: tuple[int, int],
) -> _CommunitySelection:
    packed = tuple(
        _SelectionDescriptor(
            community_id=community,
            take_count=count,
            score_key=100 - index,
        )
        for index, (community, count) in enumerate(descriptors)
    )
    return _CommunitySelection(
        descriptors=packed,
        token_count=sum(item.take_count for item in packed),
    )


def _pack_segments(
    selections: tuple[_CommunitySelection, ...],
    *,
    token_budget: int,
    num_sink: int,
) -> PackedSegments:
    selected_budget = token_budget - num_sink - 1
    descriptor_capacity = max(1, max(s.community_count for s in selections))
    tile_count = token_budget // ATTENTION_TILE
    communities = torch.full(
        (len(selections), descriptor_capacity),
        -1,
        dtype=torch.int32,
    )
    cumulative_ends = torch.zeros_like(communities)
    descriptor_counts = torch.empty((len(selections),), dtype=torch.int32)
    tile_descriptors = torch.zeros(
        (len(selections), tile_count),
        dtype=torch.int32,
    )

    for head, selection in enumerate(selections):
        if selection.token_count != selected_budget:
            raise ValueError(
                "each selection must fill token_budget - num_sink - 1 tokens"
            )
        descriptor_counts[head] = selection.community_count
        cumulative = 0
        ends: list[int] = []
        for index, descriptor in enumerate(selection.descriptors):
            communities[head, index] = descriptor.community_id
            cumulative += descriptor.take_count
            cumulative_ends[head, index] = cumulative
            ends.append(cumulative)

        for tile in range(tile_count):
            selected_offset = max(0, tile * ATTENTION_TILE - num_sink)
            if selected_offset >= selected_budget:
                tile_descriptors[head, tile] = len(ends) - 1
            else:
                tile_descriptors[head, tile] = bisect_right(
                    ends,
                    selected_offset,
                )

    return PackedSegments(
        communities=communities,
        cumulative_ends=cumulative_ends,
        descriptor_counts=descriptor_counts,
        tile_descriptors=tile_descriptors,
        token_budget=token_budget,
        num_sink=num_sink,
    )


def test_pack_segments_records_only_tile_start_descriptors() -> None:
    # 128 total positions: 10 sinks, 117 selected members, one current token.
    first = _selection((7, 20), (3, 70), (9, 27))
    second = _selection((4, 54), (8, 63))
    packed = _pack_segments(
        (first, second),
        token_budget=128,
        num_sink=10,
    )

    assert packed.communities.tolist() == [[7, 3, 9], [4, 8, -1]]
    assert packed.cumulative_ends.tolist() == [[20, 90, 117], [54, 117, 0]]
    assert packed.descriptor_counts.tolist() == [3, 2]
    # Tile 0 starts in sinks and enters descriptor 0. Tile 1 starts at selected
    # offset 54, which is descriptor 1 for both heads.
    assert packed.tile_descriptors.tolist() == [[0, 1], [0, 1]]
    assert all(tensor.dtype == torch.int32 for tensor in (
        packed.communities,
        packed.cumulative_ends,
        packed.descriptor_counts,
        packed.tile_descriptors,
    ))


def test_pack_segments_requires_exact_budget_fill() -> None:
    selection = _selection((0, 116))
    try:
        _pack_segments(
            (selection,),
            token_budget=128,
            num_sink=10,
        )
    except ValueError as error:
        assert "must fill" in str(error)
    else:
        raise AssertionError("underfilled selection was accepted")


def test_pack_segments_records_singleton_tile_starts() -> None:
    selection = _selection(*((community, 1) for community in range(127)))
    packed = _pack_segments(
        (selection,),
        token_budget=128,
        num_sink=0,
    )

    assert packed.tile_descriptors.tolist() == [[0, 64]]
