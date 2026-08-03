from __future__ import annotations

from dataclasses import dataclass

import torch

from community_kv.graph.state import DecodeDeltaChunks, MutableGraphState, PrefillCSR


@dataclass(frozen=True, slots=True)
class _OnlineUpdateResult:
    assigned_communities: torch.Tensor
    normalized_topk_scores: torch.Tensor


def _materialize_positions(
    *,
    csr: PrefillCSR,
    deltas: DecodeDeltaChunks,
    graph: int,
    descriptors: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    chunks = []
    for community, take_count in descriptors:
        prefill = csr.members(graph, community)
        delta = deltas.members(graph, community)
        members = torch.cat((delta.flip(0), prefill))
        chunks.append(members[:take_count])
    return torch.cat(chunks)


def _online_assign_and_update(
    *,
    state: MutableGraphState,
    topk_scores: torch.Tensor,
    topk_positions: torch.Tensor,
    representative_lse: torch.Tensor,
    current_key: torch.Tensor,
    current_position: int,
    lam: float,
) -> _OnlineUpdateResult:
    state.validate()
    graph_count, kappa = topk_scores.shape
    valid = topk_positions >= 0
    normalized = torch.where(
        valid,
        torch.exp(topk_scores.float() - representative_lse.float().unsqueeze(1)),
        0.0,
    )
    safe_positions = topk_positions.clamp(
        min=0,
        max=state.token_communities.shape[1] - 1,
    ).to(torch.long)
    topk_communities = torch.gather(
        state.token_communities,
        1,
        safe_positions,
    )
    topk_communities = torch.where(valid, topk_communities, -1)
    assigned = torch.empty((graph_count,), dtype=torch.int32)

    for graph in range(graph_count):
        score = normalized[graph]
        direct_weight = score * (lam * 0.5)
        node_degree = direct_weight.sum()
        score_sum = score.sum()
        pair_weight = (
            (score_sum * score_sum - (score * score).sum())
            * (0.5 * (1.0 - lam))
        )

        candidate_weights: dict[int, torch.Tensor] = {}
        for rank in range(kappa):
            community = int(topk_communities[graph, rank])
            if not bool(valid[graph, rank]) or community < 0:
                continue
            candidate_weights[community] = (
                candidate_weights.get(community, torch.tensor(0.0))
                + direct_weight[rank]
            )

        best_community = -1
        best_delta = -float("inf")
        two_m = max(2.0 * float(state.total_weight[graph]), 1.0)
        for community in sorted(candidate_weights):
            delta = float(
                candidate_weights[community]
                - node_degree
                * state.community_weight[graph, community]
                / two_m
            )
            if delta > best_delta:
                best_delta = delta
                best_community = community

        join = best_delta > 0.0
        community = (
            best_community
            if join
            else int(state.community_counts[graph])
        )
        assigned[graph] = community

        for candidate, weight in candidate_weights.items():
            state.community_weight[graph, candidate] += weight
        state.community_weight[graph, community] += node_degree
        state.total_weight[graph] += node_degree + pair_weight
        if not join:
            state.community_counts[graph] += 1

        old_size = int(state.community_sizes[graph, community])
        new_size = old_size + 1
        old_centroid = state.centroids[graph, community].float()
        key = current_key[graph].float()
        updated = (
            (old_centroid * old_size + key) / new_size
            if old_size > 0
            else key
        )
        limit = torch.finfo(torch.float8_e4m3fn).max
        state.centroids[graph, community] = updated.clamp(
            min=-limit,
            max=limit,
        ).to(state.centroids.dtype)
        state.community_sizes[graph, community] = new_size
        state.token_communities[graph, current_position] = community
        state.deltas.append(graph, community, current_position)

    return _OnlineUpdateResult(
        assigned_communities=assigned,
        normalized_topk_scores=normalized,
    )


def _csr() -> PrefillCSR:
    state = PrefillCSR(
        member_offsets=torch.tensor([[0, 2, 3, 6]], dtype=torch.int32),
        member_positions=torch.tensor([[4, 9, 2, 0, 6, 8]], dtype=torch.int32),
        community_counts=torch.tensor([3], dtype=torch.int32),
    )
    state.validate()
    return state


def test_delta_chunks_append_without_cross_community_compaction() -> None:
    chunks = DecodeDeltaChunks.allocate(
        graph_count=1,
        community_capacity=4,
        max_decode_tokens=8,
    )
    for position in (10, 11, 12):
        chunks.append(0, 1, position)
    chunks.append(0, 2, 13)
    assert chunks.members(0, 1).tolist() == [10, 11, 12]
    assert chunks.members(0, 2).tolist() == [13]
    assert chunks.chunk_capacity[0, :3].tolist() == [1, 2, 1]
    assert int(chunks.next_free_position[0]) == 4


def test_reference_materialization_reads_prefill_then_delta() -> None:
    csr = _csr()
    chunks = DecodeDeltaChunks.allocate(
        graph_count=1,
        community_capacity=4,
        max_decode_tokens=8,
    )
    chunks.append(0, 0, 10)
    chunks.append(0, 0, 11)
    positions = _materialize_positions(
        csr=csr,
        deltas=chunks,
        graph=0,
        descriptors=((0, 4), (2, 1)),
    )
    assert positions.tolist() == [11, 10, 4, 9, 0]


def test_csr_rejects_negative_graph_before_tensor_indexing() -> None:
    csr = _csr()
    try:
        csr.members(-1, 0)
    except IndexError as error:
        assert str(error) == "graph is outside the CSR"
    else:
        raise AssertionError("negative graph index was accepted")


def _mutable_state(
    *,
    graph_count: int = 1,
    community_capacity: int = 8,
    head_dim: int = 4,
    max_decode_tokens: int = 16,
    sequence_capacity: int = 70_000,
) -> MutableGraphState:
    state = MutableGraphState(
        centroids=torch.zeros(
            (graph_count, community_capacity, head_dim),
            dtype=torch.float8_e4m3fn,
        ),
        community_sizes=torch.zeros(
            (graph_count, community_capacity),
            dtype=torch.int32,
        ),
        community_counts=torch.full(
            (graph_count,),
            2,
            dtype=torch.int32,
        ),
        community_weight=torch.zeros(
            (graph_count, community_capacity),
            dtype=torch.float32,
        ),
        total_weight=torch.ones((graph_count,), dtype=torch.float32),
        token_communities=torch.full(
            (graph_count, sequence_capacity),
            -1,
            dtype=torch.int32,
        ),
        deltas=DecodeDeltaChunks.allocate(
            graph_count=graph_count,
            community_capacity=community_capacity,
            max_decode_tokens=max_decode_tokens,
        ),
        retrieval_to_graph=torch.arange(graph_count, dtype=torch.int32),
        retrieval_to_kv=torch.arange(graph_count, dtype=torch.int32),
    )
    state.community_sizes[:, :2] = torch.tensor([2, 1], dtype=torch.int32)
    state.centroids[:, 0] = 1.0
    state.centroids[:, 1] = -1.0
    state.token_communities[:, 10:12] = 0
    state.token_communities[:, 12:13] = 1
    state.token_communities[:, 4:5] = 0
    state.token_communities[:, 5:6] = 1
    state.validate()
    return state


def test_online_update_joins_best_weighted_neighbor_and_coalesces() -> None:
    state = _mutable_state()
    state.community_weight[0, :2] = torch.tensor([0.2, 0.8])
    result = _online_assign_and_update(
        state=state,
        topk_scores=torch.tensor([[2.0, 1.5, 0.5, -1.0]]),
        topk_positions=torch.tensor([[10, 11, 12, 13]], dtype=torch.int32),
        representative_lse=torch.tensor([2.5]),
        current_key=torch.tensor([[3.0, 5.0, 7.0, 9.0]], dtype=torch.bfloat16),
        current_position=20,
        lam=0.5,
    )

    assert result.assigned_communities.tolist() == [0]
    assert state.community_counts.tolist() == [2]
    assert state.community_sizes[0, :2].tolist() == [3, 1]
    assert state.deltas.members(0, 0).tolist() == [20]
    assert torch.equal(
        state.centroids[0, 0].float(),
        torch.tensor([5 / 3, 7 / 3, 3.0, 11 / 3]).to(
            torch.float8_e4m3fn
        ).float(),
    )
    probabilities = result.normalized_topk_scores[0]
    expected_direct_zero = (probabilities[0] + probabilities[1]) * 0.25
    assert torch.isclose(
        state.community_weight[0, 0],
        torch.tensor(0.2) + expected_direct_zero + probabilities.sum() * 0.25,
    )


def test_online_update_creates_singleton_when_modularity_gain_is_not_positive() -> None:
    state = _mutable_state()
    state.community_weight[0, :2] = 100.0
    result = _online_assign_and_update(
        state=state,
        topk_scores=torch.tensor([[1.0, 0.5]]),
        topk_positions=torch.tensor([[4, 5]], dtype=torch.int32),
        representative_lse=torch.tensor([2.0]),
        current_key=torch.tensor([[2.0, 4.0, 6.0, 8.0]], dtype=torch.bfloat16),
        current_position=21,
        lam=1.0,
    )

    assert result.assigned_communities.tolist() == [2]
    assert state.community_counts.tolist() == [3]
    assert state.community_sizes[0, 2].item() == 1
    assert state.centroids[0, 2].tolist() == [2.0, 4.0, 6.0, 8.0]
    assert state.deltas.members(0, 2).tolist() == [21]


def test_online_update_remains_consistent_across_thousands_of_appends() -> None:
    steps = 4096
    state = _mutable_state(
        community_capacity=steps + 2,
        max_decode_tokens=steps,
    )
    state.community_weight[0, :2] = 100.0
    for step in range(steps):
        result = _online_assign_and_update(
            state=state,
            topk_scores=torch.tensor([[1.0]]),
            topk_positions=torch.tensor([[step]], dtype=torch.int32),
            representative_lse=torch.tensor([1.5]),
            current_key=torch.full((1, 4), float(step), dtype=torch.bfloat16),
            current_position=64_000 + step,
            lam=0.5,
        )
        assert int(result.assigned_communities[0]) == step + 2

    assert state.community_counts.tolist() == [steps + 2]
    assert int(state.deltas.next_free_chunk[0]) == steps
    assert int(state.deltas.next_free_position[0]) == steps
    assert int(state.community_sizes[0, 2:].sum()) == steps
    assert state.deltas.members(0, steps + 1).tolist() == [64_000 + steps - 1]
