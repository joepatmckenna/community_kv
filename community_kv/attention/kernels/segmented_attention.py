"""Direct segmented GQA decode attention kernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - CPU-only development hosts.
    triton = None
    tl = None
    libdevice = None

from community_kv.attention.kernels.packed_segments import PackedSegments
from community_kv.graph.state import DecodeDeltaChunks, MutableGraphState, PrefillCSR
from community_kv.graph.kernels.online_update import OnlineUpdateWorkspace

HEAD_DIM = 128
KAPPA = 8
BLOCK_N = 64
MAX_DELTA_CHUNKS = 16


if triton is not None:

    @triton.jit
    def _direct_segmented_partials_kernel(
        query,
        key,
        value,
        member_offsets,
        member_positions,
        delta_positions,
        delta_chunk_start,
        delta_chunk_size,
        delta_chunk_previous,
        delta_community_tail,
        delta_community_sizes,
        descriptor_communities,
        descriptor_cumulative_ends,
        descriptor_counts,
        tile_descriptors,
        sink_positions,
        current_position,
        retrieval_to_graph,
        retrieval_to_kv,
        partial_max,
        partial_sum,
        partial_acc,
        partial_topk_packed,
        sequence_capacity,
        softmax_scale,
        stride_q_h,
        stride_q_d,
        stride_k_h,
        stride_k_s,
        stride_k_d,
        stride_v_h,
        stride_v_s,
        stride_v_d,
        stride_offsets_h,
        stride_offsets_c,
        stride_members_h,
        stride_members_n,
        stride_delta_positions_h,
        stride_delta_positions_n,
        stride_delta_chunks_h,
        stride_delta_chunks_n,
        stride_delta_table_h,
        stride_delta_table_c,
        stride_descriptors_h,
        stride_descriptors_n,
        stride_tiles_h,
        stride_tiles_n,
        stride_partial_h,
        stride_partial_split,
        stride_acc_h,
        stride_acc_split,
        stride_acc_d,
        stride_topk_h,
        stride_topk_split,
        stride_topk_k,
        NUM_SINK: tl.constexpr,
        TOKEN_BUDGET: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        HEAD_DIM_CONST: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_N_CONST: tl.constexpr,
        MAX_DELTA_CHUNKS_CONST: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        retrieval_head = tl.program_id(0)
        graph = tl.load(retrieval_to_graph + retrieval_head).to(tl.int32)
        kv_head = tl.load(retrieval_to_kv + retrieval_head).to(tl.int32)
        split = tl.program_id(1)
        local_token = tl.arange(0, BLOCK_N_CONST)
        logical_token = split * BLOCK_N_CONST + local_token
        is_sink = logical_token < NUM_SINK
        selected_token = logical_token - NUM_SINK
        is_current = logical_token == TOKEN_BUDGET - 1
        is_selected = ~(is_sink | is_current)

        descriptor_count = tl.load(descriptor_counts + retrieval_head)
        descriptor_start = tl.load(
            tile_descriptors
            + retrieval_head * stride_tiles_h
            + split * stride_tiles_n
        )
        boundary_descriptor = descriptor_start + local_token
        boundary_end = tl.load(
            descriptor_cumulative_ends
            + retrieval_head * stride_descriptors_h
            + boundary_descriptor * stride_descriptors_n,
            mask=boundary_descriptor < descriptor_count - 1,
            other=0,
        )
        boundary_lane = boundary_end + NUM_SINK - split * BLOCK_N_CONST
        valid_boundary = (
            (boundary_descriptor < descriptor_count - 1)
            & (boundary_lane > 0)
            & (boundary_lane < BLOCK_N_CONST)
        )
        low_shift = tl.maximum(0, tl.minimum(boundary_lane, 31))
        high_shift = tl.maximum(
            0,
            tl.minimum(boundary_lane - 32, 31),
        )
        one = tl.full((BLOCK_N_CONST,), 1, tl.uint32)
        low_bits = tl.where(
            valid_boundary & (boundary_lane < 32),
            one << low_shift,
            0,
        )
        high_bits = tl.where(
            valid_boundary & (boundary_lane >= 32),
            one << high_shift,
            0,
        )
        low_boundaries = tl.sum(low_bits, axis=0).to(tl.uint32)
        high_boundaries = tl.sum(high_bits, axis=0).to(tl.uint32)

        lane_in_word = local_token & 31
        prefix_mask = tl.full((BLOCK_N_CONST,), 0xFFFFFFFF, tl.uint32) >> (
            31 - lane_in_word
        )
        masked_low = tl.where(
            local_token < 32,
            low_boundaries & prefix_mask,
            low_boundaries,
        )
        masked_high = tl.where(
            local_token < 32,
            0,
            high_boundaries & prefix_mask,
        )
        boundary_count = libdevice.popc(
            masked_low.to(tl.int32, bitcast=True)
        ) + libdevice.popc(masked_high.to(tl.int32, bitcast=True))
        descriptor = descriptor_start + boundary_count
        previous_end = tl.load(
            descriptor_cumulative_ends
            + retrieval_head * stride_descriptors_h
            + (descriptor - 1) * stride_descriptors_n,
            mask=is_selected & (descriptor > 0),
            other=0,
        )
        community = tl.load(
            descriptor_communities
            + retrieval_head * stride_descriptors_h
            + descriptor * stride_descriptors_n,
            mask=is_selected,
            other=0,
        )
        member = selected_token - previous_end
        member_start = tl.load(
            member_offsets + graph * stride_offsets_h + community * stride_offsets_c,
            mask=is_selected,
            other=0,
        )
        delta_count = tl.load(
            delta_community_sizes
            + graph * stride_delta_table_h
            + community * stride_delta_table_c,
            mask=is_selected,
            other=0,
        ).to(tl.int32)
        is_delta = is_selected & (member < delta_count)
        remaining = member
        chunk = tl.load(
            delta_community_tail
            + graph * stride_delta_table_h
            + community * stride_delta_table_c,
            mask=is_delta,
            other=-1,
        ).to(tl.int32)
        delta_start = tl.zeros(
            (BLOCK_N_CONST,),
            dtype=tl.int32,
        )
        delta_slot = tl.zeros(
            (BLOCK_N_CONST,),
            dtype=tl.int32,
        )
        found = ~is_delta
        for _ in tl.static_range(MAX_DELTA_CHUNKS_CONST):
            safe_chunk = tl.maximum(chunk, 0)
            active = ~found & (chunk >= 0)
            chunk_size = tl.load(
                delta_chunk_size
                + graph * stride_delta_chunks_h
                + safe_chunk * stride_delta_chunks_n,
                mask=active,
                other=0,
            ).to(tl.int32)
            in_chunk = active & (remaining < chunk_size)
            chunk_start = tl.load(
                delta_chunk_start
                + graph * stride_delta_chunks_h
                + safe_chunk * stride_delta_chunks_n,
                mask=in_chunk,
                other=0,
            ).to(tl.int32)
            delta_start = tl.where(
                in_chunk,
                chunk_start,
                delta_start,
            )
            delta_slot = tl.where(
                in_chunk,
                chunk_size - 1 - remaining,
                delta_slot,
            )
            found = found | in_chunk
            remaining = tl.where(
                active & ~in_chunk,
                remaining - chunk_size,
                remaining,
            )
            chunk = tl.load(
                delta_chunk_previous
                + graph * stride_delta_chunks_h
                + safe_chunk * stride_delta_chunks_n,
                mask=active & ~in_chunk,
                other=-1,
            ).to(tl.int32)
        delta_position = tl.load(
            delta_positions
            + graph * stride_delta_positions_h
            + (delta_start + delta_slot) * stride_delta_positions_n,
            mask=is_delta & found,
            other=0,
        )
        prefill_member = member - delta_count
        prefill_position = tl.load(
            member_positions
            + graph * stride_members_h
            + (member_start + prefill_member) * stride_members_n,
            mask=is_selected & ~is_delta,
            other=0,
        )
        selected_position = tl.where(
            is_delta,
            delta_position,
            prefill_position,
        )
        sink_position = tl.load(
            sink_positions + logical_token,
            mask=is_sink,
            other=0,
        )
        current = tl.load(current_position + graph)
        token_position = tl.where(
            is_sink,
            sink_position,
            tl.where(is_current, current, selected_position),
        )
        group_lane = tl.arange(0, BLOCK_M)
        dim = tl.arange(0, BLOCK_D)
        group_mask = group_lane < GROUP_SIZE
        dim_mask = dim < HEAD_DIM_CONST
        query_head = retrieval_head * GROUP_SIZE + group_lane
        q = tl.load(
            query + query_head[:, None] * stride_q_h + dim[None, :] * stride_q_d,
            mask=group_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        k = tl.load(
            key
            + kv_head * stride_k_h
            + token_position[None, :] * stride_k_s
            + dim[:, None] * stride_k_d,
            mask=dim_mask[:, None],
            other=0.0,
        )
        score = tl.dot(q, k) * softmax_scale
        score = tl.where(group_mask[:, None], score, -float("inf"))

        representative_score = tl.max(
            tl.where(group_lane[:, None] == 0, score, -float("inf")),
            axis=0,
        )
        candidate_score = tl.where(is_sink, -float("inf"), representative_score)
        score_bits = candidate_score.to(tl.uint32, bitcast=True)
        ordered_score = tl.where(
            (score_bits & 0x80000000) != 0,
            ~score_bits,
            score_bits ^ 0x80000000,
        )
        tie_key = 0xFFFFFFFF - token_position.to(tl.uint32)
        packed = (ordered_score.to(tl.uint64) << 32) | tie_key.to(tl.uint64)
        pair_candidates = tl.reshape(packed, (2, BLOCK_N_CONST // 2))
        pair_head = tl.max(pair_candidates, axis=0)
        pair_tail = tl.min(pair_candidates, axis=0)
        for rank in range(0, TOPK):
            top_packed = tl.max(pair_head, axis=0)
            tl.store(
                partial_topk_packed
                + retrieval_head * stride_topk_h
                + split * stride_topk_split
                + rank * stride_topk_k,
                top_packed,
            )
            winner = pair_head == top_packed
            pair_head = tl.where(winner, pair_tail, pair_head)
            pair_tail = tl.where(winner, 0, pair_tail)

        local_max = tl.max(score, axis=1)
        probability = tl.exp(score - local_max[:, None])
        probability = tl.where(group_mask[:, None], probability, 0.0)
        local_sum = tl.sum(probability, axis=1)
        v = tl.load(
            value
            + kv_head * stride_v_h
            + token_position[:, None] * stride_v_s
            + dim[None, :] * stride_v_d,
            mask=dim_mask[None, :],
            other=0.0,
        )
        local_acc = tl.dot(probability.to(tl.bfloat16), v)

        partial_offset = query_head * stride_partial_h + split * stride_partial_split
        tl.store(
            partial_max + partial_offset,
            local_max,
            mask=group_mask,
        )
        tl.store(
            partial_sum + partial_offset,
            local_sum,
            mask=group_mask,
        )
        tl.store(
            partial_acc
            + query_head[:, None] * stride_acc_h
            + split * stride_acc_split
            + dim[None, :] * stride_acc_d,
            local_acc,
            mask=group_mask[:, None] & dim_mask[None, :],
        )

    @triton.jit
    def _direct_segmented_reduce_update_kernel(
        partial_max,
        partial_sum,
        partial_acc,
        partial_topk_packed,
        output,
        lse_out,
        topk_scores,
        topk_positions,
        key,
        current_position,
        retrieval_to_graph,
        retrieval_to_kv,
        centroids,
        community_sizes,
        community_counts,
        community_weight,
        total_weight,
        token_communities,
        delta_positions,
        chunk_start,
        chunk_capacity,
        chunk_size,
        chunk_next,
        chunk_previous,
        community_head,
        community_tail,
        delta_sizes,
        next_free_chunk,
        next_free_position,
        assigned_out,
        overflow_out,
        lam,
        community_capacity,
        chunks_per_graph,
        delta_position_capacity,
        sequence_capacity,
        stride_partial_h,
        stride_partial_split,
        stride_acc_h,
        stride_acc_split,
        stride_acc_d,
        stride_topk_h,
        stride_topk_split,
        stride_topk_k,
        stride_output_h,
        stride_output_d,
        stride_scores_h,
        stride_scores_k,
        stride_positions_h,
        stride_positions_k,
        stride_key_h,
        stride_key_s,
        stride_key_d,
        stride_centroids_h,
        stride_centroids_c,
        stride_centroids_d,
        stride_sizes_h,
        stride_sizes_c,
        stride_weight_h,
        stride_weight_c,
        stride_token_communities_h,
        stride_token_communities_s,
        stride_delta_positions_h,
        stride_delta_positions_n,
        stride_chunks_h,
        stride_chunks_n,
        stride_table_h,
        stride_table_c,
        NUM_SPLITS: tl.constexpr,
        HEAD_DIM_CONST: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_SPLITS: tl.constexpr,
        BLOCK_CANDIDATES: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        query_head = tl.program_id(0)
        retrieval_head = query_head // GROUP_SIZE
        graph = tl.load(retrieval_to_graph + retrieval_head).to(tl.int32)
        kv_head = tl.load(retrieval_to_kv + retrieval_head).to(tl.int32)
        split = tl.arange(0, BLOCK_SPLITS)
        split_mask = split < NUM_SPLITS
        dim = tl.arange(0, BLOCK_D)
        dim_mask = dim < HEAD_DIM_CONST

        partial_offset = query_head * stride_partial_h + split * stride_partial_split
        split_max = tl.load(
            partial_max + partial_offset,
            mask=split_mask,
            other=-float("inf"),
        )
        global_max = tl.max(split_max, axis=0)
        split_scale = tl.exp(split_max - global_max)
        split_sum = tl.load(
            partial_sum + partial_offset,
            mask=split_mask,
            other=0.0,
        )
        denominator = tl.sum(split_sum * split_scale, axis=0)
        split_acc = tl.load(
            partial_acc
            + query_head * stride_acc_h
            + split[:, None] * stride_acc_split
            + dim[None, :] * stride_acc_d,
            mask=split_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        result = tl.sum(split_acc * split_scale[:, None], axis=0) / denominator
        lse = global_max + tl.log(denominator)
        tl.store(
            output + query_head * stride_output_h + dim * stride_output_d,
            result,
            mask=dim_mask,
        )
        tl.store(lse_out + query_head, lse)

        if (query_head % GROUP_SIZE == 0) & (retrieval_head == graph):
            candidate_slot = tl.arange(0, BLOCK_CANDIDATES)
            candidate_split = candidate_slot // TOPK
            candidate_rank = candidate_slot % TOPK
            candidate_mask = candidate_split < NUM_SPLITS
            packed = tl.load(
                partial_topk_packed
                + retrieval_head * stride_topk_h
                + candidate_split * stride_topk_split
                + candidate_rank * stride_topk_k,
                mask=candidate_mask,
                other=0,
            )
            pair_candidates = tl.reshape(
                packed,
                (2, BLOCK_CANDIDATES // 2),
            )
            pair_head = tl.max(pair_candidates, axis=0)
            pair_tail = tl.min(pair_candidates, axis=0)
            rank_lane = tl.arange(0, BLOCK_K)
            rank_mask = rank_lane < TOPK
            selected_score = tl.full((BLOCK_K,), -float("inf"), dtype=tl.float32)
            selected_position = tl.full((BLOCK_K,), -1, dtype=tl.int32)
            for rank in range(0, TOPK):
                top_packed = tl.max(pair_head, axis=0)
                ordered_score = (top_packed >> 32).to(tl.uint32)
                score_bits = tl.where(
                    (ordered_score & 0x80000000) != 0,
                    ordered_score ^ 0x80000000,
                    ~ordered_score,
                )
                score = score_bits.to(tl.float32, bitcast=True)
                tie_key = (top_packed & 0xFFFFFFFF).to(tl.uint32)
                position = (0xFFFFFFFF - tie_key).to(tl.int32)
                tl.store(
                    topk_scores
                    + retrieval_head * stride_scores_h
                    + rank * stride_scores_k,
                    score,
                )
                tl.store(
                    topk_positions
                    + retrieval_head * stride_positions_h
                    + rank * stride_positions_k,
                    position,
                )
                selected_score = tl.where(
                    rank_lane == rank,
                    score,
                    selected_score,
                )
                selected_position = tl.where(
                    rank_lane == rank,
                    position,
                    selected_position,
                )
                winner = pair_head == top_packed
                pair_head = tl.where(winner, pair_tail, pair_head)
                pair_tail = tl.where(winner, 0, pair_tail)

            valid = (
                rank_mask
                & (selected_position >= 0)
                & (selected_position < sequence_capacity)
            )
            safe_position = tl.minimum(
                tl.maximum(selected_position, 0),
                sequence_capacity - 1,
            )
            community = tl.load(
                token_communities
                + graph * stride_token_communities_h
                + safe_position * stride_token_communities_s,
                mask=valid,
                other=-1,
            ).to(tl.int32)
            score = tl.where(
                valid,
                tl.exp(selected_score - lse),
                0.0,
            )
            direct_weight = score * (lam * 0.5)
            node_degree = tl.sum(direct_weight, axis=0)
            score_sum = tl.sum(score, axis=0)
            pair_weight = (score_sum * score_sum - tl.sum(score * score, axis=0)) * (
                0.5 * (1.0 - lam)
            )

            active_communities = tl.load(
                community_counts + retrieval_head
            ).to(tl.int32)
            candidate = valid & (community >= 0) & (community < active_communities)
            safe_community = tl.maximum(community, 0)
            same_community = safe_community[:, None] == safe_community[None, :]
            candidate_weight = tl.sum(
                tl.where(
                    same_community & candidate[None, :],
                    direct_weight[None, :],
                    0.0,
                ),
                axis=1,
            )
            first_rank = tl.min(
                tl.where(
                    same_community & candidate[None, :],
                    rank_lane[None, :],
                    BLOCK_K,
                ),
                axis=1,
            )
            leader = candidate & (rank_lane == first_rank)

            old_total_weight = tl.load(total_weight + graph).to(tl.float32)
            two_m = tl.maximum(2.0 * old_total_weight, 1.0)
            old_candidate_weight = tl.load(
                community_weight
                + graph * stride_weight_h
                + safe_community * stride_weight_c,
                mask=candidate,
                other=0.0,
            ).to(tl.float32)
            delta_q = candidate_weight - node_degree * old_candidate_weight / two_m
            delta_q = tl.where(
                candidate & (candidate_weight > 0.0),
                delta_q,
                -float("inf"),
            )
            best_delta_q = tl.max(delta_q, axis=0)
            best_community = tl.min(
                tl.where(
                    candidate & (delta_q == best_delta_q),
                    safe_community,
                    0x7FFFFFFF,
                ),
                axis=0,
            )
            join = best_delta_q > 0.0
            assigned = tl.where(join, best_community, active_communities)
            community_overflow = assigned >= community_capacity
            safe_assigned = tl.minimum(
                tl.maximum(assigned, 0),
                community_capacity - 1,
            )

            leader_increment = candidate_weight + tl.where(
                join & (safe_community == safe_assigned),
                node_degree,
                0.0,
            )
            tl.store(
                community_weight
                + graph * stride_weight_h
                + safe_community * stride_weight_c,
                old_candidate_weight + leader_increment,
                mask=leader & ~community_overflow,
            )
            tl.store(
                community_weight
                + graph * stride_weight_h
                + safe_assigned * stride_weight_c,
                node_degree,
                mask=(~join) & ~community_overflow,
            )
            tl.store(
                community_counts + retrieval_head,
                active_communities + (~join).to(tl.int32),
                mask=~community_overflow,
            )
            tl.store(
                total_weight + graph,
                old_total_weight + node_degree + pair_weight,
                mask=~community_overflow,
            )
            tl.store(
                assigned_out + graph,
                tl.where(community_overflow, -1, assigned),
            )

            old_size = tl.load(
                community_sizes
                + retrieval_head * stride_sizes_h
                + safe_assigned * stride_sizes_c,
                mask=~community_overflow,
                other=0,
            ).to(tl.int32)
            new_size = old_size + 1
            tl.store(
                community_sizes
                + retrieval_head * stride_sizes_h
                + safe_assigned * stride_sizes_c,
                new_size,
                mask=~community_overflow,
            )
            centroid_pointer = (
                centroids
                + retrieval_head * stride_centroids_h
                + safe_assigned * stride_centroids_c
                + dim * stride_centroids_d
            )
            old_centroid = tl.load(
                centroid_pointer,
                mask=dim_mask & ~community_overflow,
                other=0.0,
            ).to(tl.float32)
            current = tl.load(current_position + graph).to(tl.int32)
            valid_position = (current >= 0) & (current < sequence_capacity)
            tl.store(
                token_communities
                + graph * stride_token_communities_h
                + current * stride_token_communities_s,
                assigned,
                mask=valid_position & ~community_overflow,
            )
            key_value = tl.load(
                key
                + kv_head * stride_key_h
                + current * stride_key_s
                + dim * stride_key_d,
                mask=dim_mask & valid_position & ~community_overflow,
                other=0.0,
            ).to(tl.float32)
            updated_centroid = tl.where(
                old_size > 0,
                (old_centroid * old_size.to(tl.float32) + key_value)
                / new_size.to(tl.float32),
                key_value,
            )
            tl.store(
                centroid_pointer,
                updated_centroid,
                mask=dim_mask & valid_position & ~community_overflow,
            )

            delta_size_pointer = (
                delta_sizes + graph * stride_table_h + safe_assigned * stride_table_c
            )
            delta_size = tl.load(
                delta_size_pointer,
                mask=~community_overflow,
                other=0,
            ).to(tl.int32)
            old_tail = tl.load(
                community_tail
                + graph * stride_table_h
                + safe_assigned * stride_table_c,
                mask=~community_overflow,
                other=-1,
            ).to(tl.int32)
            safe_old_tail = tl.maximum(old_tail, 0)
            old_chunk_size = tl.load(
                chunk_size
                + graph * stride_chunks_h
                + safe_old_tail * stride_chunks_n,
                mask=(old_tail >= 0) & ~community_overflow,
                other=0,
            ).to(tl.int32)
            old_chunk_capacity = tl.load(
                chunk_capacity
                + graph * stride_chunks_h
                + safe_old_tail * stride_chunks_n,
                mask=(old_tail >= 0) & ~community_overflow,
                other=0,
            ).to(tl.int32)
            allocate_chunk = (
                (old_tail < 0) | (old_chunk_size >= old_chunk_capacity)
            ) & ~community_overflow
            free_chunk = tl.load(next_free_chunk + graph).to(tl.int32)
            free_position = tl.load(next_free_position + graph).to(tl.int32)
            new_capacity = tl.where(
                old_tail < 0,
                1,
                old_chunk_capacity * 2,
            )
            chunk_overflow = allocate_chunk & (free_chunk >= chunks_per_graph)
            position_overflow = allocate_chunk & (
                free_position + new_capacity > delta_position_capacity
            )
            allocation_overflow = chunk_overflow | position_overflow
            chunk = tl.where(allocate_chunk, free_chunk, old_tail)
            safe_chunk = tl.maximum(chunk, 0)
            valid_chunk = (
                ~community_overflow
                & ~allocation_overflow
                & (chunk >= 0)
                & (chunk < chunks_per_graph)
            )

            tl.store(
                next_free_chunk + graph,
                free_chunk + 1,
                mask=allocate_chunk & ~allocation_overflow,
            )
            tl.store(
                next_free_position + graph,
                free_position + new_capacity,
                mask=allocate_chunk & ~allocation_overflow,
            )
            tl.store(
                chunk_start + graph * stride_chunks_h + free_chunk * stride_chunks_n,
                free_position,
                mask=allocate_chunk & ~allocation_overflow,
            )
            tl.store(
                chunk_capacity
                + graph * stride_chunks_h
                + free_chunk * stride_chunks_n,
                new_capacity,
                mask=allocate_chunk & ~allocation_overflow,
            )
            tl.store(
                chunk_previous
                + graph * stride_chunks_h
                + free_chunk * stride_chunks_n,
                old_tail,
                mask=allocate_chunk & ~allocation_overflow,
            )
            tl.store(
                community_head
                + graph * stride_table_h
                + safe_assigned * stride_table_c,
                free_chunk,
                mask=(allocate_chunk & (old_tail < 0) & ~allocation_overflow),
            )
            tl.store(
                chunk_next
                + graph * stride_chunks_h
                + safe_old_tail * stride_chunks_n,
                free_chunk,
                mask=(allocate_chunk & (old_tail >= 0) & ~allocation_overflow),
            )
            tl.store(
                community_tail
                + graph * stride_table_h
                + safe_assigned * stride_table_c,
                free_chunk,
                mask=allocate_chunk & ~allocation_overflow,
            )
            existing_start = tl.load(
                chunk_start
                + graph * stride_chunks_h
                + safe_old_tail * stride_chunks_n,
                mask=(~allocate_chunk) & valid_chunk,
                other=0,
            ).to(tl.int32)
            target_start = tl.where(
                allocate_chunk,
                free_position,
                existing_start,
            )
            target_slot = tl.where(allocate_chunk, 0, old_chunk_size)
            tl.store(
                delta_positions
                + graph * stride_delta_positions_h
                + (target_start + target_slot) * stride_delta_positions_n,
                current,
                mask=valid_chunk & valid_position,
            )
            tl.store(
                chunk_size + graph * stride_chunks_h + safe_chunk * stride_chunks_n,
                target_slot + 1,
                mask=valid_chunk & valid_position,
            )
            tl.store(
                delta_size_pointer,
                delta_size + 1,
                mask=valid_chunk & valid_position,
            )
            overflow = (
                community_overflow.to(tl.int32)
                | (chunk_overflow.to(tl.int32) << 1)
                | (position_overflow.to(tl.int32) << 2)
                | ((~valid_position).to(tl.int32) << 3)
            )
            tl.store(overflow_out + graph, overflow)
            tl.store(
                current_position + graph,
                current + 1,
                mask=overflow == 0,
            )

    @triton.jit
    def _sync_shared_graph_centroids_kernel(
        centroids,
        community_sizes,
        community_counts,
        assigned,
        overflow,
        key,
        current_position,
        retrieval_to_graph,
        retrieval_to_kv,
        sequence_capacity,
        stride_centroids_h,
        stride_centroids_c,
        stride_centroids_d,
        stride_sizes_h,
        stride_sizes_c,
        stride_key_h,
        stride_key_s,
        stride_key_d,
        HEAD_DIM_CONST: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        retrieval_head = tl.program_id(0)
        graph = tl.load(retrieval_to_graph + retrieval_head).to(tl.int32)
        if retrieval_head != graph:
            dim = tl.arange(0, BLOCK_D)
            dim_mask = dim < HEAD_DIM_CONST
            kv_head = tl.load(retrieval_to_kv + retrieval_head).to(tl.int32)
            community = tl.load(assigned + graph).to(tl.int32)
            graph_overflow = tl.load(overflow + graph).to(tl.int32)
            valid = (graph_overflow == 0) & (community >= 0)
            safe_community = tl.maximum(community, 0)
            current = tl.load(current_position + graph).to(tl.int32) - 1
            valid_position = valid & (current >= 0) & (current < sequence_capacity)
            old_size = tl.load(
                community_sizes
                + retrieval_head * stride_sizes_h
                + safe_community * stride_sizes_c,
                mask=valid,
                other=0,
            ).to(tl.int32)
            new_size = old_size + 1
            tl.store(
                community_sizes
                + retrieval_head * stride_sizes_h
                + safe_community * stride_sizes_c,
                new_size,
                mask=valid,
            )
            centroid_pointer = (
                centroids
                + retrieval_head * stride_centroids_h
                + safe_community * stride_centroids_c
                + dim * stride_centroids_d
            )
            old_centroid = tl.load(
                centroid_pointer,
                mask=valid & dim_mask,
                other=0.0,
            ).to(tl.float32)
            key_value = tl.load(
                key
                + kv_head * stride_key_h
                + current * stride_key_s
                + dim * stride_key_d,
                mask=valid_position & dim_mask,
                other=0.0,
            ).to(tl.float32)
            updated = tl.where(
                old_size > 0,
                (old_centroid * old_size.to(tl.float32) + key_value)
                / new_size.to(tl.float32),
                key_value,
            )
            tl.store(
                centroid_pointer,
                updated,
                mask=valid_position & dim_mask,
            )
            tl.store(
                community_counts + retrieval_head,
                tl.load(community_counts + graph),
                mask=valid,
            )


@dataclass(slots=True)
class TritonSegmentedAttentionWorkspace:
    group_size: int
    partial_max: torch.Tensor
    partial_sum: torch.Tensor
    partial_acc: torch.Tensor
    partial_topk_packed: torch.Tensor
    output: torch.Tensor
    lse: torch.Tensor
    topk_scores: torch.Tensor
    topk_positions: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        kv_heads: int,
        group_size: int = 4,
        token_budget: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> "TritonSegmentedAttentionWorkspace":
        if triton is None:
            raise RuntimeError("Triton is unavailable")
        if kv_heads <= 0:
            raise ValueError("kv_heads must be positive")
        if not 1 <= group_size <= 16:
            raise ValueError("group_size must be in [1, 16]")
        if token_budget <= 0 or token_budget % BLOCK_N:
            raise ValueError("token_budget must be a positive multiple of 64")
        if dtype != torch.bfloat16:
            raise TypeError("the Triton specialization requires BF16 Q/K/V")
        query_heads = kv_heads * group_size
        splits = token_budget // BLOCK_N
        return cls(
            group_size=group_size,
            partial_max=torch.empty(
                (query_heads, splits), dtype=torch.float32, device=device
            ),
            partial_sum=torch.empty(
                (query_heads, splits), dtype=torch.float32, device=device
            ),
            partial_acc=torch.empty(
                (query_heads, splits, HEAD_DIM),
                dtype=torch.float32,
                device=device,
            ),
            partial_topk_packed=torch.empty(
                (kv_heads, splits, KAPPA),
                dtype=torch.uint64,
                device=device,
            ),
            output=torch.empty((query_heads, HEAD_DIM), dtype=dtype, device=device),
            lse=torch.empty((query_heads,), dtype=torch.float32, device=device),
            topk_scores=torch.empty(
                (kv_heads, KAPPA), dtype=torch.float32, device=device
            ),
            topk_positions=torch.empty(
                (kv_heads, KAPPA), dtype=torch.int32, device=device
            ),
        )


def triton_segmented_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    csr: PrefillCSR,
    deltas: DecodeDeltaChunks,
    segments: PackedSegments,
    sink_positions: torch.Tensor,
    current_position: torch.Tensor,
    softmax_scale: float,
    workspace: TritonSegmentedAttentionWorkspace,
    update_state: MutableGraphState,
    update_workspace: OnlineUpdateWorkspace,
    update_lam: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run direct CSR attention without a flat retrieval-position tensor."""

    if triton is None:
        raise RuntimeError("Triton is unavailable")
    retrieval_head_count = workspace.partial_topk_packed.shape[0]
    if query.shape != (retrieval_head_count * workspace.group_size, HEAD_DIM):
        raise ValueError(
            f"query must have {workspace.group_size} [128] heads per retrieval head"
        )
    if key.ndim != 3 or value.shape != key.shape or key.shape[-1] != HEAD_DIM:
        raise ValueError("key and value must have shape [Hkv, S, 128]")
    if query.dtype != torch.bfloat16 or key.dtype != query.dtype:
        raise TypeError("query, key, and value must use BF16")
    if segments.token_budget % BLOCK_N:
        raise ValueError("token budget must be a multiple of 64")

    kv_heads = key.shape[0]
    update_state.validate()
    graph_count = update_state.graph_count
    if current_position.dtype != torch.int32 or current_position.shape != (
        graph_count,
    ):
        raise TypeError("current_position must contain one int32 value per graph")
    if update_state.retrieval_head_count != retrieval_head_count:
        raise ValueError("update state and attention retrieval-head counts must match")
    if int(update_state.retrieval_to_kv.max().item()) >= kv_heads:
        raise ValueError("retrieval mapping exceeds the KV-cache head count")
    if update_state.head_dim != HEAD_DIM:
        raise ValueError("update state must use head_dim=128")
    if update_state.deltas is not deltas:
        raise ValueError("fused update must mutate the attention delta store")
    if not 0.0 <= update_lam <= 1.0:
        raise ValueError("update_lam must be in [0, 1]")
    if update_workspace.assigned_communities.shape != (graph_count,):
        raise ValueError("assigned update workspace has the wrong shape")
    if update_workspace.overflow.shape != (graph_count,):
        raise ValueError("overflow update workspace has the wrong shape")
    if any(
        tensor.device != query.device
        for tensor in (
            update_state.centroids,
            update_workspace.assigned_communities,
            update_workspace.overflow,
        )
    ):
        raise ValueError("fused update tensors must share the query device")
    deltas.validate()
    if deltas.graph_count != graph_count:
        raise ValueError("delta chunks must contain one row per graph")
    if deltas.community_capacity != csr.community_capacity:
        raise ValueError("delta chunks must match CSR community capacity")
    if deltas.positions.device != query.device:
        raise ValueError("delta chunks must share the attention device")
    delta_positions = deltas.positions
    delta_chunk_start = deltas.chunk_start
    delta_chunk_size = deltas.chunk_size
    delta_chunk_previous = deltas.chunk_previous
    delta_community_tail = deltas.community_tail
    delta_community_sizes = deltas.community_sizes
    query_heads = query.shape[0]
    group_size = workspace.group_size
    num_splits = segments.token_budget // BLOCK_N
    block_splits = triton.next_power_of_2(num_splits)
    block_candidates = triton.next_power_of_2(num_splits * KAPPA)
    _direct_segmented_partials_kernel[(retrieval_head_count, num_splits)](
        query,
        key,
        value,
        csr.member_offsets,
        csr.member_positions,
        delta_positions,
        delta_chunk_start,
        delta_chunk_size,
        delta_chunk_previous,
        delta_community_tail,
        delta_community_sizes,
        segments.communities,
        segments.cumulative_ends,
        segments.descriptor_counts,
        segments.tile_descriptors,
        sink_positions,
        current_position,
        update_state.retrieval_to_graph,
        update_state.retrieval_to_kv,
        workspace.partial_max,
        workspace.partial_sum,
        workspace.partial_acc,
        workspace.partial_topk_packed,
        key.shape[1],
        softmax_scale,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        csr.member_offsets.stride(0),
        csr.member_offsets.stride(1),
        csr.member_positions.stride(0),
        csr.member_positions.stride(1),
        delta_positions.stride(0),
        delta_positions.stride(1),
        delta_chunk_start.stride(0),
        delta_chunk_start.stride(1),
        delta_community_tail.stride(0),
        delta_community_tail.stride(1),
        segments.communities.stride(0),
        segments.communities.stride(1),
        segments.tile_descriptors.stride(0),
        segments.tile_descriptors.stride(1),
        workspace.partial_max.stride(0),
        workspace.partial_max.stride(1),
        workspace.partial_acc.stride(0),
        workspace.partial_acc.stride(1),
        workspace.partial_acc.stride(2),
        workspace.partial_topk_packed.stride(0),
        workspace.partial_topk_packed.stride(1),
        workspace.partial_topk_packed.stride(2),
        NUM_SINK=segments.num_sink,
        TOKEN_BUDGET=segments.token_budget,
        NUM_SPLITS=num_splits,
        HEAD_DIM_CONST=HEAD_DIM,
        GROUP_SIZE=group_size,
        TOPK=KAPPA,
        BLOCK_N_CONST=BLOCK_N,
        MAX_DELTA_CHUNKS_CONST=MAX_DELTA_CHUNKS,
        BLOCK_D=HEAD_DIM,
        BLOCK_M=16,
        num_warps=2,
    )
    update_deltas = update_state.deltas
    _direct_segmented_reduce_update_kernel[(query_heads,)](
        workspace.partial_max,
        workspace.partial_sum,
        workspace.partial_acc,
        workspace.partial_topk_packed,
        workspace.output,
        workspace.lse,
        workspace.topk_scores,
        workspace.topk_positions,
        key,
        current_position,
        update_state.retrieval_to_graph,
        update_state.retrieval_to_kv,
        update_state.centroids,
        update_state.community_sizes,
        update_state.community_counts,
        update_state.community_weight,
        update_state.total_weight,
        update_state.token_communities,
        update_deltas.positions,
        update_deltas.chunk_start,
        update_deltas.chunk_capacity,
        update_deltas.chunk_size,
        update_deltas.chunk_next,
        update_deltas.chunk_previous,
        update_deltas.community_head,
        update_deltas.community_tail,
        update_deltas.community_sizes,
        update_deltas.next_free_chunk,
        update_deltas.next_free_position,
        update_workspace.assigned_communities,
        update_workspace.overflow,
        update_lam,
        update_state.community_capacity,
        update_deltas.chunk_capacity_per_graph,
        update_deltas.position_capacity,
        key.shape[1],
        workspace.partial_max.stride(0),
        workspace.partial_max.stride(1),
        workspace.partial_acc.stride(0),
        workspace.partial_acc.stride(1),
        workspace.partial_acc.stride(2),
        workspace.partial_topk_packed.stride(0),
        workspace.partial_topk_packed.stride(1),
        workspace.partial_topk_packed.stride(2),
        workspace.output.stride(0),
        workspace.output.stride(1),
        workspace.topk_scores.stride(0),
        workspace.topk_scores.stride(1),
        workspace.topk_positions.stride(0),
        workspace.topk_positions.stride(1),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        update_state.centroids.stride(0),
        update_state.centroids.stride(1),
        update_state.centroids.stride(2),
        update_state.community_sizes.stride(0),
        update_state.community_sizes.stride(1),
        update_state.community_weight.stride(0),
        update_state.community_weight.stride(1),
        update_state.token_communities.stride(0),
        update_state.token_communities.stride(1),
        update_deltas.positions.stride(0),
        update_deltas.positions.stride(1),
        update_deltas.chunk_start.stride(0),
        update_deltas.chunk_start.stride(1),
        update_deltas.community_head.stride(0),
        update_deltas.community_head.stride(1),
        NUM_SPLITS=num_splits,
        HEAD_DIM_CONST=HEAD_DIM,
        GROUP_SIZE=group_size,
        TOPK=KAPPA,
        BLOCK_SPLITS=block_splits,
        BLOCK_CANDIDATES=block_candidates,
        BLOCK_D=HEAD_DIM,
        BLOCK_K=KAPPA,
        num_warps=2,
    )
    if retrieval_head_count != graph_count:
        _sync_shared_graph_centroids_kernel[(retrieval_head_count,)](
            update_state.centroids,
            update_state.community_sizes,
            update_state.community_counts,
            update_workspace.assigned_communities,
            update_workspace.overflow,
            key,
            current_position,
            update_state.retrieval_to_graph,
            update_state.retrieval_to_kv,
            key.shape[1],
            update_state.centroids.stride(0),
            update_state.centroids.stride(1),
            update_state.centroids.stride(2),
            update_state.community_sizes.stride(0),
            update_state.community_sizes.stride(1),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            HEAD_DIM_CONST=HEAD_DIM,
            BLOCK_D=HEAD_DIM,
            num_warps=4,
        )
    return (
        workspace.output,
        workspace.lse,
        workspace.topk_scores,
        workspace.topk_positions,
    )


__all__ = [
    "TritonSegmentedAttentionWorkspace",
    "triton_segmented_attention",
]
