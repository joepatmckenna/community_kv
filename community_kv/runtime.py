"""Standalone fused-prefill, partition, and optimized-decode runtime."""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Any, Mapping, Sequence

import torch

from community_kv.config import CommunityKVConfig, GraphAggregation, PartitionConfig
from community_kv.attention.decode import DecodeLayerWorkspace, decode_layer
from community_kv.graph.partition import PartitionedLayer
from community_kv.graph.runtime import PartitionRuntime
from community_kv.graph.state import DecodeDeltaChunks, MutableGraphState, PrefillCSR


HEAD_DIM = 128
ATTENTION_TILE = 64


def _effective_token_budget(
    *,
    configured_budget: int,
    prefill_seq_len: int,
    num_sink: int,
) -> int:
    available = ((prefill_seq_len + 1) // ATTENTION_TILE) * ATTENTION_TILE
    effective = min(configured_budget, available)
    if effective <= num_sink + 1:
        raise ValueError(
            "prefill sequence is too short for the fixed 64-token decode kernel"
        )
    return effective


@dataclass(slots=True)
class ConvertedPrefillState:
    """Validated state built from one layer's prefill partition."""

    csr: PrefillCSR
    mutable: MutableGraphState
    current_position: torch.Tensor
    initial_community_counts: tuple[int, ...]
    prefill_seq_len: int
    decode_capacity: int
    group_size: int
    aggregation: GraphAggregation
    query_heads: int
    kv_heads: int


@dataclass(slots=True)
class DecodeLayerState:
    """All fixed-address state used by one captured transformer layer."""

    converted: ConvertedPrefillState
    workspace: DecodeLayerWorkspace
    sink_positions: torch.Tensor


def _pad_member_offsets(
    member_offsets: torch.Tensor,
    *,
    community_capacity: int,
) -> torch.Tensor:
    if member_offsets.dtype != torch.int32 or member_offsets.ndim != 2:
        raise TypeError("member_offsets must be a rank-two int32 tensor")
    required_width = community_capacity + 1
    if member_offsets.shape[1] > required_width:
        raise ValueError("member offsets exceed centroid community capacity")
    if member_offsets.shape[1] == required_width:
        return member_offsets.contiguous()
    padded = torch.empty(
        (member_offsets.shape[0], required_width),
        dtype=torch.int32,
        device=member_offsets.device,
    )
    width = member_offsets.shape[1]
    padded[:, :width].copy_(member_offsets)
    padded[:, width:].copy_(member_offsets[:, -1:])
    return padded


def convert_prefill_tensors(
    *,
    member_offsets: torch.Tensor,
    member_positions: torch.Tensor,
    num_communities: torch.Tensor,
    centroids: torch.Tensor,
    community_sizes: torch.Tensor,
    community_weight: torch.Tensor,
    total_weight: torch.Tensor,
    token_communities: torch.Tensor,
    retrieval_to_graph: torch.Tensor | None = None,
    retrieval_to_kv: torch.Tensor | None = None,
    prefill_seq_len: int,
    max_decode_tokens: int,
    group_size: int = 4,
    aggregation: GraphAggregation = GraphAggregation.PER_QUERY_HEAD,
    query_heads: int | None = None,
    kv_heads: int | None = None,
) -> ConvertedPrefillState:
    """Validate explicit tensors and allocate decode-delta pages."""

    if centroids.ndim != 3 or centroids.shape[-1] != HEAD_DIM:
        raise ValueError("centroids must have shape [retrieval heads, communities, 128]")
    if centroids.dtype != torch.float8_e4m3fn:
        raise TypeError("centroids must use FP8 E4M3")
    retrieval_head_count, community_capacity, _ = centroids.shape
    graph_count = member_offsets.shape[0]
    if min(retrieval_head_count, graph_count, community_capacity) <= 0:
        raise ValueError("prefill state must contain active graphs and centroids")
    if member_positions.dtype != torch.int32 or member_positions.ndim != 2:
        raise TypeError("member_positions must be a rank-two int32 tensor")
    if member_positions.shape[0] != graph_count:
        raise ValueError("member positions must contain one row per graph")
    if num_communities.dtype != torch.int32:
        raise TypeError("num_communities must use int32")
    if num_communities.shape != (graph_count,):
        raise ValueError("num_communities must contain one value per graph")
    if community_sizes.shape != centroids.shape[:2]:
        raise ValueError("community_sizes must match centroids")
    if community_sizes.dtype != torch.int32:
        raise TypeError("community_sizes must use int32")
    if community_weight.shape != (graph_count, community_capacity):
        raise ValueError("community_weight must have one row per graph")
    if community_weight.dtype != torch.float32:
        raise TypeError("community_weight must use float32")
    if total_weight.shape != (graph_count,) or total_weight.dtype != torch.float32:
        raise TypeError("total_weight must be one float32 value per graph")
    if token_communities.dtype != torch.int32 or token_communities.ndim != 2:
        raise TypeError("token_communities must be a rank-two int32 tensor")
    if token_communities.shape[0] != graph_count:
        raise ValueError("token_communities must contain one row per graph")
    if prefill_seq_len <= 0 or max_decode_tokens <= 0:
        raise ValueError("prefill and decode capacities must be positive")
    if not 1 <= group_size <= 16:
        raise ValueError("group_size must be in [1, 16]")
    aggregation = GraphAggregation(aggregation)
    if retrieval_to_graph is None or retrieval_to_kv is None:
        if retrieval_head_count != graph_count:
            raise ValueError("aggregation mappings are required for non-identity state")
        retrieval_to_graph = torch.arange(
            retrieval_head_count,
            dtype=torch.int32,
            device=centroids.device,
        )
        retrieval_to_kv = retrieval_to_graph
    if (
        retrieval_to_graph.dtype != torch.int32
        or retrieval_to_graph.shape != (retrieval_head_count,)
        or retrieval_to_kv.dtype != torch.int32
        or retrieval_to_kv.shape != (retrieval_head_count,)
    ):
        raise TypeError("aggregation mappings must be int32 retrieval-head vectors")
    inferred_query_heads = retrieval_head_count * group_size
    query_heads = inferred_query_heads if query_heads is None else query_heads
    kv_heads = (
        int(retrieval_to_kv.max().item()) + 1
        if kv_heads is None
        else kv_heads
    )
    if query_heads != inferred_query_heads:
        raise ValueError("query-head count does not match retrieval-head geometry")
    if kv_heads <= 0 or int(retrieval_to_kv.max().item()) >= kv_heads:
        raise ValueError("KV-head count does not cover retrieval mappings")
    if token_communities.shape[1] < prefill_seq_len + max_decode_tokens:
        raise ValueError("token_communities lacks requested decode headroom")
    tensors = (
        member_offsets,
        member_positions,
        num_communities,
        community_sizes,
        community_weight,
        total_weight,
        token_communities,
        retrieval_to_graph,
        retrieval_to_kv,
    )
    if any(value.device != centroids.device for value in tensors):
        raise ValueError("all converted tensors must share one device")

    retrieval_head_counts = num_communities[retrieval_to_graph.long()].contiguous()
    initial_counts = tuple(
        int(value)
        for value in retrieval_head_counts.to(device="cpu", dtype=torch.int64).tolist()
    )
    if any(count <= 0 or count > community_capacity for count in initial_counts):
        raise ValueError("active community counts must fit centroid capacity")
    csr = PrefillCSR(
        member_offsets=_pad_member_offsets(
            member_offsets,
            community_capacity=community_capacity,
        ),
        member_positions=member_positions.contiguous(),
        community_counts=num_communities.clone(),
    )
    mutable = MutableGraphState(
        centroids=centroids.contiguous(),
        community_sizes=community_sizes.contiguous(),
        community_counts=retrieval_head_counts.clone(),
        community_weight=community_weight.contiguous(),
        total_weight=total_weight.contiguous(),
        token_communities=token_communities.contiguous(),
        deltas=DecodeDeltaChunks.allocate(
            graph_count=graph_count,
            community_capacity=community_capacity,
            max_decode_tokens=max_decode_tokens,
            device=centroids.device,
        ),
        retrieval_to_graph=retrieval_to_graph.contiguous(),
        retrieval_to_kv=retrieval_to_kv.contiguous(),
    )
    current_position = torch.full(
        (graph_count,),
        prefill_seq_len,
        dtype=torch.int32,
        device=centroids.device,
    )
    csr.validate()
    mutable.validate()
    return ConvertedPrefillState(
        csr=csr,
        mutable=mutable,
        current_position=current_position,
        initial_community_counts=initial_counts,
        prefill_seq_len=prefill_seq_len,
        decode_capacity=max_decode_tokens,
        group_size=group_size,
        aggregation=aggregation,
        query_heads=query_heads,
        kv_heads=kv_heads,
    )


class DecodeRuntime:
    """Fixed-address collection of optimized layer states."""

    def __init__(self, config: CommunityKVConfig) -> None:
        self.config = config
        self.layers: dict[int, DecodeLayerState] = {}

    def reset(self) -> None:
        self.layers.clear()

    def prepare_partitioned_layer(
        self,
        state: PartitionedLayer,
    ) -> DecodeLayerState:
        return self.prepare_layer(
            state.layer_idx,
            member_offsets=state.member_offsets,
            member_positions=state.member_positions,
            num_communities=state.community_counts,
            centroids=state.centroids,
            community_sizes=state.community_sizes,
            community_weight=state.community_weight,
            total_weight=state.total_weight,
            token_communities=state.token_communities,
            retrieval_to_graph=state.retrieval_to_graph,
            retrieval_to_kv=state.retrieval_to_kv,
            prefill_seq_len=state.prefill_seq_len,
            max_decode_tokens=(
                state.token_communities.shape[1] - state.prefill_seq_len
            ),
            group_size=state.query_heads // state.retrieval_to_graph.numel(),
            aggregation=state.aggregation,
            query_heads=state.query_heads,
            kv_heads=state.kv_heads,
        )

    def prepare_layer(
        self,
        layer_idx: int,
        *,
        member_offsets: torch.Tensor,
        member_positions: torch.Tensor,
        num_communities: torch.Tensor,
        centroids: torch.Tensor,
        community_sizes: torch.Tensor,
        community_weight: torch.Tensor,
        total_weight: torch.Tensor,
        token_communities: torch.Tensor,
        retrieval_to_graph: torch.Tensor | None = None,
        retrieval_to_kv: torch.Tensor | None = None,
        prefill_seq_len: int,
        max_decode_tokens: int,
        group_size: int = 4,
        aggregation: GraphAggregation = GraphAggregation.PER_QUERY_HEAD,
        query_heads: int | None = None,
        kv_heads: int | None = None,
    ) -> DecodeLayerState:
        if layer_idx in self.layers:
            raise ValueError(f"layer {layer_idx} is already prepared")
        converted = convert_prefill_tensors(
            member_offsets=member_offsets,
            member_positions=member_positions,
            num_communities=num_communities,
            centroids=centroids,
            community_sizes=community_sizes,
            community_weight=community_weight,
            total_weight=total_weight,
            token_communities=token_communities,
            retrieval_to_graph=retrieval_to_graph,
            retrieval_to_kv=retrieval_to_kv,
            prefill_seq_len=prefill_seq_len,
            max_decode_tokens=max_decode_tokens,
            group_size=group_size,
            aggregation=aggregation,
            query_heads=query_heads,
            kv_heads=kv_heads,
        )
        mutable = converted.mutable
        token_budget = _effective_token_budget(
            configured_budget=self.config.token_budget,
            prefill_seq_len=prefill_seq_len,
            num_sink=self.config.num_sink,
        )
        workspace = DecodeLayerWorkspace.allocate(
            retrieval_head_count=mutable.retrieval_head_count,
            graph_count=mutable.graph_count,
            group_size=converted.group_size,
            community_capacity=mutable.community_capacity,
            token_budget=token_budget,
            num_sink=self.config.num_sink,
            dtype=torch.bfloat16,
            device=mutable.centroids.device,
            score_community_counts=converted.initial_community_counts,
            score_headroom=max_decode_tokens,
        )
        layer = DecodeLayerState(
            converted=converted,
            workspace=workspace,
            sink_positions=torch.arange(
                self.config.num_sink,
                dtype=torch.int32,
                device=mutable.centroids.device,
            ),
        )
        self.layers[layer_idx] = layer
        return layer

    def decode(
        self,
        *,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        layer = self.layers.get(layer_idx)
        if layer is None:
            raise RuntimeError(f"decode layer {layer_idx} is not prepared")
        if (
            query.ndim != 4
            or query.shape[0] != 1
            or query.shape[2] != 1
            or query.shape[-1] != HEAD_DIM
        ):
            raise ValueError("decode requires query shape [1, Hq, 1, 128]")
        expected_query_heads = (
            layer.converted.query_heads
        )
        if query.shape[1] != expected_query_heads:
            raise ValueError(
                f"decode expected {expected_query_heads} query heads, "
                f"found {query.shape[1]}"
            )
        if key.ndim != 4 or key.shape[0] != 1 or value.shape != key.shape:
            raise ValueError("decode requires key/value shape [1, Hkv, S, 128]")
        if key.shape[1] != layer.converted.kv_heads:
            raise ValueError(
                f"decode expected {layer.converted.kv_heads} KV heads, "
                f"found {key.shape[1]}"
            )
        output, _, _, _, _ = decode_layer(
            query=query[0, :, 0, :],
            key=key[0],
            value=value[0],
            csr=layer.converted.csr,
            state=layer.converted.mutable,
            sink_positions=layer.sink_positions,
            current_position=layer.converted.current_position,
            softmax_scale=softmax_scale,
            lam=self.config.lam,
            workspace=layer.workspace,
        )
        return output.view(1, 1, query.shape[1], HEAD_DIM)

    def assert_no_overflow(self) -> None:
        failures = {
            layer_idx: layer.workspace.update.overflow.detach().cpu().tolist()
            for layer_idx, layer in self.layers.items()
            if torch.count_nonzero(layer.workspace.update.overflow).item()
        }
        if failures:
            raise RuntimeError(f"decode state overflow: {failures}")


class CommunityKVRuntime:
    """Request runtime spanning fused prefill, async partition, and decode."""

    def __init__(
        self,
        *,
        config: CommunityKVConfig,
        partition: PartitionConfig,
        num_layers: int,
        max_decode_tokens: int,
    ) -> None:
        if num_layers <= 0 or max_decode_tokens <= 0:
            raise ValueError("num_layers and max_decode_tokens must be positive")
        self.config = config
        self.num_layers = num_layers
        self.max_decode_tokens = max_decode_tokens
        self.decode_runtime = DecodeRuntime(config)
        self.partition_runtime = PartitionRuntime(
            algorithm=config,
            scheduling=partition,
            max_layers=num_layers,
        )
        self.resolutions: dict[int, float] = {}
        self._ready = False
        self._generation_cache: Any | None = None

    def __enter__(self) -> "CommunityKVRuntime":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def start_request(
        self,
        *,
        resolutions: Mapping[int, float] | Sequence[float] | None = None,
        max_decode_tokens: int | None = None,
    ) -> None:
        self.partition_runtime.reset()
        self.decode_runtime.reset()
        self._ready = False
        if max_decode_tokens is not None:
            if max_decode_tokens <= 0:
                raise ValueError("max_decode_tokens must be positive")
            self.max_decode_tokens = max_decode_tokens
        if resolutions is None:
            self.resolutions = {}
        elif isinstance(resolutions, Mapping):
            self.resolutions = {
                int(layer): float(value) for layer, value in resolutions.items()
            }
        else:
            if len(resolutions) != self.num_layers:
                raise ValueError("resolution vector must contain one value per layer")
            self.resolutions = {
                layer: float(value) for layer, value in enumerate(resolutions)
            }
        if any(value <= 0 for value in self.resolutions.values()):
            raise ValueError("all layer resolutions must be positive")

    def submit_prefill_layer(
        self,
        *,
        layer_idx: int,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor,
        keys: torch.Tensor,
        completion_event: torch.cuda.Event,
    ) -> None:
        self.partition_runtime.submit(
            layer_idx=layer_idx,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            keys=keys,
            completion_event=completion_event,
            resolution=self.resolutions.get(
                layer_idx,
                self.config.leiden_resolution,
            ),
            max_decode_tokens=self.max_decode_tokens,
        )

    def finish_prefill(self) -> None:
        if self._ready:
            return
        self.partition_runtime.wait(self.decode_runtime.prepare_partitioned_layer)
        actual_layers = set(self.decode_runtime.layers)
        expected_layers = set(range(self.num_layers))
        if actual_layers != expected_layers:
            raise RuntimeError(
                "partitioned layer set is incomplete: "
                f"missing={sorted(expected_layers - actual_layers)}, "
                f"extra={sorted(actual_layers - expected_layers)}"
            )
        self._ready = True
        if self._generation_cache is not None:
            self._generation_cache.enable_device_updates()

    def ensure_ready(self) -> None:
        if not self._ready:
            self.finish_prefill()

    def generate(
        self,
        model: Any,
        inputs: torch.Tensor | None = None,
        *,
        resolutions: Mapping[int, float] | Sequence[float] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate one sequence with CommunityKV-managed request state and cache."""

        if inputs is not None and "input_ids" in kwargs:
            raise ValueError("pass input_ids either positionally or by keyword")
        input_ids = kwargs.get("input_ids", inputs)
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("generate requires rank-two input_ids")
        if input_ids.shape[0] != 1:
            raise ValueError("CommunityKV generate currently requires batch size 1")
        if getattr(model.config, "_attn_implementation", None) != "community_kv":
            raise ValueError(
                "model.config._attn_implementation must be 'community_kv'"
            )
        if kwargs.get("past_key_values") is not None:
            raise ValueError("CommunityKV generate manages past_key_values")
        if kwargs.get("cache_implementation") is not None:
            raise ValueError("CommunityKV generate manages the cache implementation")
        if kwargs.get("assistant_model") is not None:
            raise ValueError("CommunityKV generate does not support assisted generation")
        if kwargs.get("penalty_alpha") is not None:
            raise ValueError("CommunityKV generate does not support contrastive search")
        if kwargs.get("do_sample") not in (None, False):
            raise ValueError("CommunityKV generate supports greedy decoding only")

        generation_config = kwargs.get(
            "generation_config",
            getattr(model, "generation_config", None),
        )

        def generation_option(name: str, default: Any) -> Any:
            if name in kwargs:
                return kwargs[name]
            return getattr(generation_config, name, default)

        if generation_option("num_beams", 1) != 1:
            raise ValueError("CommunityKV generate does not support beam search")
        if generation_option("num_return_sequences", 1) != 1:
            raise ValueError(
                "CommunityKV generate supports one returned sequence per prompt"
            )

        prompt_tokens = int(input_ids.shape[-1])
        max_new_tokens = generation_option("max_new_tokens", None)
        if max_new_tokens is None:
            max_length = generation_option("max_length", None)
            if max_length is None:
                raise ValueError("generate requires max_new_tokens or max_length")
            max_new_tokens = int(max_length) - prompt_tokens
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")

        model_layers = int(model.config.num_hidden_layers)
        if model_layers != self.num_layers:
            raise ValueError(
                f"runtime has {self.num_layers} layers but model has {model_layers}"
            )

        from community_kv.attention.cache import StaticCache

        cache = StaticCache(
            num_hidden_layers=self.num_layers,
            max_cache_len=prompt_tokens + max_new_tokens,
        )
        if getattr(self, "_generation_cache", None) is not None:
            raise RuntimeError("CommunityKV runtime is already generating")
        self.start_request(
            resolutions=resolutions,
            max_decode_tokens=max(1, max_new_tokens - 1),
        )
        generate_kwargs = dict(kwargs)
        generate_kwargs["past_key_values"] = cache
        generate_kwargs["do_sample"] = False
        generate_kwargs["num_beams"] = 1
        generate_kwargs["num_return_sequences"] = 1
        if hasattr(generation_config, "disable_compile"):
            generate_kwargs["disable_compile"] = True
        self._generation_cache = cache
        try:
            if inputs is None:
                output = model.generate(**generate_kwargs)
            else:
                output = model.generate(inputs, **generate_kwargs)
            self.finish_prefill()
            self.assert_no_overflow()
            return output
        finally:
            self._generation_cache = None

    def close(self) -> None:
        self.partition_runtime.shutdown()

    def assert_no_overflow(self) -> None:
        self.decode_runtime.assert_no_overflow()


__all__ = [
    "CommunityKVRuntime",
    "ConvertedPrefillState",
    "DecodeLayerState",
    "DecodeRuntime",
    "convert_prefill_tensors",
]
