"""Tests for community_kv.graph.state — graph kind enum + per-layer dataclasses."""

import pickle

import pytest
import torch

from community_kv.graph.state import GraphAggregation, LayerLog


class TestGraphAggregation:
    @pytest.mark.parametrize(
        "agg,H_q,H_kv,expected",
        [
            (GraphAggregation.PER_QUERY_HEAD, 32, 8, 32),
            (GraphAggregation.QUERY_GROUP, 32, 8, 8),
            (GraphAggregation.LAYER_WISE, 32, 8, 1),
        ],
    )
    def test_num_graphs_per_layer(self, agg, H_q, H_kv, expected):
        assert agg.num_graphs_per_layer(H_q, H_kv) == expected

    def test_string_value(self):
        assert GraphAggregation("per_query_head") is GraphAggregation.PER_QUERY_HEAD
        assert GraphAggregation("query_group") is GraphAggregation.QUERY_GROUP
        assert GraphAggregation("layer_wise") is GraphAggregation.LAYER_WISE

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            # Bypass enum membership to construct a fake GraphAggregation
            class Fake:
                pass

            GraphAggregation.num_graphs_per_layer(Fake(), 32, 8)


class TestLayerGraph:
    def test_construction_with_required_fields(self, layer_graph):
        graph = layer_graph(G=2, S=8, max_C=4, D=16)
        assert graph.layer_idx == 0
        assert graph.decode_log_size == 0
        assert graph.decode_edge_size == 0
        assert graph.version == 0
        assert graph.prefill_edge_src is None


class TestLayerLog:
    """``LayerLog`` carries CUDA timing events that are intentionally
    stripped on pickling so the dataclass survives ``dist.gather_object``
    across ranks. Tests cover construction, the no-event roundtrip, and
    the events-resolved roundtrip."""

    def test_construction_no_events(self):
        log = LayerLog(
            fwd_device="cuda:0",
            part_device="cuda:0",
            prefill_seq_len=512,
            attn_ms=3.14,
        )
        assert log.fwd_device == "cuda:0"
        assert log.attn_ms == 3.14
        assert log.ev_attn_start is None
        assert log.ev_attn_end is None

    def test_pickle_roundtrip_drops_events(self):
        log = LayerLog(
            fwd_device="cuda:1",
            part_device="cuda:1",
            prefill_seq_len=1024,
            attn_ms=12.5,
        )
        roundtripped = pickle.loads(pickle.dumps(log))
        assert isinstance(roundtripped, LayerLog)
        assert roundtripped.fwd_device == "cuda:1"
        assert roundtripped.prefill_seq_len == 1024
        assert roundtripped.attn_ms == 12.5
        assert roundtripped.ev_attn_start is None
        assert roundtripped.ev_attn_end is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA Event")
    def test_pickle_resolves_events_first(self):
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        ev_s.record()
        ev_e.record()
        torch.cuda.synchronize()
        log = LayerLog(
            fwd_device="cuda:0",
            part_device="cuda:0",
            prefill_seq_len=1,
            ev_attn_start=ev_s,
            ev_attn_end=ev_e,
        )
        # attn_ms is 0 until resolve / pickle.
        assert log.attn_ms == 0.0
        roundtripped = pickle.loads(pickle.dumps(log))
        # After roundtrip the events are stripped and attn_ms is set.
        assert roundtripped.ev_attn_start is None
        assert roundtripped.attn_ms >= 0.0  # any non-negative timing
