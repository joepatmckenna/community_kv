"""Tests for community_kv.attention.community_kv.

Two classes:

1. ``TestCommunityKVAttentionWrapper`` — CPU. Pins the impl-name constant
   and the ``CommunityKVAttention`` OOP wiring (init, register, isolation).

2. ``TestCommunityKVAttentionFunctional`` — CUDA + patched fused-attn-fwd-topk +
   Leiden CUDA extension. Exercises the real prefill / decode paths
   across (lam, aggregation, kappa, num_sink, GQA).

All shared helpers (``FakeAttnModule``, ``make_test_runtime``,
``bshd_to_bhsd``, ``drain_runtime``, ``make_qkv``, ``FA_HEAD_DIM``,
``FUSED_ATTN_FWD_TOPK_AND_LEIDEN_REQUIRED``) live in ``tests/conftest.py``.

The functional class needs both the patched fused-attn-fwd-topk build AND the
compiled native Leiden extension — the prefill path submits a partition
job to the worker pool that imports the Leiden ``.so`` on the worker
thread, and the decode path mutates graph state that was built by Leiden.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
import torch

from community_kv.attention import community_kv as ckv_mod
from community_kv.attention.community_kv import (
    COMMUNITY_KV_ATTN_IMPL,
    CommunityKVAttention,
    community_kv_attention_forward,
)
from community_kv.graph.runtime import GraphRuntime
from community_kv.graph.state import GraphAggregation
from tests.conftest import (
    FA_HEAD_DIM,
    FUSED_ATTN_FWD_TOPK_AND_LEIDEN_REQUIRED,
    FakeAttnModule,
    append_decode_step,
    bshd_to_bhsd,
    drain_runtime,
    make_qkv,
    make_test_runtime,
)


class TestCommunityKVAttentionWrapper:
    """CPU pinning for the impl-name constant + the OOP wiring around the
    free ``community_kv_attention_forward``."""

    def test_impl_name_const(self):
        assert COMMUNITY_KV_ATTN_IMPL == "COMMUNITY_KV_ATTN"
        assert CommunityKVAttention.IMPL_NAME == COMMUNITY_KV_ATTN_IMPL

    def test_init_takes_graph_runtime(self):
        graph_runtime = GraphRuntime()
        attn = CommunityKVAttention(graph_runtime=graph_runtime)
        assert attn.graph_runtime is graph_runtime

    def test_init_requires_graph_runtime(self):
        """``graph_runtime`` is a required keyword-only arg."""
        with pytest.raises(TypeError):
            CommunityKVAttention()  # type: ignore[call-arg]

    def test_register_writes_config_into_gr(self, monkeypatch):
        """``register()`` populates ``graph_runtime.config`` and adds the bound
        forward to HF's attention registry under the impl name."""
        graph_runtime = GraphRuntime()
        attn = CommunityKVAttention(graph_runtime=graph_runtime)

        fake_registry: dict = {}
        fake_modeling_utils = type(sys)("transformers.modeling_utils")
        fake_modeling_utils.ALL_ATTENTION_FUNCTIONS = fake_registry
        monkeypatch.setitem(sys.modules, "transformers.modeling_utils", fake_modeling_utils)

        name = attn.register(
            kappa=4,
            num_sink=8,
            lam=0.6,
            leiden_resolution=0.5,
            leiden_max_iter=3,
            max_new_tokens=64,
            token_budget=2048,
        )
        assert name == "COMMUNITY_KV_ATTN"
        for k, v in {
            "kappa": 4,
            "num_sink": 8,
            "lam": 0.6,
            "leiden_resolution": 0.5,
            "leiden_max_iter": 3,
            "max_new_tokens": 64,
            "token_budget": 2048,
        }.items():
            assert graph_runtime.config[k] == v
        assert "COMMUNITY_KV_ATTN" in fake_registry
        # Bound methods generate a fresh wrapper per access; identity
        # comparison fails. Equality is what we actually want.
        assert fake_registry["COMMUNITY_KV_ATTN"] == attn.forward

    def test_separate_instances_isolate_config(self):
        a = CommunityKVAttention(graph_runtime=GraphRuntime())
        b = CommunityKVAttention(graph_runtime=GraphRuntime())
        a.graph_runtime.config["kappa"] = 99
        assert b.graph_runtime.config["kappa"] == 8

    def test_forward_delegates_to_free_function(self):
        """``CommunityKVAttention.forward`` is a thin wrapper around the
        free ``community_kv_attention_forward``: it splices in the bound
        ``self.graph_runtime`` and passes everything else through. Verify
        on CPU via mock.patch."""
        gr = GraphRuntime()
        attn = CommunityKVAttention(graph_runtime=gr)
        sentinel = object()
        with patch.object(ckv_mod, "community_kv_attention_forward", return_value=sentinel) as mock:
            result = attn.forward(
                "module",
                "query",
                "key",
                "value",
                "attention_mask",
                "scaling",
                dropout=0.25,
                extra_kwarg="passed_through",
            )
        assert result is sentinel
        mock.assert_called_once_with(
            "module",
            "query",
            "key",
            "value",
            "attention_mask",
            "scaling",
            dropout=0.25,
            graph_runtime=gr,
            extra_kwarg="passed_through",
        )


# --------------------------------------------------------------------------- #
# Functional GPU tests
# --------------------------------------------------------------------------- #


@FUSED_ATTN_FWD_TOPK_AND_LEIDEN_REQUIRED
class TestCommunityKVAttentionFunctional:
    """Real prefill / decode paths with fused-attn-fwd-topk + Leiden + per-q-head
    retrieval.

    Test matrix:
        lam         in {0.0, 0.5, 1.0}    graph-edge mixing (w1 vs w2)
        aggregation in {per_query_head, query_group, layer_wise}
        kappa       in {4, 8}             topk cardinality
        num_sink    in {0, 4}             sink exclusion
        GQA         H_q:H_kv in {8:8, 8:2}
    """

    # ---- prefill path ---------------------------------------------------- #

    @pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
    def test_prefill_populates_layer_graph(self, lam):
        """Prefill must populate ``graph_runtime.graphs[layer_idx]`` with a LayerGraph
        whose shapes line up with (G, S_k, max_C). lam variants stress
        different edge-construction paths."""
        torch.manual_seed(0)
        B, S, H_q, H_kv = 1, 128, 8, 8
        graph_runtime = make_test_runtime(kappa=8, num_sink=4, lam=lam, max_new_tokens=4)
        q, k, v = make_qkv(B, S, S, H_q, H_kv, FA_HEAD_DIM, device="cuda")
        out, _ = community_kv_attention_forward(
            module=FakeAttnModule(layer_idx=0),
            query=bshd_to_bhsd(q),
            key=bshd_to_bhsd(k),
            value=bshd_to_bhsd(v),
            attention_mask=None,
            scaling=1.0 / (FA_HEAD_DIM**0.5),
            graph_runtime=graph_runtime,
        )
        drain_runtime(graph_runtime)
        assert out.shape == (B, S, H_q, FA_HEAD_DIM)
        assert 0 in graph_runtime.graphs
        graph = graph_runtime.graphs[0]
        assert graph.community_ids.shape[0] == H_q
        assert graph.prefill_seq_len == S
        assert graph.prefill_edge_src is not None
        assert graph.prefill_edge_src.numel() > 0
        assert graph.decode_log_size == 0
        graph_runtime.shutdown()

    def test_prefill_output_matches_attn_forward_topk(self):
        """The prefill output is exactly what ``attn_forward_topk`` would
        return — the only side effect is the async partition. Compared
        bit-for-bit on the same seed."""
        from community_kv.attention.fused_attn_fwd_topk import attn_forward_topk

        torch.manual_seed(0)
        B, S, H_q, H_kv = 1, 128, 8, 8
        kappa, num_sink = 8, 4
        graph_runtime = make_test_runtime(kappa=kappa, num_sink=num_sink, lam=0.5, max_new_tokens=4)
        q, k, v = make_qkv(B, S, S, H_q, H_kv, FA_HEAD_DIM, device="cuda")
        scaling = 1.0 / (FA_HEAD_DIM**0.5)
        out, _ = community_kv_attention_forward(
            module=FakeAttnModule(0),
            query=bshd_to_bhsd(q),
            key=bshd_to_bhsd(k),
            value=bshd_to_bhsd(v),
            attention_mask=None,
            scaling=scaling,
            graph_runtime=graph_runtime,
        )
        drain_runtime(graph_runtime)
        ref = attn_forward_topk(
            q,
            k,
            v,
            softmax_scale=scaling,
            topk_K=kappa,
            exclude_sink_tokens=num_sink,
        )[0]
        assert torch.equal(out, ref), "prefill output should bit-equal attn_forward_topk(...)[0]"
        graph_runtime.shutdown()

    @pytest.mark.parametrize(
        "aggregation,expected_G",
        [
            (GraphAggregation.PER_QUERY_HEAD, 8),
            (GraphAggregation.QUERY_GROUP, 2),
            (GraphAggregation.LAYER_WISE, 1),
        ],
    )
    def test_aggregation_modes(self, aggregation, expected_G):
        torch.manual_seed(0)
        B, S, H_q, H_kv = 1, 128, 8, 2
        graph_runtime = make_test_runtime(
            kappa=8,
            num_sink=4,
            lam=0.5,
            max_new_tokens=4,
            aggregation=aggregation,
        )
        q, k, v = make_qkv(B, S, S, H_q, H_kv, FA_HEAD_DIM, device="cuda")
        community_kv_attention_forward(
            module=FakeAttnModule(0),
            query=bshd_to_bhsd(q),
            key=bshd_to_bhsd(k),
            value=bshd_to_bhsd(v),
            attention_mask=None,
            scaling=1.0 / (FA_HEAD_DIM**0.5),
            graph_runtime=graph_runtime,
        )
        drain_runtime(graph_runtime)
        graph = graph_runtime.graphs[0]
        assert graph.community_ids.shape[0] == expected_G
        assert graph.aggregation == aggregation
        graph_runtime.shutdown()

    @pytest.mark.parametrize("kappa,num_sink", [(4, 0), (8, 0), (4, 4), (8, 4)])
    def test_prefill_kappa_num_sink_combinations(self, kappa, num_sink):
        torch.manual_seed(0)
        B, S, H_q, H_kv = 1, 128, 8, 8
        graph_runtime = make_test_runtime(kappa=kappa, num_sink=num_sink, lam=0.5, max_new_tokens=4)
        q, k, v = make_qkv(B, S, S, H_q, H_kv, FA_HEAD_DIM, device="cuda")
        community_kv_attention_forward(
            module=FakeAttnModule(0),
            query=bshd_to_bhsd(q),
            key=bshd_to_bhsd(k),
            value=bshd_to_bhsd(v),
            attention_mask=None,
            scaling=1.0 / (FA_HEAD_DIM**0.5),
            graph_runtime=graph_runtime,
        )
        drain_runtime(graph_runtime)
        assert 0 in graph_runtime.graphs
        assert graph_runtime.graphs[0].prefill_edge_src.numel() > 0
        graph_runtime.shutdown()

    # ---- decode path ----------------------------------------------------- #

    @pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
    def test_decode_after_prefill_mutates_graph(self, lam, prefilled_runtime):
        """After prefill + a single decode step, the graph's decode log,
        decode-edge buffer, and ``community_ids`` row at ``current_pos``
        must all be updated. Under each lam value to confirm the w1/w2
        decode edge writes both fire."""
        S = 128
        graph_runtime, k, v = prefilled_runtime(S=S, kappa=8, num_sink=4, lam=lam, max_new_tokens=4)
        graph = graph_runtime.graphs[0]
        edge_size_before = graph.decode_edge_size
        log_size_before = graph.decode_log_size

        out_dec, _, _ = append_decode_step(graph_runtime, k, v)

        assert out_dec.shape == (1, 1, 8, FA_HEAD_DIM)
        assert graph.decode_log_size == log_size_before + 1
        assert (graph.community_ids[:, S] != -1).all()
        assert graph.decode_edge_size > edge_size_before

    def test_decode_without_prefill_raises(self):
        """If decode runs before prefill populated the layer's graph, the
        forward should ``assert`` (better than silently doing nothing or
        falling back)."""
        graph_runtime = make_test_runtime(kappa=8, num_sink=4, lam=0.5, max_new_tokens=4)
        S_k = 64
        q = torch.randn(1, 1, 8, FA_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, S_k, 8, FA_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, S_k, 8, FA_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(AssertionError, match="layer 0"):
            community_kv_attention_forward(
                module=FakeAttnModule(0),
                query=bshd_to_bhsd(q),
                key=bshd_to_bhsd(k),
                value=bshd_to_bhsd(v),
                attention_mask=None,
                scaling=1.0 / (FA_HEAD_DIM**0.5),
                graph_runtime=graph_runtime,
            )
        graph_runtime.shutdown()

    # ---- GQA ------------------------------------------------------------- #

    def test_prefill_gqa_4to1(self):
        torch.manual_seed(0)
        B, S, H_q, H_kv = 1, 128, 8, 2
        graph_runtime = make_test_runtime(kappa=8, num_sink=4, lam=0.5, max_new_tokens=4)
        q, k, v = make_qkv(B, S, S, H_q, H_kv, FA_HEAD_DIM, device="cuda")
        out, _ = community_kv_attention_forward(
            module=FakeAttnModule(0),
            query=bshd_to_bhsd(q),
            key=bshd_to_bhsd(k),
            value=bshd_to_bhsd(v),
            attention_mask=None,
            scaling=1.0 / (FA_HEAD_DIM**0.5),
            graph_runtime=graph_runtime,
        )
        drain_runtime(graph_runtime)
        assert out.shape == (B, S, H_q, FA_HEAD_DIM)
        assert graph_runtime.graphs[0].community_ids.shape[0] == H_q
        graph_runtime.shutdown()

    # ---- branch coverage of the decode path ------------------------------ #

    def test_decode_snapshots_keys_when_repartition_pending(self, prefilled_runtime):
        """If a layer is in ``repartition_trigger_pending`` when the decode
        forward fires, that forward must capture a key snapshot into
        ``repartition_key_snapshots`` for the later async-repartition
        dispatch to consume."""
        S, H_kv = 128, 8
        graph_runtime, k, v = prefilled_runtime(S=S, kappa=8, num_sink=4, lam=0.5, max_new_tokens=4)

        # Mark layer 0 as eligible for snapshotting.
        graph_runtime.repartition_trigger_pending.add(0)
        assert 0 not in graph_runtime.repartition_key_snapshots

        append_decode_step(graph_runtime, k, v)

        assert 0 in graph_runtime.repartition_key_snapshots
        # Snapshot is ``key[0]``: BHSD-shaped key tensor minus the batch dim.
        snap = graph_runtime.repartition_key_snapshots[0]
        assert snap.shape == (H_kv, S + 1, FA_HEAD_DIM)

    def test_second_decode_uses_decode_log(self, prefilled_runtime):
        """The second decode step has ``log_size > 0`` because the first
        step recorded its assignment. Exercises the K>0 path of the
        log-replay branch in ``community_kv_attention_forward``."""
        graph_runtime, k, v = prefilled_runtime(
            S=512, kappa=8, num_sink=4, lam=0.5, max_new_tokens=8, token_budget=128
        )
        graph = graph_runtime.graphs[0]

        # First decode step.
        _, k, v = append_decode_step(graph_runtime, k, v)
        assert graph.decode_log_size == 1

        # Second decode step — log_size > 0 path.
        out, _, _ = append_decode_step(graph_runtime, k, v)
        assert out.shape == (1, 1, 8, FA_HEAD_DIM)
        assert graph.decode_log_size == 2

    def test_decode_with_token_budget_exceeding_cache(self, prefilled_runtime):
        """When ``token_budget > S_k``, the gather caps at ``S_k`` (dense
        attention over the whole cache, no -1 sentinels duplicated to
        ``current_pos``). ``effective_budget = S_k`` should make the
        retrieve_budget cover all community members exactly once."""
        S, H_q = 128, 8
        # token_budget=4096 against an S_k=128 cache: effective_budget caps at 128.
        graph_runtime, k, v = prefilled_runtime(
            S=S, kappa=8, num_sink=4, lam=0.5, max_new_tokens=4, token_budget=4096
        )

        out, _, _ = append_decode_step(graph_runtime, k, v)
        # Output is valid; the kernel saw an effective_budget = S_k = 129.
        assert out.shape == (1, 1, H_q, FA_HEAD_DIM)
        # All retrieved slots covered real positions — total_retrieved == S_k - num_sink - 1
        # (= prefill_body + log_size). The decode_retrieve_n entry pins this.
        per_call = graph_runtime.decode_retrieve_n[0][0]  # (H_q, 2)
        assert per_call.shape == (H_q, 2)
        # total_retrieved per head == 128 - 4 = 124 (S_k=129 - num_sink=4 - current=1).
        assert torch.all(per_call[:, 1] == 124)

    def test_decode_kappa_too_large_for_pool_uses_python_fallback(self, prefilled_runtime):
        """When the post-sink pool is smaller than ``kappa``, the kernel's
        compile-time top-K dispatch can't run. The decode path runs the
        same kernel with ``return_topk=False`` (plain FA, no probe) and
        synthesizes the graph-update inputs from all post-sink positions."""
        S = 32
        # kappa=8 in config; token_budget=11 with num_sink=4 -> pool=7 (< kappa).
        graph_runtime, k, v = prefilled_runtime(
            S=S, kappa=8, num_sink=4, lam=0.5, max_new_tokens=4, token_budget=11
        )
        graph = graph_runtime.graphs[0]

        out, _, _ = append_decode_step(graph_runtime, k, v)
        # Output is valid AND the graph still got mutated — the fallback
        # synthesized post-sink "top tokens" for decode_step_update.
        assert out.shape == (1, 1, 8, FA_HEAD_DIM)
        assert graph.decode_log_size == 1
        assert (graph.community_ids[:, S] != -1).all()

    def test_decode_zero_retrieve_budget_with_valid_pool(self, prefilled_runtime):
        """When ``retrieve_budget`` rounds to 0 but the post-sink pool
        still admits a valid ``kappa_eff`` (e.g. on the second decode
        step with ``effective_budget = num_sink + 2`` and ``log_size=1``),
        the K==0 branches in ``_retrieve`` and ``_record_retrieval_stats``
        run cleanly: no community members gathered, but sinks + decode-log
        + current still fill the gather, and FA-topK runs over the
        post-sink pool of size 2."""
        # token_budget=6, num_sink=4 -> first decode: pool=2, retrieve_budget=1, K=1.
        # Second decode (log_size=1): retrieve_budget=0, K=0 path fires.
        graph_runtime, k, v = prefilled_runtime(
            S=128, kappa=8, num_sink=4, lam=0.5, max_new_tokens=4, token_budget=6
        )
        # First decode (K > 0).
        _, k, v = append_decode_step(graph_runtime, k, v)
        # Second decode (log_size=1 -> K = 0 path fires).
        out, _, _ = append_decode_step(graph_runtime, k, v)
        assert out.shape == (1, 1, 8, FA_HEAD_DIM)
        # Second-call retrieve summary recorded the K=0 case (N==0, total_retrieved==0).
        per_call_2 = graph_runtime.decode_retrieve_n[0][1]  # (H_q, 2)
        assert per_call_2.shape == (8, 2)
        assert torch.all(per_call_2 == 0)

    def test_retrieved_indices_stay_within_cache(self, prefilled_runtime, monkeypatch):
        """Inactive headroom (the unwritten tail of the decode buffer:
        ``decode_log_position[t:max_new_tokens-1]`` and the corresponding
        rows in ``community_ids`` past ``prefill_seq_len + log_size``)
        must NEVER appear in the ``retrieved`` gather. With the
        ``effective_budget`` cap, every entry must lie in ``[0, S_k)``.
        Capture the ``retrieved`` tensor passed to ``decode_step_update``
        and assert it directly."""
        captured: list[torch.Tensor] = []
        real = ckv_mod.decode_step_update

        def capturing(*args, **kwargs):
            captured.append(kwargs["retrieved"].detach().clone())
            return real(*args, **kwargs)

        monkeypatch.setattr(ckv_mod, "decode_step_update", capturing)

        # token_budget=4096 vs S_k that grows past prefill: forces the
        # cap-at-S_k path. max_new_tokens=8 means the decode_log buffer
        # has 7 unused slots after the first decode.
        graph_runtime, k, v = prefilled_runtime(
            S=128, kappa=8, num_sink=4, lam=0.5, max_new_tokens=8, token_budget=4096
        )

        # Three decode steps: the cache grows S_k = 129, 130, 131. Each
        # step's `retrieved` must stay within [0, S_k). The "inactive
        # headroom" rows (decode-buffer slots past log_size, and any
        # community_ids columns past prefill_seq_len + log_size) must
        # never appear.
        for step in range(3):
            _, k, v = append_decode_step(graph_runtime, k, v)
            S_k = k.shape[1]
            ret = captured[-1]
            assert ret.max().item() < S_k, (
                f"step {step}: retrieved[max]={ret.max().item()} >= S_k={S_k} "
                f"(would index inactive headroom)"
            )
            assert ret.min().item() >= 0, f"step {step}: retrieved has negative index"

    def test_decode_picks_kernel_topk_path_per_pool_size(self, prefilled_runtime, monkeypatch):
        """CommunityKV branches the decode kernel call on pool size: it asks
        for the in-kernel top-K probe (``return_topk=True``) when
        ``post_sink_pool >= kappa`` and degrades to plain FA
        (``return_topk=False``, Python fallback) otherwise. Verify the flag
        CommunityKV picks per call — this is about CommunityKV's branching,
        not the kernel honoring the flag (that lives in
        test_fused_attn_fwd_topk.py)."""
        captured: list[bool] = []
        real = ckv_mod.attn_forward_topk

        def capturing(q, k, v, **kw):
            captured.append(kw.get("return_topk", True))
            return real(q, k, v, **kw)

        monkeypatch.setattr(ckv_mod, "attn_forward_topk", capturing)

        # Sparse / fused path: token_budget large enough -> return_topk=True.
        graph_runtime, k, v = prefilled_runtime(
            S=128, kappa=8, num_sink=4, lam=0.5, max_new_tokens=4, token_budget=64
        )
        captured.clear()
        append_decode_step(graph_runtime, k, v)
        # post_sink_pool = min(64, 129) - 4 = 60 >= kappa=8 -> kernel topK.
        assert captured == [True], f"expected [True], got {captured}"

        # Fallback path: token_budget small enough -> return_topk=False.
        graph_runtime, k, v = prefilled_runtime(
            S=128, kappa=8, num_sink=4, lam=0.5, max_new_tokens=4, token_budget=11
        )
        captured.clear()
        append_decode_step(graph_runtime, k, v)
        # post_sink_pool = 11 - 4 = 7 < kappa=8 -> Python fallback.
        assert captured == [False], f"expected [False], got {captured}"

    def test_decode_with_tiny_pool_still_runs_via_fallback(self, prefilled_runtime):
        """Edge case: ``post_sink_pool == 1`` (only the current token).
        Previously this raised; with the Python fallback the kernel runs
        plain FA and the graph update has exactly one "top token" (the
        current position). Sanity-check that nothing crashes."""
        S = 32
        # token_budget=5, num_sink=4 -> effective_budget=5, post_sink_pool=1.
        graph_runtime, k, v = prefilled_runtime(
            S=S, kappa=8, num_sink=4, lam=0.5, max_new_tokens=4, token_budget=5
        )
        graph = graph_runtime.graphs[0]

        out, _, _ = append_decode_step(graph_runtime, k, v)
        assert out.shape == (1, 1, 8, FA_HEAD_DIM)
        assert graph.decode_log_size == 1
