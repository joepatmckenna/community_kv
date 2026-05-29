"""Tests for community_kv.attention.fused_attn_fwd_topk.

Two classes, two layers of coverage:

1. ``TestAttnForwardTopkWrapper`` — CPU, mocked flash-attention. Pins the
   wrapper's argument shuffling so any upstream API change surfaces here
   rather than at runtime on a GPU box.
2. ``TestAttnForwardTopkFunctional`` — CUDA + patched flash-attention.
   Exercises the real forward + topk probe across (kappa, num_sink,
   causal, GQA ratio).

Helpers (``make_qkv``, ``ref_sdpa``, ``fa3_abs_q_positions``,
``fake_fused_attn_fwd_topk``, ``FUSED_ATTN_FWD_TOPK_REQUIRED``, ``FA_HEAD_DIM``) live in
``tests/conftest.py``.

The FA3 topk-probe is gated at compile time on::

    Element == bf16  &&  head_dim_v == 128  &&  pack_gqa  &&  !paged  &&  !fp8

so the functional class pins those constraints. Top-K is also restricted
to ``topk_K in {2, 4, 8, 16, 32}`` at runtime (FA3 raises otherwise).
"""

from __future__ import annotations

import sys

import pytest
import torch

import community_kv.attention.fused_attn_fwd_topk as fused
from tests.conftest import (
    FA_HEAD_DIM,
    FUSED_ATTN_FWD_TOPK_REQUIRED,
    fa3_abs_q_positions,
    make_qkv,
    ref_sdpa,
)


class TestAttnForwardTopkWrapper:
    """CPU pinning for the wrapper layer. Uses the ``fake_fused_attn_fwd_topk``
    fixture from conftest to capture and assert on call kwargs."""

    def test_import_error_message_when_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "flash_attn_interface", None)
        with pytest.raises(ImportError, match="fused-attn-fwd-topk"):
            fused._import_upstream_fused_attn_fwd_topk()

    def test_pins_topk_knobs(self, fake_fused_attn_fwd_topk):
        out = fused.attn_forward_topk(
            q="q-tensor",
            k="k-tensor",
            v="v-tensor",
            softmax_scale=0.125,
            topk_K=8,
            exclude_sink_tokens=10,
        )
        assert out[0] == "attn_output"
        assert out[4] == "topk_scores"
        assert out[5] == "topk_indices"

        kwargs = fake_fused_attn_fwd_topk.calls[-1]
        assert kwargs["softmax_scale"] == 0.125
        assert kwargs["return_topk"] is True
        assert kwargs["topk_K"] == 8
        assert kwargs["exclude_sink_tokens"] == 10

    def test_default_causal_and_pack_gqa(self, fake_fused_attn_fwd_topk):
        fused.attn_forward_topk(
            q="q",
            k="k",
            v="v",
            softmax_scale=0.5,
            topk_K=4,
            exclude_sink_tokens=2,
        )
        kwargs = fake_fused_attn_fwd_topk.calls[-1]
        assert kwargs["causal"] is True
        assert kwargs["pack_gqa"] is True

    def test_overrides_pass_through(self, fake_fused_attn_fwd_topk):
        fused.attn_forward_topk(
            q="q",
            k="k",
            v="v",
            softmax_scale=0.5,
            topk_K=4,
            exclude_sink_tokens=2,
            causal=False,
            pack_gqa=False,
        )
        kwargs = fake_fused_attn_fwd_topk.calls[-1]
        assert kwargs["causal"] is False
        assert kwargs["pack_gqa"] is False

    def test_static_args_pinned(self, fake_fused_attn_fwd_topk):
        """The wrapper hides ~25 always-None / always-default upstream args.
        Pin a few representative ones so a future upstream API change shows
        up here as a test failure rather than at runtime."""
        fused.attn_forward_topk(
            q="q",
            k="k",
            v="v",
            softmax_scale=0.5,
            topk_K=4,
            exclude_sink_tokens=2,
        )
        kwargs = fake_fused_attn_fwd_topk.calls[-1]
        for none_arg in [
            "k_new",
            "v_new",
            "qv",
            "out",
            "cu_seqlens_q",
            "cu_seqlens_k",
            "cu_seqlens_k_new",
            "rotary_cos",
            "rotary_sin",
            "q_descale",
            "k_descale",
            "v_descale",
        ]:
            assert kwargs[none_arg] is None
        assert kwargs["window_size"] == (-1, -1)
        assert kwargs["softcap"] == 0.0
        assert kwargs["num_splits"] == 1
        assert kwargs["sm_margin"] == 0


@FUSED_ATTN_FWD_TOPK_REQUIRED
class TestAttnForwardTopkFunctional:
    """Real-kernel correctness on GPU. Exercises::

        kappa     in {2, 4, 8}
        num_sink  in {0, 4}
        causal    in {True, False}
        GQA ratio H_q:H_kv in {8:8 (MHA), 8:2 (4x GQA)}

    All combinations use bf16 + head_dim 128 + pack_gqa=True (the only
    config the FA3 topk probe is compiled for).
    """

    # ---- shape contract -------------------------------------------------- #

    @pytest.mark.parametrize(
        "kappa,num_sink",
        [
            (2, 0),
            (4, 0),
            (8, 0),
            (2, 4),
            (4, 4),
            (8, 4),
        ],
    )
    def test_shape_contract(self, kappa, num_sink):
        torch.manual_seed(0)
        B, S, H_q, H_kv = 1, 256, 8, 8
        q, k, v = make_qkv(B, S, S, H_q, H_kv, FA_HEAD_DIM, device="cuda")
        scaling = 1.0 / (FA_HEAD_DIM**0.5)
        out = fused.attn_forward_topk(
            q,
            k,
            v,
            softmax_scale=scaling,
            topk_K=kappa,
            exclude_sink_tokens=num_sink,
        )
        assert out[0].shape == (B, S, H_q, FA_HEAD_DIM)
        assert out[0].dtype == torch.bfloat16

        S_elig = S - (kappa - 1 + num_sink)
        topk_scores, topk_idx = out[4], out[5]
        assert topk_scores.shape == (B, H_q, S_elig, kappa)
        assert topk_idx.shape == (B, H_q, S_elig, kappa)
        assert topk_scores.dtype == torch.float32
        assert topk_idx.dtype == torch.int32

    # ---- output correctness vs fp32 SDPA --------------------------------- #

    @pytest.mark.parametrize("kappa,num_sink", [(2, 0), (4, 0), (8, 4)])
    def test_output_matches_reference_causal(self, kappa, num_sink):
        torch.manual_seed(0)
        B, S, H_q, H_kv = 1, 256, 8, 8
        q, k, v = make_qkv(B, S, S, H_q, H_kv, FA_HEAD_DIM, device="cuda")
        scaling = 1.0 / (FA_HEAD_DIM**0.5)
        out_fa = fused.attn_forward_topk(
            q,
            k,
            v,
            softmax_scale=scaling,
            topk_K=kappa,
            exclude_sink_tokens=num_sink,
        )[0]
        out_ref, _ = ref_sdpa(q, k, v, scaling=scaling, causal=True)
        torch.testing.assert_close(out_fa.float(), out_ref, atol=5e-2, rtol=5e-2)

    def test_output_matches_reference_non_causal(self):
        torch.manual_seed(0)
        B, S, H_q, H_kv = 1, 128, 4, 4
        q, k, v = make_qkv(B, S, S, H_q, H_kv, FA_HEAD_DIM, device="cuda")
        scaling = 1.0 / (FA_HEAD_DIM**0.5)
        out_fa = fused.attn_forward_topk(
            q,
            k,
            v,
            softmax_scale=scaling,
            topk_K=4,
            exclude_sink_tokens=0,
            causal=False,
        )[0]
        out_ref, _ = ref_sdpa(q, k, v, scaling=scaling, causal=False)
        torch.testing.assert_close(out_fa.float(), out_ref, atol=5e-2, rtol=5e-2)

    # ---- topk index validity --------------------------------------------- #

    @pytest.mark.parametrize(
        "kappa,num_sink",
        [
            (2, 0),
            (4, 0),
            (8, 0),
            (2, 4),
            (4, 4),
            (8, 4),
        ],
    )
    def test_topk_indices_valid_and_causal(self, kappa, num_sink):
        torch.manual_seed(0)
        B, S, H_q, H_kv = 1, 256, 8, 8
        q, k, v = make_qkv(B, S, S, H_q, H_kv, FA_HEAD_DIM, device="cuda")
        scaling = 1.0 / (FA_HEAD_DIM**0.5)
        topk_idx = fused.attn_forward_topk(
            q,
            k,
            v,
            softmax_scale=scaling,
            topk_K=kappa,
            exclude_sink_tokens=num_sink,
        )[5]
        S_elig = topk_idx.shape[2]
        assert (topk_idx >= num_sink).all(), f"some indices are below num_sink={num_sink}"
        assert (topk_idx < S).all(), f"some indices are >= S_k={S}"
        q_abs = fa3_abs_q_positions(S_elig, kappa, num_sink, device="cuda")
        q_abs_b = q_abs.view(1, 1, S_elig, 1).to(topk_idx.dtype)
        assert (topk_idx <= q_abs_b).all(), "some indices peek past the causal-valid range"

    def test_topk_no_duplicates(self):
        torch.manual_seed(0)
        B, S, H_q, H_kv = 1, 256, 8, 8
        q, k, v = make_qkv(B, S, S, H_q, H_kv, FA_HEAD_DIM, device="cuda")
        scaling = 1.0 / (FA_HEAD_DIM**0.5)
        topk_idx = fused.attn_forward_topk(
            q,
            k,
            v,
            softmax_scale=scaling,
            topk_K=8,
            exclude_sink_tokens=0,
        )[5]
        sorted_idx, _ = topk_idx.sort(dim=-1)
        assert (
            sorted_idx[..., 1:] != sorted_idx[..., :-1]
        ).all(), "topk indices have duplicates within a row"

    # ---- topk score round-trip vs reference softmax ---------------------- #

    def test_topk_scores_match_softmax_at_indices(self):
        torch.manual_seed(0)
        B, S, H_q, H_kv = 1, 256, 8, 8
        kappa, num_sink = 4, 4
        q, k, v = make_qkv(B, S, S, H_q, H_kv, FA_HEAD_DIM, device="cuda")
        scaling = 1.0 / (FA_HEAD_DIM**0.5)
        out = fused.attn_forward_topk(
            q,
            k,
            v,
            softmax_scale=scaling,
            topk_K=kappa,
            exclude_sink_tokens=num_sink,
        )
        topk_scores = out[4][0].float()
        topk_idx = out[5][0].long()
        _, attn_full = ref_sdpa(q, k, v, scaling=scaling, causal=True)
        attn_eligible = attn_full[0, :, kappa - 1 + num_sink :, :]
        gathered = torch.gather(attn_eligible, dim=-1, index=topk_idx)
        torch.testing.assert_close(topk_scores, gathered, atol=2e-2, rtol=5e-2)

    def test_topk_score_sums_in_unit_interval(self):
        torch.manual_seed(0)
        B, S, H_q, H_kv = 1, 256, 8, 8
        q, k, v = make_qkv(B, S, S, H_q, H_kv, FA_HEAD_DIM, device="cuda")
        scaling = 1.0 / (FA_HEAD_DIM**0.5)
        topk_scores = fused.attn_forward_topk(
            q,
            k,
            v,
            softmax_scale=scaling,
            topk_K=8,
            exclude_sink_tokens=0,
        )[4]
        sums = topk_scores.float().sum(dim=-1)
        assert (sums >= 0.0).all()
        assert (sums <= 1.0 + 1e-3).all()

    # ---- top-K vs reference top-K (smallest supported kappa) ------------- #

    def test_kappa2_matches_reference_top2(self):
        """For kappa=2 (smallest FA3-supported K) the index set per row must
        equal the reference top-2 set, ignoring intra-row order. Random
        q/k/v can produce bf16 ties, so we accept ≥95% — the deterministic
        100% counterpart lives in :meth:`test_kappa_constructed_inputs_100pct`."""
        torch.manual_seed(0)
        B, S, H_q, H_kv = 1, 64, 4, 4
        kappa, num_sink = 2, 0
        q, k, v = make_qkv(B, S, S, H_q, H_kv, FA_HEAD_DIM, device="cuda")
        scaling = 1.0 / (FA_HEAD_DIM**0.5)
        topk_idx = fused.attn_forward_topk(
            q,
            k,
            v,
            softmax_scale=scaling,
            topk_K=kappa,
            exclude_sink_tokens=num_sink,
        )[5][0].long()
        _, attn_full = ref_sdpa(q, k, v, scaling=scaling, causal=True)
        attn_eligible = attn_full[0, :, kappa - 1 + num_sink :, :]
        ref_top = attn_eligible.topk(kappa, dim=-1).indices
        actual_sorted, _ = topk_idx.sort(dim=-1)
        ref_sorted, _ = ref_top.sort(dim=-1)
        match = (actual_sorted == ref_sorted).all(dim=-1).float().mean().item()
        assert match > 0.95, f"kappa=2 topk set matched only {match:.1%} of rows; expected >95%"

    @pytest.mark.parametrize("kappa,num_sink", [(2, 0), (4, 0), (8, 0), (4, 4), (8, 4)])
    def test_kappa_constructed_inputs_100pct(self, kappa, num_sink):
        """Top-K must be 100%-correct on inputs that are constructed so bf16
        rounding **cannot** cause ties.

        Construction:
          * ``k[b, s, h, :]`` is the s-th basis vector (zeros, ``1.0`` at
            position s). So the dot product ``q · k_s == q[..., s]``.
          * ``q[b, q_pos, h, :S]`` is a random permutation of
            ``[0, 1, ..., S-1]`` (different permutation per row). Higher
            value -> higher attention score.
          * ``softmax_scale=1.0``.

        Adjacent ranks differ by 1.0 in score (>> bf16 ulp), so the top-K
        ordering is unambiguous post-bf16-rounding and post-softmax.
        Random row-wise permutation prevents trivial-pass cases (e.g. the
        kernel always picking the last K positions)."""
        torch.manual_seed(0)
        B, S, H_q, H_kv = 1, 64, 4, 4
        device = "cuda"

        # k = basis vectors. k[b, s, h, d] = 1.0 if d == s else 0.0.
        # Magnitude doesn't matter for the test — what matters is that
        # q · k_s = q[..., s] independently of s.
        k = torch.zeros(B, S, H_kv, FA_HEAD_DIM, dtype=torch.bfloat16, device=device)
        arange_S = torch.arange(S, device=device)
        k[:, arange_S, :, arange_S] = 1.0
        v = torch.randn(B, S, H_kv, FA_HEAD_DIM, dtype=torch.bfloat16, device=device)

        # q: per (b, q_pos, h) a random permutation of [0..S-1] in the
        # first S components, zero elsewhere.
        q = torch.zeros(B, S, H_q, FA_HEAD_DIM, dtype=torch.bfloat16, device=device)
        rank_values = torch.arange(S, dtype=torch.bfloat16, device=device)
        for b in range(B):
            for q_pos in range(S):
                for hq in range(H_q):
                    perm = torch.randperm(S, device=device)
                    q[b, q_pos, hq, :S] = rank_values[perm]

        scaling = 1.0  # keep integer gaps post-scaling
        out = fused.attn_forward_topk(
            q,
            k,
            v,
            softmax_scale=scaling,
            topk_K=kappa,
            exclude_sink_tokens=num_sink,
        )
        topk_idx = out[5][0].long()

        # Reference: top-K via fp32 SDPA. With our construction the scores
        # are unambiguous integers so the reference top-K is deterministic.
        _, attn_full = ref_sdpa(q, k, v, scaling=scaling, causal=True)
        attn_eligible = attn_full[0, :, kappa - 1 + num_sink :, :]
        # Mask out sinks before topk so the reference matches the kernel's
        # exclude_sink_tokens behavior.
        if num_sink > 0:
            attn_eligible = attn_eligible.clone()
            attn_eligible[..., :num_sink] = float("-inf")
        ref_top = attn_eligible.topk(kappa, dim=-1).indices

        actual_sorted, _ = topk_idx.sort(dim=-1)
        ref_sorted, _ = ref_top.sort(dim=-1)
        assert torch.equal(actual_sorted, ref_sorted), (
            "100% top-K match expected with uniform-gap constructed inputs; "
            f"got mismatch on kappa={kappa}, num_sink={num_sink}"
        )

    # ---- return_topk=False (plain-FA degrade) ---------------------------- #

    def test_return_topk_false_matches_topk_true_output(self):
        """The attention ``out`` is independent of whether the top-K probe
        ran — the probe is a side channel, not part of the AV accumulation.
        So ``return_topk=False`` must produce a bit-for-bit identical ``out``
        to ``return_topk=True`` on the same inputs."""
        torch.manual_seed(0)
        B, S, H_q, H_kv = 1, 256, 8, 8
        q, k, v = make_qkv(B, S, S, H_q, H_kv, FA_HEAD_DIM, device="cuda")
        scaling = 1.0 / (FA_HEAD_DIM**0.5)
        out_probe = fused.attn_forward_topk(
            q, k, v, softmax_scale=scaling, topk_K=8, exclude_sink_tokens=4, return_topk=True
        )[0]
        out_plain = fused.attn_forward_topk(
            q, k, v, softmax_scale=scaling, topk_K=8, exclude_sink_tokens=4, return_topk=False
        )[0]
        assert torch.equal(
            out_plain, out_probe
        ), "out must be identical whether or not the top-K probe ran"

    def test_return_topk_false_empty_topk_placeholders(self):
        """When the probe is gated off, the kernel still returns a 6-tuple,
        but the score/index slots are empty placeholders — no
        ``(B, H_q, S_elig, K)`` buffers are allocated."""
        torch.manual_seed(0)
        B, S, H_q, H_kv = 1, 128, 8, 8
        q, k, v = make_qkv(B, S, S, H_q, H_kv, FA_HEAD_DIM, device="cuda")
        scaling = 1.0 / (FA_HEAD_DIM**0.5)
        out = fused.attn_forward_topk(
            q, k, v, softmax_scale=scaling, topk_K=8, exclude_sink_tokens=4, return_topk=False
        )
        assert out[0].shape == (B, S, H_q, FA_HEAD_DIM)  # attention output still valid
        assert out[4].numel() == 0, "topk_scores must be an empty placeholder"
        assert out[5].numel() == 0, "topk_indices must be an empty placeholder"

    def test_return_topk_false_accepts_unsupported_topk_K(self):
        """The ``topk_K in {2,4,8,16,32}`` runtime check lives inside the
        probe's ``if (return_topk)`` block. With the probe gated off,
        ``topk_K`` is ignored, so an unsupported value like 7 must run
        cleanly (whereas ``return_topk=True`` raises on it)."""
        torch.manual_seed(0)
        B, S, H_q, H_kv = 1, 128, 8, 8
        q, k, v = make_qkv(B, S, S, H_q, H_kv, FA_HEAD_DIM, device="cuda")
        scaling = 1.0 / (FA_HEAD_DIM**0.5)
        # Sanity: with the probe ON, topk_K=7 is rejected by FA3.
        with pytest.raises(Exception):
            fused.attn_forward_topk(
                q, k, v, softmax_scale=scaling, topk_K=7, exclude_sink_tokens=0, return_topk=True
            )
        # With the probe OFF, the same unsupported topk_K is ignored.
        out = fused.attn_forward_topk(
            q, k, v, softmax_scale=scaling, topk_K=7, exclude_sink_tokens=0, return_topk=False
        )
        assert out[0].shape == (B, S, H_q, FA_HEAD_DIM)
        assert out[5].numel() == 0

    # ---- GQA ratios ------------------------------------------------------ #

    @pytest.mark.parametrize("H_q,H_kv", [(8, 8), (8, 2)])  # MHA, 4:1 GQA
    def test_gqa_shape_and_correctness(self, H_q, H_kv):
        torch.manual_seed(0)
        B, S = 1, 128
        q, k, v = make_qkv(B, S, S, H_q, H_kv, FA_HEAD_DIM, device="cuda")
        scaling = 1.0 / (FA_HEAD_DIM**0.5)
        out = fused.attn_forward_topk(
            q,
            k,
            v,
            softmax_scale=scaling,
            topk_K=4,
            exclude_sink_tokens=0,
        )
        assert out[0].shape == (B, S, H_q, FA_HEAD_DIM)
        assert out[5].shape == (B, H_q, S - 3, 4)
        out_ref, _ = ref_sdpa(q, k, v, scaling=scaling, causal=True)
        torch.testing.assert_close(out[0].float(), out_ref, atol=5e-2, rtol=5e-2)
