"""Shared pytest fixtures + reusable test helpers. CPU-only — GPU
integration tests are gated behind the skip markers below and run on
hardware.

Most helpers are plain callables, imported directly by the tests that
need them (``make_qkv``, ``ref_sdpa``, ``make_test_runtime``,
``append_decode_step``, ...). A few are also exposed as pytest fixtures
(``make_topk``, ``layer_graph``, ``prefilled_runtime``,
``fake_fused_attn_fwd_topk``) where fixture injection reads more naturally.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

# --------------------------------------------------------------------------- #
# Skip markers
# --------------------------------------------------------------------------- #

CUDA_AVAILABLE = torch.cuda.is_available()
HAS_FUSED_ATTN_FWD_TOPK = False
if CUDA_AVAILABLE:
    try:
        import flash_attn_interface  # noqa: F401  (upstream-named module)

        HAS_FUSED_ATTN_FWD_TOPK = True
    except ImportError:
        pass

HAS_LEIDEN = False
if CUDA_AVAILABLE:
    try:
        from community_kv.graph._leiden import _community_kv_leiden  # noqa: F401

        HAS_LEIDEN = True
    except ImportError:
        pass

# Skip marker for tests that need the patched FlashAttention forward
# (the fused top-K probe lives in the hopper extension, not in stock FA).
FUSED_ATTN_FWD_TOPK_REQUIRED = pytest.mark.skipif(
    not (CUDA_AVAILABLE and HAS_FUSED_ATTN_FWD_TOPK),
    reason="needs CUDA + patched fused-attn-fwd-topk (flash-attention v2.8.3 + topk patch) build",
)

# Skip marker for tests that exercise the compiled Leiden kernel. The .so is
# lazy-imported inside ``run_leiden`` (via ``_leiden._load_module``), so a test
# can reach it without CUDA being enough — it also needs the extension built.
LEIDEN_REQUIRED = pytest.mark.skipif(
    not (CUDA_AVAILABLE and HAS_LEIDEN),
    reason="needs CUDA + the compiled native Leiden extension (community_kv.graph._leiden)",
)

# CommunityKV-attention forward tests need both: prefill submits a partition
# job to the worker pool that imports the Leiden .so on the worker thread.
FUSED_ATTN_FWD_TOPK_AND_LEIDEN_REQUIRED = pytest.mark.skipif(
    not (CUDA_AVAILABLE and HAS_FUSED_ATTN_FWD_TOPK and HAS_LEIDEN),
    reason="needs CUDA + fused-attn-fwd-topk build + native Leiden extension",
)


# --------------------------------------------------------------------------- #
# Constants for FA3 topk-probe constraints
# --------------------------------------------------------------------------- #

# The FA3 topk probe is compiled only for head_dim_v == 128.
FA_HEAD_DIM = 128


# --------------------------------------------------------------------------- #
# Plain-callable helpers — direct use inside tests
# --------------------------------------------------------------------------- #


def make_topk_indices(
    H_q: int,
    S_eligible: int,
    kappa: int,
    init_q_start: int,
    *,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize valid causal top-kappa data for partition() unit tests.

    For query at absolute pos ``p = init_q_start + r``, picks the kappa most
    recent valid keys [p, p-1, ..., p-kappa+1]. Scores uniform random.
    Returns (topk_indices, topk_scores), both shape (H_q, S_eligible, kappa).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    topk_indices = torch.zeros(H_q, S_eligible, kappa, dtype=torch.int32, device=device)
    for r in range(S_eligible):
        p = init_q_start + r
        for j in range(kappa):
            topk_indices[:, r, j] = p - j
    topk_scores = torch.rand(H_q, S_eligible, kappa, generator=gen, device=device) * 0.5 + 0.1
    return topk_indices, topk_scores


def make_layer_graph(
    G: int = 2,
    S: int = 8,
    max_C: int = 4,
    D: int = 16,
    num_centroid_heads: int | None = None,
    decode_capacity: int = 4,
    device: torch.device | str = "cpu",
):
    """Build a minimal LayerGraph with sane shapes for unit tests."""
    from community_kv.graph.runtime import LayerGraph
    from community_kv.graph.state import GraphAggregation

    nch = num_centroid_heads or G
    return LayerGraph(
        layer_idx=0,
        aggregation=GraphAggregation.PER_QUERY_HEAD,
        num_kv_heads_local=G,
        prefill_seq_len=S,
        head_dim=D,
        device=torch.device(device),
        community_ids=torch.zeros(G, S, dtype=torch.int32, device=device),
        num_communities=torch.full((G,), max_C, dtype=torch.int32, device=device),
        centroids=torch.zeros(nch, max_C, D, device=device),
        community_sizes=torch.zeros(nch, max_C, device=device),
        community_sizes_prefill=torch.zeros(nch, max_C, device=device),
        member_offsets=torch.zeros(G, max_C + 1, dtype=torch.int32, device=device),
        member_positions=torch.zeros(G, S, dtype=torch.int32, device=device),
        community_weight=torch.zeros(G, max_C, device=device),
        total_weight=torch.zeros(G, device=device),
        decode_log_position=torch.full(
            (decode_capacity,),
            -1,
            dtype=torch.int32,
            device=device,
        ),
        decode_log_community=torch.full(
            (G, decode_capacity),
            -1,
            dtype=torch.int32,
            device=device,
        ),
    )


def make_qkv(
    B: int,
    S_q: int,
    S_k: int,
    H_q: int,
    H_kv: int,
    D: int = FA_HEAD_DIM,
    *,
    device: torch.device | str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random bf16 q / k / v in FA's expected (B, S, H, D) layout.

    bf16 + ``D = FA_HEAD_DIM = 128`` are the only configurations the FA3
    topk probe is compiled for — the kernel is gated on those at compile
    time.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, S_q, H_q, D, dtype=torch.bfloat16, device=device, generator=g)
    k = torch.randn(B, S_k, H_kv, D, dtype=torch.bfloat16, device=device, generator=g)
    v = torch.randn(B, S_k, H_kv, D, dtype=torch.bfloat16, device=device, generator=g)
    return q, k, v


def ref_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scaling: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 SDPA reference for FA-style GQA.

    Args (FA layout): q (B, S_q, H_q, D), k/v (B, S_k, H_kv, D), bf16.
    Returns (out (B, S_q, H_q, D) fp32, attn (B, H_q, S_q, S_k) fp32).
    """
    B, S_q, H_q, D = q.shape
    _, S_k, H_kv, _ = k.shape
    repeat = H_q // H_kv
    k_b = k.repeat_interleave(repeat, dim=2).float()
    v_b = v.repeat_interleave(repeat, dim=2).float()
    qf = q.float()
    scores = torch.einsum("bshd,bxhd->bhsx", qf, k_b) * scaling
    if causal:
        mask = torch.ones(S_q, S_k, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = scores.softmax(dim=-1)
    out = torch.einsum("bhsx,bxhd->bshd", attn, v_b)
    return out, attn


def fa3_abs_q_positions(
    S_eligible: int,
    kappa: int,
    num_sink: int,
    *,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Absolute query position for each row of FA3's topk output.

    The patch returns ``(B, H, S_eligible, K)`` — the first
    ``kappa - 1 + num_sink`` query positions are skipped because they
    lack enough causal-valid keys to fill a top-K row.
    """
    return torch.arange(S_eligible, device=device, dtype=torch.long) + (kappa - 1 + num_sink)


# --------------------------------------------------------------------------- #
# Helpers for community_kv attention forward tests
# --------------------------------------------------------------------------- #


class FakeAttnModule:
    """Minimal stand-in for the HF attention module — only ``layer_idx``
    is read by ``community_kv_attention_forward``."""

    def __init__(self, layer_idx: int = 0):
        self.layer_idx = layer_idx


def make_test_runtime(
    *,
    kappa: int = 8,
    num_sink: int = 4,
    lam: float = 0.5,
    max_new_tokens: int = 4,
    token_budget: int = 64,
    aggregation=None,
    leiden_resolution: float = 1.0,
    leiden_max_iter: int = 2,
    max_workers: int = 2,
):
    """Build a ``GraphRuntime`` pre-configured for unit tests, with a
    ThreadPoolExecutor wired up. ``aggregation`` defaults to
    PER_QUERY_HEAD; pass any ``GraphAggregation`` member to override.

    Caller is responsible for ``graph_runtime.shutdown()`` at end of test (or in a
    pytest fixture's teardown)."""
    from concurrent.futures import ThreadPoolExecutor

    from community_kv.graph.runtime import GraphRuntime
    from community_kv.graph.state import GraphAggregation

    graph_runtime = GraphRuntime()
    graph_runtime.config.update(
        kappa=kappa,
        num_sink=num_sink,
        lam=lam,
        leiden_resolution=leiden_resolution,
        leiden_max_iter=leiden_max_iter,
        max_new_tokens=max_new_tokens,
        token_budget=token_budget,
    )
    graph_runtime.aggregation = (
        aggregation if aggregation is not None else GraphAggregation.PER_QUERY_HEAD
    )
    graph_runtime.executor = ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="test-part"
    )
    return graph_runtime


def bshd_to_bhsd(t: torch.Tensor) -> torch.Tensor:
    """FA layout (B, S, H, D) -> module forward layout (B, H, S, D).

    The HF attention module dispatcher passes ``(B, H, S, D)``-shaped
    tensors; FA's wrappers expect ``(B, S, H, D)``. This helper keeps
    test setup readable when we hand-construct FA-shaped tensors and
    feed them through the module forward path.
    """
    return t.transpose(1, 2).contiguous()


def drain_runtime(graph_runtime) -> None:
    """Block until every async-partition future submitted during a forward
    pass completes, then clear the list. Use after a prefill in a test."""
    for fut in graph_runtime.futures:
        fut.result()
    graph_runtime.futures.clear()


def run_prefill_and_drain(
    graph_runtime,
    *,
    B: int = 1,
    S: int = 128,
    H_q: int = 8,
    H_kv: int = 8,
    layer_idx: int = 0,
    seed: int = 0,
):
    """Run one prefill forward through ``community_kv_attention_forward``
    and drain the partition workers. Returns ``(k, v)`` BSHD tensors so
    subsequent decode steps can append fresh KV.

    Centralises the boilerplate that was duplicated across ~10 decode
    tests in ``test_community_kv.py``."""
    from community_kv.attention.community_kv import community_kv_attention_forward

    torch.manual_seed(seed)
    q, k, v = make_qkv(B, S, S, H_q, H_kv, FA_HEAD_DIM, device="cuda")
    community_kv_attention_forward(
        module=FakeAttnModule(layer_idx=layer_idx),
        query=bshd_to_bhsd(q),
        key=bshd_to_bhsd(k),
        value=bshd_to_bhsd(v),
        attention_mask=None,
        scaling=1.0 / (FA_HEAD_DIM**0.5),
        graph_runtime=graph_runtime,
    )
    drain_runtime(graph_runtime)
    return k, v


def append_decode_step(
    graph_runtime,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    *,
    H_q: int = 8,
    H_kv: int = 8,
    layer_idx: int = 0,
):
    """Run one decode step with random fresh KV against a runtime that has
    already been prefilled. Returns ``(out, k_full, v_full)`` where the
    cache tensors are extended by one position."""
    from community_kv.attention.community_kv import community_kv_attention_forward

    B = k_full.shape[0]
    q_dec = torch.randn(B, 1, H_q, FA_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k_full = torch.cat(
        [k_full, torch.randn(B, 1, H_kv, FA_HEAD_DIM, dtype=torch.bfloat16, device="cuda")],
        dim=1,
    )
    v_full = torch.cat(
        [v_full, torch.randn(B, 1, H_kv, FA_HEAD_DIM, dtype=torch.bfloat16, device="cuda")],
        dim=1,
    )
    out, _ = community_kv_attention_forward(
        module=FakeAttnModule(layer_idx=layer_idx),
        query=bshd_to_bhsd(q_dec),
        key=bshd_to_bhsd(k_full),
        value=bshd_to_bhsd(v_full),
        attention_mask=None,
        scaling=1.0 / (FA_HEAD_DIM**0.5),
        graph_runtime=graph_runtime,
    )
    torch.cuda.synchronize()
    return out, k_full, v_full


# --------------------------------------------------------------------------- #
# Pytest fixtures wrapping the helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def make_topk():
    return make_topk_indices


@pytest.fixture
def layer_graph():
    return make_layer_graph


@pytest.fixture
def prefilled_runtime():
    """Factory + auto-cleanup. Yields a callable
    ``(*, S=128, H_q=8, H_kv=8, **runtime_kwargs) -> (graph_runtime, k, v)``
    that builds a ``GraphRuntime``, runs one prefill forward, drains the
    partition workers, and tracks the runtime for shutdown on teardown.

    Use in decode-path tests to skip ~15 lines of prefill boilerplate:

        def test_my_decode_thing(self, prefilled_runtime):
            gr, k, v = prefilled_runtime(token_budget=4096)
            out, k_full, v_full = append_decode_step(gr, k, v)
            ...
    """
    runtimes: list = []

    def factory(*, S: int = 128, H_q: int = 8, H_kv: int = 8, seed: int = 0, **runtime_kwargs):
        gr = make_test_runtime(**runtime_kwargs)
        runtimes.append(gr)
        k, v = run_prefill_and_drain(gr, S=S, H_q=H_q, H_kv=H_kv, seed=seed)
        return gr, k, v

    yield factory
    for gr in runtimes:
        gr.shutdown()


# --------------------------------------------------------------------------- #
# Mock flash-attention for wrapper-pinning tests
# --------------------------------------------------------------------------- #


class FakeFusedAttn:
    """Records the kwargs passed to the upstream ``_flash_attn_forward``
    so wrapper-pinning tests can assert on what we forward.

    The fake returns a 6-tuple matching the upstream call's shape:
    ``(out, ?, ?, ?, topk_scores, topk_indices)``.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def _flash_attn_forward(self, **kwargs):
        self.calls.append(kwargs)
        return ("attn_output", None, None, None, "topk_scores", "topk_indices")

    def flash_attn_func(self, *args, **kwargs):
        return None


@pytest.fixture
def fake_fused_attn_fwd_topk(monkeypatch):
    """Replace upstream ``flash_attn_interface`` in ``sys.modules`` with a
    recorder fake. Yields the fake so the test can assert on captured kwargs."""
    fa = FakeFusedAttn()
    fake_module = types.SimpleNamespace(
        _flash_attn_forward=fa._flash_attn_forward,
        flash_attn_func=fa.flash_attn_func,
    )
    monkeypatch.setitem(sys.modules, "flash_attn_interface", fake_module)
    return fa


__all__ = [
    "CUDA_AVAILABLE",
    "FA_HEAD_DIM",
    "FA_VALID_KAPPAS",
    "FUSED_ATTN_FWD_TOPK_AND_LEIDEN_REQUIRED",
    "FUSED_ATTN_FWD_TOPK_REQUIRED",
    "FakeAttnModule",
    "FakeFusedAttn",
    "HAS_FUSED_ATTN_FWD_TOPK",
    "HAS_LEIDEN",
    "LEIDEN_REQUIRED",
    "bshd_to_bhsd",
    "drain_runtime",
    "fa3_abs_q_positions",
    "make_layer_graph",
    "make_qkv",
    "make_test_runtime",
    "make_topk_indices",
    "ref_sdpa",
]
