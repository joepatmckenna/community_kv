"""Bindings to the patched FlashAttention forward (with the row-wise top-K
probe).

Wraps the long upstream ``_flash_attn_forward`` argument list in
``attn_forward_topk`` — the single call shape we use, with the top-K
probe turned on. The compiled extension lives at
``third_party/flash-attention/hopper`` and is exposed under the upstream
module name ``flash_attn_interface`` by FlashAttention's own installer.
"""

from __future__ import annotations

import torch


def _import_upstream_fused_attn_fwd_topk():
    """Lazy import — keeps this module importable on a CPU dev machine.

    Returns ``(fused_attn_fwd_topk_impl, fused_attn_fwd_topk_func)``,
    aliased references to the upstream ``_flash_attn_forward`` and
    ``flash_attn_func`` symbols.
    """
    try:
        from flash_attn_interface import (
            _flash_attn_forward as fused_attn_fwd_topk_impl,
            flash_attn_func as fused_attn_fwd_topk_func,
        )
    except ImportError as e:
        raise ImportError(
            "community_kv requires the patched fused-attn-fwd-topk build "
            "(flash-attention v2.8.3 + the top-K patch) to be installed. "
            "From the package root, run `pip install --no-build-isolation "
            "-e .` (or follow the README's manual install steps). The "
            "submodule + patch live under third_party/."
        ) from e
    return fused_attn_fwd_topk_impl, fused_attn_fwd_topk_func


def attn_forward_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float,
    topk_K: int,
    exclude_sink_tokens: int,
    causal: bool = True,
    pack_gqa: bool = True,
    return_topk: bool = True,
):
    """Fused attention forward with the top-K probe extension enabled.

    Hides the ~25 always-None upstream args, lazy-imports on first call,
    and pins the topk knobs we use. Returns the upstream tuple; callers
    index out::

        out = result[0]              # (B, S_q, H_q, D)
        topk_scores = result[4]      # (B, H_q, S_eligible, K)  fp32  (only when return_topk)
        topk_indices = result[5]     # (B, H_q, S_eligible, K)  int32 (only when return_topk)

    When ``return_topk=False``, the kernel skips the in-kernel top-K
    dispatch (the patch's ``if (return_topk) {...}`` validation block is
    bypassed and the row-wise probe is gated off in the mainloop), so
    the call degrades to plain FA. ``topk_K`` and ``exclude_sink_tokens``
    are ignored in that case but must still be passed (the kernel
    accepts them positionally).
    """
    fused_attn_fwd_topk_impl, _ = _import_upstream_fused_attn_fwd_topk()
    return fused_attn_fwd_topk_impl(
        q=q,
        k=k,
        v=v,
        k_new=None,
        v_new=None,
        qv=None,
        out=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        cu_seqlens_k_new=None,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        page_table=None,
        kv_batch_idx=None,
        leftpad_k=None,
        rotary_cos=None,
        rotary_sin=None,
        seqlens_rotary=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=pack_gqa,
        sm_margin=0,
        return_topk=return_topk,
        topk_K=topk_K,
        exclude_sink_tokens=exclude_sink_tokens,
    )


__all__ = ["attn_forward_topk"]
