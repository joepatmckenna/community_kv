"""Tests for evals.utils — dataset-agnostic helpers (rope math, truncation,
chat-template tokenization)."""

import pytest

from evals.utils import (
    DEFAULT_MAX_ROPE_FACTOR,
    compute_rope_factor,
)


class TestComputeRopeFactor:
    @pytest.mark.parametrize(
        "n_tokens,native_max,expected",
        [
            (0, 40960, 1.0),
            (1, 40960, 1.0),
            (10000, 40960, 1.0),
            (40960, 40960, 1.0),  # exactly fits
            (40961, 40960, 2.0),
            (81920, 40960, 2.0),  # exactly fits at factor=2
            (81921, 40960, 4.0),
            (300000, 40960, 8.0),  # 8 * 40960 = 327_680 >= 300_000
            (1000000, 40960, 32.0),
        ],
    )
    def test_unbounded(self, n_tokens, native_max, expected):
        assert compute_rope_factor(n_tokens, native_max) == expected

    def test_capped_below_natural_returns_natural(self):
        # natural factor 4, cap 8 -> still 4
        assert compute_rope_factor(81921, 40960, max_rope_factor=8.0) == 4.0

    def test_capped_above_natural_caps(self):
        # natural would be 32, cap at default 4 -> 4
        capped = compute_rope_factor(1_000_000, 40960, max_rope_factor=DEFAULT_MAX_ROPE_FACTOR)
        assert capped == 4.0

    def test_default_cap_value(self):
        assert DEFAULT_MAX_ROPE_FACTOR == 4.0
