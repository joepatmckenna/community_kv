"""Regression tests for evals.datasets.longbench_v2 (LongBench-specific)."""

import pytest

from evals.datasets.longbench_v2 import LongBenchV2Dataset


class TestRenderPrompt:
    @pytest.fixture
    def ds(self):
        return LongBenchV2Dataset(split="short")

    def test_substitutes_all_fields(self, ds):
        sample = {
            "context": "ctx",
            "question": "Q?",
            "choice_A": "Apple",
            "choice_B": "Banana",
            "choice_C": "Cherry",
            "choice_D": "Date",
        }
        prompt = ds.render_prompt(sample)
        assert "ctx" in prompt
        assert "Q?" in prompt
        assert "Apple" in prompt and "Banana" in prompt
        assert "Cherry" in prompt and "Date" in prompt
        assert "$DOC$" not in prompt
        assert "$Q$" not in prompt
        assert "$C_A$" not in prompt

    def test_missing_fields_are_empty(self, ds):
        prompt = ds.render_prompt({})
        assert "$DOC$" not in prompt

    def test_template_matches_official_shape(self, ds):
        prompt = ds.render_prompt(
            {
                "context": "X",
                "question": "Q",
                "choice_A": "a",
                "choice_B": "b",
                "choice_C": "c",
                "choice_D": "d",
            }
        )
        assert "<text>" in prompt and "</text>" in prompt
        assert "(A) a" in prompt and "(D) d" in prompt
        assert (
            'Format your response as follows: "The correct answer is (insert answer here)".'
            in prompt
        )


class TestExtractAnswer:
    """Mirrors THUDM/LongBench pred.py extract_answer — strict two-pattern match."""

    @pytest.fixture
    def dataset(self):
        return LongBenchV2Dataset(split="short")

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("The correct answer is (A)", "A"),
            ("The correct answer is (D).", "D"),
            ("Sure. The correct answer is (B) because ...", "B"),
            ("The correct answer is C", "C"),
            ("The correct answer is C.", "C"),
            ("The correct answer is **A**", "A"),
            ("**The correct answer is (B)**", "B"),
        ],
    )
    def test_official_patterns(self, dataset, text, expected):
        assert dataset.extract_answer(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "Answer: A",
            "(A)",
            "A.",
            r"\boxed{A}",
            "I think it's A.",
            "",
            "The answer is A",  # missing 'correct'
        ],
    )
    def test_does_not_match_loose_forms(self, dataset, text):
        # Official extractor only matches the canonical "The correct answer is …" phrase.
        assert dataset.extract_answer(text) is None

    def test_paren_pattern_takes_precedence(self, dataset):
        # If both forms could match, the parens form wins (it's tried first).
        text = "The correct answer is (A) and not B"
        assert dataset.extract_answer(text) == "A"


class TestFitSample:
    """Truncation cache: short prompts pass through, long ones get cached."""

    class _FakeTok:
        """Whitespace-token stand-in: encode = split, decode = join."""

        def encode(self, s):
            return s.split()

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(ids)

    def _sample(self, ctx_words):
        return {
            "context": " ".join(f"w{i}" for i in range(ctx_words)),
            "question": "Q",
            "choice_A": "a",
            "choice_B": "b",
            "choice_C": "c",
            "choice_D": "d",
            "answer": "A",
            "_id": "x",
        }

    def test_no_truncation_when_fits(self):
        ds = LongBenchV2Dataset(split="short")
        sample = self._sample(ctx_words=5)
        assert ds.fit_sample(sample, self._FakeTok(), max_len=1000) is False
        # No cached prompt; format_prompt re-renders.
        assert ds._PROMPT_CACHE_KEY not in sample

    def test_truncation_caches_prompt(self):
        ds = LongBenchV2Dataset(split="short")
        sample = self._sample(ctx_words=2000)
        truncated = ds.fit_sample(sample, self._FakeTok(), max_len=20)
        assert truncated is True
        cached = sample[ds._PROMPT_CACHE_KEY]
        assert ds.format_prompt(sample) == cached
        # Cached form is shorter than the original render.
        assert len(cached.split()) <= 20
