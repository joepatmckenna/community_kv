"""Regression tests for evals.datasets.babilong."""

import pytest

from evals.datasets.babilong import BabilongDataset


class TestRenderPrompt:
    def test_assembles_template(self):
        ds = BabilongDataset(task="qa1", length="0k")
        sample = {
            "input": "John went to the bedroom. ",
            "question": "Where is John?",
            "target": "bedroom",
        }
        out = ds.render_prompt(sample)
        # Components from the official template are all present.
        assert "<context>" in out and "</context>" in out
        assert "Question: Where is John?" in out
        assert "John went to the bedroom" in out
        assert "<example>" in out
        # Post-prompt boilerplate is included.
        assert "The most recent location" in out

    def test_unknown_task_raises_at_construction(self):
        # render_prompt is per-instance now; an unknown task can't reach it.
        with pytest.raises(ValueError, match="unknown task"):
            BabilongDataset(task="qa999")


class TestExtractAnswer:
    """Closed-vocab matching with question-label exclusion."""

    @pytest.fixture
    def qa1(self):
        return BabilongDataset(task="qa1", length="0k")

    @pytest.fixture
    def qa6(self):
        return BabilongDataset(task="qa6", length="0k")

    @pytest.fixture
    def qa8(self):
        return BabilongDataset(task="qa8", length="0k")

    def test_qa1_simple_match(self, qa1):
        sample = {"question": "Where is Mary?", "target": "kitchen"}
        # Output has one valid label; question doesn't mention any qa1 label.
        assert (
            qa1.extract_answer("The most recent location of Mary is kitchen.", sample) == "kitchen"
        )

    def test_qa1_question_label_excluded(self, qa1):
        # If the question mentions a label, it doesn't count as a prediction.
        sample = {"question": "Is Mary in the kitchen?", "target": "kitchen"}
        # Question mentions "kitchen", so even though it's in the output it's excluded.
        # Output mentions only "kitchen" (which is in question) -> empty prediction set.
        assert qa1.extract_answer("Yes, the kitchen.", sample) is None

    def test_qa1_multiple_labels_in_output_returns_set(self, qa1):
        sample = {"question": "Where is Mary?", "target": "kitchen"}
        # Both kitchen and bedroom mentioned; pred is the joined set.
        out = qa1.extract_answer("Mary is in the kitchen or the bedroom.", sample)
        assert out == "bedroom,kitchen"

    def test_qa1_no_label_in_output(self, qa1):
        sample = {"question": "Where is Mary?", "target": "kitchen"}
        assert qa1.extract_answer("I don't know.", sample) is None

    def test_qa6_yes(self, qa6):
        sample = {"question": "Is John in the garden?", "target": "yes"}
        assert qa6.extract_answer("yes", sample) == "yes"

    def test_qa8_multi_target_set(self, qa8):
        sample = {"question": "What is Daniel carrying?", "target": "apple,milk"}
        out = qa8.extract_answer("Daniel is carrying the apple and milk.", sample)
        assert out == "apple,milk"

    def test_extract_takes_first_sentence_only(self, qa1):
        sample = {"question": "Where is Mary?", "target": "kitchen"}
        # "bedroom" is in second sentence; preprocess_output drops it.
        out = qa1.extract_answer("Mary is in the kitchen. Then she went to the bedroom.", sample)
        assert out == "kitchen"

    def test_extract_strips_hallucinated_examples(self, qa1):
        sample = {"question": "Where is Mary?", "target": "kitchen"}
        out = qa1.extract_answer("kitchen <example>Mary is in the bedroom</example>", sample)
        # Anything after <example> is dropped.
        assert out == "kitchen"

    def test_returns_none_without_sample(self, qa1):
        # extract_answer needs the sample for question-label exclusion;
        # behavior without a sample is None (better than guessing).
        assert qa1.extract_answer("kitchen", None) is None


class TestGoldCanonical:
    def test_lowercases_and_strips(self):
        ds = BabilongDataset(task="qa1", length="0k")
        assert ds.gold({"target": "  Kitchen  "}) == "kitchen"

    def test_multi_target_sorted(self):
        ds = BabilongDataset(task="qa8", length="0k")
        # Order in the upstream target string is "apple,milk"; check we still
        # canonicalize to sorted order so pred==gold works under set equality.
        assert ds.gold({"target": "milk,apple"}) == "apple,milk"

    def test_pred_equals_gold_for_correct_qa8(self):
        ds = BabilongDataset(task="qa8", length="0k")
        sample = {"question": "What is Daniel carrying?", "target": "milk,apple"}
        pred = ds.extract_answer("Daniel is carrying milk and apple.", sample)
        gold = ds.gold(sample)
        assert pred == gold


class TestRegistryAndArgs:
    def test_registered(self):
        from evals.datasets import DATASET_REGISTRY

        assert DATASET_REGISTRY["babilong"] is BabilongDataset

    def test_default_construction(self):
        ds = BabilongDataset()
        assert ds.task == "qa1"
        assert ds.length == "4k"
        assert ds.samples_repo == "100"

    def test_unknown_task_rejected(self):
        with pytest.raises(ValueError, match="unknown task"):
            BabilongDataset(task="qa999")

    def test_unknown_length_rejected(self):
        with pytest.raises(ValueError, match="unknown length"):
            BabilongDataset(length="999k")

    def test_lookup_name_includes_both_axes(self):
        ds = BabilongDataset(task="qa3", length="64k")
        assert ds.lookup_name() == "babilong:qa3:64k"

    def test_all_20_tasks_loaded(self):
        from evals.datasets import babilong as bl

        assert sorted(bl.TASK_PROMPTS.keys(), key=lambda k: int(k[2:])) == [
            f"qa{i}" for i in range(1, 21)
        ]
        assert sorted(bl.TASK_LABELS.keys(), key=lambda k: int(k[2:])) == [
            f"qa{i}" for i in range(1, 21)
        ]


class TestSampleId:
    def test_stable_across_calls(self):
        ds = BabilongDataset()
        s = {"question": "Where is John?", "input": "John went to the kitchen."}
        assert ds.sample_id(s) == ds.sample_id(s)

    def test_different_samples_different_ids(self):
        ds = BabilongDataset()
        a = {"question": "Where is John?", "input": "John went to the kitchen."}
        b = {"question": "Where is Mary?", "input": "Mary went to the bedroom."}
        assert ds.sample_id(a) != ds.sample_id(b)


class TestFitSample:
    """Same truncation contract as LongBenchV2: caches a truncated prompt."""

    class _FakeTok:
        def encode(self, s):
            return s.split()

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(ids)

    def _sample(self, ctx_words):
        return {
            "input": " ".join(f"w{i}" for i in range(ctx_words)),
            "question": "Where is Mary?",
            "target": "kitchen",
        }

    def test_no_truncation_when_fits(self):
        ds = BabilongDataset(task="qa1", length="0k")
        sample = self._sample(ctx_words=5)
        assert ds.fit_sample(sample, self._FakeTok(), max_len=10000) is False
        assert ds._PROMPT_CACHE_KEY not in sample

    def test_truncation_caches_prompt(self):
        ds = BabilongDataset(task="qa1", length="0k")
        sample = self._sample(ctx_words=2000)
        truncated = ds.fit_sample(sample, self._FakeTok(), max_len=20)
        assert truncated is True
        cached = sample[ds._PROMPT_CACHE_KEY]
        # format_prompt routes through the cache.
        assert ds.format_prompt(sample) == cached
        assert len(cached.split()) <= 20
