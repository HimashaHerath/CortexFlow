"""Tests for cortexflow.metrics_utils."""

from __future__ import annotations

from cortexflow.metrics_utils import (
    calculate_benchmark_metrics,
    calculate_f1,
    calculate_mrr,
    calculate_path_accuracy,
    calculate_precision,
    calculate_recall,
    evaluate_hop_accuracy,
    evaluate_llm_answer,
    extract_entities,
    normalize_text,
)


class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("Hello World") == "hello world"

    def test_removes_punctuation(self):
        assert normalize_text("hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_text("  hello   world  ") == "hello world"

    def test_empty_string(self):
        assert normalize_text("") == ""


class TestExtractEntities:
    def test_removes_stop_words(self):
        entities = extract_entities("the cat and the dog")
        assert "the" not in entities
        assert "and" not in entities
        assert "cat" in entities
        assert "dog" in entities

    def test_includes_bigrams(self):
        entities = extract_entities("new york city")
        assert "new york" in entities
        assert "york city" in entities

    def test_single_char_tokens_removed(self):
        entities = extract_entities("I am a cat")
        assert "i" not in entities


class TestCalculatePrecision:
    def test_all_relevant(self):
        assert calculate_precision(["cat"], ["the cat sat"]) == 1.0

    def test_none_relevant(self):
        assert calculate_precision(["cat"], ["the dog sat"]) == 0.0

    def test_empty_retrieved(self):
        assert calculate_precision(["cat"], []) == 0.0


class TestCalculateRecall:
    def test_all_found(self):
        assert calculate_recall(["cat"], ["the cat sat"]) == 1.0

    def test_none_found(self):
        assert calculate_recall(["cat"], ["the dog sat"]) == 0.0

    def test_empty_expected(self):
        assert calculate_recall([], ["anything"]) == 1.0


class TestCalculateF1:
    def test_perfect_scores(self):
        assert calculate_f1(1.0, 1.0) == 1.0

    def test_zero_scores(self):
        assert calculate_f1(0.0, 0.0) == 0.0

    def test_harmonic_mean(self):
        f1 = calculate_f1(0.5, 1.0)
        assert abs(f1 - 2 / 3) < 1e-9


class TestCalculateMRR:
    def test_first_result_relevant(self):
        assert calculate_mrr(["cat"], ["cat is here", "dog is there"]) == 1.0

    def test_second_result_relevant(self):
        assert calculate_mrr(["cat"], ["dog is there", "cat is here"]) == 0.5

    def test_no_relevant_results(self):
        assert calculate_mrr(["cat"], ["dog", "fish"]) == 0.0

    def test_empty_inputs(self):
        assert calculate_mrr([], ["anything"]) == 0.0
        assert calculate_mrr(["cat"], []) == 0.0


class TestCalculatePathAccuracy:
    def test_empty_inputs(self):
        assert calculate_path_accuracy([], "anything") == 0.0
        assert calculate_path_accuracy(["a"], "") == 0.0

    def test_perfect_match(self):
        score = calculate_path_accuracy(["A", "B"], "A → B")
        assert score > 0.0


class TestEvaluateHopAccuracy:
    def test_exact_match(self):
        assert evaluate_hop_accuracy(3, 3) == 1.0

    def test_both_zero(self):
        assert evaluate_hop_accuracy(0, 0) == 1.0

    def test_expected_zero_actual_nonzero(self):
        assert evaluate_hop_accuracy(0, 5) == 0.0

    def test_ratio(self):
        assert evaluate_hop_accuracy(4, 2) == 0.5


class TestEvaluateLLMAnswer:
    def test_all_entities_found(self):
        score = evaluate_llm_answer("The cat sat on the mat", ["cat", "mat"])
        assert score == 1.0

    def test_no_entities_found(self):
        score = evaluate_llm_answer("The dog ran", ["cat", "mat"])
        assert score == 0.0

    def test_empty_inputs(self):
        assert evaluate_llm_answer("", ["cat"]) == 0.0
        assert evaluate_llm_answer("hello", []) == 0.0


class TestCalculateBenchmarkMetrics:
    def test_returns_all_keys(self):
        results = [{"text": "the cat sat"}]
        expected = {"expected_entities": ["cat"]}
        metrics = calculate_benchmark_metrics(results, expected)
        assert set(metrics.keys()) == {"precision", "recall", "f1", "mrr"}

    def test_perfect_retrieval(self):
        results = [{"text": "cat and dog"}]
        expected = {"expected_entities": ["cat", "dog"]}
        metrics = calculate_benchmark_metrics(results, expected)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
