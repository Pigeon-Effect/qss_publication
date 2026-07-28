"""Tests for verdict and rating extraction.

Extraction is the one place where a silent bug would corrupt reported accuracy
without any visible error, so the response shapes actually observed in the
archived runs are pinned here.
"""

import pytest

from clustervalidation.parsing import (
    extract_dual_rating,
    extract_rating,
    extract_verdict,
)


class TestExtractVerdict:
    @pytest.mark.parametrize(
        "text,expected,rule",
        [
            ("Final verdict: 3", 3, "final_verdict"),
            ("some reasoning\n\nFinal verdict: 5", 5, "final_verdict"),
            ("Verdict: 2", 2, "verdict"),
            ("Answer: 1", 1, "answer"),
            ("Intruder: 4", 4, "intruder_label"),
            ("The intruder is paper 2", 2, "intruder_prose"),
            ("3", 3, "bare_line"),
        ],
    )
    def test_explicit_markers(self, text, expected, rule):
        result = extract_verdict(text)
        assert result.value == expected
        assert result.rule == rule

    def test_case_insensitive(self):
        assert extract_verdict("FINAL VERDICT: 4").value == 4

    def test_first_pattern_wins(self):
        # An explicit verdict line outranks a stray digit later in the text.
        assert extract_verdict("Final verdict: 2\nsee item 5").value == 2

    def test_falls_back_to_last_digit(self):
        result = extract_verdict("I think it must be number 4 in the end")
        assert result.value == 4
        assert result.rule == "last_digit"
        assert not result.is_explicit

    def test_out_of_range_digits_ignored(self):
        # 7 and 9 exceed the panel size and must not be returned.
        assert extract_verdict("options 7 and 9 aside, verdict: 3").value == 3

    def test_respects_panel_size(self):
        assert extract_verdict("Final verdict: 3", panel_size=3).value == 3
        # 5 is not a valid position in a 3-item panel.
        assert extract_verdict("Final verdict: 5", panel_size=3).value != 5

    @pytest.mark.parametrize("text", ["", None, "no digits here at all"])
    def test_returns_none_when_absent(self, text):
        result = extract_verdict(text)
        assert result.value is None
        assert result.rule == "none"

    def test_explicitness_flag(self):
        assert extract_verdict("Final verdict: 1").is_explicit
        assert not extract_verdict("...ends with 1").is_explicit


class TestExtractRating:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Rating: 4", 4),
            ("The set is coherent.\nRating: 5", 5),
            ("`Rating: 2`", 2),
            ("Score: 3", 3),
        ],
    )
    def test_explicit_ratings(self, text, expected):
        assert extract_rating(text).value == expected

    def test_rejects_out_of_scale(self):
        # 8 is off the Likert scale; the trailing 4 is the real rating.
        assert extract_rating("not 8 but rather 4").value == 4

    def test_missing_rating(self):
        assert extract_rating("no rating given").value is None


class TestExtractDualRating:
    def test_reads_both_scores(self):
        topic, method = extract_dual_rating(
            "Topic coherence: 4\nMethodology coherence: 2"
        )
        assert topic.value == 4
        assert method.value == 2

    def test_accepts_method_abbreviation(self):
        _, method = extract_dual_rating("Topic coherence: 3\nMethod coherence: 5")
        assert method.value == 5

    def test_partial_response(self):
        topic, method = extract_dual_rating("Topic coherence: 3")
        assert topic.value == 3
        assert method.value is None

    def test_empty_response(self):
        topic, method = extract_dual_rating(None)
        assert topic.value is None and method.value is None
