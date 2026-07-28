"""Extraction of structured verdicts from free-text model responses.

Models are asked to end their reply with a fixed marker line, but do not always
comply, so extraction proceeds from the most explicit pattern to the least and
records which rule fired. Recording the rule matters: a result recovered by the
last-resort rule is weaker evidence than one read off an explicit verdict line,
and the reports carry that distinction through to the summary.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# Ordered from most to least explicit. The first match wins.
_VERDICT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("final_verdict", r"final verdict\s*:\s*([1-9])"),
    ("verdict", r"verdict\s*:\s*([1-9])"),
    ("answer", r"answer\s*:\s*([1-9])"),
    ("intruder_label", r"intruder\s*:\s*([1-9])"),
    ("intruder_prose", r"intruder\s+is\s+(?:paper\s+)?([1-9])"),
    ("bare_line", r"^\s*([1-9])\s*$"),
    ("trailing_digit", r"\b([1-9])\s*$"),
)

_RATING_PATTERNS: tuple[tuple[str, str], ...] = (
    ("rating_label", r"rating\s*:\s*([1-5])"),
    ("topic_coherence", r"topic coherence\s*:\s*([1-5])"),
    ("score_label", r"score\s*:\s*([1-5])"),
    ("bare_line", r"^\s*([1-5])\s*$"),
    ("trailing_digit", r"\b([1-5])\s*$"),
)

_FLAGS = re.IGNORECASE | re.MULTILINE


class Extraction(NamedTuple):
    """The value read from a response and the rule that produced it."""

    value: int | None
    rule: str

    @property
    def is_explicit(self) -> bool:
        """True when the value came from a labelled marker, not a bare digit."""
        return self.rule not in {"bare_line", "trailing_digit", "last_digit", "none"}


def _extract(
    text: str | None,
    patterns: tuple[tuple[str, str], ...],
    valid: range,
) -> Extraction:
    if not text:
        return Extraction(None, "none")

    for rule, pattern in patterns:
        match = re.search(pattern, text, _FLAGS)
        if match:
            value = int(match.group(1))
            if value in valid:
                return Extraction(value, rule)

    # Last resort: the conclusion usually sits at the end of the reply, so the
    # final in-range digit anywhere in the text is the best remaining guess.
    digits = [int(d) for d in re.findall(r"[0-9]", text) if int(d) in valid]
    if digits:
        return Extraction(digits[-1], "last_digit")
    return Extraction(None, "none")


def extract_verdict(text: str | None, panel_size: int = 5) -> Extraction:
    """Read the intruder position (1..``panel_size``) from a response."""
    return _extract(text, _VERDICT_PATTERNS, range(1, panel_size + 1))


def extract_rating(text: str | None) -> Extraction:
    """Read a 1-5 Likert coherence rating from a response."""
    return _extract(text, _RATING_PATTERNS, range(1, 6))


def extract_dual_rating(text: str | None) -> tuple[Extraction, Extraction]:
    """Read separate topic and methodology ratings from a response.

    Used by the ``dual_score`` coherence variant, which asks the model to rate
    topical and methodological coherence independently.
    """
    if not text:
        return Extraction(None, "none"), Extraction(None, "none")

    topic = re.search(r"topic coherence\s*:\s*([1-5])", text, _FLAGS)
    method = re.search(r"method(?:ology)? coherence\s*:\s*([1-5])", text, _FLAGS)
    return (
        Extraction(int(topic.group(1)), "topic_coherence") if topic
        else Extraction(None, "none"),
        Extraction(int(method.group(1)), "method_coherence") if method
        else Extraction(None, "none"),
    )
