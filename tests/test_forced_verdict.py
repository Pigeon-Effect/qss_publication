"""Every trial must end in a verdict.

The intrusion task always has exactly one correct answer, so a blind guess
scores 1/panel_size while a non-answer scores zero. Leaving a trial unanswered
would depress accuracy for a reason unrelated to cluster coherence, so the
protocol resolves an unanswered trial in two tiers: ask a cheap model to read
the analysis and name the answer it was heading towards, and failing that,
guess reproducibly.

These tests pin that guarantee. No API key or network access is required.
"""

from __future__ import annotations

import sqlite3

import pytest

from clustervalidation.config import MODELS, RunConfig
from clustervalidation.corpus import load_clusters
from clustervalidation.llm import Completion
from clustervalidation.protocols import intrusion


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.db"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE works_labeled (
            id TEXT, title TEXT, cleaned_abstract TEXT,
            h1_cluster INTEGER, h2_cluster INTEGER, h3_cluster INTEGER
        )
        """
    )
    rows = [
        (
            f"W{h1}{h2}{h3}{n}",
            f"Title {h1}{h2}{h3}-{n}",
            f"Abstract about topic {h1}{h2}{h3} " * 40,
            h1,
            h2,
            h3,
        )
        for h1 in range(3)
        for h2 in range(2)
        for h3 in range(2)
        for n in range(8)
    ]
    connection.executemany("INSERT INTO works_labeled VALUES (?,?,?,?,?,?)", rows)
    connection.commit()
    connection.close()
    return str(path)


def _completion(content: str, reasoning: str = "", finish: str = "stop") -> Completion:
    return Completion(
        content=content,
        reasoning=reasoning,
        finish_reason=finish,
        prompt_tokens=100,
        completion_tokens=10,
        cost_usd=0.000017,
        attempts=1,
    )


class SilentClient:
    """Never states a verdict; the follow-up answers with a bare digit."""

    def __init__(self, follow_up: str | None = "4", reasoning: str = "notes ..."):
        self.follow_up = follow_up
        self.reasoning = reasoning
        self.follow_up_calls = 0

    def complete(self, prompt, system_message=None, model=None):
        if model is not None:  # the forced-choice follow-up
            self.follow_up_calls += 1
            if self.follow_up is None:
                raise RuntimeError("follow-up unavailable")
            return _completion(self.follow_up)
        # Primary call: burned its budget reasoning, emitted nothing.
        return _completion("", reasoning=self.reasoning, finish="length")


def _config(**overrides) -> RunConfig:
    defaults = dict(
        protocol="intrusion",
        level="h3",
        model=MODELS["deepseek-v4-flash"],
        prompt_variant="decisive",
        trials=6,
        seed=20250628,
        panel_size=5,
        force_choice_model=MODELS["deepseek-chat"],
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


def test_follow_up_supplies_the_verdict(corpus):
    """A trial with no verdict is resolved by the cheap follow-up, not lost."""
    clusters = load_clusters(corpus, "h3")
    client = SilentClient(follow_up="4")
    outcome = intrusion.run(clusters, _config(db_path=corpus), client, progress=False)

    assert client.follow_up_calls == 6
    assert all(t.predicted == 4 for t in outcome.completed)
    assert all(t.forced_choice for t in outcome.completed)
    assert not any(t.forced_guess for t in outcome.completed)
    assert outcome.summary()["unparsed_responses"] == 0
    assert outcome.summary()["forced_choice_calls"] == 6


def test_guess_when_the_follow_up_fails(corpus):
    """A broken follow-up must not cost the trial its verdict."""
    clusters = load_clusters(corpus, "h3")
    client = SilentClient(follow_up=None)
    outcome = intrusion.run(clusters, _config(db_path=corpus), client, progress=False)

    assert all(t.forced_guess for t in outcome.completed)
    assert all(t.predicted in range(1, 6) for t in outcome.completed)
    assert outcome.summary()["unparsed_responses"] == 0


def test_no_reasoning_skips_straight_to_the_guess(corpus):
    """With nothing to read there is nothing to ask about."""
    clusters = load_clusters(corpus, "h3")
    client = SilentClient(reasoning="")
    outcome = intrusion.run(clusters, _config(db_path=corpus), client, progress=False)

    assert client.follow_up_calls == 0
    assert all(t.forced_guess for t in outcome.completed)
    assert all(t.predicted in range(1, 6) for t in outcome.completed)


def test_guesses_are_reproducible(corpus):
    """The same seed must reproduce the same guesses."""
    clusters = load_clusters(corpus, "h3")
    first = intrusion.run(
        clusters, _config(db_path=corpus), SilentClient(follow_up=None), progress=False
    )
    second = intrusion.run(
        clusters, _config(db_path=corpus), SilentClient(follow_up=None), progress=False
    )
    assert [t.predicted for t in first.completed] == [
        t.predicted for t in second.completed
    ]

    other_seed = intrusion.run(
        clusters,
        _config(db_path=corpus, seed=12345),
        SilentClient(follow_up=None),
        progress=False,
    )
    assert [t.predicted for t in first.completed] != [
        t.predicted for t in other_seed.completed
    ]


def test_follow_up_cost_is_accounted(corpus):
    """The extra call is cheap, but it is not free and must be recorded."""
    clusters = load_clusters(corpus, "h3")
    outcome = intrusion.run(
        clusters, _config(db_path=corpus), SilentClient(follow_up="2"), progress=False
    )
    # Primary plus follow-up, both stubbed at 0.000017.
    assert all(t.cost_usd == pytest.approx(0.000034) for t in outcome.completed)


def test_disabling_the_follow_up_still_yields_a_verdict(corpus):
    """force_choice_model=None falls straight through to the guess."""
    clusters = load_clusters(corpus, "h3")
    client = SilentClient(follow_up="4")
    outcome = intrusion.run(
        clusters,
        _config(db_path=corpus, force_choice_model=None),
        client,
        progress=False,
    )
    assert client.follow_up_calls == 0
    assert all(t.forced_guess for t in outcome.completed)
    assert outcome.summary()["unparsed_responses"] == 0
