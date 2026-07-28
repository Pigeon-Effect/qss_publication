"""End-to-end tests over a synthetic corpus.

These exercise the whole chain - SQLite loading, panel construction, prompt
rendering, response parsing and report writing - with a stub client, so no API
credit is spent and no network access is needed.
"""

import json
import sqlite3

import pytest

from clustervalidation.config import MODELS, RunConfig
from clustervalidation.corpus import load_clusters
from clustervalidation.llm import Completion
from clustervalidation.protocols import coherence, intrusion
from clustervalidation.reporting import write_reports


@pytest.fixture
def corpus(tmp_path):
    """A small SQLite corpus with the schema the real database uses."""
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
    rows = []
    for h1 in range(3):
        for h2 in range(2):
            for h3 in range(2):
                for n in range(8):
                    rows.append(
                        (
                            f"W{h1}{h2}{h3}{n}",
                            f"Title {h1}{h2}{h3}-{n}",
                            f"Abstract about topic {h1}{h2}{h3} " * 40,
                            h1,
                            h2,
                            h3,
                        )
                    )
    # One document with an empty abstract, which must be filtered out.
    rows.append(("EMPTY", "No abstract", "   ", 0, 0, 0))
    connection.executemany("INSERT INTO works_labeled VALUES (?,?,?,?,?,?)", rows)
    connection.commit()
    connection.close()
    return str(path)


class StubClient:
    """Returns a canned response, recording the prompts it was given."""

    def __init__(self, reply="Final verdict: 1"):
        self.reply = reply
        self.prompts = []

    def complete(self, prompt, system_message=None):
        self.prompts.append(prompt)
        return Completion(
            content=self.reply,
            reasoning="",
            finish_reason="stop",
            prompt_tokens=100,
            completion_tokens=10,
            cost_usd=0.000017,
            attempts=1,
        )


def make_config(protocol, level, **overrides):
    defaults = dict(
        protocol=protocol,
        level=level,
        model=MODELS["deepseek-chat"],
        prompt_variant="reasoned" if protocol == "intrusion" else "tan_dsouza",
        trials=6,
        seed=11,
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


class TestCorpusLoading:
    @pytest.mark.parametrize("level,expected", [("h1", 3), ("h2", 6), ("h3", 12)])
    def test_cluster_counts_per_level(self, corpus, level, expected):
        assert len(load_clusters(corpus, level)) == expected

    def test_blank_abstracts_excluded(self, corpus):
        clusters = load_clusters(corpus, "h1")
        ids = {doc.id for docs in clusters.values() for doc in docs}
        assert "EMPTY" not in ids

    def test_small_clusters_dropped(self, corpus):
        # Every synthetic cluster holds 8 documents, so a threshold of 9 empties it.
        with pytest.raises(ValueError, match="at least 2 clusters"):
            load_clusters(corpus, "h3", min_cluster_size=9)

    def test_missing_database(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="data/README"):
            load_clusters(str(tmp_path / "absent.db"), "h1")

    def test_unknown_level(self, corpus):
        with pytest.raises(ValueError, match="unknown hierarchy level"):
            load_clusters(corpus, "h4")


class TestIntrusionRun:
    def test_run_and_report(self, corpus, tmp_path):
        clusters = load_clusters(corpus, "h3")
        config = make_config("intrusion", "h3")
        client = StubClient("Final verdict: 1")

        outcome = intrusion.run(clusters, config, client, progress=False)

        assert len(outcome.trials) == 6
        assert len(client.prompts) == 6
        # The stub always answers 1, so accuracy equals the share of panels
        # whose intruder genuinely landed in position 1.
        expected = sum(1 for t in outcome.trials if t.panel.true_position == 1)
        assert outcome.correct == expected
        assert outcome.baseline == pytest.approx(20.0)

        paths = write_reports(outcome, str(tmp_path), stem="run")
        manifest = json.loads(open(paths["json"], encoding="utf-8").read())
        assert manifest["configuration"]["level"] == "h3"
        assert manifest["configuration"]["seed"] == 11
        assert manifest["summary"]["trials_completed"] == 6

        records = [
            json.loads(line)
            for line in open(paths["jsonl"], encoding="utf-8").read().splitlines()
        ]
        assert len(records) == 6
        assert all(len(r["panel"]) == 5 for r in records)
        assert all(sum(p["is_intruder"] for p in r["panel"]) == 1 for r in records)

        transcript = open(paths["txt"], encoding="utf-8").read()
        assert "DOCUMENT INTRUSION DETECTION" in transcript
        assert "*** INTRUDER ***" in transcript

    def test_api_failure_recorded_not_raised(self, corpus, tmp_path):
        class FailingClient:
            def complete(self, prompt, system_message=None):
                raise RuntimeError("rate limited")

        clusters = load_clusters(corpus, "h2")
        outcome = intrusion.run(
            clusters, make_config("intrusion", "h2"), FailingClient(), progress=False
        )
        assert len(outcome.trials) == 6
        assert outcome.completed == []
        assert outcome.accuracy == 0.0
        assert outcome.summary()["trials_failed"] == 6

    def test_verdict_recovered_from_reasoning(self, corpus):
        class ReasoningOnlyClient:
            def complete(self, prompt, system_message=None):
                return Completion(
                    content="",
                    reasoning="weighing them up... Final verdict: 2",
                    finish_reason="stop",
                    prompt_tokens=10,
                    completion_tokens=5,
                    cost_usd=0.0,
                    attempts=1,
                )

        clusters = load_clusters(corpus, "h3")
        outcome = intrusion.run(
            clusters,
            make_config("intrusion", "h3", trials=3),
            ReasoningOnlyClient(),
            progress=False,
        )
        assert all(t.predicted == 2 for t in outcome.trials)
        assert all(t.recovered_from_reasoning for t in outcome.trials)
        assert outcome.summary()["recovered_from_reasoning"] == 3


class TestCoherenceRun:
    def test_run_and_report(self, corpus, tmp_path):
        clusters = load_clusters(corpus, "h2")
        config = make_config("coherence", "h2", trials=4)
        outcome = coherence.run(
            clusters, config, StubClient("Rating: 4"), progress=False
        )

        assert len(outcome.trials) == 4
        assert outcome.mean_rating == pytest.approx(4.0)

        paths = write_reports(outcome, str(tmp_path), stem="coh")
        manifest = json.loads(open(paths["json"], encoding="utf-8").read())
        assert manifest["summary"]["mean_rating"] == 4.0
        assert manifest["summary"]["valid_ratings"] == 4
        assert "CLUSTER COHERENCE RATING" in open(
            paths["txt"], encoding="utf-8"
        ).read()

    def test_dual_score_variant(self, corpus):
        clusters = load_clusters(corpus, "h3")
        config = make_config("coherence", "h3", prompt_variant="dual_score", trials=3)
        client = StubClient("Topic coherence: 5\nMethodology coherence: 3")
        outcome = coherence.run(clusters, config, client, progress=False)

        assert all(t.topic_rating == 5 for t in outcome.trials)
        assert all(t.method_rating == 3 for t in outcome.trials)
        summary = outcome.summary()
        assert summary["mean_topic_coherence"] == 5.0
        assert summary["mean_method_coherence"] == 3.0

    def test_each_cluster_rated_at_most_once(self, corpus):
        clusters = load_clusters(corpus, "h3")
        config = make_config("coherence", "h3", trials=100)
        samples = coherence.build_samples(clusters, config)
        assert len({s.cluster for s in samples}) == len(samples) == 12


class TestPromptVariants:
    @pytest.mark.parametrize(
        "variant",
        ["reasoned", "minimal", "expert", "narrow", "chain_of_thought", "structured"],
    )
    def test_every_intrusion_variant_renders(self, corpus, variant):
        clusters = load_clusters(corpus, "h3")
        config = make_config("intrusion", "h3", prompt_variant=variant, trials=2)
        client = StubClient()
        intrusion.run(clusters, config, client, progress=False)
        # Each rendered prompt must carry all five numbered panel positions.
        for prompt in client.prompts:
            for position in range(1, 6):
                assert f"[{position}]" in prompt

    @pytest.mark.parametrize("level", ["h1", "h2", "h3"])
    def test_tan_dsouza_adapts_to_level(self, corpus, level):
        clusters = load_clusters(corpus, level)
        config = make_config("coherence", level, trials=1)
        client = StubClient("Rating: 3")
        coherence.run(clusters, config, client, progress=False)
        prompt = client.prompts[0]
        expected = {
            "h1": "broad scientific discipline",
            "h2": "research subfield",
            "h3": "specific research topic",
        }[level]
        assert expected in prompt
