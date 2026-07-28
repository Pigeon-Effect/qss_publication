"""Tests for panel and sample construction.

These guarantee the two properties the protocols depend on: that a run is
reproducible from its seed, and that panels are well-formed (exactly one
intruder, drawn from a different cluster than the home documents).
"""

import pytest

from clustervalidation.config import MODELS, RunConfig
from clustervalidation.corpus import Document
from clustervalidation.protocols import coherence, intrusion


@pytest.fixture
def clusters():
    """Six synthetic clusters of ten documents each."""
    return {
        f"c{index}": [
            Document(f"{index}-{n}", f"Title {index}-{n}", f"Abstract {index}-{n} " * 60)
            for n in range(10)
        ]
        for index in range(6)
    }


def make_config(protocol="intrusion", **overrides):
    defaults = dict(
        protocol=protocol,
        level="h3",
        model=MODELS["deepseek-chat"],
        prompt_variant="reasoned" if protocol == "intrusion" else "tan_dsouza",
        trials=20,
        seed=42,
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


class TestIntrusionPanels:
    def test_builds_requested_number(self, clusters):
        panels = intrusion.build_panels(clusters, make_config(trials=15))
        assert len(panels) == 15

    def test_exactly_one_intruder(self, clusters):
        for panel in intrusion.build_panels(clusters, make_config()):
            assert sum(item.is_intruder for item in panel.items) == 1

    def test_panel_has_configured_size(self, clusters):
        config = make_config(panel_size=4)
        for panel in intrusion.build_panels(clusters, config):
            assert len(panel.items) == 4

    def test_intruder_comes_from_another_cluster(self, clusters):
        for panel in intrusion.build_panels(clusters, make_config()):
            assert panel.home_cluster != panel.intruder_cluster

    def test_true_position_points_at_intruder(self, clusters):
        for panel in intrusion.build_panels(clusters, make_config()):
            assert panel.items[panel.true_position - 1].is_intruder

    def test_home_documents_are_distinct(self, clusters):
        for panel in intrusion.build_panels(clusters, make_config()):
            ids = [item.document.id for item in panel.items]
            assert len(ids) == len(set(ids))

    def test_abstracts_are_truncated(self, clusters):
        config = make_config(max_words=25)
        for panel in intrusion.build_panels(clusters, config):
            for item in panel.items:
                assert len(item.document.abstract.split()) <= 25

    def test_same_seed_reproduces_run(self, clusters):
        first = intrusion.build_panels(clusters, make_config(seed=7))
        second = intrusion.build_panels(clusters, make_config(seed=7))
        assert [p.home_cluster for p in first] == [p.home_cluster for p in second]
        assert [p.true_position for p in first] == [p.true_position for p in second]

    def test_different_seed_changes_run(self, clusters):
        first = intrusion.build_panels(clusters, make_config(seed=1, trials=40))
        second = intrusion.build_panels(clusters, make_config(seed=2, trials=40))
        assert [p.true_position for p in first] != [p.true_position for p in second]

    def test_rejects_insufficient_clusters(self):
        with pytest.raises(ValueError, match="at least 2 clusters"):
            intrusion.build_panels(
                {"only": [Document(str(n), "t", "a") for n in range(10)]},
                make_config(),
            )


class TestCoherenceSamples:
    def test_one_sample_per_cluster(self, clusters):
        samples = coherence.build_samples(clusters, make_config("coherence", trials=4))
        assert len({s.cluster for s in samples}) == len(samples) == 4

    def test_caps_at_available_clusters(self, clusters):
        # Only six clusters exist, so a request for 50 yields six.
        samples = coherence.build_samples(clusters, make_config("coherence", trials=50))
        assert len(samples) == 6

    def test_sample_size_matches_panel_size(self, clusters):
        config = make_config("coherence", panel_size=3)
        for sample in coherence.build_samples(clusters, config):
            assert len(sample.documents) == 3

    def test_same_seed_reproduces_run(self, clusters):
        config = make_config("coherence", seed=99)
        first = coherence.build_samples(clusters, config)
        second = coherence.build_samples(clusters, config)
        assert [s.cluster for s in first] == [s.cluster for s in second]
        assert [[d.id for d in s.documents] for s in first] == [
            [d.id for d in s.documents] for s in second
        ]
