"""Direct coherence rating of cluster samples.

Following Tan and D'Souza (2025), a sample of documents from a single cluster is
shown to a language model, which rates on a 5-point Likert scale how well they
belong together as one unit of analysis. Unlike intrusion detection this has no
random baseline: it is an absolute judgement, and is therefore reported as a
mean rating with its spread rather than as an accuracy.

The ``dual_score`` variant separates topical from methodological coherence,
which a single score conflates.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field

from clustervalidation.config import RunConfig
from clustervalidation.corpus import ClusterMap, Document, truncate_words
from clustervalidation.llm import ChatClient
from clustervalidation.parsing import extract_dual_rating, extract_rating
from clustervalidation.prompts import (
    COHERENCE_PROMPTS,
    COHERENCE_SYSTEM_MESSAGE,
    render_panel,
)


@dataclass(frozen=True)
class Sample:
    """A set of documents drawn from one cluster."""

    cluster: str
    documents: list[Document]


@dataclass
class Trial:
    """The outcome of rating one sample."""

    index: int
    sample: Sample
    rating: int | None = None
    topic_rating: int | None = None
    method_rating: int | None = None
    extraction_rule: str = "none"
    content: str = ""
    reasoning: str = ""
    finish_reason: str = ""
    truncated: bool = False
    cost_usd: float = 0.0
    error: str | None = None


@dataclass
class Outcome:
    """Aggregate results of a run."""

    config: RunConfig
    trials: list[Trial] = field(default_factory=list)

    @property
    def completed(self) -> list[Trial]:
        return [t for t in self.trials if t.error is None]

    @property
    def ratings(self) -> list[int]:
        return [t.rating for t in self.completed if t.rating is not None]

    @property
    def mean_rating(self) -> float | None:
        return statistics.mean(self.ratings) if self.ratings else None

    def summary(self) -> dict:
        ratings = self.ratings
        topic = [t.topic_rating for t in self.completed if t.topic_rating is not None]
        method = [
            t.method_rating for t in self.completed if t.method_rating is not None
        ]
        result = {
            "clusters_requested": self.config.trials,
            "clusters_rated": len(self.completed),
            "clusters_failed": len(self.trials) - len(self.completed),
            "valid_ratings": len(ratings),
            "mean_rating": round(statistics.mean(ratings), 3) if ratings else None,
            "median_rating": statistics.median(ratings) if ratings else None,
            "stdev_rating": (
                round(statistics.stdev(ratings), 3) if len(ratings) > 1 else None
            ),
            "min_rating": min(ratings) if ratings else None,
            "max_rating": max(ratings) if ratings else None,
            "truncated_responses": sum(1 for t in self.completed if t.truncated),
            "total_cost_usd": round(sum(t.cost_usd for t in self.trials), 6),
        }
        if topic:
            result["mean_topic_coherence"] = round(statistics.mean(topic), 3)
        if method:
            result["mean_method_coherence"] = round(statistics.mean(method), 3)
        return result


def build_samples(clusters: ClusterMap, config: RunConfig) -> list[Sample]:
    """Draw one sample per cluster, up to ``config.trials`` clusters.

    Clusters are sampled without replacement: each cluster is rated at most
    once, so the mean rating is an average over clusters rather than over
    draws, and is not dominated by whichever cluster happened to be picked
    repeatedly.
    """
    rng = random.Random(config.seed)
    eligible = sorted(
        cid for cid, docs in clusters.items() if len(docs) >= config.panel_size
    )
    if not eligible:
        raise ValueError(
            f"no cluster has the {config.panel_size} documents needed for a sample"
        )

    count = min(config.trials, len(eligible))
    selected = rng.sample(eligible, count)

    samples = []
    for cluster in selected:
        documents = rng.sample(clusters[cluster], config.panel_size)
        samples.append(
            Sample(
                cluster=cluster,
                documents=[
                    Document(d.id, d.title, truncate_words(d.abstract, config.max_words))
                    for d in documents
                ],
            )
        )
    return samples


def run(
    clusters: ClusterMap,
    config: RunConfig,
    client: ChatClient,
    progress: bool = True,
) -> Outcome:
    """Execute the protocol and return the outcome."""
    template = COHERENCE_PROMPTS[config.prompt_variant]
    samples = build_samples(clusters, config)
    outcome = Outcome(config=config)
    dual = config.prompt_variant == "dual_score"

    for index, sample in enumerate(samples, start=1):
        trial = Trial(index=index, sample=sample)
        rendered = render_panel([(d.title, d.abstract) for d in sample.documents])
        prompt = template(rendered, config.panel_size, config.level)

        try:
            completion = client.complete(prompt, COHERENCE_SYSTEM_MESSAGE)
        except Exception as error:  # noqa: BLE001 - recorded per trial
            trial.error = str(error)
            outcome.trials.append(trial)
            if progress:
                print(f"Cluster {sample.cluster:>4} | ERROR: {error}")
            continue

        if dual:
            topic, method = extract_dual_rating(completion.content)
            trial.topic_rating = topic.value
            trial.method_rating = method.value
            # The topic score is the headline figure for the dual variant.
            trial.rating = topic.value
            trial.extraction_rule = topic.rule
        else:
            extraction = extract_rating(completion.content)
            trial.rating = extraction.value
            trial.extraction_rule = extraction.rule

        trial.content = completion.content
        trial.reasoning = completion.reasoning
        trial.finish_reason = completion.finish_reason
        trial.truncated = completion.truncated
        trial.cost_usd = completion.cost_usd
        outcome.trials.append(trial)

        if progress:
            if dual:
                print(
                    f"Cluster {sample.cluster:>4} | topic={trial.topic_rating} "
                    f"method={trial.method_rating}"
                )
            else:
                print(f"Cluster {sample.cluster:>4} | rating={trial.rating}")

    return outcome
