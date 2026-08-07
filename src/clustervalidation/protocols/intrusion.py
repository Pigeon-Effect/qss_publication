"""Document-intrusion detection.

Adapted from the "reading tea leaves" paradigm of Chang et al. (2009), which
validates topic models by testing whether a human (here, a language model) can
spot an item that does not belong. A panel holds ``panel_size - 1`` genuine
members of a target cluster plus one intruder drawn from a different cluster at
the same hierarchy level. Detection accuracy - the share of trials in which the
intruder is correctly located - is the coherence signal: a cluster that is
semantically tight makes the intruder conspicuous.

The random-guess baseline is ``1 / panel_size`` (20% for the default panel of
five), which is what accuracy must be compared against.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from clustervalidation.config import RunConfig
from clustervalidation.corpus import ClusterMap, Document, truncate_words
from clustervalidation.llm import ChatClient
from clustervalidation.parsing import Extraction, extract_verdict
from clustervalidation.prompts import INTRUSION_PROMPTS, force_choice, render_panel


@dataclass(frozen=True)
class PanelItem:
    """One document as placed in a panel."""

    document: Document
    is_intruder: bool


@dataclass(frozen=True)
class Panel:
    """A single intrusion trial before it is sent to the model."""

    home_cluster: str
    intruder_cluster: str
    items: list[PanelItem]

    @property
    def true_position(self) -> int:
        """1-based index of the intruder after shuffling."""
        return next(
            i for i, item in enumerate(self.items, start=1) if item.is_intruder
        )


@dataclass
class Trial:
    """The outcome of one panel."""

    index: int
    panel: Panel
    predicted: int | None = None
    extraction_rule: str = "none"
    recovered_from_reasoning: bool = False
    # A verdict obtained by asking a cheap model to read the analysis and name
    # the answer it was heading towards.
    forced_choice: bool = False
    # Last resort: a seeded random pick, used only when even the follow-up
    # produced nothing. Expected accuracy 1/panel_size, not zero.
    forced_guess: bool = False
    correct: bool = False
    content: str = ""
    reasoning: str = ""
    finish_reason: str = ""
    truncated: bool = False
    cost_usd: float = 0.0
    error: str | None = None
    # Wall-clock time the response was recorded, so a long run can be audited
    # trial by trial rather than only through the run-level manifest.
    timestamp_utc: str = ""


@dataclass
class Outcome:
    """Aggregate results of a run."""

    config: RunConfig
    trials: list[Trial] = field(default_factory=list)

    @property
    def completed(self) -> list[Trial]:
        return [t for t in self.trials if t.error is None]

    @property
    def correct(self) -> int:
        return sum(1 for t in self.completed if t.correct)

    @property
    def accuracy(self) -> float:
        """Share of completed trials answered correctly, in percent."""
        if not self.completed:
            return 0.0
        return 100.0 * self.correct / len(self.completed)

    @property
    def baseline(self) -> float:
        """Random-guess accuracy for this panel size, in percent."""
        return 100.0 / self.config.panel_size

    def extraction_rule_counts(self) -> dict[str, int]:
        """How often each parsing rule fired, most frequent first.

        Reported so that a reader can see how much of the accuracy figure rests
        on explicit verdict lines versus the permissive fallback rules.
        """
        counts: dict[str, int] = {}
        for trial in self.completed:
            counts[trial.extraction_rule] = counts.get(trial.extraction_rule, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def summary(self) -> dict:
        completed = self.completed
        return {
            "trials_requested": self.config.trials,
            "trials_completed": len(completed),
            "trials_failed": len(self.trials) - len(completed),
            "correct": self.correct,
            "accuracy_percent": round(self.accuracy, 2),
            "random_baseline_percent": round(self.baseline, 2),
            "recovered_from_reasoning": sum(
                1 for t in completed if t.recovered_from_reasoning
            ),
            "non_explicit_verdicts": sum(
                1
                for t in completed
                if t.extraction_rule in {"bare_line", "trailing_digit", "last_digit"}
            ),
            "truncated_responses": sum(1 for t in completed if t.truncated),
            "forced_choice_calls": sum(1 for t in completed if t.forced_choice),
            "forced_guesses": sum(1 for t in completed if t.forced_guess),
            "unparsed_responses": sum(1 for t in completed if t.predicted is None),
            "extraction_rule_counts": self.extraction_rule_counts(),
            "total_cost_usd": round(sum(t.cost_usd for t in self.trials), 6),
        }


def build_panels(clusters: ClusterMap, config: RunConfig) -> list[Panel]:
    """Draw ``config.trials`` panels using a seeded generator.

    A dedicated ``random.Random`` instance is used rather than the module-level
    functions, so panel construction is reproducible from ``config.seed`` and
    cannot be perturbed by other code touching global random state.
    """
    rng = random.Random(config.seed)
    cluster_ids = sorted(clusters)  # sorted so the seed alone fixes the draw
    home_count = config.panel_size - 1

    eligible = [cid for cid in cluster_ids if len(clusters[cid]) >= home_count]
    if len(eligible) < 2:
        raise ValueError(
            f"need at least 2 clusters with >={home_count} documents to build "
            f"panels of size {config.panel_size}"
        )

    panels: list[Panel] = []
    for _ in range(config.trials):
        home = rng.choice(eligible)
        home_docs = rng.sample(clusters[home], home_count)

        intruder_cluster = rng.choice([c for c in eligible if c != home])
        intruder_doc = rng.choice(clusters[intruder_cluster])

        items = [
            PanelItem(_truncate(doc, config.max_words), False) for doc in home_docs
        ]
        items.append(PanelItem(_truncate(intruder_doc, config.max_words), True))
        rng.shuffle(items)

        panels.append(Panel(home, intruder_cluster, items))
    return panels


def _force_verdict(
    reasoning: str,
    config: RunConfig,
    client: ChatClient,
    index: int,
) -> tuple[Extraction, bool, bool, float]:
    """Obtain a verdict for a trial whose primary response gave none.

    Two tiers, cheapest sufficient one wins:

    1. **Forced choice.** A small, non-reasoning model reads the original
       analysis and names the answer it was heading towards. This is extraction,
       not adjudication - the follow-up sees no abstracts. It costs a few
       hundred tokens and only fires on the rare unanswered trial.
    2. **Forced guess.** If there is no analysis to read, or the follow-up
       fails or returns nothing usable, pick uniformly at random from a
       generator seeded by the run seed and the trial index, so the guess is
       reproducible. Expected accuracy is the random baseline rather than zero.

    Returns the extraction, flags for which tier produced it, and the cost of
    the follow-up call so the caller can add it to the trial's total.
    """
    attempted = bool(reasoning) and config.force_choice_model is not None
    extra_cost = 0.0

    if attempted:
        try:
            follow_up = client.complete(
                force_choice(reasoning, config.panel_size),
                model=config.force_choice_model,
            )
            extra_cost = follow_up.cost_usd
            # The follow-up is asked for a bare digit, so the permissive rules
            # are appropriate here in a way they are not for a severed trace.
            recovered = extract_verdict(follow_up.content, config.panel_size)
            if recovered.value is not None:
                return (
                    Extraction(recovered.value, "forced_choice"),
                    True,
                    False,
                    extra_cost,
                )
        except Exception:  # noqa: BLE001 - a failed follow-up falls through
            # The follow-up is a convenience, never a reason to lose the trial.
            pass

    guesser = random.Random(f"{config.seed}:{index}")
    return (
        Extraction(guesser.randint(1, config.panel_size), "forced_guess"),
        attempted,
        True,
        extra_cost,
    )


def _truncate(doc: Document, max_words: int) -> Document:
    return Document(doc.id, doc.title, truncate_words(doc.abstract, max_words))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(
    clusters: ClusterMap,
    config: RunConfig,
    client: ChatClient,
    progress: bool = True,
    on_trial: Callable[[Trial], None] | None = None,
) -> Outcome:
    """Execute the protocol and return the outcome.

    A failing trial is recorded and the run continues; only the panels that
    produced a response contribute to accuracy.

    ``on_trial`` is called with each trial as it completes. A run of a thousand
    reasoning trials takes hours, so the caller uses this to checkpoint results
    to disk rather than holding them in memory until the end, where a killed
    process would take every completed trial with it.
    """
    template = INTRUSION_PROMPTS[config.prompt_variant]
    panels = build_panels(clusters, config)
    outcome = Outcome(config=config)

    for index, panel in enumerate(panels, start=1):
        trial = Trial(index=index, panel=panel)
        rendered = render_panel(
            [(item.document.title, item.document.abstract) for item in panel.items]
        )
        prompt = template(rendered, config.panel_size)

        try:
            completion = client.complete(prompt)
        except Exception as error:  # noqa: BLE001 - recorded per trial
            trial.error = str(error)
            trial.timestamp_utc = _now()
            outcome.trials.append(trial)
            if on_trial is not None:
                on_trial(trial)
            if progress:
                print(f"Trial {index:4d} | ERROR: {error}")
            continue

        # Primary read is the visible answer; reasoning models sometimes emit
        # the verdict only in the reasoning trace, so that is the fallback.
        #
        # Only an *explicit* verdict is recoverable from a trace. The permissive
        # rules exist for replies that are merely terse, not for traces severed
        # by the token ceiling: a trace cut mid-sentence ends on an enumeration
        # fragment ("... paper 2 clay bricks; paper 3"), and reading that as the
        # answer scores a coin flip as though the model had decided. It is wrong
        # four times in five and right once, so it adds noise in both
        # directions. Left unparsed, the trial stays in the denominator and
        # counts as incorrect - which is what a non-answer is.
        extraction = extract_verdict(completion.content, config.panel_size)
        if extraction.value is None and completion.reasoning:
            recovered = extract_verdict(completion.reasoning, config.panel_size)
            if recovered.is_explicit:
                extraction = recovered
                trial.recovered_from_reasoning = True

        # Every trial must end in a verdict. The task has exactly one correct
        # answer, so a blind guess scores 1/panel_size while a non-answer
        # scores zero; leaving a trial unanswered would depress accuracy for a
        # reason unrelated to cluster coherence.
        follow_up_cost = 0.0
        if extraction.value is None:
            (
                extraction,
                trial.forced_choice,
                trial.forced_guess,
                follow_up_cost,
            ) = _force_verdict(completion.reasoning, config, client, index)

        trial.predicted = extraction.value
        trial.extraction_rule = extraction.rule
        trial.correct = extraction.value == panel.true_position
        trial.content = completion.content
        trial.reasoning = completion.reasoning
        trial.finish_reason = completion.finish_reason
        trial.truncated = completion.truncated
        trial.cost_usd = completion.cost_usd + follow_up_cost
        trial.timestamp_utc = _now()
        outcome.trials.append(trial)
        if on_trial is not None:
            on_trial(trial)

        if progress:
            mark = "OK" if trial.correct else " X"
            flags = "".join(
                (
                    " [reasoning]" if trial.recovered_from_reasoning else "",
                    " [truncated]" if trial.truncated else "",
                )
            )
            print(
                f"Trial {index:4d} | true={panel.true_position} "
                f"pred={trial.predicted} | {mark}{flags}"
            )

    return outcome
