# 07 — LLM-based cluster validation

Tests whether the clusters produced by
[stage 04](../04_subdiscipline_clustering/) capture coherent semantic units
rather than analytical artefacts (manuscript §4.1, *External Validation via
Document Intrusion*).

## Why an external validation is needed at all

Stage 04's taxonomy is the product of an expert consolidating an
over-segmented K-Means partition. That is a defensible procedure, but it is
also a procedure in which a human decided the answer, so "the clusters look
coherent" is not evidence — the same person drew them. The validation has to
come from outside the loop.

Document intrusion, adapted from the *reading tea leaves* paradigm of Chang et
al. (2009), supplies it. A judge is shown four genuine members of a target
cluster plus one **intruder** drawn from a different cluster at the same
hierarchy level, in randomised order, and must identify the intruder. Nothing
about the cluster's label, its TF-IDF terms, or the expert's reasoning is
shown — only the documents.

The logic: a semantically tight cluster makes the intruder conspicuous, a
diffuse one does not. **Detection accuracy is therefore the coherence signal**,
with a random-guess baseline of 1 / panel size — 20 % for the five-document
panel used here. The judge is a large language model
(`deepseek-v4-flash` with reasoning enabled), which is what makes a thousand
trials per level feasible where human annotators would not be.

Accuracy is expected to *rise* as the hierarchy descends, and does — 40.5 % at
h1, 64.7 % at h2, 77.8 % at h3. Intrusion detection depends on the ratio of
within-cluster spread to between-cluster separation, and a macro-domain like
Computer Science spans computer vision, NLP and graph learning, so an intruder
is less conspicuous there than among narrow, lexically distinctive research
fronts.

## Where the implementation lives

- **Package:** [`src/clustervalidation/`](../../src/clustervalidation/) — an
  installable, CLI-driven implementation of two protocols: document intrusion
  detection and Likert coherence rating. Usage, options and reproducibility
  notes are in the top-level [`README.md`](../../README.md).
- **Tests:** [`tests/`](../../tests/) — 75 tests. No API key, no network, no
  spend.
- **Results:** [`results/`](../../results/) — the complete record of all 3,000
  reported trials: manifest, per-trial records, readable transcript and run log
  for each level, documented in
  [`results/README.md`](../../results/README.md).

One parameterised, seed-reproducible implementation replaced the nineteen
near-duplicate per-experiment scripts the protocol was first developed as. Panel
construction uses a dedicated seeded generator, so a run is rebuildable from its
manifest.

## The second protocol

**Likert coherence rating** (after Tan and D'Souza 2025) is also implemented: a
sample from a single cluster is rated on a five-point scale for whether it forms
one recognisable unit — a discipline at h1, a subfield at h2, a research topic
at h3. A `dual_score` variant rates topical and methodological coherence
separately, since a cluster can be methodologically tight while topically
diffuse.

It is not what the article reports. In development it proved far more sensitive
to the wording of the rubric than to the clusters being rated: the same clusters
can move most of the way across the five-point scale on a rubric change alone,
and there is no random baseline against which to calibrate an absolute
judgement. A rating that unstable cannot carry an external validation. Document
intrusion, being a forced choice with a known chance level, can — which is why
it does.
