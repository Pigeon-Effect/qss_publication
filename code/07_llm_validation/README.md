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
with a random-guess baseline of 1 / panel size — 20 % for the default
five-document panel. The judge here is a large language model
(`deepseek-v4-flash` with reasoning enabled), which makes running a thousand
trials per level feasible where human annotators would not be.

Accuracy is expected to *rise* as the hierarchy descends, and does: intrusion
detection depends on the ratio of within-cluster spread to between-cluster
separation, and a macro-domain like Computer Science spans computer vision,
NLP and graph learning, so an intruder is less conspicuous there than among
narrow, lexically distinctive research fronts.

## Where the implementation lives

This is the one stage that is fully implemented, tested, and packaged — it is
the part of the pipeline this repository originated from.

- **Package:** [`src/clustervalidation/`](../../src/clustervalidation/) — an
  installable, CLI-driven implementation of two protocols: document intrusion
  detection and Likert coherence rating. Usage, options and reproducibility
  notes are in the top-level [`README.md`](../../README.md).
- **Tests:** [`tests/`](../../tests/) — 62 tests. No API key, no network, no
  spend.
- **Archived runs:** [`results/`](../../results/) — every run's manifest,
  per-trial records and readable transcript, including the reported ones, with
  a per-file inventory in [`results/README.md`](../../results/README.md).

A second protocol, **Likert coherence rating** (Tan and D'Souza 2025), is also
implemented but is *not* what the manuscript reports. It proved far more
prompt-sensitive than intrusion detection — the same h3 clusters score 1.50 or
3.97 depending only on the rubric — and that instability is the reason
intrusion detection carries the external validation.

## `archive/intrusion_detection_scripts/`

The original per-hierarchy-level scripts
(`h1`/`h2`/`h3_intrusion_detection_v4_thinking.py`) that predate the
`clustervalidation` package. They are **superseded** by
`src/clustervalidation/protocols/intrusion.py`, which replaced 19
near-duplicate scripts with one parameterised, seed-reproducible
implementation.

They are kept for provenance, the same way `paper/archive/` keeps superseded
manuscript drafts, and because the transcripts in `results/` were produced by
them. **Prefer the package.** These scripts predate seeded sampling, so they
cannot rebuild a panel sequence; a re-run draws different panels.

They have been updated in two respects only, neither touching their logic:
absolute paths from a different machine now resolve relative to the repository,
and the API key is loaded from `.env` as well as the environment.
