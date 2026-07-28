# Cluster validation for a hierarchical topic taxonomy of AI research

Reference implementation and archived results for the large-language-model
validation of the AI-research taxonomy reported in:

> **Mapping the AI Research of China, the US and the EU: A Scientometric
> Analysis of Citation Shares within AI Subdisciplines (2020–2025)**
> Julius Pfundstein, Thomas Efer, Manuel Burghardt
> Computational Humanities, University of Leipzig — *manuscript in preparation*

The study identifies AI subdisciplines in 1,986,659 OpenAlex records through a
semi-supervised, hierarchical topic model (SPECTER embeddings → high-granularity
K-Means → expert consolidation), yielding **5 domains, 31 fields and 106
research fronts**. This repository holds the code and transcripts for the step
that asks whether those clusters are semantically real rather than analytical
artefacts.

---

## Scope

**This repository contains the validation component only.** It does *not*
contain OpenAlex retrieval, the topic-modeling pipeline, the citation analysis,
or figure generation — those live in a
[separate repository](https://github.com/Pigeon-Effect/scientometric-analysis-of-ai-research).

What is here:

| | |
|---|---|
| `src/clustervalidation/` | the two validation protocols, as an installable package |
| `results/` | transcripts of every run, including the reported ones |
| `paper/` | manuscript sources, figures and bibliography |
| `data/` | where the corpus is expected at runtime (not redistributed) |
| `tests/` | test suite, runs without network access or an API key |

---

## The two protocols

### Document intrusion detection

Adapted from the *reading tea leaves* paradigm of Chang et al. (2009). A panel
of four genuine members of a target cluster is shown to a language model
together with one **intruder** drawn from a different cluster at the same
hierarchy level, in randomised order. The model must identify the intruder.

A semantically tight cluster makes the intruder conspicuous, so **detection
accuracy is the coherence signal**. The random-guess baseline is 1/*panel size*
— 20 % for the default five-document panel.

### Coherence rating

Following Tan and D'Souza (2025). A sample from a single cluster is rated on a
five-point Likert scale for whether it forms one recognisable unit — a
discipline at h1, a subfield at h2, a research topic at h3. There is no random
baseline; this is an absolute judgement, reported as a mean with its spread.

A `dual_score` variant scores topical and methodological coherence separately,
since a cluster can be methodologically tight while topically diffuse.

> Coherence rating proved **far more prompt-sensitive** than intrusion
> detection — the same h3 clusters score 1.50 or 3.97 depending only on the
> rubric. That instability is why the manuscript reports intrusion detection as
> the external validation. See [`results/README.md`](results/README.md) for the
> evidence.

---

## Quick start

Requires Python 3.10+ and a [DeepSeek](https://platform.deepseek.com) API key.

```bash
git clone https://github.com/Pigeon-Effect/qss_publication.git
cd qss_publication

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Provide the API key through the environment — never in a file that gets
committed:

```bash
export DEEPSEEK_API_KEY='sk-...'          # bash / zsh
$env:DEEPSEEK_API_KEY = 'sk-...'          # PowerShell
```

Place the corpus at `data/merged_works_labeled.db` (see
[`data/README.md`](data/README.md)), then:

```bash
# Inspect the corpus — no API calls, no cost
python -m clustervalidation inspect --level h3

# See exactly what would be sent, without spending anything
python -m clustervalidation intrusion --level h3 --trials 100 --dry-run

# Run the reported configuration
python -m clustervalidation intrusion --level h3 --trials 100
python -m clustervalidation coherence --level h1 --trials 50
```

Reports land in `results/<protocol>/` as three files sharing a stem: `.json`
(manifest and summary), `.jsonl` (one record per trial), and `.txt` (readable
transcript).

### Options

| Flag | Default | Notes |
|---|---|---|
| `--level` | *required* | `h1`, `h2` or `h3` |
| `--model` | `deepseek-v4-flash` | also `deepseek-chat`, `deepseek-reasoner`, `deepseek-v4-pro` |
| `--prompt` | `reasoned` / `tan_dsouza` | see `--help` for all registered variants |
| `--trials` | `100` | |
| `--seed` | `20250628` | fixes panel construction |
| `--max-words` | `200` | abstract truncation |
| `--panel-size` | `5` | changes the random baseline |
| `--dry-run` | off | build panels, print one prompt, make no API calls |

Every one of these is written into the run manifest, so a result file always
carries the parameters that produced it.

---

## Reproducibility

**Panel construction is deterministic.** A dedicated seeded generator is used
rather than global random state, so the same `--seed`, `--level`, `--trials`,
`--panel-size` and `--max-words` rebuild the identical panel sequence:

```bash
python -m clustervalidation intrusion --level h3 --trials 100 --seed 20250628
```

**Model responses are not.** Sampling is non-deterministic server-side and the
API is a moving target, so accuracy will vary between runs of the same panels.
Treat a single 100-trial run as a point estimate with meaningful sampling error
(±~5 pp at *n* = 100), not an exact figure.

**Prompts are versioned, not edited.** Every wording ever run is registered by
name in `prompts.py`. Editing a registered variant would silently invalidate the
results that cite it, so new wordings get new names.

**Extraction is auditable.** Models do not always emit the requested verdict
line. Extraction proceeds from the most explicit pattern to the least and
records *which rule fired*, so a value recovered by the last-resort rule is
distinguishable from one read off an explicit marker. Runs report both counts.

### Caveat on the archived runs

The transcripts in `results/` were produced by the original per-experiment
scripts, which used unseeded sampling. They record which panels were shown but
**cannot be regenerated panel-for-panel**. Seeded sampling begins at `v1.0.0`.

---

## Known discrepancy

⚠ **The manuscript and the archived results do not currently agree.**

`paper/main.tex` (section *External Validation via Document Intrusion*) states
1,000 trials per hierarchy level, reporting 46.0 % / 75.0 % / over 84.0 % for
h1 / h2 / h3. The archived runs are 100 trials at 44.0 % / 68.0 % / 82.0 %.

|  | h1 | h2 | h3 | Trials |
|---|---:|---:|---:|---:|
| `paper/main.tex` | 46.0 % | 75.0 % | > 84.0 % | 1,000 |
| `results/intrusion/` | 44.0 % | 68.0 % | 82.0 % | 100 |

Either a larger run exists that was never archived here, or the manuscript
figures need revising. **This has to be reconciled before deposit** — the whole
point of an archived record is that it reproduces the published numbers. Note
that the qualitative claim (monotonic rise across levels, all well above the
20 % baseline) holds under both sets.

---

## Tests

```bash
pytest
```

62 tests covering verdict and rating extraction, panel and sample construction,
seed determinism, corpus loading, and an end-to-end run against a synthetic
SQLite corpus with a stub client. **No API key or network access required** —
nothing in the suite spends credit.

---

## Repository layout

```
├── src/clustervalidation/
│   ├── config.py            models, pricing, taxonomy levels, RunConfig
│   ├── corpus.py            SQLite loading, cluster grouping, truncation
│   ├── llm.py               API client, retries, cost accounting
│   ├── parsing.py           verdict/rating extraction with rule tracking
│   ├── prompts.py           every registered prompt variant
│   ├── reporting.py         JSON / JSONL / text reports
│   ├── cli.py               command-line interface
│   └── protocols/
│       ├── intrusion.py     document-intrusion detection
│       └── coherence.py     Likert coherence rating
├── results/
│   ├── intrusion/           reported + supporting runs
│   ├── coherence/           coherence-rating runs
│   └── exploratory/         model selection, pilots, diagnostics
├── paper/
│   ├── main.tex             current manuscript
│   ├── bibliography.bib
│   ├── figures/
│   └── archive/             superseded drafts and the originating thesis
├── data/                    corpus location (gitignored)
└── tests/
```

Compile the manuscript with `latexmk -pdf main.tex` from `paper/`
(requires `biber`).

---

## Citing

Cite **both** the article and this software. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff); GitHub renders a "Cite this repository" button
from it.

A Zenodo DOI will be minted at publication and added here. Until then, cite the
repository and commit hash.

---

## Licence

| Content | Licence |
|---|---|
| Software (`src/`, `tests/`) | [MIT](LICENSE) |
| Results, figures, manuscript | [CC BY 4.0](LICENSE-DATA) |

Source metadata from [OpenAlex](https://openalex.org), released by OurResearch
under CC0.

---

## References

Chang, J., Boyd-Graber, J., Gerrish, S., Wang, C., & Blei, D. M. (2009).
Reading tea leaves: How humans interpret topic models. *NeurIPS 22*.

Cohan, A., Feldman, S., Beltagy, I., Downey, D., & Weld, D. S. (2020).
SPECTER: Document-level representation learning using citation-informed
transformers. *ACL 2020*.

Priem, J., Piwowar, H., & Orr, R. (2022). OpenAlex: A fully-open index of
scholarly works, authors, venues, institutions, and concepts.
*arXiv:2205.01833*.
