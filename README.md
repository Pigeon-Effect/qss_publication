# Mapping the AI Research of China, the US and the EU

Research compendium for:

> **Mapping the AI Research of China, the US and the EU: A Scientometric
> Analysis of Citation Shares within AI Subdisciplines (2020-2024)**
> Julius Pfundstein, Thomas Efer, Manuel Burghardt
> Computational Humanities, University of Leipzig

This repository holds everything behind that article: the code that built the
corpus and analysed it, the complete record of the validation experiments, the
cluster-level result tables, and the manuscript. The labelled corpus itself is
a separate Zenodo data record (see [Data](#data)).

---

## What the study did, in plain terms

The question of who is "ahead" in AI is usually answered with a single number —
total publications, total citations. That number hides something important:
**AI is not one field.** A country can dominate computer vision and be weak in
the social sciences, and a single ranking will never show it.

So this study did two things.

**First, it drew a map of AI research.** It collected the metadata of 1,986,659
AI-related publications from 2020–2024, retrieved from OpenAlex with 279 search
terms, and asked what subdisciplines they fall into. Rather than trusting an
algorithm to find "the right number" of topics (there isn't one; the evidence
is in [stage 04](code/04_subdiscipline_clustering/)), the pipeline deliberately
cuts the corpus into far too many pieces and has a human labeler consolidate
them into meaningful units, at three levels. The result is a taxonomy of
**5 domains, 31 fields and 106 research fronts.**

**Second, it measured who holds influence where.** Every publication's
citations are split among the countries that produced it, in proportion to its
authors' institutional affiliations, and summed within each cluster. Grouped
into three blocs — China (mainland, Hong Kong, Macao), the United States and the
EU-27 — and a rest of the world, this gives each bloc's share of the citations
in every part of AI research.

**What it found.** The three blocs together account for 57.4 % of the corpus's
publications and 62.3 % of its citations, in similar shares but with sharply
different profiles. China leads computer science and engineering, above all
computer vision, control systems and robotics, but trails in the social and
health sciences. The United States has its centre of gravity in health science
and in method fields with a long history in AI, such as natural language
processing. The EU-27 has the most even profile: it leads most of the social
sciences, has few large gaps, and is weak mainly in the methodological core.

### How the map was checked

A taxonomy built by a human labeler has an obvious weakness: the person who drew
the boundaries is also the person who thinks they look right. So the clusters
were tested from outside, with a protocol adapted from Chang et al.'s *reading
tea leaves* (2009). A judge sees four papers from one cluster and one "intruder"
from another, shuffled, with nothing but titles and abstracts, and names the
intruder. If a cluster is a coherent topic, the intruder stands out. The judge
is a large language model, which makes 3,000 trials feasible.

| Level | What a cluster is here | Accuracy | Chance |
|---|---|---:|---:|
| h1 | one of 5 domains | 40.5 % | 20 % |
| h2 | one of 31 fields | 64.7 % | 20 % |
| h3 | one of 106 research fronts | 77.8 % | 20 % |

Accuracy climbs as the clusters get narrower, as it should: a broad domain is
heterogeneous by design, a research front has a tight vocabulary. The level the
citation analysis leans on most, the research fronts, scores best. Details in
[`results/`](results/README.md).

A second check concerns scale. Clustering and consolidation ran on samples, and
the rest of the corpus was labelled by assigning each publication to its most
similar reference document. On held-out documents, that assignment agrees with
the consolidated K-Means label in 84 % (h1) to 88 % (h3) of cases; see
[`code/04_subdiscipline_clustering/assignment_accuracy/`](code/04_subdiscipline_clustering/assignment_accuracy/).

---

## What is in this repository

| | |
|---|---|
| [`code/`](code/README.md) | the full pipeline, one numbered stage per step of the study |
| [`src/clustervalidation/`](src/clustervalidation/) | the validation protocols, as an installable Python package |
| [`results/`](results/README.md) | all 3,000 validation trials, a paired prompt experiment, and the assignment-accuracy results |
| [`supplementary/`](supplementary/README.md) | the article's Supplementary Information: bloc citation shares for every cluster at all three levels (PDF + CSV) |
| [`paper/`](paper/) | the manuscript (`main.tex`, compiled `main.pdf`), its bibliography and the five figures it uses |
| [`data/`](data/README.md) | where the corpus lives, its schema, and the two small intermediate tables later stages read |
| [`tests/`](tests/) | test suite for `clustervalidation`: no network, no API key, no spend |

Changes between versions are listed in [`CHANGELOG.md`](CHANGELOG.md).

### Data

The labelled corpus is deposited on Zenodo as a **separate data record**,
[10.5281/zenodo.22791584](https://doi.org/10.5281/zenodo.22791584) (CC BY 4.0):
1,986,659 publications with their OpenAlex IDs, citation counts, country shares
and cluster labels, as a gzip-compressed SQLite database (1.2 GB; 4.3 GB
uncompressed), together with `taxonomy.csv`, which names every cluster, and
SHA-256 checksums. The two records are kept apart so that each carries its own
DOI and licence and the corpus is not re-archived with every code release.
[`data/published_dataset/README.md`](data/published_dataset/README.md) documents
the schema; `taxonomy.csv` and `SHA256SUMS.txt` are also in that folder.

### The pipeline, stage by stage

```
01  125 survey papers ──KeyBERT──▶ candidate terms ──manual curation──▶ 279 search terms
                                                                          │
02  ◀─────────────── Boolean-OR query, batched, deduplicated ──────────────┘
    3,346,705 OpenAlex records
                                                                          │
03  abstracts reconstructed · country shares derived · completeness filter
    1,986,659 publications
                                                                          │
04  SPECTER embeddings ─▶ K-Means, k = 50 / 30 (over-segment) ─▶ contrastive terms + UMAP
                                        │
                          human consolidation ─▶ mapping ─▶ labels via reference documents
                          repeated at h1 → h2 → h3
    5 domains · 31 fields · 106 research fronts
                    │                                    │
05  fractional citation sum per cluster per bloc    07  document-intrusion validation
                    │                                    40.5 / 64.7 / 77.8 %
06  semantic landscape (Figure 1) · bloc heatmap (Figure 2)
```

Each stage directory has its own README explaining what it does, **why it does
it that way**, how to run it, and what it reads and writes.

---

## Quick start

Requires Python 3.10+ and, for the validation protocols, a
[DeepSeek](https://platform.deepseek.com) API key.

```bash
git clone https://github.com/Pigeon-Effect/qss_publication.git
cd qss_publication

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

That installs the validation package and the test suite. The numbered pipeline
stages in `code/` need further libraries (KeyBERT, SPECTER, UMAP, matplotlib
and others), installed with `pip install -e ".[pipeline]"`.

Provide the API key through a local `.env` file or the environment. `.env` is
gitignored; `.env.example` documents the variable name.

```bash
cp .env.example .env                      # then fill in the key
export DEEPSEEK_API_KEY='sk-...'          # bash / zsh — takes precedence
$env:DEEPSEEK_API_KEY = 'sk-...'          # PowerShell
```

Download the corpus from Zenodo and place it at `data/merged_works_labeled.db`
(see [`data/README.md`](data/README.md)), then:

```bash
# Inspect the corpus — no API calls, no cost
python -m clustervalidation inspect --level h3

# See exactly what would be sent, without spending anything
python -m clustervalidation intrusion --level h3 --trials 1000 --dry-run

# Reproduce the reported run
python -m clustervalidation intrusion --level h3 --trials 1000 \
    --prompt decisive --max-tokens 8000 --seed 20250628
```

Reports land in `results/<protocol>/` as files sharing a stem: `.json`
(manifest and summary), `.jsonl` (one record per trial), `.txt` (readable
transcript) and `.log` (run log).

Regenerating the tables and figures needs no API key:

```bash
python code/05_impact_analysis/fractional_citation_share.py
python code/05_impact_analysis/supplementary_tables.py
python code/06_visualization/bloc_share_heatmap_cluster_across_microclusters.py   # Figure 2
python code/06_visualization/semantic_landscape/render_landscape.py              # Figure 1
```

Figure 1 redraws from the layout deposited alongside it, so it needs neither the
corpus nor a GPU. Rebuilding that layout from the corpus is a separate step; see
[`code/06_visualization/semantic_landscape/`](code/06_visualization/semantic_landscape/).

---

## The validation protocols in detail

### Document intrusion detection

A panel of `--panel-size` documents is built: `panel_size - 1` genuine members
of a target cluster plus one intruder drawn from a different cluster at the same
level, in random order, with titles and abstracts truncated to `--max-words`.
Clusters, not documents, are sampled uniformly, so small clusters are tested as
often as large ones. The model receives the panel and nothing else — no cluster
label, no terms, no hint — and names the intruder. The random-guess baseline is
1 / panel size, so 20 % at the default five.

**The reported configuration** is `deepseek-v4-flash` with reasoning enabled,
the `decisive` prompt variant, 200-word truncation, five-document panels, an
8,000-token ceiling and seed 20250628, at 1,000 trials per level.

The `decisive` variant exists for a measured reason. In the h3 pilot, wrong
answers reasoned 4.1× longer than correct ones (mean 6,820 vs 1,680
characters): testing "what if paper *k* is the intruder" for every *k* re-reads
the panel again and again, where labelling each paper once and taking the
minority reads it once. `decisive` asks for that single pass and forbids
revisiting. Whether another framing would have scored higher is tested directly
in [`results/prompt_experiments/`](results/prompt_experiments/README.md): on
identical panels, the alternative scores 9 percentage points *lower* at every
level.

### Coherence rating

A second protocol, Likert coherence rating after Tan and D'Souza (2025), is
implemented in the package: a sample from a single cluster is rated 1–5 for
whether it forms one recognisable unit. It is **not** what the article reports.
In development it proved far more sensitive to the wording of the rubric than to
the clusters being rated, and an absolute judgement that unstable cannot carry
an external validation.

### How a verdict is extracted

Models do not always emit the requested verdict line. Extraction proceeds from
the most explicit pattern to the least and **records which rule fired**, so a
value read off an explicit marker is distinguishable from one recovered by a
fallback. Two recovery paths appear in the reported runs:

- **Reasoning-trace recovery.** A model that states its answer inside its
  reasoning but never writes the final line has answered; the verdict is read
  from the trace. This fired 22 / 11 / 7 times across h1 / h2 / h3.
- **Forced choice.** A response cut off by the token ceiling is passed to a
  second, non-reasoning model (`deepseek-chat`) that only reads the truncated
  trace and reports which paper it was converging on, without seeing the panel.
  This fired 103 / 70 / 37 times.

Neither path guesses. Across all 3,000 trials, `forced_guesses` is 0 and
`unparsed_responses` is 0.

---

## Reproducibility

**Panel construction is deterministic.** A dedicated seeded generator rebuilds
the identical panel sequence from the same `--seed`, `--level`, `--trials`,
`--panel-size` and `--max-words`.

**Model responses are not.** Sampling is non-deterministic server-side, so
accuracy varies between runs over the same panels. At n = 1,000 the sampling
error on one accuracy figure is roughly ±3 percentage points.

**Prompts are versioned, not edited.** Every wording ever run is registered by
name in `prompts.py`; new wordings get new names, so no result silently loses
the prompt it was produced with.

**The taxonomy rests on human judgement.** Applying a consolidation decision is
mechanical, but the decisions themselves are not reproducible: another labeler
would draw some boundaries differently. The scripts in stage 04 document the
procedure; the authoritative taxonomy is the set of labels deposited with the
corpus, which every downstream stage reads.

**The survey papers are not redistributed.** They are third-party publications.
The vocabulary built from them,
[`search_terms.txt`](code/01_keyword_construction/search_terms.txt), is
deposited, so retrieval and everything after it runs without them.

---

## Command reference

| Flag | Default | Notes |
|---|---|---|
| `--level` | *required* | `h1`, `h2` or `h3` |
| `--model` | `deepseek-v4-flash` | also `deepseek-chat`, `deepseek-reasoner`, `deepseek-v4-pro` |
| `--prompt` | `reasoned` / `tan_dsouza` | `decisive` is the reported intrusion variant; see `--help` for all |
| `--trials` | `100` | |
| `--seed` | `20250628` | fixes panel construction |
| `--max-words` | `200` | abstract truncation |
| `--panel-size` | `5` | changes the random baseline |
| `--max-tokens` | model default (3,000) | 8,000 in the reported runs |
| `--db` | `data/merged_works_labeled.db` | |
| `--dry-run` | off | build panels, print one prompt, make no API calls |

Every one of these is written into the run manifest, so a result file always
carries the parameters that produced it.

### Environment variables

| Variable | Default | Used by |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | stage 07 |
| `OPENALEX_MAILTO`, `OPENALEX_API_KEY` | — | stage 02 |
| `QSS_INTERIM_DIR` | `data/interim/` | stages 01–06 |
| `QSS_DB_PATH` | `data/merged_works_labeled.db` | stage 04 TF-IDF and assignment accuracy, stage 07 |
| `QSS_H3_DB` | `data/interim/h1_cluster_subsets/engineering_dataset.db` | stage 04 h3 scripts |
| `QSS_SEARCH_TERMS` | `code/01_keyword_construction/search_terms.txt` | stages 01–02 |
| `QSS_SURVEY_PDF_DIR` | `data/interim/ai_discipline_surveys_pdf/` | stage 01 |
| `QSS_SURVEY_TXT_DIR` | `data/interim/ai_discipline_surveys_txt/` | stage 01 |
| `QSS_CITSHARE_CSV` | `data/interim/citshare_h3x4entities.csv` | stages 05–06 |

No script contains an absolute path; each derives the repository root from its
own location, so a fresh clone runs anywhere.

---

## Repository layout

```
├── code/                            full pipeline, one stage per step of the study
│   ├── 01_keyword_construction/     survey papers ─▶ KeyBERT ─▶ the 279 search terms
│   ├── 02_data_collection/          batched, resumable OpenAlex retrieval and merge
│   ├── 03_data_processing/          abstract reconstruction, country shares, filtering
│   ├── 04_subdiscipline_clustering/
│   │   ├── finding_optimal_k/       evidence that no natural k exists
│   │   ├── SPECTER/                 h1 / h2 / h3: over-segment, then consolidate
│   │   └── assignment_accuracy/     how faithfully the full corpus inherits the sample clustering
│   ├── 05_impact_analysis/          fractional citation sum; supplementary tables; taxonomy.csv
│   ├── 06_visualization/
│   │   ├── semantic_landscape/      projection and density summarization (Figure 1)
│   │   └── …                        citation-share heatmap (Figure 2)
│   └── 07_llm_validation/           document intrusion — see src/clustervalidation/
├── src/clustervalidation/
│   ├── config.py                    models, pricing, taxonomy levels, RunConfig
│   ├── corpus.py                    SQLite loading, cluster grouping, truncation
│   ├── llm.py                       API client, retries, outage handling, cost accounting
│   ├── parsing.py                   verdict/rating extraction with rule tracking
│   ├── prompts.py                   every registered prompt variant
│   ├── reporting.py                 JSON / JSONL / text reports
│   ├── cli.py                       command-line interface
│   └── protocols/
│       ├── intrusion.py             document-intrusion detection
│       └── coherence.py             Likert coherence rating
├── results/
│   ├── intrusion/                   the three reported runs, 1,000 trials each
│   ├── prompt_experiments/          paired comparison of two prompt framings
│   └── assignment_accuracy/         held-out agreement of the label assignment
├── supplementary/                   Supplementary Information (PDF + CSV)
├── paper/                           main.tex, main.pdf, bibliography.bib, figures/
├── data/                            corpus location, schema, taxonomy.csv, checksums
└── tests/                           75 tests, no network required
```

Compile the manuscript with `latexmk -pdf main.tex` from `paper/` (requires
`biber`), and the Supplementary Information the same way from `supplementary/`.

---

## Tests

```bash
pytest
```

75 tests covering verdict and rating extraction, panel and sample construction,
seed determinism, corpus loading, forced-choice fallback, outage retry, and an
end-to-end run against a synthetic SQLite corpus with a stub client. No API key
or network access required; nothing in the suite spends credit.

---

## Citing

Cite **both** the article and this compendium. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff).

## Licence

| Content | Licence |
|---|---|
| Software (`src/`, `code/`, `tests/`) | [MIT](LICENSE) |
| Results, tables, figures, manuscript | [CC BY 4.0](LICENSE-DATA) |

Source metadata from [OpenAlex](https://openalex.org), released by OurResearch
under CC0.

## References

Chang, J., Boyd-Graber, J., Gerrish, S., Wang, C., & Blei, D. M. (2009).
Reading tea leaves: How humans interpret topic models. *NeurIPS 22*.

Cohan, A., Feldman, S., Beltagy, I., Downey, D., & Weld, D. S. (2020).
SPECTER: Document-level representation learning using citation-informed
transformers. *ACL 2020*.

Priem, J., Piwowar, H., & Orr, R. (2022). OpenAlex: A fully-open index of
scholarly works, authors, venues, institutions, and concepts.
*arXiv:2205.01833*.
