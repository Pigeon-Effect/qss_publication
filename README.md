# Mapping the AI Research of China, the US and the EU

Research compendium for:

> **Mapping the AI Research of China, the US and the EU: A Scientometric
> Analysis of Citation Shares within AI Subdisciplines (2020–2025)**
> Julius Pfundstein, Thomas Efer, Manuel Burghardt
> Computational Humanities, University of Leipzig

This repository holds everything behind that article: the code that built the
dataset, the code that analysed it, the complete record of the validation
experiments, the cluster-level result tables, and the manuscript sources.

---

## What the study did, in plain terms

Almost every government with global ambitions now has an AI strategy, and the
question of who is "ahead" is usually answered with a single number — total
publications, total citations, a benchmark score. Those numbers hide something
important: **AI is not one field.** A country can dominate computer vision and
be nearly absent from AI ethics, and a single ranking will never show that.

So this study did two things.

**First, it drew a map of AI research.** It collected the metadata for
1,986,659 research publications from 2020–2025 — every paper that matched a
carefully built list of 279 AI-related search terms — and asked what
subdisciplines they naturally fall into. Rather than trusting an algorithm to
find "the right number" of topics (there isn't one; the evidence for that is in
[stage 04](code/04_subdiscipline_clustering/)), the pipeline deliberately cuts
the corpus into far too many pieces and then has a human expert consolidate
them into meaningful units, three times over. The result is a three-level
taxonomy: **5 broad domains, 31 fields inside them, and 106 specific research
fronts inside those.**

**Second, it measured who holds influence where.** Every publication's
citations were split among the countries that produced it, in proportion to
where its authors work. Summed inside each of the 106 research fronts and
grouped into three blocs — China, the United States, and the EU-27 — this shows
each bloc's share of the scholarly attention in that specific corner of AI.

The headline results: the three blocs together hold 58.9 % of publications and
63.5 % of citations, but their profiles differ sharply. China leads computer
science and engineering and trails in the social and health sciences. The US
leads health science and the older computer-science fields. The EU-27 is the
most balanced, with the fewest blind spots. Governance and ethics — the part of
AI that dominates public debate — accounts for 2.7 % of the literature.

### How we checked the map was real

A taxonomy built partly by a human expert has an obvious weakness: the person
who drew the boundaries is also the person who thinks they look right. That is
not evidence. So the clusters were tested from outside, using a protocol
adapted from Chang et al.'s *reading tea leaves* (2009).

The idea is simple. Take four papers from one cluster, add one "intruder" from
a different cluster, shuffle them, and ask a judge which one does not belong —
showing nothing but the papers themselves, no labels, no hints. If the cluster
is a genuine, coherent topic, the intruder sticks out. If the cluster is an
artefact, it doesn't. Detection accuracy is therefore a measure of how real the
cluster is. The judge here is a large language model, which makes it feasible
to run this **3,000 times** where human annotators would not be.

The result, at 1,000 trials for each of the three levels:

| Level | What a cluster is here | Accuracy | Chance |
|---|---|---:|---:|
| Macro (h1) | one of 5 broad domains | 40.5 % | 20 % |
| Meso (h2) | one of 31 fields | 64.7 % | 20 % |
| Micro (h3) | one of 106 research fronts | 77.8 % | 20 % |

Accuracy climbs as the clusters get more specific, which is exactly what should
happen. A broad domain like *Natural Science* legitimately contains structural
engineering, ionospheric physics and land-use change all at once, so an
intruder hides easily. A narrow research front like *Cancer Detection &
Screening* has a tight vocabulary, and an outsider is obvious. Every level beats
chance by a wide margin, and the levels the analysis leans on hardest are the
ones that score best.

---

## What is in this repository

| | |
|---|---|
| [`code/`](code/README.md) | the full pipeline, one numbered stage per step of the study |
| [`src/clustervalidation/`](src/clustervalidation/) | the validation protocols, as an installable Python package |
| [`results/`](results/README.md) | the complete record of all 3,000 validation trials, plus one paired prompt experiment |
| [`supplementary/`](supplementary/) | cluster-level result tables (PDF + CSV) — the article's Supplementary Information |
| [`paper/`](paper/) | manuscript source, figures, bibliography |
| [`data/`](data/README.md) | where the corpus lives, and how to get it |
| [`tests/`](tests/) | test suite for `clustervalidation` — no network, no API key, no spend |

The corpus itself — 1,986,659 labelled records with their citation counts,
country shares and cluster labels — is deposited on Zenodo as a **separate data
record**, [10.5281/zenodo.22791584](https://doi.org/10.5281/zenodo.22791584),
rather than carried in git: 1.2 GB gzipped, 4.3 GB as a SQLite file. This
repository is the software record; the two are deposited apart so each carries
its own DOI and licence (code MIT, data CC BY 4.0) and so the corpus is not
re-archived with every code release. See [`data/README.md`](data/README.md) for
the schema and how to place the file.

### The pipeline, stage by stage

```
01  survey papers ──KeyBERT──▶ candidate terms ──manual curation──▶ 279 search terms
                                                                          │
02  ◀─────────────── Boolean-OR query, batched, deduplicated ──────────────┘
    3,346,705 raw OpenAlex records
                                                                          │
03  abstracts reconstructed · country shares derived · length/completeness filter
    1,986,659 publications
                                                                          │
04  SPECTER embeddings ─▶ K-Means k=50 (over-segment) ─▶ TF-IDF + UMAP
                                        │
                          expert consolidation ─▶ remap dict ─▶ labels
                          repeated at h1 → h2 → h3
    5 domains · 31 fields · 106 research fronts
                    │                                    │
05  fractional citation sum per cluster per bloc    07  document-intrusion
                    │                                    validation
06  semantic landscape · Figure 2 heatmap                 40.5 / 64.7 / 77.8 %
```

Each stage directory carries its own README explaining what it does, **why it
does it that way**, how to run it, and what it reads and writes.

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
stages in `code/` bring their own libraries — KeyBERT, SPECTER, UMAP,
matplotlib and so on — which are installed with `pip install -e ".[pipeline]"`
when you want to re-run the pipeline rather than the protocols.

Provide the API key through a local `.env` file or the environment. `.env` is
gitignored; `.env.example` documents the variable name.

```bash
cp .env.example .env                      # then fill in the key
export DEEPSEEK_API_KEY='sk-...'          # bash / zsh — takes precedence
$env:DEEPSEEK_API_KEY = 'sk-...'          # PowerShell
```

Place the corpus at `data/merged_works_labeled.db` (see
[`data/README.md`](data/README.md)), then:

```bash
# Inspect the corpus — no API calls, no cost
python -m clustervalidation inspect --level h3

# See exactly what would be sent, without spending anything
python -m clustervalidation intrusion --level h3 --trials 1000 --dry-run

# Reproduce the reported run
python -m clustervalidation intrusion --level h3 --trials 1000 \
    --prompt decisive --max-tokens 8000 --seed 20250628
```

Reports land in `results/<protocol>/` as four files sharing a stem: `.json`
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
hierarchy level, in randomised order, with titles and abstracts truncated to
`--max-words`. The model receives the panel and nothing else — no cluster label,
no TF-IDF terms, no indication of which document came from where — and names the
intruder. The random-guess baseline is 1 / panel size, so 20 % at the default
five.

**The reported configuration** is `deepseek-v4-flash` with reasoning enabled,
the `decisive` prompt variant, 200-word truncation, five-document panels, an
8,000-token ceiling and seed 20250628, at 1,000 trials per level.

The `decisive` variant exists for a measurable reason. In the h3 pilot, wrong
answers reasoned 4.1× longer than correct ones (mean 6,820 vs 1,680 characters),
and the one truncated trial spent ~3,500 tokens cycling hypotheses without ever
committing. The cause is structural: testing "what if paper *k* is the intruder"
for every *k* re-reads the panel O(n²) times, where labelling each paper once
and taking the minority is O(n). `decisive` forbids the re-reading and supplies
an explicit tie-break, so the model has an exit from the deadlock rather than
looping into the token ceiling.

### Coherence rating

A second protocol, Likert coherence rating after Tan and D'Souza (2025), is
implemented in the package: a sample from a single cluster is rated 1–5 for
whether it forms one recognisable unit. It is **not** what the article reports.
In development it proved far more sensitive to the wording of the rubric than to
the clusters being rated — the same clusters can move most of the way across the
five-point scale on a rubric change alone. An absolute judgement that unstable
cannot carry an external validation, which is why document intrusion does.

### How a verdict is extracted

Models do not always emit the requested verdict line. Extraction proceeds from
the most explicit pattern to the least and **records which rule fired**, so a
value read off an explicit marker is distinguishable from one recovered by a
last-resort rule. Every run reports the counts.

Two specific recovery paths matter, because both appear in the reported runs:

- **Reasoning-trace recovery.** A model that states its answer inside its
  reasoning but never writes the final line is not a failed trial. The verdict
  is read from the trace. This fired 22 / 11 / 7 times across h1 / h2 / h3.
- **Forced choice.** A response cut off by the token ceiling mid-reasoning is
  passed to a second, non-reasoning model (`deepseek-chat`) whose only job is to
  read the truncated trace and report which paper it was converging on. This
  fired 103 / 70 / 37 times.

Neither path guesses. Across all 3,000 trials, `forced_guesses` is 0 and
`unparsed_responses` is 0 — every trial was scored from something the model
actually said.

---

## Reproducibility

**Panel construction is deterministic.** A dedicated seeded generator is used
rather than global random state, so the same `--seed`, `--level`, `--trials`,
`--panel-size` and `--max-words` rebuild the identical panel sequence.

**Model responses are not.** Sampling is non-deterministic server-side and the
API is a moving target, so accuracy will vary between runs over the same panels.
At n = 1,000 the sampling error on a single accuracy figure is roughly ±3 pp.

**Prompts are versioned, not edited.** Every wording ever run is registered by
name in `prompts.py`. Editing a registered variant would silently invalidate the
results that cite it, so new wordings get new names.

**The taxonomy is a human judgement, recorded as a lookup table.** The expert's
consolidation of each over-segmented partition is stored as a plain remapping
dictionary and applied by nearest reference abstract, so the step is
inspectable and re-runnable rather than an unrepeatable act. Clustering and
projection still depend on random seeds, so an independent re-run of stage 04
reproduces the procedure and the structure it finds, not an identical set of
boundaries. The published labels are the authoritative version of the taxonomy
and are deposited with the corpus, and every downstream stage reads them.

The survey papers that seeded the keyword extraction are third-party
publications and are not redistributed. The vocabulary built from them,
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
| `--max-tokens` | `3000` | 8,000 in the reported runs |
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
| `QSS_DB_PATH` | `data/merged_works_labeled.db` | stage 04 TF-IDF, stage 07 |
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
├── code/                          full pipeline, one stage per manuscript section
│   ├── 01_keyword_construction/   survey papers ─▶ KeyBERT ─▶ the 279 search terms
│   ├── 02_data_collection/        batched, resumable OpenAlex retrieval and merge
│   ├── 03_data_processing/        abstract reconstruction, country shares, filtering
│   ├── 04_subdiscipline_clustering/
│   │   ├── finding_optimal_k/     evidence that no natural k exists
│   │   └── SPECTER/               h1 / h2 / h3: over-segment, then consolidate
│   ├── 05_impact_analysis/        fractional citation sum; supplementary tables
│   ├── 06_visualization/
│   │   ├── semantic_landscape/    projection, density summarization (Figure 1)
│   │   └── …                      citation-share heatmap (Figure 2)
│   └── 07_llm_validation/         document intrusion — see src/clustervalidation/
├── src/clustervalidation/
│   ├── config.py                  models, pricing, taxonomy levels, RunConfig
│   ├── corpus.py                  SQLite loading, cluster grouping, truncation
│   ├── llm.py                     API client, retries, outage handling, cost accounting
│   ├── parsing.py                 verdict/rating extraction with rule tracking
│   ├── prompts.py                 every registered prompt variant
│   ├── reporting.py               JSON / JSONL / text reports
│   ├── cli.py                     command-line interface
│   └── protocols/
│       ├── intrusion.py           document-intrusion detection
│       └── coherence.py           Likert coherence rating
├── results/intrusion/             the three reported runs, 1,000 trials each
├── supplementary/                 cluster-level tables (PDF + CSV)
├── paper/                         manuscript, figures, bibliography, archive/
├── data/                          corpus location and schema documentation
└── tests/                         75 tests, no network required
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
[`CITATION.cff`](CITATION.cff); GitHub renders a "Cite this repository" button
from it.

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
