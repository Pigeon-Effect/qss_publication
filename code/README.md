# Code

The processing pipeline behind [`paper/main.pdf`](../paper/main.pdf), organized
as one numbered stage per step of the study. Each stage's own `README.md`
explains what it does, **why it does it that way**, how to run it, and what it
reads and writes.

| # | Stage | Manuscript section |
|---|---|---|
| [01](01_keyword_construction/) | Keyword construction | §2.2 Dataset Extraction Pipeline |
| [02](02_data_collection/) | Data collection (OpenAlex retrieval) | §2.1–2.2 |
| [03](03_data_processing/) | Data processing (quality control) | §2.3 Quality Control and Postprocessing |
| [04](04_subdiscipline_clustering/) | Subdiscipline clustering (embedding → over-segmentation → expert consolidation) | §3.1 Hierarchical Topic Modeling |
| [05](05_impact_analysis/) | Impact analysis (fractional citation sum) | §3.2 Assessment of Research Impact |
| [06](06_visualization/) | Visualization (semantic landscape; citation-share heatmap) | §4 Findings |
| [07](07_llm_validation/) | LLM-based cluster validation | §4.1 External Validation via Document Intrusion |

## How the pipeline fits together

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

The two branches below stage 04 are independent: stage 07 tests whether the
taxonomy is real, stages 05–06 measure what it shows.

## Conventions

**Paths.** No script contains an absolute path. Each derives the repository
root from its own location (`Path(__file__).resolve().parents[N]`), so a fresh
clone runs anywhere. Locations that legitimately vary are environment
variables:

| Variable | Default | Used by |
|---|---|---|
| `QSS_INTERIM_DIR` | `data/interim/` | stages 01–04, 06 |
| `QSS_DB_PATH` | `data/merged_works_labeled.db` | stage 04 TF-IDF, stage 07 |
| `QSS_H3_DB` | `data/interim/h1_cluster_subsets/engineering_dataset.db` | stage 04 h3 scripts |
| `QSS_SEARCH_TERMS` | `code/01_keyword_construction/search_terms.txt` | stages 01–02 |
| `QSS_SURVEY_PDF_DIR` | `data/interim/ai_discipline_surveys_pdf/` | stage 01 |
| `QSS_SURVEY_TXT_DIR` | `data/interim/ai_discipline_surveys_txt/` | stage 01 |
| `QSS_CITSHARE_CSV` | `data/interim/citshare_h3x4entities.csv` | stages 05–06 |

**Data.** The labelled corpus and the large intermediate databases are derived
data measured in gigabytes; they are deposited with the article rather than
carried in git. The two small stage-05 outputs that stages 06 and the
Supplementary Information read are in `data/interim/`. See
[`data/README.md`](../data/README.md).

**Credentials.** Read from a local `.env` (copy
[`.env.example`](../.env.example)) or the environment. `.env` is gitignored.
Nothing in this repository contains a key.

**Outputs.** Scripts write beside themselves: `resources/` for artefacts that
are part of the archived record — the interactive UMAPs and abstract summaries
the expert consolidated from — and `output/` for regenerable products.

## What a re-run reproduces

Two things about this pipeline are worth knowing before running it.

The **survey papers** that seeded the keyword extraction (stage 01) are
third-party publications and are not redistributed; the input folder is
assembled locally. The vocabulary they produced —
[`search_terms.txt`](01_keyword_construction/search_terms.txt) — is deposited,
so every stage from 02 onwards runs without them.

The **taxonomy** (stage 04) is the product of an expert consolidating an
over-segmented K-Means partition, recorded as plain remapping dictionaries and
applied by nearest reference abstract, so each labelling step is an inspectable
lookup rather than an unrepeatable act. Clustering and projection still depend
on random seeds, so a re-run reproduces the procedure and the structure it
finds, not an identical set of boundaries. The labels deposited with the corpus
are the authoritative version of the taxonomy, and every downstream stage reads
them.

See the top-level [`README.md`](../README.md) for installation, reproducibility
notes and licensing.
