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
| [06](06_visualization/) | Visualization | §4 Findings |
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
06  Figure 2 heatmap · Supplementary tables              40.5 / 64.7 / 77.8 %
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
| `QSS_SEARCH_TERMS` | `code/01_keyword_construction/search_terms.txt` | stage 02 |
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

## What the code cannot reproduce on its own

Three things about this pipeline are worth knowing before running it.

The **survey PDFs** that seeded the keyword extraction (stage 01) are
third-party copyrighted material and are not redistributed; the input folder has
to be assembled locally.

The **expert consolidation** at h2 and h3 (stage 04) was recorded only in part.
`H3_MAP` held one slice at a time and was overwritten between runs, and a final
h2 pass that moved three cluster groups between macro-domains left no record.
Stage 04's README states exactly which files diverge from the published taxonomy
and how. The published labels are authoritative and ship with the corpus; a
re-run of stage 04 produces a similar but not identical taxonomy. Nothing has
been reconstructed by guesswork, because inventing a cluster assignment would
fabricate part of the taxonomy.

**Figure 1's** rendering code — the density-summarised UMAP with its taxonomy
legend — is not part of this deposit. The per-level UMAP projections stage 04
writes are the diagnostic scatterplots the expert consolidated from, not that
figure.

See the top-level [`README.md`](../README.md) for installation, reproducibility
notes and licensing.
