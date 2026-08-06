# Code

The processing pipeline behind [`paper/main.pdf`](../paper/main.pdf), organized
as one numbered stage per step of the manuscript. Each stage's own `README.md`
explains what it does, **why it does it that way**, how to run it, and what it
reads and writes.

| # | Stage | Manuscript section | State |
|---|---|---|---|
| [01](01_keyword_construction/) | Keyword construction | §2.2 Dataset Extraction Pipeline | ✅ code present; the curated 279-term list itself is not |
| [02](02_data_collection/) | Data collection (OpenAlex retrieval) | §2.1–2.2 | ✅ |
| [03](03_data_processing/) | Data processing (quality control) | §2.3 Quality Control and Postprocessing | ✅ |
| [04](04_subdiscipline_clustering/) | Subdiscipline clustering (embedding → over-segmentation → expert consolidation) | §3.1 Hierarchical Topic Modeling | ✅ with three documented gaps |
| [05](05_impact_analysis/) | Impact analysis (fractional citation sum) | §3.2 Assessment of Research Impact | ❌ not yet imported |
| [06](06_visualization/) | Visualization | §4 Findings | ◐ Figure 2 present, Figure 1 missing |
| [07](07_llm_validation/) | LLM-based cluster validation | §4.1 External Validation via Document Intrusion | ✅ packaged as [`clustervalidation`](../src/clustervalidation/) |

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
06  Figure 2 heatmap                                     46% / 75% / 84%+
```

## Conventions

**Paths.** No script contains an absolute path. Each derives the repository
root from its own location (`Path(__file__).resolve().parents[N]`), so a fresh
clone runs anywhere. Locations that legitimately vary are environment
variables:

| Variable | Default | Used by |
|---|---|---|
| `QSS_INTERIM_DIR` | `data/interim/` | stages 01–04, 06 |
| `QSS_DB_PATH` | `data/merged_works_labeled.db` | stage 04 TF-IDF, stage 07 archive |
| `QSS_H3_DB` | `data/interim/h1_cluster_subsets/engineering_dataset.db` | stage 04 h3 scripts |
| `QSS_SEARCH_TERMS` | `code/01_keyword_construction/search_terms.txt` | stage 02 |
| `QSS_SURVEY_TXT_DIR` | `data/interim/ai_discipline_surveys_txt/` | stage 01 |
| `QSS_CITSHARE_CSV` | `data/interim/citshare_h3x4entities.csv` | stage 06 |

**Data.** Everything under `data/` is gitignored except its README. Corpora and
intermediate databases are derived data, measured in gigabytes, and are
rebuilt rather than shipped — see [`data/README.md`](../data/README.md).

**Credentials.** Read from a local `.env` (copy
[`.env.example`](../.env.example)) or the environment. `.env` is gitignored.
Nothing in this repository contains a key.

**Outputs.** Scripts write beside themselves: `resources/` for artefacts that
are part of the archived record (the interactive UMAPs and abstract summaries
the expert consolidated from), `output/` for regenerable products.

## What is not here

Stated plainly so a reader does not go looking:

- The **279-term search list** (stage 01, step 4 — manual curation).
- **Stage 05** in its entirety, the fractional citation sum.
- **Figure 1's** generating code (stage 06).
- The record of the **final h2 consolidation pass** that moved three cluster
  groups between macro-domains (stage 04, "Known gaps").
- The **h3 mappings for 30 of the 31 slices** — `H3_MAP` held one slice at a
  time and was overwritten between runs (stage 04).

See the top-level [`README.md`](../README.md) for installation,
reproducibility notes and licensing.
