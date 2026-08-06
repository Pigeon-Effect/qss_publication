# Data

This directory is where the labelled corpus is expected at runtime. **The
corpus itself is not distributed with this repository** — it is a ~20 GB SQLite
file, far beyond what a code repository or a Zenodo software record should
carry, and it is derived data that can be rebuilt from its open source.

```
data/
├── merged_works_labeled.db     the final labelled corpus (not in git; see below)
└── interim/                    everything the pipeline builds on the way there
```

Everything in this directory except this README is gitignored.

## `interim/`

Where stages 01–06 read and write their intermediate artefacts. Nothing here is
distributed: all of it is derived data, rebuilt by re-running the pipeline.
Override the location with `QSS_INTERIM_DIR`.

```
data/interim/
├── ai_discipline_surveys_txt/                      stage 01 input: survey papers as .txt
├── openalex_raw/                                   stage 02: per-term-group results + cursors
├── openalex_ai_works_2020-2024_raw.json            stage 02: merged, deduplicated
├── openalex_ai_works_merged_deduplicated.db        stage 03 input (table `works`)
├── openalex_ai_works_merged_deduplicated_cleaned.db          + cleaned_abstract
├── openalex_ai_works_merged_deduplicated_country_with_origin.db  + country_of_origin
├── openalex_ai_works_merged_deduplicated_cleaned_filtered.db  after the quality filter
├── openalex_ai_works_sample_10k.db                 stage 04: finding_optimal_k
├── openalex_ai_works_sample_100k.db                stage 04: finding_optimal_k
├── openalex_ai_filtered_dataset_sample_100k.db     stage 04: h1 pre-clustering
├── h1_cluster_subsets/                             stage 04: one database per macro-domain
│   ├── <domain>_dataset.db                           full subset, labelled in place
│   └── <domain>_sample.db                            sample used for the UMAP figures
└── citshare_h3x4entities.csv                       stage 05 output → stage 06 input
```

`<domain>` is one of `computer_science`, `biomedical`, `social_science`,
`natural_science`, `engineering`.

## What the file contains

One table, `works_labeled`, holding OpenAlex publication metadata for the AI
corpus described in the manuscript, plus the three cluster labels assigned by
the topic-modeling pipeline.

Verified against the database itself:

| Property | Value |
|---|---|
| Rows in `works_labeled` | 1,986,659 |
| Rows with a usable `cleaned_abstract` | 1,986,659 (100 %) |
| Distinct `h1` clusters | 5 |
| Distinct `h2` clusters | 31 |
| Distinct `h3` clusters | 106 |
| File size | ~20 GB |

These match the figures reported in the manuscript. Because every row carries
an abstract, the protocols' `cleaned_abstract IS NOT NULL` filter removes
nothing on this corpus — it guards against a differently-built database.

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT | OpenAlex work ID; primary identifier |
| `doi` | TEXT | may be null; OpenAlex IDs cover records without one |
| `title` | TEXT | |
| `publication_year` | INTEGER | 2020–2025 |
| `language` | TEXT | filtered to English abstracts |
| `type`, `type_crossref` | TEXT | article, preprint, review, … |
| `is_oa` | INTEGER | open-access flag |
| `authorships` | TEXT | JSON; source for country attribution |
| `country_of_origin` | TEXT | JSON list of country–share tuples summing to 1 |
| `countries_distinct_count` | REAL | |
| `institutions_distinct_count` | REAL | |
| `cited_by_count` | INTEGER | complete for all records |
| `fwci` | REAL | field-weighted citation impact |
| `cited_by_percentile_year` | TEXT | |
| `referenced_works` | TEXT | JSON; outgoing references (~92% complete after filtering) |
| `abstract_inverted_index` | TEXT | JSON, as delivered by OpenAlex |
| `cleaned_abstract` | TEXT | reconstructed plain-text abstract — **the field the protocols read** |
| `h1_cluster` | INTEGER | macro level: domain |
| `h2_cluster` | INTEGER | meso level: field within domain |
| `h3_cluster` | INTEGER | micro level: research front within field |

Additional bibliographic columns (`relevance_score`, `host_organization_name`,
`source_issn_1`, `corresponding_author_ids`, `corresponding_institution_ids`,
`apc_list`, `apc_paid`, `biblio`, `grants`) are retained from retrieval but are
not used by the validation protocols.

### Cluster identifiers

The three label columns are hierarchical and are read as a **path**, not as
independent values. `h2_cluster = 4` is only meaningful relative to its parent,
so the code concatenates labels down to the requested level:

| Level | Identifier | Meaning |
|---|---|---|
| `h1` | `h1_cluster` | 5 domains |
| `h2` | `h1_cluster ‖ h2_cluster` | 31 fields |
| `h3` | `h1_cluster ‖ h2_cluster ‖ h3_cluster` | 106 research fronts |

This concatenation is defined once in `src/clustervalidation/config.py`
(`CLUSTER_ID_SQL`).

## Obtaining the corpus

The corpus is derived data. Three routes, in order of preference:

1. **Zenodo data deposit.** If a dataset record accompanies the published
   article, download the database from there and place it at
   `data/merged_works_labeled.db`.
2. **Rebuild from OpenAlex.** The retrieval and topic-modeling pipeline is in
   [`code/`](../code/README.md), stages 01–04, with each stage's inputs and
   outputs documented in its own README. Note two things: the 279-term search
   list is not currently in the repository (see
   [stage 01](../code/01_keyword_construction/README.md#what-is-not-here)), and
   clustering depends on random seeds and on an expert consolidation step whose
   record is incomplete, so an independent rebuild will not reproduce cluster
   identifiers exactly.
3. **Contact the authors.** See `CITATION.cff` for the corresponding author.

## Using a different path

Nothing requires the database to live here. Every command accepts `--db`:

```bash
python -m clustervalidation inspect --level h3 --db /mnt/big/corpus.db
```

## Provenance and licence

Source metadata retrieved from the [OpenAlex](https://openalex.org) API in July
2025, released by OurResearch under a CC0 public-domain dedication. Because the
youngest records were roughly six months old at retrieval, citation counts sit
well below the typical accumulation peak — a constraint that conditions every
citation-based figure in the manuscript.

Derived content in this repository is licensed CC BY 4.0; see
[`LICENSE-DATA`](../LICENSE-DATA).
