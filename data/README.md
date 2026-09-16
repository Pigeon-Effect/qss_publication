# Data

This directory is where the labelled corpus is expected at runtime. The corpus
is too large for a git repository, so it is deposited on Zenodo as a data
record alongside the code:
**[10.5281/zenodo.22791584](https://doi.org/10.5281/zenodo.22791584)**. See
[Obtaining the corpus](#obtaining-the-corpus).

```
data/
├── merged_works_labeled.db     the final labelled corpus (Zenodo; see below)
├── published_dataset/          README and checksums for the Zenodo deposit
└── interim/                    everything the pipeline builds on the way there
```

## `interim/`

Where stages 01–06 read and write their intermediate artefacts. Two files here
are small and are carried in git, because stage 06 and the Supplementary
Information read them: `citshare_h3x4entities.csv` and
`fractional_citations_by_country_h3.csv`, both written by
[stage 05](../code/05_impact_analysis/). Everything else is multi-gigabyte
derived data, rebuilt by re-running the pipeline. Override the location with
`QSS_INTERIM_DIR`.

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
├── citshare_h3x4entities.csv                       stage 05 output → stage 06 input (in git)
└── fractional_citations_by_country_h3.csv          stage 05 output, per country (in git)
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
| File size | 4.3 GB (1.2 GB gzipped, as deposited) |

These match the figures reported in the manuscript. Because every row carries
an abstract, the protocols' `cleaned_abstract IS NOT NULL` filter removes
nothing on this corpus — it guards against a differently-built database.

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT | OpenAlex work ID; primary identifier, unique |
| `doi` | TEXT | may be null (32,595 records); OpenAlex IDs cover records without one |
| `title` | TEXT | empty for 1,897 records |
| `publication_year` | INTEGER | 2020–2024 |
| `cited_by_count` | INTEGER | complete for all records; an OpenAlex snapshot of July 2025 |
| `country_of_origin` | TEXT | JSON list of country–share tuples summing to 1 |
| `cleaned_abstract` | TEXT | reconstructed plain-text abstract — **the field the protocols read** |
| `h1_cluster` | INTEGER | macro level: domain |
| `h2_cluster` | INTEGER | meso level: field within domain |
| `h3_cluster` | INTEGER | micro level: research front within field |

The working database carried twenty further columns retrieved from OpenAlex
(`language`, `type`, `is_oa`, `authorships`, `fwci`, `cited_by_percentile_year`,
`referenced_works`, `abstract_inverted_index`, `grants` and others). They are
not part of the deposit: no analysis in this repository reads them, and most
change as OpenAlex updates its records, so a snapshot of them would go stale.
Current values can be fetched from the [OpenAlex API](https://docs.openalex.org)
with the `id` column. `country_of_origin`, `cleaned_abstract` and the three
cluster labels are derived by us and exist nowhere else.

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

**Download it from the Zenodo data record**
[10.5281/zenodo.22791584](https://doi.org/10.5281/zenodo.22791584), which
accompanies the article. That deposit is the authoritative copy: it carries the
published cluster labels, which the code alone cannot reproduce exactly.

```bash
gunzip merged_works_labeled_published.db.gz
mv merged_works_labeled_published.db data/merged_works_labeled.db
```

The deposited file is named `merged_works_labeled_published.db` to distinguish
it from the full working database. Rename it as above, or keep its own name and
pass `--db` to every command. Table and column names are unchanged either way.
`data/published_dataset/` holds the deposit's README and its SHA-256 checksums.

Rebuilding it from OpenAlex instead is possible — the retrieval and
topic-modeling pipeline is [`code/`](../code/README.md) stages 01–04, each
stage's inputs and outputs documented in its own README — but it will not
reproduce the same cluster identifiers. Clustering depends on random seeds, and
the expert-consolidation step was recorded only in part. A rebuild yields a
similar taxonomy, not this one.

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
