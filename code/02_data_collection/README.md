# 02 — Data collection

Retrieves the raw corpus from OpenAlex (manuscript §2.1–2.2). Corresponds to
[`data_extraction_pipeline_diagram.pdf`](../../paper/figures/data_extraction_pipeline_diagram.pdf).

## Why OpenAlex

The choice of scientific knowledge graph is a design decision with consequences
for every downstream number, so the reasoning is recorded here even though it
leaves no code. Three SKGs can credibly claim broad, current coverage —
Clarivate's *Web of Science*, Elsevier's *Scopus*, and OurResearch's
*OpenAlex*. This study needs an SKG that supplies abstracts (subdiscipline
identification depends on them), attributes publications to a country of
origin, and carries at least one impact indicator.

OpenAlex was chosen on two grounds. Coverage: 243.1 million indexed
publications, roughly four times the proprietary databases. And, decisively for
a study comparing world regions, **geographic balance**: measured as each
region's share in the SKG against its share in the ROAD reference database,
Scopus and Web of Science underrepresent Asia (coverage index below 0.4) and
overrepresent Europe (above 1.6), whereas OpenAlex sits close to parity across
the AI hotspots — Asia 0.97, Europe 1.01, North America 1.03. A study asking
how China, the US and the EU compare cannot be built on a corpus that is
already skewed between them.

The trade-off is lower metadata completeness than the curated databases, which
is what stage 03 then has to filter for.

OpenAlex has its own thematic taxonomy, but cross-database studies report it as
insufficiently granular and unstable across releases for application-oriented
work — hence the study builds its own topic model in stage 04.

## How the corpus was retrieved

OpenAlex caps the length of a single `search` expression, so the 279 terms from
[stage 01](../01_keyword_construction/) cannot be sent as one Boolean-OR query.
Retrieval therefore ran in batches: each run takes a slice of the term list,
queries it as one OR expression against the full-text index (titles and
abstracts), filters to `publication_year`, sorts newest first, and writes the
matching works into its own SQLite database. The batch databases are then merged
and deduplicated on the OpenAlex work id.

The runs that produced the published corpus:

| Terms | Years | Database |
|---|---|---|
| 1–50 | 2020 | `openalex_ai_works_1-50_search_terms_2020.db` |
| 1–50 | 2021 | `openalex_ai_works_1-50_search_terms_2021.db` |
| 1–50 | 2022–2024 | `openalex_ai_works_1-50_search_terms_2022-2024.db` |
| 51–100 | 2020–2024 | `openalex_ai_works_51-100_search_terms_2020-2024.db` |
| 101–200 | 2020–2024 | `openalex_ai_works_101-200_search_terms_2020-2024.db` |
| 201–279 | 2020–2024 | `openalex_ai_works_201-279_search_terms_2020-2024.db` |

Splitting the first batch by year keeps each run short enough to supervise. The
year filter is part of the query, so the union over the year slices is the same
set of works a single 2020–2024 run would return.

Retrieval yielded **3,346,705 distinct publications** before the quality control
applied in [stage 03](../03_data_processing/).

## `download_openalex_works.py`

Cursor-paginated retrieval of one term batch straight into SQLite.

**Resumption.** Every page commits its rows and stores the next cursor beside
the database, so a run interrupted after hours continues where it stopped.
Rows are written with `INSERT OR IGNORE` on the work id, which makes a resumed
or repeated run idempotent.

**Rate limits.** A `429` response backs off 60 s without consuming a retry;
other transport errors retry up to five times with linear backoff. `--pause`
adds a wait between pages when a gentler pace is wanted.

**Selected fields.** Only the metadata the study uses is requested —
identifiers, title, year, language, type, open-access status, `authorships`
(the source for country attribution in stage 03), `cited_by_count`, `fwci`,
`referenced_works` and `abstract_inverted_index`. Requesting the full record
would multiply transfer volume for fields nothing reads. Nested objects are
stored as JSON text; `primary_location` and `open_access` are flattened into
`host_organization_name`, `source_issn_1` and `is_oa`.

**Run it**

```bash
# one batch
python code/02_data_collection/download_openalex_works.py --terms 201-279 --years 2020-2024

# the first batch, split by year
python code/02_data_collection/download_openalex_works.py --terms 1-50 --years 2020
python code/02_data_collection/download_openalex_works.py --terms 1-50 --years 2021
python code/02_data_collection/download_openalex_works.py --terms 1-50 --years 2022-2024

# a quick smoke test
python code/02_data_collection/download_openalex_works.py --terms 1-5 --max-works 400
```

| Variable | Required | Purpose |
|---|---|---|
| `OPENALEX_MAILTO` | no | puts requests in OpenAlex's faster "polite pool" |
| `OPENALEX_API_KEY` | no | raised rate limits, if you have a key |
| `QSS_SEARCH_TERMS` | no | override the search-term list location |
| `QSS_INTERIM_DIR` | no | override the output location |

**Input** — `code/01_keyword_construction/search_terms.txt`, the 279 curated
terms.
**Output** — `data/interim/openalex_raw/openalex_ai_works_<terms>_search_terms_<years>.db`
(table `works`) plus a small `.state.json` holding the resume cursor. Gitignored:
this is multi-gigabyte derived data.

## `merge_term_batches.py`

Merges the batch databases into one corpus. A publication matching terms in
several batches is retrieved several times, so the batches cannot simply be
concatenated: every batch is copied into a single `works` table keyed on the
OpenAlex work id and inserted with `INSERT OR IGNORE`, keeping each publication
exactly once. The run prints rows read against distinct publications kept, which
is where the 3,346,705 figure comes from.

```bash
python code/02_data_collection/merge_term_batches.py
```

**Input** — `data/interim/openalex_raw/*.db`.
**Output** — `data/interim/openalex_ai_works_merged_deduplicated.db` (table
`works`), the database [stage 03](../03_data_processing/) reads. Pass
`--overwrite` to replace an existing merge.
