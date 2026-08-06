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
work — hence the study builds its own topic model in stage 04 rather than
adopting it.

## `download_full_dataset.py`

Cursor-paginated retrieval of every work matching the search vocabulary.

**Query.** A Boolean-OR expression over the 279 search terms from
[stage 01](../01_keyword_construction/), matched against OpenAlex's full-text
index (titles and abstracts), filtered to `publication_year:2020-2024`, sorted
newest first.

**Batching.** OpenAlex caps the length of a single `search` expression, so the
279 terms cannot be sent as one query. They are split into groups of
`TERMS_PER_QUERY` (default 25), each group retrieved independently, and the
results merged and **deduplicated on the OpenAlex work id** — a paper matching
terms in several groups is kept once.

**Resumption.** Each group writes its results and its `next_cursor` to disk
every `save_interval` records (default 100,000). Re-running picks up from the
saved cursor rather than restarting, which matters for a retrieval measured in
days. A group whose saved cursor is `null` is treated as finished.

**Rate limiting.** Requests are paced at 11.38 s per 200-record page,
targeting roughly one million works per 24 hours — chosen to stay well within
OpenAlex's limits rather than to go as fast as possible. A `429` response backs
off 60 s without consuming a retry; other transport errors retry up to five
times with linear backoff.

**Selected fields.** Only the metadata the study uses is requested — identifiers,
title, year, language, type, open-access status, `authorships` (the source for
country attribution in stage 03), `cited_by_count`, `fwci`,
`referenced_works`, and `abstract_inverted_index`. Requesting the full record
would multiply transfer volume for fields nothing reads.

**Run it**

```bash
# credentials are optional but recommended - see .env.example
python code/02_data_collection/download_full_dataset.py
```

| Variable | Required | Purpose |
|---|---|---|
| `OPENALEX_MAILTO` | no | puts requests in OpenAlex's faster "polite pool" |
| `OPENALEX_API_KEY` | no | raised rate limits, if you have a key |
| `QSS_SEARCH_TERMS` | no | override the search-term list location |
| `QSS_INTERIM_DIR` | no | override the output location |

**Input** — `code/01_keyword_construction/search_terms.txt`: the 279 curated
terms, one per line. **This file is not currently in the repository** (see
[stage 01](../01_keyword_construction/#what-is-not-here)); the script exits
with an explicit message if it is missing rather than silently querying a
partial vocabulary.

**Output**, all gitignored:

```
data/interim/
├── openalex_raw/
│   ├── works_group_000.json     raw records for term group 0
│   ├── state_group_000.json     {"next_cursor": ...} for resumption
│   └── ...                      one pair per term group
└── openalex_ai_works_2020-2024_raw.json    merged, deduplicated
```

Retrieval yielded **3,346,705 publications** before the quality control applied
in [stage 03](../03_data_processing/).

## Note on the merged output format

The original run stored the merged corpus in SQLite as a `works` table; stage
03 reads that table. This script writes JSON, which is what its
`get_all_works` has always produced. Loading the merged JSON into
`data/interim/openalex_ai_works_merged_deduplicated.db` is the one step between
stages 02 and 03 that is not scripted here.
