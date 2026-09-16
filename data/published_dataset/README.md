# Published corpus: `merged_works_labeled_published.db`

**Zenodo data record: [10.5281/zenodo.22791584](https://doi.org/10.5281/zenodo.22791584)**
(that DOI always resolves to the latest version). The database itself is not
carried in git — only this README and `SHA256SUMS.txt` are.

The labelled corpus behind the manuscript: 1,986,659 AI-related publications
from 2020–2024, each with its citation count, its country composition and its
position in the three-level cluster taxonomy. It is the subset of the full
working database that the analyses need, and nothing more.

| File | Size | Contents |
|---|---|---|
| `merged_works_labeled_published.db.gz` | 1.2 GB | gzip-compressed SQLite database |
| `SHA256SUMS.txt` | | checksums for the compressed and uncompressed file |

## Usage

```bash
gunzip merged_works_labeled_published.db.gz   # → 4.3 GB SQLite file
sha256sum -c SHA256SUMS.txt                   # optional integrity check
python -m clustervalidation inspect --level h3 --db merged_works_labeled_published.db
```

The code in this repository looks for `data/merged_works_labeled.db` by
default. Either pass `--db` as above or rename the file and place it there.
The table name and column names are the ones the code expects, so no other
change is needed.

## Contents

One table, `works_labeled`, one row per work, with a unique index on `id`.

| Column | Type | Description |
|---|---|---|
| `id` | TEXT | OpenAlex work ID as a full URL, e.g. `https://openalex.org/W1002758560`. Unique, never null. The key for fetching any further metadata from OpenAlex. |
| `doi` | TEXT | DOI as a full URL (`https://doi.org/…`). Null for 32,595 works (1.6 %) that have none. |
| `title` | TEXT | Title as delivered by OpenAlex. Empty for 1,897 works. |
| `publication_year` | INTEGER | 2020–2024. |
| `cited_by_count` | INTEGER | Citations received, as counted by OpenAlex at retrieval (July 2025). Never null. |
| `country_of_origin` | TEXT | JSON list of `[country, share]` pairs, e.g. `[["US", 0.7], ["IT", 0.3]]`. Countries are ISO 3166-1 alpha-2 codes taken from the authors' institutional affiliations (falling back to the author's own country field); each share is that country's proportion of all affiliation country codes on the work, so shares sum to 1. Derived by the authors — not an OpenAlex field. |
| `cleaned_abstract` | TEXT | Plain-text abstract reconstructed from OpenAlex's `abstract_inverted_index`. Present for every work. This is the text the validation protocols read. |
| `h1_cluster` | INTEGER | Macro level (domain), 0–4. |
| `h2_cluster` | INTEGER | Meso level (field), 0–6, numbered **within** its `h1_cluster`. |
| `h3_cluster` | INTEGER | Micro level (research front), 0–5, numbered **within** its `h2_cluster`. |

### Reading the cluster labels

The three labels form a path and are only meaningful together:
`h2_cluster = 4` means different fields in different domains. Concatenate them
to get a unique identifier — 5 domains, 31 fields and 106 research fronts in
total. [`code/05_impact_analysis/taxonomy.csv`](../../code/05_impact_analysis/taxonomy.csv)
maps each `h1‖h2‖h3` code (e.g. `011`) to its name.

| `h1_cluster` | Domain | Works |
|---|---|---:|
| 0 | Computer Science | 441,661 |
| 1 | Health Science | 454,264 |
| 2 | Social Science | 373,534 |
| 3 | Natural Science | 340,221 |
| 4 | Engineering | 376,979 |

The cluster labels and `country_of_origin` are the authors' contribution and
cannot be recovered from OpenAlex. The labels in particular cannot be
reproduced exactly by re-running the pipeline (see [`../README.md`](../README.md)),
so this file is the authoritative record of them.

## What is not included

OpenAlex holds considerably more metadata for each work than this file carries:
full authorships with author and institution identifiers, venues and
publishers, open-access status and APCs, funders and grants, reference lists,
field-weighted citation impact and citation percentiles, concepts and topics,
and more. The working database retrieved for this study held about twenty such
columns.

They are left out because they are not directly relevant to the research
questions, and because most of them keep changing as OpenAlex updates its
records, so a snapshot of them would soon be out of date anyway. Anyone who
needs them can fetch current values from the
[OpenAlex API](https://docs.openalex.org) using the `id` column, e.g.
`https://api.openalex.org/works/W1002758560`.

Note that `cited_by_count` is also a snapshot: OpenAlex's current values will
differ, usually upwards. The figures in the manuscript are based on the counts in this file.

## Provenance and licence

Source metadata was retrieved from [OpenAlex](https://openalex.org) in July
2025. OpenAlex releases its data under a CC0 public-domain dedication. The
derived columns (`country_of_origin`, `cleaned_abstract`, `h1_cluster`,
`h2_cluster`, `h3_cluster`) and this compilation are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by Julius Pfundstein,
Thomas Efer and Manuel Burghardt; see [`LICENSE-DATA`](../../LICENSE-DATA).

The file was built from the full working database by selecting the columns
above, with every row checked against the source for equality.
