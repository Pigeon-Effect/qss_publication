# 05 — Impact analysis

Computes each bloc's research impact within every cluster of the taxonomy
(manuscript §3.2, *Assessment of Research Impact*), producing the input to
Figure 2 in [stage 06](../06_visualization/).

## The metric

The **fractional citation sum**. A work's incoming citations are split among
the countries that produced it, in proportion to the `country_of_origin` shares
built in [stage 03](../03_data_processing/) from author institutions. A paper
with 12 citations and shares `[("US", 0.75), ("DE", 0.25)]` contributes 9.0
fractional citations to the US and 3.0 to Germany. Summed over a cluster and
grouped into blocs, that gives the shares the manuscript reports.

Two premises sit underneath it, and both are choices worth stating:

**Authors contributed equally.** Citations are split by institutional slot, not
weighted by author position, seniority, or declared contribution. Author-order
conventions differ sharply across the disciplines this corpus spans — first
author, last author and alphabetical are all in use — so any weighting scheme
would encode one field's convention as though it were universal. Equal
splitting is the assumption that treats every field alike.

**Institutions indicate research systems better than personal origin would.**
The question is which research system produced the work — its funding,
infrastructure and training — not where its authors were born or hold
citizenship. Affiliation is the observable that tracks that. Nationality is
neither recorded by OpenAlex nor the quantity of interest.

## Why not something more sophisticated

Two families were considered and rejected, each for a reason specific to this
study's structure:

- **Network-centrality measures** (closeness, eigenvector, PageRank over
  citation or co-authorship graphs) are heavily distorted here by collaboration
  culture and institutional architecture. A country whose researchers publish
  in large international consortia scores differently from one whose
  researchers publish in small national teams, independently of impact. These
  measures end up tracking network structure rather than research impact.
- **Field- and year-normalized indicators** such as FWCI are unstable across
  the many small and young clusters this topic model produces. When a cluster
  holds a few hundred publications, a handful of papers can swing the
  expected-citation baseline the indicator divides by.

The fractional citation sum is the most transparent and least abstracted of the
options, makes no assumption that needs defending at this granularity, and is
**complete in the dataset** — every record carries a citation count, whereas
outgoing references are missing for 7.8 % of records.

Its one structural property has to be stated plainly, and the manuscript does
so under *Limitations*: shares within a cluster sum to 100 %, so one bloc's
gain is mechanically another's loss. That is a property of the measure, not a
finding about AI competition.

## `fractional_citation_share.py`

Streams the labelled corpus, sums fractional citations per (micro-cluster,
country), aggregates countries into blocs, and writes the CSVs.

**Bloc definitions**, all explicit constants at the top of the file:

| Bloc | Members |
|---|---|
| China | `CN`, `HK`, `MO`, `TW` — the Greater China reading |
| USA | `US` |
| EU-27 | the 27 member states; the UK is **not** one |
| RoW | the residual: everything in no other bloc |

OpenAlex reports Hong Kong, Macao and Taiwan as separate institution country
codes, so whether they count as China is a choice, and a consequential one:

| Definition | Corpus-wide citation share |
|---|---:|
| `CN` only | 21.82 % |
| `CN + HK + MO` | 23.05 % |
| **`CN + HK + MO + TW`** (used) | **24.00 %** |
| manuscript reports | 24.10 % |

It is a named constant rather than something buried in a query, so it can be
re-cut and re-checked. The un-aggregated per-country output below makes
re-cutting cheap.

RoW is computed as `total − (China + US + EU-27)`, so the four values sum to the
cluster total exactly and no country can be double-counted or silently dropped.

**Correctness checks built into the run.** The script reports rather than hides
what it skipped, and fails loudly on a real inconsistency:

- Every cluster code found in the corpus must exist in `taxonomy.csv`, or the
  run aborts — a mismatch there would propagate into every downstream number.
  Codes in the taxonomy with no cited works are reported as a warning.
- The per-work country shares are checked against summing to 1.0, and the
  worst deviation is printed. Anything above floating-point noise points at
  stage 03.
- Rows skipped for zero citations, missing country, or unparseable
  `country_of_origin` are counted and printed separately.

Works with zero citations are skipped: they contribute exactly zero to every
sum, so skipping is equivalent to processing and avoids parsing JSON for a
large share of the corpus.

**Run it**

```bash
python code/05_impact_analysis/fractional_citation_share.py
```

| Flag | Default |
|---|---|
| `--db` | `data/merged_works_labeled.db` (or `QSS_DB_PATH`) |
| `--taxonomy` | `taxonomy.csv` beside the script |
| `--out` | `data/interim/citshare_h3x4entities.csv` |
| `--out-countries` | `data/interim/fractional_citations_by_country_h3.csv` |
| `--progress-every` | `250000` rows; `0` to silence |

Expect a single full scan of the corpus. On a ~20 GB database that is minutes,
not seconds.

## Reproduction against the manuscript

Run on `data/merged_works_labeled.db` (1,986,659 works; 1,426,844 carry at
least one citation; 19,817,595 fractional citations in total):

**Macro-domain shares of the corpus total — exact match to Figure 2:**

| Domain | Computed | Figure 2 |
|---|---:|---:|
| Computer Science | 22.48 % | 22.48 % |
| Health Science | 26.91 % | 26.91 % |
| Social Science | 15.07 % | 15.07 % |
| Natural Science | 17.99 % | 17.99 % |
| Engineering | 17.56 % | 17.56 % |

**Corpus-wide bloc shares — within 0.22 pp of §4.2:**

| Bloc | Computed | Manuscript |
|---|---:|---:|
| China | 24.00 % | 24.1 % |
| USA | 18.52 % | 18.60 % |
| EU-27 | 20.75 % | 20.79 % |
| RoW | 36.73 % | 36.51 % |

Every macro-level bloc figure quoted in §4.2 also reproduces within 0.22 pp
(e.g. China in Computer Science 32.89 % vs 32.8 %, the US in Health Science
23.49 % vs 23.36 %, the EU-27 in Social Science 25.60 % vs 25.38 %). The
residual is a consistent ~0.2 pp of citation mass sitting in RoW here that the
manuscript assigns to the three blocs — small enough to be a marginally
different corpus snapshot or a one-country difference in a bloc list, and left
unreconciled rather than tuned away.

## Inputs

The cluster-labelled corpus from
[stage 04](../04_subdiscipline_clustering/) — specifically `h1_cluster`,
`h2_cluster`, `h3_cluster`, `country_of_origin` and `cited_by_count`. The
cluster identifier is the concatenation of the three labels, matching
`CLUSTER_ID_SQL` in `src/clustervalidation/config.py`, so stages 05 and 07
address clusters identically.

### `taxonomy.csv`

The display names of the 5 domains, 31 fields and 106 research fronts, keyed by
3-digit code. Committed alongside the script because the database stores only
integer codes — the names live in the manuscript, not in the data.

```csv
code,macro_id,macro,meso,micro
000,0,Computer Science,Computer Vision & Image Processing,Image Generation & Segmentation
```

**Transcribed from manuscript Figure 1**, and worth checking against it: the
structure is verified in code (106 unique codes, 31 unique fields, 5 domains,
no cluster with more than ten children), but the spelling of a label is not
something code can check.

## Outputs

Both land in `data/interim/` and are gitignored.

**`citshare_h3x4entities.csv`** — one row per micro-cluster, sorted by macro
then code, which is the order stage 06 needs (it groups meso-clusters into
contiguous blocks and relies on it):

```
macro_id, macro, meso, micro, code, total_citations,
cn_cits, us_cits, eu27_cits, row_cits,
cn_share, us_share, eu27_share, row_share
```

**`fractional_citations_by_country_h3.csv`** — the same sums before bloc
aggregation, one row per (micro-cluster, country), sorted by citations
descending within a cluster. Keeping it means the bloc definitions can be
re-cut — adding Hong Kong to China, say, or splitting the EU-27 — without
another pass over two million rows.

Macro- and meso-level figures are not written to file. They are exact sums of
the micro rows, stage 06 aggregates them itself, and a second file holding
derivable numbers is a second file that can fall out of sync. The run prints
them instead, since they are the figures quoted in manuscript §4.2.
