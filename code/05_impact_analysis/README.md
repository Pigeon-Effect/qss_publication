# 05 — Impact analysis

**Status: not yet in this repository.** No code here; this README states what
belongs here and why, so the gap is explicit rather than silent.

Computes each bloc's research-impact metric within every cluster of the
taxonomy (manuscript §3.2, *Assessment of Research Impact*), which is the input
to the citation-share figure produced in [stage 06](../06_visualization/).

## The metric and why it was chosen

The primary metric is the **fractional citation sum**: an incoming citation is
split equally among the citing work's contributing authors, and thus among
their countries, using the `country_of_origin` shares built in
[stage 03](../03_data_processing/). A paper with three US and one German
institutional slot contributes 0.75 of each of its citations to the US and 0.25
to Germany.

Two more sophisticated families were considered and rejected, both for reasons
specific to this study's structure:

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

Its main property has to be stated plainly, and the manuscript does so under
*Limitations*: shares within a cluster sum to 100 %, so one bloc's gain is
mechanically another's loss. That is a property of the measure, not a finding
about AI competition.

## What belongs here

Code that, given the cluster-labelled corpus, produces per-cluster,
per-bloc citation shares for China, the US, the EU-27 and the residual RoW
category, at all three hierarchy levels.

**Input** — the labelled corpus from [stage 04](../04_subdiscipline_clustering/),
specifically `country_of_origin`, `cited_by_count`, `referenced_works` and the
three `h*_cluster` columns.

**Output** — `data/interim/citshare_h3x4entities.csv`, which
[stage 06](../06_visualization/) reads. The heatmap script requires these
columns:

```
macro_id, macro, meso, micro, code, total_citations,
cn_cits, us_cits, eu27_cits, row_cits,
cn_share, us_share, eu27_share, row_share
```

Until this stage lands, that CSV must be supplied by other means for the stage
06 figure to be regenerable. Point `QSS_CITSHARE_CSV` elsewhere if it lives
somewhere else.
