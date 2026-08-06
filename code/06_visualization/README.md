# 06 — Visualization

Generates the manuscript's findings figures (§4, *Findings*). The manuscript
includes exactly two figures, and this stage covers one of them.

| | Figure | Code |
|---|---|---|
| Figure 1 | Summarized UMAP and taxonomy of AI subdisciplines | ⚠ not in this repository — see below |
| Figure 2 | Fractional citation shares of the blocs and RoW across hierarchical AI subdisciplines | `bloc_share_heatmap_cluster_across_microclusters.py` |

`paper/figures/` also holds figures carried over from the originating master's
thesis (country-contribution Aitoff projection, institution network, percentile
composition, search-term bar chart, H1/H2 CAGR bars, dataset-completeness
charts, pipeline diagrams). **None of them appears in the manuscript**, so
their generating code is deliberately not part of this deposit.

## `bloc_share_heatmap_cluster_across_microclusters.py`

Draws Figure 2: five columns, one per macro-domain, each a treemap-style stack
of meso-cluster boxes, each box holding a mini-heatmap row per micro-cluster
showing the four blocs' shares of that cluster's fractional citation sum.

The design choices are all in service of one comparison — *within a cluster,
how do the four blocs divide the citations* — while keeping the hierarchy
legible:

- **Rows are the unit of comparison, not columns.** Each micro-cluster's four
  cells sum to 100 %, so colour intensity is read across a row. The shared
  greyscale colour bar runs 0–50 %, above which cells saturate; a bloc holding
  more than half of a cluster is already the extreme case and finer resolution
  there would cost resolution in the range where most values sit.
- **Nesting is drawn, not implied.** Rounded H1 column frames carry the
  macro-domain colours used throughout the manuscript (purple computer science,
  blue health, teal social, yellow natural, orange engineering); off-white H2
  boxes sit inside them. The reader can see the tree without consulting the
  numeric codes.
- **A TOTAL row sits at the top of each column**, giving the macro-level split
  before the eye descends into the meso and micro rows.
- **Columns are height-equalised.** Domains have different numbers of clusters,
  so extra vertical space is distributed uniformly across available "slots"
  (after the TOTAL block, between rows inside a meso box, and between meso
  boxes) rather than padding the bottom — otherwise the five columns end at
  different heights and read as if they carried different amounts of
  information.
- **Geometry is computed in pixels, then converted**, so gaps and corner radii
  stay visually constant regardless of the figure's final scale.
- `svg.fonttype = 'none'` keeps text as text in the SVG, so the figure remains
  editable and its fonts substitutable downstream.

**Run it**

```bash
python code/06_visualization/bloc_share_heatmap_cluster_across_microclusters.py
```

**Input** — `data/interim/citshare_h3x4entities.csv` (override with
`QSS_CITSHARE_CSV`), the per-cluster per-bloc citation shares from
[stage 05](../05_impact_analysis/). Required columns:

```
macro_id, macro, meso, micro, code, total_citations,
cn_cits, us_cits, eu27_cits, row_cits,
cn_share, us_share, eu27_share, row_share
```

Since stage 05 is not yet in this repository, that CSV must be supplied
separately for this figure to be regenerable.

**Output** — `output/bloc_citation_shares_across_subdisciplines.svg`. The
manuscript embeds the PDF conversion of this file.

## The missing Figure 1

`cluster_umap_with_legend_2025_10_09_bigger_clusters.pdf` — the
density-summarized UMAP of all 106 micro-clusters, where each cluster is drawn
as an ellipse whose size reflects **semantic spread, not publication volume**,
accompanied by the full taxonomy legend.

Its generating code is not here. The UMAP projections written by
[stage 04](../04_subdiscipline_clustering/) are the per-level diagnostic
scatterplots the expert consolidated from; none of them produces the
density-summarized ellipse rendering or the legend layout used in the
manuscript. This is a genuine gap in the deposit, not a documentation
oversight.
