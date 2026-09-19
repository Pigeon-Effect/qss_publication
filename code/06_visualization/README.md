# 06 — Visualization

Draws the manuscript's two figures (§4, *Findings*):

| Figure | What it shows | Where |
|---|---|---|
| Figure 1 | the semantic landscape of the 106 micro-clusters | [`semantic_landscape/`](semantic_landscape/) |
| Figure 2 | each bloc's citation share within every cluster | `bloc_share_heatmap_cluster_across_microclusters.py` |

Both read the deposited outputs of earlier stages, so neither needs an API key
and neither re-runs the pipeline.

## Figure 1 — the semantic landscape

See [`semantic_landscape/README.md`](semantic_landscape/README.md). The folder
holds the three steps that turn the corpus into the map — projection, density
summarization, rendering — and the published layout, so the figure can be
redrawn without recomputing the embedding.

## Figure 2 — `bloc_share_heatmap_cluster_across_microclusters.py`

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
`QSS_CITSHARE_CSV`), the per-cluster per-bloc citation shares written by
[stage 05](../05_impact_analysis/) and deposited with this repository. Required
columns:

```
macro_id, macro, meso, micro, code, total_citations,
cn_cits, us_cits, eu27_cits, row_cits,
cn_share, us_share, eu27_share, row_share
```

**Output** — `output/bloc_citation_shares_across_subdisciplines.svg`. The
manuscript embeds the PDF conversion of this file.

## A note on `paper/figures/`

That folder also holds figures carried over from the originating master's
thesis — country-contribution Aitoff projection, institution network, percentile
composition, H1/H2 CAGR bars, dataset-completeness charts, pipeline diagrams.
They are part of the manuscript's lineage rather than of its argument, and the
code here covers the two figures the manuscript itself presents.
