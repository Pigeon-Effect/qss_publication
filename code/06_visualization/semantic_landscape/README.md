# The semantic landscape (Figure 1)

Draws the map of the 106 micro-clusters that opens the *Findings* section of the
manuscript: every research front as one outline, coloured by the macro-domain it
belongs to and labelled with its three-digit code.

The map answers a question the citation tables cannot: **how the subdisciplines
of AI sit relative to one another.** The taxonomy built in
[stage 04](../../04_subdiscipline_clustering/) is a tree, and a tree says which
cluster is inside which parent, but nothing about which clusters are neighbours.
The embedding space does carry that information, and this figure is how it is
read.

## How to read it

- **Position and adjacency carry meaning.** Two clusters sit next to each other
  when their publications use similar language. Adjacency across a domain
  boundary is the interesting case: it marks a shared method rather than a
  shared subject, which is why computer-vision fronts border medical imaging and
  earth observation.
- **Size is semantic spread, not publication volume.** An outline encloses the
  region a cluster occupies in the embedding space. A large, tightly written
  field is drawn small; a small field whose papers range widely is drawn large.
  Machine-learning foundations covers much more of the map than the far more
  numerous computer-vision fronts, and that is the point being made.
- **Rotation is arbitrary.** UMAP fixes relative arrangement, not orientation.
  Compass directions used in the text are a convenience for describing the
  figure, not a property of the data.
- **Colour and digits repeat the hierarchy.** The first digit is the
  macro-domain and matches the colour; the second identifies the field within it;
  the third the research front.

## The three steps

```
corpus ──project_sample.py──▶ scatter ──summarize_clusters.py──▶ outlines ──render_landscape.py──▶ figure
          SPECTER + UMAP                  80 % density contour               one polygon per cluster
```

### 1. `project_sample.py` — from documents to a scatter

Draws a seeded random sample (100,000 publications by default) and encodes each
one exactly as stage 04 did: title and abstract, lowercased, non-alphabetic
characters stripped, tokens of one or two characters dropped, stop words kept;
SPECTER over 100-token windows with stride 50, window vectors averaged; then
UMAP (`n_neighbors=15`, `min_dist=0.1`, cosine). Using the same encoder and the
same settings is what makes the map a picture of the space the clusters were
actually built in.

A sample is used because embedding two million abstracts for one figure buys
nothing: at 100,000 points every micro-cluster still holds several hundred
publications, which is far more than a density estimate needs.

**Output** — `data/interim/semantic_landscape/points.csv.gz` (id, the three
cluster labels, x, y) and `run_config.json` with the parameters of the run.

### 2. `summarize_clusters.py` — from a scatter to outlines

A scatter of 100,000 points is unreadable, and overlapping clusters hide exactly
the structure the figure is for. Each cluster is therefore replaced by a single
outline:

1. a Gaussian kernel density estimate over that cluster's points, on a regular
   grid spanning the cluster's extent, with Scott's-rule bandwidth scaled by the
   cluster's covariance, so an elongated cluster gets an elongated kernel;
2. the density level enclosing **80 %** of the cluster's points, read off the
   density values at the points themselves;
3. the contour at that level, keeping the largest closed ring.

Following a density level rather than the outermost points is what makes size
mean spread: outliers move the extremes but not the level, so a handful of
far-flung papers cannot inflate a cluster. The trade-off is stated in the
manuscript — diffuse clusters are drawn generously, dense ones conservatively,
so the map reads semantic reach and not output.

**Output** — `data/interim/semantic_landscape/semantic_landscape_layout.geojson`,
one polygon per micro-cluster with its code, its taxonomy names, the number of
points behind it and the coverage used.

### 3. `render_landscape.py` — from outlines to the figure

Fills each outline in its macro-domain colour at 80 % opacity, outlines it in
black, and places the three-digit code in a white box at the polygon's centre.
Outlines are drawn largest first so a small cluster lying on top of a big one
stays visible. `svg.fonttype = 'none'` keeps the text editable in the SVG.

**Output** — `output/semantic_landscape.svg` (and a PNG with `--png`).

## The published layout

[`resources/semantic_landscape_layout.geojson`](resources/semantic_landscape_layout.geojson)
holds the 106 outlines exactly as published in Figure 1, each carrying its
cluster code, its taxonomy names and the label position used in the figure.
Coordinates are figure units with y pointing up; only relative position, size
and shape carry meaning.

It is deposited so the figure can be redrawn, recoloured or reused — for
instance to shade the same map by some per-cluster quantity — without a GPU and
without recomputing the embedding. `render_landscape.py` reads it by default.

A UMAP layout is not deterministic across runs: a fresh projection reproduces
the topology and the neighbourhood structure, with a different rotation and
slightly different shapes. Use the deposited layout when the published figure
itself is wanted, and steps 1–2 when the aim is to rebuild the map from the
corpus.

## Running it

```bash
# redraw the published figure — no model, no GPU, seconds
python code/06_visualization/semantic_landscape/render_landscape.py --png

# rebuild the map from the corpus
python code/06_visualization/semantic_landscape/project_sample.py
python code/06_visualization/semantic_landscape/summarize_clusters.py
python code/06_visualization/semantic_landscape/render_landscape.py \
    --layout data/interim/semantic_landscape/semantic_landscape_layout.geojson

# the scatter behind the outlines, as a diagnostic
python code/06_visualization/semantic_landscape/render_landscape.py \
    --layout data/interim/semantic_landscape/semantic_landscape_layout.geojson \
    --points data/interim/semantic_landscape/points.csv.gz --png
```

| Variable | Purpose |
|---|---|
| `QSS_DB_PATH` | the labelled corpus (default `data/merged_works_labeled.db`) |
| `QSS_INTERIM_DIR` | where the projection and layout are written (default `data/interim/`) |

Useful flags: `--sample-size` and `--seed` for the projection; `--coverage`,
`--resolution` and `--max-points` for the summarization; `--width`, `--png` and
`--no-labels` for the rendering. Every script takes `--help`.

Embedding 100,000 abstracts takes roughly ten minutes on a CUDA device and hours
on a CPU; `--save-embeddings` keeps the vectors so the projection can be redone
without re-encoding. The summarization and rendering steps run in well under a
minute on any machine.
