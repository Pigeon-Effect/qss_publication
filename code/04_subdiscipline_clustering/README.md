# 04 — Subdiscipline clustering (semi-supervised hierarchical topic model)

Builds the taxonomy of **5 domains, 31 fields and 106 research fronts** that the
rest of the study is organised around (manuscript §3.1, *Hierarchical Topic
Modeling*). Corresponds to
[`topic_modeling_pipeline_diagram_clean.pdf`](../../paper/figures/topic_modeling_pipeline_diagram_clean.pdf)
and
[`exemplar_clustering_hierarchies_cancer_detection_clean_002.pdf`](../../paper/figures/exemplar_clustering_hierarchies_cancer_detection_clean_002.pdf).

## The problem this stage solves

There is no natural number of clusters in this corpus. `finding_optimal_k/`
exists to demonstrate that, not to find an answer: cohesion keeps improving as
*k* grows, with no elbow, and only a noisy plateau once a complexity penalty is
added. That is the expected result. Linguistic corpora reflect disciplinary
conventions, which grade continuously into one another; they do not partition
into discrete natural islands. Asking an algorithm for *the* right *k* is
asking the wrong question.

So the granularity at which the space is cut becomes an **analytical choice**,
and the pipeline is built to make that choice explicit and reviewable rather
than hiding it inside a hyperparameter. It does so in two clearly separated
moves, repeated at each level:

> **Over-segment algorithmically, then consolidate by hand.**

**Why over-segment.** K-Means is run at a deliberately high *k* = 50, far more
clusters than any level of the final taxonomy needs. A small but distinctive
field — say, neuromorphic hardware — would be swallowed by a larger neighbour
at a "sensible" *k*, and once absorbed, the human step cannot recover it:
merging is possible, unmixing is not. Over-segmentation is cheap insurance
against an irreversible loss.

**Why consolidate by hand.** Deciding that seventeen algorithmic clusters are
all *computer vision* is a judgement about how a research community understands
itself. The expert (an author) makes it with two complementary views:

| Clue | Artefact | What it shows |
|---|---|---|
| **Topological** | interactive UMAP (`.html`, hover-labelled, zoomable) | which clusters sit adjacent in embedding space |
| **Lexical** | TF-IDF terms + 20 most central and 100 random abstracts per cluster (`.json`) | what each cluster is actually about |

Neither alone suffices. Adjacency in a UMAP can be a projection artefact; TF-IDF
terms can be ambiguous across fields. Read together they are usually decisive.

**Why the result is still deterministic.** The expert's decision is recorded as
a plain `remap` / `H2_MAP` / `H3_MAP` dictionary — fine cluster id → super-cluster.
Labelling then assigns each document to its nearest *reference abstract* by
cosine similarity and inherits that reference's code. So the human judgement is
a lookup table, inspectable and re-runnable, not an unrepeatable act.

## The two-script rhythm

Every hierarchy level is the same pair of steps. This is the single most
important thing to understand about this folder:

```
   ALGORITHMIC PRE-CLUSTERING              →  EXPERT CONSOLIDATION
   (embed → k=50 KMeans → TF-IDF →            (read the artefacts → write the
    export UMAP + abstracts)                   remap dict → propagate labels)

h1 specter_embedding_with_central_          specter_embedding_with_curated_
   abstract_extraction.py                      cluster_mapping.py
h2 abstract_extraction_and_                  h2_labeling_<domain>.py
   interactive_umap.py                       (+ h2_umap_<domain>.py countercheck)
h3 h3_clustering_interactive_umap_and_       h3_label_selected_cluster.py
   abstract_extraction.py
```

The naming does not currently make this rhythm obvious — see the restructuring
proposal in the top-level [`README.md`](../../README.md).

## Shared method

Identical at all three levels, so it is stated once:

- **Vectorization.** Title + abstract, lowercased, non-alphabetic characters
  stripped, tokens of ≤2 characters dropped, **no stop-word removal**. Encoded
  with SPECTER (`allenai-specter`), a transformer trained on citation networks
  rather than on generic text, so proximity reflects how documents cite and are
  cited. Documents are tokenized into overlapping **100-token windows with
  stride 50** and the chunk embeddings averaged into one 768-dimensional
  vector — the windowing is what lets an abstract longer than the model's
  context still contribute in full.
- **Clustering.** K-Means, `k = 50`, `random_state = 42`, `n_init = 10`.
- **Outlier removal.** 1 % of points are dropped before the final projection
  (LocalOutlierFactor at h1, IsolationForest at h2) and the UMAP refitted.
  This is a *display* decision: a handful of extreme points otherwise compress
  the entire projection into an unreadable blob.
- **Lexical labelling.** Per-cluster TF-IDF centroids, scored as a weighted
  blend of raw magnitude and an inverse-cluster-frequency distinctiveness term,
  filtered for digits and short tokens. Top-10 terms per cluster.
- **UMAP.** `n_neighbors=15`, `min_dist=0.1`, `metric='cosine'`,
  `random_state=42`.

## Hierarchy and identifier scheme

The procedure recurses: each consolidated super-cluster becomes the corpus for
the next level. Codes are positional, so an identifier *is* a path.

| Level | Digits | Count | Example |
|---|---|---|---|
| h1 macro — domain | 1 | 5 | `1` Health Science |
| h2 meso — field | 2 | 31 | `10` Medical Imaging & Radiology |
| h3 micro — research front | 3 | 106 | `102` Cancer Detection & Screening |

No cluster is split into more than ten children, which is what keeps one digit
per level sufficient. Every micro-cluster belongs to exactly one meso- and one
macro-cluster: the taxonomy is a strict tree.

At h3, K-Means runs at `k = 30` (not 50) on a sample of up to 30,000 documents
per (h1, h2) slice — a slice is small enough that 30 over-segments it
adequately, and sampling keeps 31 separate runs tractable.

## Contents

### `finding_optimal_k/`

Evidence that no natural *k* exists. Not part of the production pipeline.

| Script | What it does |
|---|---|
| `silhouette_score.py` | TF-IDF + MiniBatchKMeans over *k* = 5…40, 10 seeded trials each, cosine silhouette on a 5,000-doc sample; writes `output/silhouette_scores.csv` and a mean ±1 SD plot |
| `bic_score.py` | Same sweep with additional criteria (Calinski-Harabasz, Davies-Bouldin, SVD-reduced) on a 10k sample |
| `silhouette_score_for_h1_clusters.py` | Elbow + silhouette over *k* = 2…20 on **SPECTER** embeddings of one h1 subset, i.e. the same question asked of the representation the pipeline actually uses |

### `SPECTER/h1_cluster_topic_modeling/`

| Script | Role |
|---|---|
| `specter_embedding_with_central_abstract_extraction.py` | **Algorithmic.** Embeds a 100k sample, K-Means *k*=50, TF-IDF names, exports `resources/interactive_umap.html` and `resources/cluster_abstracts_summary.json` |
| `specter_embedding_with_curated_cluster_mapping.py` | **Expert.** Holds the `remap` dict assigning all 50 clusters to the five macro-domains; assigns every document by nearest reference abstract; draws `resources/mapped_clusters_umap.svg` |

### `SPECTER/h2_cluster_topic_modeling/<domain>/`

One folder per macro-domain (`h2_cluster_biomedical`, `_computer_science`,
`_engineering`, `_natural_science`, `_social_sciences`), each with the same
three scripts:

| Script | Role |
|---|---|
| `abstract_extraction_and_interactive_umap.py` | **Algorithmic.** Same as h1, on that domain's subset |
| `h2_labeling_<domain>.py` | **Expert.** Holds `H2_MAP`; writes the `h2_cluster` column |
| `h2_umap_<domain>.py` | **Countercheck.** Re-applies the same mapping and draws a static UMAP, so the expert can see whether the merged groups occupy coherent regions |

Four of the five carry a completeness assertion — every one of the 50 fine
cluster ids must appear in the mapping exactly once, or the script raises.
`h2_labeling_natural_science.py` does not, and silently assigns `-1` to
anything unmapped.

### `SPECTER/h3_cluster_topic_modeling/`

| Script | Role |
|---|---|
| `h3_clustering_interactive_umap_and_abstract_extraction.py` | **Algorithmic.** Takes `--h1` / `--h2`, samples ≤30k docs from that slice, K-Means *k*=30, writes `resources/h1_XX_h2_YY/` |
| `h3_label_selected_cluster.py` | **Expert.** Holds `H3_MAP` for one slice; writes `h3_cluster`. Refuses to overwrite rows that already carry a value, so a mis-specified `--h1`/`--h2` cannot corrupt a previously labelled slice |
| `check_db_state.py` | Read-only pre-flight: shows how many rows in a slice are eligible and whether any already have `h3_cluster` |

`resources/` holds the exported artefacts for all **31** (h1, h2) slices — this
is the archived record of what the expert was looking at when consolidating.

Note that `H3_MAP` in `h3_label_selected_cluster.py` holds **one slice at a
time**: it was edited between runs. The version archived here is the last one
run (h1=4, h2=5 — Robotics & Mechatronics). The other 30 mappings were
overwritten and are not recoverable from this repository.

### `SPECTER/tfidf_terms_for_micro_meso_macro.py`

Post-hoc term extraction for the finished taxonomy, and the reason the
descriptors in the manuscript read as *field*-specific rather than generic.

Contrast is computed **parent-scoped**: an h3 cluster is contrasted only
against its siblings under the same h2, an h2 only against siblings under the
same h1. Contrasting against the whole corpus would return generic AI
vocabulary at every level; contrasting against siblings returns what actually
separates *this* cluster from the ones it could be confused with — which is why
descriptors grow more specific as the hierarchy descends. Text is lemmatized
(NLTK WordNet, verb then noun) so inflectional variants do not compete, and
terms are ranked by Δ mean TF-IDF with a smoothed log₂ ratio reported alongside.

Writes `output/tfidf_contrast_lemmas_parent_scoped_h1_<a>_h2_<b>_h3_<c>.csv`.
The target micro-cluster is the `MICRO_ID` constant (currently 102, the cancer
detection exemplar used in the manuscript figure).

## Running it

```bash
# 1. algorithmic pre-clustering of the whole corpus
python code/04_subdiscipline_clustering/SPECTER/h1_cluster_topic_modeling/specter_embedding_with_central_abstract_extraction.py
# 2. inspect resources/interactive_umap.html + cluster_abstracts_summary.json,
#    then edit the `remap` dict in the next script
python code/04_subdiscipline_clustering/SPECTER/h1_cluster_topic_modeling/specter_embedding_with_curated_cluster_mapping.py
# 3. split the labelled corpus into one database per macro-domain, then repeat
#    the same two steps inside each - and again for each (h1, h2) slice at h3
python code/04_subdiscipline_clustering/SPECTER/h3_cluster_topic_modeling/h3_clustering_interactive_umap_and_abstract_extraction.py --h1 0 --h2 0
```

A CUDA device is used if available; SPECTER on CPU is slow but works.

| Variable | Purpose |
|---|---|
| `QSS_INTERIM_DIR` | location of the intermediate databases (default `data/interim/`) |
| `QSS_DB_PATH` | the final labelled corpus, read by the TF-IDF script |
| `QSS_H3_DB` | which subset database the h3 scripts read |

**Inputs** — the filtered corpus from [stage 03](../03_data_processing/), plus
the per-domain subset databases derived from it
(`data/interim/h1_cluster_subsets/<domain>_dataset.db`, and `_sample.db`
variants used for the visualisations).

**Output** — the corpus labelled with `h1_cluster` / `h2_cluster` /
`h3_cluster`, consumed by [stage 05](../05_impact_analysis/),
[stage 06](../06_visualization/) and [stage 07](../07_llm_validation/).

## Known gaps between this code and the published taxonomy

These scripts are the working record of an iterative process, and the record is
incomplete in three places. Each is flagged in the affected file's header. None
has been reconstructed by guesswork, because inventing a cluster assignment
would silently fabricate part of the taxonomy.

1. **`h2_labeling_computer_science.py` is superseded.** It records eight meso
   codes; the manuscript gives Computer Science five. Neuromorphic Hardware
   Accelerators moved to Natural Science (meso 35), Ethical & Creative AI to
   Social Science (meso 26), and Recommendation Systems does not survive. That
   pass moved fine clusters *across macro-domains*, which a script writing only
   `h2_cluster` within one subset cannot express.
2. **`h2_labeling_natural_science.py` and `h2_labeling_social_science.py` are
   incomplete.** They lack, respectively, meso 35/36 and meso 26 — the groups
   that arrived in that same later pass.
3. **`h2_cluster_social_sciences/h2_umap_social_science.py` is not a
   social-science script.** Its paths, `remap` and palette are all Engineering;
   it is an unadapted copy of `h2_umap_engineering.py`. The social-science
   static-UMAP countercheck was therefore never produced.

Reconciling 1–2 needs the record of which fine cluster ids were reassigned in
the final consolidation pass. `h2_labeling_engineering.py` and
`h2_labeling_biomedical.py` match the published taxonomy exactly.
