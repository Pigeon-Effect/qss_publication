# Assignment accuracy — how faithfully the full corpus inherits the sample clustering

K-Means and the expert consolidation ran on **samples** (100,000 publications at
h1 and h2, at most 30,000 per field at h3). The other publications never saw
K-Means. Each was labeled by a cheaper rule: find the most similar *reference
document* (20 central + 100 random members of every K-Means cluster) by cosine
similarity and inherit its cluster, then let the expert mapping carry it to its
consolidated cluster.

This folder measures how often that rule gives the same answer K-Means would
have given — the figure reported in manuscript §4.1 (*Scaling to the full
corpus*) and in the Limitations (§7).

## Result

Procedure exactly as run, weighted by cluster size:

| Level | Same K-Means cluster | Same consolidated cluster | Method |
|---|---:|---:|---|
| h1 — 5 domains | 69.0 % | **84.3 %** | re-run, 94,038 held-out documents |
| h2 — 31 fields | 69.6 % | 88.0 % | leave-one-out, 4 archived domains |
| h3 — 106 research fronts | 66.6 % | **88.1 %** | leave-one-out, all 31 archived slices |

**Reading it.** About two in three documents land in exactly the K-Means cluster
they belong to. That is the strict reading, and it is expected: K-Means was set
to over-segment (k = 50 / 30), so many neighbouring clusters share a topic and
documents on their border can go either way. After consolidation, most of
those confusions disappear, because they were between clusters the expert
merged anyway: about 84–88 % of documents keep their consolidated label. The
consolidated figure is what decides which field a publication is counted in.

Errors compound down the hierarchy. A publication is labeled correctly at all
three levels with probability of roughly 0.84 × 0.88 × 0.88 ≈ 65 %, if the
levels erred independently (they do not entirely: errors concentrate on
documents that sit between topics).

## Two methods, and why both

**h2 and h3 — `loo_archived_references.py` (leave-one-out on the original
artefacts).** The `cluster_abstracts_summary*.json` files in `../SPECTER/*/resources/`
hold the original reference documents together with their original K-Means
cluster. A cluster's 100 *random* references are a random sample of its
members, so they stand in for held-out documents: each is embedded the way a
document is, assigned to its nearest *other* reference, and scored against its
own cluster. Nothing is re-clustered; this is the original run.

**h1 — `rerun_h1_heldout.py` (fresh re-run).** The h1 reference JSON was not
archived, so the h1 step is repeated on a new random sample of 100,000
publications (seed 42): embed, K-Means k = 50, pick references, assign the
94,038 sample documents that are not references, and compare. It also scores
10,000 publications outside the sample (K-Means label = nearest centroid):
68.1 % / 83.6 %, practically the same as held-out. Deviations from the original
run: 3 instead of 10 K-Means initializations, and no 1 % LOF outlier removal
before choosing references.

**The re-run validates the leave-one-out estimator.** On the re-run's own random
references, leave-one-out gives 70.5 % / 84.9 %, against 69.0 % / 84.3 % for the
full held-out set. So leave-one-out is accurate to about one point, slightly
optimistic; the h2/h3 figures should be read with that margin.

## Details that matter

**The consolidation is read from the deposited labels.** The deposited labels
are the published taxonomy, so each K-Means cluster is assigned the most common
**deposited** code among its members (h1) or references (h2, h3; matched to the
corpus by title). At h1 this is an approximation: the fresh K-Means clusters do
not coincide with the original ones, so the consolidation derived this way
agrees with the deposited h1 label for only 79 % of the sample. The h1
consolidated figure is therefore best read as a conservative estimate.

**References and documents are embedded from different text.** Every labeling
script embeds references from the raw `title. abstract` but documents from the
preprocessed text (lowercased, letters only, words of ≤ 2 characters dropped).
The figures above keep that asymmetry. Embedding both sides the same way
changes them by 1–3 points, in no consistent direction (h1: +2.2 / +1.2; h2:
−3.2 / −1.5; h3: −1.7 / −0.7), so the asymmetry does not matter for the
result. Both variants are in the result files.

**Embeddings run through ONNX.** `specter_onnx.py` reproduces the labeling
scripts' `embed_with_sliding()` (100-token windows, stride 50, [CLS] pooling,
mean over windows) with the ONNX export of `allenai-specter` that
sentence-transformers publishes, so the check needs neither torch nor CUDA.
DirectML and CPU agree to 1e-5.

## Contents

| File | Role |
|---|---|
| `specter_onnx.py` | SPECTER sliding-window embeddings via ONNX Runtime |
| `loo_archived_references.py` | h2 / h3: leave-one-out on the archived reference sets |
| `rerun_h1_heldout.py` | h1: re-run of the h1 step on a fresh 100k sample |

Results are written to [`results/assignment_accuracy/`](../../../results/assignment_accuracy/):
`loo_per_cluster.csv` (one row per K-Means cluster, level, slice and variant),
`loo_summary.json`, and `h1_rerun.json`.

## Running it

```bash
pip install onnxruntime tokenizers huggingface_hub numpy   # or onnxruntime-directml on Windows
cd code/04_subdiscipline_clustering/assignment_accuracy
python loo_archived_references.py   # ~1 h on a GPU, embeddings cached in output/
python rerun_h1_heldout.py          # ~45 min on a GPU
```

Both read the labeled corpus (`QSS_DB_PATH`, default
`data/merged_works_labeled.db`; see [`data/README.md`](../../../data/README.md))
read-only. The published corpus is sufficient: it carries the titles,
abstracts and labels the scripts need, in the same row order as the working
database the deposited results were computed on, so the h1 re-run draws the
same sample. Embeddings are cached in `output/` (gitignored), so a second run
only redoes the scoring.
