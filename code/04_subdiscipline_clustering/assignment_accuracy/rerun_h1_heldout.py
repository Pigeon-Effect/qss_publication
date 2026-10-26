"""Assignment accuracy at h1, by re-running the h1 procedure.

The h1 reference set was not archived, so the procedure is repeated on a fresh
random sample of 100,000 publications: embed (preprocessed text), K-Means with
k = 50, 20 central + 100 random references per cluster (re-embedded from raw
text, as in the labeling script), then every sample document that is not a
reference is assigned to its nearest reference by cosine similarity and
compared with its K-Means label.

Also reported:
  out_of_sample  10,000 publications outside the sample, whose K-Means label
                 is their nearest centroid
  loo            the leave-one-out estimator of loo_archived_references.py,
                 computed on this run's random references; it should match
                 the held-out figure, which validates the h2/h3 numbers

Consolidated accuracy uses a reconstructed mapping: each K-Means cluster maps
to the most common deposited h1 code among its members.

Deviations from the original run: K-Means uses 3 instead of 10 random
initializations (numpy implementation, k-means++), and the 1 % LOF outlier
removal before choosing references is skipped.

Writes results/assignment_accuracy/h1_rerun.json. Embeddings are cached in
output/; a full run takes about 45 minutes on a GPU.
"""

import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

import specter_onnx as S

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = Path(os.environ.get(
    "QSS_DB_PATH", PROJECT_ROOT / "data" / "original_dataset" / "merged_works_labeled.db"))
OUT = Path(__file__).resolve().parent / "output" / "h1_rerun"
RESULTS = PROJECT_ROOT / "results" / "assignment_accuracy"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

SAMPLE, OUT_OF_SAMPLE, K, SEED = 100_000, 10_000, 50, 42
rng = np.random.default_rng(SEED)
con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def fetch(rowids):
    found = {}
    for b in range(0, len(rowids), 900):
        q = ",".join(map(str, rowids[b:b + 900]))
        sql = f"SELECT rowid, title, cleaned_abstract, h1_cluster FROM works_labeled WHERE rowid IN ({q})"
        for row in con.execute(sql):
            found[row[0]] = row[1:]
    return [found[i] for i in rowids]


def cached(name, compute):
    path = OUT / f"{name}.npy"
    if not path.exists():
        np.save(path, compute())
    return np.load(path)


def processed(rows):
    return [S.preprocess((t or "") + ". " + (a or "")) for t, a, _ in rows]


def kmeans(X, k, n_init=3, max_iter=300, seed=SEED):
    r = np.random.default_rng(seed)
    sq = (X ** 2).sum(1)
    best = None
    for _ in range(n_init):
        centers = [X[r.integers(len(X))]]
        d = ((X - centers[0]) ** 2).sum(1)
        for _ in range(1, k):  # k-means++
            centers.append(X[r.choice(len(X), p=d / d.sum())])
            d = np.minimum(d, ((X - centers[-1]) ** 2).sum(1))
        C = np.array(centers)
        for _ in range(max_iter):
            dist = sq[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None]
            labels = dist.argmin(1)
            new = np.array([X[labels == j].mean(0) if (labels == j).any() else C[j]
                            for j in range(k)])
            shift = ((new - C) ** 2).sum()
            C = new
            if shift < 1e-6:
                break
        inertia = dist[np.arange(len(X)), labels].sum()
        if best is None or inertia < best[0]:
            best = (inertia, C, labels)
    return best[1], best[2]


def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def nearest(queries, refs_unit, ref_labels, block=10_000):
    return np.concatenate([ref_labels[(unit(queries[b:b + block]) @ refs_unit.T).argmax(1)]
                           for b in range(0, len(queries), block)])


n_rows = con.execute("SELECT max(rowid) FROM works_labeled").fetchone()[0]
ids = rng.choice(np.arange(1, n_rows + 1), SAMPLE + OUT_OF_SAMPLE, replace=False)
docs, oos_docs = fetch(ids[:SAMPLE]), fetch(ids[SAMPLE:])
X = cached("X", lambda: S.embed(processed(docs)))
X_oos = cached("XO", lambda: S.embed(processed(oos_docs)))

centers, L = kmeans(X.astype(np.float64), K)
print("K-Means done", flush=True)

ref_idx, ref_lab, rand_idx = [], [], []
for j in range(K):
    members = np.where(L == j)[0]
    dist = np.linalg.norm(X[members] - X[members].mean(0), axis=1)
    central = members[np.argsort(dist)[:20]]
    random = rng.choice(members, min(100, len(members)), replace=False)
    ref_idx += list(central) + list(random)
    ref_lab += [j] * (len(central) + len(random))
    rand_idx += list(random)
ref_idx, ref_lab, rand_idx = np.array(ref_idx), np.array(ref_lab), np.array(rand_idx)
R_raw = cached("R_raw", lambda: S.embed(
    [f"{(docs[i][0] or '').strip()}. {(docs[i][1] or '').strip()}" for i in ref_idx]))

deposited = np.array([r[2] for r in docs])
mapping = {j: Counter(deposited[L == j]).most_common(1)[0][0] for j in range(K)}
to_cons = np.vectorize(mapping.get)
L_oos = ((X_oos[:, None, :] - centers[None]) ** 2).sum(2).argmin(1)
held = np.setdiff1d(np.arange(SAMPLE), ref_idx)
sizes = np.bincount(L, minlength=K)

result = {"n_heldout": int(len(held)), "n_out_of_sample": OUT_OF_SAMPLE}
# as_implemented: references embedded from raw text; preprocessed_both: from the
# same preprocessed text as the documents
for variant, refs in (("as_implemented", R_raw), ("preprocessed_both", X[ref_idx])):
    ru = unit(refs)
    pred = nearest(X[held], ru, ref_lab)
    pred_oos = nearest(X_oos, ru, ref_lab)
    sim = unit(X[rand_idx]) @ ru.T
    for r, i in enumerate(rand_idx):
        sim[r, ref_idx == i] = -np.inf
    p, t = ref_lab[sim.argmax(1)], L[rand_idx]
    result[variant] = {
        "heldout_fine": float((pred == L[held]).mean()),
        "heldout_consolidated": float((to_cons(pred) == to_cons(L[held])).mean()),
        "out_of_sample_fine": float((pred_oos == L_oos).mean()),
        "out_of_sample_consolidated": float((to_cons(pred_oos) == to_cons(L_oos)).mean()),
        "loo_fine": float(sum(sizes[j] * (p[t == j] == j).mean() for j in range(K)) / sizes.sum()),
        "loo_consolidated": float(sum(sizes[j] * (to_cons(p[t == j]) == mapping[j]).mean()
                                      for j in range(K)) / sizes.sum()),
    }
# context: how well the reconstructed consolidation of K-Means matches the deposit
result["kmeans_consolidated_vs_deposited_h1"] = float((to_cons(L) == deposited).mean())

json.dump(result, open(RESULTS / "h1_rerun.json", "w"), indent=2)
print(json.dumps(result, indent=2))
