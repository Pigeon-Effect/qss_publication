"""Assignment accuracy at h2 and h3, from the archived reference sets.

The full corpus was labeled by assigning each document to the cluster of its
most similar reference document. This script measures how often that
assignment reproduces the K-Means label, using the original K-Means clusters
and the original references archived in `SPECTER/*/resources/`.

Each cluster's 100 random references are a random sample of its members, so
they stand in for held-out documents. Each is embedded the way a document is
(preprocessed text), assigned to its nearest *other* reference (references
embedded from raw text, as the labeling scripts do), and scored against its own
cluster: leave-one-out. Per-cluster accuracy is weighted by cluster size.

Two accuracies are reported:
  fine          same K-Means cluster
  consolidated  same consolidated cluster after the expert mapping

Only one h3 mapping survives in the scripts, so the mapping is reconstructed
from the deposited labels: each K-Means cluster maps to the most common final
code among its references (matched to the corpus by title).

Writes results/assignment_accuracy/loo_per_cluster.csv and loo_summary.json.
Embeddings are cached in output/; a full run takes about an hour on a GPU.
"""

import csv
import glob
import json
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import specter_onnx as S

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPECTER_DIR = Path(__file__).resolve().parents[1] / "SPECTER"
DB_PATH = Path(os.environ.get(
    "QSS_DB_PATH", PROJECT_ROOT / "data" / "original_dataset" / "merged_works_labeled.db"))
OUT = Path(__file__).resolve().parent / "output"
RESULTS = PROJECT_ROOT / "results" / "assignment_accuracy"
OUT.mkdir(exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

SOURCES = [("h2", Path(f).parents[1].name, Path(f)) for f in sorted(glob.glob(
    str(SPECTER_DIR / "h2_cluster_topic_modeling/*/resources/cluster_abstracts_summary.json")))]
SOURCES += [("h3", Path(f).parent.name, Path(f)) for f in sorted(glob.glob(
    str(SPECTER_DIR / "h3_cluster_topic_modeling/resources/*/*.json")))]


def load_refs(path):
    """References in the order the labeling scripts read them."""
    data = json.load(open(path, encoding="utf-8"))
    texts, labels, keys, queries = [], [], [], []
    for cid, entry in data.items():
        for section in ("most_central_abstracts", "random_abstracts"):
            for item in entry[section]:
                title = (item.get("title") or "").strip()
                abstract = (item.get("cleaned_abstract") or "").strip()
                if section == "random_abstracts":
                    queries.append(len(texts))
                texts.append(f"{title}. {abstract}")
                labels.append(cid)
                keys.append((title, abstract))
    return data, texts, np.array(labels), keys, np.array(queries)


def deposited_labels_by_title():
    cache = OUT / "deposited_labels_by_title.json"
    if not cache.exists():
        wanted = set()
        for _, _, path in SOURCES:
            wanted.update(k[0] for k in load_refs(path)[3])
        found = {}
        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        query = "SELECT title, h1_cluster, h2_cluster, h3_cluster FROM works_labeled"
        for title, h1, h2, h3 in con.execute(query):
            title = (title or "").strip()
            if title in wanted:
                found.setdefault(title, [h1, h2, h3])
        json.dump(found, open(cache, "w"))
    return json.load(open(cache))


def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


deposited = deposited_labels_by_title()
rows = []
for level, name, path in SOURCES:
    data, texts, labels, keys, queries = load_refs(path)
    cache = OUT / f"{level}_{name}.npz"
    if not cache.exists():
        print(f"[{level} {name}] embedding {len(texts)} references", flush=True)
        refs = S.embed(texts, progress=False)
        docs = S.embed([S.preprocess(texts[i]) for i in queries], progress=False)
        np.savez(cache, R=refs, Q=docs)
    z = np.load(cache)
    refs, docs = z["R"], z["Q"]

    code = 1 if level == "h2" else 2  # position of the final code in [h1, h2, h3]
    votes = defaultdict(Counter)
    for (title, _), cid in zip(keys, labels):
        hit = deposited.get(title)
        if hit and hit[code] is not None:
            votes[cid][hit[code]] += 1
    mapping = {cid: v.most_common(1)[0][0] for cid, v in votes.items()}

    key_ids = np.unique([hash(k) for k in keys], return_inverse=True)[1]
    # as_implemented: documents preprocessed, references raw (what was run)
    # raw_both: documents embedded from raw text too (the asymmetry's cost)
    for variant, q in (("as_implemented", docs), ("raw_both", refs[queries])):
        sim = unit(q) @ unit(refs).T
        for r, i in enumerate(queries):  # leave the document and its duplicates out
            sim[r, key_ids == key_ids[i]] = -np.inf
        pred = labels[sim.argmax(1)]
        true = labels[queries]
        for cid in data:
            m = true == cid
            rows.append({
                "level": level, "slice": name, "variant": variant, "cluster": cid,
                "cluster_size": data[cid]["num_documents"], "n_tested": int(m.sum()),
                "fine": round(float((pred[m] == cid).mean()), 4),
                "consolidated": round(float(np.mean(
                    [mapping.get(p) == mapping.get(cid) for p in pred[m]])), 4),
            })
    print(f"[{level} {name}] done", flush=True)

with open(RESULTS / "loo_per_cluster.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary = {}
for level in ("h2", "h3"):
    for variant in ("as_implemented", "raw_both"):
        sel = [r for r in rows if r["level"] == level and r["variant"] == variant]
        w = np.array([r["cluster_size"] for r in sel], float)
        summary[f"{level}_{variant}"] = {
            "fine": round(float(np.dot(w, [r["fine"] for r in sel]) / w.sum()), 4),
            "consolidated": round(float(np.dot(w, [r["consolidated"] for r in sel]) / w.sum()), 4),
            "clusters": len(sel), "documents_represented": int(w.sum()),
        }
json.dump(summary, open(RESULTS / "loo_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
