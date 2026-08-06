"""
h2_umap_natural_science.py
---------------------------------------------------
Static UMAP of the consolidated meso-clusters for the NATURAL SCIENCE subset
(h1 = 3).

This is the countercheck for the expert consolidation, not a separate decision:
it re-applies the same `remap` dictionary used by the labelling script and
draws the result as a static UMAP, so the expert can see whether the merged
groups actually occupy coherent regions of the embedding space.

Meso codes (manuscript Figure 1, macro-domain 3 - Natural Science):

    0 : Climate & Hydrology
    1 : Material Science
    2 : Energy & Thermodynamics
    3 : Agriculture & Forestry
    4 : Astrophysics & Quantumphysics

INCOMPLETE relative to the manuscript, which gives this domain seven
meso-clusters. See the header of h2_labeling_natural_science.py.
"""
import os, json, re, sqlite3, torch
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── File paths ──────────────────────────────────────────────────────────────────
# Derived from this file's location, so the script runs from any checkout.
# .../SPECTER/h2_cluster_topic_modeling/<domain>/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[5]
INTERIM_DIR = Path(os.environ.get("QSS_INTERIM_DIR", PROJECT_ROOT / "data" / "interim"))
SUBSETS_DIR = INTERIM_DIR / "h1_cluster_subsets"
RESOURCES_DIR = Path(__file__).resolve().parent / "resources"

JSON_PATH = RESOURCES_DIR / "cluster_abstracts_summary_002.json"
DB_PATH   = SUBSETS_DIR / "natural_science_sample.db"
SVG_PATH  = RESOURCES_DIR / "mapped_clusters_umap.svg"

# ── Load curated clusters (reference documents) ────────────────────────────────
with open(JSON_PATH, encoding="utf-8") as f:
    curated = json.load(f)

reference_texts, reference_labels = [], []
for cl_id, entry in curated.items():
    label = f"cluster_{cl_id}" if cl_id.isdigit() else cl_id
    for section in ("most_central_abstracts", "random_abstracts"):
        for item in entry[section]:
            title = item.get("title") or ""
            abstract = item.get("cleaned_abstract") or ""
            txt = (title.strip() + ". " + abstract.strip())
            reference_texts.append(txt)
            reference_labels.append(label)

# ── Embedding model & helper ────────────────────────────────────────────────────
device   = "cuda" if torch.cuda.is_available() else "cpu"
model    = SentenceTransformer("allenai-specter", device=device)
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")

def embed_sliding(texts, window=100, stride=50):
    vecs = []
    for t in tqdm(texts, desc="Embedding"):
        toks   = tokenizer.tokenize(t)
        chunks = [toks[i:i+window] for i in range(0, len(toks), stride)]
        chunk_texts = [tokenizer.convert_tokens_to_string(c) for c in chunks if c]
        emb = model.encode(chunk_texts or [''], device=device)
        vecs.append(emb.mean(axis=0))
    return np.vstack(vecs)

ref_emb = embed_sliding(reference_texts)

# ── Load natural‑science subsample from SQLite ──────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df   = pd.read_sql_query("""
        SELECT id, title, cleaned_abstract
        FROM   works_labeled
        WHERE  cleaned_abstract IS NOT NULL
           AND language = 'en';
       """, conn)
conn.close()

# ── Light text cleaning ────────────────────────────────────────────────────────
def clean(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r'[^a-z\s]', ' ', txt)
    return ' '.join(w for w in txt.split() if len(w) > 2)

df["processed_text"] = (df["title"].fillna('') + '. ' +
                        df["cleaned_abstract"].fillna('')).apply(clean)

abs_emb = embed_sliding(df["processed_text"].tolist())

# ── Cosine‑similarity assignment to nearest reference document ─────────────────
sim     = cosine_similarity(abs_emb, ref_emb)
nearest = np.argmax(sim, axis=1)
df["ref_label"] = [reference_labels[i] for i in nearest]

# ── Outlier detection and removal ──────────────────────────────────────────────
print(f"Original dataset size: {len(df)}")

# Use Isolation Forest for outlier detection on embeddings
scaler = StandardScaler()
abs_emb_scaled = scaler.fit_transform(abs_emb)

# Detect outliers using Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
outlier_labels = iso_forest.fit_predict(abs_emb_scaled)

# Filter out outliers (outlier_labels == -1 means outlier)
outlier_mask = outlier_labels == 1  # Keep only inliers (1 = inlier, -1 = outlier)
df_filtered = df[outlier_mask].reset_index(drop=True)
abs_emb_filtered = abs_emb[outlier_mask]

print(f"Filtered dataset size: {len(df_filtered)} (removed {len(df) - len(df_filtered)} outliers)")

# Update dataframe reference for subsequent operations
df = df_filtered
abs_emb = abs_emb_filtered

# ── Macro‑area mapping  ────────────────────────────────────────────────────────
remap = {
    # ―― Climate & Hydrology ――
    "1":"Climate & Hydrology",  "5":"Climate & Hydrology",
    "11":"Climate & Hydrology", "15":"Climate & Hydrology",
    "18":"Climate & Hydrology", "43":"Climate & Hydrology",
    "49":"Climate & Hydrology", "0":"Climate & Hydrology",
    "44":"Climate & Hydrology", "31":"Climate & Hydrology",
    "13":"Climate & Hydrology", "27":"Climate & Hydrology",
    "32":"Climate & Hydrology", "12":"Climate & Hydrology",
    "16":"Climate & Hydrology",

    # ―― Agriculture & Forestry ――
    "4":"Agriculture & Forestry", "6":"Agriculture & Forestry",
    "7":"Agriculture & Forestry", "20":"Agriculture & Forestry",
    "22":"Agriculture & Forestry", "25":"Agriculture & Forestry",
    "35":"Agriculture & Forestry", "47":"Agriculture & Forestry",
    "38":"Agriculture & Forestry",

    # ―― Material Science ――
    "2":"Material Science",  "17":"Material Science",
    "19":"Material Science", "28":"Material Science",
    "34":"Material Science", "36":"Material Science",
    "37":"Material Science", "46":"Material Science",
    "10":"Material Science", "29":"Material Science",
    "21":"Material Science",
    "8":"Material Science",  "39":"Material Science",
    "42":"Material Science",

    # ―― Energy & Thermodynamics ――
    "9":"Energy & Thermodynamics",  "24":"Energy & Thermodynamics",
    "33":"Energy & Thermodynamics", "41":"Energy & Thermodynamics", "48":"Energy & Thermodynamics",
    "3":"Energy & Thermodynamics",  "26":"Energy & Thermodynamics", "30":"Energy & Thermodynamics",

    # ―― Astrophysics & Quantumphysics ――
    "14":"Astrophysics & Quantumphysics", "23":"Astrophysics & Quantumphysics",
    "40":"Astrophysics & Quantumphysics", "45":"Astrophysics & Quantumphysics"
}

# ----- Sanity‑check: each numeric key appears only once  -----------------------
dupes = [k for k in set(remap) if list(remap.keys()).count(k) > 1]
if dupes:
    raise ValueError(
        f"Cluster ID(s) {dupes} appear in more than one macro‑cluster."
        " Resolve duplicates (see TODO comments) before proceeding."
    )

df["macro_cluster"] = [
    remap.get(lbl.split('_')[-1], "Unmapped") for lbl in df["ref_label"]
]

# ── 2‑D UMAP -------------------------------------------------------------------
umap_xy = umap.UMAP(n_neighbors=15, min_dist=0.1,
                    metric='cosine', random_state=42).fit_transform(abs_emb)
df["x"], df["y"] = umap_xy[:,0], umap_xy[:,1]

# ── Plot settings --------------------------------------------------------------
plt.rcParams['font.family'] = 'Times New Roman'
palette = {
    "Climate & Hydrology":        "#3a86ff",
    "Agriculture & Forestry":     "#2dd4bf",
    "Material Science":           "#ffbe0b",
    "Energy & Thermodynamics":    "#fb5607",
    "Astrophysics & Quantumphysics":          "#7209b7"
}

fig, ax = plt.subplots(figsize=(13, 15))
order = df["macro_cluster"].value_counts().index.tolist()

for cl in order:
    sub   = df[df["macro_cluster"] == cl]
    color = palette.get(cl, "#d3d3d3")
    ax.scatter(sub["x"], sub["y"], s=5, alpha=0.6, color=color)

# Legend with counts
handles, labels = [], []
for cl in order:
    handles.append(ax.scatter([], [], s=50, color=palette.get(cl, "#d3d3d3")))
    labels.append(f"{cl} ({len(df[df['macro_cluster']==cl])})")

ax.set_xlabel("UMAP 1", fontsize=12)
ax.set_ylabel("UMAP 2", fontsize=12)
ax.legend(handles, labels, loc='lower left',
          fontsize=12, framealpha=0.9)
plt.tight_layout()
plt.savefig(SVG_PATH, format="svg", bbox_inches="tight")
plt.close()

print("✓ UMAP written to:", SVG_PATH)
