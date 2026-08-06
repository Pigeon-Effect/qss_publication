"""H1 step 2 of 2 - applying the expert's consolidation of the 50 clusters.

The `remap` dictionary below is the human decision itself: it records which of
the 50 algorithmic clusters the expert judged to belong to each of the five
macro-domains, after inspecting the interactive UMAP and the TF-IDF terms
produced by `specter_embedding_with_central_abstract_extraction.py`.

Every document is then assigned to its nearest reference abstract by cosine
similarity, and inherits that reference's macro-domain. The mapping is a lookup
table, so the step is deterministic and re-runnable.
"""

import os
import json
import numpy as np
import pandas as pd
import sqlite3
import torch
import re
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import umap.umap_ as umap
import matplotlib.pyplot as plt

# --- File Paths ---
# Derived from this file's location, so the script runs from any checkout.
# .../SPECTER/h1_cluster_topic_modeling/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
INTERIM_DIR = Path(os.environ.get("QSS_INTERIM_DIR", PROJECT_ROOT / "data" / "interim"))
RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

json_path = RESOURCES_DIR / "cluster_abstracts_summary.json"
db_path = INTERIM_DIR / "openalex_ai_filtered_dataset_sample_100k.db"
svg_path = RESOURCES_DIR / "mapped_clusters_umap.svg"

# --- Load curated clusters from JSON ---
with open(json_path, "r", encoding="utf-8") as f:
    curated = json.load(f)

# --- Flatten representative texts for embedding ---
reference_texts = []
reference_labels = []
for cluster_id, entry in curated.items():
    name = cluster_id if not cluster_id.isdigit() else f"cluster_{cluster_id}"
    for section in ["most_central_abstracts", "random_abstracts"]:
        for item in entry[section]:
            title = item.get("title") or ""
            abstract = item.get("cleaned_abstract") or ""
            ref_text = title.strip() + ". " + abstract.strip()
            reference_texts.append(ref_text)
            reference_labels.append(name)

# --- Embed reference documents ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('allenai-specter', device=device)
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')

def embed_with_sliding(texts, model, tokenizer, window_size=100, stride=50):
    all_embeds = []
    for text in tqdm(texts, desc="Embedding Texts"):
        tokens = tokenizer.tokenize(text)
        chunks = [tokens[i:i + window_size] for i in range(0, len(tokens), stride)]
        chunk_texts = [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks if chunk]
        if not chunk_texts:
            all_embeds.append(model.encode([''], device=device))
            continue
        embeddings = model.encode(chunk_texts, device=device)
        avg_embedding = np.mean(embeddings, axis=0)
        all_embeds.append(avg_embedding)
    return np.vstack(all_embeds)

ref_embeddings = embed_with_sliding(reference_texts, model, tokenizer)

# --- Load and preprocess database entries ---
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT id, title, cleaned_abstract FROM works WHERE cleaned_abstract IS NOT NULL AND language = 'en'", conn)
conn.close()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if len(word) > 2])
    return text

df['processed_text'] = (df['title'].fillna('') + '. ' + df['cleaned_abstract'].fillna('')).apply(preprocess_text)

# --- Embed all database entries ---
abstract_embeddings = embed_with_sliding(df['processed_text'].tolist(), model, tokenizer)

# --- Compute nearest cluster by cosine similarity ---
from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity(abstract_embeddings, ref_embeddings)
best_match_indices = np.argmax(sims, axis=1)
best_match_labels = [reference_labels[i] for i in best_match_indices]

# --- Map to 5 cluster names ---
remap = {
    # Biomedical & Health Sciences
    "30": "Biomedical & Health Sciences", "47": "Biomedical & Health Sciences", "29": "Biomedical & Health Sciences",
    "1": "Biomedical & Health Sciences", "27": "Biomedical & Health Sciences", "17": "Biomedical & Health Sciences",
    "49": "Biomedical & Health Sciences", "48": "Biomedical & Health Sciences", "35": "Biomedical & Health Sciences",
    "10": "Biomedical & Health Sciences", "9": "Biomedical & Health Sciences",

    # Engineering & Applied Sciences (includes distributed systems and robotics)
    "11": "Engineering & Applied Sciences", "16": "Engineering & Applied Sciences", "24": "Engineering & Applied Sciences",
    "45": "Engineering & Applied Sciences", "34": "Engineering & Applied Sciences", "4": "Engineering & Applied Sciences",
    "43": "Engineering & Applied Sciences", "22": "Engineering & Applied Sciences", "3": "Engineering & Applied Sciences",

    # Natural Sciences
    "41": "Natural Sciences", "25": "Natural Sciences", "7": "Natural Sciences", "28": "Natural Sciences",
    "18": "Natural Sciences", "32": "Natural Sciences", "38": "Natural Sciences", "33": "Natural Sciences",
    "5": "Natural Sciences",

    # Social Sciences (includes education and behavioral sciences)
    "13": "Social Sciences", "39": "Social Sciences", "0": "Social Sciences",
    "31": "Social Sciences", "44": "Social Sciences", "42": "Social Sciences",
    "21": "Social Sciences", "40": "Social Sciences", "6": "Social Sciences",


    # Computer Science & Mathematics
    "23": "Computer Science & Mathematics", "19": "Computer Science & Mathematics", "14": "Computer Science & Mathematics",
    "46": "Computer Science & Mathematics", "20": "Computer Science & Mathematics", "2": "Computer Science & Mathematics",
    "36": "Computer Science & Mathematics", "26": "Computer Science & Mathematics", "8": "Computer Science & Mathematics",
    "12": "Computer Science & Mathematics", "15": "Computer Science & Mathematics", "37": "Computer Science & Mathematics"
}


mapped_final_labels = [remap.get(label.split('_')[-1], label) for label in best_match_labels]
df['mapped_cluster'] = mapped_final_labels

# --- UMAP Projection ---
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
umap_embeddings = umap_model.fit_transform(abstract_embeddings)
df['x'] = umap_embeddings[:, 0]
df['y'] = umap_embeddings[:, 1]

# --- UMAP Plot ---
import matplotlib.pyplot as plt

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Define specific colors for each of the 5 main clusters
cluster_color_map = {
    "Computer Science & Mathematics": "#7b2cbf",  # Purple
    "Biomedical & Health Sciences": "#3a86ff",  # Blue
    "Social Sciences": "#2dd4bf",  # Green
    "Natural Sciences": "#ffd500",  # Yellow
    "Engineering & Applied Sciences": "#fb5607"  # Orange
}

fig, ax = plt.subplots(figsize=(14, 16))

# Count documents per cluster and sort by count (descending)
cluster_counts = df['mapped_cluster'].value_counts()
sorted_clusters_for_legend = cluster_counts.index.tolist()  # Keep original order for legend

# Create plotting order: move "Computer Science & Mathematics" to the end so it's plotted last (on top)
sorted_clusters_for_plotting = sorted_clusters_for_legend.copy()
if "Computer Science & Mathematics" in sorted_clusters_for_plotting:
    sorted_clusters_for_plotting.remove("Computer Science & Mathematics")
    sorted_clusters_for_plotting.append("Computer Science & Mathematics")

for cluster in sorted_clusters_for_plotting:
    sub = df[df['mapped_cluster'] == cluster]
    count = len(sub)
    color = cluster_color_map.get(cluster, "#FF0000")  # Default to red if cluster not found
    ax.scatter(sub['x'], sub['y'], s=5, alpha=0.6,
              color=color)

# Create legend entries in count order (not plotting order)
legend_handles = []
legend_labels = []
for cluster in sorted_clusters_for_legend:
    count = cluster_counts[cluster]
    color = cluster_color_map.get(cluster, "#FF0000")
    # Create a dummy scatter plot for legend
    handle = ax.scatter([], [], s=50, color=color, alpha=0.8)
    legend_handles.append(handle)
    legend_labels.append(f"{cluster} ({count})")

ax.set_xlabel("UMAP 1", fontsize=12, fontfamily='Times New Roman')
ax.set_ylabel("UMAP 2", fontsize=12, fontfamily='Times New Roman')
ax.legend(legend_handles, legend_labels, loc='lower left', fontsize=12, markerscale=1.0, framealpha=0.9, prop={'family': 'Times New Roman'})
plt.tight_layout()
plt.savefig(svg_path, format='svg', bbox_inches='tight')
plt.close()

print("[✓] UMAP saved to:", svg_path)
