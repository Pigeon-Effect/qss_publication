import sqlite3
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --------------------------
# Config
# --------------------------
# Paths are derived from this file's location, so the script runs from any
# checkout. .../code/04_subdiscipline_clustering/finding_optimal_k/ -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
INTERIM_DIR = Path(os.environ.get("QSS_INTERIM_DIR", PROJECT_ROOT / "data" / "interim"))

k_search_output_path = Path(__file__).resolve().parent / "output" / "k_search"
os.makedirs(k_search_output_path, exist_ok=True)

# --------------------------
# Load and Preprocess from SQLite
# --------------------------
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if len(word) > 2])
    return text

db_path = INTERIM_DIR / "h1_cluster_subsets" / "natural_science_sample.db"
conn = sqlite3.connect(db_path)
query = "SELECT title, cleaned_abstract FROM works_labeled WHERE cleaned_abstract IS NOT NULL AND language = 'en'"
df = pd.read_sql_query(query, conn)
conn.close()

df['processed_text'] = (df['title'].fillna('') + '. ' + df['cleaned_abstract']).apply(preprocess_text)

# --------------------------
# Generate Embeddings using SciBERT
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('allenai-specter', device=device)
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')

texts = df['processed_text'].tolist()
window_size = 100
stride = 50

all_embeddings = []
for text in tqdm(texts, desc="Generating Embeddings"):
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + window_size] for i in range(0, len(tokens), stride)]
    chunk_texts = [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks if len(chunk) > 0]
    if not chunk_texts:
        all_embeddings.append(model.encode([''], device=device))
        continue
    embeddings = model.encode(chunk_texts, device=device)
    embeddings = np.mean(embeddings, axis=0)
    all_embeddings.append(embeddings)
embeddings = np.vstack(all_embeddings)

# --------------------------
# Elbow & Silhouette Methods
# --------------------------
k_values = list(range(2, 21))
inertias = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(embeddings, labels))

# --------------------------
# Plot Results
# --------------------------
fig, ax1 = plt.subplots(figsize=(12, 8))

color1 = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Elbow)', color=color1)
ax1.plot(k_values, inertias, marker='o', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Silhouette Score', color=color2)
ax2.plot(k_values, silhouette_scores, marker='s', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Elbow Method and Silhouette Score for Optimal k')
plt.tight_layout()
plt.savefig(os.path.join(k_search_output_path, "k_optimization.svg"), format='svg', bbox_inches='tight')
plt.show()
