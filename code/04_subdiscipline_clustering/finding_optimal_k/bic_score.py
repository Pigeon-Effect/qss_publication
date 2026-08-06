import sqlite3
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from pathlib import Path
import warnings
from joblib import Parallel, delayed
import multiprocessing
import gc
import time
from tqdm import tqdm
import scipy.sparse

warnings.filterwarnings("ignore")

# === CONFIG ===
# Paths are derived from this file's location, so the script runs from any
# checkout. .../code/04_subdiscipline_clustering/finding_optimal_k/ -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
INTERIM_DIR = Path(os.environ.get("QSS_INTERIM_DIR", PROJECT_ROOT / "data" / "interim"))

DB_PATH = INTERIM_DIR / "openalex_ai_works_sample_10k.db"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reproducibility seed
MAIN_SEED = 42
np.random.seed(MAIN_SEED)


def decompress_abstract(json_str):
    """Robust inverted index decompression with error handling"""
    if not json_str or not isinstance(json_str, str):
        return ""

    try:
        inverted_index = json.loads(json_str)
        position_word = []
        for word, positions in inverted_index.items():
            if isinstance(positions, int):
                position_word.append((positions, word))
            elif isinstance(positions, list):
                for pos in positions:
                    position_word.append((pos, word))
        position_word.sort(key=lambda x: x[0])
        return ' '.join([w for _, w in position_word])
    except (json.JSONDecodeError, TypeError, ValueError):
        return ""


def load_documents(db_path):
    """Load documents with balanced title representation"""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT title, abstract_inverted_index 
    FROM works 
    WHERE abstract_inverted_index IS NOT NULL 
        AND language = 'en'
        AND LENGTH(abstract_inverted_index) > 20
    """
    cursor = conn.execute(query)
    documents = []
    for row in tqdm(cursor, desc="Loading documents"):
        title, abstract_json = row
        abstract = decompress_abstract(abstract_json)
        if abstract and title:
            # Balanced representation: single title with separator
            full_text = f"{title}. {abstract}"
            documents.append(full_text)
    conn.close()
    return documents


def preprocess_text(text):
    """Text preprocessing without phrase patterns"""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([w for w in text.split() if len(w) > 2])

    return text


def parallel_preprocess(texts):
    """Memory-efficient parallel preprocessing"""
    num_cores = multiprocessing.cpu_count() - 1
    return Parallel(n_jobs=num_cores, prefer="processes")(
        delayed(preprocess_text)(text) for text in tqdm(texts, desc="Preprocessing")
    )


def vectorize_texts(texts):
    """Vectorization with dimensionality reduction"""
    vectorizer = TfidfVectorizer(
        max_df=0.75,  # More aggressive filtering
        min_df=15,
        stop_words='english',
        max_features=10000,  # Reduced features
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(texts)

    # Dimensionality reduction
    svd = TruncatedSVD(n_components=300, random_state=MAIN_SEED)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X_reduced = lsa.fit_transform(X)

    # Diagnostics
    print(f"Original shape: {X.shape}, Reduced shape: {X_reduced.shape}")
    return X_reduced


def compute_k_metrics(X, k, n_trials, max_sample_size):
    """Compute multiple clustering metrics"""
    scores = {
        "silhouette": [],
        "calinski": [],
        "davies": []
    }

    for trial_idx in tqdm(range(n_trials), desc=f"k={k}", leave=False):
        # Unique seed for each trial
        seed = MAIN_SEED + (k * 1000) + trial_idx

        # Spherical k-means (works with normalized vectors)
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            init='k-means++',
            random_state=seed,
            batch_size=512,  # Smaller batches
            n_init=5,  # More initializations
            max_iter=150  # More iterations
        )

        labels = kmeans.fit_predict(X)
        unique_labels = len(np.unique(labels))

        if unique_labels > 1 and unique_labels < X.shape[0]:
            # Silhouette score with sampling
            sample_size = min(max_sample_size, X.shape[0])
            sil_sample = np.random.choice(X.shape[0], sample_size, replace=False)
            sil_score = silhouette_score(
                X[sil_sample],
                labels[sil_sample],
                metric='cosine'
            )
            scores["silhouette"].append(sil_score)

            # Calinski-Harabasz Index
            scores["calinski"].append(calinski_harabasz_score(X, labels))

            # Davies-Bouldin Index (lower is better)
            scores["davies"].append(davies_bouldin_score(X, labels))

    return k, scores


def compute_cluster_metrics_parallel(X, k_values, n_trials=10, max_sample_size=5000):
    """Parallel execution of metric computation"""
    num_cores = min(len(k_values), multiprocessing.cpu_count() - 1)

    results = Parallel(n_jobs=num_cores, verbose=1)(
        delayed(compute_k_metrics)(X, k, n_trials, max_sample_size)
        for k in tqdm(k_values, desc="Cluster Sizes")
    )

    return {k: scores for k, scores in results}


def save_results(results, output_dir):
    """Save raw results to CSV for further analysis"""
    results_data = []
    for k, metrics in results.items():
        for i in range(len(metrics["silhouette"])):
            row = {'k': k}
            for metric_name in metrics.keys():
                row[metric_name] = metrics[metric_name][i]
            results_data.append(row)

    df_results = pd.DataFrame(results_data)
    csv_path = os.path.join(output_dir, 'cluster_metrics.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"Saved raw results to {csv_path}")
    return df_results


def plot_metrics(results, output_dir):
    """Enhanced visualization with multiple metrics"""
    k_values = sorted(results.keys())

    plt.figure(figsize=(14, 10))

    # Silhouette Score
    plt.subplot(3, 1, 1)
    sil_means = [np.mean(results[k]["silhouette"]) for k in k_values]
    plt.plot(k_values, sil_means, 'o-', color='b')
    plt.title('Silhouette Score (Higher is better)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(k_values)

    # Calinski-Harabasz Index
    plt.subplot(3, 1, 2)
    calinski_means = [np.mean(results[k]["calinski"]) for k in k_values]
    plt.plot(k_values, calinski_means, 'o-', color='r')
    plt.title('Calinski-Harabasz Index (Higher is better)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(k_values)

    # Davies-Bouldin Index (inverted)
    plt.subplot(3, 1, 3)
    davies_means = [1 / np.mean(results[k]["davies"]) for k in k_values]  # Invert so higher=better
    plt.plot(k_values, davies_means, 'o-', color='g')
    plt.title('1 / Davies-Bouldin Index (Higher is better)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(k_values)

    plt.tight_layout()

    # Save in multiple formats
    for ext in ['.svg', '.png']:
        output_path = os.path.join(output_dir, f'cluster_metrics_comparison{ext}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved metrics plot to {output_path}")

    plt.close()


def main():
    print(f"\n{'=' * 50}\nStarting Cluster Analysis\n{'=' * 50}")
    start_time = time.time()

    # Load and prepare data
    documents = load_documents(DB_PATH)
    print(f"Loaded {len(documents)} documents with titles and abstracts")

    # Preprocessing
    processed_texts = parallel_preprocess(documents)
    valid_indices = [i for i, text in enumerate(processed_texts) if text and len(text.split()) >= 5]
    processed_texts = [processed_texts[i] for i in valid_indices]
    print(f"Retained {len(processed_texts)} documents after preprocessing")

    # Vectorization with dimensionality reduction
    X = vectorize_texts(processed_texts)

    # Clustering evaluation
    k_values = list(range(5, 51, 5))  # 5, 10, 15, ..., 50
    print(f"Evaluating cluster counts: {k_values}")

    # Reduce trials for testing, increase for final run
    n_trials = 10  # Start with 10 trials, increase to 20-30 for final

    results = compute_cluster_metrics_parallel(
        X,
        k_values,
        n_trials=n_trials,
        max_sample_size=5000
    )

    # Save and visualize results
    df_results = save_results(results, OUTPUT_DIR)
    plot_metrics(results, OUTPUT_DIR)

    # Performance summary
    duration = (time.time() - start_time) / 60
    print(f"\n{'=' * 50}\nAnalysis completed in {duration:.1f} minutes\n{'=' * 50}")


if __name__ == "__main__":
    main()