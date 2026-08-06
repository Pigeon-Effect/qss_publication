import sqlite3
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
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

DB_PATH = INTERIM_DIR / "openalex_ai_works_sample_100k.db"
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
    """Load and filter documents with titles and double-weighted titles"""
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
            full_text = f"{title} {title} {abstract}"
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
    """Vectorization matching your clustering example parameters"""
    vectorizer = TfidfVectorizer(
        max_df=0.85,
        min_df=20,
        stop_words='english',
        max_features=15000,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(texts)

    # Make the sparse matrix writeable
    X = scipy.sparse.csr_matrix(X, copy=True)
    X.data.flags.writeable = True

    # Diagnostics
    sparsity = 1 - (X.nnz / (X.shape[0] * X.shape[1]))
    print(f"Vectorized shape: {X.shape}, Sparsity: {sparsity:.2%}")
    return X


def compute_k_silhouettes(X, k, n_trials, max_sample_size):
    """Compute silhouette scores with reproducibility"""
    # Create a writeable copy of the sparse matrix
    X_local = X.copy()
    X_local.data.flags.writeable = True

    scores = []
    sample_size = min(max_sample_size, X_local.shape[0])

    for trial_idx in tqdm(range(n_trials), desc=f"k={k}", leave=False):
        seed = MAIN_SEED + (k * 1000) + trial_idx

        kmeans = MiniBatchKMeans(
            n_clusters=k,
            init='k-means++',
            random_state=seed,
            batch_size=1000,
            n_init='auto',
            max_iter=100
        )

        labels = kmeans.fit_predict(X_local)
        unique_labels = len(np.unique(labels))

        if unique_labels > 1 and unique_labels < X_local.shape[0]:
            score = silhouette_score(
                X_local,
                labels,
                sample_size=sample_size,
                random_state=seed,
                metric='cosine'
            )
            scores.append(score)

    return k, scores


def compute_silhouette_scores_parallel(X, k_values, n_trials=100, max_sample_size=10000):
    """Parallel execution of silhouette score computation"""
    num_cores = min(len(k_values), multiprocessing.cpu_count() - 1)

    # Reduce verbosity to avoid joblib output clutter
    results = Parallel(n_jobs=num_cores, verbose=0)(
        delayed(compute_k_silhouettes)(X, k, n_trials, max_sample_size)
        for k in tqdm(k_values, desc="Cluster Sizes")
    )

    return {k: scores for k, scores in results}


def save_results(results, output_dir):
    """Save raw results to CSV for further analysis"""
    results_data = []
    for k, scores in results.items():
        for score in scores:
            results_data.append({'k': k, 'silhouette_score': score})

    df_results = pd.DataFrame(results_data)
    csv_path = os.path.join(output_dir, 'silhouette_scores.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"Saved raw results to {csv_path}")
    return df_results


def plot_silhouette_scores(results, output_dir):
    """Enhanced visualization with confidence intervals"""
    k_values = sorted(results.keys())
    means = []
    stds = []

    for k in k_values:
        scores = np.array(results[k])
        means.append(np.nanmean(scores))
        stds.append(np.nanstd(scores))

    plt.figure(figsize=(14, 8))
    plt.plot(k_values, means, 'o-', color='b', label='Mean Score')
    plt.fill_between(k_values,
                     np.array(means) - np.array(stds),
                     np.array(means) + np.array(stds),
                     color='lightblue', alpha=0.4, label='±1 SD')

    plt.xlabel('Number of Clusters (k)', fontsize=14)
    plt.ylabel('Silhouette Score', fontsize=14)
    plt.title('Cluster Quality Analysis', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(k_values)

    # Save in multiple formats
    for ext in ['.svg', '.png']:
        output_path = os.path.join(output_dir, f'silhouette_scores{ext}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved silhouette plot to {output_path}")

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

    # Vectorization
    X = vectorize_texts(processed_texts)

    # Clustering evaluation
    k_values = list(range(5, 41))
    print(f"Evaluating cluster counts: {k_values}")

    # Reduce trials for testing, increase for final run
    n_trials = 10  # Reduced for stability testing - increase to 50-100 for final

    results = compute_silhouette_scores_parallel(
        X,
        k_values,
        n_trials=n_trials,
        max_sample_size=5000  # Reduced sample size
    )

    # Save and visualize results
    df_results = save_results(results, OUTPUT_DIR)
    plot_silhouette_scores(results, OUTPUT_DIR)

    # Performance summary
    duration = (time.time() - start_time) / 60
    print(f"\n{'=' * 50}\nAnalysis completed in {duration:.1f} minutes\n{'=' * 50}")


if __name__ == "__main__":
    main()