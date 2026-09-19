"""Project a sample of the labelled corpus into two dimensions (manuscript section 4.1).

This is the first step towards the semantic landscape: it turns documents into
the scatter that ``summarize_clusters.py`` then summarises.

The corpus is too large to embed and project in full for a figure, so a random
sample is drawn - 100,000 publications by default, which is dense enough that
every one of the 106 micro-clusters is represented by a few hundred points at
least. The sample is drawn with an explicit seed, so a re-run selects the same
publications.

Each document is encoded exactly as in stage 04, so the projection shows the
same embedding space the clustering was built in:

* text is the title followed by the abstract, lowercased, with non-alphabetic
  characters stripped and tokens of one or two characters dropped; stop words
  are kept, because they carry stylistic signal SPECTER was trained on,
* SPECTER encodes 100-token windows with stride 50 and the window vectors are
  averaged, so an abstract longer than the model's context still contributes in
  full,
* the 768-dimensional vectors are projected with UMAP.

Embedding 100,000 documents takes hours on a CPU and roughly ten minutes on a
CUDA device. The vectors can be kept with ``--save-embeddings`` so the
projection can be repeated without re-encoding.

Input  - the labelled corpus (``QSS_DB_PATH``, table ``works_labeled``)
Output - ``data/interim/semantic_landscape/points.csv.gz`` with one row per
         sampled publication (id, cluster labels, x, y) and ``run_config.json``
         recording the parameters
"""

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
# Derived from this file's location, so the script runs from any checkout.
# .../code/06_visualization/semantic_landscape/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
INTERIM_DIR = Path(os.environ.get("QSS_INTERIM_DIR", PROJECT_ROOT / "data" / "interim"))
LANDSCAPE_DIR = INTERIM_DIR / "semantic_landscape"
DB_PATH = Path(os.environ.get("QSS_DB_PATH", PROJECT_ROOT / "data" / "merged_works_labeled.db"))
TABLE = "works_labeled"

# Encoder and projection settings, identical to stage 04.
MODEL_NAME = "allenai-specter"
WINDOW_TOKENS = 100
STRIDE_TOKENS = 50
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"

NON_ALPHA = re.compile(r"[^a-z\s]")


def preprocess(text: str) -> str:
    """Lowercase, strip non-alphabetic characters, drop very short tokens."""
    lowered = NON_ALPHA.sub(" ", str(text).lower())
    return " ".join(token for token in lowered.split() if len(token) > 2)


def load_sample(db_path: Path, size: int, seed: int) -> pd.DataFrame:
    """Draw a seeded random sample of labelled publications with abstracts."""
    if not db_path.exists():
        raise SystemExit(
            f"Corpus not found: {db_path}\n"
            "Place the labelled corpus there or point QSS_DB_PATH at it; see data/README.md."
        )
    connection = sqlite3.connect(db_path)
    try:
        rowids = pd.read_sql_query(
            f"SELECT rowid FROM {TABLE} "
            "WHERE cleaned_abstract IS NOT NULL AND h3_cluster IS NOT NULL",
            connection,
        )["rowid"].to_numpy()
        print(f"Publications eligible for the sample: {len(rowids):,}")

        rng = np.random.default_rng(seed)
        if size < len(rowids):
            rowids = rng.choice(rowids, size=size, replace=False)
        rowids = np.sort(rowids)

        frames = []
        for start in range(0, len(rowids), 5000):
            chunk = rowids[start: start + 5000]
            placeholders = ",".join("?" * len(chunk))
            frames.append(pd.read_sql_query(
                f"SELECT id, title, cleaned_abstract, h1_cluster, h2_cluster, h3_cluster "
                f"FROM {TABLE} WHERE rowid IN ({placeholders})",
                connection, params=[int(v) for v in chunk],
            ))
    finally:
        connection.close()

    frame = pd.concat(frames, ignore_index=True)
    print(f"Sampled {len(frame):,} publications")
    return frame


def embed(texts: list[str], batch_size: int) -> np.ndarray:
    """Average SPECTER embeddings of overlapping token windows per document."""
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding {len(texts):,} documents with {MODEL_NAME} on {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter")

    vectors = np.zeros((len(texts), model.get_sentence_embedding_dimension()), dtype=np.float32)
    windows: list[str] = []
    owners: list[int] = []

    def flush() -> None:
        if not windows:
            return
        encoded = model.encode(windows, batch_size=batch_size, show_progress_bar=False)
        for index, vector in zip(owners, encoded):
            vectors[index] += vector
        windows.clear()
        owners.clear()

    counts = np.zeros(len(texts), dtype=np.int32)
    for i, text in enumerate(texts):
        tokens = tokenizer.tokenize(text)
        pieces = [tokens[start: start + WINDOW_TOKENS]
                  for start in range(0, max(len(tokens), 1), STRIDE_TOKENS)]
        pieces = [p for p in pieces if p] or [[]]
        for piece in pieces:
            windows.append(tokenizer.convert_tokens_to_string(piece))
            owners.append(i)
            counts[i] += 1
        if len(windows) >= batch_size * 8:
            flush()
        if (i + 1) % 5000 == 0:
            print(f"  tokenised {i + 1:,}/{len(texts):,}")
    flush()

    vectors /= np.maximum(counts, 1)[:, None]
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=LANDSCAPE_DIR / "points.csv.gz")
    parser.add_argument("--save-embeddings", action="store_true",
                        help="also keep the 768-dimensional vectors as .npy")
    args = parser.parse_args()

    frame = load_sample(args.db, args.sample_size, args.seed)
    texts = [preprocess(f"{title} {abstract}")
             for title, abstract in zip(frame["title"].fillna(""), frame["cleaned_abstract"].fillna(""))]

    embeddings = embed(texts, args.batch_size)

    import umap
    print(f"Projecting with UMAP (n_neighbors={UMAP_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, "
          f"metric={UMAP_METRIC}, random_state={args.seed})")
    reducer = umap.UMAP(n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                        metric=UMAP_METRIC, random_state=args.seed)
    coordinates = reducer.fit_transform(embeddings)

    points = pd.DataFrame({
        "id": frame["id"],
        "h1_cluster": frame["h1_cluster"].astype(int),
        "h2_cluster": frame["h2_cluster"].astype(int),
        "h3_cluster": frame["h3_cluster"].astype(int),
        "x": coordinates[:, 0],
        "y": coordinates[:, 1],
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(args.output, index=False)
    print(f"Points written to: {args.output}")

    config = {
        "db_path": str(args.db), "table": TABLE,
        "model": MODEL_NAME, "window_tokens": WINDOW_TOKENS, "stride_tokens": STRIDE_TOKENS,
        "batch_size": args.batch_size, "sample_size": args.sample_size, "seed": args.seed,
        "umap": {"n_neighbors": UMAP_NEIGHBORS, "min_dist": UMAP_MIN_DIST,
                 "metric": UMAP_METRIC, "random_state": args.seed},
        "num_documents": int(len(points)), "embedding_dim": int(embeddings.shape[1]),
    }
    config_path = args.output.parent / "run_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Run parameters written to: {config_path}")

    if args.save_embeddings:
        vectors_path = args.output.parent / "embeddings.npy"
        np.save(vectors_path, embeddings)
        print(f"Embeddings written to: {vectors_path}")


if __name__ == "__main__":
    main()
