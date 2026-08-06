#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Contrastive TF-IDF (with lemmatization) for hierarchical clusters, restricting the
“others” comparison to the *same parent layer*:

- H1 (macro): target = given H1, others = all other H1s.
- H2 (meso):  target = given H2 within its H1, others = other H2s within the *same H1*.
- H3 (micro): target = given H3 within its H2, others = other H3s within the *same H2*.

Target micro id is hard-coded as 102 (=> H1=1, H2=10, H3=102).
We sample up to 100k docs from the target group and up to 100k from the others.

Ranking is by Δ mean TF-IDF = mean(target) − mean(others) (top-20 terms).
We also report a smoothed log2 ratio for interpretability.

Output CSV, written next to this script:
  output/tfidf_contrast_lemmas_parent_scoped_h1_1_h2_10_h3_102.csv
"""

import os
import re
import sqlite3
import math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List

from sklearn.feature_extraction.text import TfidfVectorizer

# --- Lemmatization (NLTK) ---
import nltk
from nltk.stem import WordNetLemmatizer

# Ensure wordnet data is available
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4", quiet=True)

LEMMATIZER = WordNetLemmatizer()

# -----------------------
# Configuration
# -----------------------
# Paths are derived from this file's location, so the script runs from any
# checkout. .../code/04_subdiscipline_clustering/SPECTER/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = Path(
    os.environ.get("QSS_DB_PATH", PROJECT_ROOT / "data" / "merged_works_labeled.db")
)

# Hard-coded hierarchical target (three-digit micro id)
MICRO_ID = 102
MACRO_ID = MICRO_ID // 100            # 1
MESO_ID  = MICRO_ID // 10             # 10
# Extract the single-digit components for H2 and H3 (assuming digit-wise coding)
H2_DIGIT = (MICRO_ID // 10) % 10      # 0
H3_DIGIT = MICRO_ID % 10              # 2

SAMPLE_TARGET_MAX = 100_000
SAMPLE_OTHERS_MAX = 100_000

TOP_K = 20  # return top-20 terms

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    f"tfidf_contrast_lemmas_parent_scoped_h1_{MACRO_ID}_h2_{MESO_ID}_h3_{MICRO_ID}.csv"
)

# -----------------------
# Text preprocessing (tokenize -> lemmatize -> rejoin)
# -----------------------
# Keep letters, digits, dash/underscore; collapse others to space
_KEEP_RE = re.compile(r"[^a-zA-Z0-9\-\_\s]+")

def preprocess_and_lemmatize(text: str) -> str:
    """
    Lowercase, strip unwanted chars, simple tokenization,
    and lemmatize tokens to reduce inflectional variants.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _KEEP_RE.sub(" ", text)
    toks = [t for t in text.split() if len(t) > 2]  # drop very short tokens
    # Two-pass lemmatization (verb then noun) to reduce common variants
    lemmas = [LEMMATIZER.lemmatize(LEMMATIZER.lemmatize(t, pos="v"), pos="n") for t in toks]
    return " ".join(lemmas)

# -----------------------
# SQLite helpers
# -----------------------
BASE_FILTER = "language = 'en' AND cleaned_abstract IS NOT NULL"

def _fetch_texts(conn: sqlite3.Connection, where_sql: str, params: Tuple, sample_limit: int) -> pd.DataFrame:
    """
    Fetch a random sample of title + cleaned_abstract joined text from works_labeled
    matching where_sql, limited to sample_limit rows (if positive).
    """
    limit_clause = f" LIMIT {int(sample_limit)}" if sample_limit and sample_limit > 0 else ""
    q = f"""
        SELECT title, cleaned_abstract
        FROM works_labeled
        WHERE {BASE_FILTER} AND ({where_sql})
        ORDER BY RANDOM(){limit_clause}
    """
    df = pd.read_sql_query(q, conn, params=params)
    if df.empty:
        return df
    combined = (df["title"].fillna("") + ". " + df["cleaned_abstract"].fillna("")).astype(str)
    df["text"] = combined.apply(preprocess_and_lemmatize)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df[["text"]]

def _target_others_for_level_parent_scoped(conn: sqlite3.Connection, level: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Return (df_target, df_others, target_desc) for a given level in {'h1','h2','h3'},
    with "others" restricted to the *same parent layer*:

      H1: target = h1 == MACRO_ID;          others = h1 != MACRO_ID
      H2: target = h1 == MACRO_ID & h2 == H2_DIGIT;
          others = h1 == MACRO_ID & h2 != H2_DIGIT
      H3: target = h1 == MACRO_ID & h2 == H2_DIGIT & h3 == H3_DIGIT;
          others = h1 == MACRO_ID & h2 == H2_DIGIT & h3 != H3_DIGIT
    """
    if level == "h1":
        target_desc = f"H1={MACRO_ID}"
        where_target = "h1_cluster = ?"
        params_target = (MACRO_ID,)
        where_others = "h1_cluster != ?"
        params_others = (MACRO_ID,)

    elif level == "h2":
        target_desc = f"H2={MACRO_ID}{H2_DIGIT}"
        # Parent-scoped to the same H1
        where_target = "h1_cluster = ? AND h2_cluster = ?"
        params_target = (MACRO_ID, H2_DIGIT)
        where_others = "h1_cluster = ? AND h2_cluster != ?"
        params_others = (MACRO_ID, H2_DIGIT)

    elif level == "h3":
        target_desc = f"H3={MACRO_ID}{H2_DIGIT}{H3_DIGIT}"
        # Parent-scoped to the same H2 (same H1 and H2)
        where_target = "h1_cluster = ? AND h2_cluster = ? AND h3_cluster = ?"
        params_target = (MACRO_ID, H2_DIGIT, H3_DIGIT)
        where_others = "h1_cluster = ? AND h2_cluster = ? AND h3_cluster != ?"
        params_others = (MACRO_ID, H2_DIGIT, H3_DIGIT)

    else:
        raise ValueError("level must be one of {'h1','h2','h3'}")

    df_tgt = _fetch_texts(conn, where_target, params_target, SAMPLE_TARGET_MAX)
    df_oth = _fetch_texts(conn, where_others, params_others, SAMPLE_OTHERS_MAX)
    return df_tgt, df_oth, target_desc

# -----------------------
# TF-IDF + contrast scoring
# -----------------------
def _fit_vectorizer(texts: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    """
    Fit a TF-IDF vectorizer on `texts` and return (vectorizer, X).
    Texts are already lemmatized; let TfidfVectorizer handle tokenization.
    Auto-tune min_df for small corpora.
    """
    n_docs = len(texts)
    if n_docs == 0:
        return None, None

    min_df = 2 if n_docs < 20_000 else 5

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.6,          # push out very common terms
        min_df=min_df,
        max_features=100_000,
        dtype=np.float64
    )
    X = vectorizer.fit_transform(texts)
    return vectorizer, X

def _contrast_terms(vectorizer: TfidfVectorizer, X, y: np.ndarray, top_k: int = TOP_K):
    """
    Compute per-term contrast:
      Δ = mean_tf_idf(target) - mean_tf_idf(others)
    plus a smoothed log2 ratio for interpretability.
    """
    if X is None or X.shape[0] == 0:
        return []

    target_mask = (y == 1)
    other_mask  = (y == 0)

    if not target_mask.any() or not other_mask.any():
        return []

    mu_t = np.asarray(X[target_mask].mean(axis=0)).ravel()
    mu_o = np.asarray(X[other_mask].mean(axis=0)).ravel()

    delta = mu_t - mu_o
    eps = 1e-9
    log_ratio = np.log2((mu_t + eps) / (mu_o + eps))

    idx = np.argsort(-delta)[:top_k]
    terms = vectorizer.get_feature_names_out()[idx]
    rows = []
    for rank, j in enumerate(idx, start=1):
        rows.append({
            "rank": rank,
            "term": terms[rank-1],
            "mean_tfidf_target": float(mu_t[j]),
            "mean_tfidf_others": float(mu_o[j]),
            "delta": float(delta[j]),
            "log2_ratio": float(log_ratio[j]),
        })
    return rows

# -----------------------
# Main
# -----------------------
def main():
    print(f"[INFO] DB: {DB_PATH}")
    print(f"[INFO] Targets: H1={MACRO_ID}, H2={MESO_ID} (H2 digit={H2_DIGIT}), H3={MICRO_ID} (H3 digit={H3_DIGIT})")
    if not os.path.isfile(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    all_records = []

    for level in ["h1", "h2", "h3"]:
        df_tgt, df_oth, desc = _target_others_for_level_parent_scoped(conn, level)

        n_tgt = len(df_tgt)
        n_oth = len(df_oth)
        print(f"[INFO] {desc}: sampled target={n_tgt:,}, others(parent-scoped)={n_oth:,}")

        if n_tgt == 0 or n_oth == 0:
            print(f"[WARN] Skipping {desc} due to empty sample.")
            continue

        texts = df_tgt["text"].tolist() + df_oth["text"].tolist()
        y = np.array([1] * n_tgt + [0] * n_oth, dtype=np.int8)

        vectorizer, X = _fit_vectorizer(texts)
        if X is None or X.shape[1] == 0:
            print(f"[WARN] Vectorizer produced no features for {desc}. Skipping.")
            continue

        rows = _contrast_terms(vectorizer, X, y, top_k=TOP_K)
        for r in rows:
            r.update({
                "level": level.upper(),
                "cluster_id": (
                    MACRO_ID if level == "h1"
                    else (MESO_ID if level == "h2" else MICRO_ID)
                ),
                "n_docs_target": int(n_tgt),
                "n_docs_others": int(n_oth),
            })
        all_records.extend(rows)

        # Console preview
        print(f"\n[RESULT] Top-{TOP_K} contrastive TF-IDF terms — {desc} (parent-scoped)")
        for r in rows:
            print(f"{r['rank']:2d}. {r['term']:20s} Δ={r['delta']:.6f}  "
                  f"μ_t={r['mean_tfidf_target']:.6f}  μ_o={r['mean_tfidf_others']:.6f}  "
                  f"log2R={r['log2_ratio']:.3f}")
        print("")

    conn.close()

    if all_records:
        out_df = pd.DataFrame(all_records, columns=[
            "level", "cluster_id", "rank", "term",
            "mean_tfidf_target", "mean_tfidf_others", "delta", "log2_ratio",
            "n_docs_target", "n_docs_others"
        ])
        out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"[✓] Saved: {OUTPUT_CSV}")
    else:
        print("[INFO] No output generated.")

if __name__ == "__main__":
    main()
