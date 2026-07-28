"""Access to the labelled OpenAlex corpus.

The corpus is a SQLite database with one table, ``works_labeled``, holding the
publication metadata and the three cluster labels assigned by the topic-modeling
pipeline. See ``data/README.md`` for the schema and for how to obtain the file.
"""

from __future__ import annotations

import os
import sqlite3
from collections import defaultdict
from typing import NamedTuple

from clustervalidation.config import CLUSTER_ID_SQL, HIERARCHY_LEVELS


class Document(NamedTuple):
    """One publication as used by the validation protocols."""

    id: str
    title: str
    abstract: str


# A mapping from cluster identifier to its member documents.
ClusterMap = dict[str, list[Document]]


def truncate_words(text: str, max_words: int) -> str:
    """Return the first ``max_words`` whitespace-delimited tokens of ``text``.

    Truncation bounds prompt length and cost. It is applied identically to
    home and intruder documents so it cannot itself signal which is which.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def load_clusters(
    db_path: str,
    level: str,
    min_cluster_size: int = 5,
) -> ClusterMap:
    """Load the corpus grouped by cluster at the requested hierarchy level.

    Documents without a usable abstract are excluded, and clusters with fewer
    than ``min_cluster_size`` members are dropped, since a panel cannot be
    drawn from them.

    Raises:
        ValueError: if ``level`` is unknown or fewer than two clusters remain.
        FileNotFoundError: if the database is missing.
    """
    if level not in HIERARCHY_LEVELS:
        raise ValueError(
            f"unknown hierarchy level {level!r}; expected one of {HIERARCHY_LEVELS}"
        )
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"corpus database not found at {db_path}\n"
            "It is not distributed with this repository - see data/README.md."
        )

    query = f"""
        SELECT id, title, cleaned_abstract,
               {CLUSTER_ID_SQL[level]} AS cluster_id
        FROM works_labeled
        WHERE cleaned_abstract IS NOT NULL AND trim(cleaned_abstract) != ''
    """

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(query).fetchall()
    finally:
        connection.close()

    clusters: ClusterMap = defaultdict(list)
    for doc_id, title, abstract, cluster_id in rows:
        clusters[cluster_id].append(Document(doc_id, title or "", abstract))

    eligible = {
        cid: docs for cid, docs in clusters.items() if len(docs) >= min_cluster_size
    }
    if len(eligible) < 2:
        raise ValueError(
            f"need at least 2 clusters with >={min_cluster_size} documents at "
            f"level {level}, found {len(eligible)}"
        )
    return eligible


def corpus_summary(clusters: ClusterMap) -> dict:
    """Return descriptive statistics for a loaded cluster map."""
    sizes = sorted(len(docs) for docs in clusters.values())
    total = sum(sizes)
    return {
        "clusters": len(clusters),
        "documents": total,
        "smallest_cluster": sizes[0],
        "largest_cluster": sizes[-1],
        "median_cluster": sizes[len(sizes) // 2],
    }
