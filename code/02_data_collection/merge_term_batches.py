"""Merge the per-batch retrieval databases into one deduplicated corpus (manuscript section 2.2).

``download_openalex_works.py`` writes one SQLite database per term batch. A
publication that matches terms in several batches is retrieved several times, so
the batches cannot simply be concatenated: this script copies every batch into a
single ``works`` table whose primary key is the OpenAlex work id and inserts with
``INSERT OR IGNORE``, so each publication is kept exactly once.

The merged database is what stage 03 reads. Retrieval for the published corpus
yielded 3,346,705 distinct publications at this point, before quality control.

Input  - ``data/interim/openalex_raw/*.db`` (table ``works``)
Output - ``data/interim/openalex_ai_works_merged_deduplicated.db`` (table ``works``)
"""

import argparse
import os
import sqlite3
from pathlib import Path
from typing import List

# ── Paths ─────────────────────────────────────────────────────────────────────
# Derived from this file's location, so the script runs from any checkout.
# .../code/02_data_collection/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = Path(os.environ.get("QSS_INTERIM_DIR", PROJECT_ROOT / "data" / "interim"))
RAW_DIR = INTERIM_DIR / "openalex_raw"
DEFAULT_OUTPUT = INTERIM_DIR / "openalex_ai_works_merged_deduplicated.db"

COLUMNS = [
    "id", "doi", "title", "relevance_score",
    "publication_year", "language", "host_organization_name", "source_issn_1",
    "type", "type_crossref", "is_oa", "authorships",
    "countries_distinct_count", "institutions_distinct_count",
    "corresponding_author_ids", "corresponding_institution_ids",
    "apc_list", "apc_paid", "fwci", "cited_by_count",
    "cited_by_percentile_year", "biblio",
    "grants", "referenced_works", "abstract_inverted_index",
]

CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS works (
    id TEXT PRIMARY KEY,
    {", ".join(f"{c} {'INTEGER' if c in {'publication_year', 'is_oa', 'countries_distinct_count', 'institutions_distinct_count', 'cited_by_count'} else 'REAL' if c in {'relevance_score', 'fwci'} else 'TEXT'}" for c in COLUMNS[1:])}
)
"""


def batch_databases(folder: Path) -> List[Path]:
    paths = sorted(folder.glob("*.db"))
    if not paths:
        raise SystemExit(
            f"No batch databases found in {folder}\n"
            "Run download_openalex_works.py first, or pass --raw-dir."
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing merged database")
    args = parser.parse_args()

    sources = batch_databases(args.raw_dir)

    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(
                f"{args.output} already exists. Pass --overwrite to replace it."
            )
        args.output.unlink()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    target = sqlite3.connect(args.output)
    target.execute("PRAGMA journal_mode = OFF")
    target.execute("PRAGMA synchronous = OFF")
    target.execute(CREATE_TABLE)
    target.commit()

    column_list = ", ".join(COLUMNS)
    total_read = 0

    print(f"Merging {len(sources)} batch database(s) into {args.output.name}\n")
    for i, source in enumerate(sources, 1):
        target.execute("ATTACH DATABASE ? AS batch", (str(source),))
        try:
            rows = target.execute("SELECT COUNT(*) FROM batch.works").fetchone()[0]
            before = target.execute("SELECT COUNT(*) FROM works").fetchone()[0]
            target.execute(
                f"INSERT OR IGNORE INTO works ({column_list}) SELECT {column_list} FROM batch.works"
            )
            target.commit()
            after = target.execute("SELECT COUNT(*) FROM works").fetchone()[0]
            total_read += rows
            print(f"[{i}/{len(sources)}] {source.name}: {rows:,} rows, {after - before:,} new")
        finally:
            target.execute("DETACH DATABASE batch")

    unique = target.execute("SELECT COUNT(*) FROM works").fetchone()[0]
    print("\nVacuuming ...")
    target.execute("VACUUM")
    target.close()

    print(f"\nRows read across batches: {total_read:,}")
    print(f"Distinct publications:    {unique:,}")
    print(f"Written to: {args.output}")


if __name__ == "__main__":
    main()
