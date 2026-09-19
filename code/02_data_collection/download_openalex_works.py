"""Retrieve the raw AI corpus from OpenAlex (manuscript sections 2.1-2.2).

OpenAlex caps the length of a single ``search`` expression, so the 279-term
vocabulary from stage 01 cannot be sent as one Boolean-OR query. The corpus was
therefore retrieved in batches: each run takes a slice of the term list, queries
it as one OR expression, and writes the matching works straight into its own
SQLite database. ``merge_term_batches.py`` then merges the batch databases and
deduplicates them on the OpenAlex work id, so a paper matching terms in several
batches is kept once.

The batches that produced the published corpus were::

    --terms 1-50    --years 2020        --terms 101-200  --years 2020-2024
    --terms 1-50    --years 2021        --terms 201-279  --years 2020-2024
    --terms 1-50    --years 2022-2024
    --terms 51-100  --years 2020-2024

Splitting the first batch by year keeps each run short enough to supervise; the
year filter is part of the query, so the union over the year slices is the same
set of works as a single 2020-2024 run.

Retrieval is resumable. Every page writes its rows and stores the next cursor
next to the database, so a run interrupted after hours continues where it
stopped instead of starting over. Rows are inserted with ``INSERT OR IGNORE`` on
the work id, which makes a resumed or repeated run idempotent.

Only the metadata the study uses is requested; asking for the full record would
multiply transfer volume for fields nothing reads.

Credentials and paths come from the environment; nothing is hardcoded. See
``.env.example`` at the repository root.

Input  - ``code/01_keyword_construction/search_terms.txt``
Output - ``data/interim/openalex_raw/openalex_ai_works_<terms>_search_terms_<years>.db``
         (table ``works``), gitignored as multi-gigabyte derived data
"""

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
# Derived from this file's location, so the script runs from any checkout.
# .../code/02_data_collection/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEARCH_TERMS_FILE = Path(
    os.environ.get(
        "QSS_SEARCH_TERMS",
        PROJECT_ROOT / "code" / "01_keyword_construction" / "search_terms.txt",
    )
)
INTERIM_DIR = Path(os.environ.get("QSS_INTERIM_DIR", PROJECT_ROOT / "data" / "interim"))
RAW_DIR = INTERIM_DIR / "openalex_raw"

# The fields requested from OpenAlex. `authorships` carries the institution
# country codes stage 03 turns into country shares; `abstract_inverted_index`
# is the abstract as OpenAlex distributes it.
SELECT_FIELDS = [
    "id", "doi", "title", "relevance_score",
    "publication_year", "language",
    "primary_location", "type", "type_crossref",
    "open_access", "authorships",
    "countries_distinct_count", "institutions_distinct_count",
    "corresponding_author_ids", "corresponding_institution_ids",
    "apc_list", "apc_paid", "fwci",
    "cited_by_count", "cited_by_percentile_year",
    "biblio", "grants", "referenced_works",
    "abstract_inverted_index",
]

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS works (
    id TEXT PRIMARY KEY,
    doi TEXT,
    title TEXT,
    relevance_score REAL,
    publication_year INTEGER,
    language TEXT,
    host_organization_name TEXT,
    source_issn_1 TEXT,
    type TEXT,
    type_crossref TEXT,
    is_oa INTEGER,
    authorships TEXT,
    countries_distinct_count INTEGER,
    institutions_distinct_count INTEGER,
    corresponding_author_ids TEXT,
    corresponding_institution_ids TEXT,
    apc_list TEXT,
    apc_paid TEXT,
    fwci REAL,
    cited_by_count INTEGER,
    cited_by_percentile_year TEXT,
    biblio TEXT,
    grants TEXT,
    referenced_works TEXT,
    abstract_inverted_index TEXT
)
"""

INSERT_ROW = """
INSERT OR IGNORE INTO works (
    id, doi, title, relevance_score,
    publication_year, language, host_organization_name, source_issn_1,
    type, type_crossref, is_oa, authorships,
    countries_distinct_count, institutions_distinct_count,
    corresponding_author_ids, corresponding_institution_ids,
    apc_list, apc_paid, fwci, cited_by_count,
    cited_by_percentile_year, biblio,
    grants, referenced_works, abstract_inverted_index
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def load_search_terms(path: Path) -> List[str]:
    """Read the curated search-term list, one term per line, '#' for comments."""
    if not path.exists():
        raise SystemExit(
            f"Search-term list not found: {path}\n"
            "This file is the output of stage 01 (keyword construction): the 279\n"
            "curated terms, one per line. Point QSS_SEARCH_TERMS at it if it lives\n"
            "elsewhere. See code/01_keyword_construction/README.md."
        )
    with open(path, encoding="utf-8") as handle:
        terms = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
    if not terms:
        raise SystemExit(f"No search terms found in {path}")
    return terms


def parse_range(spec: str, total: int) -> tuple[int, int]:
    """Parse '51-100' into inclusive, 1-based bounds, clamped to the list length."""
    if "-" in spec:
        first_s, last_s = spec.split("-", 1)
        first, last = int(first_s), int(last_s)
    else:
        first = last = int(spec)
    if first < 1 or last < first:
        raise SystemExit(f"Invalid term range: {spec}")
    if first > total:
        raise SystemExit(f"Term range {spec} starts past the end of the list ({total} terms).")
    return first, min(last, total)


def as_json(value: Any) -> Optional[str]:
    """Serialise a nested OpenAlex field, or None when it is empty."""
    return json.dumps(value, ensure_ascii=False) if value else None


def row_from_work(work: Dict[str, Any]) -> tuple:
    """Flatten one OpenAlex work into the `works` table layout."""
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {} if isinstance(primary_location, dict) else {}
    open_access = work.get("open_access") or {}
    is_oa = open_access.get("is_oa")

    return (
        work.get("id"),
        work.get("doi"),
        work.get("title"),
        work.get("relevance_score"),
        work.get("publication_year"),
        work.get("language"),
        source.get("host_organization_name") if isinstance(source, dict) else None,
        source.get("issn_l") if isinstance(source, dict) else None,
        work.get("type"),
        work.get("type_crossref"),
        (1 if is_oa else 0) if is_oa is not None else None,
        as_json(work.get("authorships")),
        work.get("countries_distinct_count"),
        work.get("institutions_distinct_count"),
        as_json(work.get("corresponding_author_ids")),
        as_json(work.get("corresponding_institution_ids")),
        as_json(work.get("apc_list")),
        as_json(work.get("apc_paid")),
        work.get("fwci"),
        work.get("cited_by_count"),
        as_json(work.get("cited_by_percentile_year")),
        as_json(work.get("biblio")),
        as_json(work.get("grants")),
        as_json(work.get("referenced_works")),
        as_json(work.get("abstract_inverted_index")),
    )


class OpenAlexClient:
    """Cursor-paginated retrieval of every work matching a set of terms."""

    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = "https://api.openalex.org"
        self.session = requests.Session()
        if email:
            self.session.headers.update({"User-Agent": f"PythonClient/1.0 (mailto:{email})"})
        self._email = email
        self._api_key = api_key

    def _params(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(extra)
        if self._email:
            params["mailto"] = self._email
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    def count_works(self, terms: List[str], years: str) -> int:
        params = self._params({
            "search": " OR ".join(f'"{t}"' for t in terms),
            "filter": f"publication_year:{years}",
            "per-page": 1,
            "select": "id",
        })
        response = self.session.get(f"{self.base_url}/works", params=params, timeout=60)
        response.raise_for_status()
        return response.json().get("meta", {}).get("count", 0)

    def download(self, terms: List[str], years: str, db_path: Path, state_path: Path,
                 max_works: Optional[int] = None, pause: float = 0.0,
                 max_retries: int = 5) -> int:
        """Fetch every matching work into `db_path`, resuming if possible."""
        connection = sqlite3.connect(db_path)
        connection.execute(CREATE_TABLE)
        connection.commit()

        cursor_value = "*"
        if state_path.exists():
            saved = json.loads(state_path.read_text(encoding="utf-8")).get("next_cursor")
            if saved is None:
                # A stored cursor of null means the previous run paged to the end.
                print("This batch is already complete; delete its .state.json to re-run.")
                connection.close()
                return 0
            cursor_value = saved
            print(f"Resuming from saved cursor: {cursor_value}")

        base = self._params({
            "search": " OR ".join(f'"{t}"' for t in terms),
            "filter": f"publication_year:{years}",
            "per-page": 200,
            "select": ",".join(SELECT_FIELDS),
            "sort": "publication_date:desc",
        })

        inserted = attempts = 0
        while cursor_value and attempts < max_retries:
            params = dict(base, cursor=cursor_value)
            try:
                response = self.session.get(f"{self.base_url}/works", params=params, timeout=120)
                response.raise_for_status()
            except requests.exceptions.RequestException as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 429:
                    print("Rate limited by OpenAlex (429). Waiting 60 s.")
                    time.sleep(60)
                    continue  # a 429 is not a failed attempt
                attempts += 1
                print(f"Request failed ({exc}). Retry {attempts}/{max_retries}.")
                time.sleep(attempts * 5)
                continue

            payload = response.json()
            results = payload.get("results", [])
            connection.executemany(INSERT_ROW, [row_from_work(w) for w in results])
            connection.commit()
            inserted += len(results)

            cursor_value = payload.get("meta", {}).get("next_cursor")
            state_path.write_text(json.dumps({"next_cursor": cursor_value}, indent=2), encoding="utf-8")
            print(f"Fetched {len(results):>3} works (batch total {inserted:,}).")

            attempts = 0
            if not cursor_value:
                print("All pages retrieved.")
                break
            if max_works is not None and inserted >= max_works:
                print(f"Stopping at --max-works {max_works:,}.")
                break
            if pause:
                time.sleep(pause)

        stored = connection.execute("SELECT COUNT(*) FROM works").fetchone()[0]
        connection.close()
        print(f"\nRows in {db_path.name}: {stored:,}")
        return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--terms", default=None,
                        help="slice of the term list to query, 1-based inclusive, e.g. '201-279' (default: all)")
    parser.add_argument("--years", default="2020-2024", help="OpenAlex publication_year filter")
    parser.add_argument("--terms-file", type=Path, default=SEARCH_TERMS_FILE)
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--max-works", type=int, default=None, help="stop early; for smoke tests")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds to wait between pages")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    all_terms = load_search_terms(args.terms_file)

    first, last = parse_range(args.terms, len(all_terms)) if args.terms else (1, len(all_terms))
    terms = all_terms[first - 1: last]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"openalex_ai_works_{first}-{last}_search_terms_{args.years}"
    db_path = args.out_dir / f"{stem}.db"
    state_path = args.out_dir / f"{stem}.state.json"

    client = OpenAlexClient(
        email=os.environ.get("OPENALEX_MAILTO"),
        api_key=os.environ.get("OPENALEX_API_KEY"),
    )

    print(f"Terms {first}-{last} of {len(all_terms)} ({len(terms)} terms), publication_year:{args.years}")
    print(f"Writing to: {db_path}")
    try:
        expected = client.count_works(terms, args.years)
        print(f"OpenAlex reports {expected:,} matching works for this batch.\n")
    except requests.exceptions.RequestException as exc:
        print(f"Could not read the expected count ({exc}); continuing.\n")

    client.download(terms, args.years, db_path, state_path,
                    max_works=args.max_works, pause=args.pause)


if __name__ == "__main__":
    main()
