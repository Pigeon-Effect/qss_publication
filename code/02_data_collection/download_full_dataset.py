"""Retrieve the raw AI corpus from OpenAlex (manuscript section 2.2).

The search vocabulary is the 279-term list built in stage 01. OpenAlex caps the
length of a single ``search`` expression, so the terms cannot be sent as one
Boolean-OR query: they are split into groups, each group is retrieved with
cursor pagination, and the groups are merged and deduplicated on the OpenAlex
work id afterwards.

Credentials and paths come from the environment; nothing is hardcoded. See
``.env.example`` at the repository root.
"""

import requests
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import os

from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
# Derived from this file's location, so the script runs from any checkout.
# .../code/02_data_collection/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The 279 curated search terms produced by stage 01, one term per line.
SEARCH_TERMS_FILE = Path(
    os.environ.get(
        "QSS_SEARCH_TERMS",
        PROJECT_ROOT / "code" / "01_keyword_construction" / "search_terms.txt",
    )
)

# Raw retrieval output. Gitignored: this is multi-gigabyte derived data.
INTERIM_DIR = Path(os.environ.get("QSS_INTERIM_DIR", PROJECT_ROOT / "data" / "interim"))
RAW_DIR = INTERIM_DIR / "openalex_raw"

# OpenAlex rejects over-long `search` expressions, so the term list is queried
# in groups of this size and the results deduplicated afterwards.
TERMS_PER_QUERY = 25


class OpenAlexClient:
    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the OpenAlex client with optional authentication for higher rate limits.

        Args:
            email: Your email for polite pool (optional but recommended)
            api_key: Your API key if you have one for raised rate limits
        """
        self.base_url = "https://api.openalex.org"
        self.session = requests.Session()
        headers = {}
        if email:
            headers["User-Agent"] = f"PythonClient/1.0 (mailto:{email})"
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            self.session.headers.update(headers)
        self._email = email
        self._api_key = api_key

    def count_works_by_terms(self, terms: List[str]) -> int:
        """
        Count works that contain any of the specified terms in title or abstract.
        """
        search_query = " OR ".join([f'"{term}"' for term in terms])
        try:
            url = f"{self.base_url}/works"
            params: Dict[str, Any] = {
                "search": search_query,
                "filter": "publication_year:2020-2024",
                "per-page": 1,
                "select": "id"
            }
            if self._email:
                params['mailto'] = self._email
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("meta", {}).get("count", 0)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for count: {e}")
            return 0

    def get_all_works(self, terms: List[str],
                      output_filename: str,
                      state_filename: str,
                      save_interval: int = 100000,
                      max_retries: int = 5) -> List[Dict[str, Any]]:
        """
        Get all works, sorted newest first, with incremental saving, resumption,
        and adjusted rate limiting.
        """
        search_query = " OR ".join([f'"{term}"' for term in terms])
        url = f"{self.base_url}/works"
        select_fields = [
            "id", "doi", "title", "display_name", "relevance_score",
            "publication_year", "language",
            "primary_location", "type", "type_crossref",
            "open_access", "authorships",
            "countries_distinct_count", "institutions_distinct_count",
            "corresponding_author_ids", "corresponding_institution_ids",
            "apc_list", "apc_paid", "fwci",
            "cited_by_count",
            "cited_by_percentile_year",
            "biblio", "locations_count",
            "locations",
            "grants", "referenced_works",
            "abstract_inverted_index"
        ]

        all_fetched_works: List[Dict[str, Any]] = []
        initial_cursor = "*"

        if os.path.exists(output_filename) and os.path.exists(state_filename):
            print(
                f"Found existing data file '{output_filename}' and state file '{state_filename}'. Attempting to resume.")
            try:
                with open(output_filename, 'r', encoding='utf-8') as f_data:
                    all_fetched_works = json.load(f_data)
                with open(state_filename, 'r', encoding='utf-8') as f_state:
                    state_data = json.load(f_state)
                    initial_cursor = state_data.get("next_cursor", "*")

                if initial_cursor is None:
                    print(
                        f"Previous download completed successfully. Loaded {len(all_fetched_works)} works. No further fetching needed.")
                    return all_fetched_works
                print(f"Resuming download. Loaded {len(all_fetched_works)} works. Will use cursor: {initial_cursor}")
            except Exception as e:
                print(f"Error loading previous state ({e}), starting fresh.")
                all_fetched_works = []
                initial_cursor = "*"

        params: Dict[str, Any] = {
            "search": search_query,
            "filter": "publication_year:2020-2024",
            "per-page": 200,
            "select": ",".join(select_fields),
            "sort": "publication_date:desc",
            "cursor": initial_cursor
        }
        if self._email:
            params['mailto'] = self._email

        attempts = 0
        works_since_last_save = 0
        current_cursor = initial_cursor

        print(f"Fetching works. Initial cursor: {current_cursor}")
        print(f"Query: {search_query}, Filter: {params['filter']}, Sort: {params['sort']}")

        while current_cursor and attempts < max_retries:
            params["cursor"] = current_cursor
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])

                if results:
                    all_fetched_works.extend(results)
                    works_since_last_save += len(results)

                next_api_cursor = data.get("meta", {}).get("next_cursor")
                print(
                    f"Fetched {len(results)} new works. Total: {len(all_fetched_works)}. Next API cursor: {next_api_cursor}")

                if works_since_last_save >= save_interval and results:
                    print(f"Reached save interval ({save_interval}). Saving {len(all_fetched_works)} works...")
                    try:
                        with open(output_filename, "w", encoding="utf-8") as f_data_out:
                            json.dump(all_fetched_works, f_data_out, ensure_ascii=False, indent=2)
                        with open(state_filename, "w", encoding="utf-8") as f_state_out:
                            json.dump({"next_cursor": next_api_cursor}, f_state_out, indent=2)
                        print(f"Successfully saved to {output_filename} and state to {state_filename}.")
                        works_since_last_save = 0
                    except Exception as e:
                        print(f"Error during incremental save: {e}")

                current_cursor = next_api_cursor
                if not current_cursor:
                    print("All pages processed. No more works to fetch.")
                    break

                # Adjusted sleep time to target ~1 million works/day
                # Target rate: 1,000,000 works / 86,400 seconds/day = ~11.57 works/second.
                # Batch size is 200 works.
                # Desired time per batch = 200 works / 11.57 works/second = ~17.28 seconds.
                # Observed current time per batch (API call, processing, 0.1s sleep) ~6 seconds.
                # Estimated processing time (excluding old sleep) = 6s - 0.1s = 5.9s.
                # New sleep duration = 17.28s (desired total) - 5.9s (processing) = 11.38 seconds.
                time.sleep(11.38)

                attempts = 0

            except requests.exceptions.Timeout as e:
                attempts += 1
                print(f"Request timed out: {e}. Retrying (attempt {attempts}/{max_retries})...")
                if attempts >= max_retries:
                    print(f"Failed after {max_retries} retries due to timeout.")
                    break
                time.sleep(attempts * 5)
            except requests.exceptions.RequestException as e:
                attempts += 1
                print(f"Error fetching data: {e}. Retrying (attempt {attempts}/{max_retries})...")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response status: {e.response.status_code}, content: {e.response.text[:500]}")
                    if e.response.status_code == 429:
                        print("Rate limit hit by API (429 Too Many Requests). Waiting 60 seconds...")
                        time.sleep(60)
                        attempts -= 1
                        continue
                if attempts >= max_retries:
                    print(f"Failed after {max_retries} retries.")
                    break
                time.sleep(attempts * 3)

        print(f"Fetching loop ended. Performing final save of {len(all_fetched_works)} works.")
        try:
            with open(output_filename, "w", encoding="utf-8") as f_data_out:
                json.dump(all_fetched_works, f_data_out, ensure_ascii=False, indent=2)
            with open(state_filename, "w", encoding="utf-8") as f_state_out:
                json.dump({"next_cursor": current_cursor}, f_state_out, indent=2)
            print(
                f"Final data saved to {output_filename}. Final state (next_cursor: {current_cursor}) saved to {state_filename}.")
        except Exception as e:
            print(f"Error during final save: {e}")

        if attempts >= max_retries and current_cursor:
            print(
                f"Stopped fetching due to repeated errors after {len(all_fetched_works)} works. Last attempted cursor: {current_cursor}")
        elif not current_cursor:
            print(f"Successfully fetched all {len(all_fetched_works)} works.")
        else:
            print(f"Fetching ended. Total works: {len(all_fetched_works)}. Last attempted cursor: {current_cursor}")

        return all_fetched_works


def load_search_terms(path: Path) -> List[str]:
    """Read the curated search-term list, one term per line.

    Blank lines and ``#`` comments are ignored. The manuscript's list holds 279
    terms; the file is not redistributed with the repository, so a missing file
    is reported rather than silently substituted.
    """
    if not path.is_file():
        raise SystemExit(
            f"Search-term list not found: {path}\n"
            "This file is the output of stage 01 (keyword construction): the 279\n"
            "curated terms, one per line. Place it there, or point QSS_SEARCH_TERMS\n"
            "at it. See code/01_keyword_construction/README.md."
        )
    with open(path, encoding="utf-8") as handle:
        terms = [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not terms:
        raise SystemExit(f"Search-term list is empty: {path}")
    return terms


def chunk(items: List[str], size: int) -> List[List[str]]:
    """Split a list into consecutive groups of at most `size` items."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def main():
    # Credentials come from a local .env file (copy .env.example) or from the
    # environment. Both are optional: OpenAlex serves anonymous requests, but a
    # mailto puts them in the faster "polite pool".
    load_dotenv(PROJECT_ROOT / ".env")
    email = os.environ.get("OPENALEX_MAILTO")
    api_key = os.environ.get("OPENALEX_API_KEY")
    if not email:
        print(
            "OPENALEX_MAILTO is not set - requests will not use the polite pool "
            "and may be rate-limited more aggressively."
        )

    client = OpenAlexClient(email=email, api_key=api_key)

    search_terms = load_search_terms(SEARCH_TERMS_FILE)
    term_groups = chunk(search_terms, TERMS_PER_QUERY)

    print("Searching for works (newest first) that contain any of the search terms in title or abstract:")
    print("(Limited to publications from 2020-2024)")
    print(f"Using {len(search_terms)} search terms from {SEARCH_TERMS_FILE}")
    print(f"Split into {len(term_groups)} OR-queries of up to {TERMS_PER_QUERY} terms each,")
    print("because OpenAlex limits the length of a single search expression.")
    print("Downloads will be paced to target approximately 1 million works per 24 hours.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    save_freq = 100000

    # One retrieval per term group, each independently resumable via its own
    # state file, then merged on the OpenAlex work id.
    merged: Dict[str, Dict[str, Any]] = {}
    for group_index, terms in enumerate(term_groups):
        output_file = RAW_DIR / f"works_group_{group_index:03d}.json"
        state_file = RAW_DIR / f"state_group_{group_index:03d}.json"

        total_count_estimate = client.count_works_by_terms(terms)
        print(
            f"\n[group {group_index + 1}/{len(term_groups)}] {len(terms)} terms - "
            f"estimated matches (2020-2024): {total_count_estimate}"
        )

        if total_count_estimate == 0 and not (
            output_file.exists() and output_file.stat().st_size > 0
        ):
            print("No works estimated and no prior data for this group. Skipping.")
            continue

        group_results = client.get_all_works(
            terms,
            output_filename=str(output_file),
            state_filename=str(state_file),
            save_interval=save_freq,
        )

        for work in group_results:
            work_id = work.get("id")
            if work_id:
                merged[work_id] = work

        print(f"[group {group_index + 1}/{len(term_groups)}] cumulative unique works: {len(merged):,}")

    if not merged:
        print("No results were fetched or an error prevented fetching/loading.")
        return

    merged_path = INTERIM_DIR / "openalex_ai_works_2020-2024_raw.json"
    print(f"\nWriting {len(merged):,} deduplicated works to {merged_path} ...")
    with open(merged_path, "w", encoding="utf-8") as handle:
        json.dump(list(merged.values()), handle, ensure_ascii=False, indent=2)
    print(f"Process finished. Total unique works: {len(merged):,}.")


if __name__ == "__main__":
    main()
