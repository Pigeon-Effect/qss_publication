"""Count how many OpenAlex works each search term matches (manuscript section 2.2).

Manual curation kept a candidate term only if it was domain-exclusive,
not already covered by a shorter term it contains, and matched at least ten
retrievable works. This script supplies the evidence for that last criterion:
it queries OpenAlex once per term, in quotes and on its own, over the same
2020-2024 window stage 02 retrieves, and records the number of matching works.

Run it on the candidate pool while curating (``--terms-file`` pointing at the
candidate list), or on the final vocabulary to document it. The horizontal
log-scale chart it draws over the final 279 terms is
``paper/figures/logarithmic_bar_graph_seach_term_document_count.pdf``.

Counting 279 terms takes a few minutes and makes one request per term, so the
counts are cached in a CSV; ``--from-csv`` redraws the chart without calling
the API again.

Input  - ``search_terms.txt`` (or any one-term-per-line file)
Output - ``output/term_hit_counts.csv`` and ``output/term_hit_counts.svg``
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from matplotlib.ticker import FuncFormatter

# ── Paths ─────────────────────────────────────────────────────────────────────
# Derived from this file's location, so the script runs from any checkout.
# .../code/01_keyword_construction/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEARCH_TERMS_FILE = Path(
    os.environ.get("QSS_SEARCH_TERMS", Path(__file__).resolve().parent / "search_terms.txt")
)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

YEAR_RANGE = "2020-2024"


class OpenAlexClient:
    """Minimal OpenAlex client for counting matches of a single term."""

    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = "https://api.openalex.org"
        self.session = requests.Session()
        if email:
            self.session.headers.update({"User-Agent": f"PythonClient/1.0 (mailto:{email})"})
        self._email = email
        self._api_key = api_key

    def count_works(self, term: str, years: str = YEAR_RANGE) -> int:
        params: Dict[str, object] = {
            "search": f'"{term}"',
            "filter": f"publication_year:{years}",
            "per-page": 1,
            "select": "id",
        }
        if self._email:
            params["mailto"] = self._email
        if self._api_key:
            params["api_key"] = self._api_key
        response = self.session.get(f"{self.base_url}/works", params=params, timeout=60)
        response.raise_for_status()
        return response.json().get("meta", {}).get("count", 0)


def load_terms(path: Path) -> List[str]:
    """Read a term list: one per line, '#' starts a comment."""
    if not path.exists():
        raise SystemExit(f"Term list not found: {path}")
    with open(path, encoding="utf-8") as handle:
        terms = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
    if not terms:
        raise SystemExit(f"No terms found in {path}")
    return terms


def count_terms(client: OpenAlexClient, terms: List[str], years: str, pause: float) -> pd.DataFrame:
    rows = []
    for i, term in enumerate(terms, 1):
        count = client.count_works(term, years)
        rows.append({"search_term": term, "count": count})
        print(f"[{i:>3}/{len(terms)}] {term}: {count:,}")
        time.sleep(pause)
    return pd.DataFrame(rows)


def plot_counts(frame: pd.DataFrame, out_svg: Path, columns: int = 4) -> None:
    """Draw the terms as horizontal log-scale bars, split over several columns."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    })

    ordered = frame.sort_values("count", ascending=False)
    terms = [t.lower() for t in ordered["search_term"]]
    counts = list(ordered["count"])

    per_column = len(terms) // columns
    chunks = [
        (terms[i * per_column: (i + 1) * per_column], counts[i * per_column: (i + 1) * per_column])
        if i < columns - 1
        else (terms[i * per_column:], counts[i * per_column:])
        for i in range(columns)
    ]

    fig, axes = plt.subplots(nrows=1, ncols=columns, figsize=(8.5, 11), sharex=True)
    ticks = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
    for ax, (col_terms, col_counts) in zip(np.atleast_1d(axes), chunks):
        y = np.arange(len(col_terms))
        ax.barh(y, col_counts, align="center", height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(col_terms, fontsize=6)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(left=10)  # the ten-work curation threshold
        ax.grid(axis="x", color="grey", alpha=0.3, which="both")
        ax.margins(y=0.005)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"$10^{{{int(np.log10(v))}}}$"))

    plt.tight_layout()
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Chart written to: {out_svg}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--terms-file", type=Path, default=SEARCH_TERMS_FILE)
    parser.add_argument("--years", default=YEAR_RANGE, help="OpenAlex publication_year filter")
    parser.add_argument("--csv", type=Path, default=OUTPUT_DIR / "term_hit_counts.csv")
    parser.add_argument("--svg", type=Path, default=OUTPUT_DIR / "term_hit_counts.svg")
    parser.add_argument("--from-csv", action="store_true", help="redraw from the cached counts, no API calls")
    parser.add_argument("--pause", type=float, default=0.5, help="seconds between requests")
    parser.add_argument("--no-plot", action="store_true", help="only write the counts")
    args = parser.parse_args()

    if args.from_csv:
        if not args.csv.exists():
            raise SystemExit(f"No cached counts at {args.csv}; run without --from-csv first.")
        frame = pd.read_csv(args.csv)
        print(f"Loaded {len(frame)} cached counts from {args.csv}")
    else:
        load_dotenv(PROJECT_ROOT / ".env")
        client = OpenAlexClient(
            email=os.environ.get("OPENALEX_MAILTO"),
            api_key=os.environ.get("OPENALEX_API_KEY"),
        )
        terms = load_terms(args.terms_file)
        print(f"Counting {len(terms)} terms over publication_year:{args.years}\n")
        frame = count_terms(client, terms, args.years, args.pause)
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv, index=False)
        print(f"\nCounts written to: {args.csv}")

    below = frame[frame["count"] < 10]
    if len(below):
        print(f"{len(below)} term(s) match fewer than ten works: {', '.join(below['search_term'])}")

    if not args.no_plot:
        plot_counts(frame, args.svg)


if __name__ == "__main__":
    main()
