"""Reduce the raw KeyBERT output to a list of unique candidate terms (manuscript section 2.2).

``keyBERT_keyword_extraction.py`` writes one row per survey paper, each holding
that paper's extracted keyphrases across the four n-gram passes. The same
phrase is found in many papers, so this step flattens those rows into a single
sorted list of distinct candidates - the pool that manual curation then reduces
to the 279 search terms.

Rows whose ``keywords`` cell records an extraction failure (they start with
"ERROR:") are skipped rather than parsed, so a failed paper contributes
nothing instead of contributing a malformed entry.

Input  - ``output/keywords_mixed_<timestamp>.csv`` (newest by default)
Output - ``output/unique_keywords.csv``, one candidate term per row
"""

import argparse
import ast
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def newest_keyword_csv(folder: Path) -> Path:
    """Return the most recent KeyBERT output in ``folder``."""
    candidates = sorted(folder.glob("keywords_mixed_*.csv"))
    if not candidates:
        raise SystemExit(
            f"No keywords_mixed_*.csv found in {folder}\n"
            "Run keyBERT_keyword_extraction.py first, or pass --input."
        )
    return candidates[-1]


def collect_keywords(frame: pd.DataFrame) -> list[str]:
    """Flatten the per-paper keyword lists into one list of phrases."""
    phrases: list[str] = []
    for cell in frame["keywords"]:
        text = str(cell)
        if text.startswith("ERROR"):
            continue
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            continue
        phrases.extend(str(p) for p in parsed)
    return phrases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=None, help="KeyBERT CSV (default: newest in output/)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "unique_keywords.csv")
    args = parser.parse_args()

    source = args.input or newest_keyword_csv(OUTPUT_DIR)
    frame = pd.read_csv(source)

    phrases = collect_keywords(frame)
    unique = sorted(set(phrases))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"keyword": unique}).to_csv(args.output, index=False)

    failed = sum(1 for cell in frame["keywords"] if str(cell).startswith("ERROR"))
    print(f"Read {len(frame)} rows from {source.name} ({failed} recorded an extraction failure)")
    print(f"Collected {len(phrases):,} keyphrases, {len(unique):,} of them distinct")
    print(f"Written to: {args.output}")


if __name__ == "__main__":
    main()
