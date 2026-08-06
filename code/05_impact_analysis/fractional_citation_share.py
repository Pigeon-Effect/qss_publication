#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fractional citation sum per cluster and per bloc (manuscript section 3.2).

Computes each bloc's share of the citations accruing to every cluster of the
taxonomy built in stage 04, and writes the CSV that stage 06 draws Figure 2
from.

The metric
----------
A work's incoming citations are split among the countries that produced it, in
proportion to that work's `country_of_origin` shares - the country-share tuples
derived in stage 03 from author institutions, e.g. [("US", 0.75), ("DE", 0.25)].
A paper with 12 citations and those shares contributes 9.0 fractional citations
to the US and 3.0 to Germany. Summing over every work in a cluster gives that
cluster's fractional citation sum per country; grouping countries into blocs and
dividing by the cluster total gives the shares.

Two premises, both stated in the manuscript:

1. **Authors contributed equally.** Citations are split by institutional slot,
   not weighted by author position, seniority or declared contribution. Author
   order conventions differ across the disciplines this corpus spans - first
   author, last author and alphabetical are all in use - so any weighting would
   encode one field's convention as if it were universal. Equal splitting is
   the assumption that treats all fields alike.
2. **Institutions indicate research systems, better than personal origin
   would.** The question is which research system produced the work - its
   funding, infrastructure and training - not where its authors were born or
   hold citizenship. A researcher's affiliation is the observable that tracks
   that; nationality is neither recorded by OpenAlex nor the quantity of
   interest.

Note the consequence, which the manuscript records under *Limitations*: shares
within a cluster sum to 100 %, so one bloc's gain is mechanically another's
loss. That is a property of the measure, not a finding about AI competition.

Why this metric rather than a more sophisticated one is argued in the README.

Inputs
------
- the cluster-labelled corpus (`data/merged_works_labeled.db`, table
  `works_labeled`), for `h1_cluster`, `h2_cluster`, `h3_cluster`,
  `country_of_origin` and `cited_by_count`
- `taxonomy.csv` beside this script: the display names of the 5 domains, 31
  fields and 106 research fronts, transcribed from manuscript Figure 1

Outputs (into `data/interim/` by default)
----------------------------------------
- `citshare_h3x4entities.csv`  - one row per micro-cluster, the four blocs'
  citations and shares. This is what stage 06 reads.
- `fractional_citations_by_country_h3.csv` - the same sums before bloc
  aggregation, one row per (micro-cluster, country). Keeping it means the bloc
  definitions can be re-cut without another pass over two million rows.

Macro- and meso-level figures are not written to file: they are exact sums of
the micro rows, and stage 06 aggregates them itself. They are printed at the
end of a run because they are the numbers quoted in manuscript section 4.2.
"""

import argparse
import csv
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# Derived from this file's location, so the script runs from any checkout.
# .../code/05_impact_analysis/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = Path(os.environ.get("QSS_INTERIM_DIR", PROJECT_ROOT / "data" / "interim"))
DB_PATH = Path(
    os.environ.get("QSS_DB_PATH", PROJECT_ROOT / "data" / "merged_works_labeled.db")
)
TAXONOMY_PATH = Path(__file__).resolve().parent / "taxonomy.csv"

DEFAULT_SHARES_CSV = INTERIM_DIR / "citshare_h3x4entities.csv"
DEFAULT_COUNTRY_CSV = INTERIM_DIR / "fractional_citations_by_country_h3.csv"

# ── Bloc definitions ──────────────────────────────────────────────────────────
# ISO 3166-1 alpha-2, as OpenAlex reports institution country codes.
#
# "China" is the Greater China reading: the mainland plus Hong Kong (HK), Macao
# (MO) and Taiwan (TW), which OpenAlex reports as separate institution country
# codes. This is a choice, not a fact, and it is a consequential one - the four
# codes together carry 24.00 % of the corpus's fractional citation sum against
# the mainland's 21.82 %, and the manuscript reports 24.1 %. Kept as an explicit
# constant rather than buried in a query so it can be re-cut and re-checked.
CHINA = frozenset({"CN", "HK", "MO", "TW"})
USA = frozenset({"US"})

# The 27 member states of the European Union. The United Kingdom is not a
# member and is therefore RoW, consistent with the manuscript's "EU-27".
EU27 = frozenset({
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR",
    "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK",
    "SI", "ES", "SE",
})

# RoW is the residual: every country in no other bloc. It is reported because
# the shares have to sum to one to be readable, and because where RoW is
# smallest is where the three blocs are most consolidated.
BLOCS = (
    ("cn", CHINA),
    ("us", USA),
    ("eu27", EU27),
)

# How far the per-work country shares may deviate from summing to 1.0 before
# the run reports it. Stage 03 builds them as count/total, so the only expected
# deviation is floating-point noise.
SHARE_SUM_TOLERANCE = 1e-6


def load_taxonomy(path: Path) -> dict[str, dict[str, str]]:
    """Read the display names for the 106 micro-clusters, keyed by 3-digit code."""
    if not path.is_file():
        raise SystemExit(
            f"Taxonomy file not found: {path}\n"
            "It lists the display names of the 5 domains, 31 fields and 106 "
            "research fronts, and is required for the output CSV."
        )
    taxonomy: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            code = row["code"].strip()
            taxonomy[code] = {
                "macro_id": int(row["macro_id"]),
                "macro": row["macro"].strip(),
                "meso": row["meso"].strip(),
                "micro": row["micro"].strip(),
            }
    if not taxonomy:
        raise SystemExit(f"Taxonomy file is empty: {path}")
    return taxonomy


def accumulate(db_path: Path, progress_every: int) -> tuple[dict, dict]:
    """Stream the corpus and sum fractional citations per (cluster, country).

    Returns the accumulator and a dict of counters describing what was read,
    so the run can report what it skipped rather than skipping it silently.
    """
    if not db_path.is_file():
        raise SystemExit(
            f"Corpus not found: {db_path}\n"
            "See data/README.md for how to obtain it, or set QSS_DB_PATH."
        )

    # Streamed rather than loaded into a DataFrame: the corpus is ~2 million
    # rows in a ~20 GB database, and only five columns are needed.
    query = """
        SELECT h1_cluster, h2_cluster, h3_cluster, country_of_origin, cited_by_count
        FROM works_labeled
        WHERE h1_cluster IS NOT NULL
          AND h2_cluster IS NOT NULL
          AND h3_cluster IS NOT NULL
          AND country_of_origin IS NOT NULL
          AND country_of_origin <> '[]'
    """

    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    stats = {
        "rows_read": 0,
        "rows_counted": 0,
        "rows_zero_citations": 0,
        "rows_unparseable_country": 0,
        "rows_empty_country": 0,
        "max_share_sum_deviation": 0.0,
    }

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(query)
        while True:
            batch = cursor.fetchmany(50_000)
            if not batch:
                break
            for h1, h2, h3, country_json, cited_by in batch:
                stats["rows_read"] += 1
                if progress_every and stats["rows_read"] % progress_every == 0:
                    print(f"  ... {stats['rows_read']:,} rows", flush=True)

                citations = cited_by or 0
                if citations <= 0:
                    # An uncited work contributes exactly zero to every sum, so
                    # skipping it is equivalent to processing it - and avoids
                    # parsing JSON for a large share of the corpus.
                    stats["rows_zero_citations"] += 1
                    continue

                try:
                    shares = json.loads(country_json)
                except (TypeError, ValueError):
                    stats["rows_unparseable_country"] += 1
                    continue
                if not shares:
                    stats["rows_empty_country"] += 1
                    continue

                # The cluster identifier is the concatenation of the labels down
                # to the micro level, matching CLUSTER_ID_SQL in
                # src/clustervalidation/config.py.
                code = f"{h1}{h2}{h3}"

                bucket = totals[code]
                share_sum = 0.0
                for entry in shares:
                    country_code, share = entry[0], float(entry[1])
                    bucket[country_code] += share * citations
                    share_sum += share

                deviation = abs(share_sum - 1.0)
                if deviation > stats["max_share_sum_deviation"]:
                    stats["max_share_sum_deviation"] = deviation
                stats["rows_counted"] += 1
    finally:
        conn.close()

    return totals, stats


def to_bloc_rows(totals: dict, taxonomy: dict) -> list[dict]:
    """Aggregate per-country sums into the four blocs, one row per micro-cluster."""
    rows = []
    for code, by_country in totals.items():
        meta = taxonomy[code]
        total = sum(by_country.values())

        bloc_cits = {
            name: sum(value for cc, value in by_country.items() if cc in members)
            for name, members in BLOCS
        }
        # RoW is the residual, so the four values always sum to the total
        # exactly - no country can be counted twice or dropped.
        bloc_cits["row"] = total - sum(bloc_cits.values())

        row = {
            "macro_id": meta["macro_id"],
            "macro": meta["macro"],
            "meso": meta["meso"],
            "micro": meta["micro"],
            "code": code,
            "total_citations": total,
        }
        for name in ("cn", "us", "eu27", "row"):
            row[f"{name}_cits"] = bloc_cits[name]
        for name in ("cn", "us", "eu27", "row"):
            row[f"{name}_share"] = (bloc_cits[name] / total) if total > 0 else 0.0
        rows.append(row)

    # Sorted by macro then code, which is also what stage 06 needs: it groups
    # meso-clusters into contiguous blocks and relies on that ordering.
    rows.sort(key=lambda r: (r["macro_id"], int(r["code"])))
    return rows


def write_bloc_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "macro_id", "macro", "meso", "micro", "code", "total_citations",
        "cn_cits", "us_cits", "eu27_cits", "row_cits",
        "cn_share", "us_share", "eu27_share", "row_share",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_country_csv(totals: dict, taxonomy: dict, path: Path) -> None:
    """Long-format per-country sums, before bloc aggregation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["code", "macro_id", "macro", "meso", "micro",
                         "country", "fractional_citations"])
        for code in sorted(totals, key=lambda c: (taxonomy[c]["macro_id"], int(c))):
            meta = taxonomy[code]
            for country, value in sorted(
                totals[code].items(), key=lambda kv: -kv[1]
            ):
                writer.writerow([code, meta["macro_id"], meta["macro"],
                                 meta["meso"], meta["micro"], country, value])


def report(rows: list[dict], stats: dict) -> None:
    """Print the corpus- and macro-level figures quoted in section 4.2."""
    grand_total = sum(r["total_citations"] for r in rows)
    print("\n" + "=" * 72)
    print(f"Rows read                  {stats['rows_read']:>15,}")
    print(f"  counted                  {stats['rows_counted']:>15,}")
    print(f"  skipped, zero citations  {stats['rows_zero_citations']:>15,}")
    if stats["rows_empty_country"]:
        print(f"  skipped, no country      {stats['rows_empty_country']:>15,}")
    if stats["rows_unparseable_country"]:
        print(f"  skipped, bad country JSON{stats['rows_unparseable_country']:>15,}")
    print(f"Micro-clusters                    {len(rows):>10,}")
    print(f"Fractional citation sum      {grand_total:>15,.1f}")
    print(f"Max deviation of per-work shares from 1.0: "
          f"{stats['max_share_sum_deviation']:.2e}")
    if stats["max_share_sum_deviation"] > SHARE_SUM_TOLERANCE:
        print("  ! country_of_origin shares do not sum to 1 for every work.")
        print("    Check stage 03 (add_country_of_origin_column.py).")

    print("\nCorpus-wide bloc shares of the fractional citation sum")
    print("-" * 72)
    for name, label in (("cn", "China"), ("us", "USA"),
                        ("eu27", "EU-27"), ("row", "RoW")):
        value = sum(r[f"{name}_cits"] for r in rows)
        pct = (value / grand_total * 100.0) if grand_total else 0.0
        print(f"  {label:<7} {pct:6.2f} %   ({value:,.1f})")

    print("\nBy macro-domain")
    print("-" * 72)
    print(f"{'domain':<20}{'of corpus':>11}{'China':>9}{'USA':>9}"
          f"{'EU-27':>9}{'RoW':>9}")
    macro_ids = sorted({r["macro_id"] for r in rows})
    for macro_id in macro_ids:
        sub = [r for r in rows if r["macro_id"] == macro_id]
        name = sub[0]["macro"]
        macro_total = sum(r["total_citations"] for r in sub)
        of_corpus = (macro_total / grand_total * 100.0) if grand_total else 0.0
        cells = []
        for bloc in ("cn", "us", "eu27", "row"):
            value = sum(r[f"{bloc}_cits"] for r in sub)
            cells.append((value / macro_total * 100.0) if macro_total else 0.0)
        print(f"{name:<20}{of_corpus:>10.2f}%" + "".join(f"{c:>8.2f}%" for c in cells))
    print("=" * 72)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Fractional citation sum per cluster and bloc."
    )
    parser.add_argument("--db", type=Path, default=DB_PATH,
                        help=f"labelled corpus (default: {DB_PATH})")
    parser.add_argument("--taxonomy", type=Path, default=TAXONOMY_PATH,
                        help="cluster display names (default: taxonomy.csv)")
    parser.add_argument("--out", type=Path, default=DEFAULT_SHARES_CSV,
                        help=f"bloc-share CSV (default: {DEFAULT_SHARES_CSV})")
    parser.add_argument("--out-countries", type=Path, default=DEFAULT_COUNTRY_CSV,
                        help="per-country CSV; pass an empty string to skip")
    parser.add_argument("--progress-every", type=int, default=250_000,
                        help="progress interval in rows; 0 to silence")
    args = parser.parse_args(argv)

    taxonomy = load_taxonomy(args.taxonomy)
    print(f"[i] taxonomy : {args.taxonomy}  ({len(taxonomy)} micro-clusters)")
    print(f"[i] corpus   : {args.db}")
    print("[i] summing fractional citations ...")

    totals, stats = accumulate(args.db, args.progress_every)

    # An identifier in one source but not the other means the taxonomy file and
    # the labelled corpus disagree. Every downstream number would inherit that,
    # so it is a hard error rather than a warning.
    in_db = set(totals)
    in_taxonomy = set(taxonomy)
    if in_db - in_taxonomy:
        raise SystemExit(
            "Corpus contains cluster codes absent from the taxonomy file: "
            f"{sorted(in_db - in_taxonomy)}"
        )
    if in_taxonomy - in_db:
        print(
            "[!] taxonomy lists clusters with no cited works in the corpus: "
            f"{sorted(in_taxonomy - in_db)}"
        )

    rows = to_bloc_rows(totals, taxonomy)

    write_bloc_csv(rows, args.out)
    print(f"\n[OK] {len(rows)} rows -> {args.out}")
    if str(args.out_countries):
        write_country_csv(totals, taxonomy, args.out_countries)
        n = sum(len(v) for v in totals.values())
        print(f"[OK] {n} rows -> {args.out_countries}")

    report(rows, stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
