r"""Build the Supplementary Information tables from the stage 05 citation shares.

The manuscript reports bloc citation shares at the macro level in full and, below
that, only for clusters whose share diverges from their parent. The complete
tables for all three hierarchy levels live in the Supplementary Information, and
this script is what writes them.

Input is the micro-level CSV written by ``fractional_citation_share.py``. The
meso and macro tables are aggregations of it -- sums of fractional citations, not
means of the micro shares -- so a cluster's share is always its own citations
over its own total, at whatever level it is read.

Outputs (all under ``supplementary/``):

    citation_shares_h1.csv   5 domains
    citation_shares_h2.csv   31 fields
    citation_shares_h3.csv   106 research fronts
    tables.tex               the three tables as LaTeX, \input by
                             supplementary_information.tex

Run it:

    python code/05_impact_analysis/supplementary_tables.py
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTERIM = Path(os.environ.get("QSS_INTERIM_DIR", ROOT / "data" / "interim"))
CITSHARE_CSV = Path(os.environ.get("QSS_CITSHARE_CSV", INTERIM / "citshare_h3x4entities.csv"))
OUT_DIR = ROOT / "supplementary"

BLOCS = ("cn", "us", "eu27", "row")
BLOC_LABELS = {"cn": "China", "us": "US", "eu27": "EU-27", "row": "RoW"}


def load_micro(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def aggregate(rows: list[dict], key) -> list[dict]:
    """Sum fractional citations into the groups defined by ``key``, then re-share."""
    groups: dict[tuple, dict] = {}
    for row in rows:
        k = key(row)
        g = groups.setdefault(k, {"total": 0.0, **{b: 0.0 for b in BLOCS}})
        g["total"] += float(row["total_citations"])
        for b in BLOCS:
            g[b] += float(row[f"{b}_cits"])
    corpus_total = sum(g["total"] for g in groups.values())

    out = []
    for k in sorted(groups):
        g = groups[k]
        record = {"code": k[0], "cluster": k[-1], "total_citations": g["total"],
                  "corpus_share": g["total"] / corpus_total}
        for b in BLOCS:
            record[f"{b}_cits"] = g[b]
            record[f"{b}_share"] = g[b] / g["total"] if g["total"] else 0.0
        out.append(record)
    return out


def write_csv(records: list[dict], path: Path) -> None:
    fields = (["code", "cluster", "total_citations", "corpus_share"]
              + [f"{b}_cits" for b in BLOCS] + [f"{b}_share" for b in BLOCS])
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def tex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("%", r"\%")


def tex_table(records: list[dict], caption: str, label: str) -> str:
    head = [
        r"\begin{longtable}{@{}l l r r r r r r@{}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        r"\textbf{Code} & \textbf{Cluster} & \textbf{Frac.\ citations} & \textbf{\% corpus}"
        r" & \textbf{China} & \textbf{US} & \textbf{EU-27} & \textbf{RoW} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Code} & \textbf{Cluster} & \textbf{Frac.\ citations} & \textbf{\% corpus}"
        r" & \textbf{China} & \textbf{US} & \textbf{EU-27} & \textbf{RoW} \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    body = []
    for r in records:
        cells = [
            rf"\texttt{{{r['code']}}}",
            tex_escape(r["cluster"]),
            f"{r['total_citations']:,.0f}",
            f"{100 * r['corpus_share']:.2f}",
            *[f"{100 * r[f'{b}_share']:.2f}" for b in BLOCS],
        ]
        body.append(" & ".join(cells) + r" \\")
    return "\n".join(head + body + [r"\end{longtable}", ""])


def main() -> None:
    if not CITSHARE_CSV.exists():
        raise SystemExit(
            f"{CITSHARE_CSV} not found. Run fractional_citation_share.py first, "
            "or point QSS_CITSHARE_CSV at the CSV."
        )
    OUT_DIR.mkdir(exist_ok=True)
    rows = load_micro(CITSHARE_CSV)

    levels = [
        ("h1", lambda r: (r["macro_id"], r["macro"]), "S1",
         "Fractional citation shares by bloc across the 5 macro-clusters (domains)"),
        ("h2", lambda r: (r["code"][:2], r["meso"]), "S2",
         "Fractional citation shares by bloc across the 31 meso-clusters (fields)"),
        ("h3", lambda r: (r["code"], r["micro"]), "S3",
         "Fractional citation shares by bloc across the 106 micro-clusters (research fronts)"),
    ]

    tex_parts = []
    for level, key, label, caption in levels:
        records = aggregate(rows, key)
        write_csv(records, OUT_DIR / f"citation_shares_{level}.csv")
        tex_parts.append(tex_table(records, caption, f"tab:{label.lower()}"))
        print(f"{level}: {len(records)} clusters -> citation_shares_{level}.csv")

    (OUT_DIR / "tables.tex").write_text("\n".join(tex_parts), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'tables.tex'}")


if __name__ == "__main__":
    main()
