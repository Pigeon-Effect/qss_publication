"""Convert the collected survey papers from PDF to plain text (manuscript section 2.2).

This is the first step of stage 01. KeyBERT reads running prose, so each survey
PDF is reduced to the substantive body text of the paper:

* text is extracted page by page in reading order,
* lines that consist of nothing but a number are dropped (line and page
  numbering used by some publishers),
* runs of blank lines are collapsed,
* everything from a "References" or "Bibliography" heading onwards is cut,
  because a reference list is a dense source of terms the paper never uses
  itself and would distort the extraction.

Cover pages and other publisher boilerplate were removed by hand where the
automatic rules do not catch them.

The survey papers themselves are third-party publications and are not
redistributed with this repository; assemble the input folder locally.

Input  - one PDF per survey paper in ``QSS_SURVEY_PDF_DIR``
Output - one .txt per PDF in ``QSS_SURVEY_TXT_DIR``, read by
         ``keyBERT_keyword_extraction.py``
"""

import argparse
import os
import re
from pathlib import Path

import fitz  # PyMuPDF

# ── Paths ─────────────────────────────────────────────────────────────────────
# Derived from this file's location, so the script runs from any checkout.
# .../code/01_keyword_construction/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = Path(os.environ.get("QSS_INTERIM_DIR", PROJECT_ROOT / "data" / "interim"))

PDF_DIR = Path(os.environ.get("QSS_SURVEY_PDF_DIR", INTERIM_DIR / "ai_discipline_surveys_pdf"))
TXT_DIR = Path(os.environ.get("QSS_SURVEY_TXT_DIR", INTERIM_DIR / "ai_discipline_surveys_txt"))

# A heading that starts the reference list. Matched on its own line so that an
# in-text mention of "references" does not truncate the paper.
REFERENCES_HEADING = re.compile(r"\n\s*(references|bibliography)\s*\n", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Strip line numbering, collapse blank runs and cut the reference list."""
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    match = REFERENCES_HEADING.search(text)
    if match:
        text = text[: match.start()]
    return text.strip()


def extract_pdf_text(pdf_path: Path) -> str:
    """Return the cleaned body text of one PDF."""
    with fitz.open(pdf_path) as doc:
        pages = [page.get_text() for page in doc]
    return clean_text("\n".join(pages) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR, help="folder holding the survey PDFs")
    parser.add_argument("--txt-dir", type=Path, default=TXT_DIR, help="folder to write the .txt files to")
    parser.add_argument("--overwrite", action="store_true", help="re-convert files that already exist")
    args = parser.parse_args()

    if not args.pdf_dir.is_dir():
        raise SystemExit(
            f"Survey-paper PDF folder not found: {args.pdf_dir}\n"
            "Point QSS_SURVEY_PDF_DIR at it, or see this stage's README."
        )
    args.txt_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(p for p in args.pdf_dir.iterdir() if p.suffix.lower() == ".pdf")
    if not pdfs:
        raise SystemExit(f"No PDFs found in {args.pdf_dir}")

    converted = skipped = failed = 0
    for pdf in pdfs:
        target = args.txt_dir / (pdf.stem + ".txt")
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            text = extract_pdf_text(pdf)
        except Exception as exc:  # a single unreadable PDF must not stop the run
            print(f"[skip] {pdf.name}: {exc}")
            failed += 1
            continue
        target.write_text(text, encoding="utf-8")
        converted += 1
        print(f"[ok]   {pdf.name} -> {target.name} ({len(text):,} characters)")

    print(
        f"\nConverted {converted} of {len(pdfs)} PDFs "
        f"({skipped} already present, {failed} unreadable)."
        f"\nText written to: {args.txt_dir}"
    )


if __name__ == "__main__":
    main()
