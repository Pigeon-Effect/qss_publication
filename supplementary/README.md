# Supplementary Information

The complete cluster-level results that the article summarises.

The manuscript reports bloc citation shares at the macro level in full and,
below that, only for clusters whose share diverges from their parent — printing
all 106 research fronts in a two-column article would crowd out the argument.
This directory holds the unabridged version, both as a typeset document and as
CSV for reuse.

| File | Contents |
|---|---|
| `supplementary_information.pdf` | the document as submitted alongside the article |
| `supplementary_information.tex` | its source; `\input`s `tables.tex` |
| `tables.tex` | the three tables as LaTeX, generated |
| `citation_shares_h1.csv` | 5 domains |
| `citation_shares_h2.csv` | 31 fields |
| `citation_shares_h3.csv` | 106 research fronts |

## Reading the tables

Every row is one cluster. The four bloc columns — China, US, EU-27, RoW — are
that cluster's share of its own fractional citations, so **they sum to 100 %
across a row**. The `corpus_share` column reads the other way: it is the
cluster's share of all fractional citations in the corpus, and sums to 100 %
down each table.

Meso- and macro-level figures are aggregations of the micro-level citation
*sums*, not averages of micro-level shares. A cluster's share is therefore
always its own citations over its own total, at whatever level it is read.

Bloc membership is explicit: China is `CN + HK + MO + TW`, the EU-27 is the 27
member states, RoW is every remaining country. The definitions live as named
constants in `code/05_impact_analysis/fractional_citation_share.py` rather than
buried in a query, so they can be re-cut without guesswork — the per-country
companion CSV in `data/interim/` exists for exactly that.

## Regenerating

```bash
python code/05_impact_analysis/supplementary_tables.py
cd supplementary && latexmk -pdf supplementary_information.tex
```

The script reads `data/interim/citshare_h3x4entities.csv`, the micro-level
output of [stage 05](../code/05_impact_analysis/), which is deposited with this
repository. Override its location with `QSS_CITSHARE_CSV`.

Section 2 of the document — the document-intrusion validation summary — is
written by hand in `supplementary_information.tex` and transcribed from the run
manifests in [`results/intrusion/`](../results/README.md).
