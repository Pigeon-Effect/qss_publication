# 01 — Keyword construction

Builds the 279-term search vocabulary that stage 02 uses to query OpenAlex
(manuscript §2.2, *Dataset Extraction Pipeline*).

## Why this stage exists

A naive query for the phrase *artificial intelligence* retrieves only about
1,077,000 OpenAlex publications for 2020–2025 — far below what the scale of the
field implies. The reason is that most AI papers never self-label: they address
a specific method or application (*graph convolutional network*, *vision
transformer*, *federated learning*) and assume the reader supplies the umbrella
term. Recall therefore depends on a vocabulary broad enough to catch those
papers, and precise enough not to drag in unrelated work. That vocabulary is
what this stage produces.

The construction is deliberately corpus-driven rather than introspective. Terms
are extracted from the literature that AI researchers themselves write when
surveying their subfields, so the vocabulary reflects usage rather than the
authors' assumptions about it.

## The pipeline

```
survey papers (PDF) ──pdf_to_text.py──▶ plain text ──keyBERT_keyword_extraction.py──▶ keyphrases
                                                                                          │
                                                          deduplicate_keywords.py ◀───────┘
                                                                     │
                                            1,784 candidates ──manual curation──▶ 279 search terms
                                                                     ▲
                                                         term_hit_counts.py supplies
                                                         the hit counts curation needs
```

1. **Seed taxonomy.** An initial 24-subdiscipline taxonomy: the 15 principal
   subfields of Gargiulo et al. (2022), plus nine that gained prominence after
   2020 and are therefore absent from that source.
2. **Survey corpus.** For each subdiscipline, survey papers published from 2019
   onward were collected — 120 in total. Survey papers are used because they set
   out to summarise a subfield and therefore name the full range of its methods
   and research fronts in language meant for outsiders.
3. **Text extraction.** `pdf_to_text.py` converts each PDF to plain text and
   strips the reference list and page furniture.
4. **Candidate extraction.** `keyBERT_keyword_extraction.py` runs a
   multi-n-gram KeyBERT pass over each paper.
5. **Deduplication.** `deduplicate_keywords.py` flattens the per-paper results
   into **1,784 unique candidates**.
6. **Manual curation.** Candidates were filtered against three criteria —
   domain exclusivity (does the term appear mainly in AI work?),
   non-redundancy by subsumption (drop a term already covered by a shorter one
   it contains), and a minimum of ten retrievable works — leaving
   **279 final search terms**.

Steps 1, 2 and 6 are human judgement; steps 3–5 are the scripts below, and
`term_hit_counts.py` provides the evidence for the third curation criterion.

## `pdf_to_text.py`

Extracts the body text of each survey paper: pages in reading order, lines that
are nothing but a number dropped, blank runs collapsed, and everything from a
*References* or *Bibliography* heading onwards cut — a reference list is a dense
source of terms the paper itself never uses, and would distort the extraction.
Cover pages and other publisher boilerplate were removed by hand where these
rules do not catch them.

```bash
python code/01_keyword_construction/pdf_to_text.py
```

**Input** — one PDF per survey paper in `QSS_SURVEY_PDF_DIR`
(default `data/interim/ai_discipline_surveys_pdf/`). The survey papers are
third-party publications and are not redistributed here; assemble the folder
locally.
**Output** — one `.txt` per paper in `QSS_SURVEY_TXT_DIR`
(default `data/interim/ai_discipline_surveys_txt/`).

## `keyBERT_keyword_extraction.py`

Extracts keyphrases at four n-gram lengths from every `.txt` in the survey
folder, using KeyBERT over the `all-MiniLM-L6-v2` Sentence-BERT model.

| n-gram | top_n | MMR | Rationale |
|---|---:|---|---|
| 1 word | 5 | off | broad field markers (*segmentation*, *transformer*) |
| 2 words | 10 | off | the bulk of usable method names |
| 3 words | 5 | on (diversity 0.3) | longer method names; MMR stops the top-*n* collapsing into near-duplicate phrasings of one concept |
| 4 words | 5 | on (diversity 0.3) | same |

Papers shorter than 200 characters are rejected as extraction failures; papers
longer than 150,000 characters are truncated, since KeyBERT cost grows with
document length and the front matter of a survey already carries its
terminology. A paper that fails is recorded as an `ERROR:` row rather than
dropped, so the run's success rate is auditable.

```bash
python code/01_keyword_construction/keyBERT_keyword_extraction.py
```

**Output** — `output/keywords_mixed_<timestamp>.csv`, one row per input file:

| Column | Meaning |
|---|---|
| `file` | source filename |
| `keywords` | list of extracted phrases, all four n-gram passes concatenated |
| `time_sec` | extraction wall time |
| `text_length` | characters before truncation |

## `deduplicate_keywords.py`

The same phrase is extracted from many papers, so the per-paper rows are
flattened into one sorted list of distinct candidates — the pool curation works
from. Rows recording an extraction failure are skipped rather than parsed.

```bash
python code/01_keyword_construction/deduplicate_keywords.py
```

**Input** — the newest `output/keywords_mixed_*.csv` (override with `--input`).
**Output** — `output/unique_keywords.csv`, one candidate per row.

## `term_hit_counts.py`

Queries OpenAlex once per term, quoted and on its own, over the same 2020–2024
window stage 02 retrieves, and records how many works each term matches. This is
what the "at least ten retrievable works" criterion is checked against: run it
over the candidate list while curating, and over the final vocabulary to
document it. The chart it draws over the 279 terms is
[`logarithmic_bar_graph_seach_term_document_count.pdf`](../../paper/figures/logarithmic_bar_graph_seach_term_document_count.pdf).

```bash
# count the final vocabulary and draw the chart
python code/01_keyword_construction/term_hit_counts.py

# count a candidate list instead
python code/01_keyword_construction/term_hit_counts.py --terms-file output/unique_keywords.csv

# redraw from the cached counts, without calling the API
python code/01_keyword_construction/term_hit_counts.py --from-csv
```

Counts are cached in `output/term_hit_counts.csv`; the chart is
`output/term_hit_counts.svg`. Credentials are optional and read from the
environment — see [`.env.example`](../../.env.example).

## `search_terms.txt`

The product of step 6 and the input stage 02 reads: the **279 curated search
terms**, one per line, `#` for comments. They are listed in the order of the
published chart, i.e. by number of matching works, descending; order has no
effect on retrieval, since the terms are combined with Boolean OR, and matching
is case-insensitive.

Stage 02 finds the file automatically; `QSS_SEARCH_TERMS` overrides its location.
