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

## Procedure

1. **Seed taxonomy.** An initial 24-subdiscipline taxonomy: the 15 principal
   subfields of Gargiulo et al. (2022), plus nine that gained prominence after
   2020 and are therefore absent from that source.
2. **Survey corpus.** For each subdiscipline, survey papers from 2019 onward
   were collected and cleaned of boilerplate — 125 papers in total.
3. **Candidate extraction.** `keyBERT_keyword_extraction.py` (below) runs a
   multi-n-gram KeyBERT pass over each paper, yielding 1,784 unique candidates
   after deduplication.
4. **Manual curation.** Candidates were filtered against three criteria —
   domain exclusivity (does the term appear mainly in AI work?),
   non-redundancy by subsumption (drop a term already covered by a shorter one
   it contains), and a minimum of ten retrievable works — leaving **279 final
   search terms**.

Steps 1, 2 and 4 are human judgement and leave no code. Step 3 is this script.

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

**Run it**

```bash
python code/01_keyword_construction/keyBERT_keyword_extraction.py
```

**Input** — one plain-text file per survey paper in
`data/interim/ai_discipline_surveys_txt/`, or wherever `QSS_SURVEY_TXT_DIR`
points. The survey PDFs are third-party copyrighted material and are **not
redistributed with this repository**; the folder must be assembled locally.

**Output** — `output/keywords_mixed_<timestamp>.csv`, one row per input file:

| Column | Meaning |
|---|---|
| `file` | source filename |
| `keywords` | list of extracted phrases, all four n-gram passes concatenated |
| `time_sec` | extraction wall time |
| `text_length` | characters before truncation |

## What is not here

The 279-term list itself. It is the product of step 4, manual curation, and is
**not currently in this repository** — stage 02 expects it at
`code/01_keyword_construction/search_terms.txt` (one term per line, `#` for
comments) and fails with an explicit message if it is absent. Adding that file
is required before the retrieval stage can be re-run.
