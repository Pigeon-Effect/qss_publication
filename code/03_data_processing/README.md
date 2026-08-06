# 03 — Data processing

Turns the raw retrieval into the analysis corpus (manuscript §2.3, *Quality
Control and Postprocessing*). Corresponds to
[`initial_dataset_completeness_clean.pdf`](../../paper/figures/initial_dataset_completeness_clean.pdf)
and
[`final_dataset_completeness_clean.pdf`](../../paper/figures/final_dataset_completeness_clean.pdf).

## Why this stage exists

Every record must satisfy three requirements, each tied to something the study
actually does:

- **An abstract**, because subdiscipline identification is built on the text.
- **A country of origin**, because the whole comparison is between blocs.
- **At least one impact indicator**, which is automatic — every OpenAlex record
  carries a citation count.

OpenAlex delivers none of the first two in ready form. Abstracts arrive as an
inverted index, not as text. Country of origin does not exist as a field at
all. The three scripts here supply what is missing and then drop what still
falls short.

A side effect worth stating: because metadata completeness is strongly
correlated across fields, filtering on abstract and country also improves
everything else. Outgoing-reference completeness rises from 64.3 % to 92.2 %
without being filtered on directly.

**Run them in order.** Each reads the previous one's output.

## 1. `add_cleaned_abstract_column.py`

OpenAlex distributes abstracts as an `abstract_inverted_index` — a
`{word: [positions]}` map, a legacy of publishers licensing indices rather than
running text. The script inverts it: allocate a list as long as the highest
position, place each word at each of its positions, join. Only English records
with a non-empty index are processed; everything else gets `NULL`. Work is
spread across eight threads, since the operation is per-record and independent.

**In** `data/interim/openalex_ai_works_merged_deduplicated.db` (table `works`)
**Out** `data/interim/openalex_ai_works_merged_deduplicated_cleaned.db` — same
table plus `cleaned_abstract`

## 2. `add_country_of_origin_column.py`

Builds country attribution from author institutions. For each work, every
author's institution `country_code` is collected; authors without an
institution fall back to their `countries` field. The resulting codes are
counted and normalised into **country–share tuples summing to one**:

```python
[("US", 0.75), ("DE", 0.25)]   # 3 of 4 institution slots US, 1 DE
```

Fractional shares, not a single "main country", are what make the fractional
citation sum in stage 05 possible: a paper's citations can then be split among
the countries that produced it instead of being assigned wholesale to one.

Attribution is **institution-based only**. Inferring a country from language,
funder or publisher would introduce biases of unknown size and direction —
systematically favouring English-language and Western-published work — so
records with no institutional country are left empty and dropped by step 3
instead. This is why country of origin is available for only 64.3 % of raw
records.

**In** `data/interim/openalex_ai_works_merged_deduplicated.db`
**Out** `data/interim/openalex_ai_works_merged_deduplicated_country_with_origin.db`
— same table plus `country_of_origin` (JSON text)

## 3. `remove_abstract_OR_origin_NULLs.py`

The filter proper. Copies the `works` schema into a new database and inserts
only rows that have a non-empty `country_of_origin`, a non-`NULL`
`cleaned_abstract`, and an abstract of **100 to 1000 whitespace tokens**.

The length window is a formatting-error filter, not a content judgement.
Below 100 tokens the "abstract" is typically a copyright line, a section
heading or a truncated fragment; above 1000 it is usually a full-text dump that
escaped into the abstract field. Applying it removed **107,515 records**.

Language filtering happens implicitly: step 1 only reconstructs abstracts for
`language = 'en'`, so non-English records arrive here with a `NULL`
`cleaned_abstract` and are dropped. 95.36 % of the corpus was already English —
even non-English abstracts use English AI terminology extensively — and since
~98.89 % of outgoing OpenAlex references point at English-language documents,
restricting to English sacrifices roughly 1.11 % of citations.

**In** `data/interim/openalex_ai_works_merged_deduplicated_cleaned.db`
**Out** `data/interim/openalex_ai_works_merged_deduplicated_cleaned_filtered.db`

## Result

**1,986,659 publications** — 81.4 % articles, 11.1 % preprints, 3.7 % reviews,
1.7 % peer reviews, 1.6 % book chapters, 0.5 % datasets, 0.2 % dissertations.
This is the corpus that stage 04 clusters and that, once labelled, becomes
`data/merged_works_labeled.db` (see [`data/README.md`](../../data/README.md)).

## Caveats in the code as archived

- The two column-adding scripts both read
  `openalex_ai_works_merged_deduplicated.db` and write **separate** output
  databases; step 3 then reads only the `_cleaned` one. Merging the
  `country_of_origin` column back in is not scripted here.
- Steps 1 and 2 load the entire table into a pandas DataFrame. That is
  workable at 100k-row sample scale and memory-hungry at full corpus scale.
