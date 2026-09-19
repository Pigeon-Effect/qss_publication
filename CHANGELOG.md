# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **The search vocabulary is deposited.**
  `code/01_keyword_construction/search_terms.txt` holds the 279 curated terms,
  one per line, in the order of the published term-count chart. Stage 02 reads
  it directly, so retrieval runs from a clean checkout.
- **Stage 01 covers the whole keyword pipeline.** `pdf_to_text.py` converts the
  survey papers to the plain text KeyBERT reads, stripping reference lists and
  page furniture; `deduplicate_keywords.py` flattens the per-paper keyphrases
  into the candidate pool curation works from; `term_hit_counts.py` counts the
  works each term matches and draws the log-scale chart, which is the evidence
  for the "at least ten retrievable works" curation criterion.
- **Stage 02 retrieval matches how the corpus was built.**
  `download_openalex_works.py` retrieves one term batch straight into SQLite,
  resuming from a stored cursor and inserting idempotently on the work id;
  `merge_term_batches.py` merges the batch databases and deduplicates them into
  the database stage 03 reads. The README lists the six runs that produced the
  published corpus.
- **The semantic landscape (Figure 1) is reproducible.**
  `code/06_visualization/semantic_landscape/` holds the three steps —
  `project_sample.py` (seeded sample, SPECTER, UMAP), `summarize_clusters.py`
  (per-cluster Gaussian KDE, the contour enclosing 80 % of a cluster's points),
  `render_landscape.py` (the figure) — and, in `resources/`, the published
  layout as GeoJSON: all 106 outlines with their cluster codes, taxonomy names
  and label positions. The figure redraws from it without the corpus or a GPU,
  and the layout can be reused to shade the same map by any per-cluster
  quantity.
- **The corpus is deposited.** The labelled corpus is published as a separate
  Zenodo data record, [10.5281/zenodo.22791584](https://doi.org/10.5281/zenodo.22791584):
  1,986,659 works with their identifiers, citation counts, fractional country
  shares, abstracts and three-level cluster labels, as a gzip-compressed SQLite
  database of 1.2 GB (4.3 GB uncompressed). Every statement in the repository
  that the corpus "is deposited on Zenodo" was a forward reference until now.
  The twenty further columns retrieved from OpenAlex are not part of the
  deposit: no analysis here reads them, and most change as OpenAlex updates its
  records, so a snapshot would go stale. `data/published_dataset/` carries the
  deposit's README and its SHA-256 checksums; the database itself is not in git.
- `results/prompt_experiments/`: a paired comparison of the `subset` prompt
  framing against the `decisive` one the article used — 300 trials, same seed,
  same panels, same 8,000-token ceiling, 100 % panel alignment. `subset` scores
  9 pp lower at every level (36.0 / 63.0 / 76.0 % against 45.0 / 71.0 / 85.0 %
  on the same panels), McNemar pooled p = 6.9 × 10⁻⁵. It also reproduces, at
  90.3 %, the 90.2 % figure from an earlier token-starved pilot that had
  suggested the opposite, and shows it to be a selection artefact of
  conditioning on non-truncated responses rather than a prompt effect. Includes
  `compare.py`, which recomputes every published figure from the trial records.
  This closes the question in favour of the prompt already used; the reported
  40.5 / 64.7 / 77.8 % are unaffected.

### Changed
- `.env.example` documents the OpenAlex variables (`OPENALEX_MAILTO`,
  `OPENALEX_API_KEY`) alongside the DeepSeek key, and `pyproject.toml` gains a
  `pipeline` extra listing what the numbered stages need; the package itself
  still depends only on `openai` and `python-dotenv`.
- `code/06_visualization/README.md` now covers both manuscript figures.
- Stage 04's README and script headers describe the consolidation as the
  level-by-level judgement it is, and point to the deposited label set as the
  authoritative form of the taxonomy.
- `data/README.md`: schema table reduced to the ten deposited columns, with the
  dropped columns and how to fetch them from OpenAlex documented; corpus year
  range corrected from 2020–2025 to 2020–2024, which is what the database
  holds; file size corrected; unzip-and-place instructions added.
- `.zenodo.json`: the data record linked as `isSupplementedBy`, and the
  description's "20 GB labelled corpus" replaced with the deposited sizes.
- `CITATION.cff`: the dataset added as a `type: data` reference with its DOI.
- `README.md`: the corpus paragraph now names the data DOI and says why the
  compendium and the corpus are deposited as two records.
- `.gitignore`: databases are ignored wherever they sit in the tree. The
  published corpus is distributed through Zenodo, not GitHub, where its 1.2 GB
  would exceed the 100 MB file limit. Regenerable `output/` folders, embedding
  caches and local scratch material are ignored too, so a clone carries the
  deposit and nothing else.

### Removed
- `code/02_data_collection/download_full_dataset.py`, superseded by
  `download_openalex_works.py` and `merge_term_batches.py`, which retrieve and
  merge the corpus the way it was built.
- `h2_cluster_social_sciences/h2_umap_social_science.py`, a duplicate of the
  engineering countercheck that had been saved into the social-sciences folder;
  `h2_umap_engineering.py` is the file it copied.

## [2.0.0] — 2026-09-01

Second release, and the first complete one. Where `1.0.0` deposited the LLM
validation component alone, this release is the full research compendium behind
the article: the entire pipeline, the reported validation experiment at its
final scale, the cluster-level result tables, and a manuscript whose reported
figures match the deposited record exactly.

### Added
- Stages 01–06 of the pipeline in `code/`: keyword construction, OpenAlex
  retrieval, quality control, the semi-supervised hierarchical topic model
  (h1/h2/h3, plus the `finding_optimal_k` evidence and the parent-scoped
  contrastive TF-IDF term extraction), impact analysis, and the citation-share
  heatmap. Each stage carries a README documenting what it does, why the method
  is what it is, how to run it, and its inputs and outputs.
- `code/05_impact_analysis/fractional_citation_share.py`: the fractional
  citation sum per micro-cluster and bloc. Splits each work's citations across
  countries by its `country_of_origin` institution shares, aggregates into
  China / USA / EU-27 / RoW, and writes the CSV stage 06 reads plus an
  un-aggregated per-country companion so the bloc definitions can be re-cut
  without a second pass. Bloc membership is a set of named constants, not a
  buried query: "China" is `CN + HK + MO + TW`, which reproduces the
  manuscript's reported share where mainland-only would give 21.82 %. All five
  macro-domain citation totals in Figure 2 reproduce.
- `code/05_impact_analysis/taxonomy.csv`: the display names of the 5 domains,
  31 fields and 106 research fronts, keyed by 3-digit code. The database stores
  only integer codes.
- **The reported validation experiment at full scale.** Document intrusion at
  1,000 trials per hierarchy level — 3,000 trials in total, 0 failed, 0
  unparsed, 0 forced guesses — under `results/intrusion/`, each level with its
  manifest, per-trial records, readable transcript and run log.
- `decisive` prompt variant, which produced those runs. Written after the h3
  pilot showed that a long reasoning trace is a symptom of failure rather than
  care: wrong answers reasoned 4.1x longer than correct ones (mean 6,820 vs
  1,680 characters). Testing "what if paper k is the intruder" for every k
  re-reads the panel O(n^2) times where labelling each paper once is O(n); the
  variant forbids the re-reading and supplies an explicit tie-break.
- Forced-choice extraction: a response cut off by the token ceiling
  mid-reasoning is passed to a second, non-reasoning model that reads the
  truncated trace and reports which paper it was converging on. It never sees
  the panel, so it extracts rather than judges, and it never guesses — a trial
  it cannot resolve is recorded as unresolved rather than assigned an answer.
  `--force-choice-model`, `--max-tokens` and `--outage-wait` expose the
  surrounding controls.
- Outage handling: an unreachable API pauses the run and retries rather than
  losing it, with the wait recorded in the manifest and the notice written to
  the run's `.err` stream. All three reported runs took one such wait.
- `supplementary/`, the article's Supplementary Information: bloc citation
  shares for every cluster at all three hierarchy levels as a typeset PDF and as
  three CSVs, generated by `code/05_impact_analysis/supplementary_tables.py`.
- `tests/test_forced_verdict.py` and `tests/test_outage_retry.py`.
- `python-dotenv`: credentials are read from a local `.env` as well as from the
  environment, in both `clustervalidation` and the pipeline scripts. An exported
  variable still takes precedence.
- Environment-variable overrides for every path that legitimately varies
  (`QSS_INTERIM_DIR`, `QSS_DB_PATH`, `QSS_H3_DB`, `QSS_SEARCH_TERMS`,
  `QSS_SURVEY_TXT_DIR`, `QSS_CITSHARE_CSV`), and a `--db` flag on the h3
  clustering script.

### Changed
- **The manuscript now reports the deposited runs.** Section *External
  Validation via Document Intrusion* and the RQ1 conclusion previously gave
  46.0 % / 75.0 % / over 84.0 %, figures that no archived run produced. They now
  give 40.5 % / 64.7 % / 77.8 % at h1 / h2 / h3, transcribed from the
  1,000-trial manifests, with the 20 % random baseline stated alongside. The
  qualitative claim is unchanged: accuracy rises monotonically with granularity
  and clears chance at every level.
- The manuscript's *Data and Code* section describes this compendium and its
  Zenodo deposit; it previously pointed at a repository under a different name.
- `data/interim/citshare_h3x4entities.csv` and
  `fractional_citations_by_country_h3.csv` are now carried in git. They are
  small, and stage 06 and the Supplementary Information read them.
- All 67 hardcoded absolute paths across 32 scripts in `code/` replaced with
  paths derived from each file's own location, so a fresh clone runs anywhere.
- `download_full_dataset.py` no longer inlines a 48-term search list. It loads
  the 279 curated terms from `code/01_keyword_construction/search_terms.txt`
  and queries them in groups, because OpenAlex limits the length of a single
  `search` expression, merging and deduplicating on the OpenAlex work id. This
  matches the batched retrieval the manuscript describes; the archived script
  did not. OpenAlex credentials moved from `"xxx"` placeholders to
  `OPENALEX_MAILTO` / `OPENALEX_API_KEY`.
- `h2_labeling_biomedical.py`: meso 4 renamed *Immunology & Infectious Disease*
  → *Microbial and Immune Systems Biology*, meso 5 *Genetics, Genomics &
  Oncology* → *Genetics & Genomics*, matching Figure 1. Names only; no cluster
  assignment changed.
- `h2_labeling_social_science.py`: meso 5 *Urban Development & Tourism* →
  *Urban Development* (Tourism is micro-cluster 251 in the published taxonomy).
- Documentation rewritten to read as a finished deposit rather than a work
  plan: the top-level README opens with a plain-language account of what the
  study did and how the taxonomy was checked, and descends into protocol
  detail, reproducibility and command reference below that.
- `.gitignore` no longer applies a blanket `*.log` rule, which had been
  silently excluding the validation run logs; LaTeX artefacts are now ignored
  per directory.

### Removed
- The exploratory validation record: model-selection sweeps, prompt variants,
  truncation experiments, the Likert coherence runs, and the small-*n* pilots
  that set the final configuration. This deposit archives the experiment the
  article reports, at the scale it reports, rather than the path to it. The
  coherence *protocol* remains implemented in the package; only its exploratory
  transcripts are gone. Everything removed is preserved in git history before
  this release.
- The superseded per-level intrusion scripts under
  `code/07_llm_validation/archive/`, replaced by
  `src/clustervalidation/protocols/intrusion.py` and no longer the provenance
  of any deposited result.

### Known limitations of the record
Stated once, plainly, rather than as items awaiting action.
- The curated 279-term search list is not in this repository. Stage 02 expects
  it at `code/01_keyword_construction/search_terms.txt`.
- Figure 1's rendering code is not part of the deposit. The figure itself is in
  `paper/figures/` and the taxonomy it displays in `taxonomy.csv`.
- The expert consolidation at h2 and h3 was recorded only in part: `H3_MAP`
  held one slice at a time and was overwritten between runs, and a final h2
  pass that moved three cluster groups between macro-domains left no record.
  `h2_labeling_computer_science.py`, `_natural_science.py` and
  `_social_science.py` therefore diverge from the published taxonomy, and
  `h2_umap_social_science.py` is an unadapted copy of the engineering script.
  Nothing was reconstructed by guesswork; the published labels ship with the
  corpus and are authoritative.
- The survey PDFs behind stage 01 are third-party copyrighted material and are
  not redistributed.

## [1.0.0] — 2026-07-28

First structured release. Reorganises the project as a research compendium
suitable for archival deposit.

### Added
- `clustervalidation` package under `src/`, replacing 19 near-duplicate
  per-experiment scripts (~3,900 lines) with one parameterised implementation
  driven by a CLI.
- Seeded panel and sample construction, making a run reproducible from its
  configuration. Panels now use a dedicated `random.Random` instance rather
  than global random state.
- Run manifests: every result is written as `.json` (configuration + summary),
  `.jsonl` (one record per trial) and `.txt` (readable transcript), each
  carrying the parameters that produced it.
- Prompt registry (`prompts.py`) holding every wording used during
  development, addressable by name and recorded in each manifest.
- Extraction rule tracking: `parsing.py` reports *which* pattern produced a
  verdict, distinguishing an explicit marker from a last-resort digit match.
- Retry with exponential backoff on transient API failures; a trial that still
  fails is recorded rather than aborting the run.
- `--dry-run`, which builds panels and prints a prompt without calling the API,
  and `inspect`, which reports corpus statistics.
- Test suite (62 tests) covering extraction, sampling, seed determinism,
  corpus loading and an end-to-end run against a synthetic corpus. Requires no
  API key or network access.
- Research-compendium metadata: `CITATION.cff`, `.zenodo.json`, `LICENSE`
  (MIT), `LICENSE-DATA` (CC BY 4.0), `pyproject.toml`, `CHANGELOG.md`.
- Documentation: top-level `README.md`, `data/README.md` (schema, provenance
  and how to obtain the corpus), `results/README.md` (a per-file manifest of
  every archived run with its configuration and outcome).

### Changed
- Repository restructured: `code/` → `src/clustervalidation/`,
  `latex/` → `paper/` (current manuscript as `main.tex`, superseded drafts and
  the originating thesis under `paper/archive/`, figures under
  `paper/figures/`), and `results/` split into `intrusion/`, `coherence/` and
  `exploratory/`.
- Archived result filenames are deliberately **unchanged**, preserving the
  provenance link to the original lab record.
- All hardcoded absolute paths replaced with paths derived from the package
  location, so the repository works from any checkout directory.

### Fixed
- Verdicts emitted only inside a reasoning trace are now recovered, and
  `finish_reason` is recorded per trial. Several `deepseek-v4-pro` and
  `deepseek-v3` runs in `results/exploratory/` report 0 % accuracy purely
  because the original scripts scored an unparseable response as incorrect;
  those runs measured the harness, not the model.

### Security
- Removed a hardcoded DeepSeek API key that was present in 19 files and in the
  initial commit. Credentials are now read from `DEEPSEEK_API_KEY`; see
  `.env.example`. **The exposed key must be treated as compromised and
  rotated**, regardless of repository visibility.
- Purged the 396 MB corpus blob from git history, which exceeded GitHub's
  100 MB file limit and would have blocked any push. The corpus is now
  gitignored and documented in `data/README.md`.
