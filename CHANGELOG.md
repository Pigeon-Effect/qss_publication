# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Stages 01–04 and 06 of the pipeline, imported from the private working
  repository into `code/`: keyword construction, OpenAlex retrieval, quality
  control, the semi-supervised hierarchical topic model (h1/h2/h3, plus the
  `finding_optimal_k` evidence and the parent-scoped contrastive TF-IDF term
  extraction), and the citation-share heatmap.
- `data/interim/` as the documented location for every intermediate artefact
  the pipeline builds, with a per-file inventory in `data/README.md`.
- `python-dotenv`: credentials are now read from a local `.env` file as well as
  from the environment, in both `clustervalidation` and the archived scripts.
  An exported variable still takes precedence.
- Environment-variable overrides for every path that legitimately varies
  (`QSS_INTERIM_DIR`, `QSS_DB_PATH`, `QSS_H3_DB`, `QSS_SEARCH_TERMS`,
  `QSS_SURVEY_TXT_DIR`, `QSS_CITSHARE_CSV`), and a `--db` flag on the h3
  clustering script.

### Changed
- Every stage `README.md` rewritten against the code that is now present.
  They previously described what *should* go in each folder; they now document
  what each script does, why the method is what it is, how to run it, and its
  inputs and outputs.
- All 67 hardcoded absolute paths across 32 scripts in `code/` replaced with
  paths derived from each file's own location, so a fresh clone runs anywhere.
- `download_full_dataset.py` no longer inlines a 48-term search list. It loads
  the 279 curated terms from `code/01_keyword_construction/search_terms.txt`
  and queries them in groups, because OpenAlex limits the length of a single
  `search` expression, merging and deduplicating the groups on the OpenAlex
  work id. This matches the batched retrieval the manuscript describes; the
  archived script did not.
- OpenAlex credentials in `download_full_dataset.py` moved from `"xxx"`
  placeholders to `OPENALEX_MAILTO` / `OPENALEX_API_KEY`.
- `h2_labeling_biomedical.py`: meso 4 renamed *Immunology & Infectious Disease*
  → *Microbial and Immune Systems Biology*, meso 5 *Genetics, Genomics &
  Oncology* → *Genetics & Genomics*, matching Figure 1. Names only; no cluster
  assignment changed.
- `h2_labeling_social_science.py`: meso 5 *Urban Development & Tourism* →
  *Urban Development* (Tourism is micro-cluster 251 in the published taxonomy).

### Known gaps, now documented rather than silent
- `h2_labeling_computer_science.py` records eight meso codes; the manuscript
  gives Computer Science five. A later consolidation pass moved Neuromorphic
  Hardware Accelerators to Natural Science, Ethical & Creative AI to Social
  Science, and dissolved Recommendation Systems — reassigning fine clusters
  across macro-domains, which these per-subset scripts cannot express. The
  corresponding meso-clusters are correspondingly absent from
  `h2_labeling_natural_science.py` (35, 36) and
  `h2_labeling_social_science.py` (26). Not reconstructed: the record of which
  fine cluster ids moved is not in this repository, and guessing it would
  fabricate part of the taxonomy.
- `h2_cluster_social_sciences/h2_umap_social_science.py` is an unadapted copy
  of the engineering script — its paths, `remap` and palette are all
  Engineering. Flagged in the file; not repaired, for the same reason.
- `H3_MAP` in `h3_label_selected_cluster.py` held one (h1, h2) slice at a time
  and was overwritten between runs. Only the last (h1=4, h2=5) survives.
- Stage 05 (fractional citation sum) has no code, and Figure 1's generating
  code is absent from stage 06.

### Changed (earlier this cycle)
- Corrected repository scope. `README.md`, `.zenodo.json` and `CITATION.cff`
  previously described this repository as containing "the validation
  component only," pointing to a separate, private repository for data
  collection, topic modeling, and visualization. That was wrong: this is the
  single archival deposit intended to hold the full pipeline behind
  `paper/main.pdf`. Added `code/`, structured as one numbered stage per
  manuscript section (`01_keyword_construction` through
  `07_llm_validation`), each with a `README.md` describing what belongs there
  and its manuscript section/figures.
- Moved the pre-restructure `h1`/`h2`/`h3_intrusion_detection_v4_thinking.py`
  scripts into `code/07_llm_validation/archive/intrusion_detection_scripts/`,
  matching the existing `paper/archive/` convention for superseded material.

### To resolve before deposit
- Import stage 05 (fractional citation sum) and the generating code for
  Figure 1.
- Add the curated 279-term search list at
  `code/01_keyword_construction/search_terms.txt`.
- Recover the record of the final h2 consolidation pass, so
  `h2_labeling_computer_science.py`, `_natural_science.py` and
  `_social_science.py` can be brought in line with the published taxonomy.
- Reconcile the trial count and accuracies between `paper/main.tex` and
  `results/intrusion/` (see the README's *Known discrepancy* section).
- Update the manuscript's *Data and Code* section to cite this repository.
- Mint the Zenodo DOI and record it in `README.md`, `CITATION.cff` and
  `.zenodo.json`.

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
