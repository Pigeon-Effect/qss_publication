# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### To resolve before deposit
- Reconcile the trial count and accuracies between `paper/main.tex` and
  `results/intrusion/` (see the README's *Known discrepancy* section).
- Update the manuscript's *Data and Code* section, which currently points only
  at the pipeline repository and does not cite this validation component.
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
