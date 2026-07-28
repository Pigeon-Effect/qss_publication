# Results

Archived transcripts of every validation run. Filenames are **preserved exactly
as generated** — they are the provenance link back to the lab record, so they
are deliberately not renamed to a tidy scheme.

```
results/
├── intrusion/      document-intrusion detection runs
├── coherence/      Likert coherence-rating runs
└── exploratory/    model selection, prompt pilots, debug runs
```

Each file opens with a header stating the model, truncation length and trial
count, followed by the full transcript of every trial: the panel shown, the
model's response, and the verdict extracted from it.

> **Format note.** Everything here was produced by the original per-experiment
> scripts (git history before `v1.0.0`), which wrote plain text only. Runs made
> with the current package additionally emit `.json` (manifest + summary) and
> `.jsonl` (one record per trial) alongside the transcript. Old files were not
> retro-converted, because rewriting an experimental record to a nicer format
> is exactly the kind of silent edit an archive should not contain.

---

## Reported runs

The configuration reported in the manuscript: `deepseek-v4-flash` with
reasoning enabled, 200-word abstract truncation, 5-document panels
(4 home + 1 intruder).

| File | Level | Trials | Accuracy | Baseline |
|---|---|---:|---:|---:|
| `intrusion/intrusion_deepseek_v4_flash_h1_100.txt` | h1 — domains | 100 | **44.0 %** | 20 % |
| `intrusion/intrusion_deepseek_v4_flash_h2_100.txt` | h2 — fields | 100 | **68.0 %** | 20 % |
| `intrusion/intrusion_deepseek_v4_flash_100.txt` | h3 — research fronts | 100 | **82.0 %** | 20 % |

The monotonic rise across levels is the substantive finding: intrusion
detection depends on the ratio of within-cluster spread to between-cluster
separation, so broad domains — heterogeneous by construction — make an intruder
less conspicuous than narrow, lexically distinctive research fronts do.

> ### ⚠ Discrepancy with the current manuscript draft
>
> Section *External Validation via Document Intrusion* of `paper/main.tex`
> states **1,000 trials per hierarchy level** and reports **46.0 % / 75.0 % /
> over 84.0 %**. The archived runs above are **100 trials** at
> **44.0 % / 68.0 % / 82.0 %**.
>
> Either a larger run exists that was never archived here, or the manuscript
> figures need correcting before submission. This must be reconciled — the
> deposit should reproduce the paper's numbers. See the
> [main README](../README.md#known-discrepancy).

---

## `intrusion/` — all runs

Unless noted, the model is `deepseek-chat` at temperature 0 and panels are
4 home + 1 intruder, giving a 20 % random baseline.

| File | Level | Prompt variant | Trunc. | Trials | Accuracy |
|---|---|---|---:|---:|---:|
| `intrusion_deepseek_v4_flash_h1_100.txt` | h1 | `reasoned` (v4-flash, thinking) | 200 w | 100 | 44.0 % |
| `intrusion_deepseek_v4_flash_h2_100.txt` | h2 | `reasoned` (v4-flash, thinking) | 200 w | 100 | 68.0 % |
| `intrusion_deepseek_v4_flash_100.txt` | h3 | `reasoned` (v4-flash, thinking) | 200 w | 100 | 82.0 % |
| `intrusion_deepseek_chat_100.txt` | h3 | `reasoned` | 200 w | 100 | 68.0 % |
| `intrusion_h1_basic_results_001.txt` | h1 | `minimal`, titles included | 100 w | 100 | 22.0 % |
| `intrusion_h2_basic_results_001.txt` | h2 | `minimal`, titles included | 100 w | 100 | 35.0 % |
| `intrusion_h3_basic_results_001.txt` | h3 | `minimal`, titles included | 100 w | 100 | 39.0 % |
| `intrusion_h2_basic_results_1000_words.txt` | h2 | `minimal`, titles included | 1000 w | 100 | 39.0 % |
| `intrusion_h3_basic_results_1000_words.txt` | h3 | `minimal`, titles included | 1000 w | 100 | 27.0 % |
| `intrusion_h2_results_001.txt` | h2 | `minimal`, no titles | 100 w | 100 | 34.0 % |
| `intrusion_h2_with_titles_results.txt` | h2 | `minimal`, titles included | 100 w | 100 | 29.0 % |
| `h1_intrusion_multidisciplinary_results_001.txt` | h1 | broad-discipline framing | 100 w | 100 | 21.0 % |
| `h1_intrusion_test_results_001.txt` | h1 | `chain_of_thought` | 100 w | 100 | 24.0 % |
| `h3_intrusion_detailed_results_001.txt` | h3 | `narrow` | 100 w | 100 | 34.0 % |
| `intrusion_h3_topic_method.txt` | h3 | `topic_only` / `method_only` | 100 w | 100 | 36.0 % topic, 29.0 % method |
| `friendly_intrusion_h3_crossh1_001.txt` | h3 | 2 home + 1 intruder, cross-h1 | 150 w | 100 | 60.0 % † |

† This run uses 3-document panels, so its random baseline is 33 %, not 20 %.
It is not comparable to the rows above.

**What these show.** Non-reasoning `deepseek-chat` performs barely above
baseline at h1 and only moderately at h3. Enabling reasoning
(`deepseek-v4-flash`, thinking) is what lifts detection to the reported levels
— the single largest effect in the whole series, larger than any prompt or
truncation change. Longer truncation does not help monotonically: at h3, moving
from 100 to 1000 words *lowered* accuracy (39 % → 27 %), consistent with longer
abstracts diluting the distinguishing signal.

---

## `coherence/` — all runs

Direct Likert rating of a single cluster's sample; no random baseline, since
this is an absolute judgement rather than a forced choice.

| File | Level | Prompt variant | Trunc. | Clusters | Result |
|---|---|---|---:|---:|---|
| `coherence_h1_TanDSouza_100_words.txt` | h1 | `tan_dsouza` | 100 w | 5 | mean 2.80 |
| `coherence_h2_TanDSouza_100_words.txt` | h2 | `tan_dsouza` | 100 w | 31 | mean 1.87 |
| `coherence_h3_TanDSouza_100_words.txt` | h3 | `tan_dsouza` | 100 w | 100 | mean 1.50 |
| `coherence_rating_h3_results_001.txt` | h3 | `minimal` | 100 w | 100 | mean 3.97 |
| `coherence_h3_two_scores.txt` | h3 | `dual_score` | 500 w | 100 | topic 2.47 / method 1.91 — **32/100 parsed** |
| `coherence_h3_two_scores_001.txt` | h3 | `dual_score` | 100 w | 100 | see file header |
| `coherence_h3_two_scores_002_500_words.txt` | h3 | `dual_score` | 500 w | 100 | see file header |
| `coherence_h3_two_scores_003_friendlier_rating.txt` | h3 | `dual_score`, relaxed rubric | 500 w | 100 | topic 2.58 / method 1.93 — 91/100 parsed |

**Read these with care.** The coherence protocol is far less stable than
intrusion detection, in two distinct ways:

1. **Prompt sensitivity dwarfs the signal.** At h3 the same clusters score
   1.50 under the `tan_dsouza` rubric and 3.97 under `minimal` — a 2.5-point
   swing on a 5-point scale, driven entirely by wording. A rating that moves
   this much with the rubric cannot by itself support a claim about cluster
   quality.
2. **Extraction failure was severe and non-random.** `coherence_h3_two_scores`
   yielded only **32 of 100** parseable responses. The relaxed-rubric rerun
   recovered 91/100 at nearly identical means, which suggests the lost 68 were
   refusals-to-rate rather than a biased subsample — but the earlier file's
   means rest on a third of the intended sample and should not be quoted.

This instability is why the manuscript reports intrusion detection, not
coherence rating, as the external validation. The coherence runs are archived
as the exploratory work that led to that choice. The current package records
which extraction rule produced every value (see `parsing.py`) precisely so this
failure mode is visible in future runs rather than silent.

---

## `exploratory/` — model selection and diagnostics

Short runs used to choose a model and debug the harness. Small *n*; not
evidence for anything on their own.

| File | Purpose | Trials | Result |
|---|---|---:|---|
| `intrusion_models_comparison.txt` | multi-model sweep | 20/model | `deepseek-chat` 40.0 % |
| `intrusion_reasoner_vs_pro.txt` | reasoner vs. pro | 30/model | 0.0 % — extraction failure ‡ |
| `intrusion_reasoner_vs_pro_correct.txt` | rerun with thinking mode | 10/model | 0.0 % — extraction failure ‡ |
| `intrusion_v4pro_detailed.txt` | `deepseek-v4-pro` | 50 | 18.0 % ‡ |
| `intrusion_v4pro_optimized.txt` | `deepseek-v4-pro`, tuned prompt | 50 | 20.0 % ‡ |
| `intrusion_v4pro_concise.txt` | `deepseek-v4-pro`, forced brevity | 30 | 0.0 % ‡ |
| `intrusion_v3_500words_temp02.txt` | `deepseek-v3`, temp 0.2 | 100 | 0.0 % ‡ |
| `intrusion_debug_concise.txt` | harness debug | 10 | 90.0 % |
| `intrusion_test_debug.txt` | harness debug | 10 | 40.0 % |
| `intrusion_test_results_001.txt` | first pilot | 10 | see file |

‡ **The 0.0 % rows are not model failures.** A reasoning model that emits its
verdict only inside the reasoning trace, or that is cut off by the token
ceiling before writing the final line, yields no parseable answer — and the
original scripts scored an unparseable answer as wrong. These runs measure the
harness, not the model. Both problems are fixed in the current package
(reasoning-trace fallback, and `finish_reason` recorded per trial), which is
why the later `deepseek-v4-flash` runs succeed where these did not.

---

## Reproducing a run

The archived runs predate the seeded sampler, so they record **which panels
were shown** but cannot be regenerated panel-for-panel. Runs made with the
current package are reproducible from their manifest:

```bash
python -m clustervalidation intrusion --level h3 --trials 100 --seed 20250628
```

Identical `--seed`, `--level`, `--trials`, `--panel-size` and `--max-words`
reconstruct the identical panel sequence. Model responses may still vary:
sampling is non-deterministic server-side, and the API is a moving target.
