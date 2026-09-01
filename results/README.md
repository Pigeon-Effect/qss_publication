# Results

The complete record of the document-intrusion validation reported in the
article: 3,000 trials, 1,000 at each of the three hierarchy levels.

```
results/intrusion/
├── intrusion_h1_n1000_decisive.{json,jsonl,txt,log,err}    5 domains
├── intrusion_h2_n1000_decisive.{json,jsonl,txt,log,err}    31 fields
└── intrusion_h3_n1000_decisive.{json,jsonl,txt,log,err}    106 research fronts
```

## What was measured

Each trial shows a judge five documents: four drawn from one target cluster and
one intruder drawn from a different cluster at the same level, shuffled. Titles
and abstracts are truncated to 200 words. Nothing else is shown — no cluster
name, no keywords, no hint about which document came from where. The judge names
the intruder.

If a cluster is a genuine topic, its four members hang together and the intruder
is obvious. If the cluster is an artefact of the algorithm, it isn't. Detection
accuracy is therefore a measure of cluster coherence, against a random-guess
baseline of 1 in 5.

## The result

| Level | Granularity | Trials | Correct | Accuracy | Baseline |
|---|---|---:|---:|---:|---:|
| h1 | 5 domains | 1,000 | 405 | **40.5 %** | 20 % |
| h2 | 31 fields | 1,000 | 647 | **64.7 %** | 20 % |
| h3 | 106 research fronts | 1,000 | 778 | **77.8 %** | 20 % |

The monotonic rise is the substantive finding. Intrusion detection depends on
the ratio of within-cluster spread to between-cluster separation, and a broad
domain is heterogeneous by construction — a panel drawn from *Natural Science*
may pair structural health monitoring with ionospheric prediction and rural
land-use change, and an intruder hides easily among them. A narrow research
front has a tight vocabulary, and an outsider stands out. Every level clears
chance by a wide margin, and the finest level, which the citation analysis leans
on hardest, scores best.

## Reading a run

Each run writes five files sharing a stem.

| Extension | Contents |
|---|---|
| `.json` | the manifest: every parameter that produced the run, plus the summary counts |
| `.jsonl` | one record per trial — the panel, the intruder's position, the full response, the extracted verdict, the rule that extracted it, token counts and cost |
| `.txt` | the readable transcript: the panel as the model saw it, its reasoning, and the verdict, trial by trial |
| `.log` | the run log, as printed while it ran |
| `.err` | the error stream; here, one API-outage notice per run |

The `.json` manifest is the authoritative record of *how* a run was configured.
Reading one is the fastest way to see what a result means:

```bash
python -c "import json;print(json.dumps(json.load(open('results/intrusion/intrusion_h3_n1000_decisive.json')),indent=2))"
```

## Configuration of the reported runs

Identical across all three levels except for `--level`:

| Parameter | Value |
|---|---|
| model | `deepseek-v4-flash`, reasoning enabled |
| prompt variant | `decisive` |
| trials | 1,000 |
| seed | 20250628 |
| panel size | 5 (4 home + 1 intruder) |
| abstract truncation | 200 words |
| token ceiling | 8,000 |
| minimum cluster size | 5 |
| forced-choice extractor | `deepseek-chat` |
| package version | 1.0.0 |

Total API cost across the three runs: $2.70.

## Response extraction

Reasoning models do not always emit the requested verdict line. Extraction runs
from the most explicit pattern to the least and records **which rule fired** for
every trial, so a value read off an explicit marker is never confused with one
recovered by a fallback.

| | h1 | h2 | h3 |
|---|---:|---:|---:|
| `final_verdict` — the requested line, read directly | 894 | 930 | 961 |
| `forced_choice` — truncated trace resolved by a second model | 103 | 70 | 37 |
| other explicit rules (`answer`, `last_digit`, `bare_line`) | 3 | 0 | 2 |
| **unparsed** | **0** | **0** | **0** |
| **forced guesses** | **0** | **0** | **0** |
| responses truncated at the token ceiling | 127 | 81 | 45 |
| verdicts recovered from the reasoning trace | 22 | 11 | 7 |

Two recovery paths are worth understanding, because both were needed here.

**Reasoning-trace recovery.** A model that states its answer inside its
reasoning but never writes the final line has answered the question; the trial
is not a failure. The verdict is read from the trace, and the run records that
it was.

**Forced choice.** A response cut off by the token ceiling mid-reasoning is
passed to a second, non-reasoning model whose only job is to read the truncated
trace and report which paper it was converging on. This is an extraction step,
not a second opinion: the extractor never sees the panel and cannot form a view
of its own.

Neither path guesses. `forced_guesses` is 0 in all three runs — no trial was
resolved by falling back to a random answer — and `unparsed_responses` is 0, so
every one of the 3,000 trials was scored from something the model actually said.
Truncation is more frequent at h1 because the harder panels are exactly the ones
the model reasons longest over.

## Reproducing a run

```bash
python -m clustervalidation intrusion --level h3 --trials 1000 \
    --prompt decisive --max-tokens 8000 --seed 20250628
```

Identical `--seed`, `--level`, `--trials`, `--panel-size` and `--max-words`
reconstruct the identical panel sequence, because panel construction uses a
dedicated seeded generator rather than global random state. Model responses will
still differ: sampling is non-deterministic server-side and the API is a moving
target. At n = 1,000 the sampling error on an accuracy figure is roughly ±3 pp.

## Development history

These three runs are the reported result. The exploratory work behind them —
model selection, prompt variants, truncation sweeps, the Likert coherence
protocol, and the smaller pilot runs that set the final configuration — is not
carried in this deposit, which archives the experiment the article reports
rather than the path to it. That history is preserved in the repository's git
history before `v2.0.0` for anyone who wants it.
