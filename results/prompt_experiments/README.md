# Prompt experiments — group construction vs. outlier detection

A controlled comparison of two intrusion-detection prompt framings on the same
panels. Not part of the article's reported validation; this folder exists so the
question "would a different prompt have scored higher?" has a measured answer
rather than an extrapolated one.

```
results/prompt_experiments/
├── README.md                                    this file
├── intrusion_h1_n100_subset.{json,jsonl,txt,log,err}    5 domains
├── intrusion_h2_n100_subset.{json,jsonl,txt,log,err}    31 fields
└── intrusion_h3_n100_subset.{json,jsonl,txt,log,err}    106 research fronts
```

## Why this was run

The deposited validation in [`../intrusion/`](../intrusion/) used the `decisive`
prompt and reported 40.5 / 64.7 / 77.8 % at h1 / h2 / h3. During development a
second framing, `subset`, was written and piloted, and a belief formed that it
scored substantially better — "up to 91 percent" at h3.

That belief had no measurement behind it. It came from two places, neither of
them a completed run:

1. **An extrapolation.** A blinded human check on 20 panels the model had failed
   recovered the answer on 12 of them. Applying that recovery rate to the
   deposited h3 result gives 77.8 + (22.2 × 12/20) = **91.1 %**. It assumes the
   new prompt would recover failures as well as a human reader did, which was
   never tested.
2. **A subgroup of a token-starved pilot.** The `subset` pilots of 2026-08-07
   ran at `max_tokens: 3000`, against 8000 for the deposited runs, and 39–58 %
   of their responses were cut off mid-reasoning. Their headline accuracies came
   out *below* the deposited runs (37 / 63 / 72 %). Restricted to the
   non-truncated trials, h3 rose to **90.2 %** — but "non-truncated at 3000
   tokens" selects for easy panels far more aggressively than "non-truncated at
   8000 tokens" does, so that number compares two differently-filtered samples
   and cannot be read as a prompt effect.

This experiment removes both confounds: same token ceiling as the deposited
runs, same panels, whole sample.

## What differs between the two prompts

Both live in [`../../src/clustervalidation/prompts.py`](../../src/clustervalidation/prompts.py)
and are addressed by name (`--prompt decisive` / `--prompt subset`). The panel,
the corpus, the model and the verdict format are identical. Two things change.

**1. The task is posed as group construction, not outlier detection.**

`decisive` asks the model to label each paper and then name the one that does
not fit. `subset` asks it to *find the group of four that forms the tightest
single field*; the paper left over is the answer.

On a five-item panel these are not equivalent. Outlier-spotting compares each
paper against a "rest" that is never pinned down, so the comparison target
shifts as the model works. The subset search is constructive and makes the
"exactly four" constraint do work: a grouping is only admissible if it accounts
for four papers, which rules out readings that outlier-spotting entertains.

**2. Comparing rival groupings is permitted and requested.**

This directly reverses `decisive`, which forbids revisiting a paper once
labelled and forbids testing alternative answers against each other. Those
constraints were written after the h3 pilot showed that long reasoning traces
signal failure, not care — wrong answers reasoned 4.1× longer than correct ones
(mean 6,820 vs 1,680 characters) — and that the model could loop through
hypotheses until the token ceiling cut it off without ever committing.

The constraints solved that, but they may have overcorrected: on a genuinely
close panel, weighing two candidate groupings against each other is exactly the
operation that resolves it. `subset` re-permits the comparison and instead
bounds the cost by asking for it only at step 2, once topics are already named.

## Design

Every parameter matches the deposited runs except `--prompt` and `--trials`.

| Parameter | Value | Same as deposited run? |
|---|---|---|
| model | `deepseek-v4-flash`, reasoning enabled | yes |
| `--max-tokens` | 8000 | yes — the 2026-08-07 pilots used 3000 |
| `--seed` | 20250628 | yes |
| `--panel-size` | 5 | yes |
| `--max-words` | 200 | yes |
| `--force-choice-model` | `deepseek-chat` | yes |
| `--prompt` | `subset` | **no** — deposited runs used `decisive` |
| `--trials` | 100 | no — deposited runs used 1000 |
| package version | 2.0.0 | no — deposited runs recorded 1.0.0, being older |

**The comparison is paired.** `build_panels` draws from a single
`random.Random(seed)` sequentially, so the first 100 panels of a 1,000-trial run
are byte-identical to a 100-trial run at the same seed. Each `subset` trial here
therefore faces the same home cluster, the same four members, the same intruder
and the same shuffled position as the corresponding trial in
`../intrusion/*_n1000_decisive.jsonl`. The comparison is against those first 100
trials, not against the 1,000-trial headline, and panel difficulty is held
fixed rather than averaged over.

Reproduce with:

```bash
for level in h1 h2 h3; do
  python -m clustervalidation intrusion \
    --level $level --prompt subset --trials 100 --seed 20250628 \
    --max-tokens 8000 --results-dir results/prompt_experiments \
    --stem intrusion_${level}_n100_subset \
    > results/prompt_experiments/intrusion_${level}_n100_subset.log \
    2> results/prompt_experiments/intrusion_${level}_n100_subset.err
done
```

## Results

Run 2026-09-02. Three runs, 300 trials, 0 failed, 0 unparsed, 0 forced guesses,
$0.52 total. Panel alignment against the deposited runs is 100/100 at every
level, so every comparison below is paired.

`subset` is **worse at all three levels**, on the same panels:

| Level | n | `subset` | `decisive`, same panels | Delta | `decisive`, all 1,000 |
|---|---:|---:|---:|---:|---:|
| h1 | 100 | 36.0 % | 45.0 % | −9.0 pp | 40.5 % |
| h2 | 100 | 63.0 % | 71.0 % | −8.0 pp | 64.7 % |
| h3 | 100 | **76.0 %** | 85.0 % | −9.0 pp | 77.8 % |

McNemar's exact test on the discordant pairs — the only trials that carry
information about a prompt difference:

| Level | only `subset` right | only `decisive` right | p |
|---|---:|---:|---:|
| h1 | 2 | 11 | 0.023 |
| h2 | 5 | 13 | 0.096 |
| h3 | 1 | 10 | 0.012 |
| **pooled** | **8** | **34** | **6.9 × 10⁻⁵** |

The hypothesis is not merely unconfirmed; the effect runs the other way.

### The 91 % was an extrapolation, and the 90.2 % was a selection artefact

Both prior figures are now accounted for.

The extrapolation assumed the new prompt would recover failed panels at the
rate a human reader did. It does not: at h3 it recovers **1** panel `decisive`
missed while losing **10** it had got right.

The 90.2 % is the more instructive one, because **it reproduces here almost
exactly** — and is still wrong:

| Level | `subset`, truncated | `subset`, non-truncated | `subset`, all |
|---|---:|---:|---:|
| h1 | 21.3 % (n=47) | 49.1 % (n=53) | 36.0 % |
| h2 | 30.6 % (n=36) | 81.2 % (n=64) | 63.0 % |
| h3 | 39.3 % (n=28) | **90.3 %** (n=72) | **76.0 %** |

90.3 % against the pilot's 90.2 %, at a token ceiling nearly three times
higher. Conditioning on "the model finished its reasoning" selects the panels
the model found easy, and it does so no matter where the ceiling sits. The
subgroup is therefore not an estimate of anything — it is a restatement of
which panels were easy, and it cannot be compared to a whole-sample figure.

### Why `subset` loses: the constraints it removed were load-bearing

The mechanism is in the truncation counts, at the *same* 8,000-token ceiling:

| Level | `subset` truncated | `decisive` truncated |
|---|---:|---:|
| h1 | 47/100 | 15/100 |
| h2 | 36/100 | 2/100 |
| h3 | 28/100 | 4/100 |

`subset` runs out of tokens 7–18× more often. Re-permitting "if more than one
grouping is plausible, compare them directly" reopens exactly the loop
`decisive`'s constraints were written to close: the model weighs candidate
groupings until the ceiling severs the trace. The 2026-08-07 pilots' truncation
was blamed on their 3,000-token ceiling; most of it was the prompt.

So the `decisive` constraints were not an overcorrection that cost accuracy on
hard panels. They were what made the hard panels answerable at all.

### A limitation of this test

`subset` changes two things at once — the group-construction framing *and* the
removal of the no-comparison constraints. The experiment therefore answers "is
`subset` better than `decisive`?" (no, decisively) but not "is group
construction *per se* worse?". It remains possible that the framing is neutral
or mildly helpful and that the entire loss comes from the removed constraints.

Separating them needs a third variant: group-construction framing *with*
`decisive`'s constraints. That is not registered and has not been run.

### Effect on the article

None. The deposited 40.5 / 64.7 / 77.8 % remain the reported result and remain
the best available estimates. This experiment closes the open question in favour
of the prompt already used.

Note that `decisive` scores above its own 1,000-trial average on all three
first-100 prefixes (45.0 / 71.0 / 85.0 % against 40.5 / 64.7 / 77.8 %). Under
exchangeability the prefix count is hypergeometric, giving p = 0.195 / 0.099 /
0.041 per level and p = 0.027 combined (Fisher). Per-block accuracy over each
deposited run shows no trend and no break at the API outage, so this is a mildly
lucky prefix rather than an artefact — and it does not touch the comparison
above, which is paired: an easy panel raises both columns and cancels in the
discordant cells.

## Status

Question closed. `subset` is not registered as the default and no deposited
result uses it; it stays in `prompts.py` because these runs cite it by name.
