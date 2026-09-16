"""Paired comparison of the `subset` runs against the deposited `decisive` runs.

Both sets of trials are built from `random.Random(20250628)` drawn
sequentially, so trial *i* here faces exactly the panel trial *i* of the
1,000-trial deposited run faced. The comparison is therefore paired, and the
right test is McNemar's on the discordant pairs — the trials one prompt got
right and the other got wrong — not a two-sample test on the two accuracies.

Run from the repository root:

    python results/prompt_experiments/compare.py
"""

from __future__ import annotations

import json
import os
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUBSET = os.path.join(ROOT, "results", "prompt_experiments")
DECISIVE = os.path.join(ROOT, "results", "intrusion")
LEVELS = ("h1", "h2", "h3")
LEVEL_NAMES = {
    "h1": "5 domains",
    "h2": "31 fields",
    "h3": "106 research fronts",
}


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def binomial_two_sided(successes: int, trials: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value, by summing outcomes no more likely
    than the observed one. Used for McNemar's test on the discordant pairs,
    which is exact and appropriate at the small counts this experiment
    produces, where the chi-square approximation is not."""
    if trials == 0:
        return 1.0
    observed = comb(trials, successes) * p**successes * (1 - p) ** (trials - successes)
    total = 0.0
    for k in range(trials + 1):
        probability = comb(trials, k) * p**k * (1 - p) ** (trials - k)
        if probability <= observed * (1 + 1e-9):
            total += probability
    return min(1.0, total)


def paired(level: str) -> dict | None:
    subset_path = os.path.join(SUBSET, f"intrusion_{level}_n100_subset.jsonl")
    decisive_path = os.path.join(DECISIVE, f"intrusion_{level}_n1000_decisive.jsonl")
    if not os.path.exists(subset_path):
        return None

    subset = load(subset_path)
    decisive = load(decisive_path)[: len(subset)]

    # Confirm the panels really are the same before comparing outcomes on them.
    aligned = sum(
        1
        for s, d in zip(subset, decisive)
        if s["home_cluster"] == d["home_cluster"]
        and s["intruder_cluster"] == d["intruder_cluster"]
        and s["true_position"] == d["true_position"]
    )

    # McNemar's contingency: only the discordant cells carry information.
    subset_only = sum(
        1 for s, d in zip(subset, decisive) if s["correct"] and not d["correct"]
    )
    decisive_only = sum(
        1 for s, d in zip(subset, decisive) if d["correct"] and not s["correct"]
    )
    both = sum(1 for s, d in zip(subset, decisive) if s["correct"] and d["correct"])
    neither = len(subset) - subset_only - decisive_only - both
    discordant = subset_only + decisive_only

    return {
        "level": level,
        "n": len(subset),
        "aligned": aligned,
        "subset_correct": sum(1 for s in subset if s["correct"]),
        "decisive_correct": sum(1 for d in decisive if d["correct"]),
        "subset_truncated": sum(1 for s in subset if s["truncated"]),
        "decisive_truncated": sum(1 for d in decisive if d["truncated"]),
        "subset_cost": sum(s["cost_usd"] for s in subset),
        "both": both,
        "neither": neither,
        "subset_only": subset_only,
        "decisive_only": decisive_only,
        "p_value": binomial_two_sided(subset_only, discordant),
    }


def main() -> int:
    rows = [r for r in (paired(level) for level in LEVELS) if r is not None]
    if not rows:
        print("No subset runs found yet.")
        return 1

    print("\nPanel alignment (paired design requires 100%)")
    for r in rows:
        print(f"  {r['level']}: {r['aligned']}/{r['n']} panels identical to deposited run")

    print("\nAccuracy on the same panels")
    header = f"  {'level':<6}{'n':>5}{'subset':>10}{'decisive':>11}{'delta':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        s = 100 * r["subset_correct"] / r["n"]
        d = 100 * r["decisive_correct"] / r["n"]
        print(
            f"  {r['level']:<6}{r['n']:>5}{s:>9.1f}%{d:>10.1f}%{s - d:>+8.1f}pp"
        )

    print("\nMcNemar's test on the discordant pairs")
    for r in rows:
        print(
            f"  {r['level']}: subset-only {r['subset_only']:>3} | "
            f"decisive-only {r['decisive_only']:>3} | "
            f"both {r['both']:>3} | neither {r['neither']:>3} | "
            f"p = {r['p_value']:.4f}"
        )

    print("\nTruncation (the confound the 2026-08-07 pilots had)")
    for r in rows:
        print(
            f"  {r['level']}: subset {r['subset_truncated']:>3}/{r['n']} | "
            f"decisive {r['decisive_truncated']:>3}/{r['n']}"
        )

    print(f"\nTotal API cost of the subset runs: "
          f"${sum(r['subset_cost'] for r in rows):.4f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
