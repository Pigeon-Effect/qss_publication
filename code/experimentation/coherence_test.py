import sqlite3
import random
import os
from collections import defaultdict
from openai import OpenAI

# ================= CONFIGURATION =================
# Project root = two levels up from this file (code/<group>/<script>.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "merged_works_labeled.db")
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"                  # reliable for scalar judgments
NUM_CLUSTERS = 100                       # how many clusters to evaluate
PAPERS_PER_CLUSTER = 5                   # abstracts per evaluation
MIN_PAPERS = 5                           # only clusters with at least this many papers
# =================================================

if not API_KEY:
    raise SystemExit(
        "Set DEEPSEEK_API_KEY, e.g.\n"
        "  PowerShell: $env:DEEPSEEK_API_KEY='sk-...'"
    )

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ---------- 1. Load data from DB ----------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create the three-digit cluster ID (h1_h2_h3)
cursor.execute("""
    SELECT id, cleaned_abstract,
           CAST(h1_cluster AS TEXT) || CAST(h2_cluster AS TEXT) || CAST(h3_cluster AS TEXT) AS cluster_id
    FROM works_labeled
    WHERE cleaned_abstract IS NOT NULL AND trim(cleaned_abstract) != ''
""")
rows = cursor.fetchall()
print(f"Total valid papers loaded: {len(rows)}")

# Group papers by cluster_id
cluster_papers = defaultdict(list)
for paper_id, abstract, cid in rows:
    cluster_papers[cid].append((paper_id, abstract))

print(f"Distinct h3 clusters: {len(cluster_papers)}")

# Keep only clusters with enough papers
eligible = {cid: papers for cid, papers in cluster_papers.items() if len(papers) >= MIN_PAPERS}
print(f"Clusters with at least {MIN_PAPERS} papers: {len(eligible)}")

if len(eligible) < NUM_CLUSTERS:
    raise ValueError(f"Only {len(eligible)} eligible clusters, but need {NUM_CLUSTERS}. Reduce NUM_CLUSTERS.")

# ---------- 2. Randomly select clusters and sample papers ----------
selected_clusters = random.sample(list(eligible.keys()), NUM_CLUSTERS)

evaluation_sets = []
for cid in selected_clusters:
    pool = eligible[cid]
    sampled = random.sample(pool, PAPERS_PER_CLUSTER)   # list of (paper_id, abstract)
    evaluation_sets.append((cid, sampled))

print(f"Prepared {len(evaluation_sets)} coherence rating tasks.\n")

# ---------- 3. Rate each set using DeepSeek ----------
ratings = []               # store (cluster_id, rating, cost)
output_lines = []
total_cost = 0.0

for idx, (cid, abstracts) in enumerate(evaluation_sets, start=1):
    # Build prompt: list the abstracts
    numbered = []
    for i, (paper_id, abstract) in enumerate(abstracts, start=1):
        numbered.append(f"Abstract {i}:\n{abstract}\n")

    prompt = (
        "You are a research librarian. Below are 5 paper abstracts that have been algorithmically grouped together.\n"
        "Rate how coherent this collection is — do they all belong to a single recognizable scientific subfield "
        "or closely related area?\n"
        "Use a scale from 1 (completely unrelated) to 5 (very coherent, clearly one subfield).\n"
        "Answer with only the number (1-5).\n\n"
        + "\n".join(numbered)
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=3      # expecting a single digit
        )

        answer = response.choices[0].message.content.strip()
        # Extract rating (first integer found)
        import re
        rating_match = re.search(r'[1-5]', answer)
        rating = int(rating_match.group()) if rating_match else None

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        cost = (prompt_tokens * 0.14 + completion_tokens * 0.28) / 1_000_000
        total_cost += cost

        # Store rating (will be None if extraction failed)
        ratings.append((cid, rating, cost))

        result_str = (
            f"Cluster {idx}/{NUM_CLUSTERS}: {cid}\n"
            f"  LLM answer: '{answer}' → Rating: {rating}\n"
            f"  Tokens: prompt={prompt_tokens}, completion={completion_tokens}, cost=${cost:.6f}\n"
            f"{'─'*50}\n"
        )

    except Exception as e:
        ratings.append((cid, None, 0.0))
        result_str = (
            f"Cluster {idx}/{NUM_CLUSTERS}: {cid}\n"
            f"  ERROR: {e}\n{'─'*50}\n"
        )

    print(result_str)
    output_lines.append(result_str)

# ---------- 4. Summarize ----------
valid_ratings = [r for (_, r, _) in ratings if r is not None]
if valid_ratings:
    mean_rating = sum(valid_ratings) / len(valid_ratings)
    min_rating = min(valid_ratings)
    max_rating = max(valid_ratings)
else:
    mean_rating = min_rating = max_rating = None

summary = (
    f"\n{'='*60}\n"
    f"COHERENCE RATING SUMMARY\n"
    f"Clusters evaluated: {len(ratings)} (requested {NUM_CLUSTERS})\n"
    f"Valid ratings obtained: {len(valid_ratings)}\n"
    f"Mean coherence: {mean_rating:.2f} (1–5 scale)\n"
    f"Min / Max: {min_rating} / {max_rating}\n"
    f"Total cost: ${total_cost:.6f}\n"
    f"{'='*60}\n"
)
print(summary)

# ---------- 5. Write detailed report to file ----------
output_path = os.path.join(os.path.dirname(DB_PATH), "coherence_rating_h3_results.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("CLUSTER COHERENCE RATING (H3 GRANULAR CLUSTERS)\n")
    f.write(f"Model: {MODEL}\n")
    f.write(f"Clusters sampled: {NUM_CLUSTERS} out of {len(eligible)} eligible\n")
    f.write(f"Abstracts per cluster: {PAPERS_PER_CLUSTER}\n")
    f.write(f"Mean coherence: {mean_rating:.2f}\n")
    f.write(f"Cost: ${total_cost:.6f}\n\n")
    f.write("Prompt used:\n")
    f.write("---\n")
    f.write("You are a research librarian. Below are 5 paper abstracts that have been "
            "algorithmically grouped together.\n"
            "Rate how coherent this collection is — do they all belong to a single "
            "recognizable scientific subfield or closely related area?\n"
            "Use a scale from 1 (completely unrelated) to 5 (very coherent, clearly one subfield).\n"
            "Answer with only the number (1-5).\n---\n\n")

    for idx, (cid, (rating, cost)) in enumerate(zip(selected_clusters, [(r, c) for (_, r, c) in ratings]), start=1):
        f.write(f"Cluster {idx}: {cid} → Coherence: {rating}\n")
        # list the abstracts used
        for i, (paper_id, abstract) in enumerate(evaluation_sets[idx-1][1], start=1):
            f.write(f"  [{i}] ID={paper_id}\n      {abstract}\n\n")
        f.write("\n")

    f.write(summary)

print(f"Full report saved to {output_path}")
conn.close()