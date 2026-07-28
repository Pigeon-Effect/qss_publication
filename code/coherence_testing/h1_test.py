import sqlite3
import random
import os
import re
from collections import defaultdict
from openai import OpenAI

# ================= CONFIGURATION =================
# Project root = two levels up from this file (code/<group>/<script>.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "merged_works_labeled.db")
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
NUM_CLUSTERS = 100                        # number of h1 clusters to evaluate
PAPERS_PER_CLUSTER = 5                    # abstracts per evaluation
MAX_WORDS = 100                           # truncate abstracts to first 100 words
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
# =================================================

os.makedirs(RESULTS_DIR, exist_ok=True)

if not API_KEY:
    raise SystemExit(
        "Set DEEPSEEK_API_KEY, e.g.\n"
        "  PowerShell: $env:DEEPSEEK_API_KEY='sk-...'"
    )

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def truncate_words(text, max_words):
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text

def extract_rating(text):
    """Extract rating from 'Rating: X' format."""
    match = re.search(r'Rating:\s*(\d)', text, re.IGNORECASE)
    return int(match.group(1)) if match else None

# ---------- 1. Load data and group by h1 ----------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT id, title, cleaned_abstract, h1_cluster
    FROM works_labeled
    WHERE cleaned_abstract IS NOT NULL AND trim(cleaned_abstract) != ''
""")
rows = cursor.fetchall()
print(f"Total valid papers: {len(rows)}")

cluster_papers = defaultdict(list)
for paper_id, title, abstract, h1 in rows:
    cluster_papers[str(h1)].append((paper_id, title, abstract))

h1_list = sorted(cluster_papers.keys())
print(f"Distinct h1 clusters: {len(h1_list)}")

# Keep only clusters with at least 5 papers
eligible = {h1: papers for h1, papers in cluster_papers.items() if len(papers) >= PAPERS_PER_CLUSTER}
print(f"Clusters with ≥{PAPERS_PER_CLUSTER} papers: {len(eligible)}")

if len(eligible) < NUM_CLUSTERS:
    NUM_CLUSTERS = len(eligible)
    print(f"Only {NUM_CLUSTERS} eligible clusters – evaluating all of them.")

# Randomly select clusters
selected_clusters = random.sample(list(eligible.keys()), NUM_CLUSTERS)

# For each selected cluster, sample 5 papers
evaluation_sets = []
for h1 in selected_clusters:
    pool = eligible[h1]
    sampled = random.sample(pool, PAPERS_PER_CLUSTER)
    evaluation_sets.append((h1, sampled))

print(f"Prepared {len(evaluation_sets)} coherence evaluation tasks.\n")

# ---------- 2. Prompt template from Tan & D'Souza (2025) ----------
system_message = (
    "You are an expert research evaluator. "
    "Your task is to assess the coherence of a group of research paper abstracts, each with its title."
)

def build_prompt(numbered_abstracts):
    return (
        "You will be given a set of {n} abstracts that an algorithmic model has grouped together.\n\n"
        "For this set, determine if the {n} abstracts collectively belong to a single, recognizable "
        "**broad scientific discipline** (e.g., 'Computer Science', 'Medicine', 'Physics', 'Biology', etc.).\n\n"
        "Provide your answer in two parts:\n"
        "1. First, provide a 1-sentence explanation of why the abstracts do or do not form a coherent discipline-level group.\n"
        "2. Then, state your coherence rating on a 5-point Likert scale, where:\n"
        "   - 1 = Completely incoherent (no shared broad discipline)\n"
        "   - 2 = Somewhat incoherent\n"
        "   - 3 = Moderately coherent\n"
        "   - 4 = Very coherent\n"
        "   - 5 = Highly coherent (all clearly belong to the same broad discipline)\n\n"
        "Provide the final rating exactly in the format: `Rating: X`.\n\n"
        "---\n"
        "Abstracts to Evaluate:\n"
        "{numbered}"
    ).format(n=PAPERS_PER_CLUSTER, numbered=numbered_abstracts)

# ---------- 3. Run evaluation ----------
results = []
total_cost = 0.0

for idx, (h1, abstracts) in enumerate(evaluation_sets, start=1):
    numbered = ""
    for i, (paper_id, title, abstract) in enumerate(abstracts, start=1):
        truncated = truncate_words(abstract, MAX_WORDS)
        numbered += f"{i}. Title: {title}\n   Abstract: {truncated}\n\n"

    prompt = build_prompt(numbered)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=200
        )
        full_answer = response.choices[0].message.content.strip()
        rating = extract_rating(full_answer)

        usage = response.usage
        cost = (usage.prompt_tokens * 0.14 + usage.completion_tokens * 0.28) / 1_000_000
        total_cost += cost

        results.append((h1, rating, cost, full_answer))
        print(f"Cluster {idx}/{NUM_CLUSTERS}: h1={h1}  Rating={rating}  (cost ${cost:.5f})")
        print(f"  Explanation: {full_answer.split('Rating:')[0].strip() if 'Rating:' in full_answer else full_answer[:100]}")

    except Exception as e:
        results.append((h1, None, 0.0, str(e)))
        print(f"Cluster {idx}: h1={h1}  ERROR: {e}")

# ---------- 4. Summary ----------
valid_ratings = [r for (_, r, _, _) in results if r is not None]
if valid_ratings:
    mean_rating = sum(valid_ratings) / len(valid_ratings)
    min_rating = min(valid_ratings)
    max_rating = max(valid_ratings)
else:
    mean_rating = min_rating = max_rating = None

summary = (
    f"\n{'='*60}\n"
    f"H1 COHERENCE RATING SUMMARY\n"
    f"Clusters evaluated: {len(results)}\n"
    f"Valid ratings: {len(valid_ratings)}\n"
    f"Mean coherence: {mean_rating:.2f}\n"
    f"Min / Max: {min_rating} / {max_rating}\n"
    f"Total cost: ${total_cost:.6f}\n"
    f"{'='*60}\n"
)
print(summary)

# ---------- 5. Save report ----------
out_path = os.path.join(RESULTS_DIR, "coherence_h1_TanDSouza.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("H1 COHERENCE RATING – Tan & D'Souza (2025) prompt\n")
    f.write(f"Model: {MODEL}, truncation: first {MAX_WORDS} words\n")
    f.write(f"Clusters: {NUM_CLUSTERS} out of {len(eligible)}\n")
    f.write(summary)
    f.write("\n--- Detailed Ratings ---\n\n")
    for (h1, rating, cost, answer) in results:
        f.write(f"Cluster h1={h1}: Rating={rating}\n")
        f.write(f"Model response:\n{answer}\n")
        # Show the abstracts evaluated
        for i, (paper_id, title, abstract) in enumerate(evaluation_sets[list(selected_clusters).index(h1)][1], start=1):
            f.write(f"  {i}. ID={paper_id}\n     Title: {title}\n     Abstract: {truncate_words(abstract, MAX_WORDS)}\n")
        f.write("-"*60 + "\n")

print(f"Detailed report saved to {out_path}")
conn.close()