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
NUM_CLUSTERS = 100  # number of h3 clusters to evaluate
PAPERS_PER_CLUSTER = 5  # abstracts per evaluation
MAX_WORDS = 500  # truncate abstracts to first 100 words
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


def extract_scores(text):
    """Extract topic and methodology scores from the LLM response."""
    topic_match = re.search(r'Topic Coherence:\s*(\d)', text, re.IGNORECASE)
    method_match = re.search(r'Methodology Coherence:\s*(\d)', text, re.IGNORECASE)
    topic_score = int(topic_match.group(1)) if topic_match else None
    method_score = int(method_match.group(1)) if method_match else None
    return topic_score, method_score


# ---------- 1. Load data, group by three‑digit cluster (h1h2h3) ----------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
               SELECT id,
                      title,
                      cleaned_abstract,
                      CAST(h1_cluster AS TEXT) || CAST(h2_cluster AS TEXT) || CAST(h3_cluster AS TEXT) AS cluster_id
               FROM works_labeled
               WHERE cleaned_abstract IS NOT NULL
                 AND trim(cleaned_abstract) != ''
               """)
rows = cursor.fetchall()
print(f"Total valid papers: {len(rows)}")

cluster_papers = defaultdict(list)
for paper_id, title, abstract, cid in rows:
    cluster_papers[cid].append((paper_id, title, abstract))

h3_list = sorted(cluster_papers.keys())
print(f"Distinct h3 clusters: {len(h3_list)}")

# Keep only clusters with at least 5 papers
eligible = {cid: papers for cid, papers in cluster_papers.items() if len(papers) >= PAPERS_PER_CLUSTER}
print(f"Clusters with ≥{PAPERS_PER_CLUSTER} papers: {len(eligible)}")

if len(eligible) < NUM_CLUSTERS:
    NUM_CLUSTERS = len(eligible)
    print(f"Only {NUM_CLUSTERS} eligible clusters – evaluating all of them.")

selected_clusters = random.sample(list(eligible.keys()), NUM_CLUSTERS)

evaluation_sets = []
for cid in selected_clusters:
    pool = eligible[cid]
    sampled = random.sample(pool, PAPERS_PER_CLUSTER)
    evaluation_sets.append((cid, sampled))

print(f"Prepared {len(evaluation_sets)} coherence evaluation tasks.\n")

# ---------- 2. Simple prompt: two separate scores, no examples, no asterisks ----------
system_message = (
    "You are an expert research evaluator. Your task is to assess how coherent a set of research paper abstracts is."
)


def build_simple_prompt(numbered_abstracts):
    return f"""
You will be given a set of {PAPERS_PER_CLUSTER} abstracts that an algorithm grouped together.

First, reason step by step:

Step 1 – Read each abstract and identify the main research topic (what they study) and the main methodology (how they study it).

Step 2 – For Topic Coherence, evaluate how well these abstracts belong together as a single research subfield. Use this scale:

1 = No coherent topic – abstracts are from completely different major areas.
2 = Weak topic relation – they share only a very broad area like artificial intelligence but no clear subfield.
3 = Moderate topic coherence – they belong to the same general subfield, but within that subfield they cover diverse specific problems.
4 = Strong topic coherence – they consistently focus on a narrower theme within a subfield.
5 = Perfect topic coherence – all abstracts clearly address the same specific research topic.

Step 3 – For Methodology Coherence, evaluate how similar their research methods are. Use this scale:

1 = No shared methodology – methods are completely different.
2 = Weak method overlap – only at the highest abstraction level.
3 = Moderate method coherence – they share a common family of methods, but with meaningful differences.
4 = Strong method coherence – they use the same specific methodology throughout.
5 = Identical methodology – the experimental setups, models, and evaluation procedures are essentially the same.

Step 4 – After your reasoning, output exactly two lines with the final scores:

Topic Coherence: X
Methodology Coherence: Y

where X and Y are numbers from 1 to 5.

Do not add any extra text, explanations, or formatting.

Abstracts to evaluate:
{numbered_abstracts}
"""


# ---------- 3. Run evaluation ----------
results = []  # each element: (cid, topic_score, method_score, cost, raw_response)
total_cost = 0.0

for idx, (cid, abstracts) in enumerate(evaluation_sets, start=1):
    numbered = ""
    for i, (paper_id, title, abstract) in enumerate(abstracts, start=1):
        truncated = truncate_words(abstract, MAX_WORDS)
        numbered += f"{i}. Title: {title}\n   Abstract: {truncated}\n\n"

    prompt = build_simple_prompt(numbered)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100  # enough for just two scores
        )
        raw = response.choices[0].message.content.strip()
        topic_score, method_score = extract_scores(raw)

        usage = response.usage
        cost = (usage.prompt_tokens * 0.14 + usage.completion_tokens * 0.28) / 1_000_000
        total_cost += cost

        results.append((cid, topic_score, method_score, cost, raw))
        print(f"Cluster {idx}/{NUM_CLUSTERS}: h3={cid}  Topic={topic_score}  Method={method_score}  (cost ${cost:.5f})")

    except Exception as e:
        results.append((cid, None, None, 0.0, str(e)))
        print(f"Cluster {idx}: h3={cid}  ERROR: {e}")

# ---------- 4. Summary statistics ----------
valid_topic = [t for (_, t, _, _, _) in results if t is not None]
valid_method = [m for (_, _, m, _, _) in results if m is not None]

if valid_topic:
    mean_topic = sum(valid_topic) / len(valid_topic)
    mean_method = sum(valid_method) / len(valid_method)
else:
    mean_topic = mean_method = None

summary = (
    f"\n{'=' * 60}\n"
    f"H3 COHERENCE EVALUATION (Topic vs. Methodology scores)\n"
    f"Clusters evaluated: {len(results)}\n"
    f"Valid topic scores: {len(valid_topic)}  |  Valid method scores: {len(valid_method)}\n"
    f"Mean topic coherence: {mean_topic:.2f}\n"
    f"Mean methodology coherence: {mean_method:.2f}\n"
    f"Total cost: ${total_cost:.6f}\n"
    f"{'=' * 60}\n"
)
print(summary)

# ---------- 5. Save detailed report ----------
out_path = os.path.join(RESULTS_DIR, "coherence_h3_two_scores.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("H3 CLUSTER VALIDATION – Separate Topic and Methodology Coherence\n")
    f.write(f"Model: {MODEL}, truncation: first {MAX_WORDS} words\n")
    f.write(f"Clusters sampled: {NUM_CLUSTERS} out of {len(eligible)} eligible clusters\n")
    f.write(summary)
    f.write("\n--- Detailed Results per Cluster ---\n\n")

    for idx, (cid, topic, method, cost, raw) in enumerate(results):
        f.write(f"Cluster {idx + 1}: h3={cid}\n")
        f.write(f"Topic Coherence (1-5): {topic}\n")
        f.write(f"Methodology Coherence (1-5): {method}\n")
        f.write(f"Cost: ${cost:.5f}\n")
        f.write(f"Raw LLM response:\n{raw}\n")
        f.write("\nAbstracts used for this cluster:\n")
        # Find the corresponding papers for this cid
        for (cid2, papers) in evaluation_sets:
            if cid2 == cid:
                for i, (pid, title, abstract) in enumerate(papers, start=1):
                    f.write(f"  {i}. ID={pid}\n")
                    f.write(f"     Title: {title}\n")
                    f.write(f"     Abstract: {truncate_words(abstract, MAX_WORDS)}\n")
                break
        f.write("-" * 80 + "\n\n")

print(f"Detailed report saved to {out_path}")
conn.close()