import sqlite3
import random
import os
from openai import OpenAI

# ================= CONFIGURATION =================
# Project root = two levels up from this file (code/<group>/<script>.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "merged_works_labeled.db")
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"                   # Best for classification / instruction following
MAX_WORDS_ABSTRACT = 200
TEST_PANELS = 10                         # 5 panels × 2 clusters
# =================================================

# Connect to DeepSeek
if not API_KEY:
    raise SystemExit(
        "Set DEEPSEEK_API_KEY, e.g.\n"
        "  PowerShell: $env:DEEPSEEK_API_KEY='sk-...'"
    )

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def truncate_abstract(text, max_words):
    """Truncate abstract to the first max_words words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def extract_intruder_number(response_text):
    """Extract the first integer from the model's answer."""
    import re
    nums = re.findall(r'\d+', response_text)
    if nums:
        return int(nums[0])
    return None


# ---------- 1. Connect to DB and prepare ----------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create the hierarchical cluster ID
cursor.execute("""
    SELECT id, cleaned_abstract, 
           CAST(h1_cluster AS TEXT) || CAST(h2_cluster AS TEXT) || CAST(h3_cluster AS TEXT) AS cluster_id
    FROM works_labeled
    WHERE cleaned_abstract IS NOT NULL AND trim(cleaned_abstract) != ''
""")

rows = cursor.fetchall()
print(f"Total valid papers loaded: {len(rows)}")

# Group by cluster_id
from collections import defaultdict
cluster_papers = defaultdict(list)
for paper_id, abstract, cluster_id in rows:
    cluster_papers[cluster_id].append((paper_id, abstract))

# Keep only clusters that have at least 4 papers (to form a panel)
min_papers_per_cluster = 4
eligible_clusters = {cid: papers for cid, papers in cluster_papers.items()
                     if len(papers) >= min_papers_per_cluster}
print(f"Clusters with ≥{min_papers_per_cluster} papers: {len(eligible_clusters)}")

if len(eligible_clusters) < 3:
    raise ValueError("Not enough clusters with sufficient papers. At least 3 required.")

# ---------- 2. Pick 3 clusters for the test ----------
selected_clusters = random.sample(list(eligible_clusters.keys()), 3)
cluster_A, cluster_B, cluster_C = selected_clusters
print(f"\nSelected clusters: A={cluster_A} ({len(eligible_clusters[cluster_A])} papers), "
      f"B={cluster_B} ({len(eligible_clusters[cluster_B])} papers), "
      f"C={cluster_C} ({len(eligible_clusters[cluster_C])} papers)")

# We'll build 5 panels for cluster_A and 5 for cluster_B
# Intruders are drawn from the other two clusters (not the target)

def build_panels(target_cluster, other_clusters, num_panels, cluster_pools):
    """
    target_cluster: the cluster for which we create panels (home papers)
    other_clusters: list of clusters that provide intruders
    num_panels: how many panels to build
    cluster_pools: dict cluster_id -> list of (paper_id, abstract)
    Returns list of panels, each panel is a dict:
        { 'home_cluster': target_cluster,
          'abstracts': [ ... 5 tuples (paper_id, text, is_intruder) ... ],
          'intruder_position': (1-indexed) }
    """
    home_papers = cluster_pools[target_cluster].copy()
    random.shuffle(home_papers)
    # Ensure we have enough distinct papers for all panels (4 distinct per panel)
    if len(home_papers) < 4 * num_panels:
        raise ValueError(f"Not enough papers in cluster {target_cluster} for {num_panels} panels.")

    # Prepare intruder pool from other clusters
    intruder_pool = []
    for cid in other_clusters:
        intruder_pool.extend(cluster_pools[cid])
    random.shuffle(intruder_pool)

    panels = []
    used_papers = set()   # track paper_ids to avoid duplicates across panels

    for i in range(num_panels):
        # Choose 4 home papers not used before
        home_sample = []
        for pid, txt in home_papers:
            if pid not in used_papers:
                home_sample.append((pid, txt))
                if len(home_sample) == 4:
                    break
        if len(home_sample) < 4:
            raise RuntimeError(f"Couldn't find 4 unused papers for panel {i+1} in cluster {target_cluster}.")
        # Mark them as used
        for pid, _ in home_sample:
            used_papers.add(pid)

        # Choose an intruder not used before and from a different cluster
        intruder = None
        for pid, txt in intruder_pool:
            if pid not in used_papers:
                intruder = (pid, txt)
                used_papers.add(pid)
                break
        if intruder is None:
            raise RuntimeError("No unused intruder paper available.")

        # Combine: 4 home + 1 intruder, then randomise order
        combined = [(pid, txt, False) for pid, txt in home_sample] + \
                   [(intruder[0], intruder[1], True)]
        random.shuffle(combined)  # the intruder's new position is random

        # Find intruder position (1-indexed)
        intruder_pos = next(i+1 for i, (_, _, is_intruder) in enumerate(combined) if is_intruder)

        panel = {
            'home_cluster': target_cluster,
            'abstracts': combined,   # list of (paper_id, text, is_intruder)
            'intruder_position': intruder_pos
        }
        panels.append(panel)

    return panels


# Build 5 panels for cluster A (intruders from B and C) and 5 for cluster B (intruders from A and C)
panels_A = build_panels(cluster_A, [cluster_B, cluster_C], 5, eligible_clusters)
panels_B = build_panels(cluster_B, [cluster_A, cluster_C], 5, eligible_clusters)

all_panels = panels_A + panels_B   # 10 panels
random.shuffle(all_panels)         # optional: present in random order for evaluation

print(f"\nBuilt {len(all_panels)} intrusion detection panels.\n")

# ---------- 3. Run intrusion detection via DeepSeek ----------
output_lines = []
total_cost = 0.0

for idx, panel in enumerate(all_panels, start=1):
    home_cluster = panel['home_cluster']
    intruder_pos_true = panel['intruder_position']

    # Prepare numbered abstracts (truncated)
    numbered = []
    for i, (paper_id, abstract, is_intruder) in enumerate(panel['abstracts'], start=1):
        truncated = truncate_abstract(abstract, MAX_WORDS_ABSTRACT)
        numbered.append(f"Abstract {i}:\n{truncated}\n")

    prompt = (
        "You are an expert in AI research. I will show you 5 paper abstracts. "
        "4 of them belong to the exact same AI subfield. 1 abstract does not belong. "
        "Identify the intruder abstract. Answer ONLY with the number of the intruder (e.g., \"3\").\n\n"
        + "\n".join(numbered)
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10   # we only need a short answer
        )

        answer = response.choices[0].message.content.strip()
        pred_number = extract_intruder_number(answer)
        is_correct = (pred_number == intruder_pos_true)

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        # Pricing for deepseek-chat: $0.14 / 1M input, $0.28 / 1M output (check exact rates; I'll use approximate)
        cost = (prompt_tokens * 0.14 + completion_tokens * 0.28) / 1_000_000
        total_cost += cost

        result_str = (f"Panel {idx} | Cluster: {home_cluster}\n"
                      f"True intruder position: {intruder_pos_true}\n"
                      f"Model answer: '{answer}' → Predicted: {pred_number}\n"
                      f"Correct: {is_correct}\n"
                      f"Tokens: prompt={prompt_tokens}, completion={completion_tokens}, cost=${cost:.6f}\n"
                      f"{'='*50}\n")
    except Exception as e:
        result_str = (f"Panel {idx} | Cluster: {home_cluster}\n"
                      f"ERROR: {e}\n{'='*50}\n")
        is_correct = None

    print(result_str)
    output_lines.append(result_str)
    # Also save detailed panel info to file later

# ---------- 4. Write detailed results to file ----------
output_path = os.path.join(os.path.dirname(DB_PATH), "intrusion_test_results.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("INTRUSION DETECTION TEST – DETAILED RESULTS\n")
    f.write("Model: deepseek-chat\n")
    f.write(f"Total panels: {len(all_panels)}\n")
    f.write(f"Total estimated cost: ${total_cost:.6f}\n\n")
    for idx, panel in enumerate(all_panels, start=1):
        home_cluster = panel['home_cluster']
        intruder_pos_true = panel['intruder_position']
        f.write(f"Panel {idx} – Home cluster: {home_cluster}\n")
        for i, (paper_id, abstract, is_intruder) in enumerate(panel['abstracts'], start=1):
            marker = " *** INTRUDER ***" if is_intruder else ""
            truncated = truncate_abstract(abstract, MAX_WORDS_ABSTRACT)
            f.write(f"  [{i}] ID={paper_id}{marker}\n      {truncated}\n\n")
        f.write(f"True intruder position: {intruder_pos_true}\n")
        # Retrieve the model response from output_lines if available
        # We'll just re-print the summary we already stored
        f.write(output_lines[idx-1])   # re-use the summary string from earlier
        f.write("\n")

print(f"\nResults saved to {output_path}")
conn.close()