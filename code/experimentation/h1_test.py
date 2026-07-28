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
TEST_PANELS = 100                         # number of intrusion detection runs
# =================================================

if not API_KEY:
    raise SystemExit(
        "Set DEEPSEEK_API_KEY, e.g.\n"
        "  PowerShell: $env:DEEPSEEK_API_KEY='sk-...'"
    )

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def extract_intruder_number(response_text):
    """Extract the first integer from the model's answer (expects just a number)."""
    nums = re.findall(r'\d+', response_text)
    return int(nums[0]) if nums else None

# ---------- 1. Connect to DB and load data ----------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT id, cleaned_abstract, h1_cluster
    FROM works_labeled
    WHERE cleaned_abstract IS NOT NULL AND trim(cleaned_abstract) != ''
""")

rows = cursor.fetchall()
print(f"Total valid papers loaded: {len(rows)}")

# Group papers by h1_cluster
cluster_papers = defaultdict(list)
for paper_id, abstract, h1 in rows:
    cluster_papers[h1].append((paper_id, abstract))

# Identify how many unique h1 clusters exist and list them
unique_h1 = sorted(cluster_papers.keys())
print(f"Number of distinct h1 domains: {len(unique_h1)}")
print(f"Domain IDs: {unique_h1}")

# Keep only clusters with at least 4 papers
min_papers = 4
eligible_clusters = {h1: papers for h1, papers in cluster_papers.items()
                     if len(papers) >= min_papers}
print(f"h1 domains with ≥{min_papers} papers: {len(eligible_clusters)}")
if len(eligible_clusters) < 2:
    raise ValueError("Need at least 2 domains to perform intrusion detection.")

# ---------- 2. Build 100 random panels ----------
cluster_ids = list(eligible_clusters.keys())
panels = []

for _ in range(TEST_PANELS):
    # Select home domain
    home_cluster = random.choice(cluster_ids)
    # Select intruder domain different from home
    possible_intruders = [c for c in cluster_ids if c != home_cluster]
    intruder_cluster = random.choice(possible_intruders)

    # Randomly sample 4 distinct papers from home domain
    home_sample = random.sample(eligible_clusters[home_cluster], min(4, len(eligible_clusters[home_cluster])))
    # Randomly sample 1 intruder paper from intruder domain
    intruder_sample = random.choice(eligible_clusters[intruder_cluster])

    # Combine and shuffle positions
    combined = [(pid, txt, False) for pid, txt in home_sample] + \
               [(intruder_sample[0], intruder_sample[1], True)]
    random.shuffle(combined)
    intruder_position = next(i+1 for i, (_, _, is_intruder) in enumerate(combined) if is_intruder)

    panels.append({
        'home_cluster': home_cluster,
        'intruder_cluster': intruder_cluster,
        'abstracts': combined,  # (paper_id, text, is_intruder)
        'intruder_position': intruder_position
    })

print(f"Built {len(panels)} intrusion panels.")

# ---------- 3. Run intrusion detection ----------
output_lines = []
total_cost = 0.0
correct_count = 0

# Domain-neutral prompt, directly asks for the intruder number
for idx, panel in enumerate(panels, start=1):
    home_cluster = panel['home_cluster']
    intruder_cluster = panel['intruder_cluster']
    intruder_pos_true = panel['intruder_position']

    # Format numbered abstracts (full text)
    numbered = []
    for i, (paper_id, abstract, is_intruder) in enumerate(panel['abstracts'], start=1):
        numbered.append(f"Abstract {i}:\n{abstract}\n")

    prompt = (
        "You are a research librarian.\n"
        "Below are 5 paper abstracts from various scientific fields.\n"
        "4 of them belong to the same broad scientific discipline. "
        "1 abstract is from a completely different discipline.\n"
        "Identify the intruder abstract.\n"
        "Answer with only the number (1-5) of the intruder."
        "\n\n" + "\n".join(numbered)
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5    # only expecting a digit
        )

        full_answer = response.choices[0].message.content.strip()
        pred_number = extract_intruder_number(full_answer)
        is_correct = (pred_number == intruder_pos_true)
        if is_correct:
            correct_count += 1

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        cost = (prompt_tokens * 0.14 + completion_tokens * 0.28) / 1_000_000
        total_cost += cost

        result_str = (
            f"Panel {idx} | Home domain: h1={home_cluster}, Intruder from: h1={intruder_cluster}\n"
            f"True intruder position: {intruder_pos_true}\n"
            f"Model response: '{full_answer}' → Predicted: {pred_number}\n"
            f"Correct: {is_correct}\n"
            f"Tokens: prompt={prompt_tokens}, completion={completion_tokens}, cost=${cost:.6f}\n"
            f"{'='*60}\n"
        )
    except Exception as e:
        result_str = (
            f"Panel {idx} | Home domain: h1={home_cluster}\n"
            f"ERROR: {e}\n{'='*60}\n"
        )
        is_correct = None

    print(result_str)
    output_lines.append(result_str)

accuracy = correct_count / TEST_PANELS * 100 if TEST_PANELS else 0
summary = (
    f"\nFINAL RESULTS\n"
    f"Total panels: {TEST_PANELS}\n"
    f"Correct: {correct_count}\n"
    f"Accuracy: {accuracy:.1f}%\n"
    f"Total estimated cost: ${total_cost:.6f}\n"
)
print(summary)

# ---------- 4. Write detailed report ----------
output_path = os.path.join(os.path.dirname(DB_PATH), "h1_intrusion_multidisciplinary_results.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("H1-LEVEL INTRUSION DETECTION (MULTIDISCIPLINARY DATASET)\n")
    f.write(f"Number of h1 domains: {len(unique_h1)} ({unique_h1})\n")
    f.write(f"Model: {MODEL}\n")
    f.write(f"Total panels: {TEST_PANELS}\n")
    f.write(f"Accuracy: {correct_count}/{TEST_PANELS} = {accuracy:.1f}%\n")
    f.write(f"Total cost: ${total_cost:.6f}\n")
    f.write("Prompt used:\n")
    f.write("---\n")
    f.write("You are a research librarian.\nBelow are 5 paper abstracts from various scientific fields.\n"
            "4 of them belong to the same broad scientific discipline. 1 abstract is from a completely different discipline.\n"
            "Identify the intruder abstract.\nAnswer with only the number (1-5) of the intruder.\n")
    f.write("---\n\n")
    for idx, panel in enumerate(panels, start=1):
        f.write(f"Panel {idx}\n")
        f.write(f"Home domain: h1={panel['home_cluster']}, Intruder from: h1={panel['intruder_cluster']}\n")
        for i, (paper_id, abstract, is_intruder) in enumerate(panel['abstracts'], start=1):
            marker = " *** INTRUDER ***" if is_intruder else ""
            f.write(f"  [{i}] ID={paper_id}{marker}\n      {abstract}\n\n")
        f.write(f"True intruder position: {panel['intruder_position']}\n")
        f.write(f"Model response & result:\n{output_lines[idx-1]}\n")
        f.write("\n" + "="*70 + "\n")

print(f"Detailed results saved to {output_path}")
conn.close()