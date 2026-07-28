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
TEST_PANELS = 100
MAX_WORDS = 150                           # truncate abstracts to first 150 words
# =================================================

if not API_KEY:
    raise SystemExit(
        "Set DEEPSEEK_API_KEY, e.g.\n"
        "  PowerShell: $env:DEEPSEEK_API_KEY='sk-...'"
    )

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def truncate_words(text, max_words):
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text

def extract_intruder_number(response_text):
    nums = re.findall(r'\d+', response_text)
    return int(nums[0]) if nums else None

# ---------- 1. Load data ----------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT id, cleaned_abstract,
           CAST(h1_cluster AS TEXT) || CAST(h2_cluster AS TEXT) || CAST(h3_cluster AS TEXT) AS cluster_id
    FROM works_labeled
    WHERE cleaned_abstract IS NOT NULL AND trim(cleaned_abstract) != ''
""")
rows = cursor.fetchall()
print(f"Total valid papers: {len(rows)}")

cluster_papers = defaultdict(list)
for paper_id, abstract, cid in rows:
    cluster_papers[cid].append((paper_id, abstract))

# Only clusters with at least 2 papers (for 2 home abstracts)
eligible = {c: papers for c, papers in cluster_papers.items() if len(papers) >= 2}
print(f"h3 clusters with ≥2 papers: {len(eligible)}")

# Build mapping from cluster ID to its h1 (first digit)
def h1_of(cluster_id):
    return cluster_id[0]   # first character of the three‑digit string

# Group eligible cluster IDs by their h1
h1_to_clusters = defaultdict(list)
for cid in eligible:
    h1_to_clusters[h1_of(cid)].append(cid)

# Ensure we have at least 2 different h1 groups
if len(h1_to_clusters) < 2:
    raise ValueError("Need at least 2 distinct h1 domains for cross‑h1 intruders.")

# ---------- 2. Build panels (2 home + 1 cross‑h1 intruder) ----------
panels = []
for _ in range(TEST_PANELS):
    # 1. Pick a home cluster
    home_cluster = random.choice(list(eligible.keys()))
    home_h1 = h1_of(home_cluster)

    # 2. Pick an intruder cluster from a DIFFERENT h1
    other_h1 = random.choice([h for h in h1_to_clusters.keys() if h != home_h1])
    intruder_cluster = random.choice(h1_to_clusters[other_h1])

    # 3. Sample 2 home abstracts
    home_pool = eligible[home_cluster]
    home_sample = random.sample(home_pool, min(2, len(home_pool)))

    # 4. Sample 1 intruder abstract
    intruder_sample = random.choice(eligible[intruder_cluster])

    # 5. Combine, shuffle, record intruder position
    combined = [(pid, truncate_words(txt, MAX_WORDS), False) for pid, txt in home_sample] + \
               [(intruder_sample[0], truncate_words(intruder_sample[1], MAX_WORDS), True)]
    random.shuffle(combined)
    intruder_pos = next(i+1 for i, (_, _, is_intruder) in enumerate(combined) if is_intruder)

    panels.append({
        'home_cluster': home_cluster,
        'intruder_cluster': intruder_cluster,
        'abstracts': combined,
        'intruder_position': intruder_pos
    })

print(f"Built {len(panels)} panels (2 home + 1 cross‑h1 intruder).")

# ---------- 3. Run intrusion detection ----------
output_lines = []
total_cost = 0.0
correct = 0

for idx, panel in enumerate(panels, start=1):
    home = panel['home_cluster']
    intruder = panel['intruder_cluster']
    true_pos = panel['intruder_position']

    numbered = []
    for i, (pid, abstract, is_intr) in enumerate(panel['abstracts'], start=1):
        numbered.append(f"Abstract {i}:\n{abstract}\n")

    prompt = (
        "You are a research librarian. Below are 3 paper abstracts.\n"
        "Two of them belong to the same narrow scientific subfield. "
        "The third is from a completely different discipline.\n"
        "First, say what the common subfield of the two is (one sentence). "
        "Then write 'Intruder: X' where X is the number of the abstract that does not belong.\n\n"
        + "\n".join(numbered)
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=80
        )
        full = resp.choices[0].message.content.strip()
        pred = extract_intruder_number(full)
        ok = (pred == true_pos)
        if ok:
            correct += 1

        usage = resp.usage
        cost = (usage.prompt_tokens * 0.14 + usage.completion_tokens * 0.28) / 1e6
        total_cost += cost

        out = (
            f"Panel {idx} | Home: {home} (h1={home[0]}), Intruder: {intruder} (h1={intruder[0]})\n"
            f"True intruder: {true_pos}\n"
            f"Model output: {full}\n"
            f"Predicted: {pred}, Correct: {ok}\n"
            f"Tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, cost=${cost:.6f}\n"
            f"{'='*60}\n"
        )
    except Exception as e:
        out = f"Panel {idx} | Home: {home}\nERROR: {e}\n{'='*60}\n"
        ok = None

    print(out)
    output_lines.append(out)

acc = correct / TEST_PANELS * 100
summary = (
    f"\nFINAL RESULTS (cross‑h1, 2+1, truncated abstracts)\n"
    f"Total panels: {TEST_PANELS}\n"
    f"Correct: {correct}\n"
    f"Accuracy: {acc:.1f}%\n"
    f"Total cost: ${total_cost:.6f}\n"
)
print(summary)

# ---------- 4. Save report ----------
out_path = os.path.join(os.path.dirname(DB_PATH), "friendly_intrusion_h3_crossh1.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("FRIENDLY INTRUSION TEST (h3 clusters, cross‑h1 intruders, 2+1)\n")
    f.write(f"Model: {MODEL}, truncation: {MAX_WORDS} words\n")
    f.write(f"Panels: {TEST_PANELS}, Accuracy: {acc:.1f}%\n\n")
    for idx, panel in enumerate(panels, start=1):
        f.write(f"Panel {idx}\n")
        f.write(f"Home: {panel['home_cluster']} (h1={panel['home_cluster'][0]}), "
                f"Intruder: {panel['intruder_cluster']} (h1={panel['intruder_cluster'][0]})\n")
        for i, (pid, abstract, is_intr) in enumerate(panel['abstracts'], start=1):
            marker = " *** INTRUDER ***" if is_intr else ""
            f.write(f"  [{i}] ID={pid}{marker}\n      {abstract}\n\n")
        f.write(f"True intruder position: {panel['intruder_position']}\n")
        f.write(output_lines[idx-1] + "\n" + "="*70 + "\n")

print(f"Report saved to {out_path}")
conn.close()