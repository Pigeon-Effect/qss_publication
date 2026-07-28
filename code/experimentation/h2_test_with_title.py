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
TEST_PANELS = 100                         # how many intrusion tasks to run
MAX_WORDS = 100                           # truncate abstracts to first 100 words
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

# ---------- 1. Load data, group by two‑digit cluster (h1_h2) ----------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Now also select the title
cursor.execute("""
    SELECT id, title, cleaned_abstract,
           CAST(h1_cluster AS TEXT) || CAST(h2_cluster AS TEXT) AS cluster_id
    FROM works_labeled
    WHERE cleaned_abstract IS NOT NULL AND trim(cleaned_abstract) != ''
""")
rows = cursor.fetchall()
print(f"Total valid papers: {len(rows)}")

# Store as (paper_id, title, abstract)
cluster_papers = defaultdict(list)
for paper_id, title, abstract, cid in rows:
    cluster_papers[cid].append((paper_id, title, abstract))

print(f"Distinct h2 clusters: {len(cluster_papers)}")

# Keep only clusters with at least 4 papers (needed for home set)
min_home = 4
eligible = {cid: papers for cid, papers in cluster_papers.items() if len(papers) >= min_home}
print(f"Clusters with ≥{min_home} papers: {len(eligible)}")

if len(eligible) < 2:
    raise ValueError("Need at least 2 eligible clusters for intrusion task.")

cluster_ids = list(eligible.keys())

# ---------- 2. Build 100 panels (4 home + 1 intruder) ----------
panels = []
for _ in range(TEST_PANELS):
    # Pick home cluster
    home_cid = random.choice(cluster_ids)
    # Pick an intruder cluster different from home
    intruder_cid = random.choice([c for c in cluster_ids if c != home_cid])

    # Sample 4 home items (each is (pid, title, abstract))
    home_sample = random.sample(eligible[home_cid], 4)
    # Sample 1 intruder item
    intruder_sample = random.choice(eligible[intruder_cid])

    # Combine, shuffle, mark intruder
    combined = []
    for pid, title, abstract in home_sample:
        combined.append((pid, title, truncate_words(abstract, MAX_WORDS), False))
    pid_i, title_i, abstract_i = intruder_sample
    combined.append((pid_i, title_i, truncate_words(abstract_i, MAX_WORDS), True))
    random.shuffle(combined)

    intruder_position = next(i+1 for i, (_, _, _, is_intruder) in enumerate(combined) if is_intruder)

    panels.append({
        'home_cluster': home_cid,
        'intruder_cluster': intruder_cid,
        'items': combined,    # (paper_id, title, abstract, is_intruder)
        'intruder_position': intruder_position
    })

print(f"Built {len(panels)} panels.")

# ---------- 3. Run intrusion detection ----------
output_lines = []
total_cost = 0.0
correct_count = 0

for idx, panel in enumerate(panels, start=1):
    home = panel['home_cluster']
    intruder = panel['intruder_cluster']
    true_pos = panel['intruder_position']

    numbered = []
    for i, (pid, title, abstract, is_intr) in enumerate(panel['items'], start=1):
        # Format: Title: ... \n Abstract: ...
        numbered.append(f"Abstract {i}:\nTitle: {title}\nAbstract: {abstract}\n")

    prompt = (
        "You are a research librarian. Below are 5 paper abstracts (each with title).\n"
        "Four of them belong to the same narrow scientific subfield. "
        "One abstract is from a different subfield and does not belong.\n"
        "Identify the intruder abstract.\n"
        "Answer with only the number (1-5) of the intruder.\n\n"
        + "\n".join(numbered)
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5
        )
        answer = response.choices[0].message.content.strip()
        pred = extract_intruder_number(answer)
        is_correct = (pred == true_pos)
        if is_correct:
            correct_count += 1

        usage = response.usage
        cost = (usage.prompt_tokens * 0.14 + usage.completion_tokens * 0.28) / 1_000_000
        total_cost += cost

        out_str = (
            f"Panel {idx} | Home: {home}, Intruder: {intruder}\n"
            f"True intruder: {true_pos}\n"
            f"Model answer: '{answer}' → Predicted: {pred}\n"
            f"Correct: {is_correct}\n"
            f"Tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, cost=${cost:.6f}\n"
            f"{'='*60}\n"
        )
    except Exception as e:
        out_str = f"Panel {idx} | Home: {home}\nERROR: {e}\n{'='*60}\n"
        is_correct = None

    print(out_str)
    output_lines.append(out_str)

accuracy = correct_count / TEST_PANELS * 100 if TEST_PANELS else 0
summary = (
    f"\nFINAL RESULTS (h2 clusters, 4+1, with titles, no cross‑h1, 100 words)\n"
    f"Total panels: {TEST_PANELS}\n"
    f"Correct: {correct_count}\n"
    f"Accuracy: {accuracy:.1f}%\n"
    f"Total cost: ${total_cost:.6f}\n"
)
print(summary)

# ---------- 4. Save report ----------
out_path = os.path.join(os.path.dirname(DB_PATH), "intrusion_h2_with_titles_results.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("INTRUSION DETECTION – H2 CLUSTERS (4 HOME + 1 INTRUDER, WITH TITLES)\n")
    f.write(f"Model: {MODEL}, truncation: {MAX_WORDS} words\n")
    f.write(f"Panels: {TEST_PANELS}, Accuracy: {accuracy:.1f}%\n\n")
    for idx, panel in enumerate(panels, start=1):
        f.write(f"Panel {idx}\n")
        f.write(f"Home cluster: {panel['home_cluster']}, Intruder from: {panel['intruder_cluster']}\n")
        for i, (pid, title, abstract, is_intr) in enumerate(panel['items'], start=1):
            marker = " *** INTRUDER ***" if is_intr else ""
            f.write(f"  [{i}] ID={pid}{marker}\n      Title: {title}\n      Abstract: {abstract}\n\n")
        f.write(f"True intruder position: {panel['intruder_position']}\n")
        f.write(output_lines[idx-1] + "\n" + "="*70 + "\n")

print(f"Report saved to {out_path}")
conn.close()