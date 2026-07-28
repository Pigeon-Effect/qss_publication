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
TEST_PANELS = 100                         # number of intrusion tasks
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
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])

def extract_intruder_number(response_text):
    nums = re.findall(r'\d+', response_text)
    return int(nums[0]) if nums else None

# ---------- 1. Load data and group by three‑digit cluster (h1_h2_h3) ----------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT id, title, cleaned_abstract,
           CAST(h1_cluster AS TEXT) || CAST(h2_cluster AS TEXT) || CAST(h3_cluster AS TEXT) AS cluster_id
    FROM works_labeled
    WHERE cleaned_abstract IS NOT NULL AND trim(cleaned_abstract) != ''
""")
rows = cursor.fetchall()
print(f"Total valid papers loaded: {len(rows)}")

cluster_papers = defaultdict(list)
for paper_id, title, abstract, cid in rows:
    cluster_papers[cid].append((paper_id, title, abstract))

print(f"Distinct h3 clusters: {len(cluster_papers)}")

min_home = 4
eligible = {cid: papers for cid, papers in cluster_papers.items() if len(papers) >= min_home}
print(f"Clusters with ≥{min_home} papers: {len(eligible)}")

if len(eligible) < 2:
    raise ValueError("Need at least 2 eligible clusters for intrusion task.")

cluster_ids = list(eligible.keys())

# ---------- 2. Build panels (4 home + 1 intruder) ----------
panels = []
for _ in range(TEST_PANELS):
    home_cid = random.choice(cluster_ids)
    intruder_cid = random.choice([c for c in cluster_ids if c != home_cid])

    home_sample = random.sample(eligible[home_cid], 4)
    intruder_sample = random.choice(eligible[intruder_cid])

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
        'items': combined,
        'intruder_position': intruder_position
    })

print(f"Built {len(panels)} panels.")

# ---------- 3. Run intrusion detection for TOPIC and METHOD separately ----------
topic_correct = 0
method_correct = 0
total_cost = 0.0

# Store detailed results for each panel
detailed_results = []

for idx, panel in enumerate(panels, start=1):
    home = panel['home_cluster']
    intruder = panel['intruder_cluster']
    true_pos = panel['intruder_position']

    # Build numbered abstracts string
    numbered = []
    for i, (pid, title, abstract, is_intr) in enumerate(panel['items'], start=1):
        numbered.append(f"Abstract {i}:\nTitle: {title}\nAbstract: {abstract}\n")
    abstracts_block = "\n".join(numbered)

    # --- Topic intrusion prompt ---
    topic_prompt = (
        "You are a research librarian. Below are 5 paper abstracts (each with title).\n"
        "Four of them belong to the same narrow research topic. "
        "One abstract is from a different topic and does not belong.\n"
        "Identify the intruder abstract based on its research topic.\n"
        "Answer with only the number (1-5) of the intruder.\n\n"
        + abstracts_block
    )

    # --- Methodology intrusion prompt ---
    method_prompt = (
        "You are a research librarian. Below are 5 paper abstracts (each with title).\n"
        "Four of them use the same or very similar research methodology. "
        "One abstract uses a clearly different methodology and does not belong.\n"
        "Identify the intruder abstract based on its methodology.\n"
        "Answer with only the number (1-5) of the intruder.\n\n"
        + abstracts_block
    )

    try:
        # Topic call
        resp_topic = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": topic_prompt}],
            temperature=0,
            max_tokens=5
        )
        answer_topic = resp_topic.choices[0].message.content.strip()
        pred_topic = extract_intruder_number(answer_topic)
        topic_correct_flag = (pred_topic == true_pos)
        if topic_correct_flag:
            topic_correct += 1

        # Method call
        resp_method = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": method_prompt}],
            temperature=0,
            max_tokens=5
        )
        answer_method = resp_method.choices[0].message.content.strip()
        pred_method = extract_intruder_number(answer_method)
        method_correct_flag = (pred_method == true_pos)
        if method_correct_flag:
            method_correct += 1

        # Cost calculation (sum of both calls)
        usage_t = resp_topic.usage
        usage_m = resp_method.usage
        cost_t = (usage_t.prompt_tokens * 0.14 + usage_t.completion_tokens * 0.28) / 1_000_000
        cost_m = (usage_m.prompt_tokens * 0.14 + usage_m.completion_tokens * 0.28) / 1_000_000
        panel_cost = cost_t + cost_m
        total_cost += panel_cost

        out_str = (
            f"Panel {idx} | Home: {home}, Intruder from: {intruder}\n"
            f"True intruder position: {true_pos}\n"
            f"Topic test: answer='{answer_topic}' → predicted={pred_topic} | Correct={topic_correct_flag}\n"
            f"Method test: answer='{answer_method}' → predicted={pred_method} | Correct={method_correct_flag}\n"
            f"Cost: ${panel_cost:.6f}\n"
            f"{'='*60}\n"
        )
    except Exception as e:
        out_str = f"Panel {idx} | Home: {home}\nERROR: {e}\n{'='*60}\n"
        topic_correct_flag = method_correct_flag = None

    print(out_str)
    detailed_results.append({
        'panel_id': idx,
        'home': home,
        'intruder_from': intruder,
        'true_pos': true_pos,
        'items': panel['items'],
        'topic_answer': answer_topic if 'answer_topic' in locals() else None,
        'topic_pred': pred_topic if 'pred_topic' in locals() else None,
        'topic_correct': topic_correct_flag,
        'method_answer': answer_method if 'answer_method' in locals() else None,
        'method_pred': pred_method if 'pred_method' in locals() else None,
        'method_correct': method_correct_flag,
        'log': out_str
    })

# ---------- 4. Summary ----------
topic_accuracy = (topic_correct / TEST_PANELS) * 100 if TEST_PANELS else 0
method_accuracy = (method_correct / TEST_PANELS) * 100 if TEST_PANELS else 0

summary = (
    f"\n{'='*60}\n"
    f"INTRUSION DETECTION RESULTS (h3 clusters, 4 home + 1 intruder)\n"
    f"Total panels: {TEST_PANELS}\n"
    f"Topic-based intruder detection accuracy: {topic_accuracy:.1f}% ({topic_correct}/{TEST_PANELS})\n"
    f"Methodology-based intruder detection accuracy: {method_accuracy:.1f}% ({method_correct}/{TEST_PANELS})\n"
    f"Total cost: ${total_cost:.6f}\n"
    f"{'='*60}\n"
)
print(summary)

# ---------- 5. Save detailed report ----------
out_path = os.path.join(RESULTS_DIR, "intrusion_h3_topic_method.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("INTRUSION DETECTION – SEPARATE TOPIC AND METHODOLOGY TESTS\n")
    f.write(f"Model: {MODEL}, truncation: first {MAX_WORDS} words\n")
    f.write(f"Panels: {TEST_PANELS}\n")
    f.write(f"Topic accuracy: {topic_accuracy:.1f}%\n")
    f.write(f"Methodology accuracy: {method_accuracy:.1f}%\n")
    f.write(summary)
    f.write("\n--- DETAILED PANEL LOGS ---\n\n")
    for res in detailed_results:
        f.write(res['log'])
        f.write("\nPanel contents (showing intruder):\n")
        for i, (pid, title, abstract, is_intr) in enumerate(res['items'], start=1):
            marker = " *** INTRUDER ***" if is_intr else ""
            f.write(f"  [{i}] ID={pid}{marker}\n")
            f.write(f"      Title: {title}\n")
            f.write(f"      Abstract: {abstract}\n\n")
        f.write("-" * 80 + "\n\n")

print(f"Detailed results saved to {out_path}")
conn.close()