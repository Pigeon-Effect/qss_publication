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

# Use the model that the API accepts
MODEL = "deepseek-v4-flash"  # or "deepseek-v4-pro" if needed
TEMPERATURE = 0.2  # slight variation for better reasoning
NUM_TESTS = 100
MAX_WORDS = 500  # longer context
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


def extract_intruder_number(text):
    """Extract the final number from the model's response."""
    if not text:
        return None
    patterns = [
        r'final answer\s*:\s*([1-5])',
        r'verdict\s*:\s*([1-5])',
        r'^([1-5])\s*$',
        r'\b([1-5])\s*$',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return int(m.group(1))
    # fallback: last digit
    digits = re.findall(r'[1-5]', text)
    if digits:
        return int(digits[-1])
    return None


def build_prompt(panel):
    numbered = []
    for i, (pid, title, abstract, _) in enumerate(panel, start=1):
        numbered.append(f"[{i}] Title: {title}\nAbstract: {abstract}")
    prompt = (
            "You are a research expert. Below are 5 paper abstracts with titles.\n"
            "Four belong to the SAME research subfield. One is from a DIFFERENT subfield.\n\n"
            + "\n\n".join(numbered) +
            "\n\nThink step by step: identify the common subfield and which abstract is the odd one out.\n"
            "Then output ONLY the number of the intruder on the last line, preceded by 'FINAL ANSWER: '.\n"
            "Do not add any extra commentary after that line."
    )
    return prompt


print("Loading data from database...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT id, title, cleaned_abstract,
           CAST(h1_cluster AS TEXT) || CAST(h2_cluster AS TEXT) || CAST(h3_cluster AS TEXT) AS cluster_id
    FROM works_labeled
    WHERE cleaned_abstract IS NOT NULL AND trim(cleaned_abstract) != ''
""")
rows = cursor.fetchall()
print(f"Total papers: {len(rows)}")

cluster_papers = defaultdict(list)
for paper_id, title, abstract, cid in rows:
    cluster_papers[cid].append((paper_id, title, abstract))

eligible_clusters = {cid: papers for cid, papers in cluster_papers.items() if len(papers) >= 5}
print(f"Clusters with ≥5 papers: {len(eligible_clusters)}")

if len(eligible_clusters) < 2:
    raise ValueError("Need at least 2 clusters with 5+ papers")

cluster_ids = list(eligible_clusters.keys())

# ================= GENERATE TEST PANELS =================
print(f"Generating {NUM_TESTS} test panels...")
all_panels = []
for _ in range(NUM_TESTS):
    home_cluster = random.choice(cluster_ids)
    intruder_cluster = random.choice([c for c in cluster_ids if c != home_cluster])

    home_papers = random.sample(eligible_clusters[home_cluster], 4)
    intruder_paper = random.choice(eligible_clusters[intruder_cluster])

    panel = []
    for pid, title, abstract in home_papers:
        panel.append((pid, title, truncate_words(abstract, MAX_WORDS), False))
    pid_i, title_i, abstract_i = intruder_paper
    panel.append((pid_i, title_i, truncate_words(abstract_i, MAX_WORDS), True))

    random.shuffle(panel)
    true_position = next(i + 1 for i, (_, _, _, is_intr) in enumerate(panel) if is_intr)

    all_panels.append({
        'home_cluster': home_cluster,
        'intruder_cluster': intruder_cluster,
        'panel': panel,
        'true_position': true_position
    })

# ================= RUN TESTS =================
correct = 0
total_cost = 0.0
detailed_results = []

print(f"\nRunning {NUM_TESTS} intrusion tests with {MODEL} (temp={TEMPERATURE}, max_words={MAX_WORDS})...\n")

for test_num, panel_data in enumerate(all_panels, start=1):
    home_cluster = panel_data['home_cluster']
    intruder_cluster = panel_data['intruder_cluster']
    true_position = panel_data['true_position']
    panel = panel_data['panel']

    prompt = build_prompt(panel)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=200,
            timeout=30
        )
        output = response.choices[0].message.content.strip()
        predicted = extract_intruder_number(output)
        is_correct = (predicted == true_position)
        if is_correct:
            correct += 1

        usage = response.usage
        cost = (usage.prompt_tokens * 0.00000014 + usage.completion_tokens * 0.00000028)
        total_cost += cost

        detailed_results.append({
            'test_num': test_num,
            'home_cluster': home_cluster,
            'intruder_cluster': intruder_cluster,
            'true_position': true_position,
            'predicted': predicted,
            'correct': is_correct,
            'cost': cost,
            'model_output': output,
            'panel': panel
        })

        print(
            f"Test {test_num:3d} | Home:{home_cluster} | Intruder:{intruder_cluster} | True:{true_position} | Pred:{predicted} | {'✓' if is_correct else '✗'}")

    except Exception as e:
        print(f"Test {test_num:3d} | ERROR: {e}")
        detailed_results.append({
            'test_num': test_num,
            'error': str(e),
            'panel': panel
        })

# ================= SUMMARY =================
accuracy = (correct / NUM_TESTS) * 100 if NUM_TESTS > 0 else 0
print(f"\n{'=' * 60}")
print(f"RESULTS for {MODEL}")
print(f"Temperature: {TEMPERATURE}")
print(f"Max words: {MAX_WORDS}")
print(f"Tests: {NUM_TESTS}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.1f}%")
print(f"Total cost: ${total_cost:.6f}")
print(f"{'=' * 60}")

# ================= SAVE DETAILED REPORT =================
output_path = os.path.join(RESULTS_DIR, "intrusion_v4flash_500words_temp02.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"INTRUSION DETECTION - {MODEL}\n")
    f.write(f"Temperature: {TEMPERATURE}\n")
    f.write(f"Max words: {MAX_WORDS}\n")
    f.write(f"Tests: {NUM_TESTS}\n")
    f.write(f"Accuracy: {accuracy:.1f}%\n")
    f.write(f"Total cost: ${total_cost:.6f}\n\n")
    f.write("=" * 70 + "\n\n")

    for res in detailed_results:
        f.write(f"Test {res['test_num']}\n")
        f.write(f"Home cluster: {res.get('home_cluster', 'N/A')}\n")
        f.write(f"Intruder cluster: {res.get('intruder_cluster', 'N/A')}\n")
        f.write(f"True intruder position: {res.get('true_position', 'N/A')}\n")
        f.write(f"Predicted: {res.get('predicted', 'ERROR')}\n")
        f.write(f"Correct: {res.get('correct', False)}\n")
        if 'cost' in res:
            f.write(f"Cost: ${res['cost']:.6f}\n")
        if 'model_output' in res:
            f.write(f"\nModel output:\n{res['model_output']}\n")
        if 'error' in res:
            f.write(f"ERROR: {res['error']}\n")
        f.write("\nPanel details:\n")
        for i, (pid, title, abstract, is_intr) in enumerate(res['panel'], start=1):
            marker = " *** INTRUDER ***" if is_intr else ""
            f.write(f"  [{i}] ID={pid}{marker}\n")
            f.write(f"      Title: {title}\n")
            f.write(f"      Abstract: {abstract[:500]}\n\n")
        f.write("-" * 70 + "\n\n")

print(f"\nDetailed report saved to: {output_path}")
conn.close()