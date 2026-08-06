"""Multi-n-gram keyword extraction over the AI survey papers (manuscript 2.2).

Produces the raw KeyBERT candidate terms that manual curation then reduces to
the 279-term search vocabulary used by stage 02.
"""

import os
import time
from pathlib import Path
from keybert import KeyBERT
import pandas as pd
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
# Derived from this file's location, so the script runs from any checkout.
# .../code/01_keyword_construction/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Plain-text conversions of the collected survey papers, one .txt per paper.
# Not redistributed (third-party copyright); see this stage's README.
TXT_FOLDER = Path(
    os.environ.get(
        "QSS_SURVEY_TXT_DIR",
        PROJECT_ROOT / "data" / "interim" / "ai_discipline_surveys_txt",
    )
)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

print("Loading KeyBERT model...")
kw_model = KeyBERT(model='all-MiniLM-L6-v2')
print("Model loaded.")

if not TXT_FOLDER.is_dir():
    raise SystemExit(
        f"Survey-paper text folder not found: {TXT_FOLDER}\n"
        "Point QSS_SURVEY_TXT_DIR at it, or see this stage's README."
    )
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

txt_folder = str(TXT_FOLDER)
txt_files = [f for f in os.listdir(txt_folder) if f.endswith(".txt")]

# Limit to first 5 for test: txt_files = txt_files[:5]


# Common settings
max_text_length = 150000
min_text_length = 200

keyword_results = []

for filename in tqdm(txt_files, desc="Processing files"):
    file_path = os.path.join(txt_folder, filename)
    start_time = time.time()

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        if len(text) < min_text_length:
            raise ValueError(f"Text too short ({len(text)} chars)")

        original_length = len(text)
        if len(text) > max_text_length:
            text = text[:max_text_length]
            print(f"Trimmed {filename} from {original_length:,} to {max_text_length:,} chars")

        keywords = []

        # 1-word keywords
        kws_1 = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 1),
            stop_words='english',
            top_n=5,
            use_mmr=False
        )
        keywords.extend([kw[0] for kw in kws_1])

        # 2-word keywords
        kws_2 = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(2, 2),
            stop_words='english',
            top_n=10,
            use_mmr=False
        )
        keywords.extend([kw[0] for kw in kws_2])

        # 3-word keywords (with MMR for variety)
        kws_3 = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(3, 3),
            stop_words='english',
            top_n=5,
            use_mmr=True,
            diversity=0.3,
            nr_candidates=15
        )
        keywords.extend([kw[0] for kw in kws_3])

        # 4-word keywords (with MMR for variety)
        kws_4 = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(4, 4),
            stop_words='english',
            top_n=5,
            use_mmr=True,
            diversity=0.3,
            nr_candidates=15
        )
        keywords.extend([kw[0] for kw in kws_4])

        keyword_results.append({
            "file": filename,
            "keywords": keywords,
            "time_sec": round(time.time() - start_time, 1),
            "text_length": original_length
        })
        print(f"{filename}: {len(keywords)} keywords in {keyword_results[-1]['time_sec']}s")
        print("  Mix:", [len(kw.split()) for kw in keywords])

    except Exception as e:
        print(f"ERROR in {filename}: {str(e)}")
        keyword_results.append({
            "file": filename,
            "keywords": [f"ERROR: {str(e)}"],
            "time_sec": round(time.time() - start_time, 1),
            "text_length": original_length if 'original_length' in locals() else 0
        })

# Save results
timestamp = time.strftime("%Y%m%d_%H%M%S")
csv_output = os.path.join(OUTPUT_DIR, f"keywords_mixed_{timestamp}.csv")
df = pd.DataFrame(keyword_results)
df.to_csv(csv_output, index=False)

# Report
success_count = len([x for x in keyword_results if not str(x['keywords'][0]).startswith('ERROR')])
avg_time = df.loc[df['keywords'].apply(lambda x: not str(x[0]).startswith('ERROR')), 'time_sec'].mean()

print(f"\nCompleted processing {len(txt_files)} files")
print(f"Success rate: {success_count}/{len(txt_files)} ({success_count / len(txt_files):.1%})")
print(f"Average processing time: {avg_time:.1f}s per file")
print(f"Results saved to: {csv_output}")
print("\nTop 5 longest processing times:")
print(df.nlargest(5, 'time_sec')[['file', 'time_sec', 'text_length']].to_string(index=False))