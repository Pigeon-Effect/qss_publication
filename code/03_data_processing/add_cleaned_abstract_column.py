"""Reconstruct plain-text abstracts from OpenAlex inverted indices.

First of the three quality-control steps in manuscript section 2.3. Reads the
deduplicated raw corpus and writes a copy carrying a `cleaned_abstract` column.
"""

import os
import sqlite3
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# === CONFIG ===
# Paths are derived from this file's location, so the script runs from any
# checkout. .../code/03_data_processing/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = Path(os.environ.get("QSS_INTERIM_DIR", PROJECT_ROOT / "data" / "interim"))

INPUT_DB_PATH = INTERIM_DIR / "openalex_ai_works_merged_deduplicated.db"
OUTPUT_DB_PATH = INTERIM_DIR / "openalex_ai_works_merged_deduplicated_cleaned.db"


# === STEP 1: Read works table ===
print("🔍 Reading works table from input DB...")
conn_in = sqlite3.connect(INPUT_DB_PATH)
df = pd.read_sql_query("SELECT * FROM works", conn_in)
conn_in.close()
print(f"✅ Loaded rows: {len(df)}")


# === STEP 2: Decompression Utility ===
def decompress_abstract(json_str):
    if not isinstance(json_str, str):
        return ""
    inverted_index = json.loads(json_str)
    max_pos = max(pos for positions in inverted_index.values() for pos in positions)
    words = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return ' '.join(words)


def parallel_decompress(series, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(decompress_abstract, series))
    return results


# === STEP 3: Process Abstracts (Decompression Only) ===
print("🛠 Decompressing abstracts...")

df['abstract_inverted_index'] = df['abstract_inverted_index'].fillna("")

mask = (df['abstract_inverted_index'] != "") & (df['language'] == 'en')
abstracts = df.loc[mask, 'abstract_inverted_index']

decompressed = parallel_decompress(abstracts, workers=8)

df['cleaned_abstract'] = None
df.loc[mask, 'cleaned_abstract'] = pd.Series(decompressed, index=abstracts.index)

print("✅ Added cleaned_abstract column (decompressed only).")


# === STEP 4: Write new DB ===
print(f"💾 Writing new DB to: {OUTPUT_DB_PATH} ...")
conn_out = sqlite3.connect(OUTPUT_DB_PATH)
df.to_sql('works', conn_out, index=False, if_exists='replace')
conn_out.close()
print("✅ New database written successfully.")
