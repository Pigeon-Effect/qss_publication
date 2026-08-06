"""Derive each work's country composition from its author institutions.

Second of the three quality-control steps in manuscript section 2.3. Country of
origin is not native to OpenAlex: it is built here as a list of country-share
tuples summing to one, e.g. [("US", 0.75), ("DE", 0.25)]. Records without an
institution-based attribution are left empty and dropped by the next step,
rather than having a country inferred from language, funder or publisher.
"""

import os
import sqlite3
import pandas as pd
import json
from collections import Counter
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
# Paths are derived from this file's location, so the script runs from any
# checkout. .../code/03_data_processing/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = Path(os.environ.get("QSS_INTERIM_DIR", PROJECT_ROOT / "data" / "interim"))

INPUT_DB_PATH = INTERIM_DIR / "openalex_ai_works_merged_deduplicated.db"
OUTPUT_DB_PATH = INTERIM_DIR / "openalex_ai_works_merged_deduplicated_country_with_origin.db"
BATCH_SIZE = 10000  # adjust if needed

# === STEP 1: Read works table ===
print("🔍 Reading works table from input DB...")
conn_in = sqlite3.connect(INPUT_DB_PATH)
df = pd.read_sql_query("SELECT * FROM works", conn_in)
conn_in.close()
print(f"✅ Loaded rows: {len(df)}")

# === STEP 2: Compute country_of_origin list ===
print("🌍 Extracting country composition...")
country_origin_list = []

for authorships_json in tqdm(df['authorships'], desc="Processing authorships"):
    country_list = []
    if authorships_json:
        try:
            authorships = json.loads(authorships_json)
            country_codes = []
            for author in authorships:
                found_country = False

                # 1️⃣ Primary: institutions → country_code
                institutions = author.get('institutions', [])
                for inst in institutions:
                    cc = inst.get('country_code')
                    if cc:
                        country_codes.append(cc)
                        found_country = True

                # 2️⃣ Fallback: countries field
                if not found_country:
                    author_countries = author.get('countries', [])
                    for cc in author_countries:
                        if cc:
                            country_codes.append(cc)

            if country_codes:
                counts = Counter(country_codes)
                total = sum(counts.values())
                composition = [(cc, count / total) for cc, count in counts.most_common()]
                country_list = composition

        except json.JSONDecodeError:
            country_list = []
    country_origin_list.append(country_list)

df['country_of_origin'] = country_origin_list

print("✅ Added country_of_origin column.")

# === Serialize list to JSON text ===
df['country_of_origin'] = df['country_of_origin'].apply(json.dumps)

print(f"💾 Writing new DB to: {OUTPUT_DB_PATH} ...")
conn_out = sqlite3.connect(OUTPUT_DB_PATH)
df.to_sql('works', conn_out, index=False, if_exists='replace')
conn_out.close()
print("✅ New database written successfully.")
