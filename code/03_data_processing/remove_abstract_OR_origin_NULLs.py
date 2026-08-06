"""Drop records without an abstract or a country, and filter by abstract length.

Third of the three quality-control steps in manuscript section 2.3. Keeps only
works that have a country attribution, a reconstructed abstract, and an abstract
between 100 and 1000 tokens - the length window that removes the 107,515 records
identified as likely formatting errors.
"""

import sqlite3
import os
from pathlib import Path

# Paths are derived from this file's location, so the script runs from any
# checkout. .../code/03_data_processing/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = Path(os.environ.get("QSS_INTERIM_DIR", PROJECT_ROOT / "data" / "interim"))

db_path = INTERIM_DIR / "openalex_ai_works_merged_deduplicated_cleaned.db"
new_db_path = INTERIM_DIR / "openalex_ai_works_merged_deduplicated_cleaned_filtered.db"

# Connect to original DB
conn_src = sqlite3.connect(db_path)
cursor_src = conn_src.cursor()

# Create new DB
conn_dst = sqlite3.connect(new_db_path)
cursor_dst = conn_dst.cursor()

# Copy schema
cursor_src.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='works'")
create_table_sql = cursor_src.fetchone()[0]
cursor_dst.execute(create_table_sql)

# Filter and insert rows
select_sql = """
    SELECT * FROM works
    WHERE 
        country_of_origin IS NOT NULL
        AND country_of_origin != '[]'
        AND cleaned_abstract IS NOT NULL
"""
rows = cursor_src.execute(select_sql).fetchall()
columns = [desc[0] for desc in cursor_src.description]

insert_sql = f"INSERT INTO works ({', '.join(columns)}) VALUES ({', '.join(['?'] * len(columns))})"

def token_count(text):
    return len(text.split())

filtered_rows = [
    row for row in rows
    if 100 <= token_count(row[columns.index('cleaned_abstract')]) <= 1000
]

cursor_dst.executemany(insert_sql, filtered_rows)

# Finalize
conn_dst.commit()
conn_src.close()
conn_dst.close()
