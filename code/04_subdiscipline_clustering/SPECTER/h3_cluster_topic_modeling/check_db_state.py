"""Inspect one (h1, h2) slice of a subset database before labelling it.

A read-only sanity check: prints a sample of rows, how many documents are
eligible, and whether any of them already carry an `h3_cluster` value - which
`h3_label_selected_cluster.py` refuses to overwrite.
"""

import os
import sqlite3
from pathlib import Path

# Paths are derived from this file's location, so the script runs from any
# checkout. .../SPECTER/h3_cluster_topic_modeling/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
INTERIM_DIR = Path(os.environ.get("QSS_INTERIM_DIR", PROJECT_ROOT / "data" / "interim"))
DB_PATH = Path(
    os.environ.get(
        "QSS_H3_DB", INTERIM_DIR / "h1_cluster_subsets" / "computer_science_dataset.db"
    )
)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Check sample data for h1=0, h2=2
cursor.execute("SELECT rowid, h1_cluster, h2_cluster, h3_cluster FROM works_labeled WHERE h1_cluster=0 AND h2_cluster=2 LIMIT 10")
sample = cursor.fetchall()
print("Sample rows for h1=0, h2=2:")
for row in sample:
    print(f"  rowid={row[0]}, h1={row[1]}, h2={row[2]}, h3={row[3]}")

# Count total rows and existing h3_cluster values
cursor.execute("SELECT COUNT(*) FROM works_labeled WHERE h1_cluster=0 AND h2_cluster=2 AND cleaned_abstract IS NOT NULL AND language='en'")
total_count = cursor.fetchone()[0]
print(f"\nTotal eligible rows for h1=0, h2=2: {total_count}")

cursor.execute("SELECT COUNT(*) FROM works_labeled WHERE h1_cluster=0 AND h2_cluster=2 AND h3_cluster IS NOT NULL")
existing_count = cursor.fetchone()[0]
print(f"Rows with existing h3_cluster values: {existing_count}")

if existing_count > 0:
    cursor.execute("SELECT h3_cluster, COUNT(*) FROM works_labeled WHERE h1_cluster=0 AND h2_cluster=2 AND h3_cluster IS NOT NULL GROUP BY h3_cluster")
    distribution = cursor.fetchall()
    print("Distribution of existing h3_cluster values:")
    for h3_val, count in distribution:
        print(f"  h3_cluster={h3_val}: {count} rows")

conn.close()
