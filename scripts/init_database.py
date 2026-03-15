import duckdb
from pathlib import Path

# ensure data directory exists
Path("data").mkdir(exist_ok=True)

# create/connect database
db = duckdb.connect("data/gaira.duckdb")

# create dataset registry table
db.execute("""
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT,
    name TEXT,
    source_url TEXT,
    source_type TEXT,
    modality TEXT,
    sample_type TEXT,
    matrix_type TEXT,
    n_spectra INTEGER,
    notes TEXT
)
""")

print("GAIRA database initialized.")