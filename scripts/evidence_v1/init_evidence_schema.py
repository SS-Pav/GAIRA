from __future__ import annotations

import duckdb

from gaira.evidence_v1.constants import DB_PATH, ensure_output_dirs
from gaira.evidence_v1.schema import initialize_schema, reset_v1_tables


def main() -> None:
    ensure_output_dirs()
    with duckdb.connect(str(DB_PATH)) as connection:
        initialize_schema(connection)
        reset_v1_tables(connection)
    print(f"Initialized GAIRA evidence v1 schema in {DB_PATH}")


if __name__ == "__main__":
    main()

