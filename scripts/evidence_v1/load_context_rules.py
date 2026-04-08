from __future__ import annotations

import json

import duckdb

from gaira.evidence_v1.constants import DB_PATH
from gaira.evidence_v1.loaders import load_bsv_definitions, load_context_rules


def main() -> None:
    with duckdb.connect(str(DB_PATH)) as connection:
        counts = {}
        counts.update(load_context_rules(connection))
        counts.update(load_bsv_definitions(connection))
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

