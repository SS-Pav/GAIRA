from __future__ import annotations

import json

import duckdb

from gaira.evidence_v1.constants import DB_PATH, WAREHOUSE_REPORT_ROOT, ensure_warehouse_output_dirs
from gaira.evidence_v1.qa_warehouse_grounding import generate_warehouse_grounding_qa
from gaira.evidence_v1.warehouse_grounding import build_warehouse_backbone


def main() -> None:
    ensure_warehouse_output_dirs()
    with duckdb.connect(str(DB_PATH)) as connection:
        build_counts = build_warehouse_backbone(connection)

    qa_counts = generate_warehouse_grounding_qa(DB_PATH, build_counts)
    summary = {"build_counts": build_counts, "qa_counts": qa_counts}
    (WAREHOUSE_REPORT_ROOT / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
