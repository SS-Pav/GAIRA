from __future__ import annotations

import json
from pathlib import Path

import duckdb

from gaira.evidence_v1.assignment_patterns import build_assignment_patterns
from gaira.evidence_v1.constants import (
    DB_PATH,
    PATTERN_REFINEMENT_QA_ROOT,
    PATTERN_REFINEMENT_REPORT_ROOT,
    ensure_pattern_refinement_output_dirs,
)
from gaira.evidence_v1.qa_pattern_granularity import (
    generate_pattern_granularity_qa,
    write_example_json,
)
from gaira.evidence_v1.retrieval import PeakListRetrievalEngine
from gaira.evidence_v1.schema import reset_pattern_tables


EXAMPLE_SPECS = [
    {"name": "example_serum_sers", "peaks": [725.0, 1004.0, 1450.0, 1660.0], "domain": "serum", "modality": "sers"},
    {"name": "example_ev_sers", "peaks": [785.0, 1095.0, 1452.0, 1658.0], "domain": "ev", "modality": "sers"},
    {"name": "example_pathogen_raman", "peaks": [669.0, 772.0, 1063.0, 1447.0], "domain": "pathogen", "modality": "raman"},
]


def _pattern_counts(connection: duckdb.DuckDBPyConnection) -> dict[str, float]:
    return {
        "pattern_count": float(connection.sql("SELECT COUNT(*) FROM evidence.assignment_patterns").fetchone()[0]),
        "avg_pattern_size": float(connection.sql("SELECT COALESCE(AVG(total_member_count), 0) FROM evidence.assignment_patterns").fetchone()[0]),
        "avg_core_size": float(connection.sql("SELECT COALESCE(AVG(core_member_count), 0) FROM evidence.assignment_patterns").fetchone()[0]),
        "same_family_multi_pattern_cases": float(
            connection.sql(
                "SELECT COUNT(*) FROM (SELECT normalized_family FROM evidence.assignment_patterns GROUP BY normalized_family HAVING COUNT(*) > 1)"
            ).fetchone()[0]
        ),
    }


def _run_examples(engine: PeakListRetrievalEngine) -> dict[str, dict]:
    outputs = {}
    for spec in EXAMPLE_SPECS:
        outputs[spec["name"]] = engine.search(
            query_peaks=spec["peaks"],
            domain_hint=spec["domain"],
            modality_hint=spec["modality"],
            tolerance_cm=10.0,
            top_k=5,
        )
    return outputs


def main() -> None:
    ensure_pattern_refinement_output_dirs()
    engine = PeakListRetrievalEngine(str(DB_PATH))
    before_examples = _run_examples(engine)
    with duckdb.connect(str(DB_PATH), read_only=True) as connection:
        before_counts = _pattern_counts(connection)

    with duckdb.connect(str(DB_PATH)) as connection:
        reset_pattern_tables(connection)
        build_counts = build_assignment_patterns(connection)

    after_engine = PeakListRetrievalEngine(str(DB_PATH))
    after_examples = _run_examples(after_engine)
    for spec in EXAMPLE_SPECS:
        write_example_json(PATTERN_REFINEMENT_QA_ROOT / f"{spec['name']}_before.json", before_examples[spec["name"]])
        write_example_json(PATTERN_REFINEMENT_QA_ROOT / f"{spec['name']}_after.json", after_examples[spec["name"]])
    with duckdb.connect(str(DB_PATH), read_only=True) as connection:
        after_counts = _pattern_counts(connection)

    qa_counts = generate_pattern_granularity_qa(
        db_path=Path(DB_PATH),
        before_counts=before_counts,
        after_counts=after_counts,
        before_examples=before_examples,
        after_examples=after_examples,
    )

    summary = {
        "before_counts": before_counts,
        "after_counts": after_counts,
        "build_counts": build_counts,
        "qa_counts": qa_counts,
    }
    (PATTERN_REFINEMENT_REPORT_ROOT / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
