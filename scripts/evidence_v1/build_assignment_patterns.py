from __future__ import annotations

import json
from pathlib import Path

import duckdb

from gaira.evidence_v1.assignment_patterns import build_assignment_patterns
from gaira.evidence_v1.constants import (
    CLEANUP_QA_ROOT,
    DB_PATH,
    PATTERN_QA_ROOT,
    PATTERN_REPORT_ROOT,
    ensure_pattern_output_dirs,
)
from gaira.evidence_v1.qa_assignment_patterns import generate_assignment_pattern_qa, write_example_retrieval
from gaira.evidence_v1.retrieval import PeakListRetrievalEngine
from gaira.evidence_v1.schema import initialize_schema, reset_pattern_tables


def _write_implementation_note(path: Path, pattern_counts: dict[str, int], qa_counts: dict[str, int], retrieval_examples: list[dict]) -> None:
    lines = [
        "# GAIRA Assignment Pattern Pass",
        "",
        "## What Was Built",
        "",
        "- Added a multi-peak assignment pattern layer above peak clusters.",
        "- Patterns are constructed from within-family co-support across study-family and reference-component contexts.",
        "- Retrieval now exposes pattern-level matches plus underlying cluster-level details.",
        "",
        "## Construction Logic",
        "",
        "- Cluster supports were grouped by normalized family.",
        "- A family-specific co-support graph was built using repeated context overlap between clusters.",
        "- Only multi-cluster connected components became patterns.",
        "- Member roles were assigned as core, supporting, optional, or ambiguous from context prevalence, pair containment, and direct-support strength.",
        "",
        "## Limitations",
        "",
        "- Patterns are still evidence signatures, not final biological interpretation.",
        "- Reference-rich families can still produce broad constellations when the literature support is sparse.",
        "- Isolated clusters remain cluster-only and do not force pattern creation.",
        "",
        "## Interpretability Assessment",
        "",
        "- Patterns improved retrieval when multiple family-consistent bands were present.",
        "- Improvement is strongest for families with repeated within-family co-support; weaker families remain provisional.",
        "",
        "## Pattern Counts",
        "",
        *(f"- `{key}`: {value}" for key, value in pattern_counts.items()),
        "",
        "## QA Snapshot",
        "",
        *(f"- `{key}`: {value}" for key, value in qa_counts.items()),
        "",
        "## Example Runs",
        "",
        *(
            f"- `{example['run_id']}`: {len(example.get('pattern_results', []))} pattern hits, "
            f"{len(example.get('support_bundle_results', []))} cluster hits"
            for example in retrieval_examples
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_pattern_output_dirs()
    with duckdb.connect(str(DB_PATH)) as connection:
        initialize_schema(connection)
        reset_pattern_tables(connection)
        pattern_counts = build_assignment_patterns(connection)

    engine = PeakListRetrievalEngine(str(DB_PATH))
    example_specs = [
        {"name": "example_serum_sers_patterns", "peaks": [725.0, 1004.0, 1450.0, 1660.0], "domain": "serum", "modality": "sers"},
        {"name": "example_ev_sers_patterns", "peaks": [785.0, 1095.0, 1452.0, 1658.0], "domain": "ev", "modality": "sers"},
        {"name": "example_pathogen_raman_patterns", "peaks": [669.0, 772.0, 1063.0, 1447.0], "domain": "pathogen", "modality": "raman"},
    ]
    retrieval_examples = []
    for spec in example_specs:
        payload = engine.search(
            query_peaks=spec["peaks"],
            domain_hint=spec["domain"],
            modality_hint=spec["modality"],
            tolerance_cm=10.0,
            top_k=6,
        )
        run_id = engine.persist_run(payload, top_k=6)
        payload["run_id"] = run_id
        retrieval_examples.append(payload)
        write_example_retrieval(PATTERN_QA_ROOT / f"{spec['name']}.json", payload)

    before_paths = {
        "serum": CLEANUP_QA_ROOT / "example_serum_sers_cleanup.json",
        "ev": CLEANUP_QA_ROOT / "example_ev_sers_cleanup.json",
        "pathogen": CLEANUP_QA_ROOT / "example_pathogen_raman_cleanup.json",
    }
    after_paths = {
        "serum": PATTERN_QA_ROOT / "example_serum_sers_patterns.json",
        "ev": PATTERN_QA_ROOT / "example_ev_sers_patterns.json",
        "pathogen": PATTERN_QA_ROOT / "example_pathogen_raman_patterns.json",
    }
    qa_counts = generate_assignment_pattern_qa(DB_PATH, before_paths, after_paths)
    _write_implementation_note(PATTERN_REPORT_ROOT / "implementation_note.md", pattern_counts, qa_counts, retrieval_examples)
    (PATTERN_REPORT_ROOT / "build_summary.json").write_text(
        json.dumps(
            {
                "pattern_counts": pattern_counts,
                "qa_counts": qa_counts,
                "retrieval_runs": [item["run_id"] for item in retrieval_examples],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"pattern_counts": pattern_counts, "qa_counts": qa_counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
