from __future__ import annotations

import json
from pathlib import Path

import duckdb

from gaira.evidence_v1.constants import (
    CLEANUP_QA_ROOT,
    CLEANUP_REPORT_ROOT,
    DB_PATH,
    REFINEMENT_QA_ROOT,
    ensure_cleanup_output_dirs,
)
from gaira.evidence_v1.phase1_refinement import FAMILY_LABELS, PREVIOUS_FAMILY_LABELS, build_phase1_refinement
from gaira.evidence_v1.qa_phase1_cleanup import generate_cleanup_qa_artifacts, write_example_retrieval
from gaira.evidence_v1.retrieval import PeakListRetrievalEngine
from gaira.evidence_v1.schema import initialize_schema, reset_phase1_refinement_tables


def _write_implementation_note(path: Path, refinement_counts: dict[str, int], qa_counts: dict[str, int], retrieval_examples: list[dict]) -> None:
    family_changes = [
        f"- `{family_id}`: `{PREVIOUS_FAMILY_LABELS[family_id]}` -> `{FAMILY_LABELS[family_id]}`"
        for family_id in sorted(FAMILY_LABELS)
        if PREVIOUS_FAMILY_LABELS[family_id] != FAMILY_LABELS[family_id]
    ]
    lines = [
        "# GAIRA Phase 1 Cleanup and Audit Pass",
        "",
        "## What Changed",
        "",
        "- Tightened the Phase 1 family vocabulary so retrieval labels are cleaner while still broad enough to avoid false specificity.",
        "- Kept the peak-meaning cluster layer intact, but added mixed-family flags, overlap counts, and explicit score-component provenance.",
        "- Tightened mention handling so only hint-backed, family-coherent mentions remain as weak secondary support.",
        "- Kept the context graph compact and added a small number of edges that actually affect bundle-level caveats.",
        "- Left BSV tables in placeholder status and kept them out of retrieval.",
        "",
        "## Label Vocabulary Changes",
        "",
        *(family_changes or ["- No label text changes were required."]),
        "",
        "## Confirmed Versus Fixed",
        "",
        "- Confirmed: same-family over-fragmentation was not the main issue. The cluster count remained stable because nearby duplicate windows were not present.",
        "- Fixed: mention alignment was too permissive and is now family-hint constrained.",
        "- Fixed: confidence and ambiguity now expose explicit components and penalties instead of opaque aggregate values only.",
        "- Fixed: context graph edges now attach more directly to the families most likely to be caveated in serum/EV SERS retrieval.",
        "",
        "## Still Unresolved",
        "",
        "- Cross-family overlap remains intrinsic around several Raman windows and is represented as ambiguity, not resolved away.",
        "- Reference-heavy support bundles remain lower-confidence than curated assignment bundles, but they are still useful as meaning support, not final biological calls.",
        "- Figure digitization is still queued rather than operationalized.",
        "",
        "## Ready For Pilot Digitization",
        "",
        "- Peak-window support bundles are now cleaner retrieval targets for pilot paper digitization and subsequent structured table/figure linkage.",
        "- Mention provenance is preserved without letting weak regex artifacts dominate operational retrieval.",
        "",
        "## Deferred To Phase 2",
        "",
        "- Mature BSV inference.",
        "- Neural retrieval or representation models.",
        "- Biological dataset-scale evaluation.",
        "- Automated figure/table digitization ingestion.",
        "",
        "## Cleanup Counts",
        "",
        *(f"- `{key}`: {value}" for key, value in refinement_counts.items()),
        "",
        "## QA Snapshot",
        "",
        *(f"- `{key}`: {value}" for key, value in qa_counts.items()),
        "",
        "## Example Runs",
        "",
        *(
            f"- `{example['run_id']}`: {len(example['support_bundle_results'])} bundles, "
            f"{len(example['context_graph_results'])} context modifiers"
            for example in retrieval_examples
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_cleanup_output_dirs()
    with duckdb.connect(str(DB_PATH)) as connection:
        initialize_schema(connection)
        reset_phase1_refinement_tables(connection)
        refinement_counts = build_phase1_refinement(connection)

    engine = PeakListRetrievalEngine(str(DB_PATH))
    example_specs = [
        {"name": "example_serum_sers_cleanup", "peaks": [725.0, 1004.0, 1450.0, 1660.0], "domain": "serum", "modality": "sers"},
        {"name": "example_ev_sers_cleanup", "peaks": [785.0, 1095.0, 1452.0, 1658.0], "domain": "ev", "modality": "sers"},
        {"name": "example_pathogen_raman_cleanup", "peaks": [669.0, 772.0, 1063.0, 1447.0], "domain": "pathogen", "modality": "raman"},
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
        write_example_retrieval(CLEANUP_QA_ROOT / f"{spec['name']}.json", payload)

    before_paths = {
        "serum": REFINEMENT_QA_ROOT / "example_serum_sers_refined.json",
        "ev": REFINEMENT_QA_ROOT / "example_ev_sers_refined.json",
        "pathogen": REFINEMENT_QA_ROOT / "example_pathogen_raman_refined.json",
    }
    after_paths = {
        "serum": CLEANUP_QA_ROOT / "example_serum_sers_cleanup.json",
        "ev": CLEANUP_QA_ROOT / "example_ev_sers_cleanup.json",
        "pathogen": CLEANUP_QA_ROOT / "example_pathogen_raman_cleanup.json",
    }
    qa_counts = generate_cleanup_qa_artifacts(DB_PATH, before_paths, after_paths)
    _write_implementation_note(CLEANUP_REPORT_ROOT / "implementation_note.md", refinement_counts, qa_counts, retrieval_examples)
    (CLEANUP_REPORT_ROOT / "build_summary.json").write_text(
        json.dumps(
            {
                "cleanup_counts": refinement_counts,
                "qa_counts": qa_counts,
                "retrieval_runs": [item["run_id"] for item in retrieval_examples],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"cleanup_counts": refinement_counts, "qa_counts": qa_counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
