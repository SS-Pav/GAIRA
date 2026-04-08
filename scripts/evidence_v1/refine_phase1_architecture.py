from __future__ import annotations

import json
from pathlib import Path

import duckdb

from gaira.evidence_v1.constants import (
    DB_PATH,
    QA_ROOT,
    REFINEMENT_QA_ROOT,
    REFINEMENT_REPORT_ROOT,
    ensure_refinement_output_dirs,
)
from gaira.evidence_v1.phase1_refinement import build_phase1_refinement
from gaira.evidence_v1.qa_phase1_refinement import (
    generate_refinement_qa_artifacts,
    write_example_retrieval,
)
from gaira.evidence_v1.retrieval import PeakListRetrievalEngine
from gaira.evidence_v1.schema import initialize_schema, reset_phase1_refinement_tables


def _write_implementation_note(path: Path, refinement_counts: dict[str, int], qa_counts: dict[str, int], retrieval_examples: list[dict]) -> None:
    lines = [
        "# GAIRA Phase 1 Refinement Pass",
        "",
        "## What Changed From v1",
        "",
        "- Added a peak-meaning cluster/support layer so Phase 1 retrieval units are interpretable peak windows rather than raw molecule/reference rows.",
        "- Added operational mention handling that only keeps mention rows as aligned secondary support when they reinforce an existing meaning cluster.",
        "- Added a compact context graph with nodes and edges for sample type, modality, caveats, affected evidence families, and key peak regions.",
        "- Refactored retrieval to return support bundles with confidence, ambiguity, source diversity, and context modifiers.",
        "",
        "## Refined Phase 1 Architecture",
        "",
        "- Evidence warehouse: preserved v1 evidence tables and derived operational peak-meaning clusters plus support rows.",
        "- Retrieval layer: uses `retrieval.peak_meaning_documents` instead of raw reference-object documents.",
        "- Context relationship layer: uses `context.context_nodes` and `context.context_edges` and applies them bundle-by-bundle.",
        "- BSV remains non-operational. Retrieval ignores `interpretation.bsv_definitions` and `interpretation.evidence_bsv_links` in this pass.",
        "",
        "## What Remains Deferred To Phase 2",
        "",
        "- Mature BSV logic and inference coupling.",
        "- Neural retrieval or representation models.",
        "- Large-scale biological dataset testing over the 183K spectra.",
        "- Figure digitization and table parsing beyond queue linkage.",
        "",
        "## Refinement Counts",
        "",
        *(f"- `{key}`: {value}" for key, value in refinement_counts.items()),
        "",
        "## QA Snapshot",
        "",
        *(f"- `{key}`: {value}" for key, value in qa_counts.items()),
        "",
        "## Example Runs",
        "",
        *(f"- `{example['run_id']}`: {len(example['support_bundle_results'])} support bundles, {len(example['context_graph_results'])} context modifiers" for example in retrieval_examples),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_refinement_output_dirs()
    with duckdb.connect(str(DB_PATH)) as connection:
        initialize_schema(connection)
        reset_phase1_refinement_tables(connection)
        refinement_counts = build_phase1_refinement(connection)

    engine = PeakListRetrievalEngine(str(DB_PATH))
    example_specs = [
        {"name": "example_serum_sers_refined", "peaks": [725.0, 1004.0, 1450.0, 1660.0], "domain": "serum", "modality": "sers"},
        {"name": "example_ev_sers_refined", "peaks": [785.0, 1095.0, 1452.0, 1658.0], "domain": "ev", "modality": "sers"},
        {"name": "example_pathogen_raman_refined", "peaks": [669.0, 772.0, 1063.0, 1447.0], "domain": "pathogen", "modality": "raman"},
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
        write_example_retrieval(REFINEMENT_QA_ROOT / f"{spec['name']}.json", payload)

    before_paths = {
        "serum": QA_ROOT / "example_serum_sers.json",
        "ev": QA_ROOT / "example_ev_sers.json",
        "pathogen": QA_ROOT / "example_pathogen_raman.json",
    }
    after_paths = {
        "serum": REFINEMENT_QA_ROOT / "example_serum_sers_refined.json",
        "ev": REFINEMENT_QA_ROOT / "example_ev_sers_refined.json",
        "pathogen": REFINEMENT_QA_ROOT / "example_pathogen_raman_refined.json",
    }
    qa_counts = generate_refinement_qa_artifacts(DB_PATH, before_paths, after_paths)
    _write_implementation_note(REFINEMENT_REPORT_ROOT / "implementation_note.md", refinement_counts, qa_counts, retrieval_examples)
    (REFINEMENT_REPORT_ROOT / "build_summary.json").write_text(
        json.dumps(
            {
                "refinement_counts": refinement_counts,
                "qa_counts": qa_counts,
                "retrieval_runs": [item["run_id"] for item in retrieval_examples],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"refinement_counts": refinement_counts, "qa_counts": qa_counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
