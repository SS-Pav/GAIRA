from __future__ import annotations

import json
from pathlib import Path

import duckdb

from gaira.evidence_v1.constants import DB_PATH, QA_ROOT, REPORT_ROOT, ensure_output_dirs
from gaira.evidence_v1.loaders import load_all
from gaira.evidence_v1.qa import generate_qa_artifacts, write_example_retrieval
from gaira.evidence_v1.retrieval import PeakListRetrievalEngine
from gaira.evidence_v1.schema import initialize_schema, reset_v1_tables


def _write_implementation_note(path: Path, loader_counts: dict[str, int], qa_counts: dict[str, int], retrieval_examples: list[dict]) -> None:
    resolved_in_file = 18
    note_lines = [
        "# GAIRA Evidence Operationalization v1",
        "",
        "## What Was Built",
        "",
        "- Added isolated DuckDB schemas for registry, evidence, features, interpretation, and retrieval.",
        "- Loaded curated peak assignments, source-backed valid assignments, RamanBioLib reference bridges, low-confidence wavenumber mentions, digitization queue metadata, and minimal context rules.",
        "- Built a rule-based peak-list retrieval harness using explicit peak overlap, tolerance distance, evidence tier, domain compatibility, and ambiguity penalties.",
        "",
        "## What Was Loaded",
        "",
        f"- Curated peak assignments from existing `main.peak_assignments`: {loader_counts.get('curated_assignments_loaded', 0)}",
        f"- Source-backed valid assignments: {loader_counts.get('source_backed_valid_assignments_loaded', 0)}",
        f"- Wavenumber mentions kept separate: {loader_counts.get('wavenumber_mentions_loaded', 0)}",
        f"- RamanBioLib reference spectra bridged: {loader_counts.get('ramanbiolib_reference_items_loaded', 0)}",
        f"- RamanBioLib reference peak features bridged: {loader_counts.get('ramanbiolib_reference_features_loaded', 0)}",
        f"- Digitization registry entries: {loader_counts.get('digitization_registry_rows_loaded', 0)}",
        f"- Context rules loaded: {loader_counts.get('context_rules_loaded', 0)}",
        "",
        "## What Was Intentionally Excluded",
        "",
        "- No dataset-layer biological spectra were added or modified.",
        "- Trieste HCC duplicate spectra were not ingested into the evidence layer.",
        "- ExosomeSERS mean and diff spectra were not ingested into the evidence layer.",
        "- Wavenumber mentions were not inserted into spectral features and are not directly retrievable.",
        "- Source-data arrays were not promoted into evidence tables.",
        "",
        "## Assumptions And Missing-File Substitutions",
        "",
        "- Used the configured live DuckDB at `/Volumes/SSD_Rad/GAIRA_DATA/interim/gaira.duckdb` instead of the empty repo-local DuckDB.",
        "- Used existing `main.reference_*` tables as the RamanBioLib bridge source rather than re-ingesting raw CSVs.",
        "- Used the corrected source-backed files from `gaira_source_backed_evidence_v1_corrected` as ground truth for the 95/226/174 split.",
        f"- The corrected valid-assignment CSV contains {resolved_in_file} non-`unknown` `assigned_molecule` values. The upstream note says 19 and your prompt states 22 resolved molecules; I preserved the file contents and did not fabricate extra resolved labels.",
        "- `plasma` retrieval currently benefits from direct evidence scoring, but context append rules remain sparse relative to `ev` and `serum`.",
        "",
        "## Current Limitations",
        "",
        "- Retrieval is peak-list-only and does not yet parse tables or digitized figure traces.",
        "- Source-backed regex assignments remain lower-confidence than curated assignments by explicit tier weighting.",
        "- Context rules are append-only heuristics and not scored as direct evidence.",
        "",
        "## Next Recommended Step",
        "",
        "- Build a controlled table-parsing and figure-digitization ingestion pass for high-priority families, then convert verified assignments into higher-confidence structured evidence rows before any broader spectral-text operationalization.",
        "",
        "## QA Snapshot",
        "",
        *(f"- `{table}`: {count}" for table, count in qa_counts.items()),
        "",
        "## Example Retrieval Runs",
        "",
        *(f"- `{example['run_id']}` for peaks {example['query_peaks']}: {len(example['direct_results'])} direct hits, {len(example['context_results'])} context append rules" for example in retrieval_examples),
    ]
    path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    with duckdb.connect(str(DB_PATH)) as connection:
        initialize_schema(connection)
        reset_v1_tables(connection)
        loader_counts = load_all(connection)

    qa_counts = generate_qa_artifacts(DB_PATH)

    engine = PeakListRetrievalEngine(str(DB_PATH))
    example_specs = [
        {"name": "example_serum_sers", "peaks": [725.0, 1004.0, 1450.0, 1660.0], "domain": "serum", "modality": "sers"},
        {"name": "example_ev_sers", "peaks": [785.0, 1095.0, 1452.0, 1658.0], "domain": "ev", "modality": "sers"},
        {"name": "example_pathogen_raman", "peaks": [669.0, 772.0, 1063.0, 1447.0], "domain": "pathogen", "modality": "raman"},
    ]
    retrieval_examples = []
    for spec in example_specs:
        result = engine.search(
            query_peaks=spec["peaks"],
            domain_hint=spec["domain"],
            modality_hint=spec["modality"],
            tolerance_cm=10.0,
            top_k=5,
        )
        run_id = engine.persist_run(result, top_k=5)
        result["run_id"] = run_id
        retrieval_examples.append(result)
        write_example_retrieval(QA_ROOT / f"{spec['name']}.json", result)

    qa_counts = generate_qa_artifacts(DB_PATH)

    _write_implementation_note(
        REPORT_ROOT / "implementation_note.md",
        loader_counts=loader_counts,
        qa_counts=qa_counts,
        retrieval_examples=retrieval_examples,
    )
    summary_path = REPORT_ROOT / "build_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "loader_counts": loader_counts,
                "qa_counts": qa_counts,
                "retrieval_runs": [example["run_id"] for example in retrieval_examples],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"loader_counts": loader_counts, "qa_counts": qa_counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
