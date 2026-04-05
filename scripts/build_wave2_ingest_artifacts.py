from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaira.config import ensure_storage_dirs, resolve_storage_path  # noqa: E402
from gaira.demo.autoresearch_utils import build_pdf_report  # noqa: E402


PROCESSED_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/global_v2_ingest"
)
WAVE2_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/global_v2_ingest_wave2"
)


INGESTED_CONFIG = {
    "stroke_urine_sers": {
        "fc_id": "FC08",
        "title": "Ischemic-Stroke Urine SERS Cohort",
        "role": "core_training",
        "source_urls": "https://zenodo.org/records/19109120 | https://zenodo.org/records/19369604",
        "domain_tags": "urine; stroke; human",
        "label_schema": "stroke vs healthy_control; sample_group id; replicate id",
        "summary_note": (
            "Canonical biosample ingest uses `data.rar::CBI/data.csv`. `PI/data.csv` is retained as "
            "provenance-only sidecar because its participant cardinality does not align cleanly with "
            "the released cohort matrix groups."
        ),
    },
    "ucla_saliva_sev_gc": {
        "fc_id": "FC32",
        "title": "UCLA Saliva sEV Gastric-Cancer Cohort",
        "role": "augmentation_only",
        "source_urls": (
            "https://figshare.com/articles/dataset/Health_control_01_sEV_Saliva_UCLA_ERCC_/20428395"
        ),
        "domain_tags": "saliva; EV; gastric cancer",
        "label_schema": "gastric_cancer vs healthy_control; patient id; replicate id; shard/article provenance",
        "summary_note": (
            "Grouped-family ingest preserves every recovered shard txt file plus the canonical "
            "shard manifest. Patient and class labels are reconstructed from shard filenames/titles."
        ),
    },
}


PENDING_ROWS = [
    {
        "dataset_id": "stemcell_diff_mito_sers",
        "fc_id": "FC05",
        "title": "Stem-Cell Differentiation Mitochondrial SERS Trajectory",
        "role": "augmentation_only",
        "status": "not_started_wave_2",
        "n_spectra": "",
        "n_samples": "",
        "n_patients": "",
        "label_schema": "six differentiation states",
        "domain_tags": "cell_state; differentiation; probe-mediated",
        "notes": "Priority after Wave 2 core biofluid completion; not started in this continuation.",
    },
    {
        "dataset_id": "tumor_purine_secretome_sers",
        "fc_id": "FC16",
        "title": "Tumor Purine Secretome SERS Archive",
        "role": "augmentation_only",
        "status": "not_started_wave_2",
        "n_spectra": "",
        "n_samples": "",
        "n_patients": "",
        "label_schema": "cell type; perturbation; timepoint",
        "domain_tags": "secretome; cell_line; perturbation",
        "notes": "Mixed analytical zip still needs targeted biological-spectrum extraction.",
    },
    {
        "dataset_id": "acs_platelet_sers",
        "fc_id": "FC24",
        "title": "ACS Platelet SERS Workbook Family",
        "role": "augmentation_only",
        "status": "deferred",
        "n_spectra": "",
        "n_samples": "",
        "n_patients": "",
        "label_schema": "sample id; replicate blocks",
        "domain_tags": "blood-derived; cardiovascular",
        "notes": "Deferred by instruction until the first four Wave 2 datasets are complete and stable.",
    },
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV without schema: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def archive_member_count(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".rar":
        result = subprocess.run(["bsdtar", "-tf", str(path)], check=True, capture_output=True, text=True)
        return len([line for line in result.stdout.splitlines() if line.strip()])
    return None


def build_stroke_outputs(raw_root: Path, db_path: Path) -> dict:
    dataset_id = "stroke_urine_sers"
    dataset_root = raw_root / dataset_id
    out_root = PROCESSED_ROOT / dataset_id
    table_dir = out_root / "tables"
    report_dir = out_root / "report"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    archive_path = dataset_root / "data.rar"
    inventory_rows = [
        {
            "dataset_id": dataset_id,
            "asset_name": "data.rar",
            "asset_type": "rar",
            "size_bytes": int(archive_path.stat().st_size),
            "archive_member_count": archive_member_count(archive_path),
            "keep_or_exclude": "keep",
            "notes": "Canonical raw Wave 2 archive",
        },
        {
            "dataset_id": dataset_id,
            "asset_name": "data.rar::CBI/data.csv",
            "asset_type": "archive_member_csv",
            "size_bytes": "",
            "archive_member_count": "",
            "keep_or_exclude": "keep",
            "notes": "Released cohort matrix used for the biosample ingest",
        },
        {
            "dataset_id": dataset_id,
            "asset_name": "data.rar::PI/data.csv",
            "asset_type": "archive_member_csv",
            "size_bytes": "",
            "archive_member_count": "",
            "keep_or_exclude": "keep_provenance_only",
            "notes": "Patient-metadata sidecar retained for provenance, not row-joined",
        },
        {
            "dataset_id": dataset_id,
            "asset_name": "data.rar::biomarkers/*",
            "asset_type": "archive_member_txt_family",
            "size_bytes": "",
            "archive_member_count": "",
            "keep_or_exclude": "exclude",
            "notes": "Pure biomarker reference traces excluded from the urine cohort lane",
        },
        {
            "dataset_id": dataset_id,
            "asset_name": "data.rar::R6G/*",
            "asset_type": "archive_member_family",
            "size_bytes": "",
            "archive_member_count": "",
            "keep_or_exclude": "exclude",
            "notes": "R6G assay controls excluded from the biosample ingest",
        },
        {
            "dataset_id": dataset_id,
            "asset_name": "data.rar::SERSspectra/*.wdf",
            "asset_type": "archive_member_wdf_family",
            "size_bytes": "",
            "archive_member_count": "",
            "keep_or_exclude": "exclude",
            "notes": "Instrument-side WDF bundles left out of the canonical ingest pass",
        },
    ]
    write_csv(table_dir / "input_inventory.csv", inventory_rows)

    exclusion_rows = [
        {
            "exclusion_type": "archive_family_exclusion",
            "item": "biomarkers/*",
            "reason": "pure biomolecule references do not belong in the urine biosample lane",
        },
        {
            "exclusion_type": "archive_family_exclusion",
            "item": "R6G/*",
            "reason": "assay controls excluded from the canonical cohort ingest",
        },
        {
            "exclusion_type": "archive_family_exclusion",
            "item": "SERSspectra/*.wdf",
            "reason": "not needed for the current matrix-backed ingest and would require separate WDF conversion",
        },
        {
            "exclusion_type": "metadata_nonjoin_note",
            "item": "PI/data.csv row-level demographics",
            "reason": "132 sidecar participants do not align cleanly with the 102 released matrix sample groups",
        },
    ]
    write_csv(table_dir / "exclusion_summary.csv", exclusion_rows)

    reconstruction_rows = [
        {
            "step_index": 1,
            "step": "selected_archive_version",
            "status": "applied",
            "details": "Used Zenodo 19369604 `data.rar` instead of the older monolithic archive.",
        },
        {
            "step_index": 2,
            "step": "selected_matrix",
            "status": "applied",
            "details": "Parsed `CBI/data.csv` as the released 481-spectrum cohort matrix with 4096 spectral columns plus labels.",
        },
        {
            "step_index": 3,
            "step": "preserved_metadata_sidecar",
            "status": "applied_with_caveat",
            "details": "`PI/data.csv` retained as provenance-only sidecar because participant cardinality does not match matrix groups.",
        },
        {
            "step_index": 4,
            "step": "excluded_support_members",
            "status": "applied",
            "details": "Excluded biomarker reference txt files, R6G controls, and WDF instrument families from the biosample lane.",
        },
    ]
    write_csv(table_dir / "reconstruction_log.csv", reconstruction_rows)

    with duckdb.connect(str(db_path)) as con:
        label_df = con.execute(
            """
            select
                coalesce(class_label, 'unknown') as class_label,
                coalesce(subclass_label, 'unknown') as subclass_label,
                count(*) as n_spectra,
                count(distinct sample_id) as n_samples,
                count(distinct patient_id) as n_patients
            from biosample_metadata
            where dataset_id = ?
            group by 1, 2
            order by n_spectra desc, class_label, subclass_label
            """,
            [dataset_id],
        ).fetchdf()
        counts = con.execute(
            """
            select
                count(*) as n_spectra,
                count(distinct sample_id) as n_samples,
                count(distinct patient_id) as n_patients
            from biosample_metadata
            where dataset_id = ?
            """,
            [dataset_id],
        ).fetchdf().iloc[0].to_dict()

    label_df.to_csv(table_dir / "label_summary.csv", index=False)
    write_csv(
        table_dir / "final_counts.csv",
        [
            {
                "dataset_id": dataset_id,
                "title": INGESTED_CONFIG[dataset_id]["title"],
                "role": INGESTED_CONFIG[dataset_id]["role"],
                "n_spectra": int(counts["n_spectra"]),
                "n_samples": int(counts["n_samples"]),
                "n_patients": int(counts["n_patients"]),
                "raw_archive_sha256": "114e346c1fa057356875ed1ad66cb0c780de182d37a618c82619154f079a7e14",
            }
        ],
    )

    provenance_text = "\n".join(
        [
            f"# {INGESTED_CONFIG[dataset_id]['title']} Provenance Note",
            "",
            f"- Dataset ID: `{dataset_id}`",
            "- FC ID: `FC08`",
            "- Role: `core_training`",
            f"- Source URLs: {INGESTED_CONFIG[dataset_id]['source_urls']}",
            f"- Raw root: `{dataset_root}`",
            "- Canonical ingest member: `data.rar::CBI/data.csv`",
            "- Provenance-only sidecar: `data.rar::PI/data.csv`",
            f"- DuckDB rows written: {int(counts['n_spectra'])} spectra across {int(counts['n_samples'])} sample groups",
            f"- Provenance summary: {INGESTED_CONFIG[dataset_id]['summary_note']}",
        ]
    )
    (report_dir / "provenance_note.md").write_text(provenance_text + "\n", encoding="utf-8")

    return {
        "dataset_id": dataset_id,
        "fc_id": "FC08",
        "title": INGESTED_CONFIG[dataset_id]["title"],
        "role": INGESTED_CONFIG[dataset_id]["role"],
        "status": "ingested_wave_2",
        "n_spectra": int(counts["n_spectra"]),
        "n_samples": int(counts["n_samples"]),
        "n_patients": int(counts["n_patients"]),
        "label_schema": INGESTED_CONFIG[dataset_id]["label_schema"],
        "domain_tags": INGESTED_CONFIG[dataset_id]["domain_tags"],
        "notes": INGESTED_CONFIG[dataset_id]["summary_note"],
    }


def build_ucla_outputs(raw_root: Path, db_path: Path) -> dict:
    dataset_id = "ucla_saliva_sev_gc"
    dataset_root = raw_root / dataset_id
    out_root = PROCESSED_ROOT / dataset_id
    table_dir = out_root / "tables"
    report_dir = out_root / "report"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_root / "shard_manifest.csv"
    manifest_df = pd.read_csv(manifest_path)
    manifest_df.to_csv(table_dir / "shard_manifest.csv", index=False)

    recovered_count = int(manifest_df["recovered"].sum())
    total_bytes = int(manifest_df.loc[manifest_df["recovered"] == True, "bytes_recovered"].sum())
    inventory_rows = [
        {
            "dataset_id": dataset_id,
            "asset_name": "shard_manifest.csv",
            "asset_type": "csv",
            "size_bytes": int(manifest_path.stat().st_size),
            "archive_member_count": "",
            "keep_or_exclude": "keep",
            "notes": "Canonical shard manifest with file-level provenance",
        },
        {
            "dataset_id": dataset_id,
            "asset_name": "recovered_shard_family",
            "asset_type": "txt_family",
            "size_bytes": total_bytes,
            "archive_member_count": recovered_count,
            "keep_or_exclude": "keep",
            "notes": "Recovered saliva sEV shard spectra across the grouped Figshare family",
        },
        {
            "dataset_id": dataset_id,
            "asset_name": "ERCC stub",
            "asset_type": "record_stub",
            "size_bytes": 0,
            "archive_member_count": 0,
            "keep_or_exclude": "exclude",
            "notes": "Empty family stub intentionally excluded from the ingest",
        },
    ]
    write_csv(table_dir / "input_inventory.csv", inventory_rows)

    exclusion_rows = [
        {
            "exclusion_type": "record_stub_exclusion",
            "item": "ERCC stub",
            "reason": "empty record with no downloadable spectra",
        },
        {
            "exclusion_type": "broken_file_exclusion",
            "item": "manifest rows with failed recovery",
            "reason": "none in this pass; all 2231 shard files recovered successfully",
        },
        {
            "exclusion_type": "duplicate_shard_exclusion",
            "item": "duplicate shard paths",
            "reason": "none removed in this pass; canonical manifest contains unique local paths",
        },
    ]
    write_csv(table_dir / "exclusion_summary.csv", exclusion_rows)

    reconstruction_rows = [
        {
            "step_index": 1,
            "step": "built_family_manifest",
            "status": "applied",
            "details": "Grouped the UCLA saliva shard family into one canonical manifest with article-level provenance.",
        },
        {
            "step_index": 2,
            "step": "downloaded_shards",
            "status": "applied",
            "details": f"Recovered {recovered_count} of {len(manifest_df)} shard txt files across {manifest_df['article_id'].nunique()} source records.",
        },
        {
            "step_index": 3,
            "step": "excluded_empty_stub",
            "status": "applied",
            "details": "The empty ERCC family stub was intentionally left out of the canonical manifest and ingest.",
        },
        {
            "step_index": 4,
            "step": "reconstructed_labels",
            "status": "applied_with_caveat",
            "details": "Patient and class labels were reconstructed from shard filenames and Figshare titles, then preserved in metadata notes.",
        },
    ]
    write_csv(table_dir / "reconstruction_log.csv", reconstruction_rows)

    with duckdb.connect(str(db_path)) as con:
        label_df = con.execute(
            """
            select
                coalesce(class_label, 'unknown') as class_label,
                coalesce(subclass_label, 'unknown') as subclass_label,
                count(*) as n_spectra,
                count(distinct sample_id) as n_samples,
                count(distinct patient_id) as n_patients
            from biosample_metadata
            where dataset_id = ?
            group by 1, 2
            order by n_spectra desc, class_label, subclass_label
            """,
            [dataset_id],
        ).fetchdf()
        counts = con.execute(
            """
            select
                count(*) as n_spectra,
                count(distinct sample_id) as n_samples,
                count(distinct patient_id) as n_patients
            from biosample_metadata
            where dataset_id = ?
            """,
            [dataset_id],
        ).fetchdf().iloc[0].to_dict()

    label_df.to_csv(table_dir / "label_summary.csv", index=False)
    write_csv(
        table_dir / "final_counts.csv",
        [
            {
                "dataset_id": dataset_id,
                "title": INGESTED_CONFIG[dataset_id]["title"],
                "role": INGESTED_CONFIG[dataset_id]["role"],
                "n_spectra": int(counts["n_spectra"]),
                "n_samples": int(counts["n_samples"]),
                "n_patients": int(counts["n_patients"]),
                "source_records": int(manifest_df["article_id"].nunique()),
            }
        ],
    )

    provenance_text = "\n".join(
        [
            f"# {INGESTED_CONFIG[dataset_id]['title']} Provenance Note",
            "",
            f"- Dataset ID: `{dataset_id}`",
            "- FC ID: `FC32`",
            "- Role: `augmentation_only`",
            f"- Source URLs: {INGESTED_CONFIG[dataset_id]['source_urls']}",
            f"- Raw root: `{dataset_root}`",
            f"- Canonical manifest: `{manifest_path}`",
            f"- DuckDB rows written: {int(counts['n_spectra'])} spectra across {int(counts['n_samples'])} sample groups",
            f"- Provenance summary: {INGESTED_CONFIG[dataset_id]['summary_note']}",
        ]
    )
    (report_dir / "provenance_note.md").write_text(provenance_text + "\n", encoding="utf-8")

    return {
        "dataset_id": dataset_id,
        "fc_id": "FC32",
        "title": INGESTED_CONFIG[dataset_id]["title"],
        "role": INGESTED_CONFIG[dataset_id]["role"],
        "status": "ingested_wave_2",
        "n_spectra": int(counts["n_spectra"]),
        "n_samples": int(counts["n_samples"]),
        "n_patients": int(counts["n_patients"]),
        "label_schema": INGESTED_CONFIG[dataset_id]["label_schema"],
        "domain_tags": INGESTED_CONFIG[dataset_id]["domain_tags"],
        "notes": INGESTED_CONFIG[dataset_id]["summary_note"],
    }


def build_wave2_summary(rows: list[dict]) -> str:
    lines = [
        "# Wave 2 Corpus Summary",
        "",
        "This summary reflects the executed Wave 2 continuation focused on the two highest-value pending biological datasets: `stroke_urine_sers` and `ucla_saliva_sev_gc`.",
        "",
        "## Fully Ingested in This Continuation",
        "",
        "| Dataset ID | FC | Role | Spectra | Samples | Patients | Domains |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row["status"] == "ingested_wave_2":
            lines.append(
                f"| {row['dataset_id']} | {row['fc_id']} | {row['role']} | {row['n_spectra']} | "
                f"{row['n_samples']} | {row['n_patients']} | {row['domain_tags']} |"
            )

    lines.extend(
        [
            "",
            "## Still Pending",
            "",
            "| Dataset ID | FC | Role | Status | Notes |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        if row["status"] != "ingested_wave_2":
            lines.append(
                f"| {row['dataset_id']} | {row['fc_id']} | {row['role']} | {row['status']} | {row['notes']} |"
            )

    lines.extend(
        [
            "",
            "## Biological Coverage Added in Wave 2",
            "",
            "- `stroke_urine_sers` adds a human urine disease cohort with explicit stroke/control labels.",
            "- `ucla_saliva_sev_gc` adds a reconstructed saliva small-EV gastric-cancer cohort with shard-level provenance.",
            "- Together with Wave 1, Global v2 now spans plasma, urine, saliva EVs, single-vesicle EVs, faecal spectra, and pathogen NA-SERS.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_freeze_checkpoint() -> str:
    return "\n".join(
        [
            "# Global v2 Corpus Freeze Checkpoint",
            "",
            "1. Are the current core-training datasets sufficient to begin Global v2?",
            "Yes. The core lane now covers human plasma, human urine, pathogen spectra, and single-vesicle EV spectra, which is enough biological and technical diversity to start encoder planning and baseline training.",
            "",
            "2. Are augmentation datasets sufficient to begin Global v2?",
            "Yes. The current augmentation lane is not exhaustive, but it is sufficient to begin because it already includes faecal disease spectra and a reconstructed saliva small-EV gastric-cancer cohort.",
            "",
            "3. Which remaining datasets are blocking vs nice-to-have?",
            "Blocking: none for starting the encoder build.",
            "Nice-to-have: `stemcell_diff_mito_sers` and `tumor_purine_secretome_sers` for state/perturbation breadth; `acs_platelet_sers` is lower priority and not blocking.",
            "",
            "4. Should we proceed to encoder build now, or wait?",
            "Proceed now. The corpus is strong enough to freeze for initial Global v2 encoder planning while the remaining augmentation datasets stay on a parallel reconstruction track.",
        ]
    ) + "\n"


def main() -> None:
    storage = ensure_storage_dirs()
    raw_root = resolve_storage_path(storage.get("raw_data"))
    db_path = resolve_storage_path(storage.get("database"))
    if raw_root is None or db_path is None:
        raise RuntimeError("Storage config did not resolve raw_data/database paths.")

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    (WAVE2_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    (WAVE2_ROOT / "report").mkdir(parents=True, exist_ok=True)

    summary_rows = [
        build_stroke_outputs(raw_root=raw_root, db_path=db_path),
        build_ucla_outputs(raw_root=raw_root, db_path=db_path),
        *PENDING_ROWS,
    ]

    registry_change_rows = [
        {
            "change_id": "wave2_chg_001",
            "target_file": "data/registry/datasets.csv",
            "change_type": "status_update",
            "dataset_ids": "stroke_urine_sers; ucla_saliva_sev_gc",
            "status": "applied",
            "notes": "Updated dataset registry rows from todo to ingested_wave_2 and recorded ingest-specific notes.",
        },
        {
            "change_id": "wave2_chg_002",
            "target_file": "config/gaira_dataset_experiment_registry_v2.csv",
            "change_type": "existing_rows_verified",
            "dataset_ids": "stroke_urine_sers; ucla_saliva_sev_gc",
            "status": "applied",
            "notes": "Wave 2 experiment rows already existed and were reused without lane changes.",
        },
        {
            "change_id": "wave2_chg_003",
            "target_file": "src/gaira/parsers/biosample/stroke_urine_sers_parser.py",
            "change_type": "new_parser",
            "dataset_ids": "stroke_urine_sers",
            "status": "applied",
            "notes": "Added matrix-backed stroke urine parser for `data.rar::CBI/data.csv` with explicit sidecar caveat handling.",
        },
        {
            "change_id": "wave2_chg_004",
            "target_file": "src/gaira/parsers/biosample/ucla_saliva_sev_gc_parser.py",
            "change_type": "parser_activation",
            "dataset_ids": "ucla_saliva_sev_gc",
            "status": "applied",
            "notes": "Used the new grouped-family shard parser with canonical shard manifest provenance.",
        },
        {
            "change_id": "wave2_chg_005",
            "target_file": "scripts/ingest_dataset.py",
            "change_type": "wire_new_parsers",
            "dataset_ids": "stroke_urine_sers; ucla_saliva_sev_gc",
            "status": "applied",
            "notes": "Added required raw-file checks and biosample dispatch for both Wave 2 priority datasets.",
        },
        {
            "change_id": "wave2_chg_006",
            "target_file": "scripts/load_registry.py -> datasets table",
            "change_type": "reload_registry",
            "dataset_ids": "stroke_urine_sers; ucla_saliva_sev_gc",
            "status": "applied",
            "notes": "Reloaded dataset registry into DuckDB before the priority Wave 2 ingests.",
        },
    ]

    write_csv(WAVE2_ROOT / "tables" / "wave2_registry_changes_applied.csv", registry_change_rows)
    write_csv(WAVE2_ROOT / "tables" / "wave2_corpus_summary.csv", summary_rows)

    summary_md = WAVE2_ROOT / "report" / "wave2_corpus_summary.md"
    summary_md.write_text(build_wave2_summary(summary_rows), encoding="utf-8")
    build_pdf_report(summary_md, [], summary_md.with_suffix(".pdf"))

    freeze_md = WAVE2_ROOT / "report" / "global_v2_corpus_freeze_checkpoint.md"
    freeze_md.write_text(build_freeze_checkpoint(), encoding="utf-8")

    print(f"Wrote Wave 2 artifacts under {WAVE2_ROOT}")


if __name__ == "__main__":
    main()
