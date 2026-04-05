from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TMP_QUEUE = ROOT / "tmp" / "global_v2_ingest_queue"
TMP_OUT = ROOT / "tmp" / "global_v2_ingest_outputs"

sys.path.insert(0, str(ROOT / "src"))
from gaira.config import ensure_storage_dirs, resolve_storage_path  # noqa: E402
from gaira.demo.autoresearch_utils import build_pdf_report  # noqa: E402


DATASET_CONFIG = {
    "mycoplasma_na_sers": {
        "fc_id": "FC14",
        "title": "Mycoplasma NA-SERS Pathogen Panel",
        "role": "core_training",
        "ingest_status": "ingested_wave_1",
        "domain_tags": "pathogen; bacteria; mycoplasma",
        "label_schema": "species; strain; isolate id",
        "include_in_shared_encoder": "yes",
        "exclude_from_supervised_benchmarks": "no",
        "source_urls": "https://zenodo.org/records/4941488",
        "raw_assets": [
            {"file_name": "NA-SERS specificity spectra.csv", "asset_type": "csv", "keep_or_exclude": "keep", "notes": "Primary raw pathogen matrix"},
        ],
        "exclusions": [
            {"exclusion_type": "class_exclusion", "item": "Bkg", "reason": "technical background controls excluded from biosample ingest"},
            {"exclusion_type": "class_exclusion", "item": "Media Ctl", "reason": "media-only controls excluded from biosample ingest"},
            {"exclusion_type": "support_file_exclusion", "item": "figshare stats/performance tables", "reason": "method-support only, not raw spectra"},
        ],
        "provenance_note": "Ingested from the consolidated Zenodo CSV only. The companion figshare statistics tables were intentionally not imported into DuckDB.",
    },
    "coeliac_faecal_sers": {
        "fc_id": "FC17",
        "title": "Coeliac Faecal SERS Cohort",
        "role": "augmentation_only",
        "ingest_status": "ingested_wave_1",
        "domain_tags": "faeces; gut; coeliac",
        "label_schema": "CTR; CD; GFD; sex and age encoded in filenames",
        "include_in_shared_encoder": "yes",
        "exclude_from_supervised_benchmarks": "no",
        "source_urls": "https://zenodo.org/records/5947010",
        "raw_assets": [
            {"file_name": "coeliac_faecal_sers.zip", "asset_type": "zip", "keep_or_exclude": "keep", "notes": "Primary faecal cohort archive"},
        ],
        "exclusions": [
            {"exclusion_type": "member_exclusion", "item": "pure metabolites dataset/*.txt", "reason": "pure standards kept out of biosample cohort lane"},
            {"exclusion_type": "sidecar_exclusion", "item": "OTU_table.RData; OTU_table.xlsx; faecal_dataset.RData; metabolites_dataset.RData; R_code.R", "reason": "non-spectral or duplicated analytical sidecars"},
        ],
        "provenance_note": "The parser preserves only the faecal cohort BWSpec text files under `faecal samples dataset/`. Pure metabolite references and R sidecars are documented but excluded from the biosample lane.",
    },
    "ovarian_plasma_raman_sers": {
        "fc_id": "FC23",
        "title": "Ovarian Plasma Raman and SERS Cohort",
        "role": "core_training",
        "ingest_status": "ingested_wave_1",
        "domain_tags": "plasma; ovarian cancer; raman; sers",
        "label_schema": "healthy vs ovarian_cancer; modality; donor id; replicate id",
        "include_in_shared_encoder": "yes",
        "exclude_from_supervised_benchmarks": "no",
        "source_urls": "https://figshare.com/articles/dataset/Raman_spectroscopic_techniques_to_detect_ovarian_cancer_biomarkers_in_blood_plasma/6744206",
        "raw_assets": [
            {"file_name": "Raman dataset.zip", "asset_type": "zip", "keep_or_exclude": "keep", "notes": "Primary spontaneous-Raman plasma archive"},
            {"file_name": "SERS dataset.zip", "asset_type": "zip", "keep_or_exclude": "keep", "notes": "Primary plasma SERS archive"},
        ],
        "exclusions": [
            {"exclusion_type": "policy_note", "item": "Raman and SERS subsets", "reason": "kept in one dataset_id but must remain modality-tagged for analysis"},
            {"exclusion_type": "future_qc", "item": "broken or duplicate txt members if discovered later", "reason": "none removed in this Wave 1 ingest pass"},
        ],
        "provenance_note": "Both Figshare zip archives were ingested intact with modality preserved as `subclass_label`. No processed or figure-only sidecars were mixed into the ingest.",
    },
    "single_vesicle_ev_raman": {
        "fc_id": "FC35",
        "title": "Single-Vesicle EV Raw Raman Archive",
        "role": "core_training",
        "ingest_status": "ingested_wave_1",
        "domain_tags": "EV; single_particle; heterogeneity",
        "label_schema": "control; hras7; hras8; hras9; mapping id; scan id",
        "include_in_shared_encoder": "yes",
        "exclude_from_supervised_benchmarks": "no",
        "source_urls": "https://figshare.com/articles/dataset/Raw_Raman_data_/26059145?file=47123702",
        "raw_assets": [
            {"file_name": "fc35_raw_raman_data.rar", "asset_type": "rar", "keep_or_exclude": "keep", "notes": "Primary raw single-vesicle archive"},
        ],
        "exclusions": [
            {"exclusion_type": "policy_note", "item": "filename-derived labels", "reason": "preserved but treated as archive-level metadata rather than gold-standard biology"},
            {"exclusion_type": "future_qc", "item": "malformed txt members or failed hits", "reason": "none removed in this pass; parser accepted 525 members"},
        ],
        "provenance_note": "The ingest reads the original RAR members directly via `bsdtar -xOf` and preserves the member path in `source_file`. Labels were reconstructed from filenames only.",
    },
}


PENDING_DATASETS = [
    {
        "dataset_id": "stroke_urine_sers",
        "fc_id": "FC08",
        "title": "Ischemic-Stroke Urine SERS Cohort",
        "role": "core_training",
        "ingest_status": "pending_reconstruction",
        "domain_tags": "urine; stroke; human",
        "label_schema": "stroke vs healthy; participant id",
        "include_in_shared_encoder": "planned_yes",
        "exclude_from_supervised_benchmarks": "no",
        "notes": "Blocked on unpacking `data.rar` from the newer Zenodo release and separating spectra from imaging/results assets.",
    },
    {
        "dataset_id": "stemcell_diff_mito_sers",
        "fc_id": "FC05",
        "title": "Stem-Cell Differentiation Mitochondrial SERS Trajectory",
        "role": "augmentation_only",
        "ingest_status": "pending_reconstruction",
        "domain_tags": "cell_state; differentiation; probe-mediated",
        "label_schema": "six differentiation states",
        "include_in_shared_encoder": "planned_yes",
        "exclude_from_supervised_benchmarks": "no",
        "notes": "RAR reconstruction and probe/control exclusion logic still pending.",
    },
    {
        "dataset_id": "tumor_purine_secretome_sers",
        "fc_id": "FC16",
        "title": "Tumor Purine Secretome SERS Archive",
        "role": "augmentation_only",
        "ingest_status": "pending_reconstruction",
        "domain_tags": "secretome; cell_line; perturbation",
        "label_schema": "cell type; MTA condition; timepoint",
        "include_in_shared_encoder": "planned_yes",
        "exclude_from_supervised_benchmarks": "no",
        "notes": "Needs biological-spectrum extraction from the mixed analytical zip while excluding RNA-seq, LC-MS, and support sidecars.",
    },
    {
        "dataset_id": "acs_platelet_sers",
        "fc_id": "FC24",
        "title": "ACS Platelet SERS Workbook Family",
        "role": "augmentation_only",
        "ingest_status": "pending_reconstruction",
        "domain_tags": "blood-derived; cardiovascular",
        "label_schema": "sample id; replicate blocks",
        "include_in_shared_encoder": "planned_yes",
        "exclude_from_supervised_benchmarks": "no",
        "notes": "Workbook block structure still needs custom parsing and deduplication across the two related files.",
    },
    {
        "dataset_id": "ucla_saliva_sev_gc",
        "fc_id": "FC32",
        "title": "UCLA Saliva sEV Gastric-Cancer Cohort",
        "role": "augmentation_only",
        "ingest_status": "pending_reconstruction",
        "domain_tags": "saliva; EV; gastric cancer",
        "label_schema": "patient id; disease class",
        "include_in_shared_encoder": "planned_yes",
        "exclude_from_supervised_benchmarks": "no",
        "notes": "Still blocked on pulling and merging the fragmented Figshare shard family into one cohort.",
    },
    {
        "dataset_id": "tear_dopamine_sers_support",
        "fc_id": "FC18",
        "title": "Tear Dopamine SERS Source-Data Support",
        "role": "grounding_only",
        "ingest_status": "pending_support_import",
        "domain_tags": "assay; dopamine; support",
        "label_schema": "dopamine concentration; assay condition",
        "include_in_shared_encoder": "no",
        "exclude_from_supervised_benchmarks": "yes",
        "notes": "Deferred to Wave 3 so support data do not bleed into the core biosample lane.",
    },
    {
        "dataset_id": "sertraline_serotonin_sers_support",
        "fc_id": "FC20",
        "title": "Sertraline and Serotonin Nanocone SERS Support",
        "role": "grounding_only",
        "ingest_status": "pending_support_import",
        "domain_tags": "assay; serotonin; sertraline; support",
        "label_schema": "analyte identity; platform condition",
        "include_in_shared_encoder": "no",
        "exclude_from_supervised_benchmarks": "yes",
        "notes": "Deferred to Wave 3 support-only import.",
    },
    {
        "dataset_id": "rbc_membrane_ev_sers_support",
        "fc_id": "FC25",
        "title": "RBC Membrane and EV Nanoscale SERS Support",
        "role": "grounding_only",
        "ingest_status": "pending_support_import",
        "domain_tags": "membrane; EV; support",
        "label_schema": "modality; membrane-reference identity",
        "include_in_shared_encoder": "no",
        "exclude_from_supervised_benchmarks": "yes",
        "notes": "Deferred to Wave 3 because the readily usable content is support-heavy and the TERS OPJU needs separate conversion.",
    },
]


REGISTRY_CHANGE_ROWS = [
    {
        "change_id": "chg_001",
        "target_file": "data/registry/datasets.csv",
        "change_type": "append_dataset_rows",
        "dataset_ids": "mycoplasma_na_sers; ovarian_plasma_raman_sers; coeliac_faecal_sers; single_vesicle_ev_raman",
        "status": "applied",
        "notes": "Added Wave 1 dataset registry rows with source URLs, modality/sample metadata, and Global v2 notes.",
    },
    {
        "change_id": "chg_002",
        "target_file": "config/gaira_dataset_experiment_registry_v2.csv",
        "change_type": "append_experiment_rows",
        "dataset_ids": "mycoplasma_na_sers; ovarian_plasma_raman_sers; coeliac_faecal_sers; single_vesicle_ev_raman",
        "status": "applied",
        "notes": "Added Wave 1 experiment registry rows with subset aliases and lane defaults.",
    },
    {
        "change_id": "chg_003",
        "target_file": "scripts/ingest_dataset.py",
        "change_type": "wire_new_parsers",
        "dataset_ids": "mycoplasma_na_sers; ovarian_plasma_raman_sers; coeliac_faecal_sers; single_vesicle_ev_raman",
        "status": "applied",
        "notes": "Added required raw-file checks, parser imports, and biosample dispatch branches.",
    },
    {
        "change_id": "chg_004",
        "target_file": "scripts/load_registry.py -> datasets table",
        "change_type": "reload_registry",
        "dataset_ids": "mycoplasma_na_sers; ovarian_plasma_raman_sers; coeliac_faecal_sers; single_vesicle_ev_raman",
        "status": "applied",
        "notes": "Reloaded dataset registry into DuckDB after appending new dataset rows.",
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


def file_size(path: Path) -> int:
    return int(path.stat().st_size) if path.exists() else 0


def archive_member_count(path: Path) -> int | None:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with ZipFile(path, "r") as archive:
            return len(archive.namelist())
    if suffix == ".rar":
        result = subprocess.run(["bsdtar", "-tf", str(path)], check=True, capture_output=True, text=True)
        return len([line for line in result.stdout.splitlines() if line.strip()])
    return None


def build_dataset_artifacts(raw_root: Path, db_path: Path) -> list[dict]:
    corpus_rows: list[dict] = []
    with duckdb.connect(str(db_path)) as con:
        for dataset_id, cfg in DATASET_CONFIG.items():
            dataset_root = raw_root / dataset_id
            out_root = TMP_OUT / "global_v2_ingest" / dataset_id
            table_dir = out_root / "tables"
            report_dir = out_root / "report"
            table_dir.mkdir(parents=True, exist_ok=True)
            report_dir.mkdir(parents=True, exist_ok=True)

            inventory_rows = []
            for asset in cfg["raw_assets"]:
                asset_path = dataset_root / asset["file_name"]
                inventory_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "file_name": asset["file_name"],
                        "asset_type": asset["asset_type"],
                        "size_bytes": file_size(asset_path),
                        "archive_member_count": archive_member_count(asset_path),
                        "keep_or_exclude": asset["keep_or_exclude"],
                        "notes": asset["notes"],
                    }
                )
            write_csv(table_dir / "input_inventory.csv", inventory_rows)
            write_csv(table_dir / "exclusion_summary.csv", cfg["exclusions"])

            counts_row = con.execute(
                """
                select
                    count(*) as n_spectra,
                    count(distinct sample_id) as n_samples,
                    count(distinct patient_id) as n_patients,
                    count(distinct subclass_label) as n_subclasses
                from biosample_metadata
                where dataset_id = ?
                """,
                [dataset_id],
            ).fetchdf().iloc[0].to_dict()

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
            label_df.to_csv(table_dir / "label_summary.csv", index=False)

            final_counts_rows = [
                {
                    "dataset_id": dataset_id,
                    "title": cfg["title"],
                    "role": cfg["role"],
                    "n_spectra": int(counts_row["n_spectra"]),
                    "n_samples": int(counts_row["n_samples"]),
                    "n_patients": int(counts_row["n_patients"]),
                    "n_subclasses": int(counts_row["n_subclasses"]),
                }
            ]
            write_csv(table_dir / "final_counts.csv", final_counts_rows)

            provenance_lines = [
                f"# {cfg['title']} Provenance Note",
                "",
                f"- Dataset ID: `{dataset_id}`",
                f"- FC ID: `{cfg['fc_id']}`",
                f"- Role: `{cfg['role']}`",
                f"- Source URLs: {cfg['source_urls']}",
                f"- Raw root: `{dataset_root}`",
                f"- DuckDB rows written: {int(counts_row['n_spectra'])} spectra",
                f"- Provenance summary: {cfg['provenance_note']}",
            ]
            (report_dir / "provenance_note.md").write_text("\n".join(provenance_lines) + "\n", encoding="utf-8")

            corpus_rows.append(
                {
                    "dataset_id": dataset_id,
                    "fc_id": cfg["fc_id"],
                    "title": cfg["title"],
                    "role": cfg["role"],
                    "ingest_status": cfg["ingest_status"],
                    "n_spectra": int(counts_row["n_spectra"]),
                    "n_samples": int(counts_row["n_samples"]),
                    "n_patients": int(counts_row["n_patients"]),
                    "label_schema": cfg["label_schema"],
                    "domain_tags": cfg["domain_tags"],
                    "include_in_shared_encoder": cfg["include_in_shared_encoder"],
                    "exclude_from_supervised_benchmarks": cfg["exclude_from_supervised_benchmarks"],
                    "notes": cfg["provenance_note"],
                }
            )
    return corpus_rows


def build_corpus_report(corpus_rows: list[dict]) -> str:
    lines = [
        "# Global v2 Corpus Summary",
        "",
        "This summary reflects the current execution state of the Global v2 ingest expansion. Wave 1 core additions are ingested into DuckDB. Wave 2 and Wave 3 remain explicitly pending rather than silently approximated.",
        "",
        "## Newly Ingested Wave 1 Datasets",
        "",
        "| Dataset ID | FC | Role | Spectra | Samples | Patients | Domains |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    ingested_rows = [row for row in corpus_rows if row["ingest_status"] == "ingested_wave_1"]
    for row in ingested_rows:
        lines.append(
            f"| {row['dataset_id']} | {row['fc_id']} | {row['role']} | {row['n_spectra']} | "
            f"{row['n_samples']} | {row['n_patients']} | {row['domain_tags']} |"
        )

    lines.extend(
        [
            "",
            "## Pending Datasets",
            "",
            "| Dataset ID | FC | Role | Status | Notes |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in PENDING_DATASETS:
        lines.append(
            f"| {row['dataset_id']} | {row['fc_id']} | {row['role']} | {row['ingest_status']} | {row['notes']} |"
        )

    lines.extend(
        [
            "",
            "## Corpus State",
            "",
            "- Core training additions now available in DuckDB: `mycoplasma_na_sers`, `ovarian_plasma_raman_sers`, `single_vesicle_ev_raman`.",
            "- Augmentation additions now available in DuckDB: `coeliac_faecal_sers`.",
            "- Support-only datasets remain intentionally un-ingested in this execution pass so they do not contaminate the shared encoder lane.",
            "- Reconstruction-heavy datasets remain queued with explicit blocks instead of placeholder ingests.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    storage = ensure_storage_dirs()
    raw_root = resolve_storage_path(storage.get("raw_data"))
    db_path = resolve_storage_path(storage.get("database"))
    if raw_root is None or db_path is None:
        raise RuntimeError("Storage config did not resolve raw_data/database paths.")

    corpus_rows = build_dataset_artifacts(raw_root=raw_root, db_path=db_path)
    corpus_rows.extend(PENDING_DATASETS)

    registry_changes_path = TMP_OUT / "tables" / "global_v2_registry_changes_applied.csv"
    write_csv(registry_changes_path, REGISTRY_CHANGE_ROWS)

    corpus_summary_path = TMP_OUT / "tables" / "global_v2_corpus_summary.csv"
    write_csv(corpus_summary_path, corpus_rows)

    report_md = TMP_OUT / "report" / "global_v2_corpus_summary.md"
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(build_corpus_report(corpus_rows), encoding="utf-8")
    build_pdf_report(report_md, [], report_md.with_suffix(".pdf"))

    print(f"Wrote ingest artifacts under {TMP_OUT}")


if __name__ == "__main__":
    main()
