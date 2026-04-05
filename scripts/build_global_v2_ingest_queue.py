from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TMP_OUT = ROOT / "tmp" / "global_v2_ingest_queue"

sys.path.insert(0, str(ROOT / "src"))
from gaira.demo.autoresearch_utils import build_pdf_report  # noqa: E402


QUEUE_ROWS = [
    {
        "fc_id": "FC35",
        "dataset_id": "single_vesicle_ev_raman",
        "cleaned_dataset_title": "Single-Vesicle EV Raw Raman Archive",
        "final_role": "core_training",
        "source_urls": "https://figshare.com/articles/dataset/Raw_Raman_data_/26059145?file=47123702",
        "expected_raw_asset_types": "rar; txt",
        "parser_type_needed": "rar/zip reconstruction",
        "reconstruction_difficulty": "medium",
        "likely_exclusions": "non-EV failures; malformed txt files; empty acquisitions if found",
        "expected_biosample_type": "EV; single_particle",
        "expected_label_schema": "condition label from filename; mapping id; acquisition metadata",
        "target_gaira_location": "raw/single_vesicle_ev_raman + biosample parser + core Global v2 lane",
        "required_registry_files_to_update": "data/registry/datasets.csv; config/gaira_dataset_experiment_registry_v2.csv",
        "verification_checks_needed": "txt count; per-condition counts; shared axis check; malformed file log; final spectra/sample counts",
        "recommended_ingest_wave": "wave_1",
    },
    {
        "fc_id": "FC23",
        "dataset_id": "ovarian_plasma_raman_sers",
        "cleaned_dataset_title": "Ovarian Plasma Raman and SERS Cohort",
        "final_role": "core_training",
        "source_urls": "https://figshare.com/articles/dataset/Raman_spectroscopic_techniques_to_detect_ovarian_cancer_biomarkers_in_blood_plasma/6744206",
        "expected_raw_asset_types": "zip; txt",
        "parser_type_needed": "rar/zip reconstruction",
        "reconstruction_difficulty": "medium",
        "likely_exclusions": "duplicate processed copies if present; broken txt files; keep Raman and SERS modality-tagged",
        "expected_biosample_type": "human plasma",
        "expected_label_schema": "healthy vs ovarian_cancer; modality; donor id; replicate id",
        "target_gaira_location": "raw/ovarian_plasma_raman_sers + biosample parser + core Global v2 lane",
        "required_registry_files_to_update": "data/registry/datasets.csv; config/gaira_dataset_experiment_registry_v2.csv",
        "verification_checks_needed": "archive inventory; per-modality counts; per-class counts; donor count; shared axis coverage",
        "recommended_ingest_wave": "wave_1",
    },
    {
        "fc_id": "FC08",
        "dataset_id": "stroke_urine_sers",
        "cleaned_dataset_title": "Ischemic-Stroke Urine SERS Cohort",
        "final_role": "core_training",
        "source_urls": "https://zenodo.org/records/19109120 | https://zenodo.org/records/19369604",
        "expected_raw_asset_types": "rar; docx",
        "parser_type_needed": "grouped cohort reconstruction",
        "reconstruction_difficulty": "high",
        "likely_exclusions": "code.rar; results.rar; README.docx; carotid-ultrasound image assets",
        "expected_biosample_type": "human urine",
        "expected_label_schema": "stroke vs healthy; participant id; possibly image linkage",
        "target_gaira_location": "raw/stroke_urine_sers + biosample parser + core Global v2 lane",
        "required_registry_files_to_update": "data/registry/datasets.csv; config/gaira_dataset_experiment_registry_v2.csv",
        "verification_checks_needed": "newer-vs-older version parity; data.rar inventory; spectra count; participant count; image separation",
        "recommended_ingest_wave": "wave_2",
    },
    {
        "fc_id": "FC14",
        "dataset_id": "mycoplasma_na_sers",
        "cleaned_dataset_title": "Mycoplasma NA-SERS Pathogen Panel",
        "final_role": "core_training",
        "source_urls": "https://zenodo.org/records/4941488 | https://figshare.com/articles/dataset/_PLS_DA_of_NA_SERS_specificity_and_sensitivity_in_discriminating_M_pneumoniae_strains_/493381?file=823019 | https://figshare.com/articles/dataset/_Cross_validated_PLS_DA_modeling_statistics_for_the_prediction_performance_for_NA_SERS_typing_of_individual_type_1_and_2_M_pneumoniae_clinical_isolates_/1467505?file=2154588",
        "expected_raw_asset_types": "csv; xls",
        "parser_type_needed": "csv/xlsx parser",
        "reconstruction_difficulty": "low",
        "likely_exclusions": "figshare stats tables; Bkg columns; Media Ctl columns",
        "expected_biosample_type": "pathogen",
        "expected_label_schema": "species; strain; isolate id; control class",
        "target_gaira_location": "raw/mycoplasma_na_sers + biosample parser + core Global v2 lane",
        "required_registry_files_to_update": "data/registry/datasets.csv; config/gaira_dataset_experiment_registry_v2.csv",
        "verification_checks_needed": "CSV width; class counts; control-class exclusion summary; axis monotonicity",
        "recommended_ingest_wave": "wave_1",
    },
    {
        "fc_id": "FC05",
        "dataset_id": "stemcell_diff_mito_sers",
        "cleaned_dataset_title": "Stem-Cell Differentiation Mitochondrial SERS Trajectory",
        "final_role": "augmentation_only",
        "source_urls": "https://zenodo.org/records/10851312",
        "expected_raw_asset_types": "rar; code",
        "parser_type_needed": "rar/zip reconstruction",
        "reconstruction_difficulty": "high",
        "likely_exclusions": "code-only files; model artifacts; probe blanks and QC failures if present",
        "expected_biosample_type": "cell_state",
        "expected_label_schema": "six differentiation states; replicate ids",
        "target_gaira_location": "raw/stemcell_diff_mito_sers + biosample parser + augmentation lane",
        "required_registry_files_to_update": "data/registry/datasets.csv; config/gaira_dataset_experiment_registry_v2.csv",
        "verification_checks_needed": "rar inventory; state counts; probe-driven control separation; usable spectra count",
        "recommended_ingest_wave": "wave_2",
    },
    {
        "fc_id": "FC16",
        "dataset_id": "tumor_purine_secretome_sers",
        "cleaned_dataset_title": "Tumor Purine Secretome SERS Archive",
        "final_role": "augmentation_only",
        "source_urls": "https://zenodo.org/records/8130216",
        "expected_raw_asset_types": "zip; txt; xlsx",
        "parser_type_needed": "grouped cohort reconstruction",
        "reconstruction_difficulty": "high",
        "likely_exclusions": "RNA-seq tables; LC-MS workbooks; UV-Vis images; pure-standard rows from core bio lane",
        "expected_biosample_type": "secretome; cell_line",
        "expected_label_schema": "cell type; MTA condition; timepoint; coculture context",
        "target_gaira_location": "raw/tumor_purine_secretome_sers + biosample parser + augmentation lane",
        "required_registry_files_to_update": "data/registry/datasets.csv; config/gaira_dataset_experiment_registry_v2.csv",
        "verification_checks_needed": "txt inventory; biological-vs-sidecar split; condition counts; timepoint counts",
        "recommended_ingest_wave": "wave_2",
    },
    {
        "fc_id": "FC17",
        "dataset_id": "coeliac_faecal_sers",
        "cleaned_dataset_title": "Coeliac Faecal SERS Cohort",
        "final_role": "augmentation_only",
        "source_urls": "https://zenodo.org/records/5947010",
        "expected_raw_asset_types": "zip; txt; RData; xlsx",
        "parser_type_needed": "rar/zip reconstruction",
        "reconstruction_difficulty": "low",
        "likely_exclusions": "pure metabolite txt references from core cohort lane; OTU tables; RData; R code",
        "expected_biosample_type": "human faeces",
        "expected_label_schema": "CTR; CD; GFD; sex; age; sample id",
        "target_gaira_location": "raw/coeliac_faecal_sers + biosample parser + augmentation lane",
        "required_registry_files_to_update": "data/registry/datasets.csv; config/gaira_dataset_experiment_registry_v2.csv",
        "verification_checks_needed": "sample counts by class; metadata parse from filename; BWSpec numeric section parse",
        "recommended_ingest_wave": "wave_1",
    },
    {
        "fc_id": "FC24",
        "dataset_id": "acs_platelet_sers",
        "cleaned_dataset_title": "ACS Platelet SERS Workbook Family",
        "final_role": "augmentation_only",
        "source_urls": "https://figshare.com/articles/dataset/SERS_spectra_of_43_patients_with_ACS_xlsx/24747531?file=43481136 | https://figshare.com/articles/dataset/Suplementary_material_DIB_ACS_40_samples_xlsx/24564787?file=43183257",
        "expected_raw_asset_types": "xlsx",
        "parser_type_needed": "grouped cohort reconstruction",
        "reconstruction_difficulty": "high",
        "likely_exclusions": "empty tabs; duplicated blocks across workbooks; non-spectral helper columns",
        "expected_biosample_type": "blood-derived clinical samples",
        "expected_label_schema": "sample id; replicate blocks; ACS cohort labels if recoverable",
        "target_gaira_location": "raw/acs_platelet_sers + biosample parser + augmentation lane",
        "required_registry_files_to_update": "data/registry/datasets.csv; config/gaira_dataset_experiment_registry_v2.csv",
        "verification_checks_needed": "workbook overlap resolution; replicate-block parser; patient/sample count reconstruction",
        "recommended_ingest_wave": "wave_2",
    },
    {
        "fc_id": "FC32",
        "dataset_id": "ucla_saliva_sev_gc",
        "cleaned_dataset_title": "UCLA Saliva sEV Gastric-Cancer Cohort",
        "final_role": "augmentation_only",
        "source_urls": "https://figshare.com/articles/dataset/Health_control_01_sEV_Saliva_UCLA_ERCC_/20428395 | https://figshare.com/articles/dataset/GC_Patient_20_sEV_Saliva_UCLA_ERCC_/20427957 | https://figshare.com/articles/dataset/GC_Patient_19_sEV_Saliva_UCLA_ERCC_/20427954 | https://figshare.com/articles/dataset/GC_Patient_18_sEV_Saliva_UCLA_ERCC_/20427951 | https://figshare.com/articles/dataset/GC_Patient_17_sEV_Saliva_UCLA_ERCC_/20427948 | https://figshare.com/articles/dataset/GC_Patient_16_sEV_Saliva_UCLA_ERCC_/20427945 | https://figshare.com/articles/dataset/GC_Patient_15_sEV_Saliva_UCLA_ERCC_/20427939 | https://figshare.com/articles/dataset/GC_Patient_14_sEV_Saliva_UCLA_ERCC_/20427936 | https://figshare.com/articles/dataset/GC_Patient_13_sEV_Saliva_UCLA_ERCC_/20427933 | https://figshare.com/articles/dataset/GC_Patient_12_sEV_Saliva_UCLA_ERCC_/20427930 | https://figshare.com/articles/dataset/GC_Patient_11_sEV_Saliva_UCLA_ERCC_/20427927 | https://figshare.com/articles/dataset/GC_Patient_10_sEV_Saliva_UCLA_ERCC_/20427924 | https://figshare.com/articles/dataset/GC_Patient_9_sEV_Saliva_UCLA_ERCC_/20427921 | https://figshare.com/articles/dataset/GC_Patient_8_sEV_Saliva_UCLA_ERCC_/20427918 | https://figshare.com/articles/dataset/GC_Patient_7_sEV_Saliva_UCLA_ERCC_/20427909 | https://figshare.com/articles/dataset/GC_Patient_6_sEV_Saliva_UCLA_ERCC_/20427906 | https://figshare.com/articles/dataset/GC_Patient_5_sEV_Saliva_UCLA_ERCC_/20427903 | https://figshare.com/articles/dataset/GC_Patient_1_UG3/20282238 | https://figshare.com/articles/dataset/ERCC/20406102",
        "expected_raw_asset_types": "txt; grouped figshare shards",
        "parser_type_needed": "grouped cohort reconstruction",
        "reconstruction_difficulty": "high",
        "likely_exclusions": "ERCC metadata-only stub; malformed txt files; shard duplicates",
        "expected_biosample_type": "saliva; EV",
        "expected_label_schema": "gastric cancer vs control; patient id; shard id",
        "target_gaira_location": "raw/ucla_saliva_sev_gc + biosample parser + augmentation lane",
        "required_registry_files_to_update": "data/registry/datasets.csv; config/gaira_dataset_experiment_registry_v2.csv",
        "verification_checks_needed": "shard inventory; patient count; control coverage; duplicate suppression; merged cohort counts",
        "recommended_ingest_wave": "wave_2",
    },
    {
        "fc_id": "FC18",
        "dataset_id": "tear_dopamine_sers_support",
        "cleaned_dataset_title": "Tear Dopamine SERS Source-Data Support",
        "final_role": "grounding_only",
        "source_urls": "https://zenodo.org/records/18670010",
        "expected_raw_asset_types": "xlsx; tif; jpg",
        "parser_type_needed": "support-only import",
        "reconstruction_difficulty": "medium",
        "likely_exclusions": "HEPES; background; bare AuNS; image files; calibration-only sheets from direct grounding lane",
        "expected_biosample_type": "assay; reference",
        "expected_label_schema": "dopamine concentration; preparation condition",
        "target_gaira_location": "raw/tear_dopamine_sers_support + grounding parser + reserve support lane",
        "required_registry_files_to_update": "data/registry/datasets.csv",
        "verification_checks_needed": "sheet inventory; analyte-vs-background separation; support-only flagging",
        "recommended_ingest_wave": "wave_3",
    },
    {
        "fc_id": "FC20",
        "dataset_id": "sertraline_serotonin_sers_support",
        "cleaned_dataset_title": "Sertraline and Serotonin Nanocone SERS Support",
        "final_role": "grounding_only",
        "source_urls": "https://zenodo.org/records/18284194",
        "expected_raw_asset_types": "xlsx; tif; jpg",
        "parser_type_needed": "support-only import",
        "reconstruction_difficulty": "medium",
        "likely_exclusions": "CY3 controls; backgrounds; commercial-vs-platform benchmarking images",
        "expected_biosample_type": "assay; reference",
        "expected_label_schema": "analyte identity; concentration; platform condition",
        "target_gaira_location": "raw/sertraline_serotonin_sers_support + grounding parser + reserve support lane",
        "required_registry_files_to_update": "data/registry/datasets.csv",
        "verification_checks_needed": "sheet inventory; analyte-only subset; support-only flagging",
        "recommended_ingest_wave": "wave_3",
    },
    {
        "fc_id": "FC25",
        "dataset_id": "rbc_membrane_ev_sers_support",
        "cleaned_dataset_title": "RBC Membrane and EV Nanoscale SERS Support",
        "final_role": "grounding_only",
        "source_urls": "https://figshare.com/articles/dataset/Surface-enhanced_SERS_and_tip-enhanced_TERS_Raman_scattering_in_label-free_characterization_of_erythrocyte_membranes_and_extracellular_vesicles_in_nano-scale_and_at_the_single-molecule_level_/24105993",
        "expected_raw_asset_types": "zip; txt; opju; xlsx",
        "parser_type_needed": "support-only import",
        "reconstruction_difficulty": "high",
        "likely_exclusions": "TERS OPJU until converted cleanly; AFM size sheet; pure standard txt files from biosample lane",
        "expected_biosample_type": "membrane; EV; reference",
        "expected_label_schema": "modality; membrane-reference identity",
        "target_gaira_location": "raw/rbc_membrane_ev_sers_support + grounding parser + reserve support lane",
        "required_registry_files_to_update": "data/registry/datasets.csv",
        "verification_checks_needed": "txt vs OPJU split; RBC-ghost count; support-only separation",
        "recommended_ingest_wave": "wave_3",
    },
]


PARSER_ROWS = [
    {
        "dataset_id": row["dataset_id"],
        "fc_id": row["fc_id"],
        "parser_type_needed": row["parser_type_needed"],
        "implementation_scope": (
            "new biosample parser" if row["final_role"] != "grounding_only" else "new grounding parser"
        ),
        "expected_input_shape": row["expected_raw_asset_types"],
        "normalization_tasks": "preserve native axis; derive labels from archive path/file names; record modality and provenance",
        "explicit_exclusion_policy": row["likely_exclusions"],
        "verification_artifacts": row["verification_checks_needed"],
    }
    for row in QUEUE_ROWS
]


REGISTRY_PLAN_ROWS = [
    {
        "dataset_id": row["dataset_id"],
        "fc_id": row["fc_id"],
        "role": row["final_role"],
        "include_in_shared_encoder_training": "yes" if row["final_role"] != "grounding_only" else "no",
        "exclude_from_supervised_benchmarks": "yes" if row["final_role"] == "grounding_only" else "no",
        "planned_registry_updates": row["required_registry_files_to_update"],
        "planned_subset_alias": row["dataset_id"],
        "domain_tags": row["expected_biosample_type"],
        "notes": row["expected_label_schema"],
    }
    for row in QUEUE_ROWS
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report() -> str:
    lines = [
        "# Global v2 Ingest Queue",
        "",
        "This queue turns the reviewed Global v2 candidates into a concrete ingest plan with explicit dataset roles, parser needs, exclusions, registry touchpoints, and ingest waves.",
        "",
        "## Wave Plan",
        "",
        "- Wave 1: FC35, FC23, FC14, FC17",
        "- Wave 2: FC08, FC05, FC16, FC24, FC32",
        "- Wave 3: FC18, FC20, FC25",
        "",
        "Wave 1 stays first because these four have confirmed usable spectra plus manageable parser shapes. Wave 2 remains reconstruction-heavy. Wave 3 is intentionally isolated so support/grounding assets do not drift into the core lane.",
        "",
        "## Queue Table",
        "",
        "| FC | Dataset ID | Role | Parser | Difficulty | Biosample | Wave |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in QUEUE_ROWS:
        lines.append(
            f"| {row['fc_id']} | {row['dataset_id']} | {row['final_role']} | {row['parser_type_needed']} | "
            f"{row['reconstruction_difficulty']} | {row['expected_biosample_type']} | {row['recommended_ingest_wave']} |"
        )
    lines.extend(
        [
            "",
            "## Operational Notes",
            "",
            "- `core_training` datasets should be wired into the biosample stack and experiment registry.",
            "- `augmentation_only` datasets should be ingested into the same biosample tables but documented as non-core Global v2 additions until reconstruction is complete.",
            "- `grounding_only` datasets should stay out of the shared encoder training pool and should only land in dedicated grounding/support handling.",
            "- Rejected candidates are intentionally omitted from implementation and should only appear in exclusion notes.",
            "",
            "## Registry Update Intent",
            "",
            "- `data/registry/datasets.csv`: add all selected datasets with explicit role/domain notes and raw-availability expectations.",
            "- `config/gaira_dataset_experiment_registry_v2.csv`: add biosample experiment rows for datasets intended to participate in interpretation or validation lanes.",
            "- `config/gaira_grounding_family_registry_v1.csv`: no new family is required in this pass; support datasets can reuse existing grounding-family concepts when their parsers land.",
            "",
            "## Expected Verification Outputs",
            "",
            "- input inventory table",
            "- exclusion summary",
            "- final usable spectra count",
            "- final usable sample count",
            "- label summary",
            "- provenance note",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    write_csv(TMP_OUT / "tables" / "global_v2_ingest_queue.csv", QUEUE_ROWS)
    write_csv(TMP_OUT / "tables" / "global_v2_parser_requirements.csv", PARSER_ROWS)
    write_csv(TMP_OUT / "tables" / "global_v2_registry_update_plan.csv", REGISTRY_PLAN_ROWS)
    report_md = TMP_OUT / "report" / "global_v2_ingest_queue.md"
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(build_report(), encoding="utf-8")
    build_pdf_report(report_md, [], report_md.with_suffix(".pdf"))
    print(f"Wrote queue outputs under {TMP_OUT}")


if __name__ == "__main__":
    main()
