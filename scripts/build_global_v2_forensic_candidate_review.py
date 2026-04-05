from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INV = ROOT / "tmp/global_v2_dataset_audit/tables/global_v2_candidate_dataset_inventory.csv"
DEDUP = ROOT / "tmp/global_v2_dataset_audit/tables/global_v2_deduplication_check.csv"
BIO = ROOT / "tmp/global_v2_dataset_audit_biobias/tables/global_v2_biobias_candidate_inventory.csv"
BIO_PRI = ROOT / "tmp/global_v2_dataset_audit_biobias/tables/global_v2_biobias_priority_list.csv"
OUT = ROOT / "tmp/global_v2_forensic_candidate_review"


GROUPS = [
    ("FC01", "AI autoencoder paper only", ["https://zenodo.org/records/16895315"], "single"),
    ("FC02", "Metabolite fingerprint support PDF", ["https://zenodo.org/records/14294417"], "single"),
    ("FC03", "SMILES-vs-SERS clustering paper", ["https://zenodo.org/records/17052624"], "single"),
    ("FC04", "Pseudospectra autoencoder paper", ["https://zenodo.org/records/16912956"], "single"),
    ("FC05", "Stem-cell differentiation SERS trajectory", ["https://zenodo.org/records/10851312"], "dedup_exact"),
    ("FC06", "Sherman-metabolite fingerprint PDF", ["https://zenodo.org/records/10055068"], "single"),
    ("FC07", "ACS Omega paper with claimed online SI/zip", ["https://pubs.acs.org/doi/10.1021/acsomega.4c11078"], "single"),
    ("FC08", "Ischemic-stroke urine family", ["https://zenodo.org/records/19109120", "https://zenodo.org/records/19369604"], "version_family"),
    ("FC09", "Adenine lateral-flow controlled assay", ["https://zenodo.org/records/17035751"], "single"),
    ("FC10", "Porous carbon nanowire substrate archive", ["https://zenodo.org/records/3994312"], "single"),
    ("FC11", "UTI pathogen OPJ archive", ["https://zenodo.org/records/5021659"], "single"),
    ("FC12", "Beta-lactam resistance plasmid Raman panel", ["https://zenodo.org/records/12740805"], "single"),
    ("FC13", "Bolaform drug/concentration archive", ["https://zenodo.org/records/5806132"], "single"),
    ("FC14", "Mycoplasma NA-SERS family", ["https://zenodo.org/records/4941488", "https://figshare.com/articles/dataset/_PLS_DA_of_NA_SERS_specificity_and_sensitivity_in_discriminating_M_pneumoniae_strains_/493381?file=823019", "https://figshare.com/articles/dataset/_Cross_validated_PLS_DA_modeling_statistics_for_the_prediction_performance_for_NA_SERS_typing_of_individual_type_1_and_2_M_pneumoniae_clinical_isolates_/1467505?file=2154588"], "paper_plus_stats_family"),
    ("FC15", "Bacterial metabolism nanorattle workbook", ["https://zenodo.org/records/5806264"], "single"),
    ("FC16", "Tumor purine secretome archive", ["https://zenodo.org/records/8130216"], "single"),
    ("FC17", "Coeliac faecal SERS cohort", ["https://zenodo.org/records/5947010"], "single"),
    ("FC18", "Tear dopamine assay archive", ["https://zenodo.org/records/18670010"], "single"),
    ("FC19", "Graphene metasurface antibody archive", ["https://zenodo.org/records/17023716"], "single"),
    ("FC20", "Sertraline/serotonin nanocone archive", ["https://zenodo.org/records/18284194"], "dedup_exact"),
    ("FC21", "Head-and-neck cancer / infection support zip", ["https://zenodo.org/records/14755439"], "single"),
    ("FC22", "Ibuprofen / nicotinamide dimer zip", ["https://zenodo.org/records/7523579"], "single"),
    ("FC23", "Ovarian plasma Raman cohort", ["https://figshare.com/articles/dataset/Raman_spectroscopic_techniques_to_detect_ovarian_cancer_biomarkers_in_blood_plasma/6744206"], "single"),
    ("FC24", "ACS workbook family", ["https://figshare.com/articles/dataset/SERS_spectra_of_43_patients_with_ACS_xlsx/24747531?file=43481136", "https://figshare.com/articles/dataset/Suplementary_material_DIB_ACS_40_samples_xlsx/24564787?file=43183257"], "related_workbook_family"),
    ("FC25", "RBC membrane + EV SERS/TERS archive", ["https://figshare.com/articles/dataset/Surface-enhanced_SERS_and_tip-enhanced_TERS_Raman_scattering_in_label-free_characterization_of_erythrocyte_membranes_and_extracellular_vesicles_in_nano-scale_and_at_the_single-molecule_level_/24105993"], "single"),
    ("FC26", "Magnetically enhanced bacterial/SARS-CoV-2 PDF", ["https://figshare.com/articles/dataset/DataSheet1_Magnetically_Enhanced_Liquid_SERS_for_Ultrasensitive_Analysis_of_Bacterial_and_SARS-CoV-2_Biomarkers_PDF/16697359?file=30917875"], "single"),
    ("FC27", "Telomerase HCC docx support", ["https://figshare.com/articles/dataset/DataSheet1_A_dual-amplification_strategy-intergated_SERS_biosensor_for_ultrasensitive_hepatocellular_carcinoma-related_telomerase_activity_detection_docx/21896907?file=38839281"], "single"),
    ("FC28", "PCL scaffold support family", ["https://figshare.com/articles/dataset/Data_Sheet_1_Polycaprolactone-Based_Porous_CaCO3_and_Ag_Nanoparticle_Modified_Scaffolds_as_a_SERS_Platform_With_Molecule-Specific_Adsorption_pdf/11568420?file=20804994", "https://figshare.com/articles/dataset/Table_1_Polycaprolactone-Based_Porous_CaCO3_and_Ag_Nanoparticle_Modified_Scaffolds_as_a_SERS_Platform_With_Molecule-Specific_Adsorption_pdf/11568423?file=20804997"], "paper_plus_table_family"),
    ("FC29", "Quantitative bioanalytical hotspot-stabilization support", ["https://figshare.com/articles/dataset/HOTSPOT_STABILIZATION_OF_GOLD_NANOPARTICLES_FOR_APPLICATION_OF_QUANTITATIVE_SERS_IN_BIOANALYTICAL_SYSTEMS/11390760"], "single"),
    ("FC30", "Down-syndrome leukemogenesis PDF", ["https://figshare.com/articles/dataset/Data_Sheet_1_SERS-Based_Evaluation_of_the_DNA_Methylation_Pattern_Associated_With_Progression_in_Clonal_Leukemogenesis_of_Down_Syndrome_PDF/15042300?file=28924107"], "single"),
    ("FC31", "General applications review PDF", ["https://figshare.com/articles/dataset/Data_Sheet_1_Applications_of_Surface-Enhanced_Raman_Scattering_in_Biochemical_and_Medical_Analysis_PDF/14552937?file=27920859"], "single"),
    ("FC32", "UCLA saliva sEV gastric-cancer family", [
        "https://figshare.com/articles/dataset/Health_control_01_sEV_Saliva_UCLA_ERCC_/20428395",
        "https://figshare.com/articles/dataset/GC_Patient_20_sEV_Saliva_UCLA_ERCC_/20427957",
        "https://figshare.com/articles/dataset/GC_Patient_19_sEV_Saliva_UCLA_ERCC_/20427954",
        "https://figshare.com/articles/dataset/GC_Patient_18_sEV_Saliva_UCLA_ERCC_/20427951",
        "https://figshare.com/articles/dataset/GC_Patient_17_sEV_Saliva_UCLA_ERCC_/20427948",
        "https://figshare.com/articles/dataset/GC_Patient_16_sEV_Saliva_UCLA_ERCC_/20427945",
        "https://figshare.com/articles/dataset/GC_Patient_15_sEV_Saliva_UCLA_ERCC_/20427939",
        "https://figshare.com/articles/dataset/GC_Patient_14_sEV_Saliva_UCLA_ERCC_/20427936",
        "https://figshare.com/articles/dataset/GC_Patient_13_sEV_Saliva_UCLA_ERCC_/20427933",
        "https://figshare.com/articles/dataset/GC_Patient_12_sEV_Saliva_UCLA_ERCC_/20427930",
        "https://figshare.com/articles/dataset/GC_Patient_11_sEV_Saliva_UCLA_ERCC_/20427927",
        "https://figshare.com/articles/dataset/GC_Patient_10_sEV_Saliva_UCLA_ERCC_/20427924",
        "https://figshare.com/articles/dataset/GC_Patient_9_sEV_Saliva_UCLA_ERCC_/20427921",
        "https://figshare.com/articles/dataset/GC_Patient_8_sEV_Saliva_UCLA_ERCC_/20427918",
        "https://figshare.com/articles/dataset/GC_Patient_7_sEV_Saliva_UCLA_ERCC_/20427909",
        "https://figshare.com/articles/dataset/GC_Patient_6_sEV_Saliva_UCLA_ERCC_/20427906",
        "https://figshare.com/articles/dataset/GC_Patient_5_sEV_Saliva_UCLA_ERCC_/20427903",
        "https://figshare.com/articles/dataset/GC_Patient_1_UG3/20282238",
        "https://figshare.com/articles/dataset/ERCC/20406102",
    ], "fragmented_cohort_family"),
    ("FC33", "E. coli lateral-flow docx", ["https://figshare.com/articles/dataset/Data_Sheet_1_Rapid_Quantitative_High-Sensitive_Detection_of_Escherichia_coli_O157_H7_by_Gold-Shell_Silica-Core_Nanospheres-Based_Surface-Enhanced_Raman_Scattering_Lateral_Flow_Immunoassay_docx/13199414?file=25409663"], "single"),
    ("FC34", "Bacterial endotoxin PDF", ["https://figshare.com/articles/dataset/DataSheet_1_SERS_Sensing_of_Bacterial_Endotoxin_on_Gold_Nanoparticles_pdf/16764130?file=31020013"], "single"),
    ("FC35", "Single-vesicle EV raw Raman archive", ["https://figshare.com/articles/dataset/Raw_Raman_data_/26059145?file=47123702"], "single"),
    ("FC36", "Nature Communications source workbook", ["https://figshare.com/articles/dataset/Source_Data_file_xlsxDataset_ArticleNatureComm_Dallarietal_2024/26411992?file=48039661"], "single"),
    ("FC37", "Bladder-cancer additional file 2", ["https://figshare.com/articles/dataset/Additional_file_2_of_Combined_miRNA_and_SERS_urine_liquid_biopsy_for_the_point-of-care_diagnosis_and_molecular_stratification_of_bladder_cancer/19498603?file=34649167"], "single"),
    ("FC38", "Cytochrome-c reference spectra", ["https://figshare.com/articles/dataset/SERS_and_Raman_spectra_of_WT_and_mutant_cytochromes_c/4903091"], "single"),
    ("FC39", "Multiplexed coded nanoparticle method record", ["https://figshare.com/articles/dataset/_Method_for_Assessing_the_Reliability_of_Molecular_Diagnostics_Based_on_Multiplexed_SERS_Coded_Nanoparticles_/686466"], "single"),
    ("FC40", "Fungal-pathogen peak-assignment family", ["https://figshare.com/articles/dataset/Raman_shift_and_putative_peak_assignments_of_the_SERS_spectra_from_conidia_of_the_common_causative_pathogens_used_in_this_study_/11989566?file=22020294", "https://figshare.com/articles/dataset/Raman_shift_and_putative_peak_assignments_of_the_SERS_spectra_from_mycelia_of_the_common_causative_pathogens_used_in_this_study_/11989560?file=22020288"], "paired_support_family"),
    ("FC41", "Diagnostic-performance xls table", ["https://figshare.com/articles/dataset/Diagnostic_performance_of_PLS-DA_models_in_predicting_high_and_low_groups_of_conventional_quantitative_parameters_SER_ADC_SUV_/3882102?file=6084882"], "single"),
    ("FC42", "VOC substrate docx", ["https://figshare.com/articles/dataset/DataSheet1_AuNPs_MIL-101_Cr_as_a_SERS-Active_Substrate_for_Sensitive_Detection_of_VOCs_docx/20100728?file=35950469"], "single"),
    ("FC43", "Influenza aptasensor image-only record", ["https://figshare.com/articles/dataset/Highly_sensitive_detection_of_influenza_virus_with_SERS_aptasensor/8044163"], "single"),
    ("FC44", "Fumonisin DFT/theory zip", ["https://figshare.com/articles/dataset/DFT-Based_Theoretical_Study_on_Label-Free_SERS_Detection_of_Type_B_Fumonisins_New_Insights_into_Molecular_Substrate_Interactions_and_Quantification_Strategies/28565671?file=52894126"], "single"),
]


GROUP_NOTES = {
    "FC05": ("The manuscript frames this as six stem-cell differentiation states measured by SERS with CNN classification. The archive is a single large RAR, so the real spectra are likely present but need manual unpacking; likely exclude model/code clutter if mixed in the RAR.", "ingest_after_reconstruction", "medium"),
    "FC08": ("This is the same ischemic-stroke urine study in two Zenodo versions. The newer 19369604 supersedes 19109120 and should be the only version kept; likely exclude the older version once the newer archive is unpacked.", "ingest_after_reconstruction", "high"),
    "FC09": ("Controlled adenine / IgG assay data are real spectra, but this is a calibration/support asset rather than a core Global v2 biological diversity dataset. Most useful files are the numeric spectra; exclude narrative PDFs and duplicate processed copies.", "support_only", "high"),
    "FC10": ("The paper is about a porous carbon substrate. Even if some biomolecule traces appear in figure zips, the release is fundamentally a substrate/materials package; exclude figure-only material and keep this out of core ingest.", "method_only", "high"),
    "FC11": ("The study measures real pathogen spectra in culture matrix/artificial urine. The main caveat is packaging in OPJ/OPJU, so ingestion means converting Origin projects and excluding non-spectral plotting artifacts.", "ingest_after_reconstruction", "high"),
    "FC12": ("This is a real biological/pathogen-like Raman panel around beta-lactam resistance fragments. The usable target is the Raman zip/TXT archive; exclude AFM/XPS/UV-Vis if the goal is Global v2 spectral training.", "ingest_now", "high"),
    "FC13": ("The paper focuses on solution-based drug detection and quantification. Numeric spectra exist, but the biology is weak relative to Global v2 goals; keep only as reserve support and exclude microscopy images and materials characterization.", "support_only", "medium"),
    "FC14": ("Group contains the main Mycoplasma CSV dataset plus figshare model-statistics tables from the same study family. The ingest target is the Zenodo CSV; the figshare xls tables are context/method support and should be excluded from core ingest.", "ingest_now", "high"),
    "FC15": ("This is a biologically interesting microbial metabolism workbook. The likely usable asset is the main xlsx spectral source data; exclude manuscript-only context and any non-spectral figure artifacts.", "ingest_after_reconstruction", "medium"),
    "FC16": ("The PNAS archive appears to be a real cell-secretome/purine-state dataset. It is worth keeping alive, but it likely contains mixed support material in the zip; prioritize spectral tables and exclude non-spectral supplemental clutter.", "ingest_after_reconstruction", "medium"),
    "FC17": ("Raw faecal TXT spectra are visible in the archive and the study is a real human disease/diet cohort. Exclude only documentation/code duplicates and keep the raw text spectra as the core ingest target.", "ingest_now", "high"),
    "FC18": ("This tear/dopamine study has numeric data, but it is a targeted assay paper rather than a broad biological dataset. Keep only spectral tables if ever used; exclude microscopy and figure images.", "support_only", "medium"),
    "FC20": ("Large-scale sertraline/serotonin assay data are present but the study is targeted quantification rather than biological diversity learning. Exclude image/microscopy outputs; keep only if a future support lane needs it.", "support_only", "medium"),
    "FC21": ("The Zenodo support zip belongs to a biologically relevant head-and-neck/infection study, but the visible files look like figure/panel-level outputs. Likely useful for context or reserve support, not core ingest.", "support_only", "medium"),
    "FC23": ("This is one of the strongest candidates: a real plasma disease Raman archive with hundreds of TXT spectra visible in the zip listing. Likely exclude manuscript/support files and keep the raw plasma spectra.", "ingest_now", "high"),
    "FC24": ("Both ACS workbooks look like related human cardiovascular SERS cohorts. The usable target is the workbook spectral sheets; exclude any duplicate summary sheets or derived statistics after inspection.", "ingest_after_reconstruction", "medium"),
    "FC25": ("This archive contains real TXT spectra for erythrocyte membranes and EVs, plus supporting workbook assets. Keep the raw TXT spectra; exclude explanatory spreadsheets or figures that just duplicate the same signal.", "ingest_now", "high"),
    "FC29": ("Quantitative bioanalytical system support with xls/jpg assets. Useful for calibration/support context, but not a core biological dataset.", "support_only", "medium"),
    "FC32": ("These records are a fragmented saliva sEV cohort split into per-patient Figshare shards plus one metadata stub. Review them as one reconstructable family: keep the patient/control TXT spectra, exclude the ERCC metadata-only record, and plan a cohort reconstruction rather than separate ingests.", "ingest_after_reconstruction", "high"),
    "FC35": ("The figshare RAR is a real EV/single-vesicle raw Raman release. It looks biologically valuable but needs unpacking and triage to isolate the actual spectral matrix from any supplementary clutter.", "ingest_after_reconstruction", "medium"),
    "FC37": ("Despite the title, the uploaded xlsx appears to contain miRNA target/enrichment sheets rather than reusable spectra. Treat the paper as context only and reject the asset for ingest.", "reject", "high"),
    "FC38": ("Clean reference Raman/SERS spectra for WT and mutant cytochromes c. Good support/grounding dataset; keep the numeric txt/csv spectra and ignore nonessential text metadata.", "support_only", "high"),
    "FC43": ("The record is influenza-themed, but only image files are visible, not spectra. There is no practical ingest target here.", "reject", "high"),
    "FC44": ("Large zip with theory/fumonisin context. There may be spectra, but the study is primarily theoretical and support-oriented rather than core real-bio pretraining.", "support_only", "medium"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_assets(members: list[dict[str, str]]) -> str:
    assets = sorted({a.strip() for m in members for a in m["file_types_available"].split(";") if a.strip()})
    return ", ".join(assets) if assets else "no files visible"


def raw_status(members: list[dict[str, str]]) -> str:
    vals = {m["raw_spectra_present"] for m in members}
    if vals == {"yes"}:
        return "yes"
    if "yes" in vals:
        return "mixed"
    if vals == {"no"}:
        return "no"
    return "unclear"


def sample_type(members: list[dict[str, str]]) -> str:
    counts = Counter(m["biosample_type"] for m in members)
    return counts.most_common(1)[0][0]


def labels_present(members: list[dict[str, str]]) -> str:
    labels = sorted({l.strip() for m in members for l in m["labels_available"].split(";") if l.strip() and l.strip() != "unlabeled"})
    return "; ".join(labels) if labels else "unlabeled"


def likely_exclusions(group_id: str, members: list[dict[str, str]]) -> str:
    assets = summarize_assets(members)
    exclusions = []
    if "pdf" in assets or "docx" in assets:
        exclusions.append("narrative PDF/DOCX support files with no spectra")
    if "jpg" in assets or "png" in assets or "tif" in assets or "tiff" in assets:
        exclusions.append("figure images / microscopy-only files")
    if any(m["source_url"].endswith("19109120") for m in members):
        exclusions.append("older superseded version")
    if group_id == "FC32":
        exclusions.append("ERCC metadata-only stub and shard-level duplicates after cohort merge")
    if group_id == "FC14":
        exclusions.append("separate figshare model-statistics tables")
    if group_id == "FC24":
        exclusions.append("duplicate summary sheets if both ACS workbooks overlap")
    return "; ".join(exclusions) if exclusions else "exclude only obvious non-spectral support duplicates"


def overlap(members: list[dict[str, str]]) -> str:
    vals = sorted({m["overlap_with_existing_gaira"] for m in members})
    if "strong" in vals:
        return "strong"
    if "partial" in vals or any("fragmented" in v for v in vals):
        return "partial"
    return "none"


def usable_estimate(group_id: str, members: list[dict[str, str]]) -> str:
    if group_id == "FC23":
        return "hundreds of raw plasma TXT spectra"
    if group_id == "FC24":
        return "roughly 80+ workbook spectra if both ACS files are real spectral matrices"
    if group_id == "FC32":
        total = 0
        for m in members:
            text = m["spectra_count_estimate"]
            num = "".join(ch for ch in text if ch.isdigit())
            if num:
                total += int(num)
        return f"at least {total} txt spectra across 18 shard records" if total else "many shard-level txt spectra across the cohort"
    if group_id == "FC08":
        return "large urine cohort archive; exact count needs extraction"
    if group_id == "FC05":
        return "large RAR of stem-cell state spectra; exact count needs extraction"
    if group_id == "FC35":
        return "RAR archive of single-vesicle EV spectra"
    if group_id == "FC14":
        return "one main CSV spectral matrix plus two support xls tables"
    if len(members) == 1:
        return members[0]["spectra_count_estimate"]
    return "grouped mixed assets"


def group_context(group_id: str, title: str, members: list[dict[str, str]]) -> tuple[str, str, str]:
    if group_id in GROUP_NOTES:
        return GROUP_NOTES[group_id]
    raw = raw_status(members)
    assets = summarize_assets(members)
    if raw == "yes":
        note = f"Usable spectra appear to be present. The main targets are the numeric {assets} assets; exclude non-spectral support files."
        return note, "ingest_after_reconstruction" if any(ext in assets for ext in ["rar", "zip", "xlsx", "opj", "opju"]) else "ingest_now", "medium"
    if raw == "mixed":
        note = f"The family mixes real spectra with support-only material. Keep the spectral assets and exclude narrative/statistics-only files."
        return note, "support_only", "medium"
    if "pdf" in assets or "docx" in assets:
        note = "Only narrative/support assets are visible rather than a practical spectra release."
        return note, "method_only", "high"
    note = "No practical spectral ingest target is visible from the downloadable assets."
    return note, "reject", "high"


def main() -> None:
    inv_rows = {r["source_url"]: r for r in read_csv(INV)}
    dedup_rows = {r["source_url"]: r for r in read_csv(DEDUP)}
    bio_rows = {r["source_url"]: r for r in read_csv(BIO)}
    pri_rows = {r["source_url"]: r for r in read_csv(BIO_PRI)}

    master_rows = []
    group_log = []
    report_lines = ["# Global v2 Forensic Candidate Review", ""]
    report_lines.append("## Grouped Candidate List")
    report_lines.append("")
    for gid, label, urls, mode in GROUPS:
        report_lines.append(f"- {gid}: {label} | {len(urls)} URL(s) | {mode}")
    report_lines.append("")

    for gid, label, urls, mode in GROUPS:
        members = []
        for url in urls:
            inv = inv_rows[url]
            bio = bio_rows.get(url, {})
            pri = pri_rows.get(url, {})
            ded = dedup_rows.get(url, {})
            members.append(
                {
                    "source_url": url,
                    "title": inv["title"],
                    "source_type": inv["source_type"],
                    "file_types_available": inv["file_types_available"],
                    "raw_spectra_present": inv["raw_spectra_present"],
                    "processed_only": inv["processed_only"],
                    "archive_contents_note": inv["archive_contents_note"],
                    "metadata_quality_note": inv["metadata_quality_note"],
                    "biosample_type": bio.get("biosample_type", inv["domain_category"]),
                    "labels_available": bio.get("labels_available", inv["label_type"]),
                    "spectra_count_estimate": bio.get("spectra_count_estimate", inv["approx_dataset_size"]),
                    "cohort_size_estimate": bio.get("cohort_size_estimate", "unknown"),
                    "organism_type": bio.get("organism_type", inv["entity_type"]),
                    "disease_state_task_type": bio.get("disease_state_task_type", inv["label_type"]),
                    "overlap_with_existing_gaira": bio.get("overlap_with_existing_gaira", ded.get("duplicate_status", "none")),
                    "final_category": pri.get("final_category", ""),
                }
            )
            group_log.append(
                {
                    "candidate_id": gid,
                    "candidate_label": label,
                    "member_url": url,
                    "member_title": inv["title"],
                    "grouping_reason": mode,
                    "member_role": "primary" if url == urls[0] else "grouped_member",
                }
            )

        note, decision, confidence = group_context(gid, label, members)
        assets = summarize_assets(members)
        spectra_avail = raw_status(members)
        usable = usable_estimate(gid, members)
        lbls = labels_present(members)
        overlap_status = overlap(members)
        reconstruction = "yes" if decision == "ingest_after_reconstruction" else "no"
        exclusions = likely_exclusions(gid, members)

        master_rows.append(
            {
                "candidate_id": gid,
                "title": label,
                "urls": " | ".join(urls),
                "grouped_members": len(urls),
                "spectra_available": spectra_avail,
                "sample_type": sample_type(members),
                "labels_present": lbls,
                "usable_data_estimate": usable,
                "reconstruction_required": reconstruction,
                "likely_exclusions": exclusions,
                "overlap_with_existing_gaira": overlap_status,
                "final_recommendation": decision,
                "confidence": confidence,
            }
        )

        report_lines.extend(
            [
                f"## {gid}. {label}",
                "",
                "### A. Candidate ID",
                f"- short internal ID: {gid}",
                f"- source URL(s): {' ; '.join(urls)}",
                f"- grouped or single: {'grouped' if len(urls) > 1 else 'single'}",
                f"- title if available: {label}",
                "",
                "### B. Asset availability",
                f"- actual spectra dataset available: {spectra_avail}",
                f"- downloadable assets: {assets}",
                f"- raw / processed status: {spectra_avail if spectra_avail != 'mixed' else 'mixed real spectra plus support files'}",
                f"- estimated usable spectra count: {usable}",
                f"- estimated sample / cohort count: {', '.join(sorted({m['cohort_size_estimate'] for m in members if m['cohort_size_estimate']}))}",
                "",
                "### C. Sample / biology type",
                f"- primary sample type: {sample_type(members)}",
                f"- organism / source type: {', '.join(sorted({m['organism_type'] for m in members if m['organism_type']}))}",
                f"- disease / state / task type: {', '.join(sorted({m['disease_state_task_type'] for m in members if m['disease_state_task_type']}))}",
                f"- labels visible: {lbls}",
                "",
                "### D. Paper / manuscript / SI context",
                f"- {note}",
                "",
                "### E. Likely exclusions",
                f"- {exclusions}",
                "",
                "### F. Practical ingest value",
                f"- ingestable now: {'yes' if decision == 'ingest_now' else 'no'}",
                f"- reconstruction required: {reconstruction}",
                f"- metadata sufficient: {'yes' if confidence == 'high' and spectra_avail == 'yes' else 'partial'}",
                f"- provenance acceptable: {'yes' if spectra_avail in {'yes', 'mixed'} else 'partial'}",
                f"- overlap with existing GAIRA: {overlap_status}",
                f"- still valuable despite overlap: {'yes' if overlap_status != 'strong' or decision in {'support_only', 'ingest_after_reconstruction'} else 'limited'}",
                "",
                "### G. Recommendation",
                f"- concise summary: {note}",
                f"- ingest decision label: {decision}",
                f"- confidence: {confidence}",
                "",
            ]
        )

    write_csv(OUT / "tables/candidate_review_master.csv", master_rows)
    write_csv(OUT / "tables/grouping_and_deduplication_log.csv", group_log)
    report_path = OUT / "report/global_v2_forensic_candidate_review.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines))


if __name__ == "__main__":
    main()
