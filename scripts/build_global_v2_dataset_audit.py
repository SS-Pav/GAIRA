from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import Request


ROOT = Path(__file__).resolve().parents[1]
DATASETS_CSV = ROOT / "data/registry/datasets.csv"
DISCOVERY_CSV = ROOT / "data/discovery/processed/discovery_registry.csv"
OUT_ROOT = ROOT / "tmp/global_v2_dataset_audit"


CANDIDATE_URLS = [
    "https://zenodo.org/records/16895315",
    "https://zenodo.org/records/14294417",
    "https://zenodo.org/records/17052624",
    "https://zenodo.org/records/16912956",
    "https://zenodo.org/records/10851312",
    "https://zenodo.org/records/10055068",
    "https://pubs.acs.org/doi/10.1021/acsomega.4c11078",
    "https://zenodo.org/records/19109120",
    "https://zenodo.org/records/19369604",
    "https://zenodo.org/records/17035751",
    "https://zenodo.org/records/3994312",
    "https://zenodo.org/records/5021659",
    "https://zenodo.org/records/12740805",
    "https://zenodo.org/records/5806132",
    "https://zenodo.org/records/4941488",
    "https://zenodo.org/records/5806264",
    "https://zenodo.org/records/8130216",
    "https://zenodo.org/records/5947010",
    "https://zenodo.org/records/18670010",
    "https://zenodo.org/records/17023716",
    "https://zenodo.org/records/18284194",
    "https://zenodo.org/records/14755439",
    "https://zenodo.org/records/7523579",
    "https://figshare.com/articles/dataset/Raman_spectroscopic_techniques_to_detect_ovarian_cancer_biomarkers_in_blood_plasma/6744206",
    "https://figshare.com/articles/dataset/SERS_spectra_of_43_patients_with_ACS_xlsx/24747531?file=43481136",
    "https://figshare.com/articles/dataset/Suplementary_material_DIB_ACS_40_samples_xlsx/24564787?file=43183257",
    "https://figshare.com/articles/dataset/Surface-enhanced_SERS_and_tip-enhanced_TERS_Raman_scattering_in_label-free_characterization_of_erythrocyte_membranes_and_extracellular_vesicles_in_nano-scale_and_at_the_single-molecule_level_/24105993",
    "https://figshare.com/articles/dataset/DataSheet1_Magnetically_Enhanced_Liquid_SERS_for_Ultrasensitive_Analysis_of_Bacterial_and_SARS-CoV-2_Biomarkers_PDF/16697359?file=30917875",
    "https://figshare.com/articles/dataset/DataSheet1_A_dual-amplification_strategy-intergated_SERS_biosensor_for_ultrasensitive_hepatocellular_carcinoma-related_telomerase_activity_detection_docx/21896907?file=38839281",
    "https://figshare.com/articles/dataset/Data_Sheet_1_Polycaprolactone-Based_Porous_CaCO3_and_Ag_Nanoparticle_Modified_Scaffolds_as_a_SERS_Platform_With_Molecule-Specific_Adsorption_pdf/11568420?file=20804994",
    "https://figshare.com/articles/dataset/Table_1_Polycaprolactone-Based_Porous_CaCO3_and_Ag_Nanoparticle_Modified_Scaffolds_as_a_SERS_Platform_With_Molecule-Specific_Adsorption_pdf/11568423?file=20804997",
    "https://figshare.com/articles/dataset/HOTSPOT_STABILIZATION_OF_GOLD_NANOPARTICLES_FOR_APPLICATION_OF_QUANTITATIVE_SERS_IN_BIOANALYTICAL_SYSTEMS/11390760",
    "https://figshare.com/articles/dataset/_PLS_DA_of_NA_SERS_specificity_and_sensitivity_in_discriminating_M_pneumoniae_strains_/493381?file=823019",
    "https://figshare.com/articles/dataset/Data_Sheet_1_SERS-Based_Evaluation_of_the_DNA_Methylation_Pattern_Associated_With_Progression_in_Clonal_Leukemogenesis_of_Down_Syndrome_PDF/15042300?file=28924107",
    "https://figshare.com/articles/dataset/Data_Sheet_1_Applications_of_Surface-Enhanced_Raman_Scattering_in_Biochemical_and_Medical_Analysis_PDF/14552937?file=27920859",
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
    "https://figshare.com/articles/dataset/Data_Sheet_1_Rapid_Quantitative_High-Sensitive_Detection_of_Escherichia_coli_O157_H7_by_Gold-Shell_Silica-Core_Nanospheres-Based_Surface-Enhanced_Raman_Scattering_Lateral_Flow_Immunoassay_docx/13199414?file=25409663",
    "https://figshare.com/articles/dataset/DataSheet_1_SERS_Sensing_of_Bacterial_Endotoxin_on_Gold_Nanoparticles_pdf/16764130?file=31020013",
    "https://figshare.com/articles/dataset/ERCC/20406102",
    "https://figshare.com/articles/dataset/Raw_Raman_data_/26059145?file=47123702",
    "https://figshare.com/articles/dataset/Source_Data_file_xlsxDataset_ArticleNatureComm_Dallarietal_2024/26411992?file=48039661",
    "https://figshare.com/articles/dataset/GC_Patient_1_UG3/20282238",
    "https://figshare.com/articles/dataset/Additional_file_2_of_Combined_miRNA_and_SERS_urine_liquid_biopsy_for_the_point-of-care_diagnosis_and_molecular_stratification_of_bladder_cancer/19498603?file=34649167",
    "https://figshare.com/articles/dataset/SERS_and_Raman_spectra_of_WT_and_mutant_cytochromes_c/4903091",
    "https://figshare.com/articles/dataset/_Method_for_Assessing_the_Reliability_of_Molecular_Diagnostics_Based_on_Multiplexed_SERS_Coded_Nanoparticles_/686466",
    "https://figshare.com/articles/dataset/Raman_shift_and_putative_peak_assignments_of_the_SERS_spectra_from_conidia_of_the_common_causative_pathogens_used_in_this_study_/11989566?file=22020294",
    "https://figshare.com/articles/dataset/Raman_shift_and_putative_peak_assignments_of_the_SERS_spectra_from_mycelia_of_the_common_causative_pathogens_used_in_this_study_/11989560?file=22020288",
    "https://figshare.com/articles/dataset/Diagnostic_performance_of_PLS-DA_models_in_predicting_high_and_low_groups_of_conventional_quantitative_parameters_SER_ADC_SUV_/3882102?file=6084882",
    "https://figshare.com/articles/dataset/DataSheet1_AuNPs_MIL-101_Cr_as_a_SERS-Active_Substrate_for_Sensitive_Detection_of_VOCs_docx/20100728?file=35950469",
    "https://figshare.com/articles/dataset/Highly_sensitive_detection_of_influenza_virus_with_SERS_aptasensor/8044163",
    "https://figshare.com/articles/dataset/DFT-Based_Theoretical_Study_on_Label-Free_SERS_Detection_of_Type_B_Fumonisins_New_Insights_into_Molecular_Substrate_Interactions_and_Quantification_Strategies/28565671?file=52894126",
    "https://figshare.com/articles/dataset/_Cross_validated_PLS_DA_modeling_statistics_for_the_prediction_performance_for_NA_SERS_typing_of_individual_type_1_and_2_M_pneumoniae_clinical_isolates_/1467505?file=2154588",
]


TEXT_HEADERS = {
    "User-Agent": "GAIRA Global v2 dataset audit/1.0",
    "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
}


@dataclass
class Candidate:
    source_url: str
    source_type: str = ""
    title: str = ""
    raw_spectra_present: str = "unknown"
    processed_only: str = "unknown"
    file_types_available: str = ""
    approx_dataset_size: str = ""
    analyte_or_biosample_type: str = ""
    domain_category: str = "other"
    label_type: str = "unlabeled"
    entity_type: str = "unknown"
    likely_wavenumber_range: str = ""
    reusable_for_ml: str = "unknown"
    provenance_clean: str = "unknown"
    duplicate_status: str = "no_known_overlap"
    duplicate_notes: str = ""
    already_ingested: str = "no"
    overlap_existing_dataset: str = ""
    metadata_quality_note: str = ""
    archive_contents_note: str = ""
    source_record_id: str = ""
    description_snippet: str = ""
    files_count: int = 0
    raw_score: int = 0
    metadata_score: int = 0
    biological_relevance_score: int = 0
    diversity_score: int = 0
    shared_encoder_utility_score: int = 0
    grounding_utility_score: int = 0
    invariance_score: int = 0
    ingest_difficulty_score: int = 0
    overall_value_score: float = 0.0
    recommended_action: str = ""
    rationale: str = ""
    file_names: list[str] = field(default_factory=list)
    file_sizes: list[int] = field(default_factory=list)


def http_get_json(url: str) -> Any:
    proc = subprocess.run(
        ["curl", "-L", "-sS", "-H", f"User-Agent: {TEXT_HEADERS['User-Agent']}", url],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"curl failed for {url}")
    return json.loads(proc.stdout)


def http_get_text(url: str) -> str:
    proc = subprocess.run(
        ["curl", "-L", "-sS", "-H", f"User-Agent: {TEXT_HEADERS['User-Agent']}", url],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"curl failed for {url}")
    return proc.stdout


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def size_to_str(num: int | None) -> str:
    if not num:
        return ""
    units = ["B", "KB", "MB", "GB"]
    size = float(num)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{num} B"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def summarize_sizes(sizes: list[int]) -> str:
    if not sizes:
        return ""
    total = sum(sizes)
    return f"{len(sizes)} files; {size_to_str(total)} total"


def infer_domain(title: str, desc: str, files: list[str]) -> str:
    hay = " ".join([title, desc, *files]).lower()
    if "extracellular vesicle" in hay or " exosome" in hay or "sev" in hay:
        if "saliva" in hay:
            return "saliva"
        return "EV"
    if "serum" in hay or "blood plasma" in hay or "plasma" in hay:
        return "serum"
    if "saliva" in hay:
        return "saliva"
    if "urine" in hay:
        return "urine"
    if any(k in hay for k in ["bacteria", "bacterial", "virus", "pathogen", "influenza", "sars-cov-2", "mycoplasma", "e. coli", "endotoxin", "plasmid", "fumonisin"]):
        return "pathogen"
    if any(k in hay for k in ["drug", "cocaine", "fentanyl", "alprazolam", "buprenorphine", "thc", "triclosan"]):
        return "drug"
    if any(k in hay for k in ["glucose", "lactoglobulin", "metabolite", "cytochrome", "adenine", "amino acid", "mirna", "telomerase"]):
        return "molecule_reference"
    if any(k in hay for k in ["substrate", "nanoparticle", "scaffold", "materials", "mos2", "mil-101", "gold-shell", "nanocoral"]):
        return "substrate / materials"
    if any(k in hay for k in ["interlab", "reproduc", "protocol", "batch", "lab"]):
        return "interlab / reproducibility"
    if any(k in hay for k in ["cell", "erythrocyte", "membrane"]):
        return "cells"
    return "other"


def infer_label_type(title: str, desc: str, files: list[str]) -> str:
    hay = " ".join([title, desc, *files]).lower()
    labels = []
    if any(k in hay for k in ["concentration", "ppm", "calibration", "quantitative", "spike", "dose", "dose-response"]):
        labels.append("concentration")
    if any(k in hay for k in ["disease", "cancer", "patient", "healthy", "control", "diagnosis", "hcc", "acs", "gc", "ovarian", "bladder", "alzheimer", "copd"]):
        labels.append("disease class")
    if any(k in hay for k in ["stage", "progression", "trajectory", "time-course", "treatment", "pre-", "post-"]):
        labels.append("stage / trajectory")
    if re.search(r"patient[_ -]?\d+", hay) or "sample_code" in hay:
        labels.append("patient ID")
    if "batch" in hay or "lab" in hay or "protocol" in hay:
        labels.append("batch / lab ID")
    if "replicate" in hay or "rep" in hay:
        labels.append("replicate ID")
    return "; ".join(labels) if labels else "unlabeled"


def infer_entity_type(title: str, desc: str) -> str:
    hay = f"{title} {desc}".lower()
    if any(k in hay for k in ["human", "patient", "serum", "saliva", "plasma", "blood"]):
        return "human"
    if any(k in hay for k in ["mouse", "mice", "rat", "murine"]):
        return "animal"
    if any(k in hay for k in ["synthetic", "standard", "pure", "metabolite", "adenine", "glucose", "cytochrome"]):
        return "synthetic vs pure standard"
    return "unknown"


def infer_raw_presence(title: str, desc: str, files: list[str]) -> tuple[str, str]:
    joined = " ".join([title, desc, *files]).lower()
    exts = {Path(name).suffix.lower() for name in files}
    raw_exts = {".csv", ".txt", ".xlsx", ".mat", ".zip", ".rar", ".opj", ".opju"}
    if any(ext in exts for ext in raw_exts):
        if any(k in joined for k in ["raw", "spectrum", "spectra", "raman", "sers"]):
            return "yes", "no"
    if exts and exts.issubset({".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}):
        return "no", "yes"
    if any(k in joined for k in ["supporting information", "supplementary", "datasheet"]) and not any(ext in exts for ext in raw_exts):
        return "no", "yes"
    return "unclear", "unclear"


def infer_ml_reuse(raw: str, label_type: str, files: list[str], desc: str) -> str:
    if raw != "yes":
        return "no"
    if any(Path(f).suffix.lower() in {".csv", ".txt", ".xlsx", ".mat", ".zip", ".rar"} for f in files):
        if any(k in (label_type + " " + desc).lower() for k in ["disease", "patient", "concentration", "batch", "class", "control", "healthy"]):
            return "yes"
        return "partial"
    return "partial"


def infer_provenance_clean(source_type: str, title: str, files: list[str], desc: str) -> str:
    if source_type in {"zenodo", "figshare"} and files:
        if any(k in (title + " " + desc).lower() for k in ["dataset", "data", "raw", "spectra"]):
            return "yes"
        return "partial"
    return "no"


def infer_wavenumber(desc: str, files: list[str]) -> str:
    hay = " ".join([desc, *files])
    matches = re.findall(r"(\d{2,4}\s*-\s*\d{2,4}\s*cm)", hay)
    if matches:
        return matches[0]
    matches = re.findall(r"(\d{3,4}\s*cm)", hay)
    if matches:
        return ", ".join(matches[:4])
    return ""


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def detect_overlap(url: str, title: str, datasets: list[dict[str, str]], discovery: list[dict[str, str]]) -> tuple[str, str, str]:
    norm_url = url.rstrip("/")
    lower_title = title.lower()
    for row in datasets:
        for key in ("source_url", "provenance_url", "raw_source_url"):
            if row.get(key, "").rstrip("/") == norm_url:
                return "yes", row["dataset_id"], "already_ingested_exact"
    for row in datasets:
        row_title = " ".join([row.get("name", ""), row.get("notes", "")]).lower()
        if row_title and any(tok in row_title for tok in lower_title.split()[:4]):
            if any(k in lower_title for k in ["adenine", "hcc", "covid", "shine", "small 2023", "serum", "ev", "metabolite", "plasmid"]):
                return "partial", row["dataset_id"], "scope_overlap"
    for row in discovery:
        if row.get("doi", "").endswith(norm_url.split("/")[-1]):
            return "partial", row.get("doi", ""), "discovery_overlap"
    return "no", "", "no_known_overlap"


def score_candidate(c: Candidate) -> None:
    c.raw_score = 5 if c.raw_spectra_present == "yes" else 2 if c.raw_spectra_present == "unclear" else 0
    file_types = set(t.strip() for t in c.file_types_available.split(";") if t.strip())
    c.metadata_score = min(5, 1 + int(bool(c.title)) + int(bool(c.files_count)) + int(bool(c.label_type != "unlabeled")) + int(bool(c.description_snippet)) + int(bool(c.archive_contents_note)))
    c.biological_relevance_score = {
        "serum": 4,
        "EV": 5,
        "saliva": 4,
        "urine": 4,
        "cells": 3,
        "pathogen": 3,
        "drug": 2,
        "molecule_reference": 4,
        "interlab / reproducibility": 5,
        "substrate / materials": 1,
        "other": 1,
    }.get(c.domain_category, 1)
    c.diversity_score = 0 if c.already_ingested == "yes" else 2
    if c.duplicate_status == "scope_overlap":
        c.diversity_score = 1
    if c.domain_category in {"saliva", "urine", "cells", "interlab / reproducibility", "molecule_reference"} and c.already_ingested != "yes":
        c.diversity_score = min(5, c.diversity_score + 2)
    c.shared_encoder_utility_score = 0
    if c.raw_spectra_present == "yes":
        c.shared_encoder_utility_score += 2
    if c.domain_category in {"serum", "EV", "saliva", "urine", "cells", "pathogen"}:
        c.shared_encoder_utility_score += 2
    if "disease class" in c.label_type or "patient ID" in c.label_type or "concentration" in c.label_type:
        c.shared_encoder_utility_score += 1
    c.shared_encoder_utility_score = min(5, c.shared_encoder_utility_score)
    c.grounding_utility_score = 5 if c.domain_category == "molecule_reference" and c.raw_spectra_present == "yes" else 2 if c.domain_category in {"drug", "pathogen"} and c.raw_spectra_present == "yes" else 0
    c.invariance_score = 0
    if "batch / lab ID" in c.label_type:
        c.invariance_score += 3
    if c.domain_category in {"interlab / reproducibility", "serum", "EV"} and c.raw_spectra_present == "yes":
        c.invariance_score += 2
    c.invariance_score = min(5, c.invariance_score)
    c.ingest_difficulty_score = 5
    if any(Path(name).suffix.lower() in {".csv", ".txt", ".xlsx", ".mat"} for name in c.file_names):
        c.ingest_difficulty_score = 2
    if any(Path(name).suffix.lower() in {".zip", ".rar", ".opj", ".opju"} for name in c.file_names):
        c.ingest_difficulty_score = 3
    if c.raw_spectra_present != "yes":
        c.ingest_difficulty_score = 4
    if c.already_ingested == "yes":
        c.ingest_difficulty_score = min(5, c.ingest_difficulty_score + 1)
    c.overall_value_score = round(
        (
            1.5 * c.raw_score
            + 1.2 * c.metadata_score
            + 1.3 * c.biological_relevance_score
            + 1.2 * c.diversity_score
            + 1.5 * c.shared_encoder_utility_score
            + 1.0 * c.grounding_utility_score
            + 1.0 * c.invariance_score
            + (5 - c.ingest_difficulty_score)
        )
        / 8.7,
        2,
    )
    c.recommended_action = classify_candidate(c)


def classify_candidate(c: Candidate) -> str:
    if c.already_ingested == "yes":
        return "reject"
    if c.raw_spectra_present == "yes" and c.provenance_clean in {"yes", "partial"} and c.overall_value_score >= 3.6:
        if c.ingest_difficulty_score <= 3 and c.domain_category not in {"substrate / materials", "other"}:
            return "ingest_now"
        return "ingest_later"
    if c.raw_spectra_present == "yes" and c.overall_value_score >= 2.7:
        return "ingest_later"
    if c.raw_spectra_present == "no" and c.domain_category in {"substrate / materials", "other"}:
        return "method_only"
    if c.raw_spectra_present == "no" and c.file_names and all(Path(name).suffix.lower() in {".pdf", ".docx"} for name in c.file_names):
        return "literature_only"
    if c.raw_spectra_present == "no":
        return "reject"
    return "method_only"


def build_rationale(c: Candidate) -> str:
    bits = []
    bits.append(f"raw={c.raw_spectra_present}")
    bits.append(f"domain={c.domain_category}")
    bits.append(f"labels={c.label_type}")
    if c.already_ingested == "yes":
        bits.append("already_in_gaira")
    elif c.duplicate_status == "scope_overlap":
        bits.append(f"overlaps={c.overlap_existing_dataset}")
    if c.archive_contents_note:
        bits.append(c.archive_contents_note)
    return "; ".join(bits)


def list_archive_contents(download_url: str, suffix: str, size: int) -> str:
    if suffix not in {".zip", ".xlsx"}:
        return ""
    if size and size > 15 * 1024 * 1024:
        return "archive too large to inspect in this pass"
    tmp_dir = OUT_ROOT / "downloads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{abs(hash(download_url))}{suffix}"
    proc = subprocess.run(
        ["curl", "-L", "-sS", "--max-time", "45", "-H", f"User-Agent: {TEXT_HEADERS['User-Agent']}", "-o", str(tmp_path), download_url],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return f"archive inspection download failed: {proc.stderr.strip()}"
    try:
        if suffix == ".zip":
            with zipfile.ZipFile(tmp_path) as zf:
                names = zf.namelist()[:12]
                extra = max(0, len(zf.namelist()) - len(names))
                tail = f" (+{extra} more)" if extra else ""
                return "zip contains: " + ", ".join(names) + tail
        if suffix == ".xlsx":
            proc = subprocess.run(
                ["python3", "-c", "import sys,openpyxl; wb=openpyxl.load_workbook(sys.argv[1], read_only=True); print(', '.join(wb.sheetnames))", str(tmp_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                return "xlsx sheets: " + proc.stdout.strip()
    except Exception as exc:
        return f"archive inspection failed: {exc}"
    return ""


def fetch_zenodo(url: str) -> Candidate:
    rec_id = re.search(r"/records?/(\d+)", url).group(1)
    payload = http_get_json(f"https://zenodo.org/api/records/{rec_id}")
    md = payload.get("metadata", {})
    files = payload.get("files", [])
    names = [f.get("key") or f.get("filename") or "" for f in files]
    sizes = [int(f.get("size") or 0) for f in files]
    title = md.get("title") or payload.get("title") or ""
    desc = normalize_space(md.get("description") or payload.get("description") or "")
    c = Candidate(
        source_url=url,
        source_type="zenodo",
        title=title,
        file_types_available="; ".join(sorted({Path(n).suffix.lower().lstrip(".") or "none" for n in names if n})),
        approx_dataset_size=summarize_sizes(sizes),
        analyte_or_biosample_type=normalize_space(md.get("keywords", [""])[0] if md.get("keywords") else ""),
        likely_wavenumber_range=infer_wavenumber(desc, names),
        description_snippet=desc[:500],
        files_count=len(names),
        file_names=names,
        file_sizes=sizes,
        source_record_id=rec_id,
    )
    c.domain_category = infer_domain(title, desc, names)
    c.label_type = infer_label_type(title, desc, names)
    c.entity_type = infer_entity_type(title, desc)
    c.raw_spectra_present, c.processed_only = infer_raw_presence(title, desc, names)
    c.reusable_for_ml = infer_ml_reuse(c.raw_spectra_present, c.label_type, names, desc)
    c.provenance_clean = infer_provenance_clean(c.source_type, title, names, desc)
    c.metadata_quality_note = f"Zenodo metadata with {len(names)} files."
    if files:
        for f in files[:3]:
            dl = f.get("links", {}).get("self") or f.get("links", {}).get("download")
            suffix = Path(f.get("key") or "").suffix.lower()
            if dl and suffix in {".zip", ".xlsx"}:
                c.archive_contents_note = list_archive_contents(dl, suffix, int(f.get("size") or 0))
                if c.archive_contents_note:
                    break
    return c


def figshare_article_id(url: str) -> str:
    m = re.search(r"/articles/[^/]+/[^/]+/(\d+)", url)
    if not m:
        raise ValueError(f"Could not parse figshare article id from {url}")
    return m.group(1)


def fetch_figshare(url: str) -> Candidate:
    article_id = figshare_article_id(url)
    payload = http_get_json(f"https://api.figshare.com/v2/articles/{article_id}")
    files = payload.get("files", [])
    names = [f.get("name") or "" for f in files]
    sizes = [int(f.get("size") or 0) for f in files]
    title = payload.get("title") or ""
    desc = normalize_space(payload.get("description") or "")
    categories = payload.get("categories") or []
    category_titles = [c.get("title", "") if isinstance(c, dict) else str(c) for c in categories]
    keywords = [str(k) for k in (payload.get("keywords") or payload.get("tags") or [])]
    c = Candidate(
        source_url=url,
        source_type="figshare",
        title=title,
        file_types_available="; ".join(sorted({Path(n).suffix.lower().lstrip(".") or "none" for n in names if n})),
        approx_dataset_size=summarize_sizes(sizes),
        analyte_or_biosample_type="; ".join((keywords + category_titles)[:4]),
        likely_wavenumber_range=infer_wavenumber(desc, names),
        description_snippet=desc[:500],
        files_count=len(names),
        file_names=names,
        file_sizes=sizes,
        source_record_id=article_id,
    )
    c.domain_category = infer_domain(title, desc, names)
    c.label_type = infer_label_type(title, desc, names)
    c.entity_type = infer_entity_type(title, desc)
    c.raw_spectra_present, c.processed_only = infer_raw_presence(title, desc, names)
    c.reusable_for_ml = infer_ml_reuse(c.raw_spectra_present, c.label_type, names, desc)
    c.provenance_clean = infer_provenance_clean(c.source_type, title, names, desc)
    c.metadata_quality_note = f"Figshare metadata with {len(names)} files."
    for f in files[:3]:
        dl = f.get("download_url")
        suffix = Path(f.get("name") or "").suffix.lower()
        if dl and suffix in {".zip", ".xlsx"}:
            c.archive_contents_note = list_archive_contents(dl, suffix, int(f.get("size") or 0))
            if c.archive_contents_note:
                break
    return c


def fetch_publisher(url: str) -> Candidate:
    html = http_get_text(url)
    title_match = re.search(r"<title>(.*?)</title>", html, re.I | re.S)
    title = normalize_space(title_match.group(1)) if title_match else url
    links = re.findall(r'href="([^"]+)"', html)
    files = [link for link in links if any(link.lower().endswith(ext) for ext in [".pdf", ".zip", ".xlsx", ".csv", ".txt", ".docx"])]
    desc_match = re.search(r'<meta name="dc.Description" content="([^"]+)"', html, re.I)
    desc = normalize_space(desc_match.group(1)) if desc_match else ""
    c = Candidate(
        source_url=url,
        source_type="paper",
        title=title,
        file_types_available="; ".join(sorted({Path(f).suffix.lower().lstrip(".") for f in files})),
        approx_dataset_size=f"{len(files)} linked supplemental files" if files else "",
        likely_wavenumber_range=infer_wavenumber(desc, files),
        description_snippet=desc[:500],
        files_count=len(files),
        file_names=files,
        source_record_id=url,
    )
    c.domain_category = infer_domain(title, desc, files)
    c.label_type = infer_label_type(title, desc, files)
    c.entity_type = infer_entity_type(title, desc)
    c.raw_spectra_present, c.processed_only = infer_raw_presence(title, desc, files)
    c.reusable_for_ml = infer_ml_reuse(c.raw_spectra_present, c.label_type, files, desc)
    c.provenance_clean = "partial" if files else "no"
    c.metadata_quality_note = "Publisher page scraped for supplemental links."
    if files:
        c.archive_contents_note = "publisher-linked supplement files present"
    return c


def fetch_candidate(url: str) -> Candidate:
    if "zenodo.org" in url:
        return fetch_zenodo(url)
    if "figshare.com" in url:
        return fetch_figshare(url)
    return fetch_publisher(url)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(candidates: list[Candidate]) -> str:
    sorted_candidates = sorted(candidates, key=lambda c: (-c.overall_value_score, c.title.lower()))
    best = [c for c in sorted_candidates if c.recommended_action == "ingest_now"][:10]
    redundant = [c for c in sorted_candidates if c.already_ingested == "yes" or c.duplicate_status == "scope_overlap"][:12]
    tempting = [c for c in sorted_candidates if c.recommended_action in {"ingest_later", "method_only"}][:12]
    method_only = [c for c in sorted_candidates if c.recommended_action == "method_only"][:12]
    lines = []
    lines.append("# GAIRA Global v2 Dataset Audit")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Audited candidates: {len(candidates)}")
    counts = Counter(c.recommended_action for c in candidates)
    for key in ["ingest_now", "ingest_later", "method_only", "literature_only", "reject"]:
        lines.append(f"- {key}: {counts.get(key, 0)}")
    lines.append("")
    lines.append("## Best Additions Right Now")
    lines.append("")
    for c in best:
        lines.append(f"- {c.title} | {c.source_url} | score={c.overall_value_score} | {c.rationale}")
    lines.append("")
    lines.append("## Redundant Or Already Covered")
    lines.append("")
    for c in redundant:
        lines.append(f"- {c.title} | {c.source_url} | overlap={c.overlap_existing_dataset or c.duplicate_status}")
    lines.append("")
    lines.append("## Tempting But Low-Value")
    lines.append("")
    for c in tempting:
        lines.append(f"- {c.title} | {c.source_url} | action={c.recommended_action} | {c.rationale}")
    lines.append("")
    lines.append("## Methodology References Only")
    lines.append("")
    for c in method_only:
        lines.append(f"- {c.title} | {c.source_url}")
    lines.append("")
    lines.append("## Tiered Plan")
    lines.append("")
    lines.append("### Tier A: ingest immediately")
    for c in sorted_candidates:
        if c.recommended_action == "ingest_now":
            lines.append(f"- {c.title} | {c.source_url}")
    lines.append("")
    lines.append("### Tier B: deeper manual reconstruction")
    for c in sorted_candidates:
        if c.recommended_action == "ingest_later":
            lines.append(f"- {c.title} | {c.source_url}")
    lines.append("")
    lines.append("### Tier C: skip for now")
    for c in sorted_candidates:
        if c.recommended_action in {"method_only", "literature_only", "reject"}:
            lines.append(f"- {c.title} | {c.source_url} | {c.recommended_action}")
    lines.append("")
    lines.append("## Candidate Table")
    lines.append("")
    lines.append("| Action | Score | Domain | Raw | Candidate | Notes |")
    lines.append("|---|---:|---|---|---|---|")
    for c in sorted_candidates:
        lines.append(f"| {c.recommended_action} | {c.overall_value_score} | {c.domain_category} | {c.raw_spectra_present} | {c.title} | {c.rationale} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    datasets = read_csv_rows(DATASETS_CSV)
    discovery = read_csv_rows(DISCOVERY_CSV)
    candidates: list[Candidate] = []
    for idx, url in enumerate(dict.fromkeys(CANDIDATE_URLS), start=1):
        try:
            candidate = fetch_candidate(url)
            already, overlap_id, dup = detect_overlap(url, candidate.title, datasets, discovery)
            candidate.already_ingested = "yes" if already == "yes" else "no"
            candidate.overlap_existing_dataset = overlap_id
            candidate.duplicate_status = dup
            candidate.duplicate_notes = overlap_id
            score_candidate(candidate)
            candidate.rationale = build_rationale(candidate)
        except Exception as exc:
            candidate = Candidate(
                source_url=url,
                source_type="other",
                title="FETCH_FAILED",
                provenance_clean="no",
                duplicate_status="fetch_failed",
                recommended_action="reject",
                rationale=f"fetch_failed: {exc}",
            )
        candidates.append(candidate)
        time.sleep(0.2)

    inventory_rows = []
    scoring_rows = []
    dedup_rows = []
    priority_rows = []
    for c in candidates:
        inventory_rows.append(
            {
                "source_url": c.source_url,
                "source_type": c.source_type,
                "title": c.title,
                "raw_spectra_present": c.raw_spectra_present,
                "processed_only": c.processed_only,
                "file_types_available": c.file_types_available,
                "approx_dataset_size": c.approx_dataset_size,
                "analyte_or_biosample_type": c.analyte_or_biosample_type,
                "domain_category": c.domain_category,
                "label_type": c.label_type,
                "entity_type": c.entity_type,
                "likely_wavenumber_range": c.likely_wavenumber_range,
                "reusable_for_ml": c.reusable_for_ml,
                "provenance_clean": c.provenance_clean,
                "archive_contents_note": c.archive_contents_note,
                "metadata_quality_note": c.metadata_quality_note,
            }
        )
        scoring_rows.append(
            {
                "source_url": c.source_url,
                "title": c.title,
                "raw_data_availability_score": c.raw_score,
                "metadata_quality_score": c.metadata_score,
                "biological_relevance_score": c.biological_relevance_score,
                "diversity_contribution_score": c.diversity_score,
                "shared_encoder_training_utility_score": c.shared_encoder_utility_score,
                "grounding_utility_score": c.grounding_utility_score,
                "invariance_utility_score": c.invariance_score,
                "ingest_difficulty_score": c.ingest_difficulty_score,
                "overall_value_score": c.overall_value_score,
                "recommended_action": c.recommended_action,
                "rationale": c.rationale,
            }
        )
        dedup_rows.append(
            {
                "source_url": c.source_url,
                "title": c.title,
                "already_ingested": c.already_ingested,
                "duplicate_status": c.duplicate_status,
                "overlap_existing_dataset": c.overlap_existing_dataset,
                "duplicate_notes": c.duplicate_notes,
            }
        )
        priority_rows.append(
            {
                "priority_rank": "",
                "recommended_action": c.recommended_action,
                "overall_value_score": c.overall_value_score,
                "title": c.title,
                "source_url": c.source_url,
                "domain_category": c.domain_category,
                "raw_spectra_present": c.raw_spectra_present,
                "label_type": c.label_type,
                "ingest_difficulty_score": c.ingest_difficulty_score,
                "rationale": c.rationale,
            }
        )

    priority_rows.sort(key=lambda row: (-float(row["overall_value_score"]), row["title"]))
    for i, row in enumerate(priority_rows, start=1):
        row["priority_rank"] = i

    write_csv(OUT_ROOT / "tables/global_v2_candidate_dataset_inventory.csv", inventory_rows)
    write_csv(OUT_ROOT / "tables/global_v2_candidate_dataset_scoring.csv", scoring_rows)
    write_csv(OUT_ROOT / "tables/global_v2_deduplication_check.csv", dedup_rows)
    write_csv(OUT_ROOT / "tables/global_v2_ingest_priority_list.csv", priority_rows)
    report = build_report(candidates)
    report_path = OUT_ROOT / "report/global_v2_dataset_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(str(OUT_ROOT))


if __name__ == "__main__":
    main()
