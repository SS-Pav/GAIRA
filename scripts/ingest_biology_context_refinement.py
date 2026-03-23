from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd
from docx import Document
from pypdf import PdfReader


DIABETES_PDF = Path("/Users/suraj/Downloads/diabetes_EV_arxiv.pdf")
DIABETES_SUPP = Path("/Users/suraj/Downloads/media-1 (1).docx")
SHINE_PDF = Path("/Users/suraj/Downloads/spectra_shine.pdf")
SHINE_SUPP = Path("/Users/suraj/Downloads/1-s2.0-S0925400526000985-mmc1.docx")

EV_CONTEXT_LAYER = "GAIRA_EV_CONTEXT"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _read_pdf_text(path: Path) -> tuple[str, list[str]]:
    reader = PdfReader(str(path))
    pages = [_clean(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages), pages


def _read_docx_text(path: Path) -> str:
    document = Document(str(path))
    return "\n".join(_clean(paragraph.text) for paragraph in document.paragraphs if _clean(paragraph.text))


def _ensure_sources() -> None:
    missing = [str(path) for path in [DIABETES_PDF, DIABETES_SUPP, SHINE_PDF, SHINE_SUPP] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")


def _write_markdown(path: Path, title: str, sections: list[tuple[str, list[str]]]) -> None:
    lines = [f"# {title}", ""]
    for heading, bullets in sections:
        lines.append(f"## {heading}")
        lines.append("")
        for bullet in bullets:
            lines.append(f"- {bullet}")
        lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _build_diabetes_band_table() -> pd.DataFrame:
    rows = [
        {
            "dataset_id": "diabetes_plasma_ev_sers",
            "comparison": "OWD_vs_NWD",
            "band_label": "1060-1130",
            "direction": "higher_in_OWD",
            "assignment": "lipid C-C stretching",
            "support_level": "supplementary_difference_spectrum",
            "source_ref": "media-1 (1).docx Supplementary Figure 2",
        },
        {
            "dataset_id": "diabetes_plasma_ev_sers",
            "comparison": "OWD_vs_NWD",
            "band_label": "1440-1460",
            "direction": "higher_in_OWD",
            "assignment": "CH2/CH3 deformation; lipid-associated",
            "support_level": "supplementary_difference_spectrum",
            "source_ref": "media-1 (1).docx Supplementary Figure 2",
        },
        {
            "dataset_id": "diabetes_plasma_ev_sers",
            "comparison": "OWD_vs_NWD",
            "band_label": "1240-1280",
            "direction": "higher_in_NWD",
            "assignment": "amide III / protein-associated",
            "support_level": "supplementary_difference_spectrum",
            "source_ref": "media-1 (1).docx Supplementary Figure 2",
        },
        {
            "dataset_id": "diabetes_plasma_ev_sers",
            "comparison": "OWD_vs_NWD",
            "band_label": "830-850",
            "direction": "higher_in_NWD",
            "assignment": "tyrosine doublet region",
            "support_level": "supplementary_difference_spectrum",
            "source_ref": "media-1 (1).docx Supplementary Figure 2",
        },
        {
            "dataset_id": "diabetes_plasma_ev_sers",
            "comparison": "OWD_vs_NWD",
            "band_label": "1002-1003",
            "direction": "higher_in_OWD",
            "assignment": "phenylalanine ring breathing",
            "support_level": "supplementary_boxplot",
            "source_ref": "media-1 (1).docx Supplementary Figure 4",
        },
        {
            "dataset_id": "diabetes_plasma_ev_sers",
            "comparison": "OWD_vs_NWD",
            "band_label": "797",
            "direction": "higher_in_NWD",
            "assignment": "O-P-O backbone / nucleic-acid-associated",
            "support_level": "supplementary_boxplot",
            "source_ref": "media-1 (1).docx Supplementary Figure 4",
        },
        {
            "dataset_id": "diabetes_plasma_ev_sers",
            "comparison": "OWD_vs_NWD",
            "band_label": "1058",
            "direction": "higher_in_NWD",
            "assignment": "lipid C-C stretching",
            "support_level": "supplementary_boxplot",
            "source_ref": "media-1 (1).docx Supplementary Figure 4",
        },
        {
            "dataset_id": "diabetes_plasma_ev_sers",
            "comparison": "OWD_vs_NWD",
            "band_label": "1256-1263",
            "direction": "higher_in_NWD",
            "assignment": "amide III / protein-associated",
            "support_level": "supplementary_boxplot",
            "source_ref": "media-1 (1).docx Supplementary Figure 4",
        },
        {
            "dataset_id": "diabetes_plasma_ev_sers",
            "comparison": "OWD_vs_NWD",
            "band_label": "1482",
            "direction": "trend_higher_in_NWD",
            "assignment": "guanine/adenine-like nucleic-acid support",
            "support_level": "supplementary_boxplot",
            "source_ref": "media-1 (1).docx Supplementary Figure 4",
        },
    ]
    return pd.DataFrame(rows)


def _build_shine_band_table() -> pd.DataFrame:
    rows = [
        {
            "dataset_id": "shine_ev_sers",
            "condition": "day2_apap_response",
            "band_label": "739",
            "assignment": "RNA-like / miR-122-linked in paper table",
            "functional_link": "albumin and CCK8 correlated",
            "source_ref": "spectra_shine.pdf pp. 3, 8",
        },
        {
            "dataset_id": "shine_ev_sers",
            "condition": "day2_apap_response",
            "band_label": "960",
            "assignment": "ribose / phosphate-associated",
            "functional_link": "albumin and CCK8 correlated",
            "source_ref": "spectra_shine.pdf pp. 3, 8",
        },
        {
            "dataset_id": "shine_ev_sers",
            "condition": "day2_apap_response",
            "band_label": "1250",
            "assignment": "amide III and related biomolecular support",
            "functional_link": "albumin and CCK8 correlated",
            "source_ref": "spectra_shine.pdf pp. 3, 5, 8",
        },
        {
            "dataset_id": "shine_ev_sers",
            "condition": "day2_apap_response",
            "band_label": "1525",
            "assignment": "protein/nucleic-acid injury-linked support",
            "functional_link": "albumin and CCK8 correlated",
            "source_ref": "spectra_shine.pdf pp. 3, 5, 8",
        },
        {
            "dataset_id": "shine_ev_sers",
            "condition": "day2_apap_response",
            "band_label": "1576",
            "assignment": "protein/nucleic-acid injury-linked support",
            "functional_link": "albumin and CCK8 correlated",
            "source_ref": "spectra_shine.pdf pp. 3, 5, 8",
        },
        {
            "dataset_id": "shine_ev_sers",
            "condition": "day2_apap_response",
            "band_label": "1602",
            "assignment": "protein/nucleic-acid injury-linked support",
            "functional_link": "albumin and CCK8 correlated",
            "source_ref": "spectra_shine.pdf pp. 3, 5, 8",
        },
    ]
    return pd.DataFrame(rows)


def _build_support_document_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    support_documents = [
        {
            "document_id": "diabetes_ev_context_support_doc_001",
            "dataset_id": "diabetes_ev_context_support",
            "source_dataset_id": "diabetes_plasma_ev_sers",
            "evidence_family": "ev_dataset_context_support",
            "evidence_tier": "tier2_literature_support",
            "support_type": "text",
            "citation_label": "Parlatan_2026_diabetes_EV_context",
            "title": "Structured diabetes EV subgroup interpretation support",
            "authors": "Parlatan et al.",
            "year": "2026",
            "journal": "bioRxiv + supplementary notes",
            "doi": "10.64898/2026.03.14.711704",
            "source_file": f"{DIABETES_PDF}|{DIABETES_SUPP}",
            "is_digitized": "no",
            "use_for_primary_matching": "no",
            "use_for_supporting_comparison": "yes",
            "use_for_rag": "yes",
            "notes": (
                "Support-only EV interpretation document grounded in the provided diabetes EV paper and supplement. "
                "It informs subgroup heterogeneity, lipid/protein/nucleic-acid framing, and weak-label caution for "
                "diabetes_plasma_ev_sers without turning the paper into direct spectral truth."
            ),
        },
        {
            "document_id": "shine_spectra_context_support_doc_001",
            "dataset_id": "shine_spectra_context_support",
            "source_dataset_id": "shine_ev_sers",
            "evidence_family": "ev_dataset_context_support",
            "evidence_tier": "tier2_literature_support",
            "support_type": "text",
            "citation_label": "Parlatan_2026_SPECTRA_context",
            "title": "Structured SHINE/SPECTRA injury-response interpretation support",
            "authors": "Parlatan et al.",
            "year": "2026",
            "journal": "Sensors and Actuators B + supplementary notes",
            "doi": "10.1016/j.snb.2026.139520",
            "source_file": f"{SHINE_PDF}|{SHINE_SUPP}",
            "is_digitized": "no",
            "use_for_primary_matching": "no",
            "use_for_supporting_comparison": "yes",
            "use_for_rag": "yes",
            "notes": (
                "Support-only EV interpretation document grounded in the provided SPECTRA/SHINE paper and supplement. "
                "It informs dose-response injury framing, correlated wavenumbers, and preclinical caution for shine_ev_sers."
            ),
        },
    ]
    support_chunks = [
        {
            "chunk_id": "diabetes_ev_context_support_doc_001_chunk_01",
            "document_id": "diabetes_ev_context_support_doc_001",
            "dataset_id": "diabetes_ev_context_support",
            "chunk_order": 1,
            "section": "study_design",
            "chunk_text": (
                "diabetes_plasma_ev_sers should be interpreted against the paper's intended multimodal design: "
                "plasma EVs from Asian normal-weight diabetes, Asian overweight diabetes, White normal-weight diabetes, "
                "and White overweight diabetes were analyzed by SERS and EV-RNA sequencing. The live GAIRA archive does "
                "not reconstruct those four subgroup IDs, so current Impact and Strong-D archive labels are weak-label "
                "cohort-family surrogates rather than the full subgroup design."
            ),
            "metadata_json": json.dumps({"source_kind": "paper_structured_note", "source_page": "2-3,17"}, sort_keys=True),
        },
        {
            "chunk_id": "diabetes_ev_context_support_doc_001_chunk_02",
            "document_id": "diabetes_ev_context_support_doc_001",
            "dataset_id": "diabetes_ev_context_support",
            "chunk_order": 2,
            "section": "spectral_patterns",
            "chunk_text": (
                "The diabetes EV supplement describes subgroup-linked spectral structure rather than single-molecule truth. "
                "Overweight diabetes was enriched around 1060-1130 and 1440-1460 cm^-1 with lipid-associated wording, while "
                "normal-weight diabetes retained stronger ~797, ~1058, and ~1256-1263 cm^-1 support linked to nucleic-acid, "
                "lipid, and amide-III features. A ~1482 cm^-1 guanine or adenine-like trend was reported but not significant."
            ),
            "metadata_json": json.dumps({"source_kind": "paper_structured_note", "source_page": "supplementary_figures_2_4"}, sort_keys=True),
        },
        {
            "chunk_id": "diabetes_ev_context_support_doc_001_chunk_03",
            "document_id": "diabetes_ev_context_support_doc_001",
            "dataset_id": "diabetes_ev_context_support",
            "chunk_order": 3,
            "section": "rna_and_pathway_context",
            "chunk_text": (
                "The diabetes EV paper links higher miR-208a and miR-132 in Asian overweight diabetes and higher miR-484 "
                "in Asian normal-weight diabetes to insulin signaling, systemic energy homeostasis, beta-cell adaptation, "
                "and mitochondrial dynamics. These RNA and pathway themes are support-level interpretation aids for "
                "diabetes_plasma_ev_sers, not direct proof that any single SERS band uniquely reports one pathway."
            ),
            "metadata_json": json.dumps({"source_kind": "paper_structured_note", "source_page": "4-5,16"}, sort_keys=True),
        },
        {
            "chunk_id": "diabetes_ev_context_support_doc_001_chunk_04",
            "document_id": "diabetes_ev_context_support_doc_001",
            "dataset_id": "diabetes_ev_context_support",
            "chunk_order": 4,
            "section": "cautions",
            "chunk_text": (
                "The diabetes EV paper explicitly warns that some subgroup signatures partially overlap, including "
                "convergence between Asian normal-weight and White overweight patterns. It also notes that plasma isolates "
                "can contain lipoproteins and other non-EV nanoparticles with overlapping size and biochemical signatures. "
                "GAIRA should therefore surface subgroup overlap, weak-label, and not-EV-exclusive cautions before making "
                "patient-level or subgroup-specific claims."
            ),
            "metadata_json": json.dumps({"source_kind": "paper_structured_note", "source_page": "3,8,17,29-30"}, sort_keys=True),
        },
        {
            "chunk_id": "shine_spectra_context_support_doc_001_chunk_01",
            "document_id": "shine_spectra_context_support_doc_001",
            "dataset_id": "shine_spectra_context_support",
            "chunk_order": 1,
            "section": "study_design",
            "chunk_text": (
                "shine_ev_sers maps to the SPECTRA study's dose-resolved APAP injury design rather than a generic disease "
                "classifier. Primary rat hepatocyte cultures were sampled at Day 0 and Day 2 across 0, 10, 20, and 40 mM "
                "acetaminophen, with EVs measured on a gold nanopillar SERS surface. The task is best read as perturbation "
                "and injury-response structure with dose and time context."
            ),
            "metadata_json": json.dumps({"source_kind": "paper_structured_note", "source_page": "1-2,supplement"}, sort_keys=True),
        },
        {
            "chunk_id": "shine_spectra_context_support_doc_001_chunk_02",
            "document_id": "shine_spectra_context_support_doc_001",
            "dataset_id": "shine_spectra_context_support",
            "chunk_order": 2,
            "section": "correlated_wavenumbers",
            "chunk_text": (
                "The SPECTRA paper highlights 739, 960, 1250, 1525, 1576, and 1602 cm^-1 as strongly correlated with "
                "albumin and CCK8 viability readouts, with the 1525, 1576, and 1602 cm^-1 bands described as mainly "
                "nucleic-acid and protein vibrational support. Amide III is also used as supportive evidence for EV cargo "
                "changes beyond nucleic acids."
            ),
            "metadata_json": json.dumps({"source_kind": "paper_structured_note", "source_page": "3,5,8"}, sort_keys=True),
        },
        {
            "chunk_id": "shine_spectra_context_support_doc_001_chunk_03",
            "document_id": "shine_spectra_context_support_doc_001",
            "dataset_id": "shine_spectra_context_support",
            "chunk_order": 3,
            "section": "preprocessing_and_use",
            "chunk_text": (
                "The SPECTRA supplement describes polynomial wavenumber calibration, asymmetric least-squares baseline "
                "correction, Savitzky-Golay smoothing, internal normalization to the 642 cm^-1 silicon substrate peak, "
                "and k-means filtering of non-EV-like measurements. These steps support interpreting shine_ev_sers as a "
                "curated injury-response EV dataset, not a raw direct-molecule archive."
            ),
            "metadata_json": json.dumps({"source_kind": "paper_structured_note", "source_page": "supplement_preprocessing"}, sort_keys=True),
        },
        {
            "chunk_id": "shine_spectra_context_support_doc_001_chunk_04",
            "document_id": "shine_spectra_context_support_doc_001",
            "dataset_id": "shine_spectra_context_support",
            "chunk_order": 4,
            "section": "cautions",
            "chunk_text": (
                "The SHINE/SPECTRA study should stay preclinical and support-level in GAIRA. The paper states that the "
                "current study is limited to a monoculture system, a single hepatotoxicant, and only two biological "
                "replicates, and it calls for validation across additional toxicants, co-culture or organoid settings, "
                "and more complex EV mixtures before treating these bands as broad liver-disease truth."
            ),
            "metadata_json": json.dumps({"source_kind": "paper_structured_note", "source_page": "5,supplement"}, sort_keys=True),
        },
    ]
    return pd.DataFrame(support_documents), pd.DataFrame(support_chunks)


def _build_ev_context_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    documents = [
        {
            "document_id": "gaira_ev_context_diabetes_refined_design_note",
            "context_layer": EV_CONTEXT_LAYER,
            "intended_domain": "ev",
            "context_type": "interpretive_note",
            "evidence_basis": "derived_from_provided_paper",
            "source_dataset_id": "diabetes_plasma_ev_sers",
            "source_file": f"{DIABETES_PDF}|{DIABETES_SUPP}",
            "title": "Diabetes EV subgroup-design note",
            "use_for_rag": "yes",
            "notes": "Grounded note for subgroup design and weak-label framing of diabetes_plasma_ev_sers.",
        },
        {
            "document_id": "gaira_ev_context_diabetes_refined_biology_note",
            "context_layer": EV_CONTEXT_LAYER,
            "intended_domain": "ev",
            "context_type": "interpretive_note",
            "evidence_basis": "derived_from_provided_paper",
            "source_dataset_id": "diabetes_plasma_ev_sers",
            "source_file": f"{DIABETES_PDF}|{DIABETES_SUPP}",
            "title": "Diabetes EV spectral-biology note",
            "use_for_rag": "yes",
            "notes": "Grounded note for diabetes EV lipid/protein/nucleic-acid and RNA interpretation.",
        },
        {
            "document_id": "gaira_ev_context_diabetes_refined_caution_note",
            "context_layer": EV_CONTEXT_LAYER,
            "intended_domain": "ev",
            "context_type": "caveat",
            "evidence_basis": "derived_from_provided_paper",
            "source_dataset_id": "diabetes_plasma_ev_sers",
            "source_file": f"{DIABETES_PDF}|{DIABETES_SUPP}",
            "title": "Diabetes EV overlap and non-exclusive signal caution",
            "use_for_rag": "yes",
            "notes": "Grounded caution for overlap, lipoprotein carryover, and patient-level overclaiming.",
        },
        {
            "document_id": "gaira_ev_context_shine_refined_design_note",
            "context_layer": EV_CONTEXT_LAYER,
            "intended_domain": "ev",
            "context_type": "interpretive_note",
            "evidence_basis": "derived_from_provided_paper",
            "source_dataset_id": "shine_ev_sers",
            "source_file": f"{SHINE_PDF}|{SHINE_SUPP}",
            "title": "SHINE/SPECTRA dose-response design note",
            "use_for_rag": "yes",
            "notes": "Grounded note for shine_ev_sers as dose-resolved APAP injury-response EV data.",
        },
        {
            "document_id": "gaira_ev_context_shine_refined_band_note",
            "context_layer": EV_CONTEXT_LAYER,
            "intended_domain": "ev",
            "context_type": "interpretive_note",
            "evidence_basis": "derived_from_provided_paper",
            "source_dataset_id": "shine_ev_sers",
            "source_file": f"{SHINE_PDF}|{SHINE_SUPP}",
            "title": "SHINE/SPECTRA assay-correlation band note",
            "use_for_rag": "yes",
            "notes": "Grounded note for SHINE injury-linked wavenumbers and assay correlation.",
        },
        {
            "document_id": "gaira_ev_context_shine_refined_caution_note",
            "context_layer": EV_CONTEXT_LAYER,
            "intended_domain": "ev",
            "context_type": "caveat",
            "evidence_basis": "derived_from_provided_paper",
            "source_dataset_id": "shine_ev_sers",
            "source_file": f"{SHINE_PDF}|{SHINE_SUPP}",
            "title": "SHINE/SPECTRA preclinical caution note",
            "use_for_rag": "yes",
            "notes": "Grounded caution for monoculture, single-toxicant, and limited-replicate scope.",
        },
    ]
    chunks = [
        {
            "chunk_id": "gaira_ev_context_diabetes_refined_design_note_chunk_01",
            "document_id": "gaira_ev_context_diabetes_refined_design_note",
            "context_layer": EV_CONTEXT_LAYER,
            "intended_domain": "ev",
            "chunk_order": 1,
            "section": "diabetes_subgroup_design",
            "chunk_text": (
                "The provided diabetes EV paper shows that diabetes_plasma_ev_sers belongs to a structured metabolic "
                "heterogeneity study with Asian normal-weight, Asian overweight, White normal-weight, and White overweight "
                "T2DM subgroups. GAIRA still cannot reconstruct those subgroup labels from the released Figure3ProcessedArchive, "
                "so Impact and Strong-D remain archive-supported cohort-family labels rather than full subgroup identities."
            ),
            "metadata_json": json.dumps({"source_page": "2-3,17", "source_kind": "provided_paper_note"}, sort_keys=True),
        },
        {
            "chunk_id": "gaira_ev_context_diabetes_refined_biology_note_chunk_01",
            "document_id": "gaira_ev_context_diabetes_refined_biology_note",
            "context_layer": EV_CONTEXT_LAYER,
            "intended_domain": "ev",
            "chunk_order": 1,
            "section": "diabetes_lipid_protein_nucleic_note",
            "chunk_text": (
                "The diabetes EV supplement supports a more specific biology framing than a generic weak-label diabetes set. "
                "Overweight diabetes is described with stronger lipid-associated regions near 1060-1130 and 1440-1460 cm^-1, "
                "while normal-weight diabetes retains stronger nucleic-acid and amide-III support around ~797, ~1058, and "
                "~1256-1263 cm^-1. Paper-level RNA context adds miR-208a/miR-132 insulin-signaling support in overweight "
                "samples and miR-484 mitochondrial-dynamics support in normal-weight samples."
            ),
            "metadata_json": json.dumps({"source_page": "4-5,16,supplementary_figures_2_4_5"}, sort_keys=True),
        },
        {
            "chunk_id": "gaira_ev_context_diabetes_refined_caution_note_chunk_01",
            "document_id": "gaira_ev_context_diabetes_refined_caution_note",
            "context_layer": EV_CONTEXT_LAYER,
            "intended_domain": "ev",
            "chunk_order": 1,
            "section": "diabetes_overlap_and_nonexclusive_caution",
            "chunk_text": (
                "The diabetes EV paper explicitly reports partial overlap between some subgroup signatures, including "
                "convergence between Asian normal-weight and White overweight patterns. It also warns that plasma EV "
                "isolates may include lipoproteins and other non-EV nanoparticles with overlapping biochemical signatures. "
                "GAIRA should therefore treat diabetes_plasma_ev_sers as descriptive subgroup-pattern support with weak-label "
                "and non-EV-exclusive caution, not as patient-level subtype truth."
            ),
            "metadata_json": json.dumps({"source_page": "3,8,17,29", "source_kind": "provided_paper_note"}, sort_keys=True),
        },
        {
            "chunk_id": "gaira_ev_context_shine_refined_design_note_chunk_01",
            "document_id": "gaira_ev_context_shine_refined_design_note",
            "context_layer": EV_CONTEXT_LAYER,
            "intended_domain": "ev",
            "chunk_order": 1,
            "section": "shine_dose_response_design",
            "chunk_text": (
                "shine_ev_sers should be interpreted as SPECTRA dose-response EV biology, not a generic disease-classification "
                "set. The paper uses primary rat hepatocytes exposed to 0, 10, 20, and 40 mM APAP with Day 0 and Day 2 "
                "measurements, and frames the task as predicting injury-dose structure from EV cargo changes."
            ),
            "metadata_json": json.dumps({"source_page": "1-2,supplement", "source_kind": "provided_paper_note"}, sort_keys=True),
        },
        {
            "chunk_id": "gaira_ev_context_shine_refined_band_note_chunk_01",
            "document_id": "gaira_ev_context_shine_refined_band_note",
            "context_layer": EV_CONTEXT_LAYER,
            "intended_domain": "ev",
            "chunk_order": 1,
            "section": "shine_assay_correlated_bands",
            "chunk_text": (
                "The SPECTRA paper links 739, 960, 1250, 1525, 1576, and 1602 cm^-1 to albumin and CCK8 viability trends, "
                "with 1525/1576/1602 framed mainly as nucleic-acid and protein vibrational support and amide III used as "
                "additional cargo-change support. This makes shine_ev_sers useful for injury and stress-response reasoning, "
                "especially oxidative or protein/nucleic-acid perturbation framing."
            ),
            "metadata_json": json.dumps({"source_page": "3-5,8", "source_kind": "provided_paper_note"}, sort_keys=True),
        },
        {
            "chunk_id": "gaira_ev_context_shine_refined_caution_note_chunk_01",
            "document_id": "gaira_ev_context_shine_refined_caution_note",
            "context_layer": EV_CONTEXT_LAYER,
            "intended_domain": "ev",
            "chunk_order": 1,
            "section": "shine_preclinical_caution",
            "chunk_text": (
                "The SHINE/SPECTRA paper says the current evidence is limited to a monoculture system, a single hepatotoxicant, "
                "and only two biological replicates. GAIRA should surface shine_ev_sers as useful injury-response and hepatotoxicity "
                "support, but not as broad human liver-disease truth or universal EV biology."
            ),
            "metadata_json": json.dumps({"source_page": "5,supplement", "source_kind": "provided_paper_note"}, sort_keys=True),
        },
    ]
    return pd.DataFrame(documents), pd.DataFrame(chunks)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_database_path, get_storage_paths, require_data_root_exists

    _ensure_sources()
    storage_paths = require_data_root_exists()
    processed_root = storage_paths["processed_data"]
    extraction_root = processed_root / "context_extraction"
    extraction_root.mkdir(parents=True, exist_ok=True)
    db_path = get_database_path()

    diabetes_pdf_text, diabetes_pdf_pages = _read_pdf_text(DIABETES_PDF)
    shine_pdf_text, shine_pdf_pages = _read_pdf_text(SHINE_PDF)
    diabetes_supp_text = _read_docx_text(DIABETES_SUPP)
    shine_supp_text = _read_docx_text(SHINE_SUPP)

    required_checks = {
        "diabetes_pdf": ["A-NWD", "miR-208a", "miR-484", "lipoprotein"],
        "diabetes_supp": ["1060–1130", "1440–1460", "1256", "1482"],
        "shine_pdf": ["Acetaminophen", "1525", "1576", "1602", "monoculture"],
        "shine_supp": ["asymmetric least squares", "Savitzky-Golay", "642", "CCK8"],
    }
    corpora = {
        "diabetes_pdf": diabetes_pdf_text,
        "diabetes_supp": diabetes_supp_text,
        "shine_pdf": shine_pdf_text,
        "shine_supp": shine_supp_text,
    }
    for corpus_name, phrases in required_checks.items():
        text = corpora[corpus_name]
        missing = [phrase for phrase in phrases if phrase.lower() not in text.lower()]
        if missing:
            raise ValueError(f"Missing expected phrases in {corpus_name}: {missing}")

    diabetes_band_df = _build_diabetes_band_table()
    shine_band_df = _build_shine_band_table()
    diabetes_band_path = extraction_root / "diabetes_ev_band_annotations.csv"
    shine_band_path = extraction_root / "spectra_shine_band_annotations.csv"
    diabetes_band_df.to_csv(diabetes_band_path, index=False)
    shine_band_df.to_csv(shine_band_path, index=False)

    diabetes_md_path = extraction_root / "diabetes_ev_structured_notes.md"
    shine_md_path = extraction_root / "spectra_shine_structured_notes.md"
    _write_markdown(
        diabetes_md_path,
        "Diabetes EV Structured Notes",
        [
            (
                "Study Design",
                [
                    "The paper describes a multimodal plasma EV study across Asian normal-weight, Asian overweight, White normal-weight, and White overweight T2DM subgroups; SERS used n=65 and EV-RNA sequencing used n=39.",
                    "GAIRA's live archive still only preserves Impact and Strong-D archive families from the Figure 3 MAT release, so the paper's four-subgroup design is support-level context rather than reconstructed live labels.",
                    "The authors report partial convergence between Asian normal-weight and White overweight patterns, arguing that BMI alone does not capture all shared metabolic states.",
                ],
            ),
            (
                "Spectral Notes",
                [
                    "OWD-enriched regions: 1060-1130 and 1440-1460 cm^-1 with lipid-associated wording.",
                    "NWD-enriched regions: ~797, ~1058, and ~1256-1263 cm^-1 with nucleic-acid, lipid, and amide-III support.",
                    "1002-1003 cm^-1 phenylalanine-associated intensity is higher in OWD; ~1482 cm^-1 guanine/adenine-like support trends higher in NWD but is not significant.",
                ],
            ),
            (
                "RNA And Pathway Framing",
                [
                    "miR-208a and miR-132 are discussed as higher in Asian OWD and linked to insulin signaling, beta-cell adaptation, and systemic energy homeostasis.",
                    "miR-484 is discussed as higher in Asian NWD and linked to mitochondrial dynamics through FIS1.",
                    "The paper frames these pathway links as literature-grounded context, not direct pathway measurements in patient tissue.",
                ],
            ),
            (
                "Cautions",
                [
                    "The paper warns that plasma EV preparations may include lipoproteins and other non-EV nanoparticles with overlapping size and biochemical signatures.",
                    "Band assignments are described as tentative biochemical annotations in complex EV-enriched plasma isolates, not definitive attribution to EV membrane constituents alone.",
                    "Subgroup-linked patterns are cohort-level descriptors rather than patient-level predictions.",
                ],
            ),
        ],
    )
    _write_markdown(
        shine_md_path,
        "SPECTRA/SHINE Structured Notes",
        [
            (
                "Study Design",
                [
                    "The paper studies EVs from primary rat hepatocytes exposed to 0, 10, 20, and 40 mM APAP with Day 0 and Day 2 measurements.",
                    "The task is framed as dose-resolved hepatotoxicity and injury-response regression rather than broad disease classification.",
                    "The platform uses 1.3 uL dried EV sample on a gold nanopillar surface and reports analysis in under 30 minutes.",
                ],
            ),
            (
                "Spectral And Assay Notes",
                [
                    "The paper highlights 739, 960, 1250, 1525, 1576, and 1602 cm^-1 as strongly correlated with albumin and CCK8.",
                    "1525, 1576, and 1602 cm^-1 are described as mainly nucleic-acid and protein vibrational support; amide III is additional protein/nucleic-acid cargo support.",
                    "The main biological framing is EV injury-response and stress-associated cargo change rather than direct molecule identification.",
                ],
            ),
            (
                "Preprocessing Notes",
                [
                    "Supplementary methods describe polynomial wavenumber calibration, asymmetric least-squares baseline correction, Savitzky-Golay smoothing, normalization to the 642 cm^-1 silicon peak, and k-means filtering of uncharacteristic spectra.",
                    "Analyses focus on the 810-1610 cm^-1 region for PCA and regression.",
                ],
            ),
            (
                "Cautions",
                [
                    "The paper explicitly limits interpretation to a monoculture system, a single hepatotoxicant, and only two biological replicates.",
                    "It calls for validation across more compounds, co-culture or organoid systems, and more complex EV mixtures.",
                    "These results support injury/stress interpretation rather than broad human liver-disease truth.",
                ],
            ),
        ],
    )

    provenance_path = extraction_root / "context_source_manifest.json"
    provenance_path.write_text(
        json.dumps(
            {
                "diabetes_pdf": {"path": str(DIABETES_PDF), "n_pages": len(diabetes_pdf_pages)},
                "diabetes_supplement": {"path": str(DIABETES_SUPP)},
                "shine_pdf": {"path": str(SHINE_PDF), "n_pages": len(shine_pdf_pages)},
                "shine_supplement": {"path": str(SHINE_SUPP)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    support_documents_df, support_chunks_df = _build_support_document_rows()
    ev_documents_df, ev_chunks_df = _build_ev_context_rows()

    with duckdb.connect(str(db_path)) as connection:
        connection.execute(
            "DELETE FROM grounding_support_chunks WHERE dataset_id IN ('diabetes_ev_context_support', 'shine_spectra_context_support')"
        )
        connection.execute(
            "DELETE FROM grounding_support_documents WHERE dataset_id IN ('diabetes_ev_context_support', 'shine_spectra_context_support')"
        )
        connection.execute(
            "DELETE FROM domain_context_chunks WHERE document_id LIKE 'gaira_ev_context_diabetes_refined_%' OR document_id LIKE 'gaira_ev_context_shine_refined_%'"
        )
        connection.execute(
            "DELETE FROM domain_context_documents WHERE document_id LIKE 'gaira_ev_context_diabetes_refined_%' OR document_id LIKE 'gaira_ev_context_shine_refined_%'"
        )

        connection.register("support_documents_df", support_documents_df)
        connection.execute("INSERT INTO grounding_support_documents SELECT * FROM support_documents_df")
        connection.unregister("support_documents_df")

        connection.register("support_chunks_df", support_chunks_df)
        connection.execute("INSERT INTO grounding_support_chunks SELECT * FROM support_chunks_df")
        connection.unregister("support_chunks_df")

        connection.register("ev_documents_df", ev_documents_df)
        connection.execute("INSERT INTO domain_context_documents SELECT * FROM ev_documents_df")
        connection.unregister("ev_documents_df")

        connection.register("ev_chunks_df", ev_chunks_df)
        connection.execute("INSERT INTO domain_context_chunks SELECT * FROM ev_chunks_df")
        connection.unregister("ev_chunks_df")

    print("Biology context refinement ingest complete.")
    print(f"Wrote extraction notes to: {extraction_root}")
    print(f"Inserted support documents: {len(support_documents_df)}")
    print(f"Inserted support chunks: {len(support_chunks_df)}")
    print(f"Inserted EV context documents: {len(ev_documents_df)}")
    print(f"Inserted EV context chunks: {len(ev_chunks_df)}")


if __name__ == "__main__":
    main()
