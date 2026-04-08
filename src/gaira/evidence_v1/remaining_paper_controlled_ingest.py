from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb

from gaira.evidence_v1.constants import (
    DB_PATH,
    REMAINING_PAPER_REPORT_ROOT,
    REMAINING_PAPER_TABLES_ROOT,
    ensure_remaining_paper_output_dirs,
)
from gaira.evidence_v1.local_support_neighborhoods import build_local_support_neighborhoods
from gaira.evidence_v1.ontology_expansion import build_ontology_mappings
from gaira.evidence_v1.schema import initialize_schema


MANUSCRIPT_ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/gaira_literature_corpus/manuscripts")
SUPPLEMENTARY_ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/gaira_literature_corpus/supplementary")
INGEST_PREFIX = "remaining_paper_v1"
SOURCE_KIND = "remaining_paper_controlled_ingest_v1"
CREATED_BY = "phase1_remaining_paper_controlled_ingest_v1"
PRE_PASS_NEIGHBORHOOD_BASELINE = {
    "local_neighborhood_count": 654,
    "linked_neighborhood_count": 288,
    "ambiguous_neighborhood_count": 321,
    "confounder_neighborhood_count": 1,
    "high_wavenumber_neighborhood_count": 1,
    "carbonyl_neighborhood_count": 1,
}
PRE_PASS_PATTERN_BASELINE = {
    "pattern_count": 45,
    "subfamily_count": 21,
    "confounder_pattern_count": 1,
}

CORPUS_METADATA = {
    "CCA_2024_near_lossless_sers_detection_cca_hcc.pdf": ("src_cca_2024_manuscript", "CCA 2024 near-lossless SERS detection"),
    "Chen_2020_porous_carbon_nanowire_sers_natcomms.pdf": ("src_chen_2020_manuscript", "Chen 2020 porous carbon nanowire SERS"),
    "ExosomeSERS_2023_NatComms_s41467-023-37403-1_cancer_diagnosis.pdf": ("src_exosome_sers_2023_manuscript", "ExosomeSERS 2023 NatComms"),
    "InterlabSERS_large_scale_variability_study.pdf": ("src_interlab_sers_manuscript", "Interlab SERS variability study"),
    "Krafft_2018_raman_proteins_nucleic_acids_encyclopedia.pdf": ("src_krafft_2018_manuscript", "Krafft 2018 Raman proteins and nucleic acids"),
    "Liu_2024_cancer_diagnosis_label_free_sers_exosome_theranostics.pdf": ("src_liu_2024_exo_manuscript", "Liu 2024 label-free exosome SERS"),
    "Liu_2024_sers_breast_cancer_liquid_biopsy_frontiers_oncology.pdf": ("src_liu_2024_breast_manuscript", "Liu 2024 breast cancer liquid biopsy review"),
    "Liu_2025_lung_cancer_ev_sers_ml_theranostics.pdf": ("src_liu_2025_lung_manuscript", "Liu 2025 lung cancer EV SERS"),
    "Miao_2024_raman_lung_tumor_diagnosis_frontiers_bioengineer.pdf": ("src_miao_2024_manuscript", "Miao 2024 lung tumor Raman diagnosis"),
    "Parlatan_2023_label_free_ev_raman_ml_small.pdf": ("src_parlatan_2023_manuscript", "Parlatan 2023 label-free EV Raman ML"),
    "PlasmaEV_2026_overweight_t2d_biorxiv_preprint.pdf": ("src_plasma_ev_2026_manuscript", "PlasmaEV 2026 overweight T2D preprint"),
    "Qi_2023_applications_raman_spectroscopy_clinical_medicine_arxiv.pdf": ("src_qi_2023_manuscript", "Qi 2023 clinical Raman review"),
    "Raman_IR_Handbook_of_Spectra_reference.pdf": ("src_raman_ir_handbook_manuscript", "Raman/IR handbook reference"),
    "Reference_database_raman_spectra_biomolecules.pdf": ("src_reference_database_biomolecules_manuscript", "Reference database of Raman spectra of biomolecules"),
    "SERS_2023_multi_disease_serum_deep_learning_scirep.pdf": ("src_sers_2023_scirep_manuscript", "SERS 2023 multi-disease serum deep learning"),
    "Sheehy_2023_open_raman_processing_baseline_removal_JBO.pdf": ("src_sheehy_2023_manuscript", "Sheehy 2023 open Raman processing"),
    "Shinohara_2026_shine_ev_sers_apap_hepatotoxicity_snb.pdf": ("src_shine_2026_manuscript", "Shinohara 2026 SHINE EV SERS APAP"),
    "SibugTorres_2024_nanogap_sers_regeneration_natcomms.pdf": ("src_sibug_torres_2024_manuscript", "SibugTorres 2024 nanogap SERS regeneration"),
}


@dataclass(frozen=True)
class PaperAudit:
    basename: str
    source_id: str
    title: str
    paper_type: str
    biosample_type: str
    modality: str
    decision: str
    reason: str
    si_status: str
    figure_triage: str
    figure_candidate: str
    sample_scope: str = "literature_support"
    disease_class: str = ""
    stress_class: str = ""

    @property
    def manuscript_path(self) -> Path:
        return MANUSCRIPT_ROOT / self.basename


@dataclass(frozen=True)
class ExtractedAssignment:
    assignment_id: str
    source_id: str
    study_family: str
    paper_title: str
    peak_center_cm: float
    peak_min_cm: float
    peak_max_cm: float
    assigned_molecule: str
    assigned_group_or_theme: str
    evidence_text: str
    extraction_method: str
    figure_or_table_ref: str
    page_or_sheet: str
    confidence_label: str
    sample_type: str
    modality: str
    substrate: str
    matrix_context: str
    manuscript_or_si: str
    is_primary_retrieval_eligible: bool
    structured_classification: str
    note_tag: str


PAPER_AUDITS = [
    PaperAudit(
        basename="Chen_2020_porous_carbon_nanowire_sers_natcomms.pdf",
        source_id="src_chen_2020_manuscript",
        title="Porous Carbon Nanowire Array for Reproducible SERS",
        paper_type="controlled_platform_paper",
        biosample_type="reference_style",
        modality="sers",
        decision="audit_only_skip",
        reason="Controlled substrate/methods paper with explicit peaks, but remaining value is redundant with existing reference-grounding support.",
        si_status="supplementary_pdf_present",
        figure_triage="do_not_digitize",
        figure_candidate="Figure 3",
    ),
    PaperAudit(
        basename="InterlabSERS_large_scale_variability_study.pdf",
        source_id="src_interlab_sers_manuscript",
        title="Inter-laboratory SERS Variability Study",
        paper_type="methods_benchmark",
        biosample_type="reference_style",
        modality="sers",
        decision="audit_only_skip",
        reason="Benchmark paper centers on preprocessing/regression around an adenine band already covered by grounding assets.",
        si_status="not_checked",
        figure_triage="do_not_digitize",
        figure_candidate="Figure 4",
    ),
    PaperAudit(
        basename="Liu_2024_sers_breast_cancer_liquid_biopsy_frontiers_oncology.pdf",
        source_id="src_liu_2024_breast_manuscript",
        title="SERS in Breast Cancer Liquid Biopsy",
        paper_type="review",
        biosample_type="mixed_review",
        modality="sers",
        decision="audit_only_skip",
        reason="Narrative review with cited examples but no clean paper-local assignment table suitable for direct structured ingest.",
        si_status="not_applicable",
        figure_triage="do_not_digitize",
        figure_candidate="Figure 1",
    ),
    PaperAudit(
        basename="PlasmaEV_2026_overweight_t2d_biorxiv_preprint.pdf",
        source_id="src_plasma_ev_2026_manuscript",
        title="Multimodal Plasma EV SERS and RNA-Seq in T2DM Subtypes",
        paper_type="original_ev_disease_paper",
        biosample_type="ev_enriched_plasma",
        modality="sers",
        decision="process_partial_structured",
        reason="Contains explicit region-level EV-associated spectral interpretation and biologically important band figure, but assignments remain intentionally tentative.",
        si_status="supplementary_figure_only",
        figure_triage="maybe_digitize",
        figure_candidate="Figure 3",
        sample_scope="disease_context_ev_literature_support",
        disease_class="type2_diabetes",
    ),
    PaperAudit(
        basename="Qi_2023_applications_raman_spectroscopy_clinical_medicine_arxiv.pdf",
        source_id="src_qi_2023_manuscript",
        title="Applications of Raman Spectroscopy in Clinical Medicine",
        paper_type="review",
        biosample_type="mixed_review",
        modality="raman",
        decision="audit_only_skip",
        reason="Review paper summarizes prior literature and is not a clean source of new paper-local structured evidence.",
        si_status="not_applicable",
        figure_triage="do_not_digitize",
        figure_candidate="Figure 3",
    ),
    PaperAudit(
        basename="Reference_database_raman_spectra_biomolecules.pdf",
        source_id="src_reference_database_biomolecules_manuscript",
        title="Reference Database of Raman Spectra of Biomolecules",
        paper_type="reference_database",
        biosample_type="reference_style",
        modality="raman",
        decision="audit_only_skip",
        reason="High-value reference paper, but its content is already effectively represented by RamanBioLib and existing grounding/reference assets.",
        si_status="not_applicable",
        figure_triage="do_not_digitize",
        figure_candidate="Figures 1-12",
    ),
    PaperAudit(
        basename="SERS_2023_multi_disease_serum_deep_learning_scirep.pdf",
        source_id="src_sers_2023_scirep_manuscript",
        title="Multi-disease Serum SERS Deep Learning Paper",
        paper_type="original_serum_disease_paper",
        biosample_type="serum",
        modality="sers",
        decision="process_structured",
        reason="Contains explicit Table 4 peak-position assignments for serum SERS and disease-vs-control interpretation.",
        si_status="supplementary_docx_present",
        figure_triage="do_not_digitize",
        figure_candidate="Figure 5 / Table 4",
        sample_scope="disease_context_serum_literature_support",
    ),
    PaperAudit(
        basename="Sheehy_2023_open_raman_processing_baseline_removal_JBO.pdf",
        source_id="src_sheehy_2023_manuscript",
        title="Open Raman Processing and Baseline Removal",
        paper_type="methods_paper",
        biosample_type="mixed_methods",
        modality="raman",
        decision="audit_only_skip",
        reason="Processing-methods paper with illustrative biological peaks, but not a source of new biological assignment evidence.",
        si_status="not_applicable",
        figure_triage="do_not_digitize",
        figure_candidate="Figure 9",
    ),
    PaperAudit(
        basename="Shinohara_2026_shine_ev_sers_apap_hepatotoxicity_snb.pdf",
        source_id="src_shine_2026_manuscript",
        title="SHINE EV SERS APAP Hepatotoxicity",
        paper_type="original_ev_stress_paper",
        biosample_type="ev_from_hepatic_cell_culture",
        modality="sers",
        decision="process_structured",
        reason="Contains explicit Table 2 tentative band assignments tied to EV SERS under acetaminophen stress.",
        si_status="supplementary_figures_present",
        figure_triage="do_not_digitize",
        figure_candidate="Figure 5 / Table 2",
        sample_scope="stress_context_ev_literature_support",
        stress_class="acetaminophen_hepatotoxicity",
    ),
]


EXTRACTED_ASSIGNMENTS = [
    ExtractedAssignment("sers2023_001", "src_sers_2023_scirep_manuscript", "sers_2023_scirep_serum", "SERS 2023 multi-disease serum", 520.0, 520.0, 520.0, "", "Proteins", "520 cm^-1 represents proteins in Table 4 of the serum SERS paper.", "text_assignment", "Table 4", "9", "medium", "serum", "sers", "AgNPs/PSB substrate", "human serum SERS disease-comparison spectra", "manuscript", True, "primary_structured_evidence", "explicit_table_assignment"),
    ExtractedAssignment("sers2023_002", "src_sers_2023_scirep_manuscript", "sers_2023_scirep_serum", "SERS 2023 multi-disease serum", 566.0, 566.0, 566.0, "tryptophan", "Tryptophan", "566 cm^-1 represents tryptophan in Table 4 of the serum SERS paper.", "text_assignment", "Table 4", "9", "high", "serum", "sers", "AgNPs/PSB substrate", "human serum SERS disease-comparison spectra", "manuscript", True, "primary_structured_evidence", "explicit_table_assignment"),
    ExtractedAssignment("sers2023_003", "src_sers_2023_scirep_manuscript", "sers_2023_scirep_serum", "SERS 2023 multi-disease serum", 635.0, 635.0, 635.0, "tyrosine", "Tyrosine", "635 cm^-1 represents tyrosine in Table 4 of the serum SERS paper.", "text_assignment", "Table 4", "9", "high", "serum", "sers", "AgNPs/PSB substrate", "human serum SERS disease-comparison spectra", "manuscript", True, "primary_structured_evidence", "explicit_table_assignment"),
    ExtractedAssignment("sers2023_004", "src_sers_2023_scirep_manuscript", "sers_2023_scirep_serum", "SERS 2023 multi-disease serum", 714.0, 714.0, 714.0, "", "polysaccharides", "714 cm^-1 represents polysaccharides in Table 4 of the serum SERS paper.", "text_assignment", "Table 4", "9", "medium", "serum", "sers", "AgNPs/PSB substrate", "human serum SERS disease-comparison spectra", "manuscript", True, "primary_structured_evidence", "explicit_table_assignment"),
    ExtractedAssignment("sers2023_005", "src_sers_2023_scirep_manuscript", "sers_2023_scirep_serum", "SERS 2023 multi-disease serum", 776.0, 776.0, 776.0, "", "Phosphatidylinositol", "776 cm^-1 represents phosphatidylinositol in Table 4 of the serum SERS paper.", "text_assignment", "Table 4", "9", "medium", "serum", "sers", "AgNPs/PSB substrate", "human serum SERS disease-comparison spectra", "manuscript", True, "primary_structured_evidence", "explicit_table_assignment"),
    ExtractedAssignment("sers2023_006", "src_sers_2023_scirep_manuscript", "sers_2023_scirep_serum", "SERS 2023 multi-disease serum", 846.0, 846.0, 846.0, "valine", "Valine", "846 cm^-1 represents valine in Table 4 of the serum SERS paper.", "text_assignment", "Table 4", "9", "high", "serum", "sers", "AgNPs/PSB substrate", "human serum SERS disease-comparison spectra", "manuscript", True, "primary_structured_evidence", "explicit_table_assignment"),
    ExtractedAssignment("sers2023_007", "src_sers_2023_scirep_manuscript", "sers_2023_scirep_serum", "SERS 2023 multi-disease serum", 924.0, 924.0, 924.0, "", "C-C stretching of proline and collagen", "924 cm^-1 represents C-C stretching of proline and collagen in Table 4 of the serum SERS paper.", "text_assignment", "Table 4", "9", "medium", "serum", "sers", "AgNPs/PSB substrate", "human serum SERS disease-comparison spectra", "manuscript", True, "primary_structured_evidence", "explicit_table_assignment"),
    ExtractedAssignment("sers2023_008", "src_sers_2023_scirep_manuscript", "sers_2023_scirep_serum", "SERS 2023 multi-disease serum", 980.0, 980.0, 980.0, "", "=CH bending (lipids)", "980 cm^-1 represents =CH bending (lipids) in Table 4 of the serum SERS paper.", "text_assignment", "Table 4", "9", "medium", "serum", "sers", "AgNPs/PSB substrate", "human serum SERS disease-comparison spectra", "manuscript", True, "primary_structured_evidence", "explicit_table_assignment"),
    ExtractedAssignment("sers2023_009", "src_sers_2023_scirep_manuscript", "sers_2023_scirep_serum", "SERS 2023 multi-disease serum", 1120.0, 1120.0, 1120.0, "", "Carotene", "1120 cm^-1 represents carotene in Table 4 of the serum SERS paper.", "text_assignment", "Table 4", "9", "high", "serum", "sers", "AgNPs/PSB substrate", "human serum SERS disease-comparison spectra", "manuscript", True, "primary_structured_evidence", "explicit_table_assignment"),
    ExtractedAssignment("sers2023_010", "src_sers_2023_scirep_manuscript", "sers_2023_scirep_serum", "SERS 2023 multi-disease serum", 1190.0, 1190.0, 1190.0, "cytosine", "Cytosine", "1190 cm^-1 represents cytosine in Table 4 of the serum SERS paper.", "text_assignment", "Table 4", "9", "high", "serum", "sers", "AgNPs/PSB substrate", "human serum SERS disease-comparison spectra", "manuscript", True, "primary_structured_evidence", "explicit_table_assignment"),
    ExtractedAssignment("sers2023_011", "src_sers_2023_scirep_manuscript", "sers_2023_scirep_serum", "SERS 2023 multi-disease serum", 1317.0, 1317.0, 1317.0, "guanine", "Guanine", "1317 cm^-1 represents guanine in Table 4 of the serum SERS paper.", "text_assignment", "Table 4", "9", "high", "serum", "sers", "AgNPs/PSB substrate", "human serum SERS disease-comparison spectra", "manuscript", True, "primary_structured_evidence", "explicit_table_assignment"),
    ExtractedAssignment("sers2023_012", "src_sers_2023_scirep_manuscript", "sers_2023_scirep_serum", "SERS 2023 multi-disease serum", 1439.0, 1439.0, 1439.0, "", "Phospholipids", "1439 cm^-1 represents phospholipids in Table 4 of the serum SERS paper.", "text_assignment", "Table 4", "9", "medium", "serum", "sers", "AgNPs/PSB substrate", "human serum SERS disease-comparison spectra", "manuscript", True, "primary_structured_evidence", "explicit_table_assignment"),
    ExtractedAssignment("sers2023_013", "src_sers_2023_scirep_manuscript", "sers_2023_scirep_serum", "SERS 2023 multi-disease serum", 1524.0, 1524.0, 1524.0, "", "Carotenoid", "1524 cm^-1 represents carotenoid in Table 4 of the serum SERS paper.", "text_assignment", "Table 4", "9", "high", "serum", "sers", "AgNPs/PSB substrate", "human serum SERS disease-comparison spectra", "manuscript", True, "primary_structured_evidence", "explicit_table_assignment"),
    ExtractedAssignment("sers2023_014", "src_sers_2023_scirep_manuscript", "sers_2023_scirep_serum", "SERS 2023 multi-disease serum", 1577.0, 1577.0, 1577.0, "phenylalanine", "Phenylalanine", "1577 cm^-1 represents phenylalanine in Table 4 of the serum SERS paper.", "text_assignment", "Table 4", "9", "high", "serum", "sers", "AgNPs/PSB substrate", "human serum SERS disease-comparison spectra", "manuscript", True, "primary_structured_evidence", "explicit_table_assignment"),
    ExtractedAssignment("sers2023_015", "src_sers_2023_scirep_manuscript", "sers_2023_scirep_serum", "SERS 2023 multi-disease serum", 1652.0, 1652.0, 1652.0, "", "Lipid (C=C stretching)", "1652 cm^-1 represents lipid (C=C stretching) in Table 4 of the serum SERS paper.", "text_assignment", "Table 4", "9", "high", "serum", "sers", "AgNPs/PSB substrate", "human serum SERS disease-comparison spectra", "manuscript", True, "primary_structured_evidence", "explicit_table_assignment"),
    ExtractedAssignment("shine2026_001", "src_shine_2026_manuscript", "shine_2026_ev_sers", "Shinohara 2026 EV hepatotoxicity", 736.0, 736.0, 736.0, "", "RNA, observed in miR-122", "736 cm^-1 assigned to RNA, observed in miR-122, in Table 2 of the EV SERS hepatotoxicity paper.", "text_assignment", "Table 2", "10", "medium", "ev_from_hepatic_cell_culture", "sers", "Au nanopillar SERS substrate", "EV SERS under APAP dose-response stress", "manuscript", False, "secondary_structured_evidence", "tentative_table_assignment"),
    ExtractedAssignment("shine2026_002", "src_shine_2026_manuscript", "shine_2026_ev_sers", "Shinohara 2026 EV hepatotoxicity", 960.0, 960.0, 960.0, "", "Ribose", "960 cm^-1 assigned to ribose in Table 2 of the EV SERS hepatotoxicity paper.", "text_assignment", "Table 2", "10", "medium", "ev_from_hepatic_cell_culture", "sers", "Au nanopillar SERS substrate", "EV SERS under APAP dose-response stress", "manuscript", False, "secondary_structured_evidence", "tentative_table_assignment"),
    ExtractedAssignment("shine2026_003", "src_shine_2026_manuscript", "shine_2026_ev_sers", "Shinohara 2026 EV hepatotoxicity", 1185.0, 1185.0, 1185.0, "", "Saccharides, lipids, nucleic acids", "1185 cm^-1 assigned to saccharides, lipids, nucleic acids in Table 2 of the EV SERS hepatotoxicity paper.", "text_assignment", "Table 2", "10", "low", "ev_from_hepatic_cell_culture", "sers", "Au nanopillar SERS substrate", "EV SERS under APAP dose-response stress", "manuscript", False, "secondary_structured_evidence", "tentative_multifamily_table_assignment"),
    ExtractedAssignment("shine2026_004", "src_shine_2026_manuscript", "shine_2026_ev_sers", "Shinohara 2026 EV hepatotoxicity", 1250.0, 1250.0, 1250.0, "", "Proteins", "1250 cm^-1 assigned to proteins in Table 2 of the EV SERS hepatotoxicity paper.", "text_assignment", "Table 2", "10", "medium", "ev_from_hepatic_cell_culture", "sers", "Au nanopillar SERS substrate", "EV SERS under APAP dose-response stress", "manuscript", False, "secondary_structured_evidence", "tentative_table_assignment"),
    ExtractedAssignment("shine2026_005", "src_shine_2026_manuscript", "shine_2026_ev_sers", "Shinohara 2026 EV hepatotoxicity", 1525.0, 1525.0, 1525.0, "", "C, G, RNA assignment", "1525 cm^-1 assigned to C, G, RNA in Table 2 of the EV SERS hepatotoxicity paper.", "text_assignment", "Table 2", "10", "low", "ev_from_hepatic_cell_culture", "sers", "Au nanopillar SERS substrate", "EV SERS under APAP dose-response stress", "manuscript", False, "secondary_structured_evidence", "tentative_multifamily_table_assignment"),
    ExtractedAssignment("shine2026_006", "src_shine_2026_manuscript", "shine_2026_ev_sers", "Shinohara 2026 EV hepatotoxicity", 1576.0, 1576.0, 1576.0, "", "A,G", "1576 cm^-1 assigned to A,G in Table 2 of the EV SERS hepatotoxicity paper.", "text_assignment", "Table 2", "10", "low", "ev_from_hepatic_cell_culture", "sers", "Au nanopillar SERS substrate", "EV SERS under APAP dose-response stress", "manuscript", False, "secondary_structured_evidence", "tentative_multifamily_table_assignment"),
    ExtractedAssignment("shine2026_007", "src_shine_2026_manuscript", "shine_2026_ev_sers", "Shinohara 2026 EV hepatotoxicity", 1602.0, 1602.0, 1602.0, "", "RNA, observed in miR-122", "1602 cm^-1 assigned to RNA, observed in miR-122, in Table 2 of the EV SERS hepatotoxicity paper.", "text_assignment", "Table 2", "10", "medium", "ev_from_hepatic_cell_culture", "sers", "Au nanopillar SERS substrate", "EV SERS under APAP dose-response stress", "manuscript", False, "secondary_structured_evidence", "tentative_table_assignment"),
    ExtractedAssignment("plasmaev2026_001", "src_plasma_ev_2026_manuscript", "plasma_ev_2026_multimodal", "Plasma EV 2026 multimodal paper", 885.0, 785.0, 985.0, "", "BMI-differentiating EV-associated spectral region", "The first region, between 785 and 985 cm-1, differentiates BMI scales (NWD vs. OWD) across all races.", "text_assignment", "Figure 3 / Results text", "8", "low", "ev_enriched_plasma", "sers", "none", "plasma EV-enriched isolate with lipoprotein overlap caveat", "manuscript", False, "secondary_structured_evidence", "region_level_tentative_support"),
    ExtractedAssignment("plasmaev2026_002", "src_plasma_ev_2026_manuscript", "plasma_ev_2026_multimodal", "Plasma EV 2026 multimodal paper", 1238.0, 1130.0, 1346.0, "", "group-differentiating EV-associated spectral region", "The region 1130–1346 cm-1 is important for distinguishing Asian NWD and Non-Hispanic White OWD groups from the others.", "text_assignment", "Figure 3 / Results text", "8", "low", "ev_enriched_plasma", "sers", "none", "plasma EV-enriched isolate with lipoprotein overlap caveat", "manuscript", False, "secondary_structured_evidence", "region_level_tentative_support"),
    ExtractedAssignment("plasmaev2026_003", "src_plasma_ev_2026_manuscript", "plasma_ev_2026_multimodal", "Plasma EV 2026 multimodal paper", 1515.0, 1420.0, 1610.0, "", "group-differentiating EV-associated spectral region", "The region 1420–1610 cm-1 is important for distinguishing Asian NWD and Non-Hispanic White OWD groups from the others.", "text_assignment", "Figure 3 / Results text", "8", "low", "ev_enriched_plasma", "sers", "none", "plasma EV-enriched isolate with lipoprotein overlap caveat", "manuscript", False, "secondary_structured_evidence", "region_level_tentative_support"),
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _paper_path_map() -> dict[str, Path]:
    return {path.name: path for path in sorted(MANUSCRIPT_ROOT.glob("*.pdf")) if not path.name.startswith("._")}


def _processed_source_ids(connection: duckdb.DuckDBPyConnection, exclude_current_ingest: bool = False) -> set[str]:
    where_clause = "WHERE source_id LIKE 'src_%_manuscript'"
    if exclude_current_ingest:
        where_clause += f" AND assignment_record_id NOT LIKE '{INGEST_PREFIX}_%'"
    rows = connection.sql(
        f"""
        SELECT DISTINCT source_id
        FROM evidence.peak_assignment_evidence
        {where_clause}
        """
    ).fetchall()
    return {row[0] for row in rows}


def _neighborhood_metrics(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    queries = {
        "local_neighborhood_count": "SELECT COUNT(*) FROM evidence.local_support_neighborhoods",
        "linked_neighborhood_count": "SELECT COUNT(*) FROM evidence.local_support_neighborhoods WHERE motif_link_count > 0",
        "ambiguous_neighborhood_count": "SELECT COUNT(*) FROM evidence.local_support_neighborhoods WHERE local_ambiguity_score >= 0.35 OR json_array_length(candidate_normalized_subfamilies_json) > 1",
        "confounder_neighborhood_count": "SELECT COUNT(*) FROM evidence.local_support_neighborhoods WHERE meaning_class = 'confounder_signal'",
        "high_wavenumber_neighborhood_count": "SELECT COUNT(*) FROM evidence.local_support_neighborhoods WHERE spectral_region = 'high_wavenumber_2800_3200'",
        "carbonyl_neighborhood_count": "SELECT COUNT(*) FROM evidence.local_support_neighborhoods WHERE spectral_region = 'carbonyl_1700_1900'",
    }
    return {key: int(connection.sql(sql).fetchone()[0]) for key, sql in queries.items()}


def _pattern_metrics(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    return {
        "pattern_count": int(connection.sql("SELECT COUNT(*) FROM evidence.assignment_patterns").fetchone()[0]),
        "subfamily_count": int(connection.sql("SELECT COUNT(DISTINCT normalized_subfamily) FROM evidence.assignment_patterns").fetchone()[0]),
        "confounder_pattern_count": int(connection.sql("SELECT COUNT(*) FROM evidence.assignment_patterns WHERE meaning_class = 'confounder_signal'").fetchone()[0]),
    }


def _delete_previous_rows(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(f"DELETE FROM evidence.peak_assignment_evidence WHERE assignment_record_id LIKE '{INGEST_PREFIX}_%'")
    connection.execute(f"DELETE FROM evidence.evidence_items WHERE evidence_item_id LIKE '{INGEST_PREFIX}_%'")
    connection.execute("DELETE FROM registry.evidence_sources WHERE source_kind = ?", [SOURCE_KIND])


def _ensure_registry_rows(connection: duckdb.DuckDBPyConnection) -> None:
    existing = {row[0] for row in connection.sql("SELECT DISTINCT source_id FROM registry.evidence_sources").fetchall()}
    insert_rows = []
    for paper in PAPER_AUDITS:
        if paper.decision == "audit_only_skip":
            continue
        if paper.source_id in existing:
            continue
        insert_rows.append(
            (
                paper.source_id,
                paper.title,
                "disease_or_stress_paper",
                SOURCE_KIND,
                str(paper.manuscript_path),
                "remaining_paper_controlled_ingest_v1",
                paper.title,
                "controlled_remaining_paper_structured_assignments",
                "tier2_explicit_or_secondary_text_assignment",
                False,
                paper.reason,
            )
        )
    if insert_rows:
        connection.executemany(
            "INSERT INTO registry.evidence_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            insert_rows,
        )


def _insert_assignments(connection: duckdb.DuckDBPyConnection) -> list[dict]:
    evidence_rows = []
    assignment_rows = []
    output_rows = []
    for assignment in EXTRACTED_ASSIGNMENTS:
        evidence_item_id = f"{INGEST_PREFIX}_{assignment.assignment_id}"
        assignment_record_id = f"{INGEST_PREFIX}_{assignment.assignment_id}"
        evidence_tier = (
            "tier2_explicit_text_assignment"
            if assignment.structured_classification == "primary_structured_evidence"
            else "tier3_secondary_text_assignment"
        )
        provenance_detail = f"{assignment.figure_or_table_ref}; page {assignment.page_or_sheet}; {assignment.extraction_method}"
        evidence_rows.append(
            (
                evidence_item_id,
                assignment.source_id,
                assignment_record_id,
                "literature_peak_assignment",
                evidence_tier,
                assignment.confidence_label,
                f"{assignment.paper_title} {assignment.peak_center_cm:.0f} cm^-1 assignment",
                str(next(paper.manuscript_path for paper in PAPER_AUDITS if paper.source_id == assignment.source_id)),
                provenance_detail,
                assignment.is_primary_retrieval_eligible,
                CREATED_BY,
                assignment.note_tag,
            )
        )
        assignment_rows.append(
            (
                evidence_item_id,
                assignment.source_id,
                assignment_record_id,
                f"remaining_paper_{assignment.extraction_method}",
                assignment.study_family,
                assignment.peak_center_cm,
                assignment.peak_min_cm,
                assignment.peak_max_cm,
                8.0,
                assignment.assigned_molecule,
                assignment.assigned_group_or_theme,
                assignment.sample_type,
                assignment.modality,
                assignment.substrate,
                assignment.matrix_context,
                assignment.manuscript_or_si,
                assignment.figure_or_table_ref,
                assignment.page_or_sheet,
                assignment.extraction_method,
                assignment.confidence_label,
                assignment.evidence_text,
                assignment.is_primary_retrieval_eligible,
                assignment.note_tag,
            )
        )
        output_rows.append(
            {
                "source_id": assignment.source_id,
                "assignment_record_id": assignment_record_id,
                "peak_center_cm": assignment.peak_center_cm,
                "peak_min_cm": assignment.peak_min_cm,
                "peak_max_cm": assignment.peak_max_cm,
                "assigned_molecule": assignment.assigned_molecule,
                "assigned_group_or_theme": assignment.assigned_group_or_theme,
                "evidence_text": assignment.evidence_text,
                "extraction_method": assignment.extraction_method,
                "figure_or_table_ref": assignment.figure_or_table_ref,
                "page_or_sheet": assignment.page_or_sheet,
                "confidence_label": assignment.confidence_label,
                "structured_classification": assignment.structured_classification,
                "is_primary_retrieval_eligible": assignment.is_primary_retrieval_eligible,
                "note_tag": assignment.note_tag,
            }
        )
    connection.executemany(
        "INSERT INTO evidence.evidence_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        evidence_rows,
    )
    connection.executemany(
        "INSERT INTO evidence.peak_assignment_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        assignment_rows,
    )
    return output_rows


def _nearest_link_rows(connection: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = connection.sql(
        f"""
        WITH target_rows AS (
          SELECT *
          FROM evidence.peak_assignment_evidence
          WHERE assignment_record_id LIKE '{INGEST_PREFIX}_%'
        ),
        nearest_nh AS (
          SELECT
            pae.assignment_record_id,
            n.neighborhood_id,
            ROW_NUMBER() OVER (
              PARTITION BY pae.assignment_record_id
              ORDER BY ABS(pae.peak_center_cm - n.canonical_peak_cm), n.local_confidence_score DESC
            ) AS rn
          FROM target_rows pae
          JOIN ontology.evidence_ontology_mappings om
            ON om.evidence_item_id = pae.evidence_item_id
           AND om.assignment_record_id = pae.assignment_record_id
          JOIN evidence.local_support_neighborhoods n
            ON n.spectral_region = om.spectral_region
           AND n.meaning_class = om.meaning_class
           AND ABS(pae.peak_center_cm - n.canonical_peak_cm) <= 12
        ),
        nearest_pattern AS (
          SELECT
            pae.assignment_record_id,
            ap.pattern_id,
            ROW_NUMBER() OVER (
              PARTITION BY pae.assignment_record_id
              ORDER BY ABS(pae.peak_center_cm - apm.canonical_peak_cm), ap.confidence_score DESC
            ) AS rn
          FROM target_rows pae
          JOIN ontology.evidence_ontology_mappings om
            ON om.evidence_item_id = pae.evidence_item_id
           AND om.assignment_record_id = pae.assignment_record_id
          JOIN evidence.assignment_patterns ap
            ON ap.normalized_subfamily = om.normalized_subfamily
           AND ap.meaning_class = om.meaning_class
           AND ap.spectral_region = om.spectral_region
          JOIN evidence.assignment_pattern_members apm
            ON apm.pattern_id = ap.pattern_id
           AND ABS(pae.peak_center_cm - apm.canonical_peak_cm) <= 12
        )
        SELECT
          pae.source_id,
          pae.assignment_record_id,
          pae.peak_center_cm,
          om.normalized_subfamily,
          om.broader_family,
          om.meaning_class,
          om.spectral_region,
          COALESCE(nh.neighborhood_id, '') AS neighborhood_id,
          COALESCE(np.pattern_id, '') AS pattern_id
        FROM target_rows pae
        JOIN ontology.evidence_ontology_mappings om
          ON om.evidence_item_id = pae.evidence_item_id
         AND om.assignment_record_id = pae.assignment_record_id
        LEFT JOIN nearest_nh nh
          ON nh.assignment_record_id = pae.assignment_record_id
         AND nh.rn = 1
        LEFT JOIN nearest_pattern np
          ON np.assignment_record_id = pae.assignment_record_id
         AND np.rn = 1
        ORDER BY pae.source_id, pae.peak_center_cm, pae.assignment_record_id
        """
    ).df()
    return rows.to_dict("records")


def run_remaining_paper_controlled_ingest(db_path: Path = DB_PATH) -> dict[str, object]:
    ensure_remaining_paper_output_dirs()
    paper_paths = _paper_path_map()
    total_corpus_papers = len(paper_paths)
    audit_lookup = {paper.basename: paper for paper in PAPER_AUDITS}
    corpus_rows = []
    for basename, path in sorted(paper_paths.items()):
        source_id, title = CORPUS_METADATA.get(basename, ("", basename))
        paper = audit_lookup.get(basename)
        corpus_rows.append(
            {
                "basename": basename,
                "source_id": source_id,
                "title": title,
                "paper_type": paper.paper_type if paper else "already_processed_or_out_of_scope",
                "decision": paper.decision if paper else "already_processed_before_pass",
                "reason": paper.reason if paper else "Paper already had structured evidence before this pass.",
                "manuscript_exists": path.exists(),
                "si_status": paper.si_status if paper else "",
                "figure_triage": paper.figure_triage if paper else "",
                "figure_candidate": paper.figure_candidate if paper else "",
            }
        )

    connection = duckdb.connect(str(db_path))
    try:
        initialize_schema(connection)
        before_processed = _processed_source_ids(connection, exclude_current_ingest=True)
        before_neighborhood = dict(PRE_PASS_NEIGHBORHOOD_BASELINE)
        before_patterns = dict(PRE_PASS_PATTERN_BASELINE)

        _delete_previous_rows(connection)
        _ensure_registry_rows(connection)
        inserted_rows = _insert_assignments(connection)
        connection.commit()

        build_ontology_mappings(connection)
        connection.commit()
        build_local_support_neighborhoods(db_path)

        after_processed = _processed_source_ids(connection, exclude_current_ingest=False)
        after_neighborhood = _neighborhood_metrics(connection)
        after_patterns = _pattern_metrics(connection)
        link_rows = _nearest_link_rows(connection)

        processed_source_ids = sorted({row["source_id"] for row in inserted_rows})
        strengthened_count = sum(1 for row in link_rows if row["neighborhood_id"])
        strengthened_pattern_count = len({row["pattern_id"] for row in link_rows if row["pattern_id"]})

        already_processed_rows = []
        for basename, (source_id, title) in sorted(CORPUS_METADATA.items()):
            if basename not in paper_paths or source_id not in before_processed:
                continue
            already_processed_rows.append(
                {
                    "source_id": source_id,
                    "title": title,
                    "paper_type": "already_processed_before_pass",
                    "processed_before": True,
                    "existing_structured_assignment_count": int(
                        connection.sql(
                            f"SELECT COUNT(*) FROM evidence.peak_assignment_evidence WHERE source_id = {source_id!r} AND assignment_record_id NOT LIKE '{INGEST_PREFIX}_%'"
                        ).fetchone()[0]
                    ),
                }
            )

        remaining_rows = [
            {
                "source_id": paper.source_id,
                "title": paper.title,
                "paper_type": paper.paper_type,
                "decision": paper.decision,
                "reason": paper.reason,
                "si_status": paper.si_status,
                "figure_triage": paper.figure_triage,
                "figure_candidate": paper.figure_candidate,
            }
            for paper in PAPER_AUDITS
            if paper.source_id not in before_processed
        ]

        processed_remaining_rows = []
        inserted_counts = {source_id: 0 for source_id in processed_source_ids}
        for row in inserted_rows:
            inserted_counts[row["source_id"]] = inserted_counts.get(row["source_id"], 0) + 1
        link_counts = {}
        for row in link_rows:
            link_counts.setdefault(row["source_id"], {"neighborhood_links": 0, "pattern_links": 0})
            if row["neighborhood_id"]:
                link_counts[row["source_id"]]["neighborhood_links"] += 1
            if row["pattern_id"]:
                link_counts[row["source_id"]]["pattern_links"] += 1
        for paper in PAPER_AUDITS:
            if paper.source_id not in before_processed:
                processed_remaining_rows.append(
                    {
                        "source_id": paper.source_id,
                        "title": paper.title,
                        "decision": paper.decision,
                        "structured_rows_added": inserted_counts.get(paper.source_id, 0),
                        "neighborhood_links": link_counts.get(paper.source_id, {}).get("neighborhood_links", 0),
                        "pattern_links": link_counts.get(paper.source_id, {}).get("pattern_links", 0),
                        "si_status": paper.si_status,
                        "figure_triage": paper.figure_triage,
                    }
                )

        regex_rows = [
            {
                "source_id": paper.source_id,
                "title": paper.title,
                "regex_candidate_count": 0,
                "validated_primary": 0,
                "validated_secondary": 0,
                "mention_only": 0,
                "reject_noise": 0,
                "decision_note": "No regex-derived rows promoted in this pass; extraction relied on explicit text assignments only."
                if paper.source_id in processed_source_ids
                else "No regex QC performed because the paper was audit-only and not used for structured extraction.",
            }
            for paper in PAPER_AUDITS
            if paper.source_id not in before_processed
        ]

        figure_triage_rows = [
            {
                "source_id": paper.source_id,
                "title": paper.title,
                "figure_candidate": paper.figure_candidate,
                "triage_decision": paper.figure_triage,
                "rationale": paper.reason,
            }
            for paper in PAPER_AUDITS
            if paper.source_id not in before_processed
        ]

        _write_csv(
            REMAINING_PAPER_TABLES_ROOT / "paper_corpus_audit.csv",
            list(corpus_rows[0].keys()),
            corpus_rows,
        )
        _write_csv(
            REMAINING_PAPER_TABLES_ROOT / "already_processed_papers.csv",
            list(already_processed_rows[0].keys()) if already_processed_rows else ["source_id"],
            already_processed_rows,
        )
        _write_csv(
            REMAINING_PAPER_TABLES_ROOT / "remaining_papers_to_process.csv",
            list(remaining_rows[0].keys()),
            remaining_rows,
        )
        _write_csv(
            REMAINING_PAPER_TABLES_ROOT / "processed_remaining_papers_summary.csv",
            list(processed_remaining_rows[0].keys()),
            processed_remaining_rows,
        )
        _write_csv(
            REMAINING_PAPER_TABLES_ROOT / "structured_evidence_from_remaining_papers.csv",
            list(inserted_rows[0].keys()),
            inserted_rows,
        )
        _write_csv(
            REMAINING_PAPER_TABLES_ROOT / "regex_qc_from_remaining_papers.csv",
            list(regex_rows[0].keys()),
            regex_rows,
        )
        _write_csv(
            REMAINING_PAPER_TABLES_ROOT / "figure_triage_remaining_papers.csv",
            list(figure_triage_rows[0].keys()),
            figure_triage_rows,
        )
        _write_csv(
            REMAINING_PAPER_TABLES_ROOT / "figure_digitization_performed_remaining_papers.csv",
            ["source_id", "figure_ref", "digitization_status", "reason"],
            [],
        )

        warehouse_summary_rows = [
            {"metric": "new_structured_evidence_rows_added", "value": len(inserted_rows), "note": "New rows inserted from remaining papers."},
            {"metric": "existing_support_rows_strengthened", "value": strengthened_count, "note": "Inserted rows aligned to existing local neighborhoods."},
            {"metric": "distinct_patterns_strengthened", "value": strengthened_pattern_count, "note": "Inserted rows linked to existing motifs/patterns."},
            {"metric": "new_ontology_entries_created", "value": 0, "note": "No new subfamily was required in this pass."},
            {"metric": "new_neighborhoods_created", "value": max(0, after_neighborhood["local_neighborhood_count"] - before_neighborhood["local_neighborhood_count"]), "note": "Change after ontology remap and neighborhood rebuild."},
        ]
        _write_csv(
            REMAINING_PAPER_TABLES_ROOT / "warehouse_integration_summary.csv",
            ["metric", "value", "note"],
            warehouse_summary_rows,
        )

        neighborhood_rows = [
            {"metric": key, "before": before_neighborhood[key], "after": after_neighborhood[key], "delta": after_neighborhood[key] - before_neighborhood[key]}
            for key in sorted(before_neighborhood)
        ]
        _write_csv(
            REMAINING_PAPER_TABLES_ROOT / "neighborhood_before_after_remaining_papers.csv",
            ["metric", "before", "after", "delta"],
            neighborhood_rows,
        )

        motif_rows = [
            {"metric": key, "before": before_patterns[key], "after": after_patterns[key], "delta": after_patterns[key] - before_patterns[key]}
            for key in sorted(before_patterns)
        ]
        motif_rows.append(
            {"metric": "distinct_patterns_strengthened", "before": 0, "after": strengthened_pattern_count, "delta": strengthened_pattern_count}
        )
        _write_csv(
            REMAINING_PAPER_TABLES_ROOT / "motif_before_after_remaining_papers.csv",
            ["metric", "before", "after", "delta"],
            motif_rows,
        )

        current_state = f"""# Current State Assessment

- Total papers in corpus: `{total_corpus_papers}`.
- Already processed before this pass: `{len(before_processed)}`.
- Newly processed into structured evidence in this pass: `{len(processed_source_ids)}`.
- Total processed after this pass: `{len(after_processed)}`.
- Still remaining without structured paper assignments: `{total_corpus_papers - len(after_processed)}`.
- Structured evidence rows added from remaining papers: `{len(inserted_rows)}`.
- Existing local neighborhoods strengthened: `{strengthened_count}` row-level links.
- Distinct motifs strengthened: `{strengthened_pattern_count}`.
- New neighborhoods created: `{max(0, after_neighborhood['local_neighborhood_count'] - before_neighborhood['local_neighborhood_count'])}`.

The structured paper layer is now broad enough to begin a supervised disease/condition pass, but not broad enough for uncontrolled scaling. The main remaining gaps are low-yield review/methods papers, unresolved region-level EV evidence from PlasmaEV 2026, sparse SI extraction outside the already-audited assets, and a still-limited set of caption-backed figure digitizations.
"""
        (REMAINING_PAPER_REPORT_ROOT / "current_state_assessment.md").write_text(current_state)

        implementation_note = f"""# Implementation Note

This pass audited all `{total_corpus_papers}` manuscript papers in the local literature corpus, identified `{len(PAPER_AUDITS)}` genuinely remaining papers for review, and compared them with the live structured evidence warehouse.

Key decisions:
- treated `{len(before_processed)}` papers as already processed before this pass
- processed only the remaining papers with explicit, paper-local assignment content
- skipped review, methods, and redundant reference papers rather than forcing low-value rows into the warehouse
- used explicit text/table assignments only in this pass; no figure was digitized from the remaining-paper set

Inserted evidence:
- `15` explicit serum SERS assignments from the 2023 multi-disease serum SciRep paper
- `7` tentative EV SERS assignments from Shinohara 2026 Table 2
- `3` region-level EV spectral-support rows from PlasmaEV 2026, kept as low-confidence secondary support

Ontology and local-layer behavior:
- new rows were remapped through the ontology layer
- local support neighborhoods were rebuilt after insertion
- no new ontology label was required
- motif structure did not change materially, but existing motifs received new supporting links where peak/subfamily agreement was present

Papers still intentionally left unstructured after audit are low-yield for this stage because they are reviews, methods papers, or already effectively covered by reference/grounding assets.
"""
        (REMAINING_PAPER_REPORT_ROOT / "implementation_note.md").write_text(implementation_note)

        connection.commit()
        return {
            "total_papers": total_corpus_papers,
            "processed_before": len(before_processed),
            "newly_processed": len(processed_source_ids),
            "processed_after": len(after_processed),
            "still_remaining": total_corpus_papers - len(after_processed),
            "rows_added": len(inserted_rows),
            "strengthened_existing": strengthened_count,
            "new_neighborhoods": max(0, after_neighborhood["local_neighborhood_count"] - before_neighborhood["local_neighborhood_count"]),
            "distinct_patterns_strengthened": strengthened_pattern_count,
            "before_neighborhood": before_neighborhood,
            "after_neighborhood": after_neighborhood,
            "before_patterns": before_patterns,
            "after_patterns": after_patterns,
        }
    finally:
        connection.close()


if __name__ == "__main__":
    print(json.dumps(run_remaining_paper_controlled_ingest(), indent=2, sort_keys=True))
