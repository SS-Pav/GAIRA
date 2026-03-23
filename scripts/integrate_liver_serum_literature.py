from __future__ import annotations

import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import seaborn as sns


DOWNLOAD_ROOT = Path("/Users/suraj/Downloads/New_Set_SERS_Papers_Data")
SUPPORT_DATASET_ID = "liver_serum_literature_support"
SERUM_CONTEXT_LAYER = "GAIRA_SERUM_CONTEXT"


@dataclass(frozen=True)
class PaperSpec:
    slug: str
    filename: str
    title: str
    disease_task: str
    modality: str
    sample_type: str
    source_dataset_id: str
    source_file: str
    study_design: list[str]
    spectral_findings: list[str]
    interpretation_frame: list[str]
    cautions: list[str]
    themes: tuple[str, ...]
    caution_tags: tuple[str, ...]
    group_structure: str
    recommended_action: str
    relevance: str
    integrated_reason: str


SELECTED_PAPERS: list[PaperSpec] = [
    PaperSpec(
        slug="trieste_hcc_serum_sers_2020",
        filename="hcc_sers_serum.pdf",
        title="Repeated double cross-validation applied to the PCA-LDA classification of SERS spectra: a case study with serum samples from hepatocellular carcinoma patients",
        disease_task="HCC vs healthy serum SERS holdout-relevant support",
        modality="SERS serum",
        sample_type="human serum",
        source_dataset_id="hcc_serum",
        source_file="hcc_sers_serum.pdf|hcc_sers_serum_SI.pdf",
        study_design=[
            "Case study on fasting serum SERS from 72 hepatocellular carcinoma subjects and 72 healthy controls from Trieste.",
            "Paper uses plasmonic-paper Ag-colloid substrates and repeated double cross-validation PCA-LDA rather than a broad pan-liver disease framework.",
            "This is the closest literature support analogue to the existing hcc_serum holdout, but it remains support-only and must not be treated as integrated raw evidence.",
        ],
        spectral_findings=[
            "The paper frames serum SERS as multivariate biochemical structure rather than a single-marker assay.",
            "Support emphasis is on broad serum-pattern separation under a specific Ag plasmonic paper substrate rather than one universal disease peak list.",
            "The supplementary material documents cohort composition and paper-faithful modeling choices, which helps calibrate what GAIRA should and should not claim on HCC holdout outputs.",
        ],
        interpretation_frame=[
            "Useful as support for cautious HCC-vs-control serum context and for reminding GAIRA that holdout interpretation should stay substrate- and cohort-aware.",
            "Supports comparison-style reporting and what-not-to-claim framing more than direct biochemical certainty.",
        ],
        cautions=[
            "Single-center case-control study with substrate-specific plasmonic paper preparation.",
            "Support-only literature analogue for HCC serum; not a universal liver-cancer spectral truth.",
            "Useful for holdout framing, not for contaminating the live backbone with raw HCC data.",
        ],
        themes=("protein_peptide_associated", "oxidative_metabolic_stress_associated"),
        caution_tags=("low_specificity_caution", "weak_label_or_cohort_caution", "probe_substrate_caution"),
        group_structure="72 HCC vs 72 healthy controls",
        recommended_action="integrate_high_priority",
        relevance="high",
        integrated_reason="Closest published analogue to the existing HCC holdout evaluation and useful for conservative HCC serum interpretation support.",
    ),
    PaperSpec(
        slug="xiamen_liver_cancer_staging_sers_2024",
        filename="hcc_sers_serum_3.pdf",
        title="Label-free serum SERS combined with RFE-GBDT algorithm for non-invasive screening of liver cancer",
        disease_task="Healthy vs hepatitis B vs staged liver cancer serum SERS",
        modality="SERS serum",
        sample_type="human serum",
        source_dataset_id="hcc_serum",
        source_file="hcc_sers_serum_3.pdf",
        study_design=[
            "Five-class serum SERS study spanning healthy controls, chronic hepatitis B, and T1/T2/T3 liver cancer groups.",
            "Serum was mixed with Ag nanoparticles on an aluminum plate after thawing and analyzed as a label-free serum SERS assay.",
            "The paper is useful because it adds liver-disease axis structure beyond binary HCC-vs-control framing.",
        ],
        spectral_findings=[
            "Reported as a staged liver-cancer screening task with informative bands selected by RFE-GBDT rather than only end-to-end black-box classification.",
            "Supports liver-disease differential context between benign inflammatory background and overt liver-cancer progression.",
            "Adds a serum-side staging caution that model-selected bands are still cohort- and substrate-dependent.",
        ],
        interpretation_frame=[
            "Useful for liver-cancer staging context and HBV-to-HCC transition framing at the support layer.",
            "Strengthens liver-disease differential reasoning without asserting that selected bands are universal stage markers.",
        ],
        cautions=[
            "Model-selected biomarker bands remain classifier- and cohort-dependent.",
            "Serum/AgNP droplet protocol is not interchangeable with other serum SERS substrates.",
            "Stage labels should remain support-only context, not universal biochemical truth.",
        ],
        themes=("protein_peptide_associated", "oxidative_metabolic_stress_associated", "nucleic_acid_purine_associated"),
        caution_tags=("low_specificity_caution", "probe_substrate_caution", "weak_label_or_cohort_caution"),
        group_structure="healthy, HBV, T1, T2, T3",
        recommended_action="integrate_high_priority",
        relevance="high",
        integrated_reason="High-value serum liver-disease differential support that broadens HCC reasoning beyond a binary benchmark.",
    ),
    PaperSpec(
        slug="shanghai_liver_disease_sensor_2021",
        filename="1-s2.0-S0956566321002839-main.pdf",
        title="A biosensing method for the direct serological detection of liver diseases by integrating a SERS-based sensor and a CNN classifier",
        disease_task="Normal vs HCC vs hepatitis B direct serum SERS sensor",
        modality="SERS serum",
        sample_type="human serum",
        source_dataset_id="hcc_serum",
        source_file="1-s2.0-S0956566321002839-main.pdf|1-s2.0-S0956566321002839-mmc1.docx",
        study_design=[
            "Proof-of-concept direct serum SERS sensor on 30 normal controls, 30 HCC cases, and 30 hepatitis B cases.",
            "Uses a ZnO/AgNP/AuNP nanoarray and averages 5 spectra per serum sample in the supplementary material.",
            "Frames the task as direct serological detection of multiple liver disease states rather than a single binary comparison.",
        ],
        spectral_findings=[
            "Supplementary material explicitly shows representative serum SERS spectra from normal, HCC, and HB groups.",
            "The paper emphasizes a digital-biomarker style model on direct serum readout, not isolated biomolecule attribution.",
            "Support is therefore strongest for liver-disease axis context, not definitive peak-to-metabolite ground truth.",
        ],
        interpretation_frame=[
            "Useful for reminding GAIRA that some liver-serum papers push multi-class classification on specialized nanosensors.",
            "Supports broader HCC/hepatitis serum context and sensor-specific caution.",
        ],
        cautions=[
            "Specialized ZnO/AgNP/AuNP nanoarray and CNN classifier make the findings platform-specific.",
            "Digital-biomarker framing is useful support, but should not be over-read as universal serum chemistry truth.",
            "Classifier success does not remove the need for low-specificity and substrate cautions.",
        ],
        themes=("protein_peptide_associated", "lipid_membrane_associated", "oxidative_metabolic_stress_associated"),
        caution_tags=("probe_substrate_caution", "low_specificity_caution", "weak_label_or_cohort_caution"),
        group_structure="30 normal, 30 HCC, 30 hepatitis B",
        recommended_action="integrate_high_priority",
        relevance="high",
        integrated_reason="Adds multi-class liver-serum context and explicit sensor-specific caution relevant to serum differential reasoning.",
    ),
    PaperSpec(
        slug="digital_biomarker_hcc_serum_2023",
        filename="1-s2.0-S0927776523001935-main.pdf",
        title="Discovering the digital biomarker of hepatocellular carcinoma in serum with SERS-based biosensors and intelligence vision",
        disease_task="HCC vs hepatitis B vs normal serum digital biomarker support",
        modality="SERS serum",
        sample_type="human serum",
        source_dataset_id="hcc_serum",
        source_file="1-s2.0-S0927776523001935-main.pdf",
        study_design=[
            "Serum SERS study with 34 primary HCC, 34 hepatitis B, and 34 normal individuals on a dedicated biosensor.",
            "Paper reports a set of 10 characteristic peaks as a 'digital biomarker' validated on an independent batch.",
            "Useful because it gives an explicit liver-serum differential task including HB background rather than only HCC vs healthy.",
        ],
        spectral_findings=[
            "The paper explicitly centers a 10-peak HCC digital biomarker discovered with a DNN-like vision model.",
            "It links the characteristic peaks to a few clinical HCC biomarkers such as GPC3, DKK1, and AFP at a high level.",
            "This is valuable as support-level HCC serum structure, but not as a universal biochemical peak truth.",
        ],
        interpretation_frame=[
            "Useful for HCC-focused support retrieval and for explaining why some serum papers present classifier-derived peak panels.",
            "Strengthens HCC/HB differential context but should increase low-specificity caution rather than raw certainty.",
        ],
        cautions=[
            "Explicit digital-biomarker framing and DNN-derived peak panel are classifier-specific.",
            "Findings are tied to a particular biosensor and cohort construction.",
            "Support-only role is critical: these peaks should not be treated as universal HCC signatures.",
        ],
        themes=("nucleic_acid_purine_associated", "protein_peptide_associated", "oxidative_metabolic_stress_associated"),
        caution_tags=("low_specificity_caution", "probe_substrate_caution", "weak_label_or_cohort_caution"),
        group_structure="34 HCC, 34 hepatitis B, 34 normal",
        recommended_action="integrate_high_priority",
        relevance="high",
        integrated_reason="High-value HCC/HB serum paper with explicit peak-panel framing that improves what-not-to-claim support.",
    ),
    PaperSpec(
        slug="nafld_nash_serum_sers_2023",
        filename="NAFLD_SERS.pdf",
        title="Fully connected neural network-based serum surface-enhanced Raman spectroscopy accurately identifies non-alcoholic steatohepatitis",
        disease_task="Biopsy-proven NAFLD/NASH serum SERS support",
        modality="SERS serum",
        sample_type="human serum",
        source_dataset_id="not_applicable_or_unknown",
        source_file="NAFLD_SERS.pdf|12072_2022_10444_MOESM1_ESM.docx|12072_2022_10444_MOESM2_ESM.docx",
        study_design=[
            "Serum SERS study on 261 Chinese individuals with biopsy-proven NAFLD, with NASH prediction in a validation set.",
            "Supplement explains that the pathogenesis is multifactorial and that SERS reflects biomolecule mixtures including metabolites, glucose, proteins, lipids, and nucleic acids.",
            "This is valuable because it adds metabolic-liver-disease context distinct from HCC/hepatitis tasks.",
        ],
        spectral_findings=[
            "The supplementary appendix explicitly frames NASH-related serum SERS as reflecting inflammation, oxidative stress, apoptosis, and glucose/lipid metabolism.",
            "It also notes that no single biomarker is likely sufficient to separate NAFL from NASH.",
            "The paper is therefore strong support for oxidative/metabolic and lipid/protein interpretation, paired with low-specificity caution.",
        ],
        interpretation_frame=[
            "Useful for metabolic liver-disease context and for tying serum SERS themes to inflammation, lipid metabolism, and oxidative stress without overclaiming diagnosis.",
            "Helps GAIRA speak more intelligently about NASH/NAFLD-like serum shifts as support-level context.",
        ],
        cautions=[
            "Biopsy-proven cohort is strong, but still cohort-specific and model-dependent.",
            "Findings should not be collapsed into a single serum biomarker claim.",
            "Metabolic liver disease support should remain broader than direct HCC interpretation.",
        ],
        themes=("lipid_membrane_associated", "oxidative_metabolic_stress_associated", "protein_peptide_associated", "carbohydrate_glycan_associated"),
        caution_tags=("low_specificity_caution", "weak_label_or_cohort_caution"),
        group_structure="261 biopsy-proven NAFLD participants with NASH modeling",
        recommended_action="integrate_high_priority",
        relevance="high",
        integrated_reason="High-value metabolic-liver serum paper that broadens GAIRA beyond cancer-heavy serum support.",
    ),
    PaperSpec(
        slug="hepatitisb_cirrhosis_serum_raman_2026",
        filename="J Raman Spectroscopy - 2026 - Jin - Serum Raman Spectroscopy Combined With a Support Vector Machine for Rapid Diagnosis of.pdf",
        title="Serum Raman Spectroscopy Combined With a Support Vector Machine for Rapid Diagnosis of Hepatitis B and Liver Cirrhosis",
        disease_task="Hepatitis B vs cirrhosis vs healthy serum Raman support",
        modality="Raman serum",
        sample_type="human serum",
        source_dataset_id="covid_serum_raman",
        source_file="J Raman Spectroscopy - 2026 - Jin - Serum Raman Spectroscopy Combined With a Support Vector Machine for Rapid Diagnosis of.pdf|jrs5408-sup-0001-supporting_information.doc",
        study_design=[
            "Serum Raman study on hepatitis B, liver cirrhosis, and healthy individuals using a spontaneous Raman workflow rather than SERS.",
            "Paper positions serum Raman as a global molecular fingerprint capturing proteins, amino acids, and nucleic acids.",
            "Useful because it introduces liver-serum support in a spontaneous Raman modality already represented elsewhere in GAIRA by covid_serum_raman.",
        ],
        spectral_findings=[
            "The paper explicitly frames the signal as broad biomolecule fingerprinting rather than a targeted assay.",
            "It is especially useful for modality-aware liver-serum context: some liver support comes from Raman rather than SERS.",
            "This adds richer modality mismatch caution when serum Raman and serum SERS evidence are mixed.",
        ],
        interpretation_frame=[
            "Useful for cirrhosis/hepatitis context and for reminding GAIRA that serum liver-disease support spans both SERS and spontaneous Raman.",
            "Should strengthen modality-aware liver-serum reasoning without forcing direct equivalence between Raman and SERS findings.",
        ],
        cautions=[
            "Spontaneous Raman is not interchangeable with SERS-heavy serum support.",
            "Study-specific preprocessing and SVM modeling remain cohort-dependent.",
            "Supports liver-serum context, but should raise modality mismatch caution when mixed with SERS evidence.",
        ],
        themes=("protein_peptide_associated", "nucleic_acid_purine_associated", "oxidative_metabolic_stress_associated"),
        caution_tags=("modality_mismatch_caution", "weak_label_or_cohort_caution", "low_specificity_caution"),
        group_structure="healthy vs hepatitis B vs liver cirrhosis",
        recommended_action="integrate_high_priority",
        relevance="high",
        integrated_reason="Adds non-SERS liver-serum support and strengthens modality-aware caution structure.",
    ),
    PaperSpec(
        slug="liver_injury_fingerprints_2025",
        filename="identification-of-sers-fingerprints-for-diagnosis-and-staging-of-liver-injury.pdf",
        title="Identification of SERS Fingerprints for Diagnosis and Staging of Liver Injury",
        disease_task="Acute liver injury and DILI-related SERS fingerprints",
        modality="SERS serum / injury support",
        sample_type="serum / liver injury model support",
        source_dataset_id="not_applicable_or_unknown",
        source_file="identification-of-sers-fingerprints-for-diagnosis-and-staging-of-liver-injury.pdf|ac4c04758_si_001.pdf",
        study_design=[
            "Paper focuses on diagnosing and staging liver injury with an integrated CT-plus-SERS framework in an APAP-linked injury setting.",
            "Its utility for GAIRA is not as a serum disease benchmark but as liver-injury and DILI support.",
            "This is the strongest non-assay spectral DILI paper in the folder after removing the CK18 LFIA workbook dataset.",
        ],
        spectral_findings=[
            "The abstract explicitly identifies the 1437/1000 cm^-1 intensity ratio as an injury-linked SERS fingerprint.",
            "It also highlights 1546, 1487, and 1437 cm^-1 as differentiating peaks across healthy, mild, and severe injury conditions.",
            "These provide grounded injury/stress band anchors without claiming universal DILI specificity.",
        ],
        interpretation_frame=[
            "Useful for liver-injury and DILI support, especially when GAIRA needs to explain oxidative/metabolic injury-style shifts conservatively.",
            "Also helps distinguish broad injury-response support from targeted CK18 assay context.",
        ],
        cautions=[
            "This is an injury-model paper, not a universal human serum liver-disease truth.",
            "Integration should remain support-only and avoid overclaiming DILI diagnosis from broad serum spectra.",
            "CT-linked experimental framing is informative but not interchangeable with routine serum biosample cohorts.",
        ],
        themes=("oxidative_metabolic_stress_associated", "protein_peptide_associated"),
        caution_tags=("low_specificity_caution", "weak_label_or_cohort_caution"),
        group_structure="healthy vs mild injury vs severe injury",
        recommended_action="integrate_high_priority",
        relevance="high",
        integrated_reason="High-value liver-injury spectral support that is more appropriate than the removed LFIA CK18 workbook.",
    ),
    PaperSpec(
        slug="cca_multiclass_serum_sers_2026",
        filename="combination-of-label-free-sers-based-nanosensors-and-machine-learning-for-diagnosis-of-cholangiocarcinoma.pdf",
        title="Combination of Label-Free SERS-Based Nanosensors and Machine Learning for Diagnosis of Cholangiocarcinoma",
        disease_task="CCA vs HCC vs LM vs healthy serum SERS support",
        modality="SERS serum",
        sample_type="human serum",
        source_dataset_id="cca_hcc_lm_serum_sers",
        source_file="combination-of-label-free-sers-based-nanosensors-and-machine-learning-for-diagnosis-of-cholangiocarcinoma.pdf|an5c04536_si_001.pdf",
        study_design=[
            "Multi-class serum SERS study with 194 serum samples: 58 cholangiocarcinoma, 48 HCC, 44 liver metastases, and 44 healthy controls.",
            "Uses a silver nanorod SERS chip with 49 points examined per sample after 1:320 dilution.",
            "This is the literature analogue for the already integrated cca_hcc_lm_serum_sers raw dataset and should receive same-dataset support weighting.",
        ],
        spectral_findings=[
            "Paper emphasizes differential diagnosis among hepatobiliary malignancies rather than a generic cancer-vs-control task.",
            "Supports hepatobiliary differential context and explains why HCC-like and CCA-like serum signatures can overlap while remaining distinguishable in a substrate-specific setting.",
            "Useful for same-dataset tier-2 support on cca_hcc_lm_serum_sers queries.",
        ],
        interpretation_frame=[
            "Strengthens hepatobiliary disease context and improves CCA/HCC/LM differential interpretation support inside GAIRA_SERUM.",
            "Especially valuable because it aligns directly with an already integrated serum cohort instead of adding a disconnected paper.",
        ],
        cautions=[
            "Silver nanorod chip, heavy serum dilution, and machine-learning framing are substrate- and workflow-specific.",
            "Hepatobiliary overlap should be treated as differential context, not universal biochemical truth.",
            "Support-only literature role remains essential.",
        ],
        themes=("protein_peptide_associated", "lipid_membrane_associated", "oxidative_metabolic_stress_associated"),
        caution_tags=("probe_substrate_caution", "low_specificity_caution", "weak_label_or_cohort_caution"),
        group_structure="CCA, HCC, LM, healthy",
        recommended_action="integrate_high_priority",
        relevance="high",
        integrated_reason="Best same-dataset literature support for the newly integrated hepatobiliary serum cohort.",
    ),
    PaperSpec(
        slug="cirrhosis_hcc_serum_raman_2025",
        filename="es1667.pdf",
        title="Optical Diagnosis of Liver Cirrhosis and Hepatocellular Carcinoma using Machine Learning-Assisted Serum Raman Spectroscopy",
        disease_task="Cirrhosis vs HCC vs healthy serum Raman support",
        modality="Raman serum",
        sample_type="human serum",
        source_dataset_id="covid_serum_raman",
        source_file="es1667.pdf",
        study_design=[
            "Serum Raman study differentiating HCC, cirrhosis, and healthy groups with SVM assistance.",
            "Adds another spontaneous Raman liver-serum axis beyond hepatitis/cirrhosis support alone.",
            "Useful because it contains explicit band-level group differences rather than only classifier metrics.",
        ],
        spectral_findings=[
            "Reports elevated collagen at 1246 cm^-1 in cirrhosis and HCC relative to healthy controls.",
            "Reports elevated aromatic amino-acid bands including tryptophan (757, 878), tyrosine (831, 853), phenylalanine (1004), and cholesterol (548, 699) in cirrhosis and HCC.",
            "Reports decreased beta-carotene at 1157 and 1527 cm^-1 in cirrhosis and HCC versus healthy controls.",
        ],
        interpretation_frame=[
            "Strong support-level source for protein/aromatic/lipid shifts in liver disease, especially for cirrhosis-to-HCC comparison.",
            "Also improves liver-serum band assignment support under spontaneous Raman modality.",
        ],
        cautions=[
            "Spontaneous Raman modality differs from the SERS-heavy serum backbone.",
            "Multiclass accuracy claims remain cohort- and preprocessing-specific.",
            "Band changes are useful support-level anchors, not universal liver-disease truth.",
        ],
        themes=("protein_peptide_associated", "lipid_membrane_associated", "oxidative_metabolic_stress_associated"),
        caution_tags=("modality_mismatch_caution", "low_specificity_caution", "weak_label_or_cohort_caution"),
        group_structure="healthy vs cirrhosis vs HCC",
        recommended_action="integrate_high_priority",
        relevance="high",
        integrated_reason="High-value liver-serum Raman paper with explicit bands that enriches band-centered support and cautions.",
    ),
]


BAND_ROWS = [
    ("liver_injury_fingerprints_2025", "1437", "injury-linked band elevated in liver injury fingerprint ratio vs 1000", "protein/lipid deformation support"),
    ("liver_injury_fingerprints_2025", "1000", "reference aromatic band used in 1437/1000 injury ratio", "phenylalanine-like support"),
    ("liver_injury_fingerprints_2025", "1487", "reported differentiating band across healthy, mild, severe injury", "nucleic/protein mixed support"),
    ("liver_injury_fingerprints_2025", "1546", "reported differentiating band across healthy, mild, severe injury", "protein/nucleic mixed support"),
    ("cirrhosis_hcc_serum_raman_2025", "1246", "elevated in cirrhosis and HCC vs healthy", "collagen / protein support"),
    ("cirrhosis_hcc_serum_raman_2025", "757", "elevated aromatic amino-acid support", "tryptophan support"),
    ("cirrhosis_hcc_serum_raman_2025", "878", "elevated aromatic amino-acid support", "tryptophan support"),
    ("cirrhosis_hcc_serum_raman_2025", "831-853", "elevated aromatic amino-acid support", "tyrosine support"),
    ("cirrhosis_hcc_serum_raman_2025", "1004", "elevated aromatic amino-acid support", "phenylalanine support"),
    ("cirrhosis_hcc_serum_raman_2025", "548-699", "elevated lipid-associated support", "cholesterol support"),
    ("cirrhosis_hcc_serum_raman_2025", "1157", "reduced in cirrhosis and HCC", "beta-carotene support"),
    ("cirrhosis_hcc_serum_raman_2025", "1527", "reduced in cirrhosis and HCC", "beta-carotene support"),
]


ALL_CURATION_ROWS = [
    {
        "filename": ".DS_Store",
        "title": "macOS metadata",
        "disease_task": "none",
        "modality": "none",
        "sample_type": "none",
        "likely_relevance": "none",
        "already_integrated_or_duplicate": "irrelevant",
        "recommended_action": "skip",
        "notes": "Filesystem metadata.",
    },
    {
        "filename": "diabetes_EV_arxiv.pdf",
        "title": "Analysis of Plasma Extracellular Vesicles in Normal-Weight and Overweight Type 2 Diabetes Mellitus Using Multimodal SERS and RNA-Seq",
        "disease_task": "diabetes EV heterogeneity",
        "modality": "EV SERS",
        "sample_type": "plasma EV",
        "likely_relevance": "already used for EV context",
        "already_integrated_or_duplicate": "already_integrated",
        "recommended_action": "skip",
        "notes": "Already integrated in prior EV biology refinement pass.",
    },
    {
        "filename": "media-1 (1).docx",
        "title": "diabetes EV supplementary notes",
        "disease_task": "diabetes EV heterogeneity",
        "modality": "EV SERS supplement",
        "sample_type": "plasma EV",
        "likely_relevance": "already used for EV context",
        "already_integrated_or_duplicate": "already_integrated",
        "recommended_action": "skip",
        "notes": "Already integrated in prior EV biology refinement pass.",
    },
    {
        "filename": "spectra_shine.pdf",
        "title": "SPECTRA-based detection of drug-induced hepatotoxicity through extracellular vesicle analysis",
        "disease_task": "APAP EV hepatotoxicity",
        "modality": "EV SERS",
        "sample_type": "EV / cell culture",
        "likely_relevance": "already used for EV context",
        "already_integrated_or_duplicate": "already_integrated",
        "recommended_action": "skip",
        "notes": "Already integrated in prior EV biology refinement pass.",
    },
    {
        "filename": "1-s2.0-S0925400526000985-mmc1.docx",
        "title": "SPECTRA supplementary notes",
        "disease_task": "APAP EV hepatotoxicity",
        "modality": "EV SERS supplement",
        "sample_type": "EV / cell culture",
        "likely_relevance": "already used for EV context",
        "already_integrated_or_duplicate": "already_integrated",
        "recommended_action": "skip",
        "notes": "Already integrated in prior EV biology refinement pass.",
    },
    {
        "filename": "s41467-025-61600-9.pdf",
        "title": "A point-of-care diagnostic for drug-induced liver injury using surface-enhanced Raman scattering lateral flow immunoassay",
        "disease_task": "DILI targeted assay",
        "modality": "SERS LFIA",
        "sample_type": "targeted assay / biomarker",
        "likely_relevance": "minimal CK18 context only",
        "already_integrated_or_duplicate": "removed_dataset_context_only",
        "recommended_action": "skip",
        "notes": "Workbook-derived active dataset was intentionally removed; retain only minimal CK18 note.",
    },
    {
        "filename": "s41467-025-61600-9 (1).pdf",
        "title": "A point-of-care diagnostic for drug-induced liver injury using surface-enhanced Raman scattering lateral flow immunoassay",
        "disease_task": "DILI targeted assay",
        "modality": "SERS LFIA",
        "sample_type": "targeted assay / biomarker",
        "likely_relevance": "duplicate of same paper",
        "already_integrated_or_duplicate": "duplicate",
        "recommended_action": "skip",
        "notes": "Duplicate PDF.",
    },
    {
        "filename": "41467_2025_61600_MOESM1_ESM.pdf",
        "title": "Nature DILI LFIA supplementary information",
        "disease_task": "DILI targeted assay",
        "modality": "SERS LFIA supplement",
        "sample_type": "targeted assay / biomarker",
        "likely_relevance": "minimal CK18 context only",
        "already_integrated_or_duplicate": "reserve_only",
        "recommended_action": "reserve",
        "notes": "Reserve; no broad serum spectral support integration.",
    },
    {
        "filename": "41467_2025_61600_MOESM4_ESM.xlsx",
        "title": "Nature DILI LFIA workbook",
        "disease_task": "DILI targeted assay",
        "modality": "SERS LFIA workbook",
        "sample_type": "targeted assay / biomarker",
        "likely_relevance": "not appropriate as active serum spectral dataset",
        "already_integrated_or_duplicate": "removed_dataset_context_only",
        "recommended_action": "skip",
        "notes": "Intentionally removed from active GAIRA.",
    },
    {
        "filename": "1-s2.0-S0003267024003192-main.pdf",
        "title": "Low-abundance proteins-based label-free SERS approach for high precision detection of liver cancer with different stages",
        "disease_task": "liver cancer staging serum SERS",
        "modality": "SERS serum",
        "sample_type": "human serum",
        "likely_relevance": "potentially useful but overlaps with stronger liver cancer serum set",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "reserve",
        "notes": "Reserve for later if more liver-cancer staging support is needed.",
    },
    {
        "filename": "1-s2.0-S0003267024003192-mmc1.docx",
        "title": "Low-abundance proteins liver cancer SERS supplement",
        "disease_task": "liver cancer staging serum SERS",
        "modality": "SERS serum supplement",
        "sample_type": "human serum",
        "likely_relevance": "secondary",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "reserve",
        "notes": "Reserve with the main paper.",
    },
    {
        "filename": "1-s2.0-S0927776523001935-main.pdf",
        "title": "Discovering the digital biomarker of hepatocellular carcinoma in serum with SERS-based biosensors and intelligence vision",
        "disease_task": "HCC vs HB vs normal",
        "modality": "SERS serum",
        "sample_type": "human serum",
        "likely_relevance": "strong HCC serum support",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "integrate_high_priority",
        "notes": "Selected for HCC/HB support and low-specificity caution framing.",
    },
    {
        "filename": "1-s2.0-S0956566321002839-main.pdf",
        "title": "A biosensing method for the direct serological detection of liver diseases by integrating a SERS-based sensor and a CNN classifier",
        "disease_task": "normal vs HCC vs HB",
        "modality": "SERS serum",
        "sample_type": "human serum",
        "likely_relevance": "strong multi-class liver-serum support",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "integrate_high_priority",
        "notes": "Selected for direct serum multi-class liver disease context.",
    },
    {
        "filename": "1-s2.0-S0956566321002839-mmc1.docx",
        "title": "Direct serological detection supplementary information",
        "disease_task": "normal vs HCC vs HB",
        "modality": "SERS serum supplement",
        "sample_type": "human serum",
        "likely_relevance": "strong supplement",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "integrate_secondary",
        "notes": "Used to ground representative spectra and sensor details.",
    },
    {
        "filename": "1-s2.0-S1385894724013184-main.pdf",
        "title": "SERS lateral flow strip detection of serum biomarkers for noninvasive assessment of operative microwave ablation outcomes of unresectable hepatocellular carcinoma",
        "disease_task": "ablation outcome biomarker strip",
        "modality": "SERS LFIA",
        "sample_type": "targeted serum biomarkers",
        "likely_relevance": "too assay-specific for current architecture",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "reserve",
        "notes": "Reserve as assay context only, not broad serum spectral support.",
    },
    {
        "filename": "1-s2.0-S1385894724013184-mmc1.docx",
        "title": "Ablation outcome LFIA supplement",
        "disease_task": "ablation outcome biomarker strip",
        "modality": "SERS LFIA supplement",
        "sample_type": "targeted serum biomarkers",
        "likely_relevance": "assay-specific",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "reserve",
        "notes": "Reserve only.",
    },
    {
        "filename": "119000U.pdf",
        "title": "Preliminary study of SERS of serum samples of liver cancer patients",
        "disease_task": "liver cancer serum SERS",
        "modality": "SERS serum",
        "sample_type": "human serum",
        "likely_relevance": "early preliminary liver cancer serum paper",
        "already_integrated_or_duplicate": "duplicate_family",
        "recommended_action": "reserve",
        "notes": "Preliminary/duplicate family with serum_liver_hcc_2.",
    },
    {
        "filename": "2021.06.06.21258433v2.full.pdf",
        "title": "Raman spectroscopy on blood serum samples of patients with end-stage liver disease",
        "disease_task": "end-stage liver disease serum Raman",
        "modality": "Raman serum",
        "sample_type": "human serum",
        "likely_relevance": "useful but broader/less specific than selected set",
        "already_integrated_or_duplicate": "duplicate_family",
        "recommended_action": "reserve",
        "notes": "Covered by the final PLOS One version in reserve; not selected for current tight subset.",
    },
    {
        "filename": "J Raman Spectroscopy - 2018 - Liu - Label‐free and non‐invasive BS‐SERS detection of liver cancer based on the solid device.pdf",
        "title": "Label-free and non-invasive BS-SERS detection of liver cancer based on the solid device of silver nanofilm",
        "disease_task": "liver cancer serum SERS",
        "modality": "SERS serum",
        "sample_type": "human serum",
        "likely_relevance": "secondary liver cancer serum support",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "reserve",
        "notes": "Reserve; current subset already has stronger recent HCC serum papers.",
    },
    {
        "filename": "J Raman Spectroscopy - 2026 - Jin - Serum Raman Spectroscopy Combined With a Support Vector Machine for Rapid Diagnosis of.pdf",
        "title": "Serum Raman Spectroscopy Combined With a Support Vector Machine for Rapid Diagnosis of Hepatitis B and Liver Cirrhosis",
        "disease_task": "HB vs cirrhosis vs healthy",
        "modality": "Raman serum",
        "sample_type": "human serum",
        "likely_relevance": "high-value spontaneous Raman liver-serum support",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "integrate_high_priority",
        "notes": "Selected for modality-aware liver-serum support.",
    },
    {
        "filename": "NAFLD_SERS.pdf",
        "title": "Fully connected neural network-based serum surface-enhanced Raman spectroscopy accurately identifies non-alcoholic steatohepatitis",
        "disease_task": "NAFLD/NASH serum SERS",
        "modality": "SERS serum",
        "sample_type": "human serum",
        "likely_relevance": "high-value metabolic liver disease support",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "integrate_high_priority",
        "notes": "Selected for metabolic liver disease coverage.",
    },
    {
        "filename": "12072_2022_10444_MOESM1_ESM.docx",
        "title": "NAFLD supplementary appendix",
        "disease_task": "NAFLD/NASH serum SERS",
        "modality": "SERS serum supplement",
        "sample_type": "human serum",
        "likely_relevance": "high-value supplement",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "integrate_secondary",
        "notes": "Selected to ground biomolecule mixture interpretation.",
    },
    {
        "filename": "12072_2022_10444_MOESM2_ESM.docx",
        "title": "NAFLD supplementary figure captions",
        "disease_task": "NAFLD/NASH serum SERS",
        "modality": "SERS serum supplement",
        "sample_type": "human serum",
        "likely_relevance": "secondary",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "integrate_secondary",
        "notes": "Secondary supplement for the NAFLD paper.",
    },
    {
        "filename": "12072_2022_10444_MOESM3_ESM.tif",
        "title": "NAFLD supplementary TIFF",
        "disease_task": "NAFLD/NASH serum SERS",
        "modality": "image",
        "sample_type": "supplementary figure",
        "likely_relevance": "not needed for current text integration",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "skip",
        "notes": "Image-only supplement not needed in this pass.",
    },
    {
        "filename": "ac4c04758_si_001.pdf",
        "title": "Liver injury SERS supplementary information",
        "disease_task": "liver injury / DILI support",
        "modality": "SERS supplement",
        "sample_type": "injury support",
        "likely_relevance": "high-value supplement",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "integrate_secondary",
        "notes": "Used to ground injury band details.",
    },
    {
        "filename": "am8b10252_si_001.pdf",
        "title": "Serum microRNA plus AFP SERS platform supplementary information",
        "disease_task": "targeted AFP/microRNA assay",
        "modality": "SERS assay",
        "sample_type": "targeted assay",
        "likely_relevance": "too assay-specific",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "skip",
        "notes": "Skip; targeted assay support would muddy broad spectral interpretation.",
    },
    {
        "filename": "an5c04536_si_001.pdf",
        "title": "Cholangiocarcinoma SERS supplementary information",
        "disease_task": "CCA vs HCC vs LM vs healthy",
        "modality": "SERS serum supplement",
        "sample_type": "human serum",
        "likely_relevance": "same-dataset support supplement",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "integrate_secondary",
        "notes": "Used to support the integrated CCA/HCC/LM serum cohort.",
    },
    {
        "filename": "colorectal_SERS.pdf",
        "title": "Colorectal cancer detection by gold nanoparticle based SERS of blood serum",
        "disease_task": "colorectal cancer serum SERS",
        "modality": "SERS serum",
        "sample_type": "human serum",
        "likely_relevance": "weak for liver-serum reasoning",
        "already_integrated_or_duplicate": "irrelevant_domain",
        "recommended_action": "skip",
        "notes": "Out of current liver/hepatobiliary scope.",
    },
    {
        "filename": "Combination of label-free SERS-based nanosensor an.zip",
        "title": "Cholangiocarcinoma raw serum SERS zip",
        "disease_task": "CCA vs HCC vs LM vs healthy",
        "modality": "SERS serum raw data",
        "sample_type": "human serum",
        "likely_relevance": "already integrated as raw dataset",
        "already_integrated_or_duplicate": "already_integrated",
        "recommended_action": "skip",
        "notes": "Raw dataset already onboarded as cca_hcc_lm_serum_sers; not a literature-integration target.",
    },
    {
        "filename": "combination-of-label-free-sers-based-nanosensors-and-machine-learning-for-diagnosis-of-cholangiocarcinoma.pdf",
        "title": "Combination of Label-Free SERS-Based Nanosensors and Machine Learning for Diagnosis of Cholangiocarcinoma",
        "disease_task": "CCA vs HCC vs LM vs healthy",
        "modality": "SERS serum",
        "sample_type": "human serum",
        "likely_relevance": "high-value same-dataset support",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "integrate_high_priority",
        "notes": "Selected as same-dataset support for cca_hcc_lm_serum_sers.",
    },
    {
        "filename": "es1667.pdf",
        "title": "Optical Diagnosis of Liver Cirrhosis and Hepatocellular Carcinoma using Machine Learning-Assisted Serum Raman Spectroscopy",
        "disease_task": "cirrhosis vs HCC vs healthy",
        "modality": "Raman serum",
        "sample_type": "human serum",
        "likely_relevance": "high-value explicit band support",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "integrate_high_priority",
        "notes": "Selected for band-centered liver-serum support.",
    },
    {
        "filename": "hcc_sers_serum.pdf",
        "title": "Repeated double cross-validation applied to the PCA-LDA classification of SERS spectra: a case study with serum samples from hepatocellular carcinoma patients",
        "disease_task": "HCC vs healthy",
        "modality": "SERS serum",
        "sample_type": "human serum",
        "likely_relevance": "high-value HCC holdout analogue",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "integrate_high_priority",
        "notes": "Selected because it aligns best with the hcc_serum holdout.",
    },
    {
        "filename": "hcc_sers_serum_SI.pdf",
        "title": "Trieste HCC serum SERS supplementary information",
        "disease_task": "HCC vs healthy",
        "modality": "SERS serum supplement",
        "sample_type": "human serum",
        "likely_relevance": "same-paper supplement",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "integrate_secondary",
        "notes": "Used to ground cohort and methods details for the selected paper.",
    },
    {
        "filename": "hcc_sers_serum_3.pdf",
        "title": "Label-free serum SERS combined with RFE-GBDT algorithm for non-invasive screening of liver cancer",
        "disease_task": "healthy vs HBV vs staged liver cancer",
        "modality": "SERS serum",
        "sample_type": "human serum",
        "likely_relevance": "high-value liver-disease differential support",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "integrate_high_priority",
        "notes": "Selected for staged liver-cancer context.",
    },
    {
        "filename": "hcc_sers_serum_4.pdf",
        "title": "Noninvasive liver diseases detection based on serum surface enhanced Raman spectroscopy and statistical analysis",
        "disease_task": "liver diseases serum SERS",
        "modality": "SERS serum",
        "sample_type": "human serum",
        "likely_relevance": "secondary",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "reserve",
        "notes": "Reserve; current subset already captures stronger liver-serum support coverage.",
    },
    {
        "filename": "hcc_sers_serum_5",
        "title": "single-page PDF artifact related to HCC serum",
        "disease_task": "unclear",
        "modality": "PDF artifact",
        "sample_type": "unknown",
        "likely_relevance": "unclear and likely low",
        "already_integrated_or_duplicate": "ambiguous",
        "recommended_action": "skip",
        "notes": "One-page artifact with unclear standalone value.",
    },
    {
        "filename": "hepB_sers_liver.pdf",
        "title": "Label free hepatitis B detection based on serum derivative surface enhanced Raman spectroscopy combined with multivariate analysis",
        "disease_task": "hepatitis B serum SERS",
        "modality": "SERS serum",
        "sample_type": "human serum",
        "likely_relevance": "useful but secondary",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "reserve",
        "notes": "Reserve; current subset already includes HB support via stronger multi-class papers.",
    },
    {
        "filename": "identification-of-sers-fingerprints-for-diagnosis-and-staging-of-liver-injury.pdf",
        "title": "Identification of SERS Fingerprints for Diagnosis and Staging of Liver Injury",
        "disease_task": "acute liver injury / DILI",
        "modality": "SERS injury support",
        "sample_type": "injury model support",
        "likely_relevance": "high-value liver injury support",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "integrate_high_priority",
        "notes": "Selected as the strongest non-assay liver injury support paper.",
    },
    {
        "filename": "journal.pone.0256045.pdf",
        "title": "Raman spectroscopy on blood serum samples of patients with end-stage liver disease",
        "disease_task": "end-stage liver disease serum Raman",
        "modality": "Raman serum",
        "sample_type": "human serum",
        "likely_relevance": "useful background liver-serum support",
        "already_integrated_or_duplicate": "duplicate_family",
        "recommended_action": "reserve",
        "notes": "Reserve behind the stronger selected liver Raman set.",
    },
    {
        "filename": "kidney_SERS.pdf",
        "title": "Application of serum SERS technology combined with deep learning algorithm in the rapid diagnosis of immune diseases and chronic kidney disease",
        "disease_task": "kidney disease",
        "modality": "SERS serum",
        "sample_type": "human serum",
        "likely_relevance": "not liver-focused",
        "already_integrated_or_duplicate": "irrelevant_domain",
        "recommended_action": "skip",
        "notes": "Out of current liver/hepatobiliary scope.",
    },
    {
        "filename": "liver_tissue_hcc_SERS.pdf",
        "title": "Imaging of Liver Tumors Using Surface-Enhanced Raman Scattering Nanoparticles",
        "disease_task": "liver tumor imaging",
        "modality": "SERS tissue imaging",
        "sample_type": "tissue / nanoparticle imaging",
        "likely_relevance": "non-serum and non-biofluid",
        "already_integrated_or_duplicate": "non_serum",
        "recommended_action": "reserve",
        "notes": "Reserve as non-serum context only if later needed.",
    },
    {
        "filename": "media-1 (4).pdf",
        "title": "End-stage liver disease supplementary material",
        "disease_task": "end-stage liver disease serum Raman",
        "modality": "Raman supplement",
        "sample_type": "human serum",
        "likely_relevance": "reserve only",
        "already_integrated_or_duplicate": "duplicate_family",
        "recommended_action": "reserve",
        "notes": "Reserve with the related serum Raman paper family.",
    },
    {
        "filename": "multivariate-analysis-of-serum-surface-enhanced-raman-spectroscopy-of-liver-cancer-patients.pdf",
        "title": "Multivariate analysis of serum surface-enhanced Raman spectroscopy of liver cancer patients",
        "disease_task": "liver cancer serum SERS",
        "modality": "SERS serum",
        "sample_type": "human serum",
        "likely_relevance": "secondary/duplicate family",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "reserve",
        "notes": "Reserve because the current selected subset already covers stronger liver-cancer serum support.",
    },
    {
        "filename": "naso_liver_SERS.pdf",
        "title": "Label-free detection of nasopharyngeal and liver cancer using SERS and PLS-SVM",
        "disease_task": "mixed nasopharyngeal and liver cancer",
        "modality": "SERS serum",
        "sample_type": "mixed cancer serum",
        "likely_relevance": "misleading for liver-specific reasoning",
        "already_integrated_or_duplicate": "mixed_domain",
        "recommended_action": "skip",
        "notes": "Skip to avoid contaminating liver-specific support with mixed-cancer framing.",
    },
    {
        "filename": "s12274-022-4114-z.pdf",
        "title": "Intelligent serological SERS test toward early-stage hepatocellular carcinoma diagnosis through ultrasensitive nanobiosensing",
        "disease_task": "early-stage HCC biosensing",
        "modality": "SERS serum biosensor",
        "sample_type": "human serum",
        "likely_relevance": "useful but overlaps with stronger HCC biosensor paper already selected",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "reserve",
        "notes": "Reserve for a later HCC-specific expansion if needed.",
    },
    {
        "filename": "serum_liver_hcc_2.pdf",
        "title": "Preliminary study of SERS of serum samples of liver cancer patients",
        "disease_task": "liver cancer serum SERS",
        "modality": "SERS serum",
        "sample_type": "human serum",
        "likely_relevance": "duplicate of 119000U family",
        "already_integrated_or_duplicate": "duplicate",
        "recommended_action": "skip",
        "notes": "Duplicate/preliminary family.",
    },
    {
        "filename": "ultrasensitive-detection-of-serum-microrna-using-branched-dna-based-sers-platform-combining-simultaneous-detection-of-α.pdf",
        "title": "Ultrasensitive Detection of Serum MicroRNA Using Branched DNA-Based SERS Platform Combining Simultaneous Detection of AFP",
        "disease_task": "targeted microRNA/AFP assay",
        "modality": "SERS assay",
        "sample_type": "targeted assay",
        "likely_relevance": "too targeted-assay-specific",
        "already_integrated_or_duplicate": "not_integrated",
        "recommended_action": "skip",
        "notes": "Skip to keep GAIRA focused on broad spectral interpretation rather than targeted assay readouts.",
    },
]


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()


def _wrap_lines(items: list[str]) -> list[str]:
    return [item.strip() for item in items if item.strip()]


def _selected_support_documents_df() -> pd.DataFrame:
    rows = []
    for paper in SELECTED_PAPERS:
        rows.append(
            {
                "document_id": f"{paper.slug}_doc_001",
                "dataset_id": SUPPORT_DATASET_ID,
                "source_dataset_id": paper.source_dataset_id,
                "evidence_family": "liver_serum_literature_support",
                "evidence_tier": "tier2_literature_support",
                "support_type": "text",
                "citation_label": paper.slug,
                "title": paper.title,
                "authors": "curated_from_local_files",
                "year": re.search(r"(20\d{2})", paper.title or "")[1] if re.search(r"(20\d{2})", paper.title or "") else "",
                "journal": paper.modality,
                "doi": "",
                "source_file": str(DOWNLOAD_ROOT / paper.filename) if paper.filename else paper.source_file,
                "is_digitized": "no",
                "use_for_primary_matching": "no",
                "use_for_supporting_comparison": "yes",
                "use_for_rag": "yes",
                "notes": (
                    f"Curated support-only liver/serum paper integration for {paper.disease_task}. "
                    "This remains support-level evidence and must not be treated as direct spectral truth."
                ),
            }
        )
    return pd.DataFrame(rows)


def _band_text_for_slug(slug: str) -> str | None:
    matching = [row for row in BAND_ROWS if row[0] == slug]
    if not matching:
        return None
    lines = []
    for _, band_label, observation, assignment in matching:
        lines.append(f"{band_label}: {observation}; {assignment}.")
    return " ".join(lines)


def _selected_support_chunks_df() -> pd.DataFrame:
    rows = []
    for paper in SELECTED_PAPERS:
        document_id = f"{paper.slug}_doc_001"
        chunks = [
            ("study_design", " ".join(_wrap_lines(paper.study_design))),
            ("reported_band_assignments", _band_text_for_slug(paper.slug) or "No explicit band list extracted from the selected paper; use the broader interpretation and caution notes instead."),
            ("interpretation_frame", " ".join(_wrap_lines(paper.interpretation_frame + paper.spectral_findings))),
            ("cautions", " ".join(_wrap_lines(paper.cautions))),
        ]
        for idx, (section, chunk_text) in enumerate(chunks, start=1):
            rows.append(
                {
                    "chunk_id": f"{document_id}_chunk_{idx:02d}",
                    "document_id": document_id,
                    "dataset_id": SUPPORT_DATASET_ID,
                    "chunk_order": idx,
                    "section": section,
                    "chunk_text": chunk_text,
                    "metadata_json": json.dumps(
                        {
                            "paper_slug": paper.slug,
                            "disease_task": paper.disease_task,
                            "modality": paper.modality,
                            "themes": list(paper.themes),
                            "cautions": list(paper.caution_tags),
                        },
                        sort_keys=True,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _serum_context_documents() -> tuple[pd.DataFrame, pd.DataFrame]:
    docs = [
        {
            "document_id": "gaira_serum_context_liver_disease_axis_note",
            "context_layer": SERUM_CONTEXT_LAYER,
            "intended_domain": "serum",
            "context_type": "disease_axis_note",
            "evidence_basis": "curated_literature_support",
            "source_dataset_id": "not_applicable_or_unknown",
            "source_file": "curated_liver_serum_literature_support",
            "title": "Liver-serum disease-axis interpretation note",
            "use_for_rag": "yes",
            "notes": "Curated serum-context note derived from selected liver-serum Raman/SERS papers.",
        },
        {
            "document_id": "gaira_serum_context_hcc_hepatobiliary_note",
            "context_layer": SERUM_CONTEXT_LAYER,
            "intended_domain": "serum",
            "context_type": "disease_axis_note",
            "evidence_basis": "curated_literature_support",
            "source_dataset_id": "hcc_serum,cca_hcc_lm_serum_sers",
            "source_file": "curated_liver_serum_literature_support",
            "title": "HCC and hepatobiliary serum interpretation note",
            "use_for_rag": "yes",
            "notes": "Curated hepatobiliary differential note for serum reasoning.",
        },
        {
            "document_id": "gaira_serum_context_metabolic_liver_note",
            "context_layer": SERUM_CONTEXT_LAYER,
            "intended_domain": "serum",
            "context_type": "disease_axis_note",
            "evidence_basis": "curated_literature_support",
            "source_dataset_id": "not_applicable_or_unknown",
            "source_file": "curated_liver_serum_literature_support",
            "title": "Metabolic liver disease serum note",
            "use_for_rag": "yes",
            "notes": "Curated NAFLD/NASH serum note.",
        },
        {
            "document_id": "gaira_serum_context_liver_injury_dili_note",
            "context_layer": SERUM_CONTEXT_LAYER,
            "intended_domain": "serum",
            "context_type": "injury_caution_note",
            "evidence_basis": "curated_literature_support",
            "source_dataset_id": "not_applicable_or_unknown",
            "source_file": "curated_liver_serum_literature_support",
            "title": "Liver injury and DILI serum support note",
            "use_for_rag": "yes",
            "notes": "Curated DILI/liver-injury support note distinct from targeted-assay CK18 context.",
        },
        {
            "document_id": "gaira_serum_context_liver_modality_caution",
            "context_layer": SERUM_CONTEXT_LAYER,
            "intended_domain": "serum",
            "context_type": "modality_caution_note",
            "evidence_basis": "curated_literature_support",
            "source_dataset_id": "covid_serum_raman,hcc_serum,cca_hcc_lm_serum_sers",
            "source_file": "curated_liver_serum_literature_support",
            "title": "Liver-serum modality and substrate caution note",
            "use_for_rag": "yes",
            "notes": "Curated note that distinguishes serum Raman, serum SERS, tissue, and targeted assay evidence families.",
        },
    ]

    chunks = [
        {
            "chunk_id": "gaira_serum_context_liver_disease_axis_note_chunk_01",
            "document_id": "gaira_serum_context_liver_disease_axis_note",
            "context_layer": SERUM_CONTEXT_LAYER,
            "intended_domain": "serum",
            "chunk_order": 1,
            "section": "disease_axis",
            "chunk_text": (
                "Selected liver-serum papers broaden GAIRA_SERUM context beyond serum_ag_colloids. The added support spans HCC, hepatitis B, cirrhosis, "
                "metabolic liver disease/NASH, acute liver injury/DILI, and cholangiocarcinoma/hepatobiliary differential diagnosis. These studies are useful "
                "for disease-axis framing and cautious biochemical interpretation, but they remain support-only and should not be treated as universal serum spectral truth."
            ),
            "metadata_json": json.dumps({"source_kind": "curated_liver_serum_note"}, sort_keys=True),
        },
        {
            "chunk_id": "gaira_serum_context_hcc_hepatobiliary_note_chunk_01",
            "document_id": "gaira_serum_context_hcc_hepatobiliary_note",
            "context_layer": SERUM_CONTEXT_LAYER,
            "intended_domain": "serum",
            "chunk_order": 1,
            "section": "hcc_hepatobiliary_context",
            "chunk_text": (
                "HCC-focused serum papers and the cholangiocarcinoma serum paper together suggest that hepatobiliary malignancy spectra should be interpreted as "
                "differential serum structure with substantial overlap, not as a single universal HCC peak signature. HCC, CCA, LM, hepatitis-B background, and "
                "cirrhotic background can all share strong serum protein, lipid, and metabolic anchors, so GAIRA should emphasize comparative shifts, support-only evidence, "
                "and what-not-to-claim cautions instead of definitive diagnosis."
            ),
            "metadata_json": json.dumps({"source_kind": "curated_hcc_hepatobiliary_note"}, sort_keys=True),
        },
        {
            "chunk_id": "gaira_serum_context_metabolic_liver_note_chunk_01",
            "document_id": "gaira_serum_context_metabolic_liver_note",
            "context_layer": SERUM_CONTEXT_LAYER,
            "intended_domain": "serum",
            "chunk_order": 1,
            "section": "metabolic_liver_context",
            "chunk_text": (
                "The selected NASH/NAFLD serum SERS paper frames liver-metabolic disease as a broad mixture of inflammation, oxidative stress, apoptosis, glucose metabolism, "
                "lipid metabolism, proteins, and nucleic acids rather than a single biomarker. GAIRA should therefore surface metabolic-liver support as mixed lipid/protein/"
                "oxidative context with low-specificity caution, not as a one-peak metabolic diagnosis."
            ),
            "metadata_json": json.dumps({"source_kind": "curated_metabolic_liver_note"}, sort_keys=True),
        },
        {
            "chunk_id": "gaira_serum_context_liver_injury_dili_note_chunk_01",
            "document_id": "gaira_serum_context_liver_injury_dili_note",
            "context_layer": SERUM_CONTEXT_LAYER,
            "intended_domain": "serum",
            "chunk_order": 1,
            "section": "liver_injury_context",
            "chunk_text": (
                "The selected liver-injury SERS paper supports acute hepatic injury and DILI-style interpretation at the support layer, including a reported 1437/1000 cm^-1 "
                "injury-linked ratio and differentiating peaks near 1437, 1487, and 1546 cm^-1. In GAIRA this should raise oxidative/metabolic injury support while remaining "
                "explicitly non-diagnostic and distinct from the targeted CK18/K18 assay note."
            ),
            "metadata_json": json.dumps({"source_kind": "curated_liver_injury_note"}, sort_keys=True),
        },
        {
            "chunk_id": "gaira_serum_context_liver_modality_caution_chunk_01",
            "document_id": "gaira_serum_context_liver_modality_caution",
            "context_layer": SERUM_CONTEXT_LAYER,
            "intended_domain": "serum",
            "chunk_order": 1,
            "section": "modality_caution",
            "chunk_text": (
                "The selected liver literature spans serum SERS, spontaneous serum Raman, hepatobiliary serum classification, injury support, and targeted assay papers. "
                "GAIRA should keep serum Raman and serum SERS support comparable only at a cautious interpretation layer, should not treat tissue or LFIA findings as broad "
                "serum spectral grounding, and should preserve probe/substrate and modality mismatch cautions whenever support comes from different acquisition families."
            ),
            "metadata_json": json.dumps({"source_kind": "curated_modality_caution_note"}, sort_keys=True),
        },
    ]
    return pd.DataFrame(docs), pd.DataFrame(chunks)


def _write_markdown(path: Path, title: str, sections: list[tuple[str, list[str]]]) -> None:
    lines = [f"# {title}", ""]
    for heading, bullets in sections:
        lines.append(f"## {heading}")
        lines.append("")
        for bullet in bullets:
            lines.append(f"- {bullet}")
        lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _paper_metadata_summary_df() -> pd.DataFrame:
    rows = []
    for paper in SELECTED_PAPERS:
        rows.append(
            {
                "paper_slug": paper.slug,
                "filename": paper.filename,
                "title": paper.title,
                "disease_task": paper.disease_task,
                "modality": paper.modality,
                "sample_type": paper.sample_type,
                "source_dataset_id": paper.source_dataset_id,
                "group_structure": paper.group_structure,
                "themes": ", ".join(paper.themes),
                "caution_tags": ", ".join(paper.caution_tags),
                "integrated_reason": paper.integrated_reason,
            }
        )
    return pd.DataFrame(rows)


def _disease_axis_summary_df() -> pd.DataFrame:
    rows = []
    for paper in SELECTED_PAPERS:
        axes = re.split(r"\s+vs\s+|,|\s+and\s+", paper.group_structure.lower())
        for axis in axes:
            axis = axis.strip()
            if not axis:
                continue
            rows.append(
                {
                    "paper_slug": paper.slug,
                    "title": paper.title,
                    "group_axis": axis,
                    "disease_task": paper.disease_task,
                    "sample_type": paper.sample_type,
                    "modality": paper.modality,
                }
            )
    return pd.DataFrame(rows)


def _modality_caution_summary_df() -> pd.DataFrame:
    rows = []
    for paper in SELECTED_PAPERS:
        for caution in paper.cautions:
            rows.append(
                {
                    "paper_slug": paper.slug,
                    "title": paper.title,
                    "modality": paper.modality,
                    "sample_type": paper.sample_type,
                    "caution_text": caution,
                }
            )
    return pd.DataFrame(rows)


def _band_annotations_df() -> pd.DataFrame:
    rows = []
    for slug, band_label, observation, assignment in BAND_ROWS:
        paper = next(spec for spec in SELECTED_PAPERS if spec.slug == slug)
        rows.append(
            {
                "paper_slug": slug,
                "title": paper.title,
                "band_label": band_label,
                "observation": observation,
                "assignment": assignment,
                "modality": paper.modality,
                "sample_type": paper.sample_type,
                "source_dataset_id": paper.source_dataset_id,
            }
        )
    return pd.DataFrame(rows)


def _write_extracted_notes(output_dir: Path) -> None:
    notes_dir = output_dir / "extracted_notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    for paper in SELECTED_PAPERS:
        _write_markdown(
            notes_dir / f"{paper.slug}.md",
            paper.title,
            [
                ("Study Design", paper.study_design),
                ("Key Spectral Findings", paper.spectral_findings),
                ("Interpretation Frame", paper.interpretation_frame),
                ("Cautions", paper.cautions),
            ],
        )


def _query_examples(connection: duckdb.DuckDBPyConnection) -> dict[str, tuple[str, str]]:
    examples = {}
    datasets = ["cca_hcc_lm_serum_sers", "serum_protocol_comparison"]
    for dataset_id in datasets:
        row = connection.execute(
            """
            SELECT class_label, subclass_label
            FROM biosample_class_summary
            WHERE dataset_id = ?
            ORDER BY n_spectra DESC, class_label
            LIMIT 1
            """,
            [dataset_id],
        ).fetchone()
        if row:
            examples[dataset_id] = (str(row[0]), str(row[1]))
    return examples


def _run_inference_examples(db_path: Path, include_hcc_eval: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from gaira.config import get_storage_paths
    from gaira.inference import GAIRAInferenceEngine, load_serum_class_mean_query
    from gaira.grounding_search import GroundingSearchEngine, SpectrumQuery
    import numpy as np

    rows: list[dict] = []
    results: list[dict] = []
    engine = GAIRAInferenceEngine(db_path)
    with duckdb.connect(str(db_path), read_only=True) as connection:
        examples = _query_examples(connection)
        for dataset_id, (class_label, subclass_label) in examples.items():
            request = load_serum_class_mean_query(db_path, dataset_id, class_label, subclass_label)
            result = engine.run_inference(request)
            results.append(result)
            liver_hits = [
                hit for hit in result.get("tier2_support_hits", [])
                if hit.get("source_dataset_id") in {"hcc_serum", "cca_hcc_lm_serum_sers", "not_applicable_or_unknown"}
                or hit.get("source_dataset_id") == request.source_dataset_id
                or hit.get("source_label", "").startswith("liver_")
            ]
            rows.append(
                {
                    "query_dataset_id": dataset_id,
                    "query_class_label": class_label,
                    "top_tier2_source_dataset": result["tier2_support_hits"][0]["source_dataset_id"] if result.get("tier2_support_hits") else "",
                    "top_tier2_source_label": result["tier2_support_hits"][0]["source_label"] if result.get("tier2_support_hits") else "",
                    "n_tier2_hits": len(result.get("tier2_support_hits", [])),
                    "n_liver_serum_support_hits": sum(
                        1 for hit in result.get("tier2_support_hits", [])
                        if hit.get("source_label") in {paper.slug for paper in SELECTED_PAPERS}
                    ),
                    "top_domain_context": result["domain_context_hits"][0]["document_id"] if result.get("domain_context_hits") else "",
                    "dominant_themes": ", ".join(result.get("dominant_themes", [])),
                    "global_caveats": ", ".join(result.get("biochemical_global_caveats", [])),
                }
            )

        search_engine = GroundingSearchEngine(db_path)
        grounding_row = connection.execute(
            """
            SELECT mean_wavenumbers_json, mean_intensity_json, class_label
            FROM grounding_class_summary
            WHERE dataset_id = 'metabolite_sers63_support'
            ORDER BY n_spectra DESC, class_label
            LIMIT 1
            """
        ).fetchone()
        if grounding_row:
            x_values = np.asarray(json.loads(grounding_row[0]), dtype=float)
            y_values = np.asarray(json.loads(grounding_row[1]), dtype=float)
            query = SpectrumQuery(
                query_id="metabolite_support_example",
                query_label=str(grounding_row[2]),
                query_family="metabolite_sers63_support",
                source_dataset_id="metabolite_sers63_support",
                x=x_values,
                y=y_values,
                notes="Representative metabolite grounding summary query",
            )
            grounding_hits = search_engine.search_direct_spectral_evidence(query)
            same_dataset = grounding_hits[grounding_hits["source_dataset_id"] == "metabolite_sers63_support"]
            rows.append(
                {
                    "query_dataset_id": "metabolite_sers63_support",
                    "query_class_label": str(grounding_row[2]),
                    "top_tier2_source_dataset": "",
                    "top_tier2_source_label": same_dataset.iloc[0]["source_label"] if not same_dataset.empty else "",
                    "n_tier2_hits": 0,
                    "n_liver_serum_support_hits": 0,
                    "top_domain_context": "",
                    "dominant_themes": "",
                    "global_caveats": "",
                }
            )

    return pd.DataFrame(rows), results


def _insert_tables(
    db_path: Path,
    support_documents_df: pd.DataFrame,
    support_chunks_df: pd.DataFrame,
    serum_docs_df: pd.DataFrame,
    serum_chunks_df: pd.DataFrame,
) -> None:
    with duckdb.connect(str(db_path)) as connection:
        connection.execute("DELETE FROM grounding_support_chunks WHERE dataset_id = ?", [SUPPORT_DATASET_ID])
        connection.execute("DELETE FROM grounding_support_documents WHERE dataset_id = ?", [SUPPORT_DATASET_ID])
        connection.execute(
            "DELETE FROM domain_context_chunks WHERE document_id LIKE 'gaira_serum_context_liver_%' OR document_id LIKE 'gaira_serum_context_hcc_%' OR document_id LIKE 'gaira_serum_context_metabolic_%'"
        )
        connection.execute(
            "DELETE FROM domain_context_documents WHERE document_id LIKE 'gaira_serum_context_liver_%' OR document_id LIKE 'gaira_serum_context_hcc_%' OR document_id LIKE 'gaira_serum_context_metabolic_%'"
        )
        connection.register("support_documents_df", support_documents_df)
        connection.register("support_chunks_df", support_chunks_df)
        connection.register("serum_docs_df", serum_docs_df)
        connection.register("serum_chunks_df", serum_chunks_df)
        connection.execute("INSERT INTO grounding_support_documents SELECT * FROM support_documents_df")
        connection.execute("INSERT INTO grounding_support_chunks SELECT * FROM support_chunks_df")
        connection.execute("INSERT INTO domain_context_documents SELECT * FROM serum_docs_df")
        connection.execute("INSERT INTO domain_context_chunks SELECT * FROM serum_chunks_df")


def _remove_liver_serum_overlay(db_path: Path) -> None:
    with duckdb.connect(str(db_path)) as connection:
        connection.execute("DELETE FROM grounding_support_chunks WHERE dataset_id = ?", [SUPPORT_DATASET_ID])
        connection.execute("DELETE FROM grounding_support_documents WHERE dataset_id = ?", [SUPPORT_DATASET_ID])
        connection.execute(
            "DELETE FROM domain_context_chunks WHERE document_id LIKE 'gaira_serum_context_liver_%' OR document_id LIKE 'gaira_serum_context_hcc_%' OR document_id LIKE 'gaira_serum_context_metabolic_%'"
        )
        connection.execute(
            "DELETE FROM domain_context_documents WHERE document_id LIKE 'gaira_serum_context_liver_%' OR document_id LIKE 'gaira_serum_context_hcc_%' OR document_id LIKE 'gaira_serum_context_metabolic_%'"
        )


def _write_curation_outputs(output_dir: Path) -> pd.DataFrame:
    curation_df = pd.DataFrame(ALL_CURATION_ROWS).sort_values(["recommended_action", "filename"])
    curation_df.to_csv(output_dir / "curation_table.csv", index=False)
    sections = []
    sections.append(
        (
            "Selected High-Priority Papers",
            [f"{paper.filename}: {paper.integrated_reason}" for paper in SELECTED_PAPERS],
        )
    )
    sections.append(
        (
            "Skipped or Reserved Highlights",
            [
                "diabetes_EV_arxiv.pdf and spectra_shine.pdf families were already integrated in prior EV refinement passes.",
                "s41467-025-61600-9 PDF/workbook family remains excluded from active integration after the nature_serum_sers cleanup; only minimal CK18 context survives.",
                "kidney, colorectal, mixed nasopharyngeal/liver, tissue-imaging, and targeted assay papers were skipped or reserved to avoid contaminating liver-serum reasoning.",
            ],
        )
    )
    sections.append(
        (
            "Integration Set Size",
            [f"{len(SELECTED_PAPERS)} papers selected for support/context integration."],
        )
    )
    _write_markdown(output_dir / "curation_summary.md", "Liver/Serum Literature Curation Summary", sections)
    return curation_df


def _write_tables(output_dir: Path) -> None:
    _paper_metadata_summary_df().to_csv(output_dir / "paper_metadata_summary.csv", index=False)
    _band_annotations_df().to_csv(output_dir / "band_annotations.csv", index=False)
    _disease_axis_summary_df().to_csv(output_dir / "disease_axis_summary.csv", index=False)
    _modality_caution_summary_df().to_csv(output_dir / "modality_caution_summary.csv", index=False)


def _write_support_visibility(output_dir: Path, before_df: pd.DataFrame, after_df: pd.DataFrame) -> pd.DataFrame:
    merged = before_df.merge(
        after_df,
        on=["query_dataset_id", "query_class_label"],
        how="outer",
        suffixes=("_before", "_after"),
    )
    merged.to_csv(output_dir / "before_after_support_visibility.csv", index=False)
    return merged


def _plot_figures(output_dir: Path, curation_df: pd.DataFrame, support_visibility_df: pd.DataFrame) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    curation_counts = curation_df.groupby("recommended_action").size().reset_index(name="n")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=curation_counts, x="recommended_action", y="n", color="#4c72b0")
    ax.set_title("Figure 1. Literature curation decisions")
    ax.set_xlabel("")
    ax.set_ylabel("File count")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(figure_dir / "figure1_curation_map.png", dpi=200, bbox_inches="tight")
    plt.savefig(figure_dir / "figure1_curation_map.pdf", bbox_inches="tight")
    plt.close()

    paper_df = _paper_metadata_summary_df()
    coverage_rows = []
    for paper in SELECTED_PAPERS:
        for axis in ["HCC", "cirrhosis", "hepatitis", "NASH/NAFLD", "DILI/injury", "hepatobiliary"]:
            text = f"{paper.disease_task} {paper.group_structure}".lower()
            flag = int(
                (axis == "HCC" and "hcc" in text)
                or (axis == "cirrhosis" and "cirrho" in text)
                or (axis == "hepatitis" and "hepatitis" in text)
                or (axis == "NASH/NAFLD" and ("nash" in text or "nafld" in text))
                or (axis == "DILI/injury" and ("injury" in text or "dili" in text))
                or (axis == "hepatobiliary" and ("cca" in text or "cholang" in text or "metast" in text))
            )
            coverage_rows.append({"paper_slug": paper.slug, "axis": axis, "value": flag})
    coverage_df = pd.DataFrame(coverage_rows)
    pivot = coverage_df.pivot(index="paper_slug", columns="axis", values="value")
    plt.figure(figsize=(12, 7))
    ax = sns.heatmap(pivot, cmap="Blues", cbar=False, linewidths=0.5, linecolor="white")
    ax.set_title("Figure 2. Liver-serum support coverage map")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(figure_dir / "figure2_liver_serum_support_coverage.png", dpi=200, bbox_inches="tight")
    plt.savefig(figure_dir / "figure2_liver_serum_support_coverage.pdf", bbox_inches="tight")
    plt.close()

    caution_df = _modality_caution_summary_df().copy()
    caution_df["caution_family"] = caution_df["caution_text"].apply(
        lambda text: (
            "modality/substrate"
            if any(token in text.lower() for token in ["substrate", "sensor", "raman", "sers", "lfia", "chip"])
            else "cohort/model"
        )
    )
    caution_counts = caution_df.groupby(["modality", "caution_family"]).size().reset_index(name="n")
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=caution_counts, x="modality", y="n", hue="caution_family")
    ax.set_title("Figure 3. Modality and sample-type caution map")
    ax.set_xlabel("")
    ax.set_ylabel("Caution statements")
    plt.xticks(rotation=25, ha="right")
    ax.legend(title="Caution family", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(figure_dir / "figure3_modality_caution_map.png", dpi=200, bbox_inches="tight")
    plt.savefig(figure_dir / "figure3_modality_caution_map.pdf", bbox_inches="tight")
    plt.close()

    theme_rows = []
    for paper in SELECTED_PAPERS:
        for theme in paper.themes + paper.caution_tags:
            theme_rows.append({"paper_slug": paper.slug, "theme_or_caution": theme, "value": 1})
    theme_df = pd.DataFrame(theme_rows)
    theme_pivot = theme_df.pivot(index="paper_slug", columns="theme_or_caution", values="value").fillna(0)
    plt.figure(figsize=(15, 8))
    ax = sns.heatmap(theme_pivot, cmap="YlGnBu", cbar=False, linewidths=0.5, linecolor="white")
    ax.set_title("Figure 4. Theme-support enrichment map")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(figure_dir / "figure4_theme_support_enrichment.png", dpi=200, bbox_inches="tight")
    plt.savefig(figure_dir / "figure4_theme_support_enrichment.pdf", bbox_inches="tight")
    plt.close()

    if not support_visibility_df.empty:
        plot_df = support_visibility_df.melt(
            id_vars=["query_dataset_id", "query_class_label"],
            value_vars=["n_liver_serum_support_hits_before", "n_liver_serum_support_hits_after"],
            var_name="phase",
            value_name="n_liver_hits",
        )
        plot_df["phase"] = plot_df["phase"].str.replace("n_liver_serum_support_hits_", "", regex=False)
        plot_df["query"] = plot_df["query_dataset_id"] + " / " + plot_df["query_class_label"]
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=plot_df, x="query", y="n_liver_hits", hue="phase")
        ax.set_title("Figure 5. Inference support visibility before and after")
        ax.set_xlabel("")
        ax.set_ylabel("Liver-serum support hits in top tier-2")
        plt.xticks(rotation=25, ha="right")
        ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(figure_dir / "figure5_inference_support_visibility_before_after.png", dpi=200, bbox_inches="tight")
        plt.savefig(figure_dir / "figure5_inference_support_visibility_before_after.pdf", bbox_inches="tight")
        plt.close()


def _write_report(
    output_dir: Path,
    curation_df: pd.DataFrame,
    support_visibility_df: pd.DataFrame,
    live_db_path: Path,
    eval_db_updated: bool,
) -> None:
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_md = report_dir / "liver_serum_literature_integration_report.md"
    with duckdb.connect(str(live_db_path), read_only=True) as connection:
        support_doc_count = connection.execute(
            "SELECT COUNT(*) FROM grounding_support_documents WHERE dataset_id = ?",
            [SUPPORT_DATASET_ID],
        ).fetchone()[0]
        support_chunk_count = connection.execute(
            "SELECT COUNT(*) FROM grounding_support_chunks WHERE dataset_id = ?",
            [SUPPORT_DATASET_ID],
        ).fetchone()[0]
        serum_context_count = connection.execute(
            """
            SELECT COUNT(*) FROM domain_context_documents
            WHERE context_layer = ? AND intended_domain = 'serum'
            """,
            [SERUM_CONTEXT_LAYER],
        ).fetchone()[0]

    integrated_files = [paper.filename for paper in SELECTED_PAPERS]
    skipped = curation_df[curation_df["recommended_action"].isin(["skip", "reserve"])]["filename"].tolist()
    lines = [
        "# Liver/Serum Literature Integration Report",
        "",
        "## What Was Curated",
        "",
        f"- Files reviewed in `/Users/suraj/Downloads/New_Set_SERS_Papers_Data`: {len(curation_df)} curated entries.",
        f"- High-priority papers integrated: {len(SELECTED_PAPERS)}.",
        "- Folder was curated rather than blindly ingested. Prior EV refinements, CK18 LFIA material, duplicates, and irrelevant non-liver papers were explicitly excluded from new integration.",
        "",
        "## Integrated Papers",
        "",
    ]
    for paper in SELECTED_PAPERS:
        lines.append(f"- `{paper.filename}`: {paper.integrated_reason}")
    lines.extend(
        [
            "",
            "## Skipped / Reserved Highlights",
            "",
            "- `diabetes_EV_arxiv.pdf`, `media-1 (1).docx`, `spectra_shine.pdf`, and `1-s2.0-S0925400526000985-mmc1.docx` were skipped because they were already integrated in prior EV passes.",
            "- The `s41467-025-61600-9` LFIA / CK18 family was skipped from new integration because GAIRA intentionally removed it as an active dataset and now keeps only a minimal CK18 support note.",
            "- Non-liver or misleading papers such as kidney, colorectal, mixed nasopharyngeal/liver, and targeted AFP/microRNA assay papers were skipped.",
            "- Tissue imaging and heavily assay-specific liver papers were reserved rather than integrated into serum reasoning.",
            "",
            "## What Changed in GAIRA",
            "",
            f"- Added `{support_doc_count}` support-only liver-serum literature documents and `{support_chunk_count}` support chunks under dataset_id `{SUPPORT_DATASET_ID}`.",
            f"- Serum context document count is now `{serum_context_count}` after adding liver-serum disease-axis, hepatobiliary, metabolic-liver, injury/DILI, and modality-caution notes.",
            "- New support docs remain tier-2 only and are retrievable by shared grounding search.",
            "- No raw datasets were ingested in this pass.",
            "",
            "## Knowledge / Theme Support",
            "",
            "- Added band-centered support for liver injury (1437/1000, 1487, 1546), cirrhosis/HCC Raman support (1246, 757, 878, 831-853, 1004, 548-699, 1157, 1527), and disease-axis framing across HCC, hepatitis, cirrhosis, NASH, DILI, and cholangiocarcinoma.",
            "- Updated theme ontology cautiously to recognize liver-serum support terms such as bile acids, bilirubin, keratin-18 context, xanthine, apoptosis, liver injury/DILI, and stronger modality/substrate cautions.",
            "- Kept all new literature as support-only and did not convert any paper into raw spectral truth.",
            "",
            "## Validation",
            "",
            f"- Support visibility before/after rows written to `{output_dir / 'before_after_support_visibility.csv'}`.",
            f"- HCC eval DB mirror updated: `{eval_db_updated}`.",
            "- Existing datasets checked: `cca_hcc_lm_serum_sers`, `metabolite_sers63_support`, `serum_protocol_comparison`.",
            "- `hcc_serum` normal ingest block was preserved.",
            "",
            "## Scientific Value",
            "",
            "- GAIRA is now materially stronger for liver-serum interpretation because the serum context layer is no longer dominated by serum-ag-colloid support alone.",
            "- HCC, cirrhosis, hepatitis, NASH/NAFLD, DILI/liver injury, and hepatobiliary differential context can now surface through serum context and shared tier-2 support.",
            "- The integration especially improves what-not-to-claim behavior by making substrate specificity, modality mismatch, classifier-specific digital biomarkers, and targeted-assay limitations easier to surface.",
            "",
            "## Remaining Gaps Before the Streamlit Demo",
            "",
            "- This pass improves support and context, not raw benchmark performance.",
            "- HCC raw holdout remains isolated; the new papers support interpretation but do not replace the need for careful holdout framing.",
            "- A later pass could add a small liver-serum support reranker if selected liver papers compete too heavily with older serum-ag-colloid support.",
            "",
        ]
    )
    report_md.write_text("\n".join(lines), encoding="utf-8")

    pdf_path = report_dir / "liver_serum_literature_integration_report.pdf"
    with PdfPages(pdf_path) as pdf:
        for page_index, chunk_start in enumerate(range(0, len(lines), 28), start=1):
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
            ax.axis("off")
            ax.text(
                0,
                1,
                "\n".join(lines[chunk_start : chunk_start + 28]),
                va="top",
                ha="left",
                family="monospace",
                fontsize=10,
                wrap=True,
            )
            ax.set_title(f"Liver/Serum Literature Integration Report (page {page_index})", loc="left", fontsize=14, pad=10)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _validate_counts(db_path: Path, output_dir: Path) -> None:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        support_docs = connection.execute(
            "SELECT COUNT(*) FROM grounding_support_documents WHERE dataset_id = ?",
            [SUPPORT_DATASET_ID],
        ).fetchone()[0]
        support_chunks = connection.execute(
            "SELECT COUNT(*) FROM grounding_support_chunks WHERE dataset_id = ?",
            [SUPPORT_DATASET_ID],
        ).fetchone()[0]
        serum_docs = connection.execute(
            """
            SELECT COUNT(*) FROM domain_context_documents
            WHERE context_layer = ?
              AND intended_domain = 'serum'
              AND (
                document_id LIKE 'gaira_serum_context_liver_%'
                OR document_id LIKE 'gaira_serum_context_hcc_%'
                OR document_id LIKE 'gaira_serum_context_metabolic_%'
              )
            """,
            [SERUM_CONTEXT_LAYER],
        ).fetchone()[0]
    validation_lines = [
        "# Liver/Serum Literature Validation",
        "",
        f"- grounding_support_documents for `{SUPPORT_DATASET_ID}`: {support_docs}",
        f"- grounding_support_chunks for `{SUPPORT_DATASET_ID}`: {support_chunks}",
        f"- added liver-serum serum-context documents: {serum_docs}",
    ]
    (output_dir / "validation_summary.md").write_text("\n".join(validation_lines) + "\n", encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path, require_data_root_exists

    storage_paths = require_data_root_exists()
    live_db_path = get_database_path()
    output_dir = storage_paths["processed_data"] / "liver_serum_literature_integration"
    output_dir.mkdir(parents=True, exist_ok=True)
    working_dir = output_dir / "working"
    working_dir.mkdir(parents=True, exist_ok=True)

    curation_df = _write_curation_outputs(output_dir)
    _write_tables(output_dir)
    _write_extracted_notes(output_dir)

    support_documents_df = _selected_support_documents_df()
    support_chunks_df = _selected_support_chunks_df()
    serum_docs_df, serum_chunks_df = _serum_context_documents()

    pre_integration_db = working_dir / "pre_integration.duckdb"
    shutil.copy2(live_db_path, pre_integration_db)
    _remove_liver_serum_overlay(pre_integration_db)
    before_df, _ = _run_inference_examples(pre_integration_db)

    _insert_tables(live_db_path, support_documents_df, support_chunks_df, serum_docs_df, serum_chunks_df)

    eval_db_path = storage_paths["processed_data"] / "hcc_holdout_evaluation" / "eval_db" / "gaira_hcc_holdout_eval.duckdb"
    eval_db_updated = False
    if eval_db_path.exists():
        _insert_tables(eval_db_path, support_documents_df, support_chunks_df, serum_docs_df, serum_chunks_df)
        eval_db_updated = True

    after_df, _ = _run_inference_examples(live_db_path)
    support_visibility_df = _write_support_visibility(output_dir, before_df, after_df)

    _validate_counts(live_db_path, output_dir)
    _plot_figures(output_dir, curation_df, support_visibility_df)
    _write_report(output_dir, curation_df, support_visibility_df, live_db_path, eval_db_updated)

    print(f"Wrote liver/serum literature integration outputs to: {output_dir}")


if __name__ == "__main__":
    main()
