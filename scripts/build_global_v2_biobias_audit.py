from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "tmp/global_v2_dataset_audit"
OUT_ROOT = ROOT / "tmp/global_v2_dataset_audit_biobias"


SALIVA_SHARD_IDS = {
    "20428395",
    "20427957",
    "20427954",
    "20427951",
    "20427948",
    "20427945",
    "20427939",
    "20427936",
    "20427933",
    "20427930",
    "20427927",
    "20427924",
    "20427921",
    "20427918",
    "20427909",
    "20427906",
    "20427903",
    "20282238",
}


OVERRIDES = {
    "https://zenodo.org/records/4941488": {
        "biosample_type": "respiratory pathogen spectra / clinical isolate panel",
        "organism_type": "bacteria / human clinical isolates",
        "disease_state_task_type": "pathogen identification and strain typing",
        "cohort_size_estimate": "30 clinical isolates plus non-target mycoplasma panel",
        "spectra_count_estimate": "large single CSV matrix",
        "reconstruction_needed": "no",
        "bio_value_score": 4.8,
        "ingest_effort_score": 1.8,
        "global_v2_priority_score": 4.8,
        "final_category": "tier_A_ingest_now",
        "why": "Strong real biological diversity outside serum/EV, with pathogen/strain structure and directly reusable spectra.",
    },
    "https://zenodo.org/records/5947010": {
        "biosample_type": "faeces",
        "organism_type": "human",
        "disease_state_task_type": "coeliac disease vs control / gluten-free diet",
        "cohort_size_estimate": "small human cohort",
        "spectra_count_estimate": "dozens of TXT spectra",
        "reconstruction_needed": "no",
        "bio_value_score": 4.6,
        "ingest_effort_score": 2.0,
        "global_v2_priority_score": 4.7,
        "final_category": "tier_A_ingest_now",
        "why": "Real non-blood human biofluid cohort with disease labels and multiple spectra per subject.",
    },
    "https://figshare.com/articles/dataset/Raman_spectroscopic_techniques_to_detect_ovarian_cancer_biomarkers_in_blood_plasma/6744206": {
        "biosample_type": "blood plasma",
        "organism_type": "human",
        "disease_state_task_type": "ovarian cancer vs healthy/control plasma Raman cohort",
        "cohort_size_estimate": "human disease cohort",
        "spectra_count_estimate": "roughly 380+ TXT spectra visible from archive listing",
        "reconstruction_needed": "no",
        "bio_value_score": 4.9,
        "ingest_effort_score": 2.0,
        "global_v2_priority_score": 4.9,
        "final_category": "tier_A_ingest_now",
        "why": "High-value real plasma disease benchmark with many raw spectra and obvious shared-encoder utility.",
    },
    "https://figshare.com/articles/dataset/Surface-enhanced_SERS_and_tip-enhanced_TERS_Raman_scattering_in_label-free_characterization_of_erythrocyte_membranes_and_extracellular_vesicles_in_nano-scale_and_at_the_single-molecule_level_/24105993": {
        "biosample_type": "erythrocyte membranes and extracellular vesicles",
        "organism_type": "human biological material",
        "disease_state_task_type": "EV / membrane biochemical phenotyping",
        "cohort_size_estimate": "small focused study",
        "spectra_count_estimate": "roughly 50+ TXT spectra visible from archive listing",
        "reconstruction_needed": "no",
        "bio_value_score": 4.3,
        "ingest_effort_score": 2.1,
        "global_v2_priority_score": 4.3,
        "final_category": "tier_A_ingest_now",
        "why": "Real EV-associated raw spectra with a different biology regime than current GAIRA core assets.",
    },
    "https://zenodo.org/records/12740805": {
        "biosample_type": "bacterial plasmid / resistance gene fragment spectra",
        "organism_type": "bacteria / plasmid biological material",
        "disease_state_task_type": "antibiotic resistance fragment detection",
        "cohort_size_estimate": "experimental biological panel",
        "spectra_count_estimate": "zip-based Raman archive",
        "reconstruction_needed": "no",
        "bio_value_score": 4.1,
        "ingest_effort_score": 2.2,
        "global_v2_priority_score": 4.2,
        "final_category": "tier_A_ingest_now",
        "why": "Real pathogen-related spectra with distinct biological target space and raw files already packaged.",
    },
    "https://zenodo.org/records/19369604": {
        "biosample_type": "urine",
        "organism_type": "human",
        "disease_state_task_type": "ischemic stroke urine cohort",
        "cohort_size_estimate": "substantial human cohort likely embedded in RAR release",
        "spectra_count_estimate": "large archive; exact count pending extraction",
        "reconstruction_needed": "yes",
        "bio_value_score": 4.8,
        "ingest_effort_score": 4.6,
        "global_v2_priority_score": 4.5,
        "final_category": "tier_B_high_value_reconstruct",
        "why": "Exactly the kind of real human urine disease dataset Global v2 needs, despite heavy reconstruction cost.",
    },
    "https://zenodo.org/records/19109120": {
        "biosample_type": "urine",
        "organism_type": "human",
        "disease_state_task_type": "ischemic stroke urine cohort",
        "cohort_size_estimate": "superseded by newer record",
        "spectra_count_estimate": "unknown",
        "reconstruction_needed": "yes",
        "bio_value_score": 2.0,
        "ingest_effort_score": 4.8,
        "global_v2_priority_score": 1.7,
        "final_category": "reject",
        "why": "Superseded by the newer Zenodo 19369604 release; keep only the latest version alive.",
    },
    "https://zenodo.org/records/10851312": {
        "biosample_type": "stem cells / differentiation states",
        "organism_type": "cell line / stem-cell system",
        "disease_state_task_type": "cell-state trajectory across differentiation",
        "cohort_size_estimate": "six differentiation states",
        "spectra_count_estimate": "large RAR archive",
        "reconstruction_needed": "yes",
        "bio_value_score": 4.5,
        "ingest_effort_score": 4.2,
        "global_v2_priority_score": 4.2,
        "final_category": "tier_B_high_value_reconstruct",
        "why": "High-value cell-state trajectory dataset; biologically important even though the release is heavy and messy.",
    },
    "https://zenodo.org/records/5021659": {
        "biosample_type": "urinary tract pathogen spectra",
        "organism_type": "bacteria",
        "disease_state_task_type": "UTI pathogen identification from culture matrix / artificial urine",
        "cohort_size_estimate": "three-species panel",
        "spectra_count_estimate": "multi-file Origin project archive",
        "reconstruction_needed": "yes",
        "bio_value_score": 4.2,
        "ingest_effort_score": 4.0,
        "global_v2_priority_score": 4.0,
        "final_category": "tier_B_high_value_reconstruct",
        "why": "Real pathogen spectra in urine-relevant context; proprietary packaging is annoying but not a reason to drop it.",
    },
    "https://zenodo.org/records/5806264": {
        "biosample_type": "bacterial metabolism",
        "organism_type": "bacteria",
        "disease_state_task_type": "extracellular metabolism / optophysiology",
        "cohort_size_estimate": "focused microbial experiment",
        "spectra_count_estimate": "one large xlsx source-data workbook",
        "reconstruction_needed": "yes",
        "bio_value_score": 4.0,
        "ingest_effort_score": 3.6,
        "global_v2_priority_score": 3.9,
        "final_category": "tier_B_high_value_reconstruct",
        "why": "Biologically distinctive microbial metabolism archive worth keeping alive for non-human diversity.",
    },
    "https://zenodo.org/records/8130216": {
        "biosample_type": "cell-secreted metabolite / secretome",
        "organism_type": "cell line / tumor biology",
        "disease_state_task_type": "MTAP-deficient tumor secretome states",
        "cohort_size_estimate": "cell-state comparison experiment",
        "spectra_count_estimate": "large zip archive",
        "reconstruction_needed": "yes",
        "bio_value_score": 4.1,
        "ingest_effort_score": 3.8,
        "global_v2_priority_score": 3.9,
        "final_category": "tier_B_high_value_reconstruct",
        "why": "Not a classic patient cohort, but it adds meaningful biological state diversity around purine secretome biology.",
    },
    "https://figshare.com/articles/dataset/Raw_Raman_data_/26059145?file=47123702": {
        "biosample_type": "single-vesicle EV fractions",
        "organism_type": "EV biological material",
        "disease_state_task_type": "EV subpopulation heterogeneity",
        "cohort_size_estimate": "fractionated EV experiment",
        "spectra_count_estimate": "RAR archive; exact count pending extraction",
        "reconstruction_needed": "yes",
        "bio_value_score": 4.4,
        "ingest_effort_score": 3.8,
        "global_v2_priority_score": 4.1,
        "final_category": "tier_B_high_value_reconstruct",
        "why": "High-priority EV diversity target even though the current release needs unpacking and inspection.",
    },
    "https://figshare.com/articles/dataset/SERS_spectra_of_43_patients_with_ACS_xlsx/24747531?file=43481136": {
        "biosample_type": "cardiovascular disease biofluid SERS",
        "organism_type": "human",
        "disease_state_task_type": "acute coronary syndrome cohort",
        "cohort_size_estimate": "43 patients",
        "spectra_count_estimate": "43 workbook spectra plus metadata sheets",
        "reconstruction_needed": "yes",
        "bio_value_score": 4.0,
        "ingest_effort_score": 3.0,
        "global_v2_priority_score": 3.8,
        "final_category": "tier_B_high_value_reconstruct",
        "why": "Workbook-style but still real human disease spectra; should not be buried just because the packaging is plain.",
    },
    "https://figshare.com/articles/dataset/Suplementary_material_DIB_ACS_40_samples_xlsx/24564787?file=43183257": {
        "biosample_type": "cardiovascular disease biofluid SERS",
        "organism_type": "human",
        "disease_state_task_type": "acute coronary syndrome cohort",
        "cohort_size_estimate": "40 samples",
        "spectra_count_estimate": "40 workbook spectra plus metadata sheets",
        "reconstruction_needed": "yes",
        "bio_value_score": 3.9,
        "ingest_effort_score": 3.0,
        "global_v2_priority_score": 3.7,
        "final_category": "tier_B_high_value_reconstruct",
        "why": "Messy supplemental workbook, but it still looks like real human ACS spectra worth rebuilding.",
    },
    "https://figshare.com/articles/dataset/Additional_file_2_of_Combined_miRNA_and_SERS_urine_liquid_biopsy_for_the_point-of-care_diagnosis_and_molecular_stratification_of_bladder_cancer/19498603?file=34649167": {
        "biosample_type": "urine liquid biopsy",
        "organism_type": "human",
        "disease_state_task_type": "bladder cancer molecular stratification",
        "cohort_size_estimate": "not recoverable from uploaded workbook",
        "spectra_count_estimate": "no reusable spectral matrix in uploaded workbook",
        "reconstruction_needed": "no",
        "bio_value_score": 1.5,
        "ingest_effort_score": 4.5,
        "global_v2_priority_score": 1.2,
        "final_category": "reject",
        "why": "Biologically interesting paper, but the uploaded asset is miRNA target support, not a reusable spectral dataset.",
    },
    "https://figshare.com/articles/dataset/SERS_and_Raman_spectra_of_WT_and_mutant_cytochromes_c/4903091": {
        "biosample_type": "cytochrome c reference spectra",
        "organism_type": "pure standard / protein reference",
        "disease_state_task_type": "molecule-level reference variation",
        "cohort_size_estimate": "WT plus mutant panel",
        "spectra_count_estimate": "9 csv/txt files",
        "reconstruction_needed": "no",
        "bio_value_score": 2.8,
        "ingest_effort_score": 1.4,
        "global_v2_priority_score": 2.6,
        "final_category": "tier_C_support_or_grounding",
        "why": "Useful molecule-level support set, but not a core biological pretraining dataset.",
    },
    "https://zenodo.org/records/17035751": {
        "biosample_type": "adenine / IgG controlled assay",
        "organism_type": "pure standard / controlled assay",
        "disease_state_task_type": "calibration and controlled lateral-flow SERS",
        "cohort_size_estimate": "controlled experiment",
        "spectra_count_estimate": "30 files",
        "reconstruction_needed": "no",
        "bio_value_score": 2.6,
        "ingest_effort_score": 1.5,
        "global_v2_priority_score": 2.3,
        "final_category": "tier_C_support_or_grounding",
        "why": "Good controlled grounding asset, but already in GAIRA and not a new biological diversity win.",
    },
    "https://zenodo.org/records/14294417": {
        "biosample_type": "small-molecule fingerprint support",
        "organism_type": "pure standard",
        "disease_state_task_type": "metabolite fingerprint support",
        "cohort_size_estimate": "small controlled set",
        "spectra_count_estimate": "PDF-only support release",
        "reconstruction_needed": "no",
        "bio_value_score": 2.4,
        "ingest_effort_score": 1.0,
        "global_v2_priority_score": 2.0,
        "final_category": "tier_C_support_or_grounding",
        "why": "Keep as reserve support only; already represented in GAIRA support assets.",
    },
    "https://zenodo.org/records/10055068": {
        "biosample_type": "metabolite fingerprint support",
        "organism_type": "pure standard",
        "disease_state_task_type": "fingerprint compilation",
        "cohort_size_estimate": "support-only",
        "spectra_count_estimate": "PDF-only support release",
        "reconstruction_needed": "no",
        "bio_value_score": 2.2,
        "ingest_effort_score": 1.0,
        "global_v2_priority_score": 1.9,
        "final_category": "tier_C_support_or_grounding",
        "why": "Potentially useful reserve support, but not core real-bio training material.",
    },
    "https://zenodo.org/records/5806132": {
        "biosample_type": "drug detection / bioanalytical solutions",
        "organism_type": "small molecules / bioanalytical mixtures",
        "disease_state_task_type": "concentration-response classification",
        "cohort_size_estimate": "experimental chemical panel",
        "spectra_count_estimate": "large csv/image archive",
        "reconstruction_needed": "yes",
        "bio_value_score": 2.7,
        "ingest_effort_score": 3.2,
        "global_v2_priority_score": 2.4,
        "final_category": "tier_C_support_or_grounding",
        "why": "Not core biology, but worth keeping as auxiliary quantitative support because real spectra are present.",
    },
    "https://zenodo.org/records/14755439": {
        "biosample_type": "head and neck cancer / infection biosensor support",
        "organism_type": "human bioanalytical samples",
        "disease_state_task_type": "cancer and infection biomarker sensing",
        "cohort_size_estimate": "unclear from support zip",
        "spectra_count_estimate": "small number of panel txt files visible",
        "reconstruction_needed": "yes",
        "bio_value_score": 2.8,
        "ingest_effort_score": 2.8,
        "global_v2_priority_score": 2.5,
        "final_category": "tier_C_support_or_grounding",
        "why": "Biologically relevant topic, but the released payload looks more like support panels than a full cohort archive.",
    },
    "https://zenodo.org/records/3994312": {
        "biosample_type": "substrate/materials",
        "organism_type": "materials study",
        "disease_state_task_type": "substrate characterization",
        "cohort_size_estimate": "not a biological cohort",
        "spectra_count_estimate": "figure zips",
        "reconstruction_needed": "yes",
        "bio_value_score": 1.0,
        "ingest_effort_score": 2.5,
        "global_v2_priority_score": 0.9,
        "final_category": "tier_D_method_only",
        "why": "Numeric files exist, but the released payload is fundamentally a substrate paper rather than a biological dataset.",
    },
    "https://zenodo.org/records/18670010": {
        "biosample_type": "tear dopamine assay",
        "organism_type": "biofluid assay / targeted molecule",
        "disease_state_task_type": "targeted tear analyte detection",
        "cohort_size_estimate": "focused assay study",
        "spectra_count_estimate": "xlsx/image figure sources",
        "reconstruction_needed": "yes",
        "bio_value_score": 2.0,
        "ingest_effort_score": 2.5,
        "global_v2_priority_score": 1.8,
        "final_category": "tier_C_support_or_grounding",
        "why": "Real biofluid context, but too targeted and assay-centric for core Global v2 pretraining.",
    },
    "https://zenodo.org/records/17023716": {
        "biosample_type": "antibody sensing substrate study",
        "organism_type": "materials / sensor platform",
        "disease_state_task_type": "method development",
        "cohort_size_estimate": "not a biological cohort",
        "spectra_count_estimate": "zip archives dominated by characterization outputs",
        "reconstruction_needed": "yes",
        "bio_value_score": 1.0,
        "ingest_effort_score": 2.6,
        "global_v2_priority_score": 0.9,
        "final_category": "tier_D_method_only",
        "why": "Method/materials oriented, not a strong biological training asset.",
    },
    "https://zenodo.org/records/18284194": {
        "biosample_type": "drug / neurotransmitter assay",
        "organism_type": "small molecules",
        "disease_state_task_type": "quantification assay",
        "cohort_size_estimate": "targeted assay study",
        "spectra_count_estimate": "xlsx/image source files",
        "reconstruction_needed": "yes",
        "bio_value_score": 1.8,
        "ingest_effort_score": 2.3,
        "global_v2_priority_score": 1.6,
        "final_category": "tier_C_support_or_grounding",
        "why": "Spectral support is real, but biology is too narrow and assay-specific for core training.",
    },
    "https://zenodo.org/records/7523579": {
        "biosample_type": "small-molecule correlation study",
        "organism_type": "pure standards",
        "disease_state_task_type": "molecule correlation / dimer study",
        "cohort_size_estimate": "not a biological cohort",
        "spectra_count_estimate": "one large zip",
        "reconstruction_needed": "yes",
        "bio_value_score": 1.7,
        "ingest_effort_score": 2.8,
        "global_v2_priority_score": 1.6,
        "final_category": "tier_C_support_or_grounding",
        "why": "Keep only as reserve chemistry support; not a real biological diversity asset.",
    },
    "https://figshare.com/articles/dataset/ERCC/20406102": {
        "biosample_type": "saliva sEV cohort index",
        "organism_type": "human",
        "disease_state_task_type": "metadata-only cohort stub",
        "cohort_size_estimate": "metadata record only",
        "spectra_count_estimate": "0 downloadable files",
        "reconstruction_needed": "no",
        "bio_value_score": 0.8,
        "ingest_effort_score": 5.0,
        "global_v2_priority_score": 0.5,
        "final_category": "reject",
        "why": "Metadata record with no files; keep the actual patient shard records instead.",
    },
    "https://figshare.com/articles/dataset/Highly_sensitive_detection_of_influenza_virus_with_SERS_aptasensor/8044163": {
        "biosample_type": "influenza aptasensor method",
        "organism_type": "viral biomarker assay",
        "disease_state_task_type": "aptasensor demonstration",
        "cohort_size_estimate": "not a reusable cohort",
        "spectra_count_estimate": "image-only release",
        "reconstruction_needed": "no",
        "bio_value_score": 0.9,
        "ingest_effort_score": 4.8,
        "global_v2_priority_score": 0.6,
        "final_category": "reject",
        "why": "Biologically relevant topic, but the posted files are not reusable spectra.",
    },
    "https://figshare.com/articles/dataset/DataSheet1_Magnetically_Enhanced_Liquid_SERS_for_Ultrasensitive_Analysis_of_Bacterial_and_SARS-CoV-2_Biomarkers_PDF/16697359?file=30917875": {
        "final_category": "tier_D_method_only",
        "why": "Useful method reference only; PDF support without reusable spectra.",
    },
    "https://figshare.com/articles/dataset/DataSheet1_A_dual-amplification_strategy-intergated_SERS_biosensor_for_ultrasensitive_hepatocellular_carcinoma-related_telomerase_activity_detection_docx/21896907?file=38839281": {
        "final_category": "tier_D_method_only",
        "why": "Method note only; no reusable spectral dataset in the uploaded asset.",
    },
    "https://figshare.com/articles/dataset/Data_Sheet_1_Polycaprolactone-Based_Porous_CaCO3_and_Ag_Nanoparticle_Modified_Scaffolds_as_a_SERS_Platform_With_Molecule-Specific_Adsorption_pdf/11568420?file=20804994": {
        "final_category": "tier_D_method_only",
        "why": "Substrate/materials PDF, not a biological dataset.",
    },
    "https://figshare.com/articles/dataset/Table_1_Polycaprolactone-Based_Porous_CaCO3_and_Ag_Nanoparticle_Modified_Scaffolds_as_a_SERS_Platform_With_Molecule-Specific_Adsorption_pdf/11568423?file=20804997": {
        "final_category": "tier_D_method_only",
        "why": "Substrate/materials PDF, not a biological dataset.",
    },
    "https://figshare.com/articles/dataset/HOTSPOT_STABILIZATION_OF_GOLD_NANOPARTICLES_FOR_APPLICATION_OF_QUANTITATIVE_SERS_IN_BIOANALYTICAL_SYSTEMS/11390760": {
        "final_category": "tier_C_support_or_grounding",
        "why": "Keep as quantitative bioanalytical support, not core real-bio pretraining.",
    },
    "https://figshare.com/articles/dataset/_PLS_DA_of_NA_SERS_specificity_and_sensitivity_in_discriminating_M_pneumoniae_strains_/493381?file=823019": {
        "final_category": "tier_D_method_only",
        "why": "Model-statistics supplement, not a spectral training dataset.",
    },
    "https://figshare.com/articles/dataset/Data_Sheet_1_SERS-Based_Evaluation_of_the_DNA_Methylation_Pattern_Associated_With_Progression_in_Clonal_Leukemogenesis_of_Down_Syndrome_PDF/15042300?file=28924107": {
        "final_category": "tier_D_method_only",
        "why": "Disease topic is real, but the posted file is PDF-only support.",
    },
    "https://figshare.com/articles/dataset/Data_Sheet_1_Applications_of_Surface-Enhanced_Raman_Scattering_in_Biochemical_and_Medical_Analysis_PDF/14552937?file=27920859": {
        "final_category": "tier_D_method_only",
        "why": "General review/support PDF only.",
    },
    "https://figshare.com/articles/dataset/Data_Sheet_1_Rapid_Quantitative_High-Sensitive_Detection_of_Escherichia_coli_O157_H7_by_Gold-Shell_Silica-Core_Nanospheres-Based_Surface-Enhanced_Raman_Scattering_Lateral_Flow_Immunoassay_docx/13199414?file=25409663": {
        "final_category": "tier_D_method_only",
        "why": "Docx support file only.",
    },
    "https://figshare.com/articles/dataset/DataSheet_1_SERS_Sensing_of_Bacterial_Endotoxin_on_Gold_Nanoparticles_pdf/16764130?file=31020013": {
        "final_category": "tier_D_method_only",
        "why": "Support PDF only.",
    },
    "https://figshare.com/articles/dataset/Source_Data_file_xlsxDataset_ArticleNatureComm_Dallarietal_2024/26411992?file=48039661": {
        "final_category": "tier_D_method_only",
        "why": "Figure-wise source workbook for a materials paper, not a biological cohort.",
    },
    "https://figshare.com/articles/dataset/_Method_for_Assessing_the_Reliability_of_Molecular_Diagnostics_Based_on_Multiplexed_SERS_Coded_Nanoparticles_/686466": {
        "final_category": "tier_D_method_only",
        "why": "Method-support release, not a reusable spectrum archive.",
    },
    "https://figshare.com/articles/dataset/Raman_shift_and_putative_peak_assignments_of_the_SERS_spectra_from_conidia_of_the_common_causative_pathogens_used_in_this_study_/11989566?file=22020294": {
        "final_category": "tier_C_support_or_grounding",
        "why": "Peak-assignment support for pathogen spectra; useful reserve support, not core training data.",
    },
    "https://figshare.com/articles/dataset/Raman_shift_and_putative_peak_assignments_of_the_SERS_spectra_from_mycelia_of_the_common_causative_pathogens_used_in_this_study_/11989560?file=22020288": {
        "final_category": "tier_C_support_or_grounding",
        "why": "Peak-assignment support for pathogen spectra; useful reserve support, not core training data.",
    },
    "https://figshare.com/articles/dataset/Diagnostic_performance_of_PLS-DA_models_in_predicting_high_and_low_groups_of_conventional_quantitative_parameters_SER_ADC_SUV_/3882102?file=6084882": {
        "final_category": "tier_D_method_only",
        "why": "Model-performance table only.",
    },
    "https://figshare.com/articles/dataset/DataSheet1_AuNPs_MIL-101_Cr_as_a_SERS-Active_Substrate_for_Sensitive_Detection_of_VOCs_docx/20100728?file=35950469": {
        "final_category": "tier_D_method_only",
        "why": "Substrate/method docx only.",
    },
    "https://figshare.com/articles/dataset/DFT-Based_Theoretical_Study_on_Label-Free_SERS_Detection_of_Type_B_Fumonisins_New_Insights_into_Molecular_Substrate_Interactions_and_Quantification_Strategies/28565671?file=52894126": {
        "final_category": "tier_C_support_or_grounding",
        "why": "Keep as toxin/biomolecule support rather than core biological pretraining.",
    },
    "https://figshare.com/articles/dataset/_Cross_validated_PLS_DA_modeling_statistics_for_the_prediction_performance_for_NA_SERS_typing_of_individual_type_1_and_2_M_pneumoniae_clinical_isolates_/1467505?file=2154588": {
        "final_category": "tier_D_method_only",
        "why": "Model-statistics table only.",
    },
    "https://zenodo.org/records/16895315": {
        "final_category": "tier_D_method_only",
        "why": "Autoencoder paper only; no raw spectra released.",
    },
    "https://zenodo.org/records/17052624": {
        "final_category": "tier_D_method_only",
        "why": "Analysis paper only; useful conceptually, not a dataset ingest target.",
    },
    "https://zenodo.org/records/16912956": {
        "final_category": "tier_D_method_only",
        "why": "Method paper only; no reusable spectra package.",
    },
    "https://zenodo.org/records/18670010": {
        "final_category": "tier_C_support_or_grounding",
    },
    "https://pubs.acs.org/doi/10.1021/acsomega.4c11078": {
        "final_category": "tier_D_method_only",
        "why": "Publisher page was not directly retrievable here; keep as paper reference only.",
    },
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


def extract_file_count(size_text: str) -> int | None:
    m = re.match(r"(\d+)\s+files?", size_text or "")
    return int(m.group(1)) if m else None


def source_type(url: str) -> str:
    if "zenodo.org" in url:
        return "zenodo"
    if "figshare.com" in url:
        return "figshare"
    if "acs.org" in url:
        return "paper"
    return "other"


def infer_biosample(row: dict[str, str]) -> str:
    domain = row["domain_category"]
    mapping = {
        "serum": "serum / plasma",
        "saliva": "saliva",
        "urine": "urine",
        "EV": "extracellular vesicles",
        "cells": "cells",
        "pathogen": "pathogen / bacteria / microbial biomaterial",
        "molecule_reference": "controlled biomolecule / pure standard",
        "interlab / reproducibility": "biological cohort with protocol / reproducibility value",
        "substrate / materials": "substrate / materials",
        "other": "other",
    }
    return mapping.get(domain, domain)


def infer_organism(row: dict[str, str]) -> str:
    entity = row["entity_type"]
    mapping = {
        "human": "human",
        "animal": "animal",
        "synthetic vs pure standard": "pure standard / synthetic",
        "unknown": "unknown biological source",
    }
    return mapping.get(entity, entity or "unknown")


def infer_task(row: dict[str, str]) -> str:
    title = row["title"].lower()
    if "stroke" in title:
        return "ischemic stroke disease classification"
    if "ovarian" in title:
        return "ovarian cancer detection"
    if "acs" in title:
        return "acute coronary syndrome classification"
    if "coeliac" in title:
        return "coeliac disease / diet state"
    if "mycoplasma" in title:
        return "pathogen specificity and strain typing"
    if "stem cell" in title or "differentiation" in title:
        return "cell differentiation trajectory"
    if "vesicle" in title or "ev" in row["domain_category"].lower():
        return "EV phenotyping"
    if "bladder cancer" in title:
        return "bladder cancer liquid biopsy"
    if "cytochrome" in title:
        return "molecule reference variation"
    return row["label_type"] or "biological classification / regression"


def recoverable(row: dict[str, str]) -> str:
    if row["raw_spectra_present"] == "yes":
        return "yes"
    if any(ext in row["file_types_available"] for ext in ["xlsx", "zip", "rar", "mat", "csv", "txt", "opj", "opju"]):
        return "maybe"
    return "no"


def overlap_bucket(dedup_row: dict[str, str]) -> str:
    status = dedup_row["duplicate_status"]
    if status == "already_ingested_exact":
        return "strong"
    if status in {"scope_overlap", "fragmented_cohort_component"}:
        return "partial"
    return "none"


def parse_spectra_estimate(row: dict[str, str]) -> str:
    note = row.get("archive_contents_note", "")
    title = row["title"]
    file_count = extract_file_count(row.get("approx_dataset_size", ""))
    if "43 patients" in title:
        return "about 43 spectra or more"
    if "40_samples" in title:
        return "about 40 spectra or more"
    m = re.search(r"\(\+(\d+)\s+more\)", note)
    if m:
        return f"at least {12 + int(m.group(1))} visible archive members"
    if file_count:
        return str(file_count)
    return "unknown"


def parse_cohort_estimate(row: dict[str, str]) -> str:
    title = row["title"]
    if "43 patients" in title:
        return "43 patients"
    if "40_samples" in title:
        return "40 samples"
    if any(pid in row["source_url"] for pid in SALIVA_SHARD_IDS):
        if "Healthy" in title:
            return "single control patient shard"
        return "single patient shard"
    if "Patient" in title:
        return "single patient shard"
    if "ovarian" in title.lower():
        return "multi-subject human plasma cohort"
    if "coeliac" in title.lower():
        return "small human cohort"
    return "unknown"


def default_scores(row: dict[str, str], dedup_row: dict[str, str]) -> tuple[float, float, float]:
    bio = 0.8
    if row["raw_spectra_present"] == "yes":
        bio += 1.2
    if row["domain_category"] in {"serum", "urine", "saliva", "EV", "cells", "pathogen", "interlab / reproducibility"}:
        bio += 1.6
    if "disease class" in row["label_type"] or "stage / trajectory" in row["label_type"] or "patient ID" in row["label_type"]:
        bio += 0.9
    if any(pid in row["source_url"] for pid in SALIVA_SHARD_IDS):
        bio += 0.7
    if row["domain_category"] == "molecule_reference":
        bio = min(bio, 2.8)
    if row["raw_spectra_present"] == "no":
        bio -= 1.2
    bio = max(0.0, min(5.0, bio))

    effort = 1.2
    if any(ext in row["file_types_available"] for ext in ["rar", "opj", "opju"]):
        effort += 2.0
    elif any(ext in row["file_types_available"] for ext in ["zip", "xlsx"]):
        effort += 1.0
    if overlap_bucket(dedup_row) == "strong":
        effort += 0.4
    effort = max(0.5, min(5.0, effort))

    priority = bio - 0.25 * (effort - 1.0)
    if overlap_bucket(dedup_row) == "strong":
        priority -= 0.7
    elif overlap_bucket(dedup_row) == "partial":
        priority -= 0.2
    priority = max(0.0, min(5.0, priority))
    return round(bio, 2), round(effort, 2), round(priority, 2)


def default_category(row: dict[str, str], dedup_row: dict[str, str], bio: float, effort: float) -> str:
    if overlap_bucket(dedup_row) == "strong" and row["domain_category"] == "molecule_reference":
        return "tier_C_support_or_grounding"
    if row["raw_spectra_present"] == "no":
        if row["domain_category"] == "molecule_reference":
            return "tier_C_support_or_grounding"
        return "tier_D_method_only"
    if any(pid in row["source_url"] for pid in SALIVA_SHARD_IDS):
        return "tier_B_high_value_reconstruct"
    if row["domain_category"] in {"serum", "urine", "saliva", "EV", "cells", "pathogen", "interlab / reproducibility"}:
        if effort <= 2.4 and bio >= 3.8:
            return "tier_A_ingest_now"
        return "tier_B_high_value_reconstruct"
    if row["domain_category"] == "molecule_reference":
        return "tier_C_support_or_grounding"
    return "tier_D_method_only"


def main() -> None:
    inventory = read_csv(SRC_ROOT / "tables/global_v2_candidate_dataset_inventory.csv")
    dedup_rows = {r["source_url"]: r for r in read_csv(SRC_ROOT / "tables/global_v2_deduplication_check.csv")}
    inventory_out = []
    scoring_out = []
    dedup_out = []
    priority_out = []

    for row in inventory:
        url = row["source_url"]
        dedup_row = dedup_rows.get(url, {"duplicate_status": "none", "overlap_existing_dataset": ""})
        bio, effort, priority = default_scores(row, dedup_row)
        category = default_category(row, dedup_row, bio, effort)
        over = OVERRIDES.get(url, {})
        if any(pid in url for pid in SALIVA_SHARD_IDS):
            over = {
                **over,
                "biosample_type": "saliva small-EV cohort shard",
                "organism_type": "human",
                "disease_state_task_type": "gastric cancer saliva EV cohort component",
                "cohort_size_estimate": "single patient shard",
                "spectra_count_estimate": row.get("approx_dataset_size", "").split(";")[0] or "dozens of txt spectra",
                "reconstruction_needed": "yes",
                "bio_value_score": 4.4,
                "ingest_effort_score": 3.6,
                "global_v2_priority_score": 3.8,
                "final_category": "tier_B_high_value_reconstruct",
                "why": "Real saliva EV patient spectra; high-value only when reconstructed as one cohort rather than ingested shard-by-shard.",
                "overlap_note_override": "fragmented_cohort_component",
            }
        bio = over.get("bio_value_score", bio)
        effort = over.get("ingest_effort_score", effort)
        priority = over.get("global_v2_priority_score", priority)
        category = over.get("final_category", category)

        downloadable = row["file_types_available"] or "none listed"
        if row.get("archive_contents_note"):
            downloadable = f"{downloadable}; {row['archive_contents_note']}"

        overlap = over.get("overlap_note_override", overlap_bucket(dedup_row))
        biosample = over.get("biosample_type", infer_biosample(row))
        organism = over.get("organism_type", infer_organism(row))
        task = over.get("disease_state_task_type", infer_task(row))
        cohort = over.get("cohort_size_estimate", parse_cohort_estimate(row))
        spectra = over.get("spectra_count_estimate", parse_spectra_estimate(row))
        reconstruction = over.get("reconstruction_needed", "yes" if recoverable(row) == "yes" and any(ext in row["file_types_available"] for ext in ["zip", "rar", "xlsx", "opj", "opju"]) else "no")
        rationale = over.get("why", "")

        inventory_out.append(
            {
                "source_url": url,
                "title": row["title"],
                "source_type": source_type(url),
                "actual_downloadable_assets_present": downloadable,
                "raw_spectra_appear_present": row["raw_spectra_present"],
                "spectra_recoverable_from_release": recoverable(row),
                "biosample_type": biosample,
                "organism_type": organism,
                "disease_state_task_type": task,
                "labels_available": row["label_type"],
                "cohort_size_estimate": cohort,
                "spectra_count_estimate": spectra,
                "reconstruction_needed": reconstruction,
                "overlap_with_existing_gaira": overlap,
                "final_category": category,
                "notes": rationale or row.get("metadata_quality_note", ""),
            }
        )
        scoring_out.append(
            {
                "source_url": url,
                "title": row["title"],
                "biological_realism_score": bio,
                "biological_diversity_contribution_score": round(min(5.0, bio + (0.5 if row["domain_category"] in {"urine", "saliva", "pathogen", "cells"} else 0.0)), 2),
                "spectral_reuse_potential_score": 5.0 if row["raw_spectra_present"] == "yes" else 2.0 if recoverable(row) == "maybe" else 0.5,
                "label_richness_score": round(min(5.0, 1.0 + 0.8 * row["label_type"].count(";") + (1.0 if row["label_type"] != "unlabeled" else 0.0)), 2),
                "cross_domain_shared_encoder_value_score": round(min(5.0, bio + (0.4 if row["domain_category"] in {"pathogen", "cells", "urine", "saliva"} else 0.0)), 2),
                "reconstruction_feasibility_score": round(5.0 - effort if effort <= 5 else 0.0, 2),
                "immediate_ingest_difficulty_score": effort,
                "redundancy_penalty_score": 2.5 if overlap == "strong" else 1.0 if overlap == "partial" else 0.0,
                "bio_value_score": bio,
                "ingest_effort_score": effort,
                "global_v2_priority_score": priority,
                "final_category": category,
                "rationale": rationale,
            }
        )
        dedup_out.append(
            {
                "source_url": url,
                "title": row["title"],
                "already_ingested": dedup_row.get("already_ingested", "no"),
                "overlap_with_existing_gaira": overlap,
                "deduplication_status": dedup_row.get("duplicate_status", "none"),
                "overlap_dataset_id_or_family": dedup_row.get("overlap_existing_dataset", ""),
                "deduplication_note": rationale if overlap != "none" else "",
            }
        )
        priority_out.append(
            {
                "priority_rank": "",
                "source_url": url,
                "title": row["title"],
                "biosample_type": biosample,
                "organism_type": organism,
                "bio_value_score": bio,
                "ingest_effort_score": effort,
                "global_v2_priority_score": priority,
                "final_category": category,
                "reconstruction_needed": reconstruction,
                "overlap_with_existing_gaira": overlap,
                "rationale": rationale,
            }
        )

    category_order = {
        "tier_A_ingest_now": 0,
        "tier_B_high_value_reconstruct": 1,
        "tier_C_support_or_grounding": 2,
        "tier_D_method_only": 3,
        "reject": 4,
    }
    priority_out.sort(key=lambda r: (category_order[r["final_category"]], -float(r["global_v2_priority_score"]), -float(r["bio_value_score"]), r["title"]))
    for i, row in enumerate(priority_out, start=1):
        row["priority_rank"] = str(i)

    shortlist = [
        {
            "shortlist_rank": 1,
            "dataset_name": "Ovarian cancer plasma Raman cohort",
            "source_url": "https://figshare.com/articles/dataset/Raman_spectroscopic_techniques_to_detect_ovarian_cancer_biomarkers_in_blood_plasma/6744206",
            "why_it_matters": "Large real human plasma disease archive with many raw spectra and immediate benchmark value.",
            "recommended_bucket": "top_5_immediate_additions",
        },
        {
            "shortlist_rank": 2,
            "dataset_name": "Mycoplasma clinical isolate SERS panel",
            "source_url": "https://zenodo.org/records/4941488",
            "why_it_matters": "Strong pathogen/strain diversity far from current serum/EV emphasis.",
            "recommended_bucket": "top_5_immediate_additions",
        },
        {
            "shortlist_rank": 3,
            "dataset_name": "Coeliac faecal SERS cohort",
            "source_url": "https://zenodo.org/records/5947010",
            "why_it_matters": "Real non-blood human biofluid cohort with disease labels.",
            "recommended_bucket": "top_5_immediate_additions",
        },
        {
            "shortlist_rank": 4,
            "dataset_name": "RBC membrane and EV SERS/TERS archive",
            "source_url": "https://figshare.com/articles/dataset/Surface-enhanced_SERS_and_tip-enhanced_TERS_Raman_scattering_in_label-free_characterization_of_erythrocyte_membranes_and_extracellular_vesicles_in_nano-scale_and_at_the_single-molecule_level_/24105993",
            "why_it_matters": "Adds membrane/EV biology and raw text spectra in a compact release.",
            "recommended_bucket": "top_5_immediate_additions",
        },
        {
            "shortlist_rank": 5,
            "dataset_name": "Beta-lactam resistance plasmid Raman panel",
            "source_url": "https://zenodo.org/records/12740805",
            "why_it_matters": "Pathogen-related biological spectra with clear labels and different biology from current GAIRA assets.",
            "recommended_bucket": "top_5_immediate_additions",
        },
        {
            "shortlist_rank": 6,
            "dataset_name": "Ischemic stroke urine SERS cohort",
            "source_url": "https://zenodo.org/records/19369604",
            "why_it_matters": "High-value human urine disease dataset; reconstruction cost is high but worth it.",
            "recommended_bucket": "next_5_reconstruction_targets",
        },
        {
            "shortlist_rank": 7,
            "dataset_name": "UCLA saliva sEV gastric-cancer shard cohort",
            "source_url": "https://figshare.com/articles/dataset/GC_Patient_1_UG3/20282238",
            "why_it_matters": "Fragmented but biologically rich saliva/EV cohort; should be reconstructed as a combined dataset rather than rejected per shard.",
            "recommended_bucket": "next_5_reconstruction_targets",
        },
        {
            "shortlist_rank": 8,
            "dataset_name": "Stem-cell differentiation SERS trajectory",
            "source_url": "https://zenodo.org/records/10851312",
            "why_it_matters": "Adds cell-state trajectory learning, which the conservative pass underweighted.",
            "recommended_bucket": "next_5_reconstruction_targets",
        },
        {
            "shortlist_rank": 9,
            "dataset_name": "UTI pathogen OPJ archive",
            "source_url": "https://zenodo.org/records/5021659",
            "why_it_matters": "Real urine/pathogen spectra; inconvenient packaging but biologically strong.",
            "recommended_bucket": "next_5_reconstruction_targets",
        },
        {
            "shortlist_rank": 10,
            "dataset_name": "Single-vesicle EV raw Raman archive",
            "source_url": "https://figshare.com/articles/dataset/Raw_Raman_data_/26059145?file=47123702",
            "why_it_matters": "High-value EV heterogeneity target that broadens Global v2 beyond current EV datasets.",
            "recommended_bucket": "next_5_reconstruction_targets",
        },
        {
            "shortlist_rank": 11,
            "dataset_name": "Cytochrome c Raman/SERS references",
            "source_url": "https://figshare.com/articles/dataset/SERS_and_Raman_spectra_of_WT_and_mutant_cytochromes_c/4903091",
            "why_it_matters": "Strong reserve molecule-level grounding support.",
            "recommended_bucket": "support_reserve",
        },
        {
            "shortlist_rank": 12,
            "dataset_name": "Adenine lateral-flow control archive",
            "source_url": "https://zenodo.org/records/17035751",
            "why_it_matters": "Already-ingested calibration/support asset worth keeping in reserve.",
            "recommended_bucket": "support_reserve",
        },
    ]

    write_csv(OUT_ROOT / "tables/global_v2_biobias_candidate_inventory.csv", inventory_out)
    write_csv(OUT_ROOT / "tables/global_v2_biobias_scoring.csv", scoring_out)
    write_csv(OUT_ROOT / "tables/global_v2_biobias_deduplication.csv", dedup_out)
    write_csv(OUT_ROOT / "tables/global_v2_biobias_priority_list.csv", priority_out)
    write_csv(OUT_ROOT / "tables/global_v2_biobias_realbio_shortlist.csv", shortlist)

    counts = Counter(r["final_category"] for r in priority_out)
    tier_a = [r for r in priority_out if r["final_category"] == "tier_A_ingest_now"]
    tier_b = [r for r in priority_out if r["final_category"] == "tier_B_high_value_reconstruct"]
    tier_c = [r for r in priority_out if r["final_category"] == "tier_C_support_or_grounding"]
    tier_d = [r for r in priority_out if r["final_category"] == "tier_D_method_only"]
    rejected = [r for r in priority_out if r["final_category"] == "reject"]
    underrated = [
        "Zenodo 19369604 ischemic-stroke urine was downgraded mainly for cleanup cost in the prior pass, but biologically it is one of the strongest human disease additions.",
        "Zenodo 10851312 stem-cell differentiation was too heavily penalized for being a large RAR even though Global v2 needs cell-state trajectories.",
        "Zenodo 5021659 UTI pathogen spectra and Zenodo 5806264 bacterial metabolism are inconvenient archives, but they materially expand pathogen biology.",
        "The UCLA saliva sEV Figshare shards were previously treated too punitively record-by-record; biologically they form a valuable reconstructable cohort.",
    ]
    zenodo_strong = [
        "4941488 Mycoplasma clinical isolates",
        "5947010 coeliac faecal cohort",
        "19369604 ischemic-stroke urine",
        "10851312 stem-cell differentiation",
        "12740805 beta-lactam resistance plasmid",
        "5021659 UTI pathogen spectra",
        "5806264 bacterial metabolism",
        "8130216 tumor secretome purine states",
    ]

    lines = []
    lines.append("# GAIRA Global v2 Biobias Audit")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Audited candidates: {len(priority_out)}")
    for key in ["tier_A_ingest_now", "tier_B_high_value_reconstruct", "tier_C_support_or_grounding", "tier_D_method_only", "reject"]:
        lines.append(f"- {key}: {counts.get(key, 0)}")
    lines.append("")
    lines.append("## 1. Which datasets maximize REAL biological diversity for Global v2?")
    lines.append("")
    for r in shortlist[:10]:
        lines.append(f"- {r['dataset_name']} | {r['source_url']} | {r['why_it_matters']}")
    lines.append("")
    lines.append("## 2. Which datasets were underrated in the conservative prior pass?")
    lines.append("")
    for item in underrated:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 3. Which Zenodo records are actually strong biological candidates even if messy?")
    lines.append("")
    for item in zenodo_strong:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 4. Which datasets should be treated as support/grounding only?")
    lines.append("")
    for r in tier_c[:12]:
        lines.append(f"- {r['title']} | {r['source_url']}")
    lines.append("")
    lines.append("## 5. What is the best 10-dataset shortlist for Global v2 expansion?")
    lines.append("")
    for r in shortlist[:10]:
        lines.append(f"- {r['shortlist_rank']}. {r['dataset_name']} | {r['source_url']}")
    lines.append("")
    lines.append("## 6. What is the best ingest order if we optimize for biological coverage rather than convenience?")
    lines.append("")
    for r in shortlist[:10]:
        lines.append(f"- {r['shortlist_rank']}. {r['dataset_name']} | {r['recommended_bucket']}")
    lines.append("")
    lines.append("## Tier A — ingest soon")
    lines.append("")
    for r in tier_a:
        lines.append(f"- {r['title']} | {r['source_url']} | {r['rationale']}")
    lines.append("")
    lines.append("## Tier B — biologically strong, reconstruct next")
    lines.append("")
    for r in tier_b:
        lines.append(f"- {r['title']} | {r['source_url']} | {r['rationale']}")
    lines.append("")
    lines.append("## Tier C — support / grounding")
    lines.append("")
    for r in tier_c:
        lines.append(f"- {r['title']} | {r['source_url']} | {r['rationale']}")
    lines.append("")
    lines.append("## Tier D — method only")
    lines.append("")
    for r in tier_d:
        lines.append(f"- {r['title']} | {r['source_url']}")
    lines.append("")
    lines.append("## Rejected")
    lines.append("")
    for r in rejected:
        lines.append(f"- {r['title']} | {r['source_url']} | {r['rationale']}")
    lines.append("")
    lines.append("## Global v2 max-real-bio recommendation")
    lines.append("")
    lines.append("Top 5 immediate additions:")
    for r in shortlist[:5]:
        lines.append(f"- {r['dataset_name']} | {r['source_url']}")
    lines.append("")
    lines.append("Next 5 reconstruction targets:")
    for r in shortlist[5:10]:
        lines.append(f"- {r['dataset_name']} | {r['source_url']}")
    lines.append("")
    lines.append("Support datasets worth keeping in reserve:")
    for r in shortlist[10:]:
        lines.append(f"- {r['dataset_name']} | {r['source_url']}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This pass deliberately downweights archive neatness and upweights real biological spectral diversity.")
    lines.append("- Single-patient saliva EV shards were kept alive as Tier B reconstruction components instead of being dismissed record-by-record.")
    lines.append("- Already-ingested but still useful controlled assets were moved to Tier C rather than rejected when they remain relevant as support.")
    lines.append("")
    lines.append("## Priority Table")
    lines.append("")
    lines.append("| Rank | Category | Priority | Bio Value | Effort | Candidate |")
    lines.append("|---:|---|---:|---:|---:|---|")
    for r in priority_out:
        lines.append(f"| {r['priority_rank']} | {r['final_category']} | {r['global_v2_priority_score']} | {r['bio_value_score']} | {r['ingest_effort_score']} | {r['title']} |")

    report_path = OUT_ROOT / "report/global_v2_biobias_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
