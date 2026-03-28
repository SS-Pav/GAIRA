from __future__ import annotations

import re


COHERENT_CLASS_DATASETS = {
    "cca_hcc_lm_serum_sers",
    "small2023_ev",
    "shine_ev_sers",
    "diabetes_plasma_ev_sers",
}


def normalize_label(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def family_label_for_dataset(dataset_id: str, sample_type: str) -> str:
    mapping = {
        "adenine_sers_control": "grounding_analyte",
        "metabolite_sers63_support": "grounding_analyte",
        "amino_acid_raman_grounding": "grounding_analyte",
        "small2023_ev": "ev_general",
        "shine_ev_sers": "ev_disease_or_stress",
        "diabetes_plasma_ev_sers": "ev_disease_or_stress",
        "covid_serum_raman": "serum_general",
        "serum_protocol_comparison": "serum_general",
        "serum_ag_colloids": "serum_general",
        "cspp_serum": "serum_general",
        "ergothioneine_serum": "serum_general",
        "cca_hcc_lm_serum_sers": "serum_liver_hepatobiliary",
    }
    return mapping.get(dataset_id, sample_type)


def semantic_group(dataset_id: str, sample_type: str, label_optional: str | None) -> str:
    label = normalize_label(label_optional)
    if not label:
        return ""
    if sample_type == "grounding":
        return f"molecule::{label}"
    if dataset_id in COHERENT_CLASS_DATASETS:
        return f"class::{dataset_id}::{label}"
    return ""


def hard_negative_scope(dataset_id: str, sample_type: str, label_optional: str | None) -> str:
    label = normalize_label(label_optional)
    if not label:
        return ""
    if sample_type == "grounding":
        return "grounding"
    if dataset_id in COHERENT_CLASS_DATASETS:
        return f"dataset::{dataset_id}"
    return ""
