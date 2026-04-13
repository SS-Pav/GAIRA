"""
Literature-grounded BSV profile builder for GAIRA text queries.

Loads pre-computed BSV matrices from GAIRA landscape v4 and returns
condition-level biochemical composition profiles. These are
literature-grounded, NOT spectra-derived.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import pandas as pd

from gaira.retrieval.text_query_retriever import RetrievedItem


BSV_COMPONENTS = [
    "membrane_lipid", "protein_backbone", "aromatic_amino_acid",
    "purine_nucleotide", "pyrimidine_nucleotide", "glycan_carbohydrate",
    "redox_metabolite", "nucleic_acid_backbone",
]

LANDSCAPE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "outputs" / "landscape_v4"

# Condition name aliases: query terms → canonical BSV matrix condition
CONDITION_ALIASES = {
    "hcc": "HCC",
    "hepatocellular carcinoma": "HCC",
    "nafld": "NAFLD_NASH",
    "nash": "NAFLD_NASH",
    "fatty liver": "NAFLD_NASH",
    "cca": "cholangiocarcinoma",
    "cholangiocarcinoma": "cholangiocarcinoma",
    "healthy": "healthy_control",
    "healthy control": "healthy_control",
    "control": "healthy_control",
    "lm": "liver_cancer_unspecified",
    "liver metastasis": "liver_cancer_unspecified",
    "liver metastases": "liver_cancer_unspecified",
    "hepatitis": "hepatitis",
    "fibrosis": "fibrosis",
    "breast cancer": "breast_cancer",
    "lung cancer": "lung_cancer",
    "covid": "covid19",
    "covid-19": "covid19",
    "diabetes": "type_2_diabetes_subtype_context",
    "leukemia": "leukemia",
    "ovarian cancer": "ovarian_cancer",
    "prostate cancer": "prostate_cancer",
}


def _load_bsv_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load compositional and delta BSV matrices."""
    comp = pd.read_csv(LANDSCAPE_DIR / "bsv_compositional_matrix.csv", index_col="condition")
    delta = pd.read_csv(LANDSCAPE_DIR / "bsv_delta_matrix.csv", index_col="condition")
    return comp, delta


def detect_conditions(query: str) -> list[str]:
    """Detect condition names mentioned in the query.

    Returns canonical condition names from the BSV matrix.
    """
    query_lower = query.lower()
    found = []
    # Sort aliases by length descending to match longer phrases first
    for alias, canonical in sorted(CONDITION_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in query_lower and canonical not in found:
            found.append(canonical)
    return found


def build_literature_bsv_profile(
    query: str,
    retrieved_items: Sequence[RetrievedItem] | None = None,
) -> dict:
    """Build literature-grounded BSV profiles for detected conditions.

    Args:
        query: User's text query.
        retrieved_items: Optional retrieved items (for future evidence-weighted override).

    Returns:
        dict with:
          conditions: list of detected condition names
          profiles: {condition: {component: score}}
          deltas: {condition: {component: delta_vs_healthy}}
          available: bool
          notes: str
    """
    conditions = detect_conditions(query)

    try:
        comp_df, delta_df = _load_bsv_matrices()
    except FileNotFoundError:
        return {
            "conditions": conditions,
            "profiles": {},
            "deltas": {},
            "available": False,
            "notes": "BSV landscape matrices not found.",
        }

    profiles = {}
    deltas = {}
    missing = []

    for cond in conditions:
        if cond in comp_df.index:
            profiles[cond] = {
                c: round(float(comp_df.loc[cond, c]), 4) for c in BSV_COMPONENTS
            }
        else:
            missing.append(cond)

        if cond in delta_df.index:
            deltas[cond] = {
                c: round(float(delta_df.loc[cond, c]), 4) for c in BSV_COMPONENTS
            }

    notes_parts = []
    if not conditions:
        notes_parts.append("No specific condition detected in query.")
    if missing:
        notes_parts.append(f"No BSV data for: {', '.join(missing)}.")
    if profiles:
        notes_parts.append(
            f"Literature-grounded profiles available for: {', '.join(profiles.keys())}."
        )

    return {
        "conditions": conditions,
        "profiles": profiles,
        "deltas": deltas,
        "available": len(profiles) > 0,
        "notes": " ".join(notes_parts),
    }
