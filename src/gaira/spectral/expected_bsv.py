"""
Expected literature BSV comparator — maps spectral cohorts to their
closest literature-grounded BSV profiles for observed-vs-expected analysis.

Each cohort gets its own independent expected comparator. No disease priors
are injected into the spectral projection — this is a post-hoc comparison layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LANDSCAPE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "outputs" / "landscape_v4"

BSV_COMPONENTS = [
    "membrane_lipid", "protein_backbone", "aromatic_amino_acid",
    "purine_nucleotide", "pyrimidine_nucleotide", "glycan_carbohydrate",
    "redox_metabolite", "nucleic_acid_backbone",
]


@dataclass
class ExpectedComparator:
    """Expected literature BSV comparator for a spectral cohort."""
    cohort: str
    comparator_name: str
    match_type: str          # "direct" | "approximate" | "unavailable"
    bsv: dict[str, float] | None
    confidence: str          # "high" | "moderate" | "low" | "none"
    provenance: list[str]
    explanation: str


# ── Cohort-to-comparator mappings ──────────────────────────────────────

_DATASET_COHORT_MAP: dict[str, dict[str, dict]] = {
    "cca_hcc_lm_serum_sers": {
        "hcc": {"comparator": "HCC", "match": "direct",
                "explain": "Direct HCC serum literature profile from GAIRA landscape."},
        "healthy_control": {"comparator": "healthy_control", "match": "direct",
                            "explain": "Direct healthy serum literature baseline from GAIRA landscape."},
        "cca": {"comparator": "cholangiocarcinoma", "match": "direct",
                "explain": "Direct cholangiocarcinoma literature profile from GAIRA landscape."},
        "lm": {"comparator": "liver_cancer_unspecified", "match": "approximate",
               "explain": "No specific liver-metastasis profile available. "
                          "Using liver_cancer_unspecified as closest justified approximation."},
    },
    "hcc_holdout_vornoli2020": {
        "hcc": {"comparator": "HCC", "match": "direct",
                "explain": "Direct HCC serum literature profile from GAIRA landscape."},
        "healthy_control": {"comparator": "healthy_control", "match": "direct",
                            "explain": "Direct healthy serum literature baseline from GAIRA landscape."},
    },
    "diabetes_plasma_ev_sers": {
        "impact": {"comparator": "diabetic_nephropathy", "match": "approximate",
                   "explain": "BMI>25 cohort. No direct adiposity-specific EV profile in GAIRA corpus. "
                              "Using diabetic_nephropathy as closest metabolic/obesity-adjacent comparator. "
                              "Interpret with caution."},
        "strong_d": {"comparator": "healthy_control", "match": "approximate",
                     "explain": "BMI≤25 / normal cohort. Using healthy_control as approximate baseline. "
                                "This is a metabolic cohort, not a healthy-vs-disease comparison."},
    },
}

_GENERIC_COHORT_MAP = {
    "hcc": {"comparator": "HCC", "match": "direct"},
    "healthy_control": {"comparator": "healthy_control", "match": "direct"},
    "healthy": {"comparator": "healthy_control", "match": "direct"},
    "control": {"comparator": "healthy_control", "match": "direct"},
    "cca": {"comparator": "cholangiocarcinoma", "match": "direct"},
    "nafld": {"comparator": "NAFLD_NASH", "match": "direct"},
    "hepatitis": {"comparator": "hepatitis", "match": "direct"},
}

COHORT_DISPLAY_NAMES: dict[str, dict[str, str]] = {
    "diabetes_plasma_ev_sers": {
        "impact": "BMI > 25",
        "strong_d": "BMI ≤ 25 / Normal",
    },
}


def _load_literature_bsv() -> pd.DataFrame | None:
    path = LANDSCAPE_DIR / "bsv_compositional_matrix.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col="condition")


def get_cohort_display_name(dataset_id: str, cohort: str) -> str:
    return COHORT_DISPLAY_NAMES.get(dataset_id, {}).get(cohort, cohort.replace("_", " "))


def build_expected_comparators(
    dataset_id: str,
    cohort_names: list[str],
) -> dict[str, ExpectedComparator]:
    """Build expected literature BSV comparators for each cohort."""
    lit_bsv = _load_literature_bsv()
    ds_map = _DATASET_COHORT_MAP.get(dataset_id, {})

    results = {}
    for cohort in cohort_names:
        mapping = ds_map.get(cohort, _GENERIC_COHORT_MAP.get(cohort))

        if mapping is None:
            results[cohort] = ExpectedComparator(
                cohort=cohort, comparator_name="(none)",
                match_type="unavailable", bsv=None, confidence="none",
                provenance=[], explanation=f"No comparator available for '{cohort}'.",
            )
            continue

        comp_name = mapping["comparator"]
        match_type = mapping["match"]
        explain = mapping.get("explain", f"Mapped to {comp_name} ({match_type}).")

        if lit_bsv is not None and comp_name in lit_bsv.index:
            bsv_vals = {c: round(float(lit_bsv.loc[comp_name, c]), 4) for c in BSV_COMPONENTS}
            nonzero = sum(1 for v in bsv_vals.values() if v > 0)
            confidence = "high" if nonzero >= 5 else "moderate" if nonzero >= 3 else "low"
            if match_type == "approximate":
                confidence = min(confidence, "moderate", key=["high", "moderate", "low"].index)
            provenance = ["GAIRA landscape v4 BSV compositional matrix", f"Condition: {comp_name}"]
        else:
            bsv_vals = None
            confidence = "none"
            provenance = []
            explain += f" Literature BSV for '{comp_name}' not found."
            match_type = "unavailable"

        results[cohort] = ExpectedComparator(
            cohort=cohort, comparator_name=comp_name, match_type=match_type,
            bsv=bsv_vals, confidence=confidence, provenance=provenance,
            explanation=explain,
        )

    return results


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_observed_vs_expected(
    observed_bsv: dict[str, float],
    expected: ExpectedComparator,
) -> dict | None:
    """Compute observed-vs-expected metrics for one cohort."""
    if expected.bsv is None:
        return None

    obs = np.array([observed_bsv.get(c, 0) for c in BSV_COMPONENTS])
    exp = np.array([expected.bsv.get(c, 0) for c in BSV_COMPONENTS])

    return {
        "delta": {c: round(float(obs[i] - exp[i]), 6) for i, c in enumerate(BSV_COMPONENTS)},
        "cosine": round(_cosine(obs, exp), 4),
        "observed": observed_bsv,
        "expected": expected.bsv,
    }


def compute_cross_similarity_matrix(
    cohort_bsvs: dict[str, dict[str, float]],
    expected_comparators: dict[str, ExpectedComparator],
) -> dict:
    """Compute full cohort × expected similarity matrix.

    For every observed cohort, compute cosine similarity against every
    available expected comparator. This reveals whether each cohort
    aligns preferentially with its own expected profile.

    Returns:
        {
            "observed_labels": [str],
            "expected_labels": [str],
            "matrix": [[float]],  # rows=observed, cols=expected
            "alignment_summary": [{
                "cohort", "own_expected", "own_cosine",
                "best_alt_expected", "best_alt_cosine", "margin"
            }]
        }
    """
    # Collect available expected profiles (deduplicated by comparator name)
    exp_profiles: dict[str, dict[str, float]] = {}
    exp_labels: list[str] = []
    for coh, exp in expected_comparators.items():
        if exp.bsv is not None and exp.comparator_name not in exp_profiles:
            exp_profiles[exp.comparator_name] = exp.bsv
            exp_labels.append(exp.comparator_name)

    if not exp_labels:
        return {"observed_labels": [], "expected_labels": [], "matrix": [],
                "alignment_summary": []}

    observed_labels = list(cohort_bsvs.keys())
    matrix = []

    for obs_coh in observed_labels:
        obs_vec = np.array([cohort_bsvs[obs_coh].get(c, 0) for c in BSV_COMPONENTS])
        row = []
        for exp_name in exp_labels:
            exp_vec = np.array([exp_profiles[exp_name].get(c, 0) for c in BSV_COMPONENTS])
            row.append(round(_cosine(obs_vec, exp_vec), 4))
        matrix.append(row)

    # Alignment summary: for each cohort, compare own-expected vs best-alternative
    alignment = []
    for i, obs_coh in enumerate(observed_labels):
        exp_comp = expected_comparators.get(obs_coh)
        if not exp_comp or exp_comp.bsv is None:
            continue

        own_name = exp_comp.comparator_name
        if own_name not in exp_labels:
            continue

        own_idx = exp_labels.index(own_name)
        own_cos = matrix[i][own_idx]

        # Best alternative
        best_alt_cos = -2.0
        best_alt_name = ""
        for j, en in enumerate(exp_labels):
            if j != own_idx and matrix[i][j] > best_alt_cos:
                best_alt_cos = matrix[i][j]
                best_alt_name = en

        margin = round(own_cos - best_alt_cos, 4) if best_alt_name else 0.0

        alignment.append({
            "cohort": obs_coh,
            "own_expected": own_name,
            "own_cosine": own_cos,
            "best_alt_expected": best_alt_name,
            "best_alt_cosine": round(best_alt_cos, 4),
            "margin": margin,
            "match_type": exp_comp.match_type,
        })

    return {
        "observed_labels": observed_labels,
        "expected_labels": exp_labels,
        "matrix": matrix,
        "alignment_summary": alignment,
    }


def generate_interpretation(alignment_summary: list[dict], dataset_id: str) -> str:
    """Generate a concise scientific interpretation of the alignment results."""
    if not alignment_summary:
        return "No expected comparators available for alignment analysis."

    lines = []
    n_correct = sum(1 for a in alignment_summary if a["margin"] > 0)
    n_total = len(alignment_summary)

    if n_correct == n_total:
        lines.append(
            f"All {n_total} cohorts align most strongly with their own expected literature profile."
        )
    elif n_correct > 0:
        lines.append(
            f"{n_correct}/{n_total} cohorts align most strongly with their own expected profile."
        )
    else:
        lines.append(
            "No cohort aligns preferentially with its own expected profile. "
            "This may indicate substrate-dependent spectral differences or "
            "weak literature coverage for these conditions."
        )

    # Best and worst margins
    sorted_align = sorted(alignment_summary, key=lambda a: -a["margin"])
    best = sorted_align[0]
    lines.append(
        f"Strongest alignment: {best['cohort'].replace('_', ' ')} "
        f"(margin = {best['margin']:+.3f} vs {best['best_alt_expected'].replace('_', ' ')})."
    )
    if len(sorted_align) > 1:
        worst = sorted_align[-1]
        if worst["margin"] <= 0:
            lines.append(
                f"Weakest: {worst['cohort'].replace('_', ' ')} aligns more with "
                f"{worst['best_alt_expected'].replace('_', ' ')} than its own expected profile."
            )

    return " ".join(lines)
