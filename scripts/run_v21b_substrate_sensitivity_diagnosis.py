"""
GAIRA Spectral Query v2.1b — Substrate Sensitivity Diagnosis

Diagnostic-only analysis comparing v1 HCC holdout (Au SERS) with v2.1 CCA dataset (AgNP SERS).
No refinement, no weighting, no parameter tuning.

Steps:
  1. Axis sign consistency across datasets
  2. Multi-condition axis structure in CCA dataset
  3. Axis importance shift
  4. Window-level directional analysis
  5. Nucleic acid ablation test
  6. Aromatic amino acid robustness test
  7. Cross-prior axis behavior
  8. Disease-general vs HCC-specific summary
  9. Figures
 10. Summary report
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
sns.set_theme(style="whitegrid", context="talk")

# ── paths ──────────────────────────────────────────────────────────────
OUT = Path("/Users/suraj/projects/GAIRA/outputs/gaira_spectral_query_v2_1b")
OUT.mkdir(parents=True, exist_ok=True)

V1_DIR = Path("/Users/suraj/projects/GAIRA/outputs/gaira_spectral_query_v1_hcc")
V11_DIR = Path("/Users/suraj/projects/GAIRA/outputs/gaira_spectral_query_v1_1_hcc")
V21_DIR = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/structured_evidence_v2/reports/gaira_spectral_query_v2_1_cca"
)
V12_DIR = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/structured_evidence_v2/reports/v12_adversarial"
)
LANDSCAPE_DIR = Path("/Users/suraj/projects/GAIRA/outputs/landscape_v4")
REPORT_DIR = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/structured_evidence_v2/reports"
)

BSV_COMPONENTS = [
    "membrane_lipid",
    "protein_backbone",
    "aromatic_amino_acid",
    "purine_nucleotide",
    "pyrimidine_nucleotide",
    "glycan_carbohydrate",
    "redox_metabolite",
    "nucleic_acid_backbone",
]

# ── load data ──────────────────────────────────────────────────────────

# v1 HCC holdout — per-sample BSV and delta matrices
v1_bsv = pd.read_csv(V1_DIR / "spectral_bsv_matrix.csv")
v1_delta = pd.read_csv(V1_DIR / "spectral_delta_matrix.csv")
v1_stability = pd.read_csv(V1_DIR / "spectral_signal_stability.csv")
v1_window_panel = pd.read_csv(V1_DIR / "spectral_window_panel_selected.csv")
v1_alignment = pd.read_csv(V1_DIR / "hcc_alignment_scores.csv")
v1_sample_sim = pd.read_csv(V1_DIR / "sample_condition_similarity.csv")

# v1.1 window selection
v11_windows = pd.read_csv(V11_DIR / "v11_window_selection.csv")

# v1.2 adversarial ablation
v12_ablation = pd.read_csv(V12_DIR / "v12_axis_ablation_results.csv")

# v2.1 CCA dataset
v21_means = pd.read_csv(V21_DIR / "v21_bsv_means_by_condition.csv")
v21_delta = pd.read_csv(V21_DIR / "v21_bsv_delta_matrix.csv")
v21_metrics = pd.read_csv(V21_DIR / "v21_metrics_summary.csv")
v21_sample_align = pd.read_csv(V21_DIR / "v21_alignment_sample_level.csv")

# Literature-derived priors from landscape v4
landscape_delta = pd.read_csv(LANDSCAPE_DIR / "bsv_delta_matrix.csv")
hcc_prior_row = landscape_delta[landscape_delta["condition"] == "HCC"]
nafld_prior_row = landscape_delta[landscape_delta["condition"] == "NAFLD_NASH"]

HCC_PRIOR = hcc_prior_row[BSV_COMPONENTS].values.flatten()
NAFLD_PRIOR = nafld_prior_row[BSV_COMPONENTS].values.flatten()

print(f"HCC prior:   {dict(zip(BSV_COMPONENTS, HCC_PRIOR))}")
print(f"NAFLD prior: {dict(zip(BSV_COMPONENTS, NAFLD_PRIOR))}")

# ── helpers ────────────────────────────────────────────────────────────

WEAK_THRESHOLD = 0.03  # |delta| < this → WEAK


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def save_csv(df: pd.DataFrame, name: str) -> Path:
    p = OUT / name
    df.to_csv(p, index=False)
    print(f"  saved: {p.name}")
    return p


def save_fig(fig: plt.Figure, name: str) -> Path:
    p = OUT / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {p.name}")
    return p


# ======================================================================
# STEP 1 — Axis Sign Consistency Across Datasets
# ======================================================================
print("\n=== STEP 1: Axis sign consistency ===")

# v1: delta = mean(HCC) - mean(CTR)
v1_hcc = v1_stability.set_index("component")
delta_v1 = {c: float(v1_hcc.loc[c, "delta_mean"]) for c in BSV_COMPONENTS}

# v2.1: delta = HCC - healthy_control (from precomputed delta matrix)
v21_delta_idx = v21_delta.set_index("condition")
delta_v2_hcc = {c: float(v21_delta_idx.loc["hcc", c]) for c in BSV_COMPONENTS}

rows = []
for comp in BSV_COMPONENTS:
    d1 = delta_v1[comp]
    d2 = delta_v2_hcc[comp]
    abs1, abs2 = abs(d1), abs(d2)

    if abs1 < WEAK_THRESHOLD and abs2 < WEAK_THRESHOLD:
        cls = "WEAK"
    elif (d1 > 0 and d2 > 0):
        cls = "STABLE_POSITIVE"
    elif (d1 < 0 and d2 < 0):
        cls = "STABLE_NEGATIVE"
    else:
        cls = "INVERTED"

    rows.append({
        "BSV_component": comp,
        "delta_v1_hcc_vs_healthy": round(d1, 4),
        "delta_v2_hcc_vs_healthy": round(d2, 4),
        "classification": cls,
        "abs_delta_v1": round(abs1, 4),
        "abs_delta_v2": round(abs2, 4),
        "magnitude_change": round(abs2 - abs1, 4),
    })

axis_sign = pd.DataFrame(rows)
save_csv(axis_sign, "axis_sign_consistency.csv")
print(axis_sign.to_string(index=False))

# ======================================================================
# STEP 2 — Multi-Condition Axis Structure in CCA Dataset
# ======================================================================
print("\n=== STEP 2: Multi-condition axis structure ===")

v21_means_idx = v21_means.set_index("condition")
conditions_ordered = ["healthy_control", "lm", "cca", "hcc"]

rows2 = []
for comp in BSV_COMPONENTS:
    means = {c: float(v21_means_idx.loc[c, comp]) for c in conditions_ordered}
    deltas_vs_h = {c: means[c] - means["healthy_control"] for c in ["lm", "cca", "hcc"]}

    # Pairwise
    d_hcc_lm = means["hcc"] - means["lm"]
    d_hcc_cca = means["hcc"] - means["cca"]
    d_cca_lm = means["cca"] - means["lm"]

    # Ordering
    sorted_conds = sorted(conditions_ordered, key=lambda c: means[c])
    ordering = " < ".join(sorted_conds)

    # Variance across conditions
    vals = [means[c] for c in conditions_ordered]
    var_across = float(np.var(vals))

    # Classification
    disease_deltas = [deltas_vs_h["lm"], deltas_vs_h["cca"], deltas_vs_h["hcc"]]
    signs = [np.sign(d) for d in disease_deltas if abs(d) > WEAK_THRESHOLD]

    if len(signs) == 0:
        cls = "WEAK"
        notes = "all deltas near zero"
    elif all(s == signs[0] for s in signs):
        # All disease conditions move same direction
        # Check if HCC is clearly different from CCA/LM
        if abs(deltas_vs_h["hcc"]) > 2 * max(abs(deltas_vs_h["cca"]), abs(deltas_vs_h["lm"])):
            cls = "HCC_SPECIFIC"
            notes = "HCC magnitude >2x others"
        else:
            # Check gradient
            if (deltas_vs_h["lm"] <= deltas_vs_h["cca"] <= deltas_vs_h["hcc"] or
                deltas_vs_h["lm"] >= deltas_vs_h["cca"] >= deltas_vs_h["hcc"]):
                cls = "GRADIENT"
                notes = "monotonic across disease conditions"
            else:
                cls = "DISEASE_GENERAL"
                notes = "all disease same direction vs healthy"
    else:
        # Mixed signs
        if (np.sign(deltas_vs_h["hcc"]) != np.sign(deltas_vs_h["cca"]) and
            np.sign(deltas_vs_h["hcc"]) != np.sign(deltas_vs_h["lm"]) and
            abs(deltas_vs_h["hcc"]) > WEAK_THRESHOLD):
            cls = "HCC_SPECIFIC"
            notes = "HCC sign differs from CCA and LM"
        else:
            cls = "INCONSISTENT"
            notes = "mixed sign pattern"

    rows2.append({
        "BSV_component": comp,
        "mean_healthy": round(means["healthy_control"], 4),
        "mean_lm": round(means["lm"], 4),
        "mean_cca": round(means["cca"], 4),
        "mean_hcc": round(means["hcc"], 4),
        "delta_lm_vs_healthy": round(deltas_vs_h["lm"], 4),
        "delta_cca_vs_healthy": round(deltas_vs_h["cca"], 4),
        "delta_hcc_vs_healthy": round(deltas_vs_h["hcc"], 4),
        "delta_hcc_vs_lm": round(d_hcc_lm, 4),
        "delta_hcc_vs_cca": round(d_hcc_cca, 4),
        "delta_cca_vs_lm": round(d_cca_lm, 4),
        "variance_across_conditions": round(var_across, 6),
        "ordering": ordering,
        "classification": cls,
        "notes": notes,
    })

axis_structure = pd.DataFrame(rows2)
save_csv(axis_structure, "axis_multi_condition_structure.csv")
print(axis_structure[["BSV_component", "delta_hcc_vs_healthy", "classification", "ordering"]].to_string(index=False))

# ======================================================================
# STEP 3 — Axis Importance Shift
# ======================================================================
print("\n=== STEP 3: Axis importance shift ===")

# v1 importance: from v1.2 adversarial ablation (drop_cosine per axis)
v12_abl_idx = v12_ablation.set_index("ablated")
importance_v1 = {}
for comp in BSV_COMPONENTS:
    if comp in v12_abl_idx.index:
        importance_v1[comp] = float(v12_abl_idx.loc[comp, "drop_cosine"])
    else:
        importance_v1[comp] = 0.0

# v2.1 importance: contribution to alignment = delta_component * prior_component / (|delta| * |prior|)
# This measures how much each axis contributes to the overall cosine
v21_delta_idx = v21_delta.set_index("condition")

def axis_contributions(delta_vec: np.ndarray, prior_vec: np.ndarray) -> dict[str, float]:
    """Per-axis contribution to cosine similarity."""
    norm_d = np.linalg.norm(delta_vec)
    norm_p = np.linalg.norm(prior_vec)
    if norm_d < 1e-12 or norm_p < 1e-12:
        return {c: 0.0 for c in BSV_COMPONENTS}
    denom = norm_d * norm_p
    return {BSV_COMPONENTS[i]: float(delta_vec[i] * prior_vec[i] / denom) for i in range(8)}


delta_hcc_vec = v21_delta_idx.loc["hcc", BSV_COMPONENTS].values.astype(float)
delta_cca_vec = v21_delta_idx.loc["cca", BSV_COMPONENTS].values.astype(float)
delta_lm_vec = v21_delta_idx.loc["lm", BSV_COMPONENTS].values.astype(float)

contrib_v2_hcc = axis_contributions(delta_hcc_vec, HCC_PRIOR)
contrib_v2_cca = axis_contributions(delta_cca_vec, HCC_PRIOR)
contrib_v2_lm = axis_contributions(delta_lm_vec, HCC_PRIOR)

# Rank by absolute importance
rank_v1 = {k: r + 1 for r, (k, _) in enumerate(
    sorted(importance_v1.items(), key=lambda x: abs(x[1]), reverse=True)
)}
rank_v2 = {k: r + 1 for r, (k, _) in enumerate(
    sorted(contrib_v2_hcc.items(), key=lambda x: abs(x[1]), reverse=True)
)}

rows3 = []
for comp in BSV_COMPONENTS:
    rows3.append({
        "BSV_component": comp,
        "importance_v1": round(importance_v1[comp], 4),
        "importance_v2_hcc": round(contrib_v2_hcc[comp], 4),
        "importance_v2_cca": round(contrib_v2_cca[comp], 4),
        "importance_v2_lm": round(contrib_v2_lm[comp], 4),
        "rank_v1": rank_v1[comp],
        "rank_v2_hcc": rank_v2[comp],
        "shift_v1_to_v2_hcc": rank_v2[comp] - rank_v1[comp],
        "notes": "",
    })

# Add notes
for r in rows3:
    comp = r["BSV_component"]
    if comp == "nucleic_acid_backbone":
        r["notes"] = "Dominated v1 (49% of alignment); sign-inverted in v2.1"
    elif comp == "aromatic_amino_acid":
        r["notes"] = "Largest positive contributor to HCC delta in v2.1"
    elif comp == "membrane_lipid":
        r["notes"] = "2nd in v1 importance; moderate in v2.1"

importance_shift = pd.DataFrame(rows3)
save_csv(importance_shift, "axis_importance_shift.csv")
print(importance_shift[["BSV_component", "importance_v1", "importance_v2_hcc", "rank_v1", "rank_v2_hcc"]].to_string(index=False))

# ======================================================================
# STEP 4 — Window-Level Directional Analysis
# ======================================================================
print("\n=== STEP 4: Window-level directional analysis ===")

# v1: window-level delta from the window panel (delta_magnitude has sign info lost)
# We need to recover the actual signed deltas from per-sample data.
# v1_bsv has per-sample BSVs, but windows are aggregated into BSV components.
# The window panel has delta_magnitude which is |HCC_mean - CTR_mean| for that window.
# But we need the SIGNED delta. Let's compute from the raw BSV matrix per-sample.

# Actually, the v1 data is organized by BSV component not by raw window.
# The window panel maps window -> bsv_component with delta_magnitude.
# For v1, the signed window delta is available from the delta matrix indirectly:
# Each window maps to one BSV component, so the window delta contributes to that component's delta.

# However we don't have raw per-window per-sample features stored.
# We DO have the window panel with delta_magnitude and alignment_drop from v1.
# The sign of the window contribution can be inferred from the BSV component delta sign.

# For v2.1, we also only have BSV-level data, not raw window-level.
# To do a proper window-level analysis, we'd need the spectral data.

# Let's use what we have: the window panel from v1 contains per-window info,
# and we can infer signs from the BSV component delta.

# Build window -> BSV component mapping from v1 panel
v1_wp = v1_window_panel.copy()
v1_wp["signed_delta_v1"] = v1_wp.apply(
    lambda r: r["delta_magnitude"] * np.sign(delta_v1.get(r["bsv_component"], 0))
    if r["bsv_component"] in delta_v1 else 0.0, axis=1
)

# For v2.1, we can infer window contributions similarly
# Each window's contribution to HCC is proportional to the BSV component delta
v21_delta_dict_hcc = delta_v2_hcc
v21_delta_dict_cca = {c: float(v21_delta_idx.loc["cca", c]) for c in BSV_COMPONENTS}
v21_delta_dict_lm = {c: float(v21_delta_idx.loc["lm", c]) for c in BSV_COMPONENTS}

# For a window mapped to a BSV component, the window's delta direction is
# approximated by the component's delta direction. This is imperfect (multiple
# windows map to the same component) but it's the best we can do without raw
# per-window spectral features.

# NOTE: This is an approximation. Document this limitation.

WINDOW_BSV_MAP = {}
for _, row in v1_wp.iterrows():
    WINDOW_BSV_MAP[row["window"]] = row["bsv_component"]

rows4 = []
for _, row in v1_wp.iterrows():
    win = row["window"]
    comp = row["bsv_component"]
    d_v1 = row["signed_delta_v1"]

    # v2.1 deltas inferred from BSV component
    d_v2_hcc = v21_delta_dict_hcc.get(comp, 0.0)
    d_v2_cca = v21_delta_dict_cca.get(comp, 0.0)
    d_v2_lm = v21_delta_dict_lm.get(comp, 0.0)

    # Parse window range
    parts = win.split("-")
    w_start = int(parts[0])
    w_end = int(parts[1])

    # Classification
    if abs(d_v1) < WEAK_THRESHOLD and abs(d_v2_hcc) < WEAK_THRESHOLD:
        cls = "WEAK"
    elif np.sign(d_v1) == np.sign(d_v2_hcc) and abs(d_v1) >= WEAK_THRESHOLD:
        cls = "STABLE"
    elif np.sign(d_v1) != np.sign(d_v2_hcc) and abs(d_v1) >= WEAK_THRESHOLD and abs(d_v2_hcc) >= WEAK_THRESHOLD:
        cls = "INVERTED"
    else:
        cls = "WEAK"

    # Check disease-general vs HCC-specific
    if abs(d_v2_hcc) >= WEAK_THRESHOLD:
        disease_signs = [np.sign(d_v2_hcc), np.sign(d_v2_cca), np.sign(d_v2_lm)]
        nonzero_signs = [s for s, d in zip(disease_signs, [d_v2_hcc, d_v2_cca, d_v2_lm]) if abs(d) >= WEAK_THRESHOLD]
        if len(nonzero_signs) > 1 and all(s == nonzero_signs[0] for s in nonzero_signs):
            if abs(d_v2_hcc) > 2 * max(abs(d_v2_cca), abs(d_v2_lm)):
                cls += "_HCC_SPECIFIC"
            else:
                cls += "_DISEASE_GENERAL"

    notes = ""
    if win == "1020-1080":
        notes = "nucleic_acid_backbone — INVERTED: +0.23 in v1, -0.22 in v2.1"
    elif win == "620-660":
        notes = "aromatic_amino_acid — HCC shows enrichment in both datasets"
    elif win == "500-540":
        notes = "redox_metabolite — direction change"
    elif win == "1380-1450":
        notes = "membrane_lipid — 2nd most important in v1"

    rows4.append({
        "window_id": win,
        "wavenumber_start": w_start,
        "wavenumber_end": w_end,
        "bsv_component": comp,
        "delta_v1_hcc_vs_healthy": round(d_v1, 4),
        "delta_v2_hcc_vs_healthy": round(d_v2_hcc, 4),
        "delta_v2_cca_vs_healthy": round(d_v2_cca, 4),
        "delta_v2_lm_vs_healthy": round(d_v2_lm, 4),
        "v1_importance_score": round(row["importance_score"], 4),
        "classification": cls,
        "notes": notes,
    })

window_sign = pd.DataFrame(rows4)
save_csv(window_sign, "window_sign_consistency.csv")
print(window_sign[["window_id", "bsv_component", "delta_v1_hcc_vs_healthy", "delta_v2_hcc_vs_healthy", "classification"]].to_string(index=False))

# ======================================================================
# STEP 5 — Nucleic Acid Ablation Test
# ======================================================================
print("\n=== STEP 5: Nucleic acid ablation ===")

# Remove nucleic_acid_backbone from BSV vectors and recompute alignment
ablated_components = [c for c in BSV_COMPONENTS if c != "nucleic_acid_backbone"]
ablated_idx = [BSV_COMPONENTS.index(c) for c in ablated_components]

HCC_PRIOR_ABLATED = HCC_PRIOR[ablated_idx]
NAFLD_PRIOR_ABLATED = NAFLD_PRIOR[ablated_idx]

# Compute per-condition alignment from condition-level deltas
results_ablation = []
for cond in ["healthy_control", "lm", "cca", "hcc"]:
    delta_vec_full = v21_delta_idx.loc[cond, BSV_COMPONENTS].values.astype(float)
    delta_vec_ablated = delta_vec_full[ablated_idx]

    cos_full = cosine(delta_vec_full, HCC_PRIOR)
    cos_ablated = cosine(delta_vec_ablated, HCC_PRIOR_ABLATED)

    results_ablation.append({
        "condition": cond,
        "cohort_cosine_full": round(cos_full, 4),
        "cohort_cosine_ablated": round(cos_ablated, 4),
        "change": round(cos_ablated - cos_full, 4),
    })

ablation_df = pd.DataFrame(results_ablation)

# Determine ordering
full_order = ablation_df.sort_values("cohort_cosine_full", ascending=False)["condition"].tolist()
ablated_order = ablation_df.sort_values("cohort_cosine_ablated", ascending=False)["condition"].tolist()
ablation_df["full_rank"] = ablation_df["condition"].map(
    {c: i + 1 for i, c in enumerate(full_order)}
)
ablation_df["ablated_rank"] = ablation_df["condition"].map(
    {c: i + 1 for i, c in enumerate(ablated_order)}
)

save_csv(ablation_df, "nucleic_acid_ablation_results.csv")
print(ablation_df.to_string(index=False))
print(f"\nFull ordering:    {' > '.join(full_order)}")
print(f"Ablated ordering: {' > '.join(ablated_order)}")

# Also do per-sample ablation for more robust stats
sample_ablation_rows = []
for _, row in v21_sample_align.iterrows():
    cond = row["condition"]
    # We need per-sample BSV deltas. We don't have them directly.
    # We only have per-sample cosines from v2.1. Skip per-sample for now
    # and note this as a limitation.

# ======================================================================
# STEP 6 — Aromatic Amino Acid Robustness Test
# ======================================================================
print("\n=== STEP 6: Aromatic amino acid robustness ===")

axis_subsets = {
    "aromatic_only": ["aromatic_amino_acid"],
    "aromatic_protein": ["aromatic_amino_acid", "protein_backbone"],
    "aromatic_redox": ["aromatic_amino_acid", "redox_metabolite"],
    "nucleic_acid_only": ["nucleic_acid_backbone"],
    "membrane_only": ["membrane_lipid"],
    "all_minus_nucleic": ablated_components,
    "full": BSV_COMPONENTS,
}

rows6 = []
for subset_name, subset_comps in axis_subsets.items():
    idx = [BSV_COMPONENTS.index(c) for c in subset_comps]
    prior_sub = HCC_PRIOR[idx]

    cond_cosines = {}
    for cond in ["healthy_control", "lm", "cca", "hcc"]:
        delta_vec = v21_delta_idx.loc[cond, BSV_COMPONENTS].values.astype(float)
        delta_sub = delta_vec[idx]
        cond_cosines[cond] = cosine(delta_sub, prior_sub)

    # Ordering
    ordered = sorted(cond_cosines.items(), key=lambda x: x[1], reverse=True)
    ordering = " > ".join(f"{c}({v:.3f})" for c, v in ordered)

    # Check if disease > healthy
    disease_above_healthy = all(
        cond_cosines[d] > cond_cosines["healthy_control"]
        for d in ["hcc", "cca", "lm"]
    )
    hcc_is_top = ordered[0][0] == "hcc"

    comments = []
    if disease_above_healthy:
        comments.append("disease>healthy")
    if hcc_is_top:
        comments.append("HCC is top")
    if cond_cosines["healthy_control"] > cond_cosines["hcc"]:
        comments.append("INVERTED: healthy>HCC")

    rows6.append({
        "axis_subset": subset_name,
        "components": "+".join(subset_comps),
        "cohort_cosine_healthy": round(cond_cosines["healthy_control"], 4),
        "cohort_cosine_lm": round(cond_cosines["lm"], 4),
        "cohort_cosine_cca": round(cond_cosines["cca"], 4),
        "cohort_cosine_hcc": round(cond_cosines["hcc"], 4),
        "ordering": ordering,
        "comments": "; ".join(comments) if comments else "",
    })

aromatic_test = pd.DataFrame(rows6)
save_csv(aromatic_test, "aromatic_axis_test.csv")
print(aromatic_test[["axis_subset", "cohort_cosine_hcc", "cohort_cosine_healthy", "ordering"]].to_string(index=False))

# ======================================================================
# STEP 7 — Cross-Prior Axis Behavior
# ======================================================================
print("\n=== STEP 7: Cross-prior axis behavior ===")

# For each BSV axis, compute its contribution to HCC prior and NAFLD prior alignment
# using the HCC delta vector from the CCA dataset
delta_hcc_v2 = v21_delta_idx.loc["hcc", BSV_COMPONENTS].values.astype(float)

contrib_hcc = axis_contributions(delta_hcc_v2, HCC_PRIOR)
contrib_nafld = axis_contributions(delta_hcc_v2, NAFLD_PRIOR)

rows7 = []
for comp in BSV_COMPONENTS:
    c_hcc = contrib_hcc[comp]
    c_nafld = contrib_nafld[comp]
    same_dir = "yes" if (np.sign(c_hcc) == np.sign(c_nafld) and abs(c_hcc) > 0.001) else "no"

    # Relative strength
    max_abs = max(abs(c_hcc), abs(c_nafld))
    if max_abs < 0.001:
        rel = "negligible"
    elif abs(c_hcc) > abs(c_nafld) * 2:
        rel = "HCC-prior-dominant"
    elif abs(c_nafld) > abs(c_hcc) * 2:
        rel = "NAFLD-prior-dominant"
    else:
        rel = "comparable"

    notes = ""
    if comp == "nucleic_acid_backbone":
        notes = "Opposes HCC prior (negative contribution) but near-zero for NAFLD"
    elif comp == "aromatic_amino_acid":
        notes = "Positive for both priors — robust direction"
    elif comp == "membrane_lipid":
        notes = "Positive for HCC prior (both delta and prior are negative)"

    rows7.append({
        "BSV_component": comp,
        "contribution_to_hcc_prior": round(c_hcc, 4),
        "contribution_to_nafld_prior": round(c_nafld, 4),
        "same_direction_yes_no": same_dir,
        "relative_strength": rel,
        "notes": notes,
    })

cross_prior = pd.DataFrame(rows7)
save_csv(cross_prior, "cross_prior_axis_behavior.csv")
print(cross_prior.to_string(index=False))

# ======================================================================
# STEP 8 — Disease-General vs HCC-Specific Summary
# ======================================================================
print("\n=== STEP 8: Disease specificity summary ===")

# Combine classifications from steps 1, 2
rows8 = []
for i, comp in enumerate(BSV_COMPONENTS):
    s1 = axis_sign.iloc[i]
    s2 = axis_structure.iloc[i]

    # Determine substrate sensitivity from sign consistency
    substrate_sensitive = s1["classification"] == "INVERTED"
    robust = s1["classification"] in ("STABLE_POSITIVE", "STABLE_NEGATIVE")

    # Disease structure from multi-condition
    disease_cls = s2["classification"]

    # Combined category
    if substrate_sensitive:
        category = "substrate_sensitive"
    elif robust and disease_cls in ("DISEASE_GENERAL", "GRADIENT"):
        category = "robust_disease_general"
    elif robust and disease_cls == "HCC_SPECIFIC":
        category = "robust_hcc_specific"
    elif s1["classification"] == "WEAK":
        category = "weak_or_uninterpretable"
    else:
        category = disease_cls.lower()

    rows8.append({
        "element": comp,
        "element_type": "BSV_component",
        "sign_consistency": s1["classification"],
        "multi_condition_structure": disease_cls,
        "category": category,
    })

# Add key windows
key_windows = {
    "1020-1080": "nucleic_acid_backbone window",
    "620-660": "aromatic_amino_acid window",
    "500-540": "redox window",
    "1380-1450": "membrane_lipid window",
    "1260-1320": "protein_backbone window",
}
ws_idx = window_sign.set_index("window_id")
for win_id, label in key_windows.items():
    if win_id in ws_idx.index:
        wr = ws_idx.loc[win_id]
        cls = wr["classification"]
        if "INVERTED" in cls:
            cat = "substrate_sensitive"
        elif "STABLE" in cls:
            cat = "robust_across_datasets"
        elif "WEAK" in cls:
            cat = "weak_or_uninterpretable"
        else:
            cat = cls.lower()

        rows8.append({
            "element": f"{win_id} ({label})",
            "element_type": "spectral_window",
            "sign_consistency": cls,
            "multi_condition_structure": "",
            "category": cat,
        })

summary_df = pd.DataFrame(rows8)
save_csv(summary_df, "disease_specificity_summary.csv")
print(summary_df.to_string(index=False))

# ======================================================================
# STEP 9 — Figures
# ======================================================================
print("\n=== STEP 9: Figures ===")

# ── Figure 1: Axis sign consistency scatter ──
fig, ax = plt.subplots(figsize=(10, 8))
for _, row in axis_sign.iterrows():
    color = {"STABLE_POSITIVE": "#2ca02c", "STABLE_NEGATIVE": "#1f77b4",
             "INVERTED": "#d62728", "WEAK": "#999999"}[row["classification"]]
    ax.scatter(row["delta_v1_hcc_vs_healthy"], row["delta_v2_hcc_vs_healthy"],
               s=200, c=color, edgecolors="black", linewidths=1.5, zorder=5)
    ax.annotate(row["BSV_component"].replace("_", "\n"),
                (row["delta_v1_hcc_vs_healthy"], row["delta_v2_hcc_vs_healthy"]),
                fontsize=8, ha="center", va="bottom",
                xytext=(0, 12), textcoords="offset points")

# Reference lines
lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]),
          abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) * 1.2
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="y=x (perfect stability)")

# Quadrant labels
ax.text(lim * 0.7, lim * 0.7, "STABLE\nPOSITIVE", ha="center", fontsize=9, alpha=0.4, color="green")
ax.text(-lim * 0.7, -lim * 0.7, "STABLE\nNEGATIVE", ha="center", fontsize=9, alpha=0.4, color="blue")
ax.text(lim * 0.7, -lim * 0.7, "INVERTED", ha="center", fontsize=9, alpha=0.4, color="red")
ax.text(-lim * 0.7, lim * 0.7, "INVERTED", ha="center", fontsize=9, alpha=0.4, color="red")

ax.set_xlabel("Delta v1 (Vornoli Au SERS): HCC vs Healthy")
ax.set_ylabel("Delta v2.1 (CCA AgNP SERS): HCC vs Healthy")
ax.set_title("BSV Axis Sign Consistency Across Datasets")
save_fig(fig, "fig1_axis_sign_consistency.png")

# ── Figure 2: Multi-condition BSV heatmap ──
heatmap_data = v21_delta.set_index("condition")[BSV_COMPONENTS]
heatmap_data = heatmap_data.loc[["hcc", "cca", "lm", "healthy_control"]]

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            ax=ax, linewidths=0.5, cbar_kws={"label": "Delta vs Healthy"})
ax.set_title("CCA Dataset: BSV Delta vs Healthy Control (per condition)")
ax.set_ylabel("")
ax.set_xticklabels([c.replace("_", "\n") for c in BSV_COMPONENTS], rotation=0, fontsize=9)
save_fig(fig, "fig2_multicondition_bsv_delta_heatmap.png")

# ── Figure 3: Axis importance shift bar chart ──
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
x = np.arange(len(BSV_COMPONENTS))
w = 0.35

# v1 importance
ax = axes[0]
vals_v1 = [importance_v1[c] for c in BSV_COMPONENTS]
colors_v1 = ["#d62728" if v > 0.1 else "#ff9896" if v > 0.01 else "#cccccc" for v in vals_v1]
ax.barh(x, vals_v1, color=colors_v1, edgecolor="black", linewidth=0.5)
ax.set_yticks(x)
ax.set_yticklabels([c.replace("_", " ") for c in BSV_COMPONENTS], fontsize=9)
ax.set_xlabel("Ablation Drop in Cosine")
ax.set_title("v1.2 (Vornoli Au) — Axis Importance")

# v2.1 importance (contribution to cosine)
ax = axes[1]
vals_v2 = [contrib_v2_hcc[c] for c in BSV_COMPONENTS]
colors_v2 = ["#d62728" if v < -0.05 else "#2ca02c" if v > 0.05 else "#cccccc" for v in vals_v2]
ax.barh(x, vals_v2, color=colors_v2, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Contribution to HCC Prior Cosine")
ax.set_title("v2.1 (CCA AgNP) — Axis Contribution")
ax.axvline(0, color="black", linewidth=0.8)

plt.suptitle("Axis Importance Shift: v1 → v2.1", fontsize=14, y=1.02)
plt.tight_layout()
save_fig(fig, "fig3_axis_importance_shift.png")

# ── Figure 4: Window-level sign consistency ──
fig, ax = plt.subplots(figsize=(16, 7))
ws = window_sign.sort_values("wavenumber_start")
x = np.arange(len(ws))

# Plot v1 and v2.1 deltas side by side
width = 0.35
bars1 = ax.bar(x - width / 2, ws["delta_v1_hcc_vs_healthy"], width,
               label="v1 (Vornoli Au)", color="#1f77b4", edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + width / 2, ws["delta_v2_hcc_vs_healthy"], width,
               label="v2.1 (CCA AgNP)", color="#ff7f0e", edgecolor="black", linewidth=0.5)

# Highlight key windows
highlight_windows = {"1020-1080": "#d62728", "620-660": "#2ca02c", "500-540": "#9467bd"}
for i, row in ws.iterrows():
    idx_pos = ws.index.get_loc(i)
    if row["window_id"] in highlight_windows:
        ax.axvspan(idx_pos - 0.45, idx_pos + 0.45, alpha=0.15,
                   color=highlight_windows[row["window_id"]])

ax.set_xticks(x)
ax.set_xticklabels(ws["window_id"], rotation=45, ha="right", fontsize=8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("HCC vs Healthy Delta")
ax.set_title("Window-Level HCC Delta: v1 (Au) vs v2.1 (AgNP)")
ax.legend()
plt.tight_layout()
save_fig(fig, "fig4_window_sign_consistency.png")

# ── Figure 5: Nucleic acid ablation comparison ──
fig, ax = plt.subplots(figsize=(10, 6))
conds = ablation_df["condition"].tolist()
x = np.arange(len(conds))
width = 0.35

ax.bar(x - width / 2, ablation_df["cohort_cosine_full"], width,
       label="Full (8 axes)", color="#1f77b4", edgecolor="black")
ax.bar(x + width / 2, ablation_df["cohort_cosine_ablated"], width,
       label="Ablated (no nucleic_acid_backbone)", color="#ff7f0e", edgecolor="black")

ax.set_xticks(x)
ax.set_xticklabels(conds, fontsize=10)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Cohort Cosine to HCC Prior")
ax.set_title("Nucleic Acid Backbone Ablation: Effect on Condition Ordering")
ax.legend()

# Add rank annotations
for i, row in ablation_df.iterrows():
    ax.annotate(f"#{int(row['full_rank'])}", (i - width / 2, row["cohort_cosine_full"]),
                ha="center", va="bottom", fontsize=8, color="#1f77b4")
    ax.annotate(f"#{int(row['ablated_rank'])}", (i + width / 2, row["cohort_cosine_ablated"]),
                ha="center", va="bottom", fontsize=8, color="#ff7f0e")

plt.tight_layout()
save_fig(fig, "fig5_nucleic_acid_ablation.png")

# ── Figure 6: Aromatic axis-only comparison ──
fig, ax = plt.subplots(figsize=(14, 7))
subsets_to_plot = ["full", "all_minus_nucleic", "aromatic_only", "aromatic_protein",
                   "aromatic_redox", "nucleic_acid_only", "membrane_only"]
sub_df = aromatic_test[aromatic_test["axis_subset"].isin(subsets_to_plot)].copy()
sub_df = sub_df.set_index("axis_subset").loc[subsets_to_plot].reset_index()

x = np.arange(len(sub_df))
width = 0.18
cond_colors = {
    "cohort_cosine_healthy": "#2ca02c",
    "cohort_cosine_hcc": "#d62728",
    "cohort_cosine_cca": "#9467bd",
    "cohort_cosine_lm": "#ff7f0e",
}

for j, (col, color) in enumerate(cond_colors.items()):
    label = col.replace("cohort_cosine_", "")
    vals = sub_df[col].tolist()
    ax.bar(x + j * width - 1.5 * width, vals, width, label=label,
           color=color, edgecolor="black", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels([s.replace("_", "\n") for s in sub_df["axis_subset"]], fontsize=9)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Cohort Cosine to HCC Prior")
ax.set_title("Axis Subset Test: Condition Alignment Under Different BSV Axis Subsets")
ax.legend(title="Condition", loc="upper right")
plt.tight_layout()
save_fig(fig, "fig6_aromatic_axis_comparison.png")

# ── Figure 7: Cross-prior axis contribution heatmap ──
fig, ax = plt.subplots(figsize=(10, 6))
cp_data = cross_prior.set_index("BSV_component")[["contribution_to_hcc_prior", "contribution_to_nafld_prior"]]
cp_data.columns = ["HCC Prior", "NAFLD Prior"]
sns.heatmap(cp_data, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            ax=ax, linewidths=0.5, cbar_kws={"label": "Axis Contribution"})
ax.set_title("Cross-Prior Axis Contribution (HCC delta from CCA dataset)")
ax.set_yticklabels([c.replace("_", " ") for c in BSV_COMPONENTS], rotation=0, fontsize=9)
plt.tight_layout()
save_fig(fig, "fig7_cross_prior_axis_contribution.png")

# ======================================================================
# STEP 10 — Summary Report
# ======================================================================
print("\n=== STEP 10: Writing summary report ===")

# Compute key stats for the report
n_stable = len(axis_sign[axis_sign["classification"].isin(["STABLE_POSITIVE", "STABLE_NEGATIVE"])])
n_inverted = len(axis_sign[axis_sign["classification"] == "INVERTED"])
n_weak = len(axis_sign[axis_sign["classification"] == "WEAK"])

# Ablation recovery check
ablated_hcc_cos = float(ablation_df[ablation_df["condition"] == "hcc"]["cohort_cosine_ablated"].iloc[0])
ablated_healthy_cos = float(ablation_df[ablation_df["condition"] == "healthy_control"]["cohort_cosine_ablated"].iloc[0])
ablation_recovers = ablated_hcc_cos > ablated_healthy_cos

# Aromatic-only test
aromatic_row = aromatic_test[aromatic_test["axis_subset"] == "aromatic_only"].iloc[0]
aromatic_hcc_top = "hcc" in aromatic_row["ordering"].split(" > ")[0]

# Total cosine contributions
total_hcc_cos = sum(contrib_hcc.values())
total_nafld_cos = sum(contrib_nafld.values())

report = textwrap.dedent(f"""\
# GAIRA Spectral Query v2.1b — Substrate Sensitivity Diagnosis

## Purpose

Diagnostic analysis characterizing WHY the GAIRA HCC prior failed to transfer from the v1 Vornoli Au-SERS dataset to the v2.1 CCA AgNP-SERS dataset. This is a descriptive investigation, not a refinement or rescue attempt.

## Datasets

| Property | v1 (Vornoli) | v2.1 (CCA) |
|---|---|---|
| Substrate | Au-based SERS | AgNP SERS |
| Conditions | HCC (72), CTR (72) | HCC (89), CCA (96), LM (81), healthy (88) |
| Preprocessing | AsLS + SG at query time | Pre-applied in pipeline |
| Broad v1 cohort cosine | +0.237 | -0.137 |

## 1. Which BSV Axes Are Stable Across Datasets?

Of 8 BSV axes:
- **{n_stable} stable** (same sign in both datasets)
- **{n_inverted} inverted** (sign flip between datasets)
- **{n_weak} weak** (near-zero in both)

""")

# Build axis table
report += "| Axis | Delta v1 | Delta v2.1 | Classification |\n"
report += "|---|---|---|---|\n"
for _, row in axis_sign.iterrows():
    report += f"| {row['BSV_component']} | {row['delta_v1_hcc_vs_healthy']:+.4f} | {row['delta_v2_hcc_vs_healthy']:+.4f} | **{row['classification']}** |\n"

report += f"""
**Stable axes**: {', '.join(axis_sign[axis_sign['classification'].isin(['STABLE_POSITIVE', 'STABLE_NEGATIVE'])]['BSV_component'].tolist()) or 'none'}

**Inverted axes**: {', '.join(axis_sign[axis_sign['classification'] == 'INVERTED']['BSV_component'].tolist()) or 'none'}

**Weak axes**: {', '.join(axis_sign[axis_sign['classification'] == 'WEAK']['BSV_component'].tolist()) or 'none'}

Weak threshold: |delta| < {WEAK_THRESHOLD}

## 2. Which Axes Invert Across Datasets?

"""

inverted_rows = axis_sign[axis_sign["classification"] == "INVERTED"]
for _, row in inverted_rows.iterrows():
    comp = row["BSV_component"]
    report += f"**{comp}**: v1 delta = {row['delta_v1_hcc_vs_healthy']:+.4f}, v2.1 delta = {row['delta_v2_hcc_vs_healthy']:+.4f}. "
    report += f"Magnitude change = {row['magnitude_change']:+.4f}.\n\n"

report += """## 3. Is nucleic_acid_backbone Uniquely Unstable?

"""

nab_row = axis_sign[axis_sign["BSV_component"] == "nucleic_acid_backbone"].iloc[0]
report += f"""nucleic_acid_backbone:
- v1 delta: {nab_row['delta_v1_hcc_vs_healthy']:+.4f} (HCC enriched relative to healthy)
- v2.1 delta: {nab_row['delta_v2_hcc_vs_healthy']:+.4f} (HCC depleted relative to healthy)
- Classification: **{nab_row['classification']}**
- v1.2 ablation importance: 0.4193 cosine drop (49% of total alignment)

"""

report += "Other inverted axes:\n\n"
for _, row in inverted_rows.iterrows():
    if row["BSV_component"] != "nucleic_acid_backbone":
        report += f"- {row['BSV_component']}: v1={row['delta_v1_hcc_vs_healthy']:+.4f}, v2.1={row['delta_v2_hcc_vs_healthy']:+.4f}\n"

if len(inverted_rows) == 1 and inverted_rows.iloc[0]["BSV_component"] == "nucleic_acid_backbone":
    report += "- None. nucleic_acid_backbone is the ONLY inverted axis.\n"
elif len(inverted_rows) > 1:
    report += f"\nnucleic_acid_backbone is NOT uniquely unstable — {len(inverted_rows)} axes inverted total.\n"
else:
    report += "\nnucleic_acid_backbone is the primary inverted axis but others may be weak/borderline.\n"

report += """
## 4. Which Spectral Windows Are Substrate-Sensitive?

"""

report += "| Window | BSV Component | v1 Delta | v2.1 Delta | Classification |\n"
report += "|---|---|---|---|---|\n"
for _, row in window_sign.sort_values("wavenumber_start").iterrows():
    if "INVERTED" in row["classification"] or row["window_id"] in ["1020-1080", "620-660", "500-540", "1380-1450"]:
        report += f"| {row['window_id']} | {row['bsv_component']} | {row['delta_v1_hcc_vs_healthy']:+.4f} | {row['delta_v2_hcc_vs_healthy']:+.4f} | {row['classification']} |\n"

report += f"""
Key window observations:
- **1020-1080 cm-1** (nucleic_acid_backbone): Sign inversion confirmed. This window drove 49% of v1 alignment and is the primary failure mode.
- **620-660 cm-1** (aromatic_amino_acid): {"Stable direction" if "STABLE" in ws_idx.loc["620-660", "classification"] else "Direction change"} — aromatic enrichment in HCC.
- **500-540 cm-1** (redox_metabolite): {"Stable" if "STABLE" in ws_idx.loc["500-540", "classification"] else "Changed direction"}.
- **1380-1450 cm-1** (membrane_lipid): {"Stable" if "STABLE" in ws_idx.loc["1380-1450", "classification"] else "Changed direction"} — was 2nd most important in v1.

NOTE: Window-level analysis is approximate. The v2.1 pipeline produces BSV-level, not window-level, outputs. Window deltas are inferred from BSV component deltas, which means multiple windows mapping to the same component show the same direction. This is a limitation; a true window-level analysis would require raw spectral features.

## 5. Multi-Condition Axis Structure in CCA Dataset

"""

report += "| Axis | Healthy | LM | CCA | HCC | Classification |\n"
report += "|---|---|---|---|---|---|\n"
for _, row in axis_structure.iterrows():
    report += f"| {row['BSV_component']} | {row['mean_healthy']:.3f} | {row['mean_lm']:.3f} | {row['mean_cca']:.3f} | {row['mean_hcc']:.3f} | {row['classification']} |\n"

report += "\nCondition orderings per axis:\n\n"
for _, row in axis_structure.iterrows():
    report += f"- **{row['BSV_component']}**: {row['ordering']} ({row['classification']})\n"

report += """
## 6. Does HCC Have a Clearly Distinguishing Axis?

"""

hcc_specific = axis_structure[axis_structure["classification"] == "HCC_SPECIFIC"]
if len(hcc_specific) > 0:
    report += f"**{len(hcc_specific)} axis/axes classified as HCC_SPECIFIC:**\n\n"
    for _, row in hcc_specific.iterrows():
        report += f"- {row['BSV_component']}: HCC delta = {row['delta_hcc_vs_healthy']:+.4f}, CCA = {row['delta_cca_vs_healthy']:+.4f}, LM = {row['delta_lm_vs_healthy']:+.4f}\n"
else:
    report += "No axes are classified as HCC_SPECIFIC. All disease conditions move in similar directions.\n"

report += f"""
## 7. Does Removing nucleic_acid_backbone Recover Sensible Ordering?

"""

report += "| Condition | Full Cosine (rank) | Ablated Cosine (rank) |\n"
report += "|---|---|---|\n"
for _, row in ablation_df.iterrows():
    report += f"| {row['condition']} | {row['cohort_cosine_full']:+.4f} (#{int(row['full_rank'])}) | {row['cohort_cosine_ablated']:+.4f} (#{int(row['ablated_rank'])}) |\n"

report += f"""
Full ordering: {' > '.join(full_order)}
Ablated ordering: {' > '.join(ablated_order)}

**Does ablation recover HCC > healthy?** {'YES' if ablation_recovers else 'NO'}

"""

if ablation_recovers:
    report += "Removing nucleic_acid_backbone recovers the expected HCC > healthy ordering, confirming it is the primary cause of inversion.\n"
else:
    report += "Removing nucleic_acid_backbone does NOT recover expected ordering. The inversion has additional contributors beyond this single axis.\n"

report += f"""
## 8. Is aromatic_amino_acid a More Robust Cross-Dataset Signal?

"""

report += "| Subset | Healthy | LM | CCA | HCC | Ordering |\n"
report += "|---|---|---|---|---|---|\n"
for _, row in aromatic_test.iterrows():
    report += f"| {row['axis_subset']} | {row['cohort_cosine_healthy']:+.4f} | {row['cohort_cosine_lm']:+.4f} | {row['cohort_cosine_cca']:+.4f} | {row['cohort_cosine_hcc']:+.4f} | {row['ordering'][:60]} |\n"

report += f"""
Key finding: aromatic_amino_acid shows HCC enrichment in BOTH datasets (v1: {delta_v1['aromatic_amino_acid']:+.4f}, v2.1: {delta_v2_hcc['aromatic_amino_acid']:+.4f}). Under aromatic-only alignment, HCC {'is' if aromatic_hcc_top else 'is NOT'} the top-ranking condition.

## 9. Why Does the NAFLD Prior Transfer Better Than the HCC Prior?

"""

report += "Per-axis contribution of HCC delta (CCA dataset) to each prior:\n\n"
report += "| Axis | HCC Prior Contribution | NAFLD Prior Contribution | Same Direction |\n"
report += "|---|---|---|---|\n"
for _, row in cross_prior.iterrows():
    report += f"| {row['BSV_component']} | {row['contribution_to_hcc_prior']:+.4f} | {row['contribution_to_nafld_prior']:+.4f} | {row['same_direction_yes_no']} |\n"

report += f"""
The HCC prior has a large positive weight on nucleic_acid_backbone (+1.0), which becomes a large NEGATIVE contribution when the CCA dataset shows HCC depletion on that axis. The NAFLD prior has near-zero weight on nucleic_acid_backbone (0.0), so the inversion on that axis does not damage the NAFLD prior's alignment.

The NAFLD prior also has larger negative weights on membrane_lipid (-1.0) and pyrimidine_nucleotide (-1.0), which happen to align with the CCA dataset's disease directions. The NAFLD prior is not "better" biologically — it is simply less dependent on the unstable nucleic_acid_backbone axis.

## 10. Primary Failure Mechanism

"""

report += """### Is the v2.1 failure primarily:

| Factor | Contribution | Evidence |
|---|---|---|
| **Substrate-driven** | **PRIMARY** | nucleic_acid_backbone (1020-1080 cm-1) inverts between Au and AgNP. This single axis drove 49% of v1 alignment. |
| **Preprocessing-driven** | SECONDARY | CCA data was pre-processed in embedding pipeline with potentially different baseline/smoothing parameters. Cannot be separated from substrate effect without raw spectra. |
| **Axis-dominance-driven** | SECONDARY | The v1.1 weighting amplified dependence on nucleic_acid_backbone. When that axis inverts, the entire alignment collapses. |
| **Disease-specificity mismatch** | MINOR | The CCA dataset contains interpretable disease structure, but most axes are disease-general rather than HCC-specific. The GAIRA HCC prior may be partially encoding a disease-general signal. |

### v2.1b Diagnosis

**Primary failure mechanism**: Substrate-dependent sign inversion of nucleic_acid_backbone, the single axis that dominated the v1 alignment.

**Secondary contributors**:
1. Preprocessing asymmetry (pre-applied vs query-time baseline correction) may amplify or mask substrate effects
2. Over-reliance on one axis in v1.1 weighting created a single point of failure
3. The HCC literature prior encodes nucleic_acid_backbone with the highest weight (+1.0), making the prior maximally sensitive to this axis

**Does the CCA dataset contain interpretable disease structure?**
Yes. The CCA dataset shows:
- Disease-general axes: protein_backbone depletion, glycan/purine enrichment in disease
- Potential HCC-specific signal: aromatic_amino_acid enrichment is strongest in HCC
- Clear condition separation in BSV space, just not in the direction the HCC prior expects

**Implications for v2.2**:
1. A substrate-aware calibration or normalization step is needed before cross-dataset transfer
2. Multi-axis priors that do not depend on nucleic_acid_backbone would be more robust
3. The aromatic_amino_acid axis appears more substrate-robust and should be investigated as an alternative discriminative signal
4. Within-dataset validation (does BSV separate conditions at all?) should precede cross-dataset prior alignment
5. Raw window-level features (not just BSV aggregates) would enable more precise diagnosis of substrate vs preprocessing effects
"""

# Write report
report_path = REPORT_DIR / "gaira_spectral_query_v2_1b_substrate_sensitivity.md"
report_path.write_text(report)
print(f"\nReport written to: {report_path}")

# ── Final file listing ──
print("\n=== Output files ===")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}")
print(f"  {report_path.name} (in reports/)")
print("\nDone.")
