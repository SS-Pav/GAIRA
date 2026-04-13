"""
BSV comparison — observed spectral vs expected literature.

Primary mode: raw observed BSV. Expected BSV is post-hoc comparator only.
Shared min-max scaling for cross-space visualization.
Delta comparison: observed shift vs expected shift relative to a reference cohort.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from gaira.spectral.expected_bsv import BSV_COMPONENTS, ExpectedComparator


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _to_vec(bsv: dict[str, float]) -> np.ndarray:
    return np.array([bsv.get(c, 0.0) for c in BSV_COMPONENTS], dtype=float)


def _zscore(v: np.ndarray) -> np.ndarray:
    m, s = v.mean(), v.std()
    return (v - m) / s if s > 1e-12 else np.zeros_like(v)


def _rank_norm(v: np.ndarray) -> np.ndarray:
    return rankdata(v, method="average") / len(v)


# ── Shared min-max scaling ─────────────────────────────────────────────

def shared_minmax_scale(
    obs_bsv: dict[str, float],
    exp_bsv: dict[str, float],
    all_obs_bsvs: dict[str, dict[str, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale observed and expected BSV into a shared [0,1] display range per axis.

    For each BSV axis, the display range spans from the minimum to the maximum
    of: all observed cohort values for that axis, plus the expected comparator
    value. This preserves relative structure while making both profiles comparable.

    Args:
        obs_bsv: The observed cohort BSV to compare.
        exp_bsv: The expected literature BSV.
        all_obs_bsvs: All observed cohort BSVs from the dataset (for range computation).

    Returns:
        (obs_scaled, exp_scaled) as numpy arrays in [0,1] range per axis.
    """
    obs = _to_vec(obs_bsv)
    exp = _to_vec(exp_bsv)

    # Collect all values per axis for range computation
    per_axis_vals: list[list[float]] = [[] for _ in BSV_COMPONENTS]

    if all_obs_bsvs:
        for coh_bsv in all_obs_bsvs.values():
            for i, c in enumerate(BSV_COMPONENTS):
                per_axis_vals[i].append(coh_bsv.get(c, 0.0))

    # Also include the current observed and expected
    for i in range(len(BSV_COMPONENTS)):
        per_axis_vals[i].append(float(obs[i]))
        per_axis_vals[i].append(float(exp[i]))

    obs_scaled = np.zeros(len(BSV_COMPONENTS))
    exp_scaled = np.zeros(len(BSV_COMPONENTS))

    for i in range(len(BSV_COMPONENTS)):
        lo = min(per_axis_vals[i])
        hi = max(per_axis_vals[i])
        rng = hi - lo
        if rng < 1e-12:
            obs_scaled[i] = 0.5
            exp_scaled[i] = 0.5
        else:
            obs_scaled[i] = (obs[i] - lo) / rng
            exp_scaled[i] = (exp[i] - lo) / rng

    return obs_scaled, exp_scaled


# ── Delta comparison ───────────────────────────────────────────────────

def compute_delta_comparison(
    obs_cohort_bsv: dict[str, float],
    obs_ref_bsv: dict[str, float],
    exp_cohort_bsv: dict[str, float],
    exp_ref_bsv: dict[str, float],
) -> dict:
    """Compare observed delta (cohort - ref) vs expected delta (cohort - ref).

    Returns dict with observed_delta, expected_delta, delta_cosine, per_axis.
    """
    obs_c = _to_vec(obs_cohort_bsv)
    obs_r = _to_vec(obs_ref_bsv)
    exp_c = _to_vec(exp_cohort_bsv)
    exp_r = _to_vec(exp_ref_bsv)

    obs_delta = obs_c - obs_r
    exp_delta = exp_c - exp_r
    delta_cos = _cosine(obs_delta, exp_delta)

    # Per-axis agreement on deltas
    per_axis = []
    max_abs = max(max(abs(obs_delta).max(), 1e-12), max(abs(exp_delta).max(), 1e-12))

    for i, comp in enumerate(BSV_COMPONENTS):
        od = float(obs_delta[i])
        ed = float(exp_delta[i])
        both_small = abs(od) / max_abs < 0.1 and abs(ed) / max_abs < 0.1
        same_sign = (od >= 0 and ed >= 0) or (od < 0 and ed < 0)
        rel_diff = abs(od - ed) / max_abs

        if both_small:
            cat = "weak"
        elif same_sign and rel_diff < 0.3:
            cat = "aligned"
        elif same_sign:
            cat = "partial"
        else:
            cat = "divergent"

        per_axis.append({
            "component": comp,
            "obs_delta": round(od, 6),
            "exp_delta": round(ed, 4),
            "category": cat,
        })

    return {
        "observed_delta": {BSV_COMPONENTS[i]: round(float(obs_delta[i]), 6) for i in range(8)},
        "expected_delta": {BSV_COMPONENTS[i]: round(float(exp_delta[i]), 4) for i in range(8)},
        "delta_cosine": round(delta_cos, 4),
        "per_axis": per_axis,
    }


# ── Normalization modes (kept for advanced use) ────────────────────────

NORM_MODES = {
    "raw": "Raw (primary biology)",
    "zscore": "Z-score (relative prominence; advanced)",
    "rank": "Rank (profile shape only; advanced)",
}


def normalize_pair(obs_bsv, exp_bsv, mode):
    obs, exp = _to_vec(obs_bsv), _to_vec(exp_bsv)
    if mode == "zscore":
        return _zscore(obs), _zscore(exp)
    elif mode == "rank":
        return _rank_norm(obs), _rank_norm(exp)
    return obs, exp


def compute_similarity(obs_bsv, exp_bsv, mode="raw"):
    obs_n, exp_n = normalize_pair(obs_bsv, exp_bsv, mode)
    return round(_cosine(obs_n, exp_n), 4)


def compute_all_mode_similarities(obs_bsv, exp_bsv):
    return {m: compute_similarity(obs_bsv, exp_bsv, m) for m in NORM_MODES}


def compute_cross_matrix_normalized(cohort_bsvs, expected_comparators, mode="raw"):
    """Compute cohort × expected similarity matrix."""
    exp_profiles, exp_labels = {}, []
    for coh, exp in expected_comparators.items():
        if exp.bsv is not None and exp.comparator_name not in exp_profiles:
            exp_profiles[exp.comparator_name] = exp.bsv
            exp_labels.append(exp.comparator_name)
    if not exp_labels:
        return {"observed_labels": [], "expected_labels": [], "matrix": [], "alignment_summary": []}

    observed_labels = list(cohort_bsvs.keys())
    matrix = [[compute_similarity(cohort_bsvs[oc], exp_profiles[en], mode)
                for en in exp_labels] for oc in observed_labels]

    alignment = []
    for i, obs_coh in enumerate(observed_labels):
        ec = expected_comparators.get(obs_coh)
        if not ec or ec.bsv is None or ec.comparator_name not in exp_labels:
            continue
        own_idx = exp_labels.index(ec.comparator_name)
        own_cos = matrix[i][own_idx]
        best_alt_cos, best_alt_name = -2.0, ""
        for j, en in enumerate(exp_labels):
            if j != own_idx and matrix[i][j] > best_alt_cos:
                best_alt_cos, best_alt_name = matrix[i][j], en
        alignment.append({
            "cohort": obs_coh, "own_expected": ec.comparator_name,
            "own_cosine": own_cos, "best_alt_expected": best_alt_name,
            "best_alt_cosine": round(best_alt_cos, 4),
            "margin": round(own_cos - best_alt_cos, 4) if best_alt_name else 0.0,
            "match_type": ec.match_type,
        })

    return {"observed_labels": observed_labels, "expected_labels": exp_labels,
            "matrix": matrix, "alignment_summary": alignment}


# ── Substrate context ──────────────────────────────────────────────────

DATASET_SUBSTRATE = {
    "hcc_holdout_vornoli2020": {"substrate": "Au nanoparticles", "compatibility": "favorable",
        "note": "Gold-substrate dataset. Literature has strong Au-SERS coverage."},
    "cca_hcc_lm_serum_sers": {"substrate": "AgNP colloids", "compatibility": "mixed",
        "note": "Silver-substrate dataset. Some axes (nucleic_acid_backbone) are substrate-sensitive."},
    "diabetes_plasma_ev_sers": {"substrate": "Au (EV SERS)", "compatibility": "uncertain",
        "note": "EV SERS compared to serum-dominated literature. Limited EV-specific coverage."},
}


def get_substrate_context(dataset_id):
    return DATASET_SUBSTRATE.get(dataset_id, {
        "substrate": "unknown", "compatibility": "uncertain",
        "note": "No substrate metadata available.",
    })


# ── Interpretation ─────────────────────────────────────────────────────

def generate_interpretation(alignment_summary, delta_results, substrate_ctx):
    """Generate scientific interpretation from alignment and delta analysis."""
    if not alignment_summary:
        return "No expected comparators available for alignment analysis."

    lines = []
    n_correct = sum(1 for a in alignment_summary if a["margin"] > 0)
    n_total = len(alignment_summary)

    if n_correct == n_total:
        lines.append(f"All {n_total} cohorts align preferentially with their own expected profile.")
    elif n_correct > 0:
        lines.append(f"{n_correct}/{n_total} cohorts show preferential alignment.")
    else:
        lines.append("No cohort aligns preferentially with its own expected profile.")

    # Delta results
    if delta_results:
        pos_deltas = [(c, d) for c, d in delta_results.items() if d["delta_cosine"] > 0]
        if pos_deltas:
            best = max(pos_deltas, key=lambda x: x[1]["delta_cosine"])
            aligned_axes = [a["component"] for a in best[1]["per_axis"] if a["category"] == "aligned"]
            if aligned_axes:
                lines.append(
                    f"Disease-vs-reference shift for {best[0].replace('_', ' ')} tracks the "
                    f"expected literature shift (delta cosine {best[1]['delta_cosine']:+.3f}), "
                    f"driven by {', '.join(aligned_axes[:3])}."
                )
        neg_deltas = [(c, d) for c, d in delta_results.items() if d["delta_cosine"] <= 0]
        if neg_deltas:
            worst = min(neg_deltas, key=lambda x: x[1]["delta_cosine"])
            div_axes = [a["component"] for a in worst[1]["per_axis"] if a["category"] == "divergent"]
            if div_axes:
                lines.append(
                    f"Shift for {worst[0].replace('_', ' ')} diverges from expected "
                    f"(delta cosine {worst[1]['delta_cosine']:+.3f}), "
                    f"particularly on {', '.join(div_axes[:3])}."
                )

    compat = substrate_ctx.get("compatibility", "uncertain")
    if compat == "mixed":
        lines.append("Substrate mismatch may contribute to axis divergence.")
    elif compat == "uncertain":
        lines.append("Substrate/sample context is uncertain — interpret with caution.")

    return " ".join(lines)
