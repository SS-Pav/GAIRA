"""Stage B0 — rejection rules, Pareto front and final candidate selection.

Selection uses the INNER validation folds only. A candidate must first survive the
hard rejection rules (spectral collapse, replicate destruction, peak damage,
over-subtraction), then compete on the Pareto front, where ties are broken toward
the SIMPLEST pipeline.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Rejection thresholds — frozen before the search is run.
REJECT = {
    "replicate_drop_frac": 0.90,      # replicate cosine must stay >= 0.90 x baseline
    "chem_drop_frac": 0.90,           # within-modality analyte 1-NN >= 0.90 x baseline
    "min_peak_retention": 0.90,
    "max_peak_invention": 0.02,
    "max_width_ratio": 1.25,          # median peak width may not broaden > 25%
    "min_effective_rank_frac": 0.70,  # vs baseline effective rank
    "max_duplicate_frac": 0.05,
    "max_negative_lobe": 0.60,
    "max_edge_artefact": 3.0,
}


def apply_rejection(df, base_row):
    """Flag candidates violating any hard rule. Returns df with reject columns."""
    d = df.copy()
    def _b(k, default=np.nan):
        v = base_row.get(k, default)
        return v if (v is not None and np.isfinite(v)) else np.nan

    rr = _b("rep_raman_replicate_cos"); rs = _b("rep_sers_replicate_cos")
    cr = _b("chem_raman_1nn"); cs = _b("chem_sers_1nn"); er = _b("si_effective_rank")

    reasons = []
    for _, r in d.iterrows():
        why = []
        if np.isfinite(rr) and r.get("rep_raman_replicate_cos", np.nan) < REJECT["replicate_drop_frac"] * rr:
            why.append("raman_replicate_drop")
        if np.isfinite(rs) and r.get("rep_sers_replicate_cos", np.nan) < REJECT["replicate_drop_frac"] * rs:
            why.append("sers_replicate_drop")
        if np.isfinite(cr) and r.get("chem_raman_1nn", np.nan) < REJECT["chem_drop_frac"] * cr:
            why.append("raman_chemistry_drop")
        if np.isfinite(cs) and r.get("chem_sers_1nn", np.nan) < REJECT["chem_drop_frac"] * cs:
            why.append("sers_chemistry_drop")
        if r.get("si_peak_retention", 1) < REJECT["min_peak_retention"]:
            why.append("peak_loss")
        if r.get("si_peak_invention", 0) > REJECT["max_peak_invention"]:
            why.append("peak_invention")
        if r.get("si_peak_width_ratio", 1) > REJECT["max_width_ratio"]:
            why.append("peak_broadening")
        if np.isfinite(er) and r.get("si_effective_rank", np.nan) < REJECT["min_effective_rank_frac"] * er:
            why.append("rank_collapse")
        if r.get("si_cross_analyte_duplicate_frac", 0) > REJECT["max_duplicate_frac"]:
            why.append("analyte_collapse")
        if r.get("si_negative_lobe_burden", 0) > REJECT["max_negative_lobe"]:
            why.append("over_subtraction")
        if r.get("si_edge_artefact_ratio", 0) > REJECT["max_edge_artefact"]:
            why.append("edge_artefact")
        reasons.append("|".join(why))
    d["reject_reasons"] = reasons
    d["rejected"] = d.reject_reasons.str.len() > 0
    return d


# Pareto objectives: (column, maximize?)
PARETO_OBJECTIVES = [
    ("cm_mrr", True),                       # cross-modal retrieval
    ("pk_effect_vs_mismatched", True),      # matched-specific peak correspondence
    ("rep_sers_replicate_cos", True),       # Ag-SERS replicate preservation
    ("chem_raman_1nn", True),               # within-modality chemistry
    ("si_peak_retention", True),            # spectral integrity
    ("n_stages", False),                    # simplicity
]


def pareto_front(df, objectives=PARETO_OBJECTIVES):
    """Non-dominated set. Rows with NaN in an objective are excluded from the front."""
    cols = [c for c, _ in objectives]
    d = df.dropna(subset=cols).copy()
    if d.empty:
        return d.assign(on_front=False)
    V = d[cols].values.astype(float)
    sign = np.array([1.0 if mx else -1.0 for _, mx in objectives])
    V = V * sign
    n = len(V); front = np.ones(n, dtype=bool)
    for i in range(n):
        if not front[i]:
            continue
        dom = np.all(V >= V[i], axis=1) & np.any(V > V[i], axis=1)
        if dom.any():
            front[i] = False
    d["on_front"] = front
    return d


def select(df, base_row, prefer_simple=True):
    """Apply rejection, compute the front, and pick the simplest eligible candidate
    that improves cross-modal MRR and matched-specificity over the baseline."""
    d = apply_rejection(df, base_row)
    elig = d[~d.rejected].copy()
    b_mrr = base_row.get("cm_mrr", np.nan)
    b_eff = base_row.get("pk_effect_vs_mismatched", np.nan)
    elig = elig[(elig.cm_mrr > b_mrr) & (elig.pk_effect_vs_mismatched >= b_eff)]
    if elig.empty:
        return {"selected": None, "reason": "no candidate improved MRR and peak specificity "
                                            "over the baseline while passing rejection rules",
                "n_eligible": 0}, d
    pf = pareto_front(elig)
    front = pf[pf.on_front]
    if front.empty:
        front = pf
    if prefer_simple:
        front = front.sort_values(["n_stages", "cm_mrr"], ascending=[True, False])
        # among the simplest tier, take the best MRR
        simplest = front[front.n_stages == front.n_stages.min()]
        pick = simplest.sort_values("cm_mrr", ascending=False).iloc[0]
    else:
        pick = front.sort_values("cm_mrr", ascending=False).iloc[0]
    return {"selected": pick.cid, "n_eligible": int(len(elig)),
            "n_on_front": int(front.on_front.sum() if "on_front" in front else len(front)),
            "delta_mrr_vs_baseline": float(pick.cm_mrr - b_mrr),
            "delta_peak_effect": float(pick.pk_effect_vs_mismatched - b_eff)}, d
