"""
Spectral band driver analysis — identifies which windows drive cohort BSV differences.

Data-driven: computed from spectral window features only, no motif logic.
"""
from __future__ import annotations

import numpy as np

from gaira.spectral.window_panel import BSV_COMPONENTS, WINDOW_DEFS


def compute_window_importance(
    window_features: np.ndarray,
    cohorts: np.ndarray,
    reference: str,
) -> list[dict]:
    """Rank windows by their contribution to disease-vs-reference separation.

    For each window, computes:
    - delta (cohort mean - reference mean)
    - F-ratio (between-group / within-group variance)

    Returns list sorted by importance (descending).
    """
    ref_mask = cohorts == reference
    if ref_mask.sum() == 0:
        return []

    ref_mean = window_features[ref_mask].mean(axis=0)
    dis_mask = ~ref_mask
    dis_mean = window_features[dis_mask].mean(axis=0)

    results = []
    for i, (wid, ws, we, comp) in enumerate(WINDOW_DEFS):
        delta = float(dis_mean[i] - ref_mean[i])

        # F-ratio
        grand = window_features[:, i].mean()
        n_ref, n_dis = ref_mask.sum(), dis_mask.sum()
        ss_between = n_ref * (ref_mean[i] - grand)**2 + n_dis * (dis_mean[i] - grand)**2
        ss_within = ((window_features[ref_mask, i] - ref_mean[i])**2).sum() + \
                     ((window_features[dis_mask, i] - dis_mean[i])**2).sum()
        f_ratio = (ss_between / 1) / (ss_within / max(len(cohorts) - 2, 1)) if ss_within > 0 else 0

        results.append({
            "window_id": wid,
            "wavenumber_start": ws,
            "wavenumber_end": we,
            "bsv_component": comp,
            "delta": round(delta, 6),
            "f_ratio": round(float(f_ratio), 4),
            "direction": "enriched" if delta > 0 else "depleted",
        })

    results.sort(key=lambda r: -r["f_ratio"])
    return results


def compute_per_cohort_window_importance(
    window_features: np.ndarray,
    cohorts: np.ndarray,
    reference: str,
) -> dict[str, list[dict]]:
    """Compute window importance for each non-reference cohort separately."""
    ref_mask = cohorts == reference
    if ref_mask.sum() == 0:
        return {}

    ref_mean = window_features[ref_mask].mean(axis=0)
    result = {}

    for coh in sorted(set(cohorts)):
        if coh == reference:
            continue
        coh_mask = cohorts == coh
        coh_mean = window_features[coh_mask].mean(axis=0)

        windows = []
        for i, (wid, ws, we, comp) in enumerate(WINDOW_DEFS):
            delta = float(coh_mean[i] - ref_mean[i])
            # Effect size relative to pooled std
            pooled = np.concatenate([window_features[ref_mask, i], window_features[coh_mask, i]])
            std = pooled.std()
            effect = delta / std if std > 1e-12 else 0.0

            windows.append({
                "window_id": wid, "wavenumber_start": ws, "wavenumber_end": we,
                "bsv_component": comp, "delta": round(delta, 6),
                "effect_size": round(float(effect), 4),
                "direction": "enriched" if delta > 0 else "depleted",
            })

        windows.sort(key=lambda w: -abs(w["effect_size"]))
        result[coh] = windows

    return result


def top_windows_per_axis(
    window_importance: list[dict],
    top_n: int = 3,
) -> dict[str, list[dict]]:
    """Group top contributing windows by BSV axis."""
    by_axis: dict[str, list[dict]] = {c: [] for c in BSV_COMPONENTS}
    for w in window_importance:
        comp = w["bsv_component"]
        if comp in by_axis:
            by_axis[comp].append(w)
    # Sort each by F-ratio and keep top N
    return {axis: sorted(wins, key=lambda w: -w["f_ratio"])[:top_n]
            for axis, wins in by_axis.items() if wins}
