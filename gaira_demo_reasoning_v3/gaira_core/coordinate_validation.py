"""GAIRA Demo v3 — coordinate validation & analysis helpers.

Pure analysis over the frozen calibration and the reference-samples table.
Used by tests, the validation UI mode, and the reports. Reads only; never
refits the calibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from . import global_coordinates as gc

REF_SAMPLES = cfg.GENERATED_DIR / "global_coordinate_reference_samples_v1.csv"


def load_reference_samples() -> pd.DataFrame | None:
    if REF_SAMPLES.exists() and REF_SAMPLES.stat().st_size > 0:
        return pd.read_csv(REF_SAMPLES)
    return None


def raw_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[[f"raw_{a}" for a in cfg.BSV_AXES]].to_numpy(float)


def global_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[[f"global_{a}" for a in cfg.BSV_AXES]].to_numpy(float)


# ── axis dominance / variance before vs after calibration ──────────────

def variance_before_after(df: pd.DataFrame) -> pd.DataFrame:
    R, G = raw_matrix(df), global_matrix(df)
    rows = []
    raw_var = R.var(axis=0)
    glob_var = G.var(axis=0)
    raw_range = R.max(axis=0) - R.min(axis=0)
    glob_range = G.max(axis=0) - G.min(axis=0)
    raw_rank = (-raw_var).argsort().argsort() + 1        # 1 = largest variance
    glob_rank = (-glob_var).argsort().argsort() + 1
    for j, a in enumerate(cfg.BSV_AXES):
        rows.append(dict(axis=a, axis_short=cfg.axis_short(a),
                         raw_variance=float(raw_var[j]), global_variance=float(glob_var[j]),
                         raw_dyn_range=float(raw_range[j]), global_dyn_range=float(glob_range[j]),
                         raw_var_rank=int(raw_rank[j]), global_var_rank=int(glob_rank[j])))
    return pd.DataFrame(rows)


def redox_dominance(df: pd.DataFrame, redox_axis="G10_sulfur_thiol_redox") -> dict:
    va = variance_before_after(df)
    row = va[va.axis == redox_axis].iloc[0]
    R, G = raw_matrix(df), global_matrix(df)
    j = list(cfg.BSV_AXES).index(redox_axis)
    return {
        "redox_axis": redox_axis,
        "raw_dynamic_range": float(row["raw_dyn_range"]),
        "global_dynamic_range": float(row["global_dyn_range"]),
        "raw_variance_rank": int(row["raw_var_rank"]),
        "global_variance_rank": int(row["global_var_rank"]),
        "raw_max_abs": float(np.max(np.abs(R[:, j]))),
        "global_max_abs": float(np.max(np.abs(G[:, j]))),
        "n_axes": len(cfg.BSV_AXES),
    }


# ── effect sizes between labelled groups (on global coords; labels not used in fit) ──

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if sp < 1e-12:
        return 0.0
    return float((x.mean() - y.mean()) / sp)


def group_effect_sizes(df: pd.DataFrame, label_col: str, a: str, b: str,
                       coord_prefix: str = "global_") -> pd.DataFrame:
    ga = df[df[label_col] == a]
    gb = df[df[label_col] == b]
    rows = []
    for ax in cfg.BSV_AXES:
        col = f"{coord_prefix}{ax}"
        d = cohens_d(ga[col].to_numpy(float), gb[col].to_numpy(float))
        rows.append(dict(axis=ax, axis_short=cfg.axis_short(ax),
                         mean_a=float(ga[col].mean()), mean_b=float(gb[col].mean()),
                         cohens_d=d))
    return pd.DataFrame(rows).sort_values("cohens_d", key=lambda s: s.abs(), ascending=False)


# ── cohort invariance (the central V3 property) ────────────────────────

def invariance_check(raw_bsv: dict, calib, comparison_sets: list[list[dict]],
                     atol: float = 1e-9) -> dict:
    """Confirm a sample's global coords are identical regardless of the
    comparison set it is shown with (global coords are a pure function of the
    frozen calibration). Also confirms cohort-relative coords DO change.
    """
    base = gc.global_unbounded_vector(raw_bsv, calib)
    max_dev = 0.0
    for cs in comparison_sets:
        withset = gc.global_unbounded_vector(raw_bsv, calib)   # ignores set by design
        max_dev = max(max_dev, float(np.max(np.abs(withset - base))))
    # cohort-relative SHOULD differ across sets
    cr_devs = []
    for cs in comparison_sets:
        z_alone = gc.cohort_relative_zscores([raw_bsv])[0]
        z_withset = gc.cohort_relative_zscores([raw_bsv] + cs)[0]
        cr_devs.append(max(abs(z_alone[a] - z_withset[a]) for a in cfg.BSV_AXES))
    return {
        "global_max_deviation": max_dev,
        "global_invariant": max_dev <= atol,
        "cohort_relative_max_deviation": float(max(cr_devs)) if cr_devs else 0.0,
        "cohort_relative_changes": (max(cr_devs) > atol) if cr_devs else False,
    }


# ── nuisance association (diagnostic; eta^2 of a categorical on each axis) ──

def nuisance_eta_squared(df: pd.DataFrame, nuisance_col: str,
                         coord_prefix: str = "global_") -> pd.DataFrame:
    rows = []
    groups = [g for _, g in df.groupby(nuisance_col)]
    for ax in cfg.BSV_AXES:
        col = f"{coord_prefix}{ax}"
        allv = df[col].to_numpy(float)
        grand = allv.mean()
        ss_tot = float(((allv - grand) ** 2).sum())
        ss_between = float(sum(len(g) * (g[col].mean() - grand) ** 2 for g in groups))
        eta2 = (ss_between / ss_tot) if ss_tot > 1e-12 else 0.0
        rows.append(dict(axis=ax, axis_short=cfg.axis_short(ax),
                         eta_squared=float(eta2)))
    return pd.DataFrame(rows).sort_values("eta_squared", ascending=False)
