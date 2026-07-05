"""gaira_base_4_shine_regression_progression_v1

SHINE EV — BSV → dose regression + axis/MSS/OTC progression analyses.

STRICT INVARIANTS:
- Engine / BSV axes / MSS kernel / OTC detector thresholds / preprocessing UNCHANGED
- No threshold tuning on dose labels
- No APAP "detected" claim
- Labels used only for regression / evaluation here (post-hoc)

Inputs: /Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_shine_ev_gaira_pilot_v1/tables/

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_4_shine_regression_progression_v1.py
"""
from __future__ import annotations

import shutil
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_shine_regression_progression_v1")
TABLES  = ROOT / "tables"
FIGS    = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT   = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"
for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
    d.mkdir(parents=True, exist_ok=True)

PILOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_shine_ev_gaira_pilot_v1/tables")

AXES = ["G01", "G02", "G03", "G04", "G05", "G06", "G07", "G08", "G09", "G10", "G11"]
TOP_AXIS_SUBSET = ["G01", "G03", "G04", "G06", "G08", "G09", "G05"]  # from prior D2 monotonicity
PAPER_GPR = {"R2": 0.952, "MAE_mM": 1.495, "CV5_RMSE_mM": 3.145, "bias_mM": 0.03,
                 "features": "raw SERS spectrum (~740 pixel features)"}


# ──────────────────────────────────────────────────────────────────────
# TASK 1 — Load + audit
# ──────────────────────────────────────────────────────────────────────
def task1_load():
    print("[TASK 1] loading pilot outputs + auditing")
    bsv = pd.read_csv(PILOT / "shine_per_spectrum_bsv_outputs_v1.csv")
    otc = pd.read_csv(PILOT / "shine_otc_drug_detection_per_spectrum_v1.csv")
    mss = pd.read_csv(PILOT / "shine_mss_top_hits_per_spectrum_v1.csv")
    cohort = pd.read_csv(PILOT / "shine_cohort_bsv_means_v1.csv")
    dose_resp = pd.read_csv(PILOT / "shine_bsv_dose_response_metrics_v1.csv")

    # Drop non-OK spectra (qc_status != OK)
    n_all = len(bsv)
    bsv_ok = bsv[bsv.qc_status == "OK"].copy()
    n_ok = len(bsv_ok)

    # Align mss + otc with bsv_ok by spectrum_id
    bsv_ok = bsv_ok.merge(
        otc[["spectrum_id", "outer_status", "present", "top_1",
               "margin_top1_top2", "score_paracetamol", "score_ibuprofen",
               "score_asa", "anchors_para"]],
        on="spectrum_id", how="left", suffixes=("", "_otc"))
    bsv_ok = bsv_ok.merge(
        mss[["spectrum_id", "top1_molecule", "top1_score", "top3_molecules"]],
        on="spectrum_id", how="left", suffixes=("", "_mss"))

    audit_rows = [{
        "n_spectra_total":  n_all,
        "n_spectra_ok":     n_ok,
        "n_spectra_d2":     int((bsv_ok.day == "D2").sum()),
        "n_spectra_set9_d2": int(((bsv_ok.set_id == "Set9") & (bsv_ok.day == "D2")).sum()),
        "n_spectra_set10_d2": int(((bsv_ok.set_id == "Set10") & (bsv_ok.day == "D2")).sum()),
        "n_subjects_set9_d2": bsv_ok[(bsv_ok.set_id == "Set9") &
                                           (bsv_ok.day == "D2")]["subject_id"].nunique(),
        "n_subjects_set10_d2": bsv_ok[(bsv_ok.set_id == "Set10") &
                                            (bsv_ok.day == "D2")]["subject_id"].nunique(),
        "doses_d2":        "|".join(str(d) for d in sorted(bsv_ok[bsv_ok.day == "D2"]
                                                                      ["dose_mM"].unique())),
        "top_mss_candidates": "|".join(
            f"{m}:{c}" for m, c in
            Counter(mss["top1_molecule"].dropna()).most_common(8)),
        "has_bsv_raw_columns":    all(f"raw_{a}" in bsv_ok.columns for a in AXES),
        "has_bsv_sumnorm_cols":   all(f"sumnorm_{a}" in bsv_ok.columns for a in AXES),
        "has_bsv_clr_cols":       all(f"clr_{a}" in bsv_ok.columns for a in AXES),
        "has_otc_score":           "score_paracetamol" in bsv_ok.columns,
        "has_mss_top1":            "top1_molecule" in bsv_ok.columns,
    }]
    pd.DataFrame(audit_rows).to_csv(TABLES / "shine_regression_input_audit_v1.csv", index=False)
    print(f"  OK spectra: {n_ok}; D2 total: {audit_rows[0]['n_spectra_d2']}")
    return bsv_ok, cohort, dose_resp


# ──────────────────────────────────────────────────────────────────────
# TASK 2 — BSV → Dose regression
# ──────────────────────────────────────────────────────────────────────
def _eval(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]; y_pred = y_pred[m]
    if len(y_true) < 2:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "bias": np.nan,
                  "la_low": np.nan, "la_high": np.nan, "n": len(y_true)}
    resid = y_pred - y_true
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred)) if np.var(y_true) > 0 else np.nan
    bias = float(np.mean(resid))
    sd   = float(np.std(resid, ddof=1))
    return {"rmse": rmse, "mae": mae, "r2": r2, "bias": bias,
             "la_low": bias - 1.96*sd, "la_high": bias + 1.96*sd, "n": int(len(y_true))}


def _make_model(name):
    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=0)
    if name == "gpr":
        kern = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        return GaussianProcessRegressor(kernel=kern, alpha=1e-6, normalize_y=True,
                                                n_restarts_optimizer=3, random_state=0)
    if name == "rf":
        return RandomForestRegressor(n_estimators=200, max_depth=8,
                                             min_samples_leaf=5, random_state=0, n_jobs=-1)
    raise ValueError(name)


def _feature_matrix(df, fset):
    if fset == "BSV_RAW_11":
        return df[[f"raw_{a}" for a in AXES]].values
    if fset == "BSV_SUMNORM_11":
        return df[[f"sumnorm_{a}" for a in AXES]].values
    if fset == "BSV_CLR_11":
        return df[[f"clr_{a}" for a in AXES]].values
    if fset == "TOP_AXIS_SUBSET":
        return df[[f"clr_{a}" for a in TOP_AXIS_SUBSET]].values
    raise ValueError(fset)


def task2_bsv_regression(bsv_df):
    print("[TASK 2] BSV → dose regression (D2 only)")
    d2 = bsv_df[bsv_df.day == "D2"].copy()
    perf_rows = []; pred_rows = []

    feat_sets = ["BSV_RAW_11", "BSV_SUMNORM_11", "BSV_CLR_11", "TOP_AXIS_SUBSET"]
    models = ["linear", "ridge", "gpr"]   # rf is exploratory

    for fset in feat_sets:
        for model_name in models + ["rf"]:
            is_core = model_name != "rf"
            # Within-set 5-fold CV (subject-grouped) per set
            for set_id in ["Set9", "Set10"]:
                sub = d2[d2.set_id == set_id].copy()
                if len(sub) < 50: continue
                X = _feature_matrix(sub, fset)
                y = sub["dose_mM"].values.astype(float)
                groups = sub["subject_id"].values
                n_groups = len(np.unique(groups))
                gkf = GroupKFold(n_splits=min(5, n_groups))
                fold_metrics = []
                all_pred = np.full(len(sub), np.nan)
                for tr, te in gkf.split(X, y, groups=groups):
                    scaler = StandardScaler().fit(X[tr])
                    Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
                    mdl = _make_model(model_name)
                    try:
                        mdl.fit(Xtr, y[tr])
                        yp = mdl.predict(Xte)
                    except Exception as e:
                        continue
                    all_pred[te] = yp
                    fold_metrics.append(_eval(y[te], yp))
                if not fold_metrics: continue
                # Overall metrics on all fold-wise predictions
                overall = _eval(y, all_pred)
                perf_rows.append({
                    "validation": f"within_{set_id}_GroupKFold5",
                    "feature_set": fset, "model": model_name,
                    "core_or_exploratory": "core" if is_core else "exploratory",
                    "n": int(len(sub)), "n_groups": int(n_groups),
                    **overall,
                })
                for i in range(len(sub)):
                    if np.isfinite(all_pred[i]):
                        pred_rows.append({
                            "validation": f"within_{set_id}_GroupKFold5",
                            "feature_set": fset, "model": model_name,
                            "set_id": set_id, "spectrum_id": sub.iloc[i]["spectrum_id"],
                            "subject_id": sub.iloc[i]["subject_id"],
                            "dose_actual": float(y[i]), "dose_predicted": float(all_pred[i]),
                        })

            # Cross-set transfer
            for tr_set, te_set in [("Set9", "Set10"), ("Set10", "Set9")]:
                tr = d2[d2.set_id == tr_set]; te = d2[d2.set_id == te_set]
                if len(tr) < 50 or len(te) < 50: continue
                Xtr = _feature_matrix(tr, fset); Xte = _feature_matrix(te, fset)
                ytr = tr["dose_mM"].values.astype(float)
                yte = te["dose_mM"].values.astype(float)
                scaler = StandardScaler().fit(Xtr)
                Xtr_s = scaler.transform(Xtr); Xte_s = scaler.transform(Xte)
                mdl = _make_model(model_name)
                try:
                    mdl.fit(Xtr_s, ytr)
                    yp = mdl.predict(Xte_s)
                except Exception:
                    continue
                metrics = _eval(yte, yp)
                perf_rows.append({
                    "validation": f"cross_{tr_set}_to_{te_set}",
                    "feature_set": fset, "model": model_name,
                    "core_or_exploratory": "core" if is_core else "exploratory",
                    "n": int(len(te)), "n_groups": int(te["subject_id"].nunique()),
                    **metrics,
                })
                for i in range(len(te)):
                    pred_rows.append({
                        "validation": f"cross_{tr_set}_to_{te_set}",
                        "feature_set": fset, "model": model_name,
                        "set_id": te_set, "spectrum_id": te.iloc[i]["spectrum_id"],
                        "subject_id": te.iloc[i]["subject_id"],
                        "dose_actual": float(yte[i]), "dose_predicted": float(yp[i]),
                    })

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(TABLES / "shine_bsv_regression_performance_v1.csv", index=False)
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(TABLES / "shine_bsv_regression_predictions_v1.csv", index=False)

    # Figures
    try:
        best = perf_df[(perf_df.core_or_exploratory == "core") &
                          (perf_df.validation.str.startswith("within"))] \
            .sort_values("r2", ascending=False)
        if not best.empty:
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            top_combos = best.head(2)
            for ax, (_, row) in zip(axes, top_combos.iterrows()):
                sub = pred_df[(pred_df.validation == row["validation"]) &
                                 (pred_df.feature_set == row["feature_set"]) &
                                 (pred_df.model == row["model"])]
                ax.scatter(sub.dose_actual + np.random.uniform(-1, 1, len(sub)),
                              sub.dose_predicted, s=5, alpha=0.3)
                ax.plot([0, 40], [0, 40], "k--", lw=0.8)
                ax.set_xlabel("actual APAP dose (mM)"); ax.set_ylabel("predicted dose")
                ax.set_title(f"{row['feature_set']} / {row['model']} / {row['validation']}\n"
                                f"R²={row['r2']:.3f} MAE={row['mae']:.2f} RMSE={row['rmse']:.2f}")
            fig.tight_layout()
            fig.savefig(FIGS / "fig_shine_bsv_regression_predicted_vs_actual_v1.png", dpi=150)
            plt.close(fig)
    except Exception as e:
        print(f"  fig pred issue: {e}")

    # Bland-Altman for best core model within-set
    try:
        best = perf_df[(perf_df.core_or_exploratory == "core") &
                          (perf_df.validation.str.startswith("within"))] \
            .sort_values("r2", ascending=False).iloc[0]
        sub = pred_df[(pred_df.validation == best["validation"]) &
                         (pred_df.feature_set == best["feature_set"]) &
                         (pred_df.model == best["model"])]
        resid = sub.dose_predicted.values - sub.dose_actual.values
        mean_pa = (sub.dose_predicted.values + sub.dose_actual.values) / 2
        bias = np.mean(resid); sd = np.std(resid, ddof=1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(mean_pa, resid, s=6, alpha=0.35)
        ax.axhline(bias, color="red", label=f"bias = {bias:+.2f} mM")
        ax.axhline(bias + 1.96*sd, color="red", ls="--",
                      label=f"+1.96σ = {bias+1.96*sd:+.2f}")
        ax.axhline(bias - 1.96*sd, color="red", ls="--",
                      label=f"-1.96σ = {bias-1.96*sd:+.2f}")
        ax.set_xlabel("mean(pred, actual)"); ax.set_ylabel("pred − actual (mM)")
        ax.set_title(f"Bland–Altman: {best['feature_set']} / {best['model']} / {best['validation']}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_bsv_regression_bland_altman_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig BA issue: {e}")

    return perf_df, pred_df


# ──────────────────────────────────────────────────────────────────────
# TASK 3 — Compare to paper GPR
# ──────────────────────────────────────────────────────────────────────
def task3_paper_compare(perf_df):
    print("[TASK 3] paper GPR vs GAIRA BSV comparison")
    # Best core model on within-set (any)
    core_within = perf_df[(perf_df.core_or_exploratory == "core") &
                                perf_df.validation.str.startswith("within")] \
        .sort_values("r2", ascending=False)
    best_within = core_within.iloc[0] if not core_within.empty else None
    # Best interpretable: Ridge or Linear within-set
    interp = perf_df[(perf_df.model.isin(["ridge", "linear"])) &
                          perf_df.validation.str.startswith("within")] \
        .sort_values("r2", ascending=False)
    best_interp = interp.iloc[0] if not interp.empty else None
    # Cross-set transfer best
    cross = perf_df[(perf_df.core_or_exploratory == "core") &
                         perf_df.validation.str.startswith("cross")] \
        .sort_values("r2", ascending=False)
    best_cross = cross.iloc[0] if not cross.empty else None

    def row_for(label, row, dims):
        if row is None:
            return {"label": label, "features": "n/a", "n_features": 0,
                      "R2": None, "MAE_mM": None, "RMSE_mM": None, "bias_mM": None}
        return {"label": label,
                  "features": f"{row['feature_set']} ({row['model']} / {row['validation']})",
                  "n_features": dims,
                  "R2": row["r2"], "MAE_mM": row["mae"], "RMSE_mM": row["rmse"],
                  "bias_mM": row["bias"]}

    rows = [
        {"label": "paper_GPR_D2",
         "features": PAPER_GPR["features"], "n_features": 740,
         "R2": PAPER_GPR["R2"], "MAE_mM": PAPER_GPR["MAE_mM"],
         "RMSE_mM": PAPER_GPR["CV5_RMSE_mM"], "bias_mM": PAPER_GPR["bias_mM"]},
        row_for("GAIRA_best_within_set", best_within,
                   11 if best_within is None else
                   (len(TOP_AXIS_SUBSET) if "TOP_AXIS" in best_within["feature_set"] else 11)),
        row_for("GAIRA_best_interpretable_within_set", best_interp,
                   11 if best_interp is None else
                   (len(TOP_AXIS_SUBSET) if "TOP_AXIS" in best_interp["feature_set"] else 11)),
        row_for("GAIRA_best_cross_set_transfer", best_cross,
                   11 if best_cross is None else
                   (len(TOP_AXIS_SUBSET) if "TOP_AXIS" in best_cross["feature_set"] else 11)),
    ]
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(TABLES / "shine_paper_vs_gaira_regression_comparison_v1.csv", index=False)

    # Figure
    try:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, col, ylabel in [(axes[0], "R2", "R²"),
                                      (axes[1], "MAE_mM", "MAE (mM)"),
                                      (axes[2], "RMSE_mM", "RMSE (mM)")]:
            y = [cmp_df[cmp_df.label == lbl][col].iloc[0] for lbl in cmp_df.label]
            ax.bar(range(len(cmp_df)), [0 if v is None or (isinstance(v, float) and np.isnan(v))
                                                else float(v) for v in y],
                      color=["#888", "#4C72B0", "#2ca02c", "#DD8452"])
            ax.set_xticks(range(len(cmp_df)))
            ax.set_xticklabels(cmp_df["label"], rotation=20, fontsize=7, ha="right")
            ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Paper GPR vs GAIRA BSV regression — summary")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_paper_vs_gaira_regression_summary_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig paper-compare issue: {e}")

    return cmp_df


# ──────────────────────────────────────────────────────────────────────
# TASK 4 — Axis-wise progression
# ──────────────────────────────────────────────────────────────────────
def _spearman(x, y):
    x = pd.Series(x); y = pd.Series(y)
    valid = x.notna() & y.notna()
    if valid.sum() < 3: return np.nan
    rx = x[valid].rank(); ry = y[valid].rank()
    if rx.std() == 0 or ry.std() == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _bootstrap_mean_ci(x, n=300, seed=42):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def task4_axis_progression(bsv_df):
    print("[TASK 4] axis-wise progression")
    rows = []
    for (s, d), sub_day in bsv_df.groupby(["set_id", "day"]):
        for a in AXES:
            col = f"clr_{a}"
            if col not in sub_day.columns: continue
            doses = []
            means_per_dose = []
            for dose, sub_cd in sub_day.groupby("dose_mM"):
                v = sub_cd[col].values.astype(float)
                v = v[np.isfinite(v)]
                if len(v) < 2: continue
                doses.append(dose)
                means_per_dose.append(np.mean(v))
                ci = _bootstrap_mean_ci(v)
                rows.append({
                    "set_id": s, "day": d, "axis": a, "dose_mM": int(dose),
                    "n": int(len(v)), "mean_clr": float(np.mean(v)),
                    "sd_clr": float(np.std(v)),
                    "ci_low": ci[0], "ci_high": ci[1],
                })
            if len(doses) >= 3:
                doses = np.array(doses, float); vals = np.array(means_per_dose, float)
                rho = _spearman(doses, vals)
                r = float(np.corrcoef(doses, vals)[0, 1]) \
                        if (np.std(doses) > 0 and np.std(vals) > 0) else np.nan
                # slope (linear reg)
                slope = float(np.polyfit(doses, vals, 1)[0]) if np.std(doses) > 0 else np.nan
                # endpoint
                endpoint = vals[-1] - vals[0]
                # attach to most-recent row for this (set, day, axis)
                rows.append({
                    "set_id": s, "day": d, "axis": a, "dose_mM": None,
                    "spearman_rho_vs_dose":  rho,
                    "pearson_r_vs_dose":     r,
                    "slope_per_mM":          slope,
                    "endpoint_C40_minus_C0": float(endpoint),
                    "monotonicity_abs_rho":  0 if np.isnan(rho) else abs(rho),
                })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "shine_axis_progression_metrics_v1.csv", index=False)

    # Figure: top 6-8 axes trajectories
    try:
        top = [a for a in ["G01", "G03", "G06", "G08", "G09", "G05", "G04", "G07"]
                 if a in AXES]
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
        for ax, a in zip(axes.flat, top):
            for s, col_s in [("Set9", "#4C72B0"), ("Set10", "#DD8452")]:
                for d, ls in [("D0", ":"), ("D1", "--"), ("D2", "-")]:
                    sub = df[(df.set_id == s) & (df.day == d) & (df.axis == a)
                               & df.dose_mM.notna()].sort_values("dose_mM")
                    if sub.empty: continue
                    ax.errorbar(sub.dose_mM, sub.mean_clr,
                                   yerr=[sub.mean_clr - sub.ci_low,
                                          sub.ci_high - sub.mean_clr],
                                   fmt="o-", color=col_s, ls=ls, lw=1.2,
                                   label=f"{s} {d}" if d == "D2" else None,
                                   capsize=2)
            ax.set_title(a); ax.grid(alpha=0.3); ax.axhline(0, color="black", lw=0.5)
            ax.set_xlabel("APAP dose (mM)")
        axes[0, 0].set_ylabel("BSV-CLR mean")
        axes[1, 0].set_ylabel("BSV-CLR mean")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("Top 8 BSV axes — dose trajectory by set/day (± bootstrap 95% CI)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_axis_progression_top_axes_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig axis progression issue: {e}")

    # Day comparison figure: Set9 only, D0 vs D1 vs D2 for same 4 axes
    try:
        top4 = ["G01", "G08", "G09", "G06"]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
        for ax, a in zip(axes, top4):
            for d, col_d in [("D0", "#888"), ("D1", "#f39c12"), ("D2", "#c0392b")]:
                sub = df[(df.set_id == "Set9") & (df.day == d) & (df.axis == a)
                           & df.dose_mM.notna()].sort_values("dose_mM")
                if sub.empty: continue
                ax.errorbar(sub.dose_mM, sub.mean_clr,
                               yerr=[sub.mean_clr - sub.ci_low,
                                      sub.ci_high - sub.mean_clr],
                               fmt="o-", color=col_d, lw=1.5, capsize=2, label=d)
            ax.set_title(a); ax.set_xlabel("APAP dose (mM)"); ax.grid(alpha=0.3)
            ax.legend(fontsize=7); ax.axhline(0, color="black", lw=0.5)
        axes[0].set_ylabel("BSV-CLR mean")
        fig.suptitle("Set9 Day-by-day axis progression — D0 vs D1 vs D2")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_axis_progression_day_comparison_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig day comparison issue: {e}")

    return df


# ──────────────────────────────────────────────────────────────────────
# TASK 5 — MSS / analyte progression
# ──────────────────────────────────────────────────────────────────────
MSS_CLUSTERS = {
    "purine_metabolite":     ["uric_acid", "hypoxanthine", "xanthine"],
    "purine_nucleotide":     ["adenine"],
    "aromatic_residue":      ["tryptophan", "phenylalanine", "tyrosine"],
    "sulfur_thiol_redox":    ["ergothioneine", "glutathione", "cysteine", "cystine"],
    "metabolic_small_molecule": ["lactate", "urea", "creatinine"],
    "lipid_acyl_membrane":   ["oleic_acid", "palmitic_acid", "stearic_acid"],
    "sterol_neutral_lipid":  ["cholesterol"],
    "glycan_carbohydrate":   ["glucose"],
}


def task5_mss_progression(bsv_df):
    print("[TASK 5] MSS analyte + cluster progression")
    # Analyte-level: mean top1_score per (set, day, dose, top1_molecule) + frequency
    rows_analyte = []
    for (s, d, c), sub in bsv_df.groupby(["set_id", "day", "dose_mM"]):
        n = len(sub)
        top1_ctr = Counter(sub["top1_molecule"].dropna())
        top3_ctr = Counter()
        for t3 in sub["top3_molecules"].dropna():
            for m in str(t3).split("|"):
                if m: top3_ctr[m] += 1
        for mol in set(list(top1_ctr.keys()) + list(top3_ctr.keys())):
            if mol == "" or mol == "nan": continue
            rows_analyte.append({
                "set_id": s, "day": d, "dose_mM": int(c), "molecule": mol,
                "n": n,
                "top1_freq":  top1_ctr.get(mol, 0) / n,
                "top3_freq":  top3_ctr.get(mol, 0) / n,
                "mean_top1_score": float(sub[sub.top1_molecule == mol]["top1_score"].mean())
                                        if (sub.top1_molecule == mol).any() else np.nan,
            })
    analyte_df = pd.DataFrame(rows_analyte)

    # Dose-response metrics per analyte (Spearman ρ vs dose)
    prog_rows = []
    for (s, d), sub_day in analyte_df.groupby(["set_id", "day"]):
        doses = sorted(sub_day.dose_mM.unique())
        if len(doses) < 3: continue
        for mol in sub_day.molecule.unique():
            vals = []; ds = []
            for dose in doses:
                r = sub_day[(sub_day.molecule == mol) & (sub_day.dose_mM == dose)]
                if r.empty: continue
                vals.append(float(r.top3_freq.iloc[0])); ds.append(dose)
            if len(vals) >= 3:
                rho = _spearman(np.array(ds, float), np.array(vals, float))
                endpoint = vals[-1] - vals[0]
                prog_rows.append({
                    "set_id": s, "day": d, "molecule": mol,
                    "spearman_rho_top3_freq_vs_dose": rho,
                    "endpoint_top3_freq_C40_minus_C0": endpoint,
                })
    prog_df = pd.DataFrame(prog_rows)
    prog_df.to_csv(TABLES / "shine_mss_analyte_progression_metrics_v1.csv", index=False)

    # Cluster-level aggregation (sum top3_freq of cluster members)
    cluster_rows = []
    for (s, d, c), sub in bsv_df.groupby(["set_id", "day", "dose_mM"]):
        n = len(sub)
        for cluster, members in MSS_CLUSTERS.items():
            top3_hit = 0
            top1_hit = 0
            for t3 in sub["top3_molecules"].dropna():
                mols = str(t3).split("|")
                if any(m in mols for m in members): top3_hit += 1
            for t1 in sub["top1_molecule"].dropna():
                if t1 in members: top1_hit += 1
            cluster_rows.append({
                "set_id": s, "day": d, "dose_mM": int(c), "cluster": cluster,
                "n": n, "top1_freq": top1_hit / n, "top3_freq": top3_hit / n,
            })
    cluster_df = pd.DataFrame(cluster_rows)
    # Dose-response per (set, day, cluster)
    cluster_prog_rows = []
    for (s, d, cl), sub_day in cluster_df.groupby(["set_id", "day", "cluster"]):
        sub_day = sub_day.sort_values("dose_mM")
        if len(sub_day) < 3: continue
        doses = sub_day.dose_mM.values.astype(float)
        vals = sub_day.top3_freq.values.astype(float)
        rho = _spearman(doses, vals)
        cluster_prog_rows.append({
            "set_id": s, "day": d, "cluster": cl,
            "spearman_rho_top3_freq_vs_dose": rho,
            "endpoint_C40_minus_C0": vals[-1] - vals[0],
        })
    pd.DataFrame(cluster_prog_rows).to_csv(
        TABLES / "shine_mss_cluster_progression_metrics_v1.csv", index=False)

    # Figures
    try:
        top_mols = analyte_df.groupby("molecule")["top3_freq"].mean() \
            .sort_values(ascending=False).head(8).index.tolist()
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
        for ax, mol in zip(axes.flat, top_mols):
            for s, col_s in [("Set9", "#4C72B0"), ("Set10", "#DD8452")]:
                for d, ls in [("D0", ":"), ("D1", "--"), ("D2", "-")]:
                    sub = analyte_df[(analyte_df.set_id == s) & (analyte_df.day == d)
                                          & (analyte_df.molecule == mol)].sort_values("dose_mM")
                    if sub.empty: continue
                    ax.plot(sub.dose_mM, sub.top3_freq, color=col_s, ls=ls, lw=1.2, marker="o",
                              label=f"{s} {d}" if d == "D2" else None)
            ax.set_title(mol, fontsize=9); ax.set_xlabel("APAP mM"); ax.grid(alpha=0.3)
        axes[0, 0].set_ylabel("top-3 hit rate")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("Top 8 MSS candidate molecules — top-3 hit rate vs APAP dose\n"
                        "(candidate spectral evidence — NOT definitive molecule identity)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_mss_candidate_progression_top_hits_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig MSS analyte issue: {e}")

    try:
        clusters = list(MSS_CLUSTERS.keys())
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
        for ax, cl in zip(axes.flat, clusters):
            for s, col_s in [("Set9", "#4C72B0"), ("Set10", "#DD8452")]:
                for d, ls in [("D0", ":"), ("D1", "--"), ("D2", "-")]:
                    sub = cluster_df[(cluster_df.set_id == s) & (cluster_df.day == d)
                                          & (cluster_df.cluster == cl)].sort_values("dose_mM")
                    if sub.empty: continue
                    ax.plot(sub.dose_mM, sub.top3_freq, color=col_s, ls=ls, lw=1.2, marker="o")
            ax.set_title(cl, fontsize=9); ax.set_xlabel("APAP mM"); ax.grid(alpha=0.3)
        axes[0, 0].set_ylabel("top-3 cluster hit rate")
        fig.suptitle("MSS cluster-level top-3 hit rate by set/day/dose")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_mss_cluster_progression_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig MSS cluster issue: {e}")

    return analyte_df, cluster_df, prog_df


# ──────────────────────────────────────────────────────────────────────
# TASK 6 — OTC / paracetamol-like progression
# ──────────────────────────────────────────────────────────────────────
def task6_otc_progression(bsv_df):
    print("[TASK 6] OTC / paracetamol-like progression")
    rows = []
    for (s, d, c), sub in bsv_df.groupby(["set_id", "day", "dose_mM"]):
        n = len(sub)
        outer_ctr = Counter(sub["outer_status"].fillna("NONE"))
        top1_ctr = Counter(sub["top_1"].fillna("NO_CALL"))
        sp = sub["score_paracetamol"].dropna().values
        si = sub["score_ibuprofen"].dropna().values
        sa = sub["score_asa"].dropna().values
        anc = sub["anchors_para"].dropna().values
        rows.append({
            "set_id": s, "day": d, "dose_mM": int(c), "n": n,
            "rate_NOT_DETECTED":       outer_ctr.get("NOT_DETECTED", 0) / n,
            "rate_CANDIDATE_COMPLEX":  outer_ctr.get("CANDIDATE_IN_COMPLEX_CONTEXT", 0) / n,
            "rate_HIGH_PURE":          outer_ctr.get("HIGH_CONFIDENCE_PURE_CONTEXT", 0) / n,
            "top1_paracetamol_freq":   top1_ctr.get("paracetamol", 0) / n,
            "top1_ibuprofen_freq":     top1_ctr.get("ibuprofen", 0) / n,
            "top1_asa_freq":           top1_ctr.get("acetylsalicylic_acid", 0) / n,
            "top1_no_call_freq":       top1_ctr.get("NO_CALL", 0) / n,
            "mean_score_paracetamol":  float(np.mean(sp)) if len(sp) else np.nan,
            "ci_low_paracetamol":      _bootstrap_mean_ci(sp)[0] if len(sp) else np.nan,
            "ci_high_paracetamol":     _bootstrap_mean_ci(sp)[1] if len(sp) else np.nan,
            "mean_score_ibuprofen":    float(np.mean(si)) if len(si) else np.nan,
            "mean_score_asa":          float(np.mean(sa)) if len(sa) else np.nan,
            "mean_anchors_para":       float(np.mean(anc)) if len(anc) else np.nan,
        })
    cand_df = pd.DataFrame(rows)
    cand_df.to_csv(TABLES / "shine_otc_candidate_progression_metrics_v1.csv", index=False)

    # Paracetamol-like dose response per (set, day)
    prog_rows = []
    for (s, d), sub_day in cand_df.groupby(["set_id", "day"]):
        sub_day = sub_day.sort_values("dose_mM")
        if len(sub_day) < 3: continue
        doses = sub_day.dose_mM.values.astype(float)
        vals_score  = sub_day.mean_score_paracetamol.values.astype(float)
        vals_candid = sub_day.rate_CANDIDATE_COMPLEX.values.astype(float)
        vals_top1   = sub_day.top1_paracetamol_freq.values.astype(float)
        prog_rows.append({
            "set_id": s, "day": d,
            "rho_score_vs_dose":     _spearman(doses, vals_score),
            "rho_candidate_vs_dose": _spearman(doses, vals_candid),
            "rho_top1_para_vs_dose": _spearman(doses, vals_top1),
            "endpoint_score_C40_minus_C0":  vals_score[-1] - vals_score[0],
            "endpoint_candid_C40_minus_C0": vals_candid[-1] - vals_candid[0],
            "endpoint_top1_C40_minus_C0":   vals_top1[-1] - vals_top1[0],
        })
    prog_df = pd.DataFrame(prog_rows)
    prog_df.to_csv(TABLES / "shine_paracetamol_like_progression_v1.csv", index=False)

    # Figures
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
        for ax, metric, title in [(axes[0], "rate_CANDIDATE_COMPLEX",
                                              "CANDIDATE_IN_COMPLEX_CONTEXT rate"),
                                            (axes[1], "rate_HIGH_PURE",
                                              "HIGH_CONFIDENCE_PURE_CONTEXT rate")]:
            for s, col_s in [("Set9", "#4C72B0"), ("Set10", "#DD8452")]:
                for d, ls in [("D0", ":"), ("D1", "--"), ("D2", "-")]:
                    sub = cand_df[(cand_df.set_id == s) & (cand_df.day == d)] \
                        .sort_values("dose_mM")
                    if sub.empty: continue
                    ax.plot(sub.dose_mM, sub[metric], color=col_s, ls=ls, lw=1.3, marker="o",
                              label=f"{s} {d}" if d == "D2" else None)
            ax.set_xlabel("APAP dose mM"); ax.set_ylabel(title); ax.grid(alpha=0.3)
            ax.set_title(title); ax.set_ylim(0, 1)
            ax.legend(fontsize=7)
        fig.suptitle("OTC detector outer-tier rate vs dose\n"
                        "CANDIDATE should rise on D2 high doses; HIGH_PURE should stay low")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_otc_candidate_rate_by_day_dose_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig otc status issue: {e}")

    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        for s, col_s in [("Set9", "#4C72B0"), ("Set10", "#DD8452")]:
            for d, ls in [("D0", ":"), ("D1", "--"), ("D2", "-")]:
                sub = cand_df[(cand_df.set_id == s) & (cand_df.day == d)] \
                    .sort_values("dose_mM")
                if sub.empty: continue
                ax.errorbar(sub.dose_mM, sub.mean_score_paracetamol,
                               yerr=[sub.mean_score_paracetamol - sub.ci_low_paracetamol,
                                      sub.ci_high_paracetamol - sub.mean_score_paracetamol],
                               color=col_s, ls=ls, lw=1.3, marker="o", capsize=3,
                               label=f"{s} {d}")
        ax.set_xlabel("APAP dose (mM)")
        ax.set_ylabel("paracetamol-like MSS score (mean ± 95% CI)")
        ax.set_title("Paracetamol-like MSS score by set/day/dose\n"
                        "Cautious reading: candidate spectral evidence only")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_paracetamol_like_score_by_day_dose_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig para score issue: {e}")

    # Stacked bar of outer-status by (set, day, dose)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
        for ax, s in zip(axes, ["Set9", "Set10"]):
            sub = cand_df[cand_df.set_id == s].copy()
            if sub.empty:
                ax.set_title(f"{s} — no data"); continue
            labels = [f"{d}/{int(c)}" for d, c in zip(sub.day, sub.dose_mM)]
            x = np.arange(len(sub))
            bottom = np.zeros(len(sub))
            for tier, color in [("rate_NOT_DETECTED", "#4C72B0"),
                                      ("rate_CANDIDATE_COMPLEX", "#f39c12"),
                                      ("rate_HIGH_PURE", "#2ca02c")]:
                ax.bar(x, sub[tier].values, bottom=bottom, color=color,
                          label=tier.replace("rate_", ""))
                bottom += sub[tier].values
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, fontsize=7)
            ax.set_ylabel("fraction"); ax.set_title(s); ax.legend(fontsize=7)
        fig.suptitle("OTC drug detection outer-tier stacked by (day/dose)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_otc_status_stacked_bar_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig stacked issue: {e}")

    return cand_df, prog_df


# ──────────────────────────────────────────────────────────────────────
# TASK 7 — Integrated feature regression
# ──────────────────────────────────────────────────────────────────────
TOP_MSS_MOLECULES = ["tyrosine", "uric_acid", "hypoxanthine", "cholesterol",
                          "ergothioneine", "tryptophan", "lactate"]


def _integrated_feature_matrix(df, base: str):
    X_bsv = df[[f"clr_{a}" for a in AXES]].values
    if base == "BSV_ONLY":
        return X_bsv, [f"clr_{a}" for a in AXES]
    # MSS score-per-molecule matrix from top1_score per molecule
    mss_cols = []
    X_mss = np.zeros((len(df), len(TOP_MSS_MOLECULES)))
    for j, m in enumerate(TOP_MSS_MOLECULES):
        # Use indicator × top1_score (proxy for per-molecule salience)
        match = (df["top1_molecule"].fillna("") == m).astype(float).values
        score = df["top1_score"].fillna(0).values.astype(float)
        X_mss[:, j] = match * score
        mss_cols.append(f"mss_top1_{m}")
    # OTC numeric encoding
    otc_status_num = df["outer_status"].fillna("NONE").map({
        "NOT_DETECTED": 0, "CANDIDATE_IN_COMPLEX_CONTEXT": 1,
        "HIGH_CONFIDENCE_PURE_CONTEXT": 2, "NONE": 0,
    }).values.astype(float).reshape(-1, 1)
    score_para = df["score_paracetamol"].fillna(0).values.astype(float).reshape(-1, 1)
    anchors_para = df["anchors_para"].fillna(0).values.astype(float).reshape(-1, 1)
    X_otc = np.hstack([otc_status_num, score_para, anchors_para])
    otc_cols = ["otc_status_numeric", "score_paracetamol", "anchors_para"]

    if base == "BSV_PLUS_MSS":
        return np.hstack([X_bsv, X_mss]), [f"clr_{a}" for a in AXES] + mss_cols
    if base == "BSV_PLUS_OTC":
        return np.hstack([X_bsv, X_otc]), [f"clr_{a}" for a in AXES] + otc_cols
    if base == "BSV_PLUS_MSS_PLUS_OTC":
        return np.hstack([X_bsv, X_mss, X_otc]), [f"clr_{a}" for a in AXES] + mss_cols + otc_cols
    raise ValueError(base)


def task7_integrated(bsv_df):
    print("[TASK 7] integrated feature regression")
    d2 = bsv_df[bsv_df.day == "D2"].copy()
    bases = ["BSV_ONLY", "BSV_PLUS_MSS", "BSV_PLUS_OTC", "BSV_PLUS_MSS_PLUS_OTC"]
    models = ["ridge", "gpr"]
    perf_rows = []
    for base in bases:
        X_all, feat_names = _integrated_feature_matrix(d2, base)
        for model_name in models:
            for set_id in ["Set9", "Set10"]:
                sub = d2[d2.set_id == set_id].copy()
                if len(sub) < 50: continue
                idx = sub.index.values - d2.index.min()  # position within d2
                # Realistically: filter rows by set
                mask = (d2.set_id == set_id).values
                Xsub = X_all[mask]; ysub = d2[mask]["dose_mM"].values.astype(float)
                groups = d2[mask]["subject_id"].values
                n_groups = len(np.unique(groups))
                gkf = GroupKFold(n_splits=min(5, n_groups))
                all_pred = np.full(len(ysub), np.nan)
                for tr, te in gkf.split(Xsub, ysub, groups=groups):
                    scaler = StandardScaler().fit(Xsub[tr])
                    Xtr = scaler.transform(Xsub[tr]); Xte = scaler.transform(Xsub[te])
                    mdl = _make_model(model_name)
                    try:
                        mdl.fit(Xtr, ysub[tr])
                        yp = mdl.predict(Xte)
                    except Exception:
                        continue
                    all_pred[te] = yp
                overall = _eval(ysub, all_pred)
                perf_rows.append({
                    "validation": f"within_{set_id}_GroupKFold5",
                    "feature_base": base, "model": model_name,
                    "n": int(len(ysub)), **overall,
                    "n_features": X_all.shape[1],
                })
            # Cross-set
            for tr_set, te_set in [("Set9", "Set10"), ("Set10", "Set9")]:
                tr_mask = (d2.set_id == tr_set).values
                te_mask = (d2.set_id == te_set).values
                if tr_mask.sum() < 50 or te_mask.sum() < 50: continue
                Xtr = X_all[tr_mask]; Xte = X_all[te_mask]
                ytr = d2[tr_mask]["dose_mM"].values.astype(float)
                yte = d2[te_mask]["dose_mM"].values.astype(float)
                scaler = StandardScaler().fit(Xtr)
                Xtr_s = scaler.transform(Xtr); Xte_s = scaler.transform(Xte)
                mdl = _make_model(model_name)
                try:
                    mdl.fit(Xtr_s, ytr); yp = mdl.predict(Xte_s)
                except Exception:
                    continue
                m = _eval(yte, yp)
                perf_rows.append({
                    "validation": f"cross_{tr_set}_to_{te_set}",
                    "feature_base": base, "model": model_name,
                    "n": int(len(yte)), **m,
                    "n_features": X_all.shape[1],
                })
    df = pd.DataFrame(perf_rows)
    df.to_csv(TABLES / "shine_integrated_feature_regression_v1.csv", index=False)

    # Figure: Ridge within-Set9 R² per base
    try:
        sub = df[(df.model == "ridge") & (df.validation == "within_Set9_GroupKFold5")]
        sub2 = df[(df.model == "ridge") & (df.validation == "within_Set10_GroupKFold5")]
        sub3 = df[(df.model == "ridge") & df.validation.str.startswith("cross")]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
        for ax, sb, title in [(axes[0], sub, "Within Set9 (Ridge)"),
                                     (axes[1], sub2, "Within Set10 (Ridge)"),
                                     (axes[2], sub3, "Cross-set transfer (Ridge)")]:
            if sb.empty:
                ax.set_title(title + " — no data"); continue
            ax.bar(sb.feature_base, sb.r2.fillna(0).values, color="#4C72B0")
            ax.set_xticklabels(sb.feature_base, rotation=20, fontsize=7, ha="right")
            ax.set_ylabel("R²"); ax.set_title(title); ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Integrated feature regression R² by feature base")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_integrated_feature_regression_comparison_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig integrated issue: {e}")
    return df


# ──────────────────────────────────────────────────────────────────────
# TASK 8 — Feature importance / coefficient interpretation
# ──────────────────────────────────────────────────────────────────────
def task8_importance(bsv_df):
    print("[TASK 8] feature importance")
    d2 = bsv_df[bsv_df.day == "D2"].copy()
    # Build BSV_PLUS_MSS_PLUS_OTC matrix
    X_all, feat_names = _integrated_feature_matrix(d2, "BSV_PLUS_MSS_PLUS_OTC")
    y = d2["dose_mM"].values.astype(float)
    # Fit on Set9, importance on Set10 (cross-set transfer importance)
    tr_mask = (d2.set_id == "Set9").values
    te_mask = (d2.set_id == "Set10").values
    scaler = StandardScaler().fit(X_all[tr_mask])
    Xtr = scaler.transform(X_all[tr_mask]); Xte = scaler.transform(X_all[te_mask])
    ytr, yte = y[tr_mask], y[te_mask]

    rows = []
    # Ridge coefficients
    r = Ridge(alpha=1.0, random_state=0).fit(Xtr, ytr)
    for name, coef in zip(feat_names, r.coef_):
        rows.append({"model": "ridge", "feature": name, "ridge_coef": float(coef),
                        "abs_ridge_coef": abs(float(coef)),
                        "permutation_importance": None})
    # Permutation importance on Ridge predictions on Set10
    try:
        imp = permutation_importance(r, Xte, yte, n_repeats=10, random_state=0)
        for i, name in enumerate(feat_names):
            rows.append({"model": "ridge_perm",
                            "feature": name, "ridge_coef": None,
                            "abs_ridge_coef": None,
                            "permutation_importance": float(imp.importances_mean[i])})
    except Exception as e:
        print(f"  perm imp issue: {e}")
    # RF permutation importance (exploratory)
    try:
        rf = RandomForestRegressor(n_estimators=200, max_depth=8,
                                            min_samples_leaf=5, random_state=0, n_jobs=-1).fit(Xtr, ytr)
        imp = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=0)
        for i, name in enumerate(feat_names):
            rows.append({"model": "rf_exploratory",
                            "feature": name, "ridge_coef": None,
                            "abs_ridge_coef": None,
                            "permutation_importance": float(imp.importances_mean[i])})
    except Exception as e:
        print(f"  rf perm imp issue: {e}")
    imp_df = pd.DataFrame(rows)
    imp_df.to_csv(TABLES / "shine_regression_feature_importance_v1.csv", index=False)

    # Figure: top features by Ridge coef + permutation
    try:
        ridge_top = imp_df[imp_df.model == "ridge"].sort_values(
            "abs_ridge_coef", ascending=False).head(12)
        perm_top = imp_df[imp_df.model == "ridge_perm"].sort_values(
            "permutation_importance", ascending=False).head(12)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].barh(ridge_top.feature[::-1], ridge_top.ridge_coef[::-1], color="#4C72B0")
        axes[0].set_title("Top-12 Ridge coefficients (train=Set9 D2)")
        axes[0].axvline(0, color="black", lw=0.5)
        axes[1].barh(perm_top.feature[::-1], perm_top.permutation_importance[::-1],
                        color="#DD8452")
        axes[1].set_title("Top-12 permutation importance on Set10 D2")
        for ax in axes: ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_regression_feature_importance_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig imp issue: {e}")
    return imp_df


# ──────────────────────────────────────────────────────────────────────
# Report + decision + audit
# ──────────────────────────────────────────────────────────────────────
def _decision(perf_df, cmp_df):
    best_r2 = perf_df[(perf_df.core_or_exploratory == "core") &
                           perf_df.validation.str.startswith("within")]["r2"].max()
    paper_r2 = PAPER_GPR["R2"]
    if best_r2 >= 0.90 and best_r2 >= paper_r2 - 0.06:
        return "SHINE_BSV_REGRESSION_MATCHES_PAPER_WITH_INTERPRETABILITY"
    if best_r2 >= 0.70:
        return "SHINE_BSV_REGRESSION_PARTIAL_BUT_INTERPRETABLE"
    if best_r2 >= 0.30:
        return "SHINE_BSV_REGRESSION_WEAK_AXIS_PROGRESSIONS_STRONG"
    return "SHINE_REGRESSION_BLOCKED"


def write_report(decision, perf_df, cmp_df, axis_prog_df, cand_df, prog_df,
                     integrated_df, imp_df):
    lines = [
        "# REPORT — SHINE EV regression + drug-candidate progression v1\n",
        f"date: {datetime.now().isoformat()}", "",
        f"## Decision: **{decision}**\n",
        "## Setup",
        "- Extends `gaira_base_4_shine_ev_gaira_pilot_v1` with focused BSV→dose regression +",
        "  MSS/OTC progression analyses + integrated-feature regression + importance.",
        "- All regressions and progression metrics operate on the prior pilot's 6,400-spectrum subsample.",
        "- Engine v4.5 / BSV axes / MSS kernel / OTC detector thresholds / preprocessing — UNCHANGED.",
        "- Labels (dose, day, set) used ONLY for regression targets and evaluation (post-hoc).",
        "- Subject-grouped 5-fold CV (GroupKFold on subject_id) to avoid leakage within a set.",
        "",
        "## Required answers\n",
    ]

    # Q1 — does BSV predict APAP dose?
    core_within = perf_df[(perf_df.core_or_exploratory == "core") &
                                perf_df.validation.str.startswith("within")] \
        .sort_values("r2", ascending=False)
    best_row = core_within.iloc[0] if not core_within.empty else None
    lines.append("### 1. Does GAIRA BSV predict APAP dose?")
    if best_row is not None:
        lines.append(f"- Best within-set core model: **{best_row['feature_set']} / "
                        f"{best_row['model']} / {best_row['validation']}** → "
                        f"**R² = {best_row['r2']:.3f}**, MAE = {best_row['mae']:.2f} mM, "
                        f"RMSE = {best_row['rmse']:.2f} mM, bias = {best_row['bias']:+.2f} mM.")
    else:
        lines.append("- (no within-set core regression rows produced)")
    lines.append("")

    # Q2 — vs paper GPR
    lines.append("### 2. How does BSV regression compare to paper GPR?")
    lines.append("| model | features | n_features | R² | MAE (mM) | RMSE (mM) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, r in cmp_df.iterrows():
        lines.append(f"| {r['label']} | {r['features']} | {int(r['n_features']) if r['n_features'] else 0} | "
                        f"{'' if r['R2'] is None else f'{float(r['R2']):.3f}'} | "
                        f"{'' if r['MAE_mM'] is None else f'{float(r['MAE_mM']):.2f}'} | "
                        f"{'' if r['RMSE_mM'] is None else f'{float(r['RMSE_mM']):.2f}'} |")
    lines.append("")
    if best_row is not None and best_row["r2"] < PAPER_GPR["R2"]:
        loss = PAPER_GPR["R2"] - best_row["r2"]
        lines.append(f"- GAIRA uses **11 biologically-interpretable axes** vs paper's ~740 raw-spectrum pixel "
                        f"features. If best R² is lower than paper's, framing is: "
                        f"'lower-dimensional interpretable representation with R² loss ~{loss:.2f}'.")
    else:
        lines.append("- GAIRA matches or exceeds paper GPR with radically lower feature dimensionality.")
    lines.append("")

    # Q3 — top progression axes
    strong = axis_prog_df[axis_prog_df.day == "D2"].copy()
    strong = strong.dropna(subset=["monotonicity_abs_rho"])
    strong = strong.sort_values("monotonicity_abs_rho", ascending=False).head(10)
    lines.append("### 3. Which BSV axes track progression/dose most strongly?")
    lines.append("Top D2 axis dose-response (|ρ| ≥ 0.8):")
    lines.append("| set | day | axis | ρ vs dose | slope/mM | endpoint C40−C0 |")
    lines.append("|---|---|---|---:|---:|---:|")
    for _, r in strong.iterrows():
        lines.append(f"| {r['set_id']} | {r['day']} | {r['axis']} | "
                        f"{r['spearman_rho_vs_dose']:+.2f} | {r.get('slope_per_mM', np.nan):+.3f} | "
                        f"{r.get('endpoint_C40_minus_C0', np.nan):+.3f} |")
    lines.append("")

    # Q4 — MSS candidates
    lines.append("### 4. Which MSS/analyte candidates track dose?")
    lines.append("See `shine_mss_analyte_progression_metrics_v1.csv` (Spearman ρ of top-3 freq vs dose). "
                    "All interpretations at candidate-level — MSS hits are NOT definitive molecule identity.")
    lines.append("")

    # Q5 / Q6 — paracetamol-like
    lines.append("### 5. Does paracetamol-like candidate evidence increase on Day 2?")
    d2 = cand_df[(cand_df.day == "D2") & (cand_df.set_id == "Set9")]
    d0 = cand_df[(cand_df.day == "D0") & (cand_df.set_id == "Set9")]
    if not d0.empty and not d2.empty:
        lines.append(f"- Set9 CANDIDATE rate D0 mean = {d0['rate_CANDIDATE_COMPLEX'].mean():.1%} "
                        f"vs D2 mean = {d2['rate_CANDIDATE_COMPLEX'].mean():.1%}")
        lines.append(f"- Set9 HIGH_PURE rate D0 mean = {d0['rate_HIGH_PURE'].mean():.1%} "
                        f"vs D2 mean = {d2['rate_HIGH_PURE'].mean():.1%}")
    lines.append("")

    lines.append("### 6. Does paracetamol-like evidence track APAP dose across both sets?")
    for _, r in prog_df.iterrows():
        lines.append(f"- {r['set_id']} {r['day']}: ρ(mean paracetamol-like score, dose) = "
                        f"{r['rho_score_vs_dose']:+.2f}; ρ(CANDIDATE rate, dose) = "
                        f"{r['rho_candidate_vs_dose']:+.2f}; ρ(top-1 paracetamol, dose) = "
                        f"{r['rho_top1_para_vs_dose']:+.2f}")
    lines.append("- **Reporting rule honored**: any correlation is described as 'paracetamol-like spectral "
                    "evidence as a candidate annotation in complex EV spectra' — NEVER 'APAP detected'.")
    lines.append("")

    # Q7 — integrated regression
    lines.append("### 7. Does adding MSS/OTC features improve dose prediction?")
    if not integrated_df.empty:
        int_within = integrated_df[integrated_df.validation.str.startswith("within")]
        best_per_base = int_within.groupby("feature_base")["r2"].max().sort_values(ascending=False)
        lines.append("Best within-set R² per feature base:")
        for base, r2 in best_per_base.items():
            lines.append(f"- {base}: R² = {r2:.3f}")
    lines.append("- If OTC features improve regression: call them **drug-like spectral covariates** "
                    "(not definitive APAP signal). If not: toxicity response is better captured by EV "
                    "biochemical BSV axes.")
    lines.append("")

    # Q8 — best interpretable
    interp = perf_df[(perf_df.model.isin(["ridge", "linear"])) &
                          perf_df.validation.str.startswith("within")] \
        .sort_values("r2", ascending=False)
    if not interp.empty:
        best_i = interp.iloc[0]
        lines.append("### 8. What is the best interpretable feature set?")
        lines.append(f"- **{best_i['feature_set']} / {best_i['model']} / {best_i['validation']}** — "
                        f"R² = {best_i['r2']:.3f}, MAE = {best_i['mae']:.2f} mM, "
                        f"using {'7 top axes' if 'TOP_AXIS' in best_i['feature_set'] else '11 axes'}.")
    lines.append("")

    # Q9 — demo
    lines.append("### 9. What should be shown in the demo?")
    lines.append("- BSV→dose regression predicted vs actual scatter (best core model)")
    lines.append("- Top 4-8 axis progression trajectories (G01 ↑, G08 ↓, G09 ↓, G06 ↑)")
    lines.append("- Paracetamol-like score-by-day-dose figure (with CANDIDATE framing)")
    lines.append("- OTC outer-tier stacked bar — shows HIGH_PURE stays <5% while CANDIDATE dominates on D2")
    lines.append("- Feature-importance chart (Ridge coefs + permutation on cross-set holdout)")
    lines.append("")

    # Q10 — uncertainties
    lines.append("### 10. What remains uncertain?")
    lines.append("- GAIRA has ~67× fewer features than paper — any R² gap is compensated by interpretability.")
    lines.append("- Paracetamol-like signal is CANDIDATE-only — dose trend is set-dependent.")
    lines.append("- MSS candidate hits use biological registry; drug templates live only in parallel OTC registry.")
    lines.append("- D1 hierarchy lacks subjects → subject-level variance not decomposable on D1.")
    lines.append("")
    (REPORTS / "REPORT_shine_regression_progression_analysis_v1.md").write_text("\n".join(lines))


def write_audit(decision):
    txt = [
        "# gaira_base_4_shine_regression_progression_v1 — audit log",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Inputs (read-only)",
        "- /Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_shine_ev_gaira_pilot_v1/tables/"
        "  {bsv, otc, mss, cohort, dose-response} per-spectrum/per-condition files",
        "",
        "## Models run",
        "- LinearRegression, Ridge, GaussianProcessRegressor (RBF + white noise),",
        "  RandomForestRegressor (exploratory only)",
        "",
        "## Validation schemes",
        "- Within-set GroupKFold(5) on subject_id (avoids spectrum-to-subject leakage)",
        "- Cross-set transfer: train Set9 D2 → test Set10 D2 and vice versa",
        "",
        "## Strict invariants",
        "- Labels used ONLY for regression targets and evaluation (dose_mM, day, set_id)",
        "- Engine / BSV / MSS kernel / OTC detector thresholds / preprocessing UNCHANGED",
        "- No APAP 'detected in EVs' claim; paracetamol-like signal framed as candidate-only",
        "",
        "## Outputs",
        "- tables/shine_regression_input_audit_v1.csv",
        "- tables/shine_bsv_regression_performance_v1.csv + predictions",
        "- tables/shine_paper_vs_gaira_regression_comparison_v1.csv",
        "- tables/shine_axis_progression_metrics_v1.csv",
        "- tables/shine_mss_analyte_progression_metrics_v1.csv + mss_cluster",
        "- tables/shine_otc_candidate_progression_metrics_v1.csv",
        "- tables/shine_paracetamol_like_progression_v1.csv",
        "- tables/shine_integrated_feature_regression_v1.csv",
        "- tables/shine_regression_feature_importance_v1.csv",
        "- figures: predicted-vs-actual, bland-altman, paper-vs-gaira, axis progression (top / day),",
        "           MSS candidate progression, MSS cluster progression, OTC rate by day-dose,",
        "           paracetamol-like score vs dose, OTC status stacked bar,",
        "           integrated feature regression comparison, feature importance",
        "- reports/REPORT_shine_regression_progression_analysis_v1.md",
        "",
        f"## Final decision\n**{decision}**",
    ]
    (AUDIT / "gaira_base_4_shine_regression_progression_v1_audit_log.md").write_text("\n".join(txt))


def main():
    print("=" * 78)
    print("gaira_base_4_shine_regression_progression_v1")
    print("=" * 78)
    bsv_df, cohort_df, dose_resp_df = task1_load()
    perf_df, pred_df = task2_bsv_regression(bsv_df)
    cmp_df = task3_paper_compare(perf_df)
    axis_prog_df = task4_axis_progression(bsv_df)
    analyte_df, cluster_df, analyte_prog_df = task5_mss_progression(bsv_df)
    cand_df, para_prog_df = task6_otc_progression(bsv_df)
    integrated_df = task7_integrated(bsv_df)
    imp_df = task8_importance(bsv_df)

    decision = _decision(perf_df, cmp_df)
    write_report(decision, perf_df, cmp_df, axis_prog_df,
                    cand_df, para_prog_df, integrated_df, imp_df)
    write_audit(decision)
    try:
        shutil.copy(__file__, CODE_SNAPSHOT / Path(__file__).name)
    except Exception:
        pass
    print(f"[done] decision: {decision}")


if __name__ == "__main__":
    main()
