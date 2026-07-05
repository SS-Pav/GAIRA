"""gaira_base_4_shine_ev_gaira_pilot_v1

SHINE EV SERS hepatotoxicity pilot — canonical GAIRA preprocessing + 11-axis
BSV + ΔBSV + MSS reporting + OPTIONAL OTC drug detection (enabled here).

STRICT INVARIANTS:
- Engine v4.5 / MSS kernel / motif registry / 11-axis BSV / OTC detector
  thresholds — UNCHANGED
- NO paper label-leaking normalization (no D0_C0 / D2_C0 / Si 642)
- NO paper k-means blank filtering
- NO classifier trained
- NO threshold tuning on dose labels
- Labels used ONLY post-hoc for cohort summaries + ΔBSV reference construction
- Drug detection runs as PARALLEL annotation — does NOT alter BSV / MSS / ΔBSV

Dataset:
  /Volumes/SSD_Rad/GAIRA_DATA/raw/shine_ev_sers/SERS-Hepatotoxicity_DATA_CODE_FIGURE

Subsampling:
  Cap N spectra per (set, day, dose) condition to 400 for tractable runtime
  (~6400 spectra × 15 ms / spectrum ≈ 2 minutes BSV).
"""
from __future__ import annotations

import json
import shutil
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaira.spectral import canonical_master_axis  # noqa: E402
from gaira.drug_detection import run_drug_detection_layer  # noqa: E402
from run_gaira_base_4_mss_resolution_reporting_layer_v1 import (  # noqa: E402
    baseline_correct, mss_anchor_score, load_templates,
)
from run_gaira_base_4_small_ev_dual_probe_analysis_v1 import (  # noqa: E402
    compute_bsv_per_spectrum, bsv_transforms, BSV_FAMILIES,
)


# ──────────────────────────────────────────────────────────────────────
# Paths + constants
# ──────────────────────────────────────────────────────────────────────
ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_shine_ev_gaira_pilot_v1")
TABLES  = ROOT / "tables"
FIGS    = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT   = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"
for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
    d.mkdir(parents=True, exist_ok=True)

SHINE = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/shine_ev_sers/"
             "SERS-Hepatotoxicity_DATA_CODE_FIGURE")
SHINE_SET9  = SHINE / "Figure4/data/Set9"
SHINE_SET10 = SHINE / "Figure4/data/Set10"

# Pixel→wavenumber calibration extracted verbatim from Fig4D.m (paper MATLAB)
SHINE_CAL_PIX = np.array([263, 367, 492, 512, 590, 782, 872, 887], dtype=float)
SHINE_CAL_CM  = np.array([620.9, 795.8, 1001.4, 1031.8, 1155.3, 1450.5, 1583.1, 1602.3])
SHINE_N_PIXELS = 1650

N_MAX_PER_CONDITION = 400   # subsample cap for tractable runtime
RNG_SEED = 42


# ──────────────────────────────────────────────────────────────────────
# STAGE 1 — INGESTION
# ──────────────────────────────────────────────────────────────────────
def stage1_ingest():
    print("[STAGE 1] ingest metadata + subsample per condition")
    rng = np.random.default_rng(RNG_SEED)
    rows_all = []    # all files (inventory)
    rows_selected = []   # subsampled

    for set_name, set_path in [("Set9", SHINE_SET9), ("Set10", SHINE_SET10)]:
        if not set_path.exists(): continue
        for cond_dir in sorted(set_path.iterdir()):
            if not cond_dir.is_dir(): continue
            cond = cond_dir.name   # e.g. D0_C0
            try:
                day_part, conc_part = cond.split("_")
                day = day_part
                dose = int(conc_part.lstrip("C"))
            except Exception:
                continue

            # Collect all spectrum files for this condition
            files_here = []
            subjects = [p for p in cond_dir.iterdir() if p.is_dir()]
            if subjects:
                for subj in sorted(subjects):
                    subj_id = subj.name
                    for f in sorted(subj.iterdir()):
                        if f.name.startswith("s_") and f.is_file():
                            files_here.append((subj_id, f))
            else:
                # Flat s_* under condition folder (Set9 D1)
                for f in sorted(cond_dir.iterdir()):
                    if f.name.startswith("s_") and f.is_file():
                        files_here.append(("flat", f))

            for subj_id, f in files_here:
                rows_all.append({
                    "set_id": set_name, "day": day, "dose_mM": dose,
                    "subject_id": subj_id, "file": str(f),
                    "rep_name": f.name,
                })
            # Subsample
            if len(files_here) > N_MAX_PER_CONDITION:
                indices = rng.choice(len(files_here), N_MAX_PER_CONDITION, replace=False)
                chosen = [files_here[i] for i in indices]
            else:
                chosen = files_here
            for subj_id, f in chosen:
                rows_selected.append({
                    "spectrum_id":   f"{set_name}/{cond}/{subj_id}/{f.name}",
                    "set_id":        set_name,
                    "day":           day,
                    "dose_mM":       dose,
                    "subject_id":    subj_id,
                    "chip_id":       subj_id,   # subject folder as chip proxy
                    "file":          str(f),
                    "rep_name":      f.name,
                })

    pd.DataFrame(rows_all).to_csv(TABLES / "shine_ingestion_inventory_v1.csv", index=False)
    sel_df = pd.DataFrame(rows_selected)
    print(f"  total files: {len(rows_all):,}; subsampled: {len(sel_df):,}")
    return sel_df


# ──────────────────────────────────────────────────────────────────────
# STAGE 2 — wavenumber calibration
# ──────────────────────────────────────────────────────────────────────
def stage2_calibration():
    print("[STAGE 2] wavenumber calibration (Fig4D.m polynomial)")
    p = np.polyfit(SHINE_CAL_PIX, SHINE_CAL_CM, 3)
    wn_axis = np.polyval(p, np.arange(1, SHINE_N_PIXELS + 1))
    cal_row = {
        "polynomial_order":  3,
        "n_calibration_points": len(SHINE_CAL_PIX),
        "calibration_pairs":  ";".join(f"{int(a)}→{b}" for a, b in zip(SHINE_CAL_PIX, SHINE_CAL_CM)),
        "full_wn_min":        float(wn_axis.min()),
        "full_wn_max":        float(wn_axis.max()),
        "full_n_pixels":      int(SHINE_N_PIXELS),
    }
    pd.DataFrame([cal_row]).to_csv(TABLES / "shine_wavenumber_calibration_qc_v1.csv", index=False)
    return wn_axis


# ──────────────────────────────────────────────────────────────────────
# STAGE 3 — canonical preprocessing
# ──────────────────────────────────────────────────────────────────────
def _read_shine_csv(path: Path) -> np.ndarray:
    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    return arr[:, 1]    # intensity column


def stage3_preprocess(meta_df, wn_axis_full, master_x):
    print("[STAGE 3] canonical preprocessing (AsLS + SG + L2)")
    n = len(meta_df)
    Y_pp = np.full((n, len(master_x)), np.nan)
    qc_rows = []
    for i, r in meta_df.iterrows():
        y_raw = _read_shine_csv(Path(r["file"]))
        if y_raw is None or len(y_raw) < 500:
            qc_rows.append({**r.to_dict(), "status": "READ_ERROR",
                               "n_finite": 0, "median": np.nan, "std": np.nan})
            continue
        # Pad / trim to 1650 pixel basis if needed
        if len(y_raw) > SHINE_N_PIXELS:
            y_raw = y_raw[:SHINE_N_PIXELS]
        elif len(y_raw) < SHINE_N_PIXELS:
            y_pad = np.full(SHINE_N_PIXELS, np.nan)
            y_pad[:len(y_raw)] = y_raw
            y_raw = y_pad
        # Interpolate from wn_axis_full → master_x
        y_interp = np.interp(master_x, wn_axis_full, y_raw, left=np.nan, right=np.nan)
        y_pp = baseline_correct(y_interp)
        n_fin = int(np.isfinite(y_pp).sum())
        is_flat = bool(np.nanstd(y_pp) < 1e-9)
        is_saturated = bool(np.nanmax(y_pp) >= 0.99 * np.nanmax(y_pp) if n_fin else False) \
                           and bool((y_pp >= np.nanmax(y_pp) * 0.99).sum() > 30)
        is_nan_majority = bool(n_fin < 0.5 * len(master_x))
        is_empty = bool(not np.any(np.isfinite(y_pp)))
        is_extreme_norm = bool(np.nanmax(np.abs(y_pp)) > 1.0) if n_fin else False
        low_signal = bool(n_fin > 0 and np.nanstd(y_pp) < 1e-4)
        if is_empty: status = "EMPTY"
        elif is_flat: status = "FLAT"
        elif is_nan_majority: status = "NAN_MAJORITY"
        elif is_saturated: status = "SATURATED"
        elif is_extreme_norm: status = "EXTREME_NORM"
        elif low_signal: status = "LOW_SIGNAL"
        else:
            status = "OK"
            Y_pp[i] = y_pp
        qc_rows.append({
            "spectrum_id": r["spectrum_id"], "set_id": r["set_id"],
            "day": r["day"], "dose_mM": r["dose_mM"],
            "status": status, "n_finite": n_fin,
            "median": float(np.nanmedian(y_pp)) if n_fin else np.nan,
            "std":    float(np.nanstd(y_pp)) if n_fin else np.nan,
        })
        if i % 500 == 0: print(f"  pp {i}/{n}")
    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(TABLES / "shine_preprocessing_qc_v1.csv", index=False)
    print(f"  QC: {dict(Counter(qc_df.status))}")
    return Y_pp, qc_df


# ──────────────────────────────────────────────────────────────────────
# STAGE 4 — label-free QC summary + exclusion log
# ──────────────────────────────────────────────────────────────────────
def stage4_qc_summary(qc_df):
    print("[STAGE 4] QC summary + exclusion log")
    summary = qc_df.groupby("status").size().reset_index(name="n")
    summary.to_csv(TABLES / "shine_qc_summary_v1.csv", index=False)
    excl = qc_df[qc_df.status != "OK"]
    excl.to_csv(TABLES / "shine_exclusion_log_v1.csv", index=False)
    return qc_df


# ──────────────────────────────────────────────────────────────────────
# STAGE 5 — 11-axis BSV scoring
# ──────────────────────────────────────────────────────────────────────
def stage5_bsv(Y_pp, master_x, meta_df, qc_df):
    print("[STAGE 5] 11-axis BSV scoring")
    # Use only OK spectra
    ok_mask = (qc_df["status"] == "OK").values
    n = len(meta_df)

    # Templates
    templates, _, _ = load_templates()
    by_mol = defaultdict(dict)
    for t in templates:
        by_mol[t["molecule"]][t["regime"]] = t

    # Score only OK spectra; others get NaN
    bsv_raw_full    = np.full((n, len(BSV_FAMILIES)), np.nan)
    bsv_sumnorm_full = np.full((n, len(BSV_FAMILIES)), np.nan)
    bsv_clr_full    = np.full((n, len(BSV_FAMILIES)), np.nan)

    ok_indices = np.where(ok_mask)[0]
    Y_ok = Y_pp[ok_indices]
    bsv_raw = compute_bsv_per_spectrum(Y_ok, master_x, by_mol)
    trans = bsv_transforms(bsv_raw)
    for k, i in enumerate(ok_indices):
        bsv_raw_full[i]    = trans["raw"][k]
        bsv_sumnorm_full[i] = trans["sumnorm"][k]
        bsv_clr_full[i]    = trans["clr"][k]

    # Per-spectrum table
    cols = ([f"raw_{f}" for f, _ in BSV_FAMILIES] +
              [f"sumnorm_{f}" for f, _ in BSV_FAMILIES] +
              [f"clr_{f}" for f, _ in BSV_FAMILIES])
    mat = np.hstack([bsv_raw_full, bsv_sumnorm_full, bsv_clr_full])
    bsv_df = pd.concat([meta_df.reset_index(drop=True)[
        ["spectrum_id", "set_id", "day", "dose_mM", "subject_id"]],
                           pd.DataFrame(mat, columns=cols)], axis=1)
    bsv_df["qc_status"] = qc_df["status"].values

    # Top-1 and top-3 axis hits per spectrum (using sumnorm)
    top1 = []; top3 = []
    for i in range(n):
        v = bsv_sumnorm_full[i]
        if not np.isfinite(v).any():
            top1.append(None); top3.append(None); continue
        idx = np.argsort(-v)
        top1.append(BSV_FAMILIES[int(idx[0])][0])
        top3.append("|".join(BSV_FAMILIES[int(k)][0] for k in idx[:3]))
    bsv_df["top1_axis"] = top1
    bsv_df["top3_axes"] = top3
    bsv_df.to_csv(TABLES / "shine_per_spectrum_bsv_outputs_v1.csv", index=False)

    # Family axis hits per condition (top-1 freq per (set, day, dose))
    rows = []
    for (s, d, c), sub in bsv_df.dropna(subset=["top1_axis"]).groupby(["set_id", "day", "dose_mM"]):
        n_sub = len(sub)
        ctr = Counter(sub["top1_axis"])
        for ax, count in ctr.items():
            rows.append({"set_id": s, "day": d, "dose_mM": c,
                            "top1_axis": ax, "freq": count / n_sub,
                            "count": count, "n_spectra": n_sub})
    pd.DataFrame(rows).to_csv(TABLES / "shine_family_axis_hits_per_condition_v1.csv", index=False)
    return bsv_df, bsv_raw_full, bsv_sumnorm_full, bsv_clr_full


# ──────────────────────────────────────────────────────────────────────
# STAGE 6 — ΔBSV computation
# ──────────────────────────────────────────────────────────────────────
def _cohens_d(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2: return np.nan
    pooled = np.sqrt(((len(x)-1)*np.var(x, ddof=1) + (len(y)-1)*np.var(y, ddof=1))
                       / max(len(x)+len(y)-2, 1))
    return float((np.mean(x) - np.mean(y)) / (pooled if pooled > 0 else 1.0))


def _bootstrap_mean_ci(x, n=300, seed=42):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stage6_delta_bsv(bsv_df, bsv_sumnorm, bsv_clr):
    print("[STAGE 6] ΔBSV computation")
    # Cohort means (per set, day, dose) of sumnorm + CLR
    axis_ids = [f for f, _ in BSV_FAMILIES]
    cohort_rows = []
    for (s, d, c), sub in bsv_df.dropna(subset=["top1_axis"]).groupby(["set_id", "day", "dose_mM"]):
        idx = sub.index.values
        row = {"set_id": s, "day": d, "dose_mM": c, "n": len(idx)}
        for k, fid in enumerate(axis_ids):
            vals = bsv_sumnorm[idx, k]
            row[f"mean_sumnorm_{fid}"] = float(np.nanmean(vals))
            row[f"sd_sumnorm_{fid}"]   = float(np.nanstd(vals))
            vals_clr = bsv_clr[idx, k]
            row[f"mean_clr_{fid}"] = float(np.nanmean(vals_clr))
        cohort_rows.append(row)
    cohort_df = pd.DataFrame(cohort_rows)
    cohort_df.to_csv(TABLES / "shine_cohort_bsv_means_v1.csv", index=False)

    # ΔBSV_A: same-day-C0 reference (per set, per day)
    delta_rows = []
    for (s, d), sub_day in cohort_df.groupby(["set_id", "day"]):
        ref_row = sub_day[sub_day.dose_mM == 0]
        if ref_row.empty: continue
        ref = ref_row.iloc[0]
        for _, r in sub_day.iterrows():
            for k, fid in enumerate(axis_ids):
                delta_rows.append({
                    "reference": "same_day_C0", "set_id": s, "day": d,
                    "dose_mM":   r["dose_mM"], "n": int(r["n"]),
                    "axis":      fid,
                    "mean_sumnorm":       r[f"mean_sumnorm_{fid}"],
                    "mean_clr":           r[f"mean_clr_{fid}"],
                    "delta_sumnorm":      r[f"mean_sumnorm_{fid}"] - ref[f"mean_sumnorm_{fid}"],
                    "delta_clr":          r[f"mean_clr_{fid}"]     - ref[f"mean_clr_{fid}"],
                })
    # ΔBSV_B: baseline D0 C0 reference (per set)
    for s in cohort_df["set_id"].unique():
        sub_set = cohort_df[cohort_df.set_id == s]
        ref_row = sub_set[(sub_set.day == "D0") & (sub_set.dose_mM == 0)]
        if ref_row.empty: continue
        ref = ref_row.iloc[0]
        for _, r in sub_set.iterrows():
            for k, fid in enumerate(axis_ids):
                delta_rows.append({
                    "reference": "D0_C0_baseline", "set_id": s, "day": r["day"],
                    "dose_mM":   r["dose_mM"], "n": int(r["n"]),
                    "axis":      fid,
                    "mean_sumnorm":       r[f"mean_sumnorm_{fid}"],
                    "mean_clr":           r[f"mean_clr_{fid}"],
                    "delta_sumnorm":      r[f"mean_sumnorm_{fid}"] - ref[f"mean_sumnorm_{fid}"],
                    "delta_clr":          r[f"mean_clr_{fid}"]     - ref[f"mean_clr_{fid}"],
                })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(TABLES / "shine_delta_bsv_summaries_v1.csv", index=False)

    # Effect sizes: C40 vs C0 per (set, day, axis)
    eff_rows = []
    for (s, d), sub_day in bsv_df.dropna(subset=["top1_axis"]).groupby(["set_id", "day"]):
        idx_c0  = sub_day[sub_day.dose_mM == 0].index.values
        idx_c40 = sub_day[sub_day.dose_mM == 40].index.values
        if len(idx_c0) < 2 or len(idx_c40) < 2: continue
        for k, fid in enumerate(axis_ids):
            d_val = _cohens_d(bsv_clr[idx_c40, k], bsv_clr[idx_c0, k])
            ci = _bootstrap_mean_ci(bsv_clr[idx_c40, k] - np.mean(bsv_clr[idx_c0, k]))
            eff_rows.append({
                "set_id": s, "day": d, "axis": fid,
                "cohens_d_C40_vs_C0_clr": d_val,
                "bootstrap_ci_low":  ci[0], "bootstrap_ci_high": ci[1],
                "n_c0":  int(len(idx_c0)), "n_c40": int(len(idx_c40)),
            })
    pd.DataFrame(eff_rows).to_csv(TABLES / "shine_effect_sizes_v1.csv", index=False)
    return cohort_df, delta_df


# ──────────────────────────────────────────────────────────────────────
# STAGE 7 — dose-response per axis per day
# ──────────────────────────────────────────────────────────────────────
def _spearman(x, y):
    x = pd.Series(x); y = pd.Series(y)
    valid = x.notna() & y.notna()
    if valid.sum() < 3: return np.nan
    rx = x[valid].rank(); ry = y[valid].rank()
    if rx.std() == 0 or ry.std() == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def stage7_dose_response(cohort_df):
    print("[STAGE 7] dose-response per axis per day")
    axis_ids = [f for f, _ in BSV_FAMILIES]
    rows = []
    for (s, d), sub in cohort_df.groupby(["set_id", "day"]):
        doses = np.array(sub["dose_mM"].tolist(), float)
        for fid in axis_ids:
            y = np.array(sub[f"mean_clr_{fid}"].tolist(), float)
            if len(doses) < 3: continue
            rho = _spearman(doses, y)
            if np.std(doses) > 0 and np.std(y) > 0:
                r = float(np.corrcoef(doses, y)[0, 1])
            else:
                r = np.nan
            # endpoint effect size
            c0 = sub[sub.dose_mM == 0]; c40 = sub[sub.dose_mM == 40]
            endpoint = (float(c40[f"mean_clr_{fid}"].iloc[0]) - float(c0[f"mean_clr_{fid}"].iloc[0])) \
                           if (not c0.empty and not c40.empty) else np.nan
            rows.append({
                "set_id": s, "day": d, "axis": fid,
                "spearman_rho_dose": rho, "pearson_r_dose": r,
                "endpoint_C40_minus_C0_clr": endpoint,
                "monotonicity_abs_rho": 0 if np.isnan(rho) else abs(rho),
            })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "shine_bsv_dose_response_metrics_v1.csv", index=False)
    return df


# ──────────────────────────────────────────────────────────────────────
# STAGE 8 — Set9 ↔ Set10 D2 transfer
# ──────────────────────────────────────────────────────────────────────
def stage8_set_transfer(cohort_df):
    print("[STAGE 8] Set9 ↔ Set10 D2 transfer")
    axis_ids = [f for f, _ in BSV_FAMILIES]
    rows = []
    for dose in sorted(cohort_df["dose_mM"].unique()):
        s9 = cohort_df[(cohort_df.set_id == "Set9") & (cohort_df.day == "D2")
                           & (cohort_df.dose_mM == dose)]
        s10 = cohort_df[(cohort_df.set_id == "Set10") & (cohort_df.day == "D2")
                             & (cohort_df.dose_mM == dose)]
        if s9.empty or s10.empty: continue
        for fid in axis_ids:
            v9 = float(s9[f"mean_clr_{fid}"].iloc[0])
            v10 = float(s10[f"mean_clr_{fid}"].iloc[0])
            rows.append({"dose_mM": int(dose), "axis": fid,
                            "set9_mean_clr": v9, "set10_mean_clr": v10,
                            "delta_set9_minus_set10": v9 - v10,
                            "direction_agreement_vs_zero":
                                bool(np.sign(v9) == np.sign(v10))})
        # per-axis correlation across all 11 axes per dose
        v9_all = np.array([float(s9[f"mean_clr_{f}"].iloc[0]) for f in axis_ids])
        v10_all = np.array([float(s10[f"mean_clr_{f}"].iloc[0]) for f in axis_ids])
        if np.std(v9_all) > 0 and np.std(v10_all) > 0:
            pearson = float(np.corrcoef(v9_all, v10_all)[0, 1])
        else:
            pearson = np.nan
        rows.append({"dose_mM": int(dose), "axis": "ALL_11_VECTOR_CORR",
                        "set9_mean_clr": None, "set10_mean_clr": None,
                        "delta_set9_minus_set10": None,
                        "direction_agreement_vs_zero": None,
                        "pearson_set9_vs_set10_clr_vector": pearson})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "shine_set9_set10_transfer_metrics_v1.csv", index=False)
    return df


# ──────────────────────────────────────────────────────────────────────
# STAGE 9 — MSS analyte reporting (narrow registry)
# ──────────────────────────────────────────────────────────────────────
def stage9_mss(Y_pp, master_x, meta_df, qc_df):
    print("[STAGE 9] MSS analyte reporting (narrow registry)")
    templates, _, _ = load_templates()
    by_mol = defaultdict(dict)
    for t in templates:
        by_mol[t["molecule"]][t["regime"]] = t
    target_mols = list(by_mol.keys())

    ok_mask = (qc_df["status"] == "OK").values
    rows = []
    for i, ok in enumerate(ok_mask):
        if not ok: continue
        y = Y_pp[i]
        scores = {}
        for mol, tps in by_mol.items():
            t = tps.get("SERS") or tps.get("Raman") or next(iter(tps.values()))
            sc, _, _ = mss_anchor_score(y, master_x, t["anchors"], t["supports"])
            scores[mol] = sc
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        r = meta_df.iloc[i]
        rows.append({
            "spectrum_id": r["spectrum_id"], "set_id": r["set_id"],
            "day": r["day"], "dose_mM": r["dose_mM"], "subject_id": r["subject_id"],
            "top1_molecule":  ranked[0][0],
            "top1_score":     ranked[0][1],
            "top3_molecules": "|".join(m for m, _ in ranked[:3]),
            "top5_molecules": "|".join(m for m, _ in ranked[:5]),
        })
        if i % 1000 == 0: print(f"  mss {i}/{len(ok_mask)}")
    mss_df = pd.DataFrame(rows)
    mss_df.to_csv(TABLES / "shine_mss_top_hits_per_spectrum_v1.csv", index=False)

    # By condition summary — top-3 molecule frequency per (set, day, dose)
    cond_rows = []
    for (s, d, c), sub in mss_df.groupby(["set_id", "day", "dose_mM"]):
        n = len(sub)
        for k in (1, 3, 5):
            ctr = Counter()
            for _, rr in sub.iterrows():
                col = f"top{k}_molecule" if k == 1 else (f"top{k}_molecules")
                if k == 1:
                    ctr[rr[col]] += 1
                else:
                    for mm in str(rr[col]).split("|"):
                        ctr[mm] += 1
            for mm, count in ctr.most_common(8):
                cond_rows.append({
                    "set_id": s, "day": d, "dose_mM": c,
                    "k": k, "molecule": mm,
                    "freq": count / n, "count": count, "n_spectra": n,
                })
    pd.DataFrame(cond_rows).to_csv(TABLES / "shine_mss_top_hits_by_condition_v1.csv", index=False)

    # MSS score effects (top-1 score distribution per condition)
    eff_rows = []
    for (s, d), sub_day in mss_df.groupby(["set_id", "day"]):
        c0 = sub_day[sub_day.dose_mM == 0]["top1_score"].values
        if len(c0) < 2: continue
        for dose in (10, 20, 40):
            cd = sub_day[sub_day.dose_mM == dose]["top1_score"].values
            if len(cd) < 2: continue
            eff_rows.append({"set_id": s, "day": d, "dose_mM": int(dose),
                                "cohens_d_top1_score_vs_C0": _cohens_d(cd, c0)})
    pd.DataFrame(eff_rows).to_csv(TABLES / "shine_mss_score_effects_by_condition_v1.csv", index=False)
    return mss_df


# ──────────────────────────────────────────────────────────────────────
# STAGE 10 — OTC drug detection (enabled)
# ──────────────────────────────────────────────────────────────────────
def stage10_drug_detection(Y_pp, master_x, meta_df, qc_df):
    print("[STAGE 10] OTC drug detection (enable_drug_detection=True)")
    ok_mask = (qc_df["status"] == "OK").values
    rows = []
    for i, ok in enumerate(ok_mask):
        if not ok: continue
        y = Y_pp[i]
        res = run_drug_detection_layer(y, master_x, enable_drug_detection=True)
        dd = res["drug_detection"]; ident = res["drug_identity"] or {}
        r = meta_df.iloc[i]
        rows.append({
            "spectrum_id": r["spectrum_id"], "set_id": r["set_id"],
            "day": r["day"], "dose_mM": r["dose_mM"], "subject_id": r["subject_id"],
            "enabled":       dd["enabled"],
            "outer_status":  dd["status"],
            "inner_confidence": dd.get("inner_confidence"),
            "present":       dd.get("present"),
            "top_1":         ident.get("top_1"),
            "top_3":         "|".join(ident.get("top_3", []) or []),
            "margin_top1_top2": ident.get("margin_top1_top2"),
            "score_paracetamol":  (ident.get("scores") or {}).get("paracetamol"),
            "score_ibuprofen":    (ident.get("scores") or {}).get("ibuprofen"),
            "score_asa":          (ident.get("scores") or {}).get("acetylsalicylic_acid"),
            "anchors_para":       (ident.get("anchor_hits") or {}).get("paracetamol"),
        })
        if i % 1000 == 0: print(f"  drug {i}/{len(ok_mask)}")
    dd_df = pd.DataFrame(rows)
    dd_df.to_csv(TABLES / "shine_otc_drug_detection_per_spectrum_v1.csv", index=False)

    # By condition
    cond_rows = []
    for (s, d, c), sub in dd_df.groupby(["set_id", "day", "dose_mM"]):
        n = len(sub)
        tier_counts = Counter(sub["outer_status"])
        top1_counts = Counter(sub["top_1"].fillna("NO_CALL"))
        cond_rows.append({
            "set_id": s, "day": d, "dose_mM": c, "n": n,
            "rate_HIGH_PURE":        tier_counts.get("HIGH_CONFIDENCE_PURE_CONTEXT", 0) / n,
            "rate_CANDIDATE_COMPLEX": tier_counts.get("CANDIDATE_IN_COMPLEX_CONTEXT", 0) / n,
            "rate_NOT_DETECTED":     tier_counts.get("NOT_DETECTED", 0) / n,
            "top1_paracetamol_rate":  top1_counts.get("paracetamol", 0) / n,
            "top1_ibuprofen_rate":    top1_counts.get("ibuprofen", 0) / n,
            "top1_asa_rate":          top1_counts.get("acetylsalicylic_acid", 0) / n,
            "top1_no_call_rate":      top1_counts.get("NO_CALL", 0) / n,
            "mean_score_paracetamol": float(sub["score_paracetamol"].mean()),
            "mean_score_ibuprofen":   float(sub["score_ibuprofen"].mean()),
            "mean_score_asa":         float(sub["score_asa"].mean()),
            "mean_anchors_para":      float(sub["anchors_para"].mean()),
        })
    cond_df = pd.DataFrame(cond_rows)
    cond_df.to_csv(TABLES / "shine_otc_drug_detection_by_condition_v1.csv", index=False)

    # paracetamol-like score dose response per (set, day)
    dose_rows = []
    for (s, d), sub in cond_df.groupby(["set_id", "day"]):
        doses = np.array(sub["dose_mM"].tolist(), float)
        scores = np.array(sub["mean_score_paracetamol"].tolist(), float)
        if len(doses) >= 3:
            rho = _spearman(doses, scores)
            r = float(np.corrcoef(doses, scores)[0, 1]) \
                    if (np.std(doses) > 0 and np.std(scores) > 0) else np.nan
        else: rho, r = np.nan, np.nan
        c0 = sub[sub.dose_mM == 0]; c40 = sub[sub.dose_mM == 40]
        endpoint = (float(c40["mean_score_paracetamol"].iloc[0])
                        - float(c0["mean_score_paracetamol"].iloc[0])) \
                       if (not c0.empty and not c40.empty) else np.nan
        dose_rows.append({"set_id": s, "day": d,
                             "spearman_rho_paracetamol_like_vs_dose": rho,
                             "pearson_r_paracetamol_like_vs_dose": r,
                             "endpoint_C40_minus_C0_mean_score": endpoint})
    pd.DataFrame(dose_rows).to_csv(TABLES / "shine_paracetamol_like_score_dose_response_v1.csv", index=False)
    return dd_df, cond_df


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────
DOSE_COLORS = {0: "#4C72B0", 10: "#DD8452", 20: "#2ca02c", 40: "#c0392b"}


def fig_mean_preprocessed_spectra(Y_pp, meta_df, qc_df, master_x):
    try:
        ok = (qc_df.status == "OK").values
        meta_ok = meta_df[ok].copy()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
        for ax, day in zip(axes, ["D0", "D1", "D2"]):
            for dose in [0, 10, 20, 40]:
                mask = (meta_ok.day == day) & (meta_ok.dose_mM == dose) & (meta_ok.set_id == "Set9")
                idx = meta_ok[mask].index.values
                if len(idx) == 0: continue
                mean = np.nanmean(Y_pp[idx], axis=0)
                ax.plot(master_x, mean, color=DOSE_COLORS[dose], lw=1.2,
                          label=f"{dose} mM (n={len(idx)})")
            ax.set_title(f"Set9 {day} — mean preprocessed spectra")
            ax.set_xlabel("Raman shift cm⁻¹"); ax.set_xlim(400, 1800)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("intensity (canonical pp)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_mean_preprocessed_spectra_by_day_dose_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig mean spectra issue: {e}")


def fig_dose_response_all_axes(dose_df):
    try:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
        for ax, day in zip(axes, ["D0", "D1", "D2"]):
            sub = dose_df[dose_df.day == day]
            if sub.empty:
                ax.set_title(f"{day} — no data"); continue
            axes_order = [f for f, _ in BSV_FAMILIES]
            for s in sorted(sub.set_id.unique()):
                sub_s = sub[sub.set_id == s].set_index("axis").reindex(axes_order)
                ax.plot(axes_order, sub_s["spearman_rho_dose"], "-o",
                          label=f"{s} ρ", lw=1.5)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_title(f"{day} — dose-response ρ per axis")
            ax.set_xlabel("BSV axis"); ax.set_ylim(-1, 1)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Spearman ρ (axis vs dose)")
        fig.suptitle("BSV axis dose-response by day")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_bsv_dose_response_all_axes_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig dose-response issue: {e}")


def fig_delta_monotonicity_heatmap(dose_df):
    try:
        axes_order = [f for f, _ in BSV_FAMILIES]
        for s in sorted(dose_df.set_id.unique()):
            sub_s = dose_df[dose_df.set_id == s]
            days = ["D0", "D1", "D2"] if s == "Set9" else ["D2"]
            mat = np.zeros((len(days), len(axes_order)))
            for i, d in enumerate(days):
                for j, ax_id in enumerate(axes_order):
                    r = sub_s[(sub_s.day == d) & (sub_s.axis == ax_id)]
                    mat[i, j] = r["spearman_rho_dose"].iloc[0] if not r.empty else 0
            fig, ax = plt.subplots(figsize=(11, 2.2 + 0.4 * len(days)))
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(axes_order))); ax.set_xticklabels(axes_order, rotation=30, fontsize=9)
            ax.set_yticks(range(len(days)));       ax.set_yticklabels(days, fontsize=10)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, f"{mat[i,j]:+.2f}", ha="center", va="center",
                              fontsize=7, color="white" if abs(mat[i,j]) > 0.5 else "black")
            ax.set_title(f"{s} — ΔBSV monotonicity (Spearman ρ vs dose) per axis per day")
            plt.colorbar(im, ax=ax, fraction=0.03, label="ρ")
            fig.tight_layout()
            fig.savefig(FIGS / f"fig_shine_delta_bsv_monotonicity_heatmap_{s}_v1.png", dpi=150)
            plt.close(fig)
    except Exception as e:
        print(f"  fig heatmap issue: {e}")


def fig_d2_radar_by_dose(cohort_df, set_id="Set9"):
    try:
        sub = cohort_df[(cohort_df.set_id == set_id) & (cohort_df.day == "D2")].sort_values("dose_mM")
        axis_ids = [f for f, _ in BSV_FAMILIES]
        angles = np.linspace(0, 2 * np.pi, len(axis_ids), endpoint=False)
        angles_closed = np.concatenate([angles, [angles[0]]])
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for _, r in sub.iterrows():
            vals = [r[f"mean_sumnorm_{fid}"] for fid in axis_ids]
            vals_closed = vals + [vals[0]]
            ax.plot(angles_closed, vals_closed, "-o", color=DOSE_COLORS[int(r.dose_mM)],
                      label=f"{int(r.dose_mM)} mM", lw=1.6)
            ax.fill(angles_closed, vals_closed, alpha=0.12, color=DOSE_COLORS[int(r.dose_mM)])
        ax.set_xticks(angles); ax.set_xticklabels(axis_ids, fontsize=8)
        ax.set_title(f"{set_id} D2 — BSV sumnorm radar by APAP dose")
        ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.15, 1.08))
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_d2_bsv_radar_by_dose_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig radar dose issue: {e}")


def fig_d2_delta_radar(delta_df, set_id="Set9"):
    try:
        sub = delta_df[(delta_df.reference == "same_day_C0") &
                          (delta_df.set_id == set_id) &
                          (delta_df.day == "D2")].sort_values("dose_mM")
        axis_ids = [f for f, _ in BSV_FAMILIES]
        angles = np.linspace(0, 2 * np.pi, len(axis_ids), endpoint=False)
        angles_closed = np.concatenate([angles, [angles[0]]])
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for dose in [10, 20, 40]:
            sub_d = sub[sub.dose_mM == dose].set_index("axis").reindex(axis_ids)
            vals = sub_d["delta_clr"].fillna(0).tolist()
            vals_closed = vals + [vals[0]]
            ax.plot(angles_closed, vals_closed, "-o", color=DOSE_COLORS[dose],
                      label=f"Δ {dose}−0 mM (D2)", lw=1.6)
        ax.set_xticks(angles); ax.set_xticklabels(axis_ids, fontsize=8)
        ax.set_title(f"{set_id} D2 — ΔBSV (CLR) radar vs same-day C0")
        ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.15, 1.08))
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_d2_delta_bsv_radar_by_dose_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig delta radar issue: {e}")


def fig_day0_vs_day2_radar(cohort_df, set_id="Set9"):
    try:
        axis_ids = [f for f, _ in BSV_FAMILIES]
        angles = np.linspace(0, 2 * np.pi, len(axis_ids), endpoint=False)
        angles_closed = np.concatenate([angles, [angles[0]]])
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), subplot_kw=dict(polar=True))
        for ax, dose in zip(axes, [0, 10, 20, 40]):
            for day, color, ls in [("D0", "#4C72B0", "-"),
                                           ("D2", "#c0392b", "--")]:
                r = cohort_df[(cohort_df.set_id == set_id) & (cohort_df.day == day)
                                & (cohort_df.dose_mM == dose)]
                if r.empty: continue
                row = r.iloc[0]
                vals = [row[f"mean_sumnorm_{fid}"] for fid in axis_ids]
                vals_closed = vals + [vals[0]]
                ax.plot(angles_closed, vals_closed, color=color, ls=ls, lw=1.5, label=day)
                ax.fill(angles_closed, vals_closed, alpha=0.12, color=color)
            ax.set_xticks(angles); ax.set_xticklabels(axis_ids, fontsize=7)
            ax.set_title(f"{set_id} {dose} mM — D0 vs D2", fontsize=10)
            ax.legend(fontsize=7)
        fig.suptitle("Day 0 vs Day 2 BSV radar by APAP dose", y=1.02)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_day0_vs_day2_radar_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig day0 vs day2 issue: {e}")


def fig_set9_set10_d2_radar(cohort_df):
    try:
        axis_ids = [f for f, _ in BSV_FAMILIES]
        angles = np.linspace(0, 2 * np.pi, len(axis_ids), endpoint=False)
        angles_closed = np.concatenate([angles, [angles[0]]])
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), subplot_kw=dict(polar=True))
        for ax, dose in zip(axes, [0, 10, 20, 40]):
            for s, color, ls in [("Set9", "#4C72B0", "-"), ("Set10", "#DD8452", "--")]:
                r = cohort_df[(cohort_df.set_id == s) & (cohort_df.day == "D2")
                                & (cohort_df.dose_mM == dose)]
                if r.empty: continue
                row = r.iloc[0]
                vals = [row[f"mean_sumnorm_{fid}"] for fid in axis_ids]
                vals_closed = vals + [vals[0]]
                ax.plot(angles_closed, vals_closed, color=color, ls=ls, lw=1.5, label=s)
                ax.fill(angles_closed, vals_closed, alpha=0.12, color=color)
            ax.set_xticks(angles); ax.set_xticklabels(axis_ids, fontsize=7)
            ax.set_title(f"D2 {dose} mM — Set9 vs Set10", fontsize=10)
            ax.legend(fontsize=7)
        fig.suptitle("Set9 vs Set10 — Day 2 BSV radar by dose", y=1.02)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_set9_set10_d2_radar_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig set9 vs set10 radar issue: {e}")


def fig_set9_set10_transfer(transfer_df):
    try:
        axis_ids = [f for f, _ in BSV_FAMILIES]
        doses = sorted([d for d in transfer_df["dose_mM"].unique() if d is not None])
        fig, axes = plt.subplots(1, len(doses), figsize=(4 * len(doses), 4), sharey=True)
        if len(doses) == 1: axes = [axes]
        for ax, dose in zip(axes, doses):
            sub = transfer_df[(transfer_df.dose_mM == dose) &
                                   (transfer_df.axis != "ALL_11_VECTOR_CORR")]
            sub = sub.set_index("axis").reindex(axis_ids)
            x = np.arange(len(axis_ids)); w = 0.4
            ax.bar(x - w/2, sub["set9_mean_clr"], w, label="Set9", color="#4C72B0")
            ax.bar(x + w/2, sub["set10_mean_clr"], w, label="Set10", color="#DD8452")
            ax.set_xticks(x); ax.set_xticklabels(axis_ids, rotation=30, fontsize=8)
            ax.set_title(f"{dose} mM  D2")
            ax.axhline(0, color="black", lw=0.5)
            ax.legend(fontsize=8)
        fig.suptitle("Set9 vs Set10 D2 BSV-CLR means by axis")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_set9_set10_d2_transfer_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig transfer bars issue: {e}")


def fig_mss_top_hits_by_condition(mss_df):
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for ax, day in zip(axes, ["D0", "D1", "D2"]):
            sub = mss_df[(mss_df.set_id == "Set9") & (mss_df.day == day)]
            if sub.empty: ax.set_title(f"{day} — no data"); continue
            top5 = sub["top1_molecule"].value_counts().head(5)
            ax.barh(top5.index[::-1], top5.values[::-1], color="#4C72B0")
            ax.set_title(f"Set9 {day} — top-1 MSS molecule count")
            ax.set_xlabel("n spectra")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_mss_top_hits_by_condition_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig mss issue: {e}")


def fig_drug_status_by_condition(dd_df):
    try:
        sub = dd_df[dd_df.set_id == "Set9"]
        tiers = ["HIGH_CONFIDENCE_PURE_CONTEXT", "CANDIDATE_IN_COMPLEX_CONTEXT", "NOT_DETECTED"]
        colors = {"HIGH_CONFIDENCE_PURE_CONTEXT": "#2ca02c",
                    "CANDIDATE_IN_COMPLEX_CONTEXT": "#f39c12",
                    "NOT_DETECTED": "#4C72B0"}
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for ax, day in zip(axes, ["D0", "D1", "D2"]):
            ss = sub[sub.day == day]
            doses = sorted(ss.dose_mM.unique())
            bottom = np.zeros(len(doses))
            for tier in tiers:
                counts = [int(((ss.dose_mM == d) & (ss.outer_status == tier)).sum())
                              for d in doses]
                totals = [int((ss.dose_mM == d).sum()) for d in doses]
                frac = [c/t if t > 0 else 0 for c, t in zip(counts, totals)]
                ax.bar([str(int(d)) for d in doses], frac, bottom=bottom,
                          color=colors[tier], label=tier.replace("_", " "))
                bottom += np.array(frac)
            ax.set_title(f"Set9 {day}")
            ax.set_xlabel("APAP dose mM"); ax.set_ylabel("fraction")
            ax.legend(fontsize=7)
        fig.suptitle("OTC drug detection outer-tier distribution by condition (Set9)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_otc_detection_status_by_condition_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig drug status issue: {e}")


def fig_paracetamol_like_vs_dose(dd_df):
    try:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
        for ax, day in zip(axes, ["D0", "D1", "D2"]):
            sub = dd_df[(dd_df.set_id == "Set9") & (dd_df.day == day)]
            if sub.empty: ax.set_title(f"{day} — no data"); continue
            for dose in [0, 10, 20, 40]:
                vals = sub[sub.dose_mM == dose]["score_paracetamol"].values
                ax.scatter([dose] * len(vals), vals, alpha=0.2, s=6,
                              color=DOSE_COLORS[dose])
            means = [float(sub[sub.dose_mM == d]["score_paracetamol"].mean())
                        for d in [0, 10, 20, 40]]
            ax.plot([0, 10, 20, 40], means, "-o", color="black", lw=1.5,
                      label="mean per dose")
            rho = _spearman(sub["dose_mM"], sub["score_paracetamol"])
            ax.set_title(f"Set9 {day}  ρ(dose, paracetamol-like) = {rho:+.2f}")
            ax.set_xlabel("APAP dose mM"); ax.set_ylim(0, None)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("paracetamol-like MSS score")
        fig.suptitle("Paracetamol-like MSS score vs APAP dose (Set9)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_paracetamol_like_score_vs_dose_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig paracetamol dose issue: {e}")


def fig_raw_vs_bsv_pca(Y_pp, bsv_clr, meta_df, qc_df):
    try:
        ok = (qc_df.status == "OK").values
        idx_ok = np.where(ok)[0]
        rng = np.random.default_rng(0)
        take = min(3000, len(idx_ok))
        sel = rng.choice(idx_ok, take, replace=False)
        Xraw = np.nan_to_num(Y_pp[sel], nan=0.0)
        Xbsv = bsv_clr[sel]
        Zraw = PCA(n_components=2).fit_transform(Xraw)
        Zbsv = PCA(n_components=2).fit_transform(Xbsv)
        meta_s = meta_df.iloc[sel]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        # raw × (day, dose, set)
        for col, (key, cmap) in enumerate([("day", None), ("dose_mM", None), ("set_id", None)]):
            ax = axes[0, col]
            if key == "dose_mM":
                ax.scatter(Zraw[:, 0], Zraw[:, 1], s=4, alpha=0.4,
                              c=meta_s["dose_mM"], cmap="viridis")
            else:
                for v in sorted(meta_s[key].unique()):
                    m = meta_s[key].values == v
                    ax.scatter(Zraw[m, 0], Zraw[m, 1], s=4, alpha=0.4, label=str(v))
                ax.legend(fontsize=7)
            ax.set_title(f"RAW PCA — color {key}"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        # bsv × (day, dose, set)
        for col, key in enumerate(["day", "dose_mM", "set_id"]):
            ax = axes[1, col]
            if key == "dose_mM":
                ax.scatter(Zbsv[:, 0], Zbsv[:, 1], s=4, alpha=0.4,
                              c=meta_s["dose_mM"], cmap="viridis")
            else:
                for v in sorted(meta_s[key].unique()):
                    m = meta_s[key].values == v
                    ax.scatter(Zbsv[m, 0], Zbsv[m, 1], s=4, alpha=0.4, label=str(v))
                ax.legend(fontsize=7)
            ax.set_title(f"BSV-CLR PCA — color {key}"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        fig.suptitle("SHINE raw vs BSV-CLR PCA — day / dose / set")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_shine_raw_vs_bsv_pca_day_dose_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig PCA issue: {e}")


# ──────────────────────────────────────────────────────────────────────
# STAGE 13 — final report + decision
# ──────────────────────────────────────────────────────────────────────
def _decision(dose_df, transfer_df):
    # Day 2 monotonicity: at least 1 axis with |ρ| ≥ 0.80 on either set
    d2 = dose_df[dose_df.day == "D2"]
    d0 = dose_df[dose_df.day == "D0"]
    n_d2_strong = int((d2["spearman_rho_dose"].abs() >= 0.80).sum())
    n_d0_strong = int((d0["spearman_rho_dose"].abs() >= 0.80).sum())
    # Set9 ↔ Set10 D2 transfer
    vec_corr = transfer_df[transfer_df.axis == "ALL_11_VECTOR_CORR"] \
        ["pearson_set9_vs_set10_clr_vector"].dropna().mean()
    if n_d2_strong >= 3 and (np.isnan(vec_corr) or vec_corr >= 0.5):
        return "SHINE_DYNAMIC_RESPONSE_CAPTURED"
    if n_d2_strong >= 1:
        return "SHINE_PARTIAL_RESPONSE_CAPTURED"
    if n_d2_strong == 0 and n_d0_strong == 0:
        return "SHINE_RESPONSE_NOT_CAPTURED"
    return "SHINE_PARTIAL_RESPONSE_CAPTURED"


def write_report(decision, cohort_df, dose_df, transfer_df, mss_df, dd_cond_df):
    lines = [
        "# REPORT — SHINE EV GAIRA pilot v1\n",
        f"date: {datetime.now().isoformat()}", "",
        f"## Decision: **{decision}**\n",
        "## Setup",
        "- SHINE/SPECTRA EV SERS hepatotoxicity dataset (Set9 D0+D1+D2, Set10 D2 only).",
        "- APAP / acetaminophen doses 0, 10, 20, 40 mM.",
        f"- Per-condition subsample cap: {N_MAX_PER_CONDITION} spectra (deterministic seed={RNG_SEED}).",
        "- Canonical preprocessing: pixel→wn polynomial (Fig4D.m) → master_x 400-1800 step 1 + "
        "AsLS (lam=1e5, p=0.001, 10 iter) + Savitzky-Golay (w=11, ord=3) + L2.",
        "- NO paper D0_C0/D2_C0 label-leaking normalization; NO Si 642 normalization; NO k-means blank filter.",
        "- 11-axis BSV via family-aggregated MSS anchor kernel (unchanged schema).",
        "- ΔBSV uses labels ONLY post-hoc (reference = same-day C0 OR baseline D0_C0).",
        "- Drug detection layer ENABLED (parallel, doesn't modify BSV/MSS/ΔBSV).",
        "",
        "## Required answers\n",
    ]

    # Q1 — dose-dependent response?
    d2_max = dose_df[dose_df.day == "D2"]["spearman_rho_dose"].abs().max()
    lines.append("### 1. Does GAIRA detect APAP dose-dependent EV biochemical response?")
    lines.append(f"- On D2, **max |ρ(axis, dose)| = {d2_max:.2f}** across 11 axes × 2 sets.")
    lines.append("- See `shine_bsv_dose_response_metrics_v1.csv` for per-axis per-day numbers.")
    lines.append("")

    # Q2 — Day 0 flat?
    d0_max = dose_df[dose_df.day == "D0"]["spearman_rho_dose"].abs().max()
    lines.append("### 2. Is Day 0 flat / negative-control-like?")
    lines.append(f"- D0 max |ρ| = {d0_max:.2f}. Expected: flat (< ~0.5).")
    lines.append("")

    # Q3 — which axes carry D2 response
    d2_sorted = dose_df[dose_df.day == "D2"].copy()
    d2_sorted["abs_rho"] = d2_sorted["spearman_rho_dose"].abs()
    lines.append("### 3. Which BSV axes carry the Day 2 dose response?")
    lines.append("Top-5 D2 axes by |ρ|:")
    for _, r in d2_sorted.sort_values("abs_rho", ascending=False).head(5).iterrows():
        lines.append(f"- {r['set_id']} / {r['axis']} — ρ = {r['spearman_rho_dose']:+.2f}, "
                        f"endpoint C40-C0 = {r['endpoint_C40_minus_C0_clr']:+.3f}")
    lines.append("")

    # Q4 — Set9 ↔ Set10
    vec_corr = transfer_df[transfer_df.axis == "ALL_11_VECTOR_CORR"] \
        ["pearson_set9_vs_set10_clr_vector"].dropna().values
    lines.append("### 4. Do Set9 and Set10 reproduce D2 behavior?")
    lines.append(f"- Per-dose 11-axis vector correlation Set9 vs Set10: "
                    f"{['{:+.2f}'.format(v) for v in vec_corr]}")
    lines.append("")

    # Q5 — family hits
    lines.append("### 5. What family-axis hits dominate each condition?")
    lines.append("See `shine_family_axis_hits_per_condition_v1.csv` — top-1 BSV axis frequency per cohort.")
    lines.append("")

    # Q6 — MSS candidate hits
    lines.append("### 6. What MSS candidate hits appear by condition?")
    top_m = mss_df["top1_molecule"].value_counts().head(5)
    for m, c in top_m.items():
        lines.append(f"- {m}: top-1 count = {c} across all cohorts")
    lines.append("See `shine_mss_top_hits_by_condition_v1.csv` for per-condition breakdowns.")
    lines.append("")

    # Q7 — drug detection
    lines.append("### 7. Does optional OTC drug detection show paracetamol-like signal?")
    dd_sub = dd_cond_df[dd_cond_df.set_id == "Set9"]
    if not dd_sub.empty:
        for day in ["D0", "D1", "D2"]:
            sub = dd_sub[dd_sub.day == day]
            if sub.empty: continue
            mean_candidate = float(sub["rate_CANDIDATE_COMPLEX"].mean())
            mean_not_detected = float(sub["rate_NOT_DETECTED"].mean())
            mean_high_pure = float(sub["rate_HIGH_PURE"].mean())
            lines.append(f"- Set9 {day}: HIGH_PURE = {mean_high_pure:.1%} / "
                            f"CANDIDATE = {mean_candidate:.1%} / "
                            f"NOT_DETECTED = {mean_not_detected:.1%}")
    lines.append("- **Reporting rule honored**: any detection in EV spectra is phrased as "
                    "'paracetamol-like spectral evidence as a candidate annotation in complex EV spectra' — "
                    "NOT 'APAP detected in EVs'.")
    lines.append("")

    # Q8 — drug signal dose-dependence
    lines.append("### 8. Does drug-like signal scale with APAP dose?")
    lines.append("See `shine_paracetamol_like_score_dose_response_v1.csv`; ρ(paracetamol-like mean score, dose) "
                    "per (set, day).")
    lines.append("")

    # Q9 — GAIRA vs paper
    lines.append("### 9. Does GAIRA add interpretation beyond paper PCA/GPR?")
    lines.append("- Paper outputs dose-prediction accuracy. GAIRA outputs per-axis biochemical ΔBSV, "
                    "per-condition MSS candidate evidence, cross-set transferability (Set9↔Set10), and "
                    "a parallel drug-detection annotation layer. These are complementary: paper = quantitative "
                    "prediction; GAIRA = biochemical theme-level interpretation + reproducibility audit.")
    lines.append("")

    # Q10 — uncertainties
    lines.append("### 10. What remains uncertain?")
    lines.append("- Set9 D1 lacks subject subfolders so subject-level variance is not decomposable.")
    lines.append("- Set10 is D2-only so full day × dose trajectories for Set10 are not testable.")
    lines.append("- Drug-detection FPs on biological corpora (prior phase) mean any paracetamol-like signal "
                    "on EV spectra is a CANDIDATE annotation only — dose-monotonicity of the candidate score is "
                    "the test, not raw detection rate.")
    lines.append("- MSS candidate hits on narrow registry are biological-only templates; expected to be "
                    "dominated by whichever biological molecules best match EV band content and should be "
                    "reported as candidate-level.")
    lines.append("")

    (REPORTS / "REPORT_shine_ev_gaira_pilot_v1.md").write_text("\n".join(lines))


def write_audit(decision):
    txt = [
        "# gaira_base_4_shine_ev_gaira_pilot_v1 — audit log",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Inputs (read-only)",
        "- /Volumes/SSD_Rad/GAIRA_DATA/raw/shine_ev_sers/SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4/data/Set9",
        "- /Volumes/SSD_Rad/GAIRA_DATA/raw/shine_ev_sers/SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4/data/Set10",
        "",
        "## Preprocessing parameters",
        f"- pixel→wavenumber: 3rd-order polynomial on {len(SHINE_CAL_PIX)} paper-provided reference pairs (from Fig4D.m)",
        "- interpolation target: canonical master_x (400-1800 step 1)",
        "- AsLS baseline correction: lam=1e5, p=0.001, n_iter=10",
        "- Savitzky-Golay smoothing: window=11, polyorder=3",
        "- L2 normalization per spectrum",
        "",
        "## Strict negative invariants",
        "- NO paper label-leaking D0_C0 or D2_C0 per-day normalization",
        "- NO 642 cm⁻¹ Si peak normalization",
        "- NO k-means blank filtering",
        "- NO classifier trained",
        "- NO threshold tuning on dose labels",
        "- Labels used ONLY post-hoc for cohort summaries + ΔBSV reference",
        "- Engine v4.5 / MSS kernel / motif registry / 11-axis BSV / OTC detector thresholds ALL UNCHANGED",
        "- Drug detection enable=True explicitly (parallel annotation; does NOT alter BSV/MSS/ΔBSV)",
        f"- Per-condition subsampling cap: {N_MAX_PER_CONDITION}, deterministic seed={RNG_SEED}",
        "",
        "## Outputs",
        "- Tables: shine_ingestion_inventory, shine_wavenumber_calibration_qc, shine_preprocessing_qc,",
        "          shine_qc_summary, shine_exclusion_log, shine_per_spectrum_bsv_outputs,",
        "          shine_family_axis_hits_per_condition, shine_cohort_bsv_means, shine_delta_bsv_summaries,",
        "          shine_effect_sizes, shine_bsv_dose_response_metrics, shine_set9_set10_transfer_metrics,",
        "          shine_mss_top_hits_per_spectrum, shine_mss_top_hits_by_condition,",
        "          shine_mss_score_effects_by_condition, shine_otc_drug_detection_per_spectrum,",
        "          shine_otc_drug_detection_by_condition, shine_paracetamol_like_score_dose_response",
        "- Figures: mean spectra, dose-response all axes, monotonicity heatmap (Set9+Set10),",
        "           D2 radar by dose, D2 ΔBSV radar, D0 vs D2 radar, Set9 vs Set10 D2 radar,",
        "           Set9↔Set10 transfer bars, MSS top hits by condition, OTC detection status by condition,",
        "           paracetamol-like dose-response, raw vs BSV PCA",
        "- Report: REPORT_shine_ev_gaira_pilot_v1.md",
        "",
        f"## Final decision\n**{decision}**",
    ]
    (AUDIT / "gaira_base_4_shine_ev_gaira_pilot_v1_audit_log.md").write_text("\n".join(txt))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("gaira_base_4_shine_ev_gaira_pilot_v1")
    print("=" * 78)
    master_x = canonical_master_axis()

    meta_df = stage1_ingest()
    wn_axis = stage2_calibration()
    Y_pp, qc_df = stage3_preprocess(meta_df, wn_axis, master_x)
    qc_df = stage4_qc_summary(qc_df)
    bsv_df, bsv_raw, bsv_sumnorm, bsv_clr = stage5_bsv(Y_pp, master_x, meta_df, qc_df)
    cohort_df, delta_df = stage6_delta_bsv(bsv_df, bsv_sumnorm, bsv_clr)
    dose_df = stage7_dose_response(cohort_df)
    transfer_df = stage8_set_transfer(cohort_df)
    mss_df = stage9_mss(Y_pp, master_x, meta_df, qc_df)
    dd_df, dd_cond_df = stage10_drug_detection(Y_pp, master_x, meta_df, qc_df)

    # Figures
    print("[FIGS]")
    fig_mean_preprocessed_spectra(Y_pp, meta_df, qc_df, master_x)
    fig_dose_response_all_axes(dose_df)
    fig_delta_monotonicity_heatmap(dose_df)
    fig_d2_radar_by_dose(cohort_df, set_id="Set9")
    fig_d2_delta_radar(delta_df)
    fig_day0_vs_day2_radar(cohort_df)
    fig_set9_set10_d2_radar(cohort_df)
    fig_set9_set10_transfer(transfer_df)
    fig_mss_top_hits_by_condition(mss_df)
    fig_drug_status_by_condition(dd_df)
    fig_paracetamol_like_vs_dose(dd_df)
    fig_raw_vs_bsv_pca(Y_pp, bsv_clr, meta_df, qc_df)

    decision = _decision(dose_df, transfer_df)
    write_report(decision, cohort_df, dose_df, transfer_df, mss_df, dd_cond_df)
    write_audit(decision)
    try:
        shutil.copy(__file__, CODE_SNAPSHOT / Path(__file__).name)
    except Exception:
        pass
    print(f"[done] decision: {decision}")


if __name__ == "__main__":
    main()
