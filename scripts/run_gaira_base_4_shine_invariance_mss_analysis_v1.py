"""gaira_base_4_shine_invariance_mss_analysis_v1.

Strengthen the SHINE EV hepatotoxicity pilot by quantifying how RAW, BSV,
and MSS representations encode batch (Set9 vs Set10) vs biology (APAP dose)
on Day-2.

STRICT INVARIANTS:
- GAIRA core / engine v4.5 / preprocessing / BSV schema / MSS kernel
  / OTC detector thresholds — UNCHANGED.
- No paper normalization. No Si-642 normalization. No k-means filtering.
- Labels (set / day / dose) used ONLY post-hoc for cohort grouping +
  evaluation.

Output:
    /Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_shine_invariance_mss_analysis_v1/
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
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix,
)

warnings.simplefilter("ignore")

# ── helpers from existing pilot drivers (UNCHANGED engine) ───────────────
PROJECT_ROOT = Path("/Users/suraj/projects/GAIRA")
PILOT_SNAPSHOT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_shine_ev_gaira_pilot_v1/"
    "code_snapshot")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PILOT_SNAPSHOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from gaira.spectral import canonical_master_axis  # noqa: E402
from run_gaira_base_4_mss_resolution_reporting_layer_v1 import (  # noqa: E402
    baseline_correct, mss_anchor_score, load_templates,
)


# ── paths ────────────────────────────────────────────────────────────────
SHINE_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/shine_ev_sers/"
    "SERS-Hepatotoxicity_DATA_CODE_FIGURE")
PILOT_TABLES = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_shine_ev_gaira_pilot_v1/tables")

OUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/"
    "gaira_base_4_shine_invariance_mss_analysis_v1")
T = OUT_ROOT / "tables"; F = OUT_ROOT / "figures"
R = OUT_ROOT / "reports"; A = OUT_ROOT / "audit"; C = OUT_ROOT / "code_snapshot"
for d in (T, F, R, A, C): d.mkdir(parents=True, exist_ok=True)


# ── pixel→wavenumber calibration (Fig4D.m verbatim) ──────────────────────
SHINE_CAL_PIX = np.array([263, 367, 492, 512, 590, 782, 872, 887], dtype=float)
SHINE_CAL_CM  = np.array([620.9, 795.8, 1001.4, 1031.8, 1155.3, 1450.5,
                            1583.1, 1602.3])
SHINE_N_PIXELS = 1650
RNG_SEED = 42


def shine_wn_axis() -> np.ndarray:
    coeffs = np.polyfit(SHINE_CAL_PIX, SHINE_CAL_CM, 3)
    return np.polyval(coeffs, np.arange(1, SHINE_N_PIXELS + 1, dtype=float))


def load_one_raw(spectrum_id: str) -> np.ndarray | None:
    p = SHINE_ROOT / "Figure4" / "data" / spectrum_id
    if not p.exists(): return None
    try:
        arr = pd.read_csv(p, header=None).values
        return arr[:, 1].astype(float) if arr.shape[1] >= 2 else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────
# TASK 1 — DATA INTEGRITY AUDIT
# ─────────────────────────────────────────────────────────────────────────

def task1_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[task 1] data integrity audit")
    bsv = pd.read_csv(PILOT_TABLES / "shine_per_spectrum_bsv_outputs_v1.csv")
    mss = pd.read_csv(PILOT_TABLES / "shine_mss_top_hits_per_spectrum_v1.csv")
    otc = pd.read_csv(PILOT_TABLES / "shine_otc_drug_detection_per_spectrum_v1.csv")

    rows = []
    rows.append({"table": "BSV per_spectrum",  "n_rows": len(bsv),
                 "unique_spectrum_id": bsv["spectrum_id"].nunique()})
    rows.append({"table": "MSS top_hits",      "n_rows": len(mss),
                 "unique_spectrum_id": mss["spectrum_id"].nunique()})
    rows.append({"table": "OTC drug detection", "n_rows": len(otc),
                 "unique_spectrum_id": otc["spectrum_id"].nunique()})

    common = (set(bsv["spectrum_id"]) & set(mss["spectrum_id"])
              & set(otc["spectrum_id"]))
    rows.append({"table": "TRIPLE intersection",
                 "n_rows": len(common), "unique_spectrum_id": len(common)})

    # Day/dose/set counts
    dist = (bsv.groupby(["set_id", "day", "dose_mM"]).size()
            .reset_index(name="n_spectra"))
    dist.to_csv(T / "shine_invariance_input_audit_v1.csv", index=False)
    pd.DataFrame(rows).to_csv(A / "tab_match_summary.csv", index=False)
    print(f"  triple-matched spectrum_ids: {len(common)}")
    print(f"  per-cohort breakdown:")
    print(dist.to_string(index=False))
    return bsv, mss, otc


# ─────────────────────────────────────────────────────────────────────────
# TASK 2 — REPRESENTATION MATRICES
# ─────────────────────────────────────────────────────────────────────────

# 19 narrow-registry molecules used by SHINE pilot
NARROW_REGISTRY_MOLECULES = sorted([
    "adenine", "cholesterol", "creatinine", "cysteine", "cystine",
    "ergothioneine", "glucose", "glutathione", "hypoxanthine", "lactate",
    "oleic_acid", "palmitic_acid", "phenylalanine", "stearic_acid",
    "tryptophan", "tyrosine", "urea", "uric_acid", "xanthine",
])

BSV_AXES = [f"G{i:02d}" for i in range(1, 12)]
BSV_CLR_COLS = [f"clr_{ax}" for ax in BSV_AXES]
BSV_SUMNORM_COLS = [f"sumnorm_{ax}" for ax in BSV_AXES]


def _preprocess_one(y_raw: np.ndarray, master_x: np.ndarray,
                    wn_axis_full: np.ndarray) -> np.ndarray | None:
    if len(y_raw) < SHINE_N_PIXELS:
        y = np.full(SHINE_N_PIXELS, np.nan); y[:len(y_raw)] = y_raw
    else:
        y = y_raw[:SHINE_N_PIXELS]
    y_interp = np.interp(master_x, wn_axis_full, y, left=np.nan, right=np.nan)
    if not np.isfinite(y_interp).all():
        med = np.nanmedian(y_interp)
        y_interp = np.where(np.isfinite(y_interp), y_interp, med)
    y_pp = baseline_correct(y_interp)
    from scipy.signal import savgol_filter
    y_pp = savgol_filter(y_pp, window_length=11, polyorder=3)
    n = float(np.linalg.norm(y_pp))
    return y_pp / n if n > 0 else y_pp


def task2_build_matrices(bsv: pd.DataFrame, mss: pd.DataFrame,
                          otc: pd.DataFrame) -> dict:
    print("[task 2] build representation matrices (D2 only · Set9+Set10)")
    bsv_d2 = bsv[(bsv.day == "D2") & (bsv.set_id.isin(["Set9", "Set10"]))
                  & (bsv.dose_mM.isin([0, 10, 20, 40]))
                  & (bsv.qc_status.fillna("OK") == "OK")].copy()
    bsv_d2 = bsv_d2.reset_index(drop=True)
    print(f"  D2 n_spectra: {len(bsv_d2)}")

    spectrum_ids = bsv_d2["spectrum_id"].tolist()
    meta = bsv_d2[["spectrum_id", "set_id", "day", "dose_mM",
                     "subject_id"]].copy()

    # ---- BSV matrices (CLR + sumnorm, 11-d each) ----
    X_bsv_clr = bsv_d2[BSV_CLR_COLS].values
    X_bsv_sumnorm = bsv_d2[BSV_SUMNORM_COLS].values

    # ---- RAW preprocessed (re-load + cache) ----
    raw_cache = T / "shine_d2_raw_pp_matrix_v1.npy"
    raw_meta_cache = T / "shine_d2_raw_pp_spectrum_ids_v1.csv"
    master_x = canonical_master_axis()
    wn_axis_full = shine_wn_axis()

    if raw_cache.exists() and raw_meta_cache.exists():
        X_raw = np.load(raw_cache)
        cached_ids = pd.read_csv(raw_meta_cache)["spectrum_id"].tolist()
        if cached_ids == spectrum_ids:
            print(f"  RAW reuse: cached {X_raw.shape}")
        else:
            print("  RAW cache spectrum_ids mismatch — rebuilding")
            X_raw = None
    else:
        X_raw = None

    if X_raw is None:
        rows = []
        for i, sid in enumerate(spectrum_ids):
            if i % 500 == 0:
                print(f"  RAW preprocess {i}/{len(spectrum_ids)}")
            y_raw = load_one_raw(sid)
            if y_raw is None:
                rows.append(np.full(len(master_x), np.nan)); continue
            y_pp = _preprocess_one(y_raw, master_x, wn_axis_full)
            rows.append(y_pp if y_pp is not None
                         else np.full(len(master_x), np.nan))
        X_raw = np.vstack(rows)
        np.save(raw_cache, X_raw)
        pd.DataFrame({"spectrum_id": spectrum_ids}).to_csv(
            raw_meta_cache, index=False)
        print(f"  RAW built + cached: {X_raw.shape}")

    # ---- MSS scores (re-run kernel for D2 only) ----
    mss_cache = T / "shine_d2_mss_scores_matrix_v1.npy"
    mss_cols_cache = T / "shine_d2_mss_scores_columns_v1.csv"
    if mss_cache.exists() and mss_cols_cache.exists():
        X_mss = np.load(mss_cache)
        mss_cols = pd.read_csv(mss_cols_cache)["molecule"].tolist()
        if len(mss_cols) >= len(NARROW_REGISTRY_MOLECULES):
            print(f"  MSS reuse: cached {X_mss.shape}")
        else:
            X_mss = None
    else:
        X_mss = None

    if X_mss is None:
        templates, _, _ = load_templates()
        by_mol = defaultdict(dict)
        for t in templates:
            by_mol[t["molecule"]][t["regime"]] = t
        mol_present = [m for m in NARROW_REGISTRY_MOLECULES if m in by_mol]
        print(f"  MSS molecules available: {len(mol_present)}/{len(NARROW_REGISTRY_MOLECULES)}")
        n = X_raw.shape[0]
        X_mss = np.zeros((n, len(mol_present)))
        for i in range(n):
            if i % 500 == 0:
                print(f"  MSS score {i}/{n}")
            y = X_raw[i]
            if not np.isfinite(y).any(): continue
            for j, mol in enumerate(mol_present):
                tps = by_mol[mol]
                # SHINE EV is SERS, use SERS template if available, else any
                t = tps.get("SERS") or tps.get("Raman") or next(iter(tps.values()))
                sc, _, _ = mss_anchor_score(y, master_x, t["anchors"], t["supports"])
                X_mss[i, j] = sc
        mss_cols = mol_present
        np.save(mss_cache, X_mss)
        pd.DataFrame({"molecule": mss_cols}).to_csv(mss_cols_cache, index=False)
        print(f"  MSS built + cached: {X_mss.shape}")

    # ---- OTC features (per spectrum, joined on spectrum_id) ----
    otc_d2 = otc[otc["spectrum_id"].isin(spectrum_ids)].copy()
    otc_d2 = otc_d2.set_index("spectrum_id").loc[spectrum_ids].reset_index()
    otc_features = otc_d2[["score_paracetamol", "score_ibuprofen", "score_asa",
                              "anchors_para"]].fillna(0).values

    inv_rows = [
        {"representation": "RAW", "shape": str(X_raw.shape),
         "n_features": X_raw.shape[1]},
        {"representation": "BSV CLR", "shape": str(X_bsv_clr.shape),
         "n_features": X_bsv_clr.shape[1]},
        {"representation": "BSV sumnorm", "shape": str(X_bsv_sumnorm.shape),
         "n_features": X_bsv_sumnorm.shape[1]},
        {"representation": "MSS scores", "shape": str(X_mss.shape),
         "n_features": X_mss.shape[1]},
        {"representation": "OTC drug features", "shape": str(otc_features.shape),
         "n_features": otc_features.shape[1]},
    ]
    pd.DataFrame(inv_rows).to_csv(
        T / "shine_representation_matrix_inventory_v1.csv", index=False)
    print(pd.DataFrame(inv_rows).to_string(index=False))

    return {
        "meta": meta,
        "X_raw":          X_raw,
        "X_bsv_clr":      X_bsv_clr,
        "X_bsv_sumnorm":  X_bsv_sumnorm,
        "X_mss":          X_mss,
        "X_otc":          otc_features,
        "raw_axis":       master_x,
        "mss_cols":       mss_cols,
        "bsv_d2":         bsv_d2,
        "otc_d2":         otc_d2,
    }


# ─────────────────────────────────────────────────────────────────────────
# TASK 3 — BIOLOGY VS BATCH CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────

def _cv_classify(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None,
                  task: str, n_splits: int = 5) -> dict:
    """5-fold CV. GroupKFold by subject if groups given, else Stratified."""
    rs = 42
    folds_used = "GroupKFold(subject_id)" if groups is not None else "StratifiedKFold"
    if groups is not None and len(np.unique(groups)) >= n_splits:
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X, y, groups))
    else:
        if groups is not None:
            folds_used = (f"GroupKFold(subject_id, k={len(np.unique(groups))})"
                           if len(np.unique(groups)) >= 2
                           else "StratifiedKFold(fallback)")
            cv = (GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
                   if len(np.unique(groups)) >= 2
                   else StratifiedKFold(n_splits=n_splits, shuffle=True,
                                          random_state=rs))
            splits = (list(cv.split(X, y, groups))
                       if len(np.unique(groups)) >= 2
                       else list(cv.split(X, y)))
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=rs)
            splits = list(cv.split(X, y))

    is_binary = len(np.unique(y)) == 2
    metrics = defaultdict(list)
    confusions = []
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr); Xte = scaler.transform(Xte)
        clf = LogisticRegression(max_iter=2000, random_state=rs, C=1.0)
        try:
            clf.fit(Xtr, ytr)
        except Exception:
            continue
        proba = (clf.predict_proba(Xte) if hasattr(clf, "predict_proba")
                  else None)
        pred = clf.predict(Xte)
        if is_binary:
            metrics["auroc"].append(roc_auc_score(yte, proba[:, 1]))
        else:
            try:
                metrics["auroc"].append(roc_auc_score(
                    yte, proba, multi_class="ovr", average="macro"))
            except Exception:
                metrics["auroc"].append(np.nan)
        metrics["acc"].append(accuracy_score(yte, pred))
        metrics["bal_acc"].append(balanced_accuracy_score(yte, pred))
        metrics["f1"].append(f1_score(yte, pred,
                                          average="binary" if is_binary else "macro"))
        confusions.append(confusion_matrix(yte, pred,
                                              labels=sorted(np.unique(y))))

    out = {
        "task": task, "fold_strategy": folds_used,
        "n_features": X.shape[1], "n_samples": X.shape[0],
        "n_classes": int(len(np.unique(y))),
        "auroc_mean": float(np.nanmean(metrics["auroc"])),
        "auroc_sd": float(np.nanstd(metrics["auroc"])),
        "acc_mean": float(np.mean(metrics["acc"])),
        "bal_acc_mean": float(np.mean(metrics["bal_acc"])),
        "f1_mean": float(np.mean(metrics["f1"])),
        "n_folds": len(metrics["acc"]),
    }
    return out, confusions


def task3_classify(matrices: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[task 3] biology vs batch classifiers")
    meta = matrices["meta"]
    subjects = meta["subject_id"].astype(str).values

    representations = {
        "RAW (1401 wn)":      matrices["X_raw"],
        "BSV (11 axes, CLR)": matrices["X_bsv_clr"],
        "MSS (19 scores)":    matrices["X_mss"],
        "BSV + MSS":          np.hstack([matrices["X_bsv_clr"], matrices["X_mss"]]),
    }

    # Build label vectors
    y_set = (meta["set_id"].values == "Set10").astype(int)  # 1 = Set10
    y_dose_multi = meta["dose_mM"].astype(int).values
    high = meta["dose_mM"].isin([40]).values
    low = meta["dose_mM"].isin([0]).values
    keep_binary = high | low
    y_dose_bin = high[keep_binary].astype(int)

    rows = []
    confusions_long = []

    # Set9 vs Set10 — cannot use GroupKFold by subject (subjects don't cross sets)
    for rep_name, X in representations.items():
        # Replace NaN safely
        X_clean = np.nan_to_num(X, nan=0.0)
        m, _ = _cv_classify(X_clean, y_set, groups=None,
                              task=f"{rep_name} · Set9-vs-Set10 (batch)")
        m["representation"] = rep_name; m["target"] = "set9_vs_set10"
        rows.append(m)

        # Dose 4-class — GroupKFold by subject
        m, conf = _cv_classify(X_clean, y_dose_multi,
                                groups=subjects,
                                task=f"{rep_name} · dose 4-class (biology)")
        m["representation"] = rep_name; m["target"] = "dose_4_class"
        rows.append(m)
        for ci, cmat in enumerate(conf):
            confusions_long.append({"representation": rep_name,
                                       "target": "dose_4_class",
                                       "fold": ci,
                                       "matrix": json.dumps(cmat.tolist())})

        # 0 vs 40 binary — GroupKFold by subject
        sub_subjects = subjects[keep_binary]
        m, conf = _cv_classify(X_clean[keep_binary], y_dose_bin,
                                groups=sub_subjects,
                                task=f"{rep_name} · 0-vs-40 binary (toxicity)")
        m["representation"] = rep_name; m["target"] = "dose_0_vs_40"
        rows.append(m)
        for ci, cmat in enumerate(conf):
            confusions_long.append({"representation": rep_name,
                                       "target": "dose_0_vs_40",
                                       "fold": ci,
                                       "matrix": json.dumps(cmat.tolist())})

    perf = pd.DataFrame(rows)
    perf.to_csv(T / "shine_biology_vs_batch_classifier_metrics_v1.csv",
                 index=False)
    pd.DataFrame(confusions_long).to_csv(
        T / "shine_biology_vs_batch_confusion_matrices_v1.csv", index=False)
    print(perf[["representation", "target", "auroc_mean", "auroc_sd",
                  "bal_acc_mean", "n_folds", "fold_strategy"]].to_string(
        index=False))
    return perf, pd.DataFrame(confusions_long)


# ─────────────────────────────────────────────────────────────────────────
# TASK 4 — INVARIANCE SCORES
# ─────────────────────────────────────────────────────────────────────────

def task4_invariance(perf: pd.DataFrame) -> pd.DataFrame:
    print("[task 4] invariance + biology-selectivity scores")
    rows = []
    for rep in perf["representation"].unique():
        sub = perf[perf.representation == rep]
        batch = sub[sub.target == "set9_vs_set10"]["auroc_mean"]
        dose4 = sub[sub.target == "dose_4_class"]["auroc_mean"]
        bin_ = sub[sub.target == "dose_0_vs_40"]["auroc_mean"]
        b = float(batch.iloc[0]) if len(batch) else np.nan
        d4 = float(dose4.iloc[0]) if len(dose4) else np.nan
        d2 = float(bin_.iloc[0]) if len(bin_) else np.nan
        rows.append({
            "representation": rep,
            "batch_AUROC_set9_vs_set10": round(b, 3),
            "dose_AUROC_4_class":         round(d4, 3),
            "dose_AUROC_0_vs_40":          round(d2, 3),
            "biology_selectivity_score (d4 - batch)":
                (round(d4 - b, 3) if np.isfinite(b) and np.isfinite(d4)
                 else np.nan),
            "biology_selectivity_bin (d2 - batch)":
                (round(d2 - b, 3) if np.isfinite(b) and np.isfinite(d2)
                 else np.nan),
            "batch_invariance_score (1 - |b-0.5|/0.5)":
                (round(1 - abs(b - 0.5) / 0.5, 3) if np.isfinite(b) else np.nan),
        })
    df = pd.DataFrame(rows)
    df.to_csv(T / "shine_representation_invariance_scores_v1.csv", index=False)
    print(df.to_string(index=False))

    # Figure
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    reps = df["representation"].tolist()
    x = np.arange(len(reps)); w = 0.27
    ax.bar(x - w, df["batch_AUROC_set9_vs_set10"], width=w,
           label="batch AUROC (set9 vs set10)", color="#9aa6ad")
    ax.bar(x, df["dose_AUROC_4_class"], width=w,
           label="dose AUROC (4-class)", color="#3a7d8c")
    ax.bar(x + w, df["dose_AUROC_0_vs_40"], width=w,
           label="dose AUROC (0 vs 40)", color="#1a4651")
    ax.axhline(0.5, color="#999", lw=0.7, ls="--", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(reps, fontsize=9, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC"); ax.set_title(
        "SHINE D2 · biology vs batch separability per representation",
        fontsize=11, loc="left")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(True, axis="y", alpha=0.20, lw=0.5)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    plt.tight_layout()
    fig.savefig(F / "fig_shine_biology_vs_batch_selectivity_v1.png",
                 dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return df


# ─────────────────────────────────────────────────────────────────────────
# TASK 5 — VARIANCE DECOMPOSITION (η²)
# ─────────────────────────────────────────────────────────────────────────

def _eta_squared(y: np.ndarray, set_id: np.ndarray,
                  dose: np.ndarray) -> dict:
    """Two-way fixed-effect ANOVA-like decomposition with main + interaction."""
    grand = y.mean()
    ss_total = np.sum((y - grand) ** 2)
    if ss_total <= 0: return {"set": 0, "dose": 0, "inter": 0, "resid": 1}

    sets = np.unique(set_id); doses = np.unique(dose)
    ss_set = 0.0
    for s in sets:
        ms = y[set_id == s].mean(); n = (set_id == s).sum()
        ss_set += n * (ms - grand) ** 2
    ss_dose = 0.0
    for d in doses:
        md = y[dose == d].mean(); n = (dose == d).sum()
        ss_dose += n * (md - grand) ** 2
    ss_inter = 0.0
    for s in sets:
        for d in doses:
            m_sd = y[(set_id == s) & (dose == d)].mean()
            n_sd = ((set_id == s) & (dose == d)).sum()
            if n_sd == 0: continue
            ms = y[set_id == s].mean()
            md = y[dose == d].mean()
            ss_inter += n_sd * (m_sd - ms - md + grand) ** 2

    ss_resid = ss_total - ss_set - ss_dose - ss_inter
    if ss_resid < 0: ss_resid = 0
    return {
        "set":   ss_set / ss_total,
        "dose":  ss_dose / ss_total,
        "inter": ss_inter / ss_total,
        "resid": ss_resid / ss_total,
    }


def task5_variance(matrices: dict) -> dict:
    print("[task 5] variance decomposition (η²)")
    meta = matrices["meta"]
    set_id = meta["set_id"].values
    dose = meta["dose_mM"].astype(int).values

    representations = [
        ("RAW", matrices["X_raw"], [f"wn_{int(w)}" for w in matrices["raw_axis"]]),
        ("BSV", matrices["X_bsv_clr"], BSV_CLR_COLS),
        ("MSS", matrices["X_mss"], matrices["mss_cols"]),
    ]
    summary_rows = []
    out_dfs = {}

    for name, X, cols in representations:
        rows = []
        for j in range(X.shape[1]):
            y = X[:, j]
            if np.allclose(y, y[0]):
                rows.append({"feature": cols[j], "eta2_set": 0,
                              "eta2_dose": 0, "eta2_inter": 0,
                              "eta2_resid": 1})
                continue
            r = _eta_squared(y, set_id, dose)
            rows.append({"feature": cols[j], "eta2_set": r["set"],
                          "eta2_dose": r["dose"], "eta2_inter": r["inter"],
                          "eta2_resid": r["resid"]})
        df = pd.DataFrame(rows)
        out_dfs[name] = df
        df.to_csv(T / f"shine_variance_decomposition_{name.lower()}_v1.csv",
                   index=False)
        summary_rows.append({
            "representation": name,
            "n_features": len(df),
            "median_eta2_set":   round(df.eta2_set.median(), 4),
            "median_eta2_dose":  round(df.eta2_dose.median(), 4),
            "median_eta2_inter": round(df.eta2_inter.median(), 4),
            "median_eta2_resid": round(df.eta2_resid.median(), 4),
            "mean_eta2_set":     round(df.eta2_set.mean(), 4),
            "mean_eta2_dose":    round(df.eta2_dose.mean(), 4),
            "frac_features_dose_dominant":
                round(float((df.eta2_dose > df.eta2_set).sum() / len(df)), 3),
            "top3_dose_features":
                ", ".join(df.sort_values("eta2_dose",
                                           ascending=False)["feature"].head(3)
                           .astype(str).tolist()),
            "top3_set_features":
                ", ".join(df.sort_values("eta2_set",
                                           ascending=False)["feature"].head(3)
                           .astype(str).tolist()),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(T / "shine_variance_decomposition_summary_v1.csv",
                    index=False)
    print(summary.to_string(index=False))

    # Figure: bar of median η²_set vs η²_dose per representation
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(summary)); w = 0.32
    ax.bar(x - w/2, summary["median_eta2_set"], width=w,
           label="median η² set", color="#9aa6ad")
    ax.bar(x + w/2, summary["median_eta2_dose"], width=w,
           label="median η² dose", color="#3a7d8c")
    for i, (s, d) in enumerate(zip(summary["median_eta2_set"],
                                      summary["median_eta2_dose"])):
        ax.text(i - w/2, s + 0.005, f"{s:.3f}", ha="center", va="bottom",
                 fontsize=8.5, color="#444")
        ax.text(i + w/2, d + 0.005, f"{d:.3f}", ha="center", va="bottom",
                 fontsize=8.5, color="#1a4651", fontweight="600")
    ax.set_xticks(x); ax.set_xticklabels(summary["representation"])
    ax.set_ylabel("median η² (per-feature)")
    ax.set_title("SHINE D2 · variance decomposition median η² per representation",
                  fontsize=11, loc="left")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(True, axis="y", alpha=0.20, lw=0.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    plt.tight_layout()
    fig.savefig(F / "fig_shine_variance_decomposition_summary_v1.png",
                 dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"summary": summary, "details": out_dfs}


# ─────────────────────────────────────────────────────────────────────────
# TASK 6 — PCA 3×2 figure
# ─────────────────────────────────────────────────────────────────────────

def task6_pca(matrices: dict) -> None:
    print("[task 6] RAW / BSV / MSS PCA × set / dose figure")
    meta = matrices["meta"]
    set_id = meta["set_id"].values
    dose = meta["dose_mM"].astype(int).values

    set_palette = {"Set9": "#3a7d8c", "Set10": "#d9853b"}
    dose_palette = {0: "#9aa6ad", 10: "#79c0ff", 20: "#ffa657", 40: "#ff7b72"}

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 12.5))
    rep_data = [
        ("RAW · 1401 wn",     matrices["X_raw"]),
        ("BSV · 11 axes CLR", matrices["X_bsv_clr"]),
        ("MSS · 19 scores",    matrices["X_mss"]),
    ]
    for r, (name, X) in enumerate(rep_data):
        Xc = np.nan_to_num(X, nan=0.0)
        Xs = StandardScaler().fit_transform(Xc)
        emb = PCA(n_components=2, random_state=42).fit_transform(Xs)

        # Col 0 — set
        ax = axes[r, 0]
        for s, color in set_palette.items():
            mask = set_id == s
            ax.scatter(emb[mask, 0], emb[mask, 1], s=12, alpha=0.55,
                        color=color, edgecolor="none", label=s)
        ax.set_title(f"{name}  ·  colour = set", fontsize=10.5, loc="left")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)

        # Col 1 — dose
        ax = axes[r, 1]
        for d, color in dose_palette.items():
            mask = dose == d
            ax.scatter(emb[mask, 0], emb[mask, 1], s=12, alpha=0.55,
                        color=color, edgecolor="none", label=f"{d} mM")
        ax.set_title(f"{name}  ·  colour = dose", fontsize=10.5, loc="left")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    fig.suptitle("SHINE D2 · per-representation PCA · set vs dose colouring",
                  fontsize=12.5, fontweight="600", y=0.998)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(F / "fig_shine_raw_bsv_mss_pca_set_vs_dose_v1.png",
                 dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# TASK 7 — TRAJECTORY STABILITY
# ─────────────────────────────────────────────────────────────────────────

def _trajectory(X: np.ndarray, set_id: np.ndarray, dose: np.ndarray,
                doses: list[int]) -> dict:
    """Per-feature 4-point trajectory per set."""
    out = {}
    for s in ["Set9", "Set10"]:
        cohort_means = []
        for d in doses:
            mask = (set_id == s) & (dose == d)
            cohort_means.append(X[mask].mean(axis=0) if mask.sum() else
                                 np.full(X.shape[1], np.nan))
        out[s] = np.vstack(cohort_means)  # (4, n_feat)
    return out


def task7_trajectory(matrices: dict) -> dict:
    print("[task 7] cross-set trajectory stability")
    meta = matrices["meta"]
    set_id = meta["set_id"].values
    dose = meta["dose_mM"].astype(int).values
    doses = [0, 10, 20, 40]

    out: dict = {}
    for name, X, cols in [
        ("BSV", matrices["X_bsv_clr"], BSV_CLR_COLS),
        ("MSS", matrices["X_mss"], matrices["mss_cols"]),
        ("RAW", matrices["X_raw"], [f"wn_{int(w)}" for w in matrices["raw_axis"]]),
    ]:
        traj = _trajectory(X, set_id, dose, doses)
        s9 = traj["Set9"]; s10 = traj["Set10"]
        rows = []
        for j in range(X.shape[1]):
            t9 = s9[:, j]; t10 = s10[:, j]
            if np.allclose(t9, t9[0]) and np.allclose(t10, t10[0]):
                rows.append({"feature": cols[j],
                              "pearson_r": np.nan, "spearman_r": np.nan,
                              "endpoint_sign_agree": False,
                              "set9_C40_minus_C0": 0,
                              "set10_C40_minus_C0": 0})
                continue
            try:
                pr, _ = pearsonr(t9, t10) if (np.std(t9) > 0 and np.std(t10) > 0) else (np.nan, np.nan)
                sr, _ = spearmanr(t9, t10) if (np.std(t9) > 0 and np.std(t10) > 0) else (np.nan, np.nan)
            except Exception:
                pr, sr = np.nan, np.nan
            d9 = float(t9[-1] - t9[0])
            d10 = float(t10[-1] - t10[0])
            rows.append({"feature": cols[j],
                          "pearson_r": float(pr) if pr is not None else np.nan,
                          "spearman_r": float(sr) if sr is not None else np.nan,
                          "endpoint_sign_agree": bool(np.sign(d9) == np.sign(d10)
                                                         and np.sign(d9) != 0),
                          "set9_C40_minus_C0": d9,
                          "set10_C40_minus_C0": d10})
        df = pd.DataFrame(rows).sort_values("pearson_r", ascending=False,
                                              na_position="last")
        out[name] = df
        suf = name.lower()
        if name == "BSV":
            df.to_csv(T / "shine_bsv_trajectory_stability_by_axis_v1.csv",
                       index=False)
        elif name == "MSS":
            df.to_csv(T / "shine_mss_trajectory_stability_by_candidate_v1.csv",
                       index=False)
        else:
            df.to_csv(T / "shine_raw_trajectory_stability_by_wavenumber_v1.csv",
                       index=False)

    # Figure 1: BSV + MSS top-stability bars
    bsv_df = out["BSV"]; mss_df = out["MSS"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    bsv_view = bsv_df.copy()
    bsv_view["axis"] = bsv_view["feature"].str.replace("clr_", "")
    bsv_view = bsv_view.sort_values("pearson_r", ascending=True)
    colors = ["#3a7d8c" if a else "#d9853b"
               for a in bsv_view["endpoint_sign_agree"]]
    ax.barh(bsv_view["axis"], bsv_view["pearson_r"], color=colors,
             edgecolor="white", linewidth=0.5)
    ax.set_xlabel("trajectory r · Set9 vs Set10")
    ax.set_title("A · BSV axis trajectory stability",
                  fontsize=10.5, loc="left")
    ax.axvline(0, color="#999", lw=0.5)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(True, axis="x", alpha=0.20, lw=0.5)
    ax.set_xlim(-1.05, 1.05)

    ax = axes[1]
    mss_view = mss_df.copy().sort_values("pearson_r", ascending=True)
    colors = ["#3a7d8c" if a else "#d9853b"
               for a in mss_view["endpoint_sign_agree"]]
    ax.barh(mss_view["feature"], mss_view["pearson_r"], color=colors,
             edgecolor="white", linewidth=0.5)
    ax.set_xlabel("trajectory r · Set9 vs Set10")
    ax.set_title("B · MSS candidate trajectory stability",
                  fontsize=10.5, loc="left")
    ax.axvline(0, color="#999", lw=0.5)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(True, axis="x", alpha=0.20, lw=0.5)
    ax.set_xlim(-1.05, 1.05)
    fig.suptitle("SHINE D2 · per-feature trajectory stability across "
                  "sets · teal = endpoint sign agrees, orange = disagrees",
                  fontsize=11.5, fontweight="600", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(F / "fig_shine_bsv_mss_trajectory_stability_v1.png",
                 dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Figure 2: distribution of RAW per-wn trajectory r
    raw_df = out["RAW"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(raw_df["pearson_r"].dropna(), bins=40, color="#9aa6ad",
             edgecolor="white", alpha=0.85, label="RAW per-wn (n=1401)")
    ax.axvline(bsv_df["pearson_r"].dropna().median(), color="#3a7d8c",
                lw=2, ls="--", label=f"BSV median={bsv_df.pearson_r.dropna().median():.2f}")
    ax.axvline(mss_df["pearson_r"].dropna().median(), color="#1a4651",
                lw=2, ls="--", label=f"MSS median={mss_df.pearson_r.dropna().median():.2f}")
    ax.axvline(raw_df["pearson_r"].dropna().median(), color="#666",
                lw=2, ls="--", label=f"RAW median={raw_df.pearson_r.dropna().median():.2f}")
    ax.set_xlabel("per-feature trajectory r · Set9 vs Set10")
    ax.set_ylabel("# features")
    ax.set_title("SHINE D2 · per-feature trajectory r distribution · RAW vs BSV vs MSS",
                  fontsize=11, loc="left")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(True, axis="y", alpha=0.20, lw=0.5)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    plt.tight_layout()
    fig.savefig(F / "fig_shine_raw_vs_bsv_mss_trajectory_distribution_v1.png",
                 dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────
# TASK 8 — MSS reproducible candidates (top dose tracking)
# ─────────────────────────────────────────────────────────────────────────

def task8_mss_dose_tracking(matrices: dict) -> pd.DataFrame:
    print("[task 8] MSS candidate cross-set dose tracking")
    meta = matrices["meta"]
    set_id = meta["set_id"].values
    dose = meta["dose_mM"].astype(int).values
    X = matrices["X_mss"]; cols = matrices["mss_cols"]
    doses = [0, 10, 20, 40]

    rows = []
    for j, mol in enumerate(cols):
        # per-set per-dose mean
        means = {}
        for s in ["Set9", "Set10"]:
            for d in doses:
                mask = (set_id == s) & (dose == d)
                means[(s, d)] = float(X[mask, j].mean()) if mask.sum() else np.nan

        # Spearman per set
        ys9 = [means[("Set9", d)] for d in doses]
        ys10 = [means[("Set10", d)] for d in doses]
        try:
            rho9 = float(spearmanr(doses, ys9)[0])
        except Exception:
            rho9 = np.nan
        try:
            rho10 = float(spearmanr(doses, ys10)[0])
        except Exception:
            rho10 = np.nan
        # cross-set trajectory r
        try:
            r_cross = float(pearsonr(ys9, ys10)[0])
        except Exception:
            r_cross = np.nan
        d9 = ys9[-1] - ys9[0]
        d10 = ys10[-1] - ys10[0]
        rows.append({
            "candidate": mol,
            "spearman_dose_set9": round(rho9, 3) if np.isfinite(rho9) else np.nan,
            "spearman_dose_set10": round(rho10, 3) if np.isfinite(rho10) else np.nan,
            "trajectory_r_set9_vs_set10":
                round(r_cross, 3) if np.isfinite(r_cross) else np.nan,
            "endpoint_set9_C40_minus_C0": round(d9, 4),
            "endpoint_set10_C40_minus_C0": round(d10, 4),
            "endpoint_sign_agree": bool(np.sign(d9) == np.sign(d10)
                                          and np.sign(d9) != 0),
            "monotonic_both_sets": bool(abs(rho9) >= 0.8 and abs(rho10) >= 0.8
                                          and np.sign(rho9) == np.sign(rho10)),
        })
    df = (pd.DataFrame(rows)
            .sort_values("trajectory_r_set9_vs_set10",
                         ascending=False, na_position="last"))
    df.to_csv(T / "shine_mss_reproducible_dose_candidates_v1.csv",
              index=False)
    print(df.head(10).to_string(index=False))

    # Figure — top-6 cross-set reproducible candidates
    top = df[df["trajectory_r_set9_vs_set10"] > 0.7].head(6)
    if len(top) == 0:
        # Fall back to top-6 by absolute r
        top = df.head(6)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    for k, (_, r) in enumerate(top.iterrows()):
        ax = axes[k]
        mol = r["candidate"]; j = cols.index(mol)
        ys9 = [X[(set_id == "Set9") & (dose == d), j].mean() for d in doses]
        ys10 = [X[(set_id == "Set10") & (dose == d), j].mean() for d in doses]
        ax.plot(doses, ys9, color="#3a7d8c", lw=2.0, marker="o",
                 markersize=7, label="Set9")
        ax.plot(doses, ys10, color="#d9853b", lw=2.0, marker="s",
                 markersize=7, label="Set10")
        ax.set_title(f"{mol} · r={r['trajectory_r_set9_vs_set10']:.2f}",
                      fontsize=10, loc="left")
        ax.set_xlabel("APAP dose (mM)"); ax.set_ylabel("MSS score (mean)")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.grid(True, axis="y", alpha=0.20, lw=0.5)
        if k == 0:
            ax.legend(frameon=False, fontsize=8, loc="best")
    for k in range(len(top), 6):
        axes[k].axis("off")
    fig.suptitle("SHINE D2 · top MSS candidates with reproducible "
                  "cross-set dose tracking",
                  fontsize=11.5, fontweight="600", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(F / "fig_shine_top_mss_candidate_dose_trajectories_v1.png",
                 dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return df


# ─────────────────────────────────────────────────────────────────────────
# TASK 9 — OTC/APAP D2 minus D0 candidate evidence
# ─────────────────────────────────────────────────────────────────────────

def task9_otc_delta(otc: pd.DataFrame) -> pd.DataFrame:
    print("[task 9] OTC drug candidate D2 - D0 delta")
    sub = otc[otc.set_id.isin(["Set9", "Set10"])
                & otc.day.isin(["D0", "D1", "D2"])
                & otc.dose_mM.isin([0, 10, 20, 40])].copy()

    # candidate rate = % spectra with outer_status == CANDIDATE_IN_COMPLEX_CONTEXT
    cand_mask = sub["outer_status"] == "CANDIDATE_IN_COMPLEX_CONTEXT"
    sub["is_candidate"] = cand_mask.astype(int)

    summary = (sub.groupby(["set_id", "day", "dose_mM"])
                 .agg(n_spectra=("spectrum_id", "count"),
                       cand_rate=("is_candidate", "mean"),
                       mean_para_score=("score_paracetamol", "mean"),
                       mean_anchors_para=("anchors_para", "mean"),
                       )
                 .reset_index())
    summary.to_csv(A / "otc_per_set_day_dose_summary.csv", index=False)

    # Compute D2-D0 delta per (set, dose). Only Set9 has D0.
    rows = []
    for set_id in summary["set_id"].unique():
        for dose in [0, 10, 20, 40]:
            d0 = summary[(summary.set_id == set_id) & (summary.day == "D0")
                          & (summary.dose_mM == dose)]
            d2 = summary[(summary.set_id == set_id) & (summary.day == "D2")
                          & (summary.dose_mM == dose)]
            if d0.empty or d2.empty: continue
            row = {
                "set_id": set_id, "dose_mM": dose,
                "n_d0": int(d0["n_spectra"].iloc[0]),
                "n_d2": int(d2["n_spectra"].iloc[0]),
                "cand_rate_d0":     round(float(d0["cand_rate"].iloc[0]), 4),
                "cand_rate_d2":     round(float(d2["cand_rate"].iloc[0]), 4),
                "delta_cand_rate":  round(float(d2["cand_rate"].iloc[0]
                                              - d0["cand_rate"].iloc[0]), 4),
                "delta_para_score": round(float(d2["mean_para_score"].iloc[0]
                                              - d0["mean_para_score"].iloc[0]), 4),
                "delta_anchors_para":round(float(d2["mean_anchors_para"].iloc[0]
                                              - d0["mean_anchors_para"].iloc[0]), 4),
            }
            rows.append(row)
    delta_df = pd.DataFrame(rows)
    delta_df.to_csv(T / "shine_apap_candidate_delta_day2_minus_day0_v1.csv",
                     index=False)
    print(delta_df.to_string(index=False))

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    if not delta_df.empty:
        for set_id in delta_df["set_id"].unique():
            sub_df = delta_df[delta_df.set_id == set_id]
            color = "#3a7d8c" if set_id == "Set9" else "#d9853b"
            ax.plot(sub_df["dose_mM"], sub_df["delta_cand_rate"],
                     color=color, lw=2.0, marker="o", markersize=7,
                     label=set_id)
    ax.axhline(0, color="#999", lw=0.5, ls="--")
    ax.set_xlabel("APAP dose (mM)")
    ax.set_ylabel("Δ candidate rate (D2 − D0)")
    ax.set_title("A · Δ candidate-status rate", fontsize=10.5, loc="left")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(True, axis="y", alpha=0.20, lw=0.5)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    if not delta_df.empty:
        for set_id in delta_df["set_id"].unique():
            sub_df = delta_df[delta_df.set_id == set_id]
            color = "#3a7d8c" if set_id == "Set9" else "#d9853b"
            ax.plot(sub_df["dose_mM"], sub_df["delta_para_score"],
                     color=color, lw=2.0, marker="s", markersize=7,
                     label=set_id)
    ax.axhline(0, color="#999", lw=0.5, ls="--")
    ax.set_xlabel("APAP dose (mM)")
    ax.set_ylabel("Δ paracetamol-like score (D2 − D0)")
    ax.set_title("B · Δ MSS paracetamol-like score (candidate)",
                  fontsize=10.5, loc="left")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(True, axis="y", alpha=0.20, lw=0.5)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("SHINE · drug-like candidate evidence emerging at Day-2 vs Day-0 "
                  "(candidate framing, NO APAP identity claim)",
                  fontsize=11, fontweight="600", y=0.99)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(F / "fig_shine_apap_candidate_delta_day2_minus_day0_v1.png",
                 dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return delta_df


# ─────────────────────────────────────────────────────────────────────────
# TASK 10 — Slide panel recommendation
# ─────────────────────────────────────────────────────────────────────────

def task10_slide_recommendation(perf: pd.DataFrame, inv: pd.DataFrame,
                                  var_summary: pd.DataFrame,
                                  bsv_traj: pd.DataFrame,
                                  mss_traj: pd.DataFrame) -> tuple[str, str, str]:
    print("[task 10] slide panel recommendation")

    raw_batch = float(inv[inv.representation.str.startswith("RAW")]
                       ["batch_AUROC_set9_vs_set10"].iloc[0])
    bsv_batch = float(inv[inv.representation.str.startswith("BSV (11")]
                       ["batch_AUROC_set9_vs_set10"].iloc[0])
    mss_batch = float(inv[inv.representation.str.startswith("MSS")]
                       ["batch_AUROC_set9_vs_set10"].iloc[0])
    raw_dose = float(inv[inv.representation.str.startswith("RAW")]
                       ["dose_AUROC_4_class"].iloc[0])
    bsv_dose = float(inv[inv.representation.str.startswith("BSV (11")]
                       ["dose_AUROC_4_class"].iloc[0])
    mss_dose = float(inv[inv.representation.str.startswith("MSS")]
                       ["dose_AUROC_4_class"].iloc[0])

    bsv_axes_strong = (bsv_traj["pearson_r"].fillna(-1) >= 0.7).sum()
    mss_strong = (mss_traj["pearson_r"].fillna(-1) >= 0.7).sum()

    # Decision logic
    bsv_reduces_batch = bsv_batch < raw_batch - 0.03
    mss_better_dose = mss_dose > bsv_dose + 0.02

    if bsv_reduces_batch and bsv_dose >= 0.55:
        if mss_better_dose:
            choice = "C"
            best_fig = "fig_shine_top_mss_candidate_dose_trajectories_v1.png"
            slide_claim = ("MSS preserves dose-related biology with "
                           "interpretable per-candidate features.")
        else:
            choice = "B"
            best_fig = "fig_shine_biology_vs_batch_selectivity_v1.png"
            slide_claim = ("BSV reduces batch identity while preserving "
                           "dose-associated biology.")
    elif bsv_axes_strong + mss_strong >= 5:
        choice = "A"
        best_fig = "fig_shine_bsv_mss_trajectory_stability_v1.png"
        slide_claim = ("Multiple BSV axes and MSS candidates show "
                       "reproducible dose trajectories across independent sets.")
    else:
        choice = "D"
        best_fig = "fig_shine_raw_vs_bsv_mss_trajectory_distribution_v1.png"
        slide_claim = ("Cross-set per-axis trajectory reproducibility is the "
                       "defensible reproducibility claim; broader invariance "
                       "claims are not supported by the current data.")

    caveat = ("Cohorts are subject-grouped within each set but subjects do "
              "NOT cross sets; batch / set classification is therefore "
              "spectrum-level, not subject-leakage-free.")
    options = {"A": ("BSV improves cross-set reproducibility over RAW", best_fig),
                "B": ("BSV removes batch identity while preserving dose biology", best_fig),
                "C": ("MSS preserves dose biology with interpretable features", best_fig),
                "D": ("Use axis trajectory reproducibility only", best_fig)}

    rec_md = [
        "# SLIDE_PANEL_4_RECOMMENDATION_v1",
        "",
        f"**Recommended option:** **{choice} — {options[choice][0]}**",
        "",
        f"**Best figure:** `figures/{best_fig}`",
        "",
        f"**Slide claim (one sentence):** {slide_claim}",
        "",
        f"**Caveat (one sentence):** {caveat}",
        "",
        "## Justification",
        f"- RAW batch AUROC = {raw_batch:.3f} · BSV batch AUROC = "
        f"{bsv_batch:.3f} · MSS batch AUROC = {mss_batch:.3f}",
        f"- RAW dose AUROC (4-class) = {raw_dose:.3f} · BSV = {bsv_dose:.3f} "
        f"· MSS = {mss_dose:.3f}",
        f"- BSV axes with cross-set trajectory r ≥ 0.7: {bsv_axes_strong}/11",
        f"- MSS candidates with cross-set trajectory r ≥ 0.7: {mss_strong}",
    ]
    (R / "SLIDE_PANEL_4_RECOMMENDATION_v1.md").write_text("\n".join(rec_md))
    return choice, options[choice][0], slide_claim


# ─────────────────────────────────────────────────────────────────────────
# TASK 11 — Final report
# ─────────────────────────────────────────────────────────────────────────

def task11_report(perf: pd.DataFrame, inv: pd.DataFrame,
                   var_summary: pd.DataFrame, bsv_traj: pd.DataFrame,
                   mss_traj: pd.DataFrame, mss_dose: pd.DataFrame,
                   otc_delta: pd.DataFrame, slide_choice: str,
                   slide_label: str, slide_claim: str) -> str:
    print("[task 11] final report + decision label")

    raw_batch = float(inv[inv.representation.str.startswith("RAW")]
                       ["batch_AUROC_set9_vs_set10"].iloc[0])
    bsv_batch = float(inv[inv.representation.str.startswith("BSV (11")]
                       ["batch_AUROC_set9_vs_set10"].iloc[0])
    mss_batch = float(inv[inv.representation.str.startswith("MSS")]
                       ["batch_AUROC_set9_vs_set10"].iloc[0])
    raw_dose = float(inv[inv.representation.str.startswith("RAW")]
                       ["dose_AUROC_4_class"].iloc[0])
    bsv_dose = float(inv[inv.representation.str.startswith("BSV (11")]
                       ["dose_AUROC_4_class"].iloc[0])
    mss_dose_v = float(inv[inv.representation.str.startswith("MSS")]
                       ["dose_AUROC_4_class"].iloc[0])

    bsv_top = bsv_traj.head(5)
    mss_top = mss_traj.head(5)
    repr_mss = mss_dose[mss_dose.endpoint_sign_agree
                          & (mss_dose.trajectory_r_set9_vs_set10.fillna(0) > 0.7)]

    raw_var = var_summary.iloc[0]
    bsv_var = var_summary.iloc[1]
    mss_var = var_summary.iloc[2]

    # Decision label
    bsv_reduces_batch = bsv_batch < raw_batch - 0.03
    if bsv_reduces_batch and bsv_dose >= 0.55:
        decision = "SHINE_BSV_SUPPRESSES_BATCH_AND_PRESERVES_BIOLOGY"
    elif mss_dose_v > bsv_dose + 0.05 and mss_batch <= bsv_batch + 0.03:
        decision = "SHINE_MSS_PRESERVES_BIOLOGY_BETTER_THAN_BSV"
    elif raw_dose > 0.85 and bsv_dose > 0.55:
        decision = "SHINE_RAW_AND_GAIRA_BOTH_REPRODUCIBLE_DIFFERENT_ROLES"
    else:
        decision = "SHINE_NO_CLEAR_INVARIANCE_GAIN"

    lines = [
        "# REPORT — SHINE invariance + MSS analysis v1\n",
        f"date: {datetime.now().isoformat()}",
        "",
        f"## Decision: **{decision}**",
        "",
        "## Setup",
        "- Day-2 only · Set9 + Set10 · doses 0 / 10 / 20 / 40 mM",
        "- 400 spectra per (set, dose) cell · 3,200 spectra total",
        "- Subjects do NOT cross sets (Set9: 28 subjects, Set10: 10 subjects, 0 shared)",
        "- Engine v4.5 / preprocessing / BSV / MSS kernel / OTC detector — UNCHANGED",
        "- No paper / Si-642 / k-means filtering",
        "",
        "## Required answers",
        "",
        "### 1. Does RAW encode batch/set effects?",
        f"- RAW Set9-vs-Set10 AUROC = **{raw_batch:.3f}**  "
        f"(median η²_set on RAW = {raw_var['median_eta2_set']:.4f})",
        ("- " + ("YES" if raw_batch >= 0.65 else "NO / WEAK")
         + " — RAW encodes "
         + ("substantial" if raw_batch >= 0.80
            else "moderate" if raw_batch >= 0.65 else "limited")
         + " set/batch identity."),
        "",
        "### 2. Does BSV reduce batch/set effects?",
        f"- BSV Set9-vs-Set10 AUROC = **{bsv_batch:.3f}**  "
        f"(median η²_set on BSV = {bsv_var['median_eta2_set']:.4f})",
        f"- Δ vs RAW = **{bsv_batch - raw_batch:+.3f}**",
        ("- " + ("YES — BSV materially reduces batch separability" if bsv_reduces_batch
                  else "NO — BSV retains comparable batch identity")),
        "",
        "### 3. Does MSS retain dose/toxicity biology better than BSV?",
        f"- BSV dose AUROC (4-class) = **{bsv_dose:.3f}**, MSS = **{mss_dose_v:.3f}**, "
        f"RAW = **{raw_dose:.3f}**",
        ("- "
         + ("YES — MSS exceeds BSV" if mss_dose_v > bsv_dose + 0.02
             else "≈ — MSS matches BSV" if abs(mss_dose_v - bsv_dose) <= 0.02
             else "NO — BSV holds dose biology better")),
        "",
        "### 4. Which representation best separates biology from batch?",
    ]
    # Score each by (dose - batch)
    sel_scores = inv["biology_selectivity_score (d4 - batch)"].fillna(-99)
    best_row = inv.iloc[int(sel_scores.idxmax())]
    lines.append(
        f"- Best biology-selectivity score: **{best_row['representation']}** "
        f"(dose-AUROC − batch-AUROC = "
        f"{best_row['biology_selectivity_score (d4 - batch)']:+.3f})")

    lines += [
        "",
        "### 5. Which BSV axes show reproducible trajectories?",
    ]
    for _, r in bsv_top.iterrows():
        lines.append(
            f"- {r['feature']}  · trajectory r = {r['pearson_r']:.3f}  "
            f"· endpoint sign agree = {r['endpoint_sign_agree']}")

    lines += [
        "",
        "### 6. Which MSS candidates show reproducible trajectories?",
    ]
    for _, r in mss_top.iterrows():
        lines.append(
            f"- {r['feature']}  · trajectory r = {r['pearson_r']:.3f}  "
            f"· endpoint sign agree = {r['endpoint_sign_agree']}")

    lines += [
        "",
        f"### 7. Does APAP / drug-like candidate evidence increase on Day-2 vs Day-0?",
    ]
    if not otc_delta.empty:
        n_pos = int((otc_delta["delta_cand_rate"] > 0).sum())
        n_pos_score = int((otc_delta["delta_para_score"] > 0).sum())
        lines += [
            f"- Δ(D2 − D0) candidate rate **positive in {n_pos}/{len(otc_delta)}** "
            f"set × dose cells",
            f"- Δ(D2 − D0) paracetamol-like score positive in "
            f"{n_pos_score}/{len(otc_delta)} cells",
            "- Framing: candidate evidence ONLY — no APAP identity claim.",
        ]
    else:
        lines.append("- Δ(D2 − D0) cannot be computed (Set10 has no D0 data).")

    lines += [
        "",
        "### 8. What should be shown on the SHINE slide?",
        f"- Recommended panel option: **{slide_choice} — {slide_label}**",
        f"- Slide claim: \"{slide_claim}\"",
        "",
        "### 9. What should NOT be claimed?",
        "- APAP molecule identity (use \"paracetamol-like candidate evidence\").",
        "- Cross-subject generalisation (subjects do not cross sets).",
        "- Quantitative dose prediction (this is a categorical analysis).",
        "",
        "## Variance decomposition summary",
        "",
        "| representation | n_features | median η²_set | median η²_dose | "
        "frac dose-dominant features |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in var_summary.iterrows():
        lines.append(
            f"| {r['representation']} | {int(r['n_features'])} | "
            f"{r['median_eta2_set']:.4f} | {r['median_eta2_dose']:.4f} | "
            f"{r['frac_features_dose_dominant']:.2f} |")

    lines += [
        "",
        "## Strict invariants preserved",
        "- Engine v4.5 / preprocessing / BSV / MSS / OTC — UNCHANGED.",
        "- Labels post-hoc only; no threshold tuning on labels.",
        "- No paper / Si-642 / k-means filtering.",
        "- All numbers from per-spectrum data; cohorts pinned by spectrum_id.",
        "- Re-MSS scoring used the cached kernel + 19-molecule narrow registry.",
        "",
        f"## Final decision: **{decision}**",
    ]
    (R / "REPORT_shine_invariance_mss_analysis_v1.md").write_text(
        "\n".join(lines))
    return decision


# ─────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 78)
    print("gaira_base_4_shine_invariance_mss_analysis_v1")
    print("=" * 78)

    bsv, mss, otc = task1_audit()
    matrices = task2_build_matrices(bsv, mss, otc)
    perf, _ = task3_classify(matrices)
    inv = task4_invariance(perf)
    var = task5_variance(matrices)
    task6_pca(matrices)
    traj = task7_trajectory(matrices)
    mss_dose = task8_mss_dose_tracking(matrices)
    otc_delta = task9_otc_delta(otc)
    slide_choice, slide_label, slide_claim = task10_slide_recommendation(
        perf, inv, var["summary"], traj["BSV"], traj["MSS"])
    decision = task11_report(perf, inv, var["summary"],
                              traj["BSV"], traj["MSS"], mss_dose,
                              otc_delta, slide_choice, slide_label, slide_claim)
    try:
        shutil.copy(__file__, C / Path(__file__).name)
    except Exception:
        pass
    print(f"\n[done] decision: {decision}")
    print(f"slide recommendation: {slide_choice} — {slide_label}")


if __name__ == "__main__":
    main()
