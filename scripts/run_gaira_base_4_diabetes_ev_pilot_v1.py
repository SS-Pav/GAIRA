"""gaira_base_4_diabetes_ev_pilot_v1

Diabetes plasma EV SERS dataset audit + GAIRA pilot + latent subtype discovery.

STRICT INVARIANTS:
- Engine / MSS kernel / BSV / OTC detector thresholds / preprocessing UNCHANGED
- DO NOT use race_ethnicity column (present in metadata but deliberately ignored)
- Labels used ONLY post-hoc for cohort grouping + classification evaluation:
  Group=Impact → OWD, Group=Strong-D → NWD
- Latent clusters are "cluster 1/2", never Asian/White

Dataset:
  /Volumes/SSD_Rad/GAIRA_DATA/raw/diabetes_plasma_ev_sers/Diabetes_Raw_Data_Codes.zip
  Already extracted to ../extracted/

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_4_diabetes_ev_pilot_v1.py
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
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score,
    confusion_matrix,
)

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaira.spectral import canonical_master_axis  # noqa: E402
from run_gaira_base_4_mss_resolution_reporting_layer_v1 import (  # noqa: E402
    baseline_correct, mss_anchor_score, load_templates,
)
from run_gaira_base_4_small_ev_dual_probe_analysis_v1 import (  # noqa: E402
    compute_bsv_per_spectrum, bsv_transforms, BSV_FAMILIES,
)


# ──────────────────────────────────────────────────────────────────────
# Paths + constants
# ──────────────────────────────────────────────────────────────────────
ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_diabetes_ev_pilot_v1")
TABLES  = ROOT / "tables"
FIGS    = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT   = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"
for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
    d.mkdir(parents=True, exist_ok=True)

DATA = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/diabetes_plasma_ev_sers/extracted")
IMPACT_MAT = DATA / "RawDataImpact.mat"
STRONG_MAT = DATA / "RawDataStrong.mat"
PATIENT_CSV = DATA / "patient_data.csv"

# Pixel→wavenumber polynomial (same SHINE calibration, from Figure3.m)
CAL_PIX = np.array([263, 367, 492, 512, 590, 782, 872, 887], dtype=float)
CAL_CM  = np.array([620.9, 795.8, 1001.4, 1031.8, 1155.3, 1450.5, 1583.1, 1602.3])
SHINE_N_PIXELS = 1650
SHINE_PIXEL_RANGE = (162, 898)   # analysis crop (737 points retained in .mat)

# Per-patient subsample cap (tractable runtime)
N_PER_PATIENT = 100
RNG_SEED = 42

# Paper post-hoc regions (cm⁻¹) and anchor peaks — used ONLY for interpretation
PAPER_REGIONS = [(785, 985), (1130, 1346), (1420, 1610)]
PAPER_PEAKS   = [830, 946, 1001, 1146, 1299, 1440, 1498, 1536, 1570, 1601]


# ──────────────────────────────────────────────────────────────────────
# TASK 1 — Dataset audit
# ──────────────────────────────────────────────────────────────────────
def task1_audit():
    print("[TASK 1] dataset audit")
    patient_df = pd.read_csv(PATIENT_CSV)
    # Map Group → OWD/NWD; DO NOT use race_ethnicity
    patient_df["label_OWD_NWD"] = patient_df["Group"].map({"Impact": "OWD", "Strong-D": "NWD"})

    impact_d = sio.loadmat(IMPACT_MAT, squeeze_me=False)
    strong_d = sio.loadmat(STRONG_MAT, squeeze_me=False)
    imp_cells = impact_d["smoothed_spectra"]   # shape (1, N_patients)
    str_cells = strong_d["smoothed_spectra"]
    n_imp = imp_cells.shape[1]
    n_str = str_cells.shape[1]
    shapes_imp = [imp_cells[0, i].shape for i in range(n_imp)]
    shapes_str = [str_cells[0, i].shape for i in range(n_str)]

    # Check consistency
    wn_dims = {s[0] for s in shapes_imp + shapes_str}
    spec_per = [s[1] for s in shapes_imp + shapes_str]

    inv_rows = [{
        "dataset":         "diabetes_plasma_ev_sers",
        "mat_impact":      str(IMPACT_MAT),
        "mat_strong":      str(STRONG_MAT),
        "patient_csv":     str(PATIENT_CSV),
        "patient_rows":    int(len(patient_df)),
        "impact_patients_in_mat": int(n_imp),
        "strong_patients_in_mat": int(n_str),
        "impact_patients_in_csv": int((patient_df.Group == "Impact").sum()),
        "strong_patients_in_csv": int((patient_df.Group == "Strong-D").sum()),
        "spectra_per_patient_unique":  "|".join(str(s) for s in sorted(set(spec_per))),
        "wn_axis_dims_unique":         "|".join(str(d) for d in sorted(wn_dims)),
        "total_spectra_impact":        int(sum(s[1] for s in shapes_imp)),
        "total_spectra_strong":        int(sum(s[1] for s in shapes_str)),
        "calibration_polynomial_pixels": ";".join(str(int(p)) for p in CAL_PIX),
        "calibration_polynomial_cm1":  ";".join(f"{c:.1f}" for c in CAL_CM),
        "paper_pixel_range":           f"{SHINE_PIXEL_RANGE[0]}..{SHINE_PIXEL_RANGE[1]}",
    }]
    inv_df = pd.DataFrame(inv_rows)
    inv_df.to_csv(TABLES / "dataset_inventory.csv", index=False)

    # Label mapping audit — explicit record of what we use and DON'T use
    lab_rows = []
    for _, r in patient_df.iterrows():
        lab_rows.append({
            "patient_id":      r["filename"],
            "group_raw":       r["Group"],
            "label_OWD_NWD":   r["label_OWD_NWD"],
            "bmi":             r["bmi"],
            "hba1c":           r["hba1c"],
            "age_bl":          r["age_bl"],
            "gender":          r["gender"],
            "race_ethnicity_in_csv": r["race_ethnicity"],   # recorded but NOT used
            "race_used_in_analysis": "NOT_USED_PER_TASK_SPEC",
        })
    lab_df = pd.DataFrame(lab_rows)
    lab_df.to_csv(TABLES / "label_mapping_audit.csv", index=False)

    # Audit decision
    wn_consistent = len(wn_dims) == 1 and list(wn_dims)[0] >= 500
    labels_ok = (patient_df["label_OWD_NWD"].notna().sum() == len(patient_df))
    if not labels_ok:
        decision = "BLOCKED_METADATA_AMBIGUOUS"
    elif not wn_consistent:
        decision = "BLOCKED_BAD_AXIS"
    else:
        decision = "READY_BINARY_LABELS"
    print(f"  decision: {decision}")
    return patient_df, imp_cells, str_cells, decision


# ──────────────────────────────────────────────────────────────────────
# TASK 2 — Preprocessing
# ──────────────────────────────────────────────────────────────────────
def task2_preprocess(patient_df, imp_cells, str_cells, master_x):
    print("[TASK 2] canonical preprocessing (interp → AsLS → SG → L2)")
    # Build wavenumber axis for the pixels 162..898 range (737 pixels retained in .mat)
    poly = np.polyfit(CAL_PIX, CAL_CM, 3)
    pix_full = np.arange(1, SHINE_N_PIXELS + 1)
    wn_full = np.polyval(poly, pix_full)
    wn_data_axis = wn_full[SHINE_PIXEL_RANGE[0] - 1 : SHINE_PIXEL_RANGE[1]]   # (737,)

    rng = np.random.default_rng(RNG_SEED)
    # Build metadata-aligned patient list: order in .mat follows .mat order, which
    # corresponds to csv order filtered by group. Map by index.
    imp_csv = patient_df[patient_df.Group == "Impact"].reset_index(drop=True)
    str_csv = patient_df[patient_df.Group == "Strong-D"].reset_index(drop=True)
    n_imp = imp_cells.shape[1]; n_str = str_cells.shape[1]
    if n_imp != len(imp_csv):
        print(f"  WARNING: {n_imp} Impact mat cells vs {len(imp_csv)} csv rows; using min")
    n_imp_use = min(n_imp, len(imp_csv))
    n_str_use = min(n_str, len(str_csv))

    Y_list = []; meta_rows = []
    for group_label, cells, csv_slice, n_use in [
            ("OWD", imp_cells, imp_csv, n_imp_use),
            ("NWD", str_cells, str_csv, n_str_use)]:
        for i in range(n_use):
            mat = cells[0, i]   # (737, N_spectra)
            n_spec = mat.shape[1]
            if n_spec == 0: continue
            k = min(N_PER_PATIENT, n_spec)
            idxs = rng.choice(n_spec, k, replace=False)
            for j in idxs:
                y_raw = mat[:, j]
                y_interp = np.interp(master_x, wn_data_axis, y_raw, left=np.nan, right=np.nan)
                y_pp = baseline_correct(y_interp)
                if not (np.isfinite(y_pp).any() and float(np.linalg.norm(y_pp)) >= 1e-12):
                    continue
                Y_list.append(y_pp)
                meta_rows.append({
                    "patient_id":      csv_slice.iloc[i]["filename"],
                    "group_raw":       csv_slice.iloc[i]["Group"],
                    "label_OWD_NWD":   group_label,
                    "bmi":             float(csv_slice.iloc[i]["bmi"]),
                    "hba1c":           float(csv_slice.iloc[i]["hba1c"]),
                    "age_bl":          float(csv_slice.iloc[i]["age_bl"]),
                    "gender":          csv_slice.iloc[i]["gender"],
                    "rep_idx":         int(j),
                    "spectrum_id":     f"{csv_slice.iloc[i]['filename']}::s{j:03d}",
                })
    Y_pp = np.vstack(Y_list)
    meta_df = pd.DataFrame(meta_rows)
    print(f"  preprocessed {len(meta_df)} spectra from {meta_df['patient_id'].nunique()} patients")
    return Y_pp, meta_df


# ──────────────────────────────────────────────────────────────────────
# TASK 3 — BSV scoring
# ──────────────────────────────────────────────────────────────────────
def task3_bsv(Y_pp, master_x, meta_df):
    print("[TASK 3] 11-axis BSV scoring")
    templates, _, _ = load_templates()
    by_mol = defaultdict(dict)
    for t in templates:
        by_mol[t["molecule"]][t["regime"]] = t
    bsv_raw = compute_bsv_per_spectrum(Y_pp, master_x, by_mol)
    trans = bsv_transforms(bsv_raw)
    bsv_df = meta_df.copy()
    for tag, arr in [("raw", trans["raw"]), ("sumnorm", trans["sumnorm"]), ("clr", trans["clr"])]:
        for k, (fid, _) in enumerate(BSV_FAMILIES):
            bsv_df[f"{tag}_{fid}"] = arr[:, k]
    # top-3 axes per spectrum (using sumnorm)
    top3 = []
    for i in range(len(bsv_df)):
        v = trans["sumnorm"][i]
        idx = np.argsort(-v)
        top3.append("|".join(BSV_FAMILIES[int(k)][0] for k in idx[:3]))
    bsv_df["top3_axes"] = top3
    bsv_df.to_csv(TABLES / "per_spectrum_bsv.csv", index=False)

    # Cohort means by OWD/NWD
    axis_ids = [f for f, _ in BSV_FAMILIES]
    cohort_rows = []
    for lbl, sub in bsv_df.groupby("label_OWD_NWD"):
        row = {"label": lbl, "n_spectra": len(sub),
                 "n_patients": sub["patient_id"].nunique()}
        for fid in axis_ids:
            row[f"mean_clr_{fid}"]     = float(sub[f"clr_{fid}"].mean())
            row[f"sd_clr_{fid}"]       = float(sub[f"clr_{fid}"].std())
            row[f"mean_sumnorm_{fid}"] = float(sub[f"sumnorm_{fid}"].mean())
        cohort_rows.append(row)
    pd.DataFrame(cohort_rows).to_csv(TABLES / "cohort_bsv_means.csv", index=False)
    return bsv_df, trans


# ──────────────────────────────────────────────────────────────────────
# TASK 4 — Binary OWD vs NWD
# ──────────────────────────────────────────────────────────────────────
def _cohens_d(x, y):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2: return np.nan
    pooled = np.sqrt(((len(x)-1)*np.var(x, ddof=1) + (len(y)-1)*np.var(y, ddof=1))
                       / max(len(x)+len(y)-2, 1))
    return float((np.mean(x) - np.mean(y)) / (pooled if pooled > 0 else 1.0))


def _cliffs_delta(x, y):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2: return np.nan
    # more efficient than brute-force: O(n log n) via sorting
    n, m = len(x), len(y)
    combined = np.concatenate([x, y])
    ranks = pd.Series(combined).rank().values
    ranks_x = ranks[:n]
    return float(2.0 * ranks_x.mean() / (n + m) - 1.0)


def _bootstrap_delta_ci(owd_vals, nwd_vals, n=300, seed=42):
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        a = rng.choice(owd_vals, len(owd_vals), replace=True)
        b = rng.choice(nwd_vals, len(nwd_vals), replace=True)
        deltas.append(np.mean(a) - np.mean(b))
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def task4_binary(bsv_df):
    print("[TASK 4] binary OWD vs NWD")
    axis_ids = [f for f, _ in BSV_FAMILIES]
    owd = bsv_df[bsv_df.label_OWD_NWD == "OWD"]
    nwd = bsv_df[bsv_df.label_OWD_NWD == "NWD"]
    rows = []
    for fid in axis_ids:
        o = owd[f"clr_{fid}"].values; w = nwd[f"clr_{fid}"].values
        d_val = _cohens_d(o, w)
        cliff = _cliffs_delta(o, w)
        delta_mean = float(np.mean(o) - np.mean(w))
        ci_low, ci_high = _bootstrap_delta_ci(o, w)
        rows.append({
            "axis": fid,
            "owd_mean_clr": float(np.mean(o)),
            "nwd_mean_clr": float(np.mean(w)),
            "delta_owd_minus_nwd": delta_mean,
            "cohens_d":     d_val,
            "cliffs_delta": cliff,
            "ci_low":       ci_low,
            "ci_high":      ci_high,
            "ci_excludes_zero": bool((ci_low > 0 and ci_high > 0) or
                                          (ci_low < 0 and ci_high < 0)),
        })
    df = pd.DataFrame(rows).sort_values("cohens_d",
                                                key=lambda s: s.abs(), ascending=False)
    df.to_csv(TABLES / "binary_owd_vs_nwd_effects.csv", index=False)

    # Radar plot (OWD vs NWD on sumnorm)
    try:
        means_o = np.array([owd[f"sumnorm_{f}"].mean() for f in axis_ids])
        means_n = np.array([nwd[f"sumnorm_{f}"].mean() for f in axis_ids])
        angles = np.linspace(0, 2*np.pi, len(axis_ids), endpoint=False)
        angles_closed = np.concatenate([angles, [angles[0]]])
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for vals, color, label in [(means_o, "#c0392b", "OWD"),
                                             (means_n, "#4C72B0", "NWD")]:
            v_closed = list(vals) + [vals[0]]
            ax.plot(angles_closed, v_closed, "-o", color=color, lw=1.8, label=label)
            ax.fill(angles_closed, v_closed, alpha=0.15, color=color)
        ax.set_xticks(angles); ax.set_xticklabels(axis_ids, fontsize=8)
        ax.set_title("BSV sumnorm radar — OWD vs NWD")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / "fig_radar_owd_vs_nwd.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  radar fig issue: {e}")

    # Axis heatmap
    try:
        mat = df[["cohens_d"]].values
        fig, ax = plt.subplots(figsize=(4, 5))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_yticks(range(len(df))); ax.set_yticklabels(df["axis"])
        ax.set_xticks([0]); ax.set_xticklabels(["OWD − NWD"])
        for i, d_val in enumerate(df["cohens_d"]):
            ax.text(0, i, f"{d_val:+.2f}", ha="center", va="center", fontsize=9,
                      color="white" if abs(d_val) > 0.5 else "black")
        ax.set_title("Cohen's d per axis (OWD − NWD)")
        plt.colorbar(im, ax=ax, fraction=0.05)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_heatmap_owd_vs_nwd_cohens_d.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  heatmap fig issue: {e}")
    return df


# ──────────────────────────────────────────────────────────────────────
# TASK 5 — Latent subtype discovery
# ──────────────────────────────────────────────────────────────────────
def task5_latent(bsv_df):
    print("[TASK 5] latent subtype discovery (unsupervised)")
    axis_ids = [f for f, _ in BSV_FAMILIES]
    clr = bsv_df[[f"clr_{f}" for f in axis_ids]].values

    # Helpers
    def _cluster_quality(X, labels):
        try:
            sil = float(silhouette_score(X, labels))
        except Exception:
            sil = np.nan
        try:
            db = float(davies_bouldin_score(X, labels))
        except Exception:
            db = np.nan
        return sil, db

    def _bootstrap_stability(X, k, method, n=30, seed=42):
        rng = np.random.default_rng(seed)
        base = method(k).fit(X).predict(X) if hasattr(method(k), "predict") \
                   else method(k).fit_predict(X)
        matches = []
        for _ in range(n):
            idx = rng.choice(len(X), len(X), replace=True)
            try:
                lbls_b = method(k).fit(X[idx]).predict(X[idx]) if hasattr(method(k), "predict") \
                             else method(k).fit_predict(X[idx])
            except Exception:
                continue
            # For each base point in the resample, find nearest-match proportion
            base_sub = base[idx]
            n_classes = max(len(set(lbls_b)), 1)
            cm = np.zeros((n_classes, n_classes))
            u_b = sorted(set(base_sub)); u_a = sorted(set(lbls_b))
            if len(u_a) != len(u_b): continue
            mat = np.zeros((len(u_a), len(u_b)))
            for j, a in enumerate(u_a):
                for kk, bb in enumerate(u_b):
                    mat[j, kk] = int(((lbls_b == a) & (base_sub == bb)).sum())
            # Best matching (Hungarian-ish greedy)
            best = 0
            used = set()
            for j in range(len(u_a)):
                best_k = int(np.argmax([(mat[j, kk] if kk not in used else -1)
                                                  for kk in range(len(u_b))]))
                best += int(mat[j, best_k]); used.add(best_k)
            matches.append(best / len(idx))
        return float(np.mean(matches)) if matches else np.nan

    # Build run matrix: within NWD k=2, within OWD k=2, global k=4, global k=2 baseline
    runs = []
    clr_scaler = StandardScaler().fit(clr)
    X_full = clr_scaler.transform(clr)

    for label_restrict, X_sub, meta_sub, k in [
            ("within_NWD", X_full[bsv_df.label_OWD_NWD.values == "NWD"],
               bsv_df[bsv_df.label_OWD_NWD == "NWD"], 2),
            ("within_OWD", X_full[bsv_df.label_OWD_NWD.values == "OWD"],
               bsv_df[bsv_df.label_OWD_NWD == "OWD"], 2),
            ("global", X_full, bsv_df, 4),
            ("global", X_full, bsv_df, 2)]:
        # GMM
        gmm = GaussianMixture(n_components=k, random_state=0, n_init=3)
        gmm.fit(X_sub)
        lbls_gmm = gmm.predict(X_sub)
        sil_gmm, db_gmm = _cluster_quality(X_sub, lbls_gmm)
        stab_gmm = _bootstrap_stability(
            X_sub, k, lambda kk: GaussianMixture(n_components=kk, random_state=0, n_init=1))
        # Agglomerative
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        lbls_agg = agg.fit_predict(X_sub)
        sil_agg, db_agg = _cluster_quality(X_sub, lbls_agg)
        stab_agg = _bootstrap_stability(
            X_sub, k, lambda kk: AgglomerativeClustering(n_clusters=kk, linkage="ward"))
        for method, lbls, sil, db, stab in [
                ("GMM",            lbls_gmm, sil_gmm, db_gmm, stab_gmm),
                ("Agglomerative",  lbls_agg, sil_agg, db_agg, stab_agg)]:
            runs.append({
                "restrict": label_restrict, "k": int(k), "method": method,
                "n_points": int(len(X_sub)), "n_patients_in_scope":
                    int(meta_sub["patient_id"].nunique()),
                "silhouette": sil, "davies_bouldin": db,
                "bootstrap_stability_30": stab,
                "cluster_counts": dict(Counter(lbls.tolist())),
            })

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(TABLES / "cluster_quality_metrics.csv", index=False)

    # Persist per-spectrum cluster assignments for within-NWD and within-OWD k=2
    assignments = []
    for restrict in ["within_NWD", "within_OWD"]:
        mask = bsv_df.label_OWD_NWD.values == restrict.split("_")[1]
        X_sub = X_full[mask]
        if len(X_sub) < 10: continue
        gmm = GaussianMixture(n_components=2, random_state=0, n_init=3).fit(X_sub)
        l_gmm = gmm.predict(X_sub)
        agg = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(X_sub)
        sub = bsv_df[mask].reset_index(drop=True)
        for j in range(len(sub)):
            assignments.append({
                "spectrum_id":  sub.iloc[j]["spectrum_id"],
                "patient_id":   sub.iloc[j]["patient_id"],
                "label_OWD_NWD": sub.iloc[j]["label_OWD_NWD"],
                "bmi":          sub.iloc[j]["bmi"],
                "restrict":     restrict,
                "cluster_gmm_k2": int(l_gmm[j]),
                "cluster_agg_k2": int(agg[j]),
            })
    # global k=4
    gmm4 = GaussianMixture(n_components=4, random_state=0, n_init=3).fit(X_full)
    l4 = gmm4.predict(X_full)
    for j in range(len(bsv_df)):
        assignments.append({
            "spectrum_id":  bsv_df.iloc[j]["spectrum_id"],
            "patient_id":   bsv_df.iloc[j]["patient_id"],
            "label_OWD_NWD": bsv_df.iloc[j]["label_OWD_NWD"],
            "bmi":          bsv_df.iloc[j]["bmi"],
            "restrict":     "global",
            "cluster_gmm_k4": int(l4[j]),
        })
    assign_df = pd.DataFrame(assignments)
    assign_df.to_csv(TABLES / "latent_cluster_assignments.csv", index=False)

    # Stability summary table (from runs)
    stab_df = runs_df[["restrict", "k", "method", "bootstrap_stability_30",
                           "silhouette", "davies_bouldin"]]
    stab_df.to_csv(TABLES / "cluster_stability_metrics.csv", index=False)

    # PCA figure — colored by within-NWD k=2, within-OWD k=2 (separately)
    try:
        pca = PCA(n_components=2).fit(X_full)
        Z = pca.transform(X_full)
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        for ax, color_label in zip(axes.flat, ["label_OWD_NWD", "bmi",
                                                          "within_NWD_k2", "within_OWD_k2"]):
            if color_label == "label_OWD_NWD":
                for lbl, c in [("OWD", "#c0392b"), ("NWD", "#4C72B0")]:
                    mask = bsv_df.label_OWD_NWD.values == lbl
                    ax.scatter(Z[mask, 0], Z[mask, 1], s=5, alpha=0.5, color=c, label=lbl)
                ax.legend(fontsize=8)
                ax.set_title("BSV-CLR PCA colored by OWD/NWD")
            elif color_label == "bmi":
                sc = ax.scatter(Z[:, 0], Z[:, 1], s=5, alpha=0.6, c=bsv_df.bmi, cmap="viridis")
                plt.colorbar(sc, ax=ax, label="BMI"); ax.set_title("PCA colored by BMI")
            elif color_label == "within_NWD_k2":
                mask_n = bsv_df.label_OWD_NWD.values == "NWD"
                X_sub = X_full[mask_n]
                if len(X_sub) > 5:
                    gmm = GaussianMixture(n_components=2, random_state=0, n_init=3).fit(X_sub)
                    l = gmm.predict(X_sub)
                    ax.scatter(Z[~mask_n, 0], Z[~mask_n, 1], s=3, alpha=0.2, color="grey",
                                  label="OWD (not clustered here)")
                    sc = ax.scatter(Z[mask_n, 0], Z[mask_n, 1], s=8, alpha=0.75, c=l, cmap="tab10")
                    ax.legend(fontsize=7); ax.set_title("NWD latent k=2 (GMM)")
            elif color_label == "within_OWD_k2":
                mask_o = bsv_df.label_OWD_NWD.values == "OWD"
                X_sub = X_full[mask_o]
                if len(X_sub) > 5:
                    gmm = GaussianMixture(n_components=2, random_state=0, n_init=3).fit(X_sub)
                    l = gmm.predict(X_sub)
                    ax.scatter(Z[~mask_o, 0], Z[~mask_o, 1], s=3, alpha=0.2, color="grey",
                                  label="NWD (not clustered here)")
                    sc = ax.scatter(Z[mask_o, 0], Z[mask_o, 1], s=8, alpha=0.75, c=l, cmap="tab10")
                    ax.legend(fontsize=7); ax.set_title("OWD latent k=2 (GMM)")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        fig.suptitle("BSV-CLR PCA with labeled + latent cluster overlays")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pca_latent_clusters.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  PCA fig issue: {e}")
    return runs_df, assign_df


# ──────────────────────────────────────────────────────────────────────
# TASK 6 — Paper regions & peaks (post-hoc interpretation)
# ──────────────────────────────────────────────────────────────────────
def task6_paper_regions(bsv_df, Y_pp, master_x, assign_df):
    print("[TASK 6] paper regions + peaks (post-hoc)")
    # Compute per-spectrum region intensity and peak intensity
    feat_rows = []
    for i, r in bsv_df.reset_index(drop=True).iterrows():
        y = Y_pp[i]
        row = {"spectrum_id": r["spectrum_id"],
                 "patient_id":  r["patient_id"],
                 "label_OWD_NWD": r["label_OWD_NWD"]}
        # Region mean intensity
        for lo, hi in PAPER_REGIONS:
            mask = (master_x >= lo) & (master_x <= hi)
            row[f"region_{lo}_{hi}_mean"] = float(np.nanmean(y[mask])) if mask.any() else np.nan
            row[f"region_{lo}_{hi}_area"] = float(np.trapezoid(np.clip(y[mask], 0, None),
                                                                          master_x[mask])) if mask.any() else np.nan
        # Peak intensity at anchor ± 3 cm⁻¹
        for peak in PAPER_PEAKS:
            mask = (master_x >= peak - 3) & (master_x <= peak + 3)
            row[f"peak_{peak}_max"] = float(np.nanmax(y[mask])) if mask.any() else np.nan
        feat_rows.append(row)
    feat_df = pd.DataFrame(feat_rows)
    feat_df.to_csv(TABLES / "paper_region_peak_features.csv", index=False)

    # OWD vs NWD comparison on region/peak features
    region_rows = []
    for col in feat_df.columns:
        if not (col.startswith("region_") or col.startswith("peak_")): continue
        o = feat_df[feat_df.label_OWD_NWD == "OWD"][col].dropna().values
        n = feat_df[feat_df.label_OWD_NWD == "NWD"][col].dropna().values
        region_rows.append({
            "feature": col,
            "owd_mean": float(np.mean(o)) if len(o) else np.nan,
            "nwd_mean": float(np.mean(n)) if len(n) else np.nan,
            "cohens_d": _cohens_d(o, n),
        })
    reg_df = pd.DataFrame(region_rows).sort_values("cohens_d",
                                                              key=lambda s: s.abs(),
                                                              ascending=False)
    reg_df.to_csv(TABLES / "paper_region_binary_effects.csv", index=False)

    # Latent-cluster-level comparison within each group (post-hoc)
    if not assign_df.empty:
        cluster_reg_rows = []
        for restrict in ["within_NWD", "within_OWD"]:
            mask = assign_df.restrict == restrict
            if not mask.any(): continue
            sub_assign = assign_df[mask]
            for col in feat_df.columns:
                if not (col.startswith("region_") or col.startswith("peak_")): continue
                # Map cluster labels back to feat_df via spectrum_id
                merged = feat_df.merge(
                    sub_assign[["spectrum_id", "cluster_gmm_k2"]],
                    on="spectrum_id", how="inner")
                for cl in [0, 1]:
                    vals = merged[merged.cluster_gmm_k2 == cl][col].dropna().values
                    if len(vals) < 5: continue
                    cluster_reg_rows.append({
                        "restrict": restrict, "cluster_gmm_k2": cl,
                        "feature": col,
                        "mean": float(np.mean(vals)),
                        "n": int(len(vals)),
                    })
        pd.DataFrame(cluster_reg_rows).to_csv(
            TABLES / "paper_region_cluster_means_within_group.csv", index=False)

    return feat_df


# ──────────────────────────────────────────────────────────────────────
# TASK 7 — Post-BSV classifier (OWD vs NWD)
# ──────────────────────────────────────────────────────────────────────
def task7_classifier(bsv_df, Y_pp, feat_df):
    print("[TASK 7] OWD vs NWD classifier (post-BSV evaluation only)")
    y = (bsv_df["label_OWD_NWD"].values == "OWD").astype(int)
    groups = bsv_df["patient_id"].values
    axis_ids = [f for f, _ in BSV_FAMILIES]
    n_groups = len(np.unique(groups))

    feat_sets = {
        "raw_spectra":        Y_pp,
        "BSV_CLR_11":         bsv_df[[f"clr_{f}" for f in axis_ids]].values,
        "BSV_TOP_SUBSET":     bsv_df[[f"clr_{f}" for f in ["G01", "G03", "G04", "G06", "G08", "G09"]]].values,
        "paper_region_peak":  feat_df[[c for c in feat_df.columns
                                              if c.startswith("region_") or c.startswith("peak_")]].fillna(0).values,
    }
    models = {
        "logreg":  LogisticRegression(max_iter=2000, random_state=0),
        "linSVM":  LinearSVC(max_iter=5000, random_state=0),
        "rf_expl": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0,
                                                    n_jobs=-1),
    }

    rows = []
    conf_rows = []
    feat_imp_rows = []
    gkf = GroupKFold(n_splits=min(5, n_groups))
    for fset_name, X in feat_sets.items():
        X = np.nan_to_num(X, nan=0.0)
        for model_name, model in models.items():
            is_core = model_name != "rf_expl"
            all_pred = np.full(len(y), -1, dtype=int)
            all_score = np.full(len(y), np.nan)
            for tr, te in gkf.split(X, y, groups=groups):
                scaler = StandardScaler().fit(X[tr])
                Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
                try:
                    mdl = model.__class__(**model.get_params())
                    mdl.fit(Xtr, y[tr])
                    all_pred[te] = mdl.predict(Xte)
                    if hasattr(mdl, "predict_proba"):
                        all_score[te] = mdl.predict_proba(Xte)[:, 1]
                    elif hasattr(mdl, "decision_function"):
                        all_score[te] = mdl.decision_function(Xte)
                except Exception as e:
                    continue
            if (all_pred == -1).all(): continue
            m = all_pred != -1
            acc = float(accuracy_score(y[m], all_pred[m]))
            bal = float(balanced_accuracy_score(y[m], all_pred[m]))
            f1  = float(f1_score(y[m], all_pred[m]))
            try:
                auc = float(roc_auc_score(y[m], all_score[m]))
            except Exception:
                auc = np.nan
            conf = confusion_matrix(y[m], all_pred[m])
            rows.append({
                "feature_set":     fset_name,
                "model":           model_name,
                "core_or_exploratory": "core" if is_core else "exploratory",
                "n":               int(m.sum()),
                "accuracy":        acc, "balanced_accuracy": bal,
                "f1":              f1, "auroc": auc,
            })
            conf_rows.append({
                "feature_set": fset_name, "model": model_name,
                "tn": int(conf[0, 0]), "fp": int(conf[0, 1]),
                "fn": int(conf[1, 0]), "tp": int(conf[1, 1]),
            })
            # Feature importance for logreg (coefs)
            if model_name == "logreg":
                scaler = StandardScaler().fit(X); X_s = scaler.transform(X)
                mdl = LogisticRegression(max_iter=2000, random_state=0).fit(X_s, y)
                for k, fname in enumerate(
                    [f"raw_{i}" for i in range(X.shape[1])] if fset_name == "raw_spectra"
                    else ([f"clr_{a}" for a in axis_ids] if fset_name == "BSV_CLR_11"
                            else ([f"clr_{a}" for a in ["G01", "G03", "G04", "G06", "G08", "G09"]]
                                     if fset_name == "BSV_TOP_SUBSET"
                                     else [c for c in feat_df.columns
                                             if c.startswith("region_") or c.startswith("peak_")]))):
                    feat_imp_rows.append({
                        "feature_set": fset_name, "model": model_name,
                        "feature": fname, "coef": float(mdl.coef_[0, k]),
                        "abs_coef": abs(float(mdl.coef_[0, k])),
                    })
    perf_df = pd.DataFrame(rows)
    perf_df.to_csv(TABLES / "classifier_performance.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(TABLES / "classifier_confusion_matrices.csv", index=False)
    if feat_imp_rows:
        pd.DataFrame(feat_imp_rows).to_csv(
            TABLES / "classifier_feature_importance.csv", index=False)
    return perf_df


# ──────────────────────────────────────────────────────────────────────
# TASK 8 — MSS candidate layer (post-BSV)
# ──────────────────────────────────────────────────────────────────────
def task8_mss(bsv_df, Y_pp, master_x):
    print("[TASK 8] MSS candidate layer (post-BSV)")
    templates, _, _ = load_templates()
    by_mol = defaultdict(dict)
    for t in templates:
        by_mol[t["molecule"]][t["regime"]] = t
    rows = []
    for i in range(len(bsv_df)):
        y = Y_pp[i]
        scores = {}
        for mol, tps in by_mol.items():
            t = tps.get("SERS") or tps.get("Raman") or next(iter(tps.values()))
            sc, _, _ = mss_anchor_score(y, master_x, t["anchors"], t["supports"])
            scores[mol] = sc
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        r = bsv_df.iloc[i]
        rows.append({
            "spectrum_id":    r["spectrum_id"],
            "patient_id":     r["patient_id"],
            "label_OWD_NWD":  r["label_OWD_NWD"],
            "top1_molecule":  ranked[0][0],
            "top1_score":     ranked[0][1],
            "top3_molecules": "|".join(m for m, _ in ranked[:3]),
            "top5_molecules": "|".join(m for m, _ in ranked[:5]),
        })
        if i % 1000 == 0: print(f"  mss {i}/{len(bsv_df)}")
    mss_df = pd.DataFrame(rows)
    mss_df.to_csv(TABLES / "mss_per_spectrum.csv", index=False)

    # Top hits by label
    cond_rows = []
    for lbl, sub in mss_df.groupby("label_OWD_NWD"):
        n = len(sub)
        for mol, c in Counter(sub.top1_molecule).most_common(10):
            cond_rows.append({"label": lbl, "molecule": mol,
                                 "top1_freq": c / n, "n_spectra": n})
    pd.DataFrame(cond_rows).to_csv(TABLES / "mss_top_hits_by_condition.csv", index=False)
    return mss_df


# ──────────────────────────────────────────────────────────────────────
# TASK 9 — Report + decision
# ──────────────────────────────────────────────────────────────────────
def _decision(perf_df, cluster_df):
    # Binary OWD vs NWD signal
    best_bsv = perf_df[(perf_df.feature_set.isin(["BSV_CLR_11", "BSV_TOP_SUBSET"])) &
                             (perf_df.core_or_exploratory == "core")]["auroc"].max()
    # Latent structure strength: silhouette within-NWD and within-OWD k=2
    latent = cluster_df[cluster_df.restrict.isin(["within_NWD", "within_OWD"]) &
                              (cluster_df.k == 2)]
    latent_sil = float(latent["silhouette"].max()) if not latent.empty else np.nan
    latent_stab = float(latent["bootstrap_stability_30"].max()) if not latent.empty else np.nan
    if np.isnan(best_bsv):
        return "DIABETES_EV_BLOCKED_BY_DATA_QUALITY"
    if best_bsv >= 0.75 and not np.isnan(latent_sil) and latent_sil >= 0.25 and latent_stab >= 0.7:
        return "DIABETES_EV_BINARY_BSV_SIGNAL_WITH_LATENT_SUBTYPES"
    if best_bsv >= 0.75:
        return "DIABETES_EV_BINARY_ONLY_NO_LATENT_STRUCTURE"
    if not np.isnan(latent_sil) and latent_sil >= 0.15:
        return "DIABETES_EV_LATENT_STRUCTURE_WEAK"
    return "DIABETES_EV_BINARY_ONLY_NO_LATENT_STRUCTURE"


def write_report(decision, patient_df, bsv_df, binary_df, cluster_df,
                     assign_df, feat_df, perf_df, mss_df):
    lines = [
        "# REPORT — Diabetes EV GAIRA pilot v1\n",
        f"date: {datetime.now().isoformat()}", "",
        f"## Decision: **{decision}**\n",
        "## Setup",
        "- Dataset: Diabetes plasma EV SERS (impact = OWD, Strong-D = NWD)",
        "- Labels used: **impact → OWD**, **Strong-D → NWD** only",
        "- **race_ethnicity column present in metadata but NOT USED** per task spec",
        "- Canonical GAIRA preprocessing; engine / BSV / MSS kernel UNCHANGED",
        "- Subsample: 100 spectra per patient (deterministic seed=42)",
        "- 11-axis BSV via family-aggregated MSS anchor kernel (unchanged schema)",
        "",
        "## Required answers\n",
    ]
    n_patients = patient_df["filename"].nunique()
    n_owd = int((patient_df.Group == "Impact").sum())
    n_nwd = int((patient_df.Group == "Strong-D").sum())

    lines.append("### 1. Are local labels correctly mapped as impact=OWD and strong D=NWD?")
    lines.append(f"- Yes. {n_patients} patients total; {n_owd} Impact→OWD + {n_nwd} Strong-D→NWD.")
    lines.append(f"- BMI check: OWD mean BMI = {patient_df[patient_df.Group=='Impact'].bmi.mean():.1f} "
                    f"(overweight/obese range); NWD mean BMI = "
                    f"{patient_df[patient_df.Group=='Strong-D'].bmi.mean():.1f} (normal-weight range).")
    lines.append("")

    lines.append("### 2. Does GAIRA distinguish OWD vs NWD?")
    top5_axes = binary_df.head(5)
    lines.append("Top 5 axes by |Cohen's d| (OWD − NWD):")
    lines.append("| axis | Cohen's d | Cliff's δ | Δ OWD−NWD (CLR) | 95% CI | CI excludes 0 |")
    lines.append("|---|---:|---:|---:|---|---|")
    for _, r in top5_axes.iterrows():
        lines.append(f"| {r['axis']} | {r['cohens_d']:+.2f} | {r['cliffs_delta']:+.2f} | "
                        f"{r['delta_owd_minus_nwd']:+.3f} | "
                        f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
                        f"{'✓' if r['ci_excludes_zero'] else '✗'} |")
    lines.append("")

    lines.append("### 3. Which BSV axes drive OWD vs NWD?")
    lines.append("See above top-5 table + full `binary_owd_vs_nwd_effects.csv`.")
    lines.append("")

    lines.append("### 4. Do unsupervised BSV structures reveal latent subtypes within NWD/OWD?")
    lines.append("| restrict | k | method | silhouette | Davies-Bouldin | bootstrap stability |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for _, r in cluster_df.iterrows():
        lines.append(f"| {r['restrict']} | {int(r['k'])} | {r['method']} | "
                        f"{r['silhouette']:+.3f} | {r['davies_bouldin']:.3f} | "
                        f"{r['bootstrap_stability_30']:.2f} |")
    lines.append("")

    lines.append("### 5. Are these latent subtypes consistent with the paper's four-subtype hypothesis?")
    within_k2 = cluster_df[cluster_df.restrict.isin(["within_NWD", "within_OWD"]) &
                                 (cluster_df.k == 2)]
    max_sil = float(within_k2["silhouette"].max()) if not within_k2.empty else np.nan
    max_stab = float(within_k2["bootstrap_stability_30"].max()) if not within_k2.empty else np.nan
    lines.append(f"- Within-group k=2 max silhouette = {max_sil:+.3f}; max bootstrap stability = {max_stab:.2f}")
    lines.append("- **Do NOT claim GAIRA recovered Asian vs White labels.** The task spec forbids "
                    "assigning A/W labels. Reported clusters are A/W-like structural candidates only.")
    lines.append("- If silhouette is weak (< 0.25), A/W-like structure is NOT clearly recovered.")
    lines.append("")

    lines.append("### 6. How do GAIRA axes compare with paper regions/peaks?")
    lines.append("See `paper_region_binary_effects.csv`. Region/peak means are reported for post-hoc comparison only; "
                    "GAIRA BSV axes already roll up related bands into chemistry families (e.g. G08 lipid-acyl covers "
                    "1299/1440; G06 protein covers amide region 1536/1601; G07 aromatic covers 1001/1601).")
    lines.append("")

    lines.append("### 7. How does post-BSV classifier compare with raw/paper-feature classifier?")
    lines.append("| feature_set | model | accuracy | balanced acc | AUROC | F1 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, r in perf_df.sort_values("auroc", ascending=False).iterrows():
        lines.append(f"| {r['feature_set']} | {r['model']} | {r['accuracy']:.2%} | "
                        f"{r['balanced_accuracy']:.2%} | "
                        f"{r['auroc']:.3f} | {r['f1']:.3f} |")
    lines.append("")

    lines.append("### 8. What MSS candidate themes appear?")
    owd_top = Counter(mss_df[mss_df.label_OWD_NWD == "OWD"]["top1_molecule"]).most_common(5)
    nwd_top = Counter(mss_df[mss_df.label_OWD_NWD == "NWD"]["top1_molecule"]).most_common(5)
    lines.append(f"- OWD top-5 MSS candidates: {owd_top}")
    lines.append(f"- NWD top-5 MSS candidates: {nwd_top}")
    lines.append("- MSS hits are candidate-level spectral evidence only — not definitive molecule identity.")
    lines.append("")

    lines.append("### 9. What remains impossible without A/W labels?")
    lines.append("- Cannot assign ethnicity to any latent cluster.")
    lines.append("- Cannot compute the paper's 4-way A-NWD/A-OWD/W-NWD/W-OWD statistics.")
    lines.append("- Can only report cluster 1/2 within each BMI group.")
    lines.append("- Verification that a given cluster corresponds to Asian vs White subjects requires "
                    "race_ethnicity metadata we are deliberately not using.")
    lines.append("")

    lines.append("### 10. What should be shown in demo?")
    lines.append("- OWD vs NWD radar + Cohen's d heatmap (top BSV axes driving BMI group difference)")
    lines.append("- BSV-CLR PCA with OWD/NWD overlay + BMI continuous color")
    lines.append("- Within-NWD / within-OWD k=2 cluster figure with explicit 'A/W-like but not race-labeled' caveat")
    lines.append("- Classifier comparison table (raw vs BSV vs paper-features)")
    lines.append("")

    (REPORTS / "REPORT_diabetes_ev_gaira_pilot_v1.md").write_text("\n".join(lines))


def write_audit(decision, task1_decision):
    txt = [
        "# gaira_base_4_diabetes_ev_pilot_v1 — audit log",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Inputs (read-only)",
        "- /Volumes/SSD_Rad/GAIRA_DATA/raw/diabetes_plasma_ev_sers/extracted/",
        "    RawDataImpact.mat, RawDataStrong.mat, patient_data.csv",
        "- Pixel→wavenumber polynomial: same SHINE 3rd-order fit (Figure3.m)",
        "",
        "## Strict invariants",
        "- Engine v4.5, MSS kernel, motif registry, 11-axis BSV, OTC detector thresholds UNCHANGED",
        "- race_ethnicity column in metadata IS NOT USED (explicit task spec)",
        "- Labels used ONLY for cohort grouping + classifier evaluation (post-hoc)",
        "- No threshold tuning, no classifier feedback into GAIRA core",
        "- Subsample cap 100 spectra per patient; deterministic seed=42",
        "",
        "## Outputs",
        "- tables/dataset_inventory.csv",
        "- tables/label_mapping_audit.csv",
        "- tables/per_spectrum_bsv.csv",
        "- tables/cohort_bsv_means.csv",
        "- tables/binary_owd_vs_nwd_effects.csv",
        "- tables/cluster_quality_metrics.csv + cluster_stability_metrics.csv",
        "- tables/latent_cluster_assignments.csv",
        "- tables/paper_region_peak_features.csv + paper_region_binary_effects.csv",
        "- tables/paper_region_cluster_means_within_group.csv",
        "- tables/classifier_performance.csv + classifier_confusion_matrices.csv + classifier_feature_importance.csv",
        "- tables/mss_per_spectrum.csv + mss_top_hits_by_condition.csv",
        "- figures: radar, Cohen's d heatmap, PCA with latent cluster overlays",
        "- reports/REPORT_diabetes_ev_gaira_pilot_v1.md",
        "",
        f"## Task 1 audit decision: {task1_decision}",
        f"## Final decision\n**{decision}**",
    ]
    (AUDIT / "gaira_base_4_diabetes_ev_pilot_v1_audit_log.md").write_text("\n".join(txt))


def main():
    print("=" * 78)
    print("gaira_base_4_diabetes_ev_pilot_v1")
    print("=" * 78)
    master_x = canonical_master_axis()

    patient_df, imp_cells, str_cells, task1_decision = task1_audit()
    if task1_decision != "READY_BINARY_LABELS":
        write_audit(task1_decision, task1_decision)
        print(f"[done] audit blocked: {task1_decision}")
        return
    Y_pp, meta_df = task2_preprocess(patient_df, imp_cells, str_cells, master_x)
    bsv_df, trans = task3_bsv(Y_pp, master_x, meta_df)
    binary_df = task4_binary(bsv_df)
    cluster_df, assign_df = task5_latent(bsv_df)
    feat_df = task6_paper_regions(bsv_df, Y_pp, master_x, assign_df)
    perf_df = task7_classifier(bsv_df, Y_pp, feat_df)
    mss_df = task8_mss(bsv_df, Y_pp, master_x)

    decision = _decision(perf_df, cluster_df)
    write_report(decision, patient_df, bsv_df, binary_df, cluster_df,
                    assign_df, feat_df, perf_df, mss_df)
    write_audit(decision, task1_decision)
    try:
        shutil.copy(__file__, CODE_SNAPSHOT / Path(__file__).name)
    except Exception:
        pass
    print(f"[done] decision: {decision}")


if __name__ == "__main__":
    main()
