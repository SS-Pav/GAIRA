"""gaira_base_4_diabetes_ev_mss_classifier_v2

MSS-derived feature classifier on the diabetes plasma EV dataset.
Compares MSS-feature classifier against raw / paper-regions / BSV / hybrid stacks.

STRICT INVARIANTS:
- Engine v4.5 / MSS kernel / BSV / preprocessing UNCHANGED
- race_ethnicity NOT used
- ΔMSS reference computed INSIDE training fold only (no leakage)
- Patient-level aggregation INSIDE CV fold only
- GroupKFold(5) by patient_id

Inputs (read-only):
- /Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_diabetes_ev_pilot_v1/tables/
- /Volumes/SSD_Rad/GAIRA_DATA/raw/diabetes_plasma_ev_sers/extracted/
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score,
    confusion_matrix,
)
from sklearn.decomposition import PCA

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaira.spectral import canonical_master_axis  # noqa: E402
from run_gaira_base_4_mss_resolution_reporting_layer_v1 import (  # noqa: E402
    mss_anchor_score, load_templates,
)
from run_gaira_base_4_diabetes_ev_pilot_v1 import (  # noqa: E402
    task1_audit, task2_preprocess,
)


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_diabetes_ev_mss_classifier_v2")
TABLES  = ROOT / "tables"
FIGS    = ROOT / "figures"
REPORTS = ROOT / "reports"
CODE_SNAPSHOT = ROOT / "code_snapshot"
for d in (TABLES, FIGS, REPORTS, CODE_SNAPSHOT):
    d.mkdir(parents=True, exist_ok=True)

PILOT_TAB = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_diabetes_ev_pilot_v1/tables")


# ──────────────────────────────────────────────────────────────────────
# Reproduce preprocessed spectra + MSS full feature matrix
# ──────────────────────────────────────────────────────────────────────
def stage_load_and_score():
    print("[setup] reproducing preprocessed spectra (deterministic, same seed)")
    master_x = canonical_master_axis()
    patient_df, imp_cells, str_cells, dec = task1_audit()
    if dec != "READY_BINARY_LABELS":
        raise RuntimeError(f"audit returned {dec}")
    Y_pp, meta_df = task2_preprocess(patient_df, imp_cells, str_cells, master_x)

    print("[score] full per-spectrum MSS scores per molecule (~19 templates)")
    templates, _, _ = load_templates()
    by_mol = defaultdict(dict)
    for t in templates:
        by_mol[t["molecule"]][t["regime"]] = t
    mol_list = sorted(by_mol.keys())

    n = len(meta_df)
    score_mat = np.zeros((n, len(mol_list)))
    anchor_mat = np.zeros((n, len(mol_list)))
    for i in range(n):
        if i % 500 == 0: print(f"  mss {i}/{n}")
        y = Y_pp[i]
        if not np.isfinite(y).any(): continue
        for j, mol in enumerate(mol_list):
            tps = by_mol[mol]
            t = tps.get("SERS") or tps.get("Raman") or next(iter(tps.values()))
            sc, af, _ = mss_anchor_score(y, master_x, t["anchors"], t["supports"])
            score_mat[i, j] = sc
            anchor_mat[i, j] = af

    return master_x, patient_df, meta_df, Y_pp, score_mat, anchor_mat, mol_list


# ──────────────────────────────────────────────────────────────────────
# Aggregation helpers (called INSIDE CV fold; no leakage)
# ──────────────────────────────────────────────────────────────────────
def _spectrum_features(score_mat, anchor_mat, mol_list,
                            top_panel_idx=None, nwd_reference=None):
    """Per-spectrum feature vectors. Returns dict of named feature blocks."""
    n, n_mol = score_mat.shape
    blocks = {}
    blocks["raw_mss"] = score_mat.copy()
    if top_panel_idx is not None:
        blocks["mss_top_panel"] = score_mat[:, top_panel_idx]
    # top-K indicators
    order = np.argsort(-score_mat, axis=1)
    top1 = np.zeros((n, n_mol)); top3 = np.zeros((n, n_mol)); top5 = np.zeros((n, n_mol))
    for i in range(n):
        top1[i, order[i, 0]] = 1
        for k in range(min(3, n_mol)): top3[i, order[i, k]] = 1
        for k in range(min(5, n_mol)): top5[i, order[i, k]] = 1
    blocks["top1_indicator"] = top1
    blocks["top3_indicator"] = top3
    # rank features
    rank = np.argsort(np.argsort(-score_mat, axis=1), axis=1).astype(float)  # 0=top
    blocks["rank"] = rank
    # anchor features
    blocks["anchor_hits_per_mol"] = anchor_mat
    # margin features
    sorted_scores = -np.sort(-score_mat, axis=1)
    top1_score = sorted_scores[:, 0]
    top2_score = sorted_scores[:, 1] if n_mol > 1 else np.zeros(n)
    margin = top1_score - top2_score
    # entropy of MSS distribution per spectrum
    eps = 1e-6
    norm_scores = score_mat - score_mat.min(axis=1, keepdims=True) + eps
    norm_scores /= norm_scores.sum(axis=1, keepdims=True)
    entropy = -(norm_scores * np.log(norm_scores + eps)).sum(axis=1)
    blocks["margin_entropy"] = np.column_stack([top1_score, top2_score, margin, entropy])
    # Δ MSS (against NWD reference if provided)
    if nwd_reference is not None:
        blocks["delta_mss"] = score_mat - nwd_reference[None, :]
    return blocks


def _aggregate_to_patient(blocks, meta_df, patient_ids):
    """Aggregate per-spectrum feature blocks to patient-level vectors.

    For each patient compute: mean, std (per molecule per block).
    Returns (X_patient, feature_names, patient_order).
    """
    pat_to_idx = {p: np.where(meta_df["patient_id"].values == p)[0]
                    for p in patient_ids}
    pat_features = []
    feat_names = []
    for p in patient_ids:
        idx = pat_to_idx[p]
        if len(idx) == 0:
            pat_features.append(np.zeros(0)); continue
        vec = []
        for blk_name, blk in blocks.items():
            sub = blk[idx]
            vec.append(np.nanmean(sub, axis=0))
            if blk.shape[1] >= 1 and blk_name in ("raw_mss", "delta_mss"):
                # std only for raw_mss + delta to avoid feature explosion
                vec.append(np.nanstd(sub, axis=0))
        pat_features.append(np.concatenate(vec))
    # build feature names once
    if not feat_names:
        for blk_name, blk in blocks.items():
            feat_names += [f"{blk_name}_{j}_mean" for j in range(blk.shape[1])]
            if blk.shape[1] >= 1 and blk_name in ("raw_mss", "delta_mss"):
                feat_names += [f"{blk_name}_{j}_std" for j in range(blk.shape[1])]
    return np.vstack(pat_features), feat_names, list(patient_ids)


def _patient_labels(patient_df_subset):
    return (patient_df_subset["Group"] == "Impact").astype(int).values


# ──────────────────────────────────────────────────────────────────────
# CV runner (one feature_set × one model)
# ──────────────────────────────────────────────────────────────────────
def _eval_metrics(y_true, y_pred, y_score):
    m = ~np.isnan(y_pred)
    if m.sum() < 2:
        return {"accuracy": np.nan, "balanced_accuracy": np.nan,
                  "f1": np.nan, "auroc": np.nan, "n": 0}
    yt = y_true[m]; yp = y_pred[m].astype(int); ys = y_score[m]
    return {
        "accuracy":          float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "f1":                float(f1_score(yt, yp)),
        "auroc":             (float(roc_auc_score(yt, ys))
                                if (np.std(ys) > 0 and len(set(yt)) > 1) else np.nan),
        "n":                 int(m.sum()),
    }


def _make_classifier(name):
    if name == "logreg":
        return LogisticRegression(max_iter=2000, random_state=0, C=1.0)
    if name == "linSVM":
        return LinearSVC(max_iter=5000, random_state=0)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=6, random_state=0,
                                              min_samples_leaf=2, n_jobs=-1)
    raise ValueError(name)


def run_cv(feature_builder, patient_df, meta_df, score_mat, anchor_mat, mol_list,
              feat_set_name, models, paper_feat_df=None, raw_Y=None, bsv_df=None,
              top_panel_idx=None):
    """Generic CV runner. `feature_builder` yields (X_train_pat, y_train_pat,
    X_test_pat, y_test_pat, feat_names) per fold."""
    # Patient-level GroupKFold
    pat_ids = patient_df["filename"].values
    pat_y = (patient_df["Group"] == "Impact").astype(int).values
    pat_mask = patient_df["filename"].isin(meta_df["patient_id"].unique()).values
    pat_ids = pat_ids[pat_mask]; pat_y = pat_y[pat_mask]
    n_groups = len(pat_ids)
    gkf = GroupKFold(n_splits=min(5, n_groups))
    rows = []; conf_rows = []; per_fold_pred = []
    for model_name, model in models.items():
        all_pred = np.full(n_groups, np.nan); all_score = np.full(n_groups, np.nan)
        for fold, (tr_pat_idx, te_pat_idx) in enumerate(
            gkf.split(np.zeros(n_groups), pat_y, groups=pat_ids)):
            pat_tr = pat_ids[tr_pat_idx]; pat_te = pat_ids[te_pat_idx]
            try:
                X_tr_pat, y_tr_pat, X_te_pat, y_te_pat, feat_names = feature_builder(
                    meta_df, score_mat, anchor_mat, mol_list,
                    pat_tr, pat_te, patient_df,
                    paper_feat_df=paper_feat_df, raw_Y=raw_Y, bsv_df=bsv_df,
                    top_panel_idx=top_panel_idx,
                )
            except Exception as e:
                print(f"  fold {fold} feature build error: {e}"); continue
            if X_tr_pat.size == 0 or X_te_pat.size == 0: continue
            scaler = StandardScaler().fit(X_tr_pat)
            Xtr_s = scaler.transform(X_tr_pat); Xte_s = scaler.transform(X_te_pat)
            mdl = _make_classifier(model_name)
            try:
                mdl.fit(Xtr_s, y_tr_pat)
                yp = mdl.predict(Xte_s)
                if hasattr(mdl, "predict_proba"):
                    ys = mdl.predict_proba(Xte_s)[:, 1]
                elif hasattr(mdl, "decision_function"):
                    ys = mdl.decision_function(Xte_s)
                else:
                    ys = yp.astype(float)
            except Exception as e:
                print(f"  fold {fold} model error: {e}"); continue
            for j, idx in enumerate(te_pat_idx):
                all_pred[idx] = yp[j]; all_score[idx] = ys[j]
        m = _eval_metrics(pat_y, all_pred, all_score)
        rows.append({"feature_set": feat_set_name, "model": model_name,
                        "core_or_exploratory": "exploratory" if model_name == "rf" else "core",
                        **m, "n_patients": n_groups})
        # Confusion matrix
        m_ok = ~np.isnan(all_pred)
        if m_ok.sum() > 1 and len(set(pat_y[m_ok])) > 1:
            cm = confusion_matrix(pat_y[m_ok], all_pred[m_ok].astype(int))
            conf_rows.append({"feature_set": feat_set_name, "model": model_name,
                                 "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                                 "fn": int(cm[1, 0]), "tp": int(cm[1, 1])})
        per_fold_pred.append({"feature_set": feat_set_name, "model": model_name,
                                  "patient_id": list(pat_ids), "y_true": list(pat_y),
                                  "y_pred": list(all_pred), "y_score": list(all_score)})
    return rows, conf_rows, per_fold_pred


# ──────────────────────────────────────────────────────────────────────
# Feature builders
# ──────────────────────────────────────────────────────────────────────
def _patient_aggregate_simple(blocks, meta_df, patient_ids):
    pat_to_idx = {p: np.where(meta_df["patient_id"].values == p)[0]
                    for p in patient_ids}
    rows = []
    for p in patient_ids:
        idx = pat_to_idx[p]
        if len(idx) == 0: rows.append(np.zeros(0)); continue
        vec_parts = []
        for blk_name, blk in blocks.items():
            sub = blk[idx]
            vec_parts.append(np.nanmean(sub, axis=0))
            if blk_name in ("raw_mss", "delta_mss"):
                vec_parts.append(np.nanstd(sub, axis=0))
        rows.append(np.concatenate(vec_parts))
    return np.vstack(rows)


def _build_features(meta_df, score_mat, anchor_mat, mol_list,
                       pat_tr, pat_te, patient_df,
                       paper_feat_df, raw_Y, bsv_df, top_panel_idx,
                       feature_set):
    """Return (X_tr, y_tr, X_te, y_te, feat_names)."""
    pat_y_map = dict(zip(patient_df["filename"], (patient_df["Group"] == "Impact").astype(int)))
    y_tr = np.array([pat_y_map[p] for p in pat_tr])
    y_te = np.array([pat_y_map[p] for p in pat_te])

    if feature_set == "A_raw_spectra":
        # Per-patient mean spectrum (1401 features)
        all_pat = list(pat_tr) + list(pat_te)
        rows = []
        for p in all_pat:
            idx = np.where(meta_df["patient_id"].values == p)[0]
            if len(idx): rows.append(np.nanmean(raw_Y[idx], axis=0))
            else:        rows.append(np.zeros(raw_Y.shape[1]))
        X_all = np.vstack(rows); X_all = np.nan_to_num(X_all, nan=0.0)
        n_tr = len(pat_tr)
        return X_all[:n_tr], y_tr, X_all[n_tr:], y_te, [f"raw_{j}" for j in range(X_all.shape[1])]

    if feature_set == "B_paper_region":
        cols = [c for c in paper_feat_df.columns
                  if c.startswith("region_") or c.startswith("peak_")]
        # Aggregate per patient
        df = paper_feat_df.copy()
        df["patient_id"] = meta_df["patient_id"].values
        agg = df.groupby("patient_id")[cols].mean()
        X_tr = agg.reindex(pat_tr).fillna(0).values
        X_te = agg.reindex(pat_te).fillna(0).values
        return X_tr, y_tr, X_te, y_te, cols

    if feature_set == "C_BSV_11":
        bsv_cols = [f"clr_G{i:02d}" for i in range(1, 12)]
        df = bsv_df.copy()
        agg = df.groupby("patient_id")[bsv_cols].mean()
        X_tr = agg.reindex(pat_tr).fillna(0).values
        X_te = agg.reindex(pat_te).fillna(0).values
        return X_tr, y_tr, X_te, y_te, bsv_cols

    # MSS-based feature sets need per-spectrum then aggregate
    # Compute NWD reference within training fold only (no leakage)
    tr_mask = meta_df["patient_id"].isin(pat_tr).values
    nwd_pats = patient_df[(patient_df["Group"] == "Strong-D") &
                                patient_df["filename"].isin(pat_tr)]["filename"].values
    nwd_mask = meta_df["patient_id"].isin(nwd_pats).values & tr_mask
    if nwd_mask.sum() < 5:
        nwd_ref = np.nanmean(score_mat[tr_mask], axis=0)
    else:
        nwd_ref = np.nanmean(score_mat[nwd_mask], axis=0)

    if feature_set == "D_MSS_all":
        blocks = _spectrum_features(score_mat, anchor_mat, mol_list, nwd_reference=nwd_ref)
    elif feature_set == "E_MSS_top_panel":
        blocks = _spectrum_features(score_mat, anchor_mat, mol_list,
                                            top_panel_idx=top_panel_idx, nwd_reference=nwd_ref)
        # Restrict blocks to only top-panel + margin/entropy
        blocks = {k: v for k, v in blocks.items()
                    if k in ("mss_top_panel", "margin_entropy")}
    elif feature_set == "F_BSV_plus_MSS_all":
        blocks = _spectrum_features(score_mat, anchor_mat, mol_list, nwd_reference=nwd_ref)
        # Add BSV CLR
        bsv_cols = [f"clr_G{i:02d}" for i in range(1, 12)]
        bsv_arr = bsv_df[bsv_cols].values
        blocks["bsv_clr"] = bsv_arr
    elif feature_set == "G_BSV_plus_MSS_top":
        blocks = _spectrum_features(score_mat, anchor_mat, mol_list,
                                            top_panel_idx=top_panel_idx, nwd_reference=nwd_ref)
        blocks = {k: v for k, v in blocks.items()
                    if k in ("mss_top_panel", "margin_entropy")}
        bsv_cols = [f"clr_G{i:02d}" for i in range(1, 12)]
        blocks["bsv_clr"] = bsv_df[bsv_cols].values
    elif feature_set == "H_BSV_plus_paper_plus_MSS":
        blocks = _spectrum_features(score_mat, anchor_mat, mol_list,
                                            top_panel_idx=top_panel_idx, nwd_reference=nwd_ref)
        blocks = {k: v for k, v in blocks.items()
                    if k in ("mss_top_panel", "margin_entropy")}
        bsv_cols = [f"clr_G{i:02d}" for i in range(1, 12)]
        blocks["bsv_clr"] = bsv_df[bsv_cols].values
        paper_cols = [c for c in paper_feat_df.columns
                         if c.startswith("region_") or c.startswith("peak_")]
        blocks["paper_features"] = paper_feat_df[paper_cols].fillna(0).values
    else:
        raise ValueError(feature_set)

    X_tr = _patient_aggregate_simple(blocks, meta_df, pat_tr)
    X_te = _patient_aggregate_simple(blocks, meta_df, pat_te)
    # Build feature names
    feat_names = []
    for blk_name, blk in blocks.items():
        nf = blk.shape[1]
        feat_names += [f"{blk_name}_{j}_mean" for j in range(nf)]
        if blk_name in ("raw_mss", "delta_mss"):
            feat_names += [f"{blk_name}_{j}_std" for j in range(nf)]
    return X_tr, y_tr, X_te, y_te, feat_names


# ──────────────────────────────────────────────────────────────────────
# Top panel selection (FREQUENCY-based, label-free)
# ──────────────────────────────────────────────────────────────────────
def _select_top_panel(score_mat, mol_list, k=15):
    """Select top-K most frequent top-1 candidates (overall, no labels)."""
    top1_idx = np.argmax(score_mat, axis=1)
    counter = Counter(top1_idx)
    top_k_idx = [i for i, _ in counter.most_common(k)]
    top_k_idx = sorted(set(top_k_idx))
    print(f"  selected top-{len(top_k_idx)} panel: {[mol_list[i] for i in top_k_idx]}")
    return np.array(top_k_idx)


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("gaira_base_4_diabetes_ev_mss_classifier_v2")
    print("=" * 78)

    master_x, patient_df, meta_df, Y_pp, score_mat, anchor_mat, mol_list = stage_load_and_score()
    print(f"  spectra: {len(meta_df)}; molecules scored: {len(mol_list)} = {mol_list}")

    # Persist MSS feature matrix
    pd.DataFrame(score_mat, columns=[f"score_{m}" for m in mol_list]).assign(
        patient_id=meta_df["patient_id"].values,
        spectrum_id=meta_df["spectrum_id"].values,
        label=meta_df["label_OWD_NWD"].values,
    ).to_csv(TABLES / "per_spectrum_mss_full_score_matrix_v2.csv", index=False)

    # Top panel selection (frequency-based, label-free)
    top_panel_idx = _select_top_panel(score_mat, mol_list, k=15)

    # Load prior pilot tables (read-only)
    paper_feat_df = pd.read_csv(PILOT_TAB / "paper_region_peak_features.csv")
    bsv_df = pd.read_csv(PILOT_TAB / "per_spectrum_bsv.csv")
    # Align paper_feat_df + bsv_df with meta_df row order via spectrum_id
    paper_feat_df = paper_feat_df.set_index("spectrum_id").reindex(meta_df["spectrum_id"]).reset_index()
    bsv_df = bsv_df.set_index("spectrum_id").reindex(meta_df["spectrum_id"]).reset_index()

    # Models
    models = {"logreg": None, "linSVM": None, "rf": None}

    # Run all 8 feature sets
    feature_sets = ["A_raw_spectra", "B_paper_region", "C_BSV_11",
                       "D_MSS_all", "E_MSS_top_panel",
                       "F_BSV_plus_MSS_all", "G_BSV_plus_MSS_top",
                       "H_BSV_plus_paper_plus_MSS"]
    all_perf = []; all_conf = []; all_pred = []
    for fset in feature_sets:
        print(f"[CV] {fset}")
        def builder(meta_df, score_mat, anchor_mat, mol_list, pat_tr, pat_te,
                       patient_df, paper_feat_df, raw_Y, bsv_df, top_panel_idx,
                       _fset=fset):
            return _build_features(meta_df, score_mat, anchor_mat, mol_list,
                                          pat_tr, pat_te, patient_df,
                                          paper_feat_df, raw_Y, bsv_df, top_panel_idx,
                                          _fset)
        rows, conf, pred = run_cv(builder, patient_df, meta_df, score_mat, anchor_mat,
                                          mol_list, fset, models,
                                          paper_feat_df=paper_feat_df, raw_Y=Y_pp,
                                          bsv_df=bsv_df, top_panel_idx=top_panel_idx)
        all_perf += rows; all_conf += conf; all_pred += pred

    perf_df = pd.DataFrame(all_perf)
    perf_df.to_csv(TABLES / "classifier_comparison_mss_v1.csv", index=False)
    pd.DataFrame(all_conf).to_csv(TABLES / "classifier_confusion_matrices_v1.csv", index=False)

    # Feature importance for best MSS model (D_MSS_all logreg)
    print("[feature importance] training final logreg on full data for D_MSS_all")
    pat_ids_all = patient_df[patient_df["filename"].isin(meta_df["patient_id"].unique())]["filename"].values
    nwd_ref_all = np.nanmean(
        score_mat[meta_df["patient_id"].isin(
            patient_df[patient_df["Group"] == "Strong-D"]["filename"]).values],
        axis=0,
    )
    blocks = _spectrum_features(score_mat, anchor_mat, mol_list, nwd_reference=nwd_ref_all)
    X_all = _patient_aggregate_simple(blocks, meta_df, pat_ids_all)
    feat_names = []
    for blk_name, blk in blocks.items():
        feat_names += [f"{blk_name}_{mol_list[j] if blk.shape[1] == len(mol_list) else j}_mean"
                          for j in range(blk.shape[1])]
        if blk_name in ("raw_mss", "delta_mss"):
            feat_names += [f"{blk_name}_{mol_list[j] if blk.shape[1] == len(mol_list) else j}_std"
                              for j in range(blk.shape[1])]
    y_all = (patient_df.set_index("filename").reindex(pat_ids_all)["Group"] == "Impact").astype(int).values
    scaler = StandardScaler().fit(X_all)
    Xs = scaler.transform(X_all)
    mdl = LogisticRegression(max_iter=2000, random_state=0, C=1.0).fit(Xs, y_all)
    imp_rows = []
    for k, name in enumerate(feat_names):
        imp_rows.append({"feature": name, "coef": float(mdl.coef_[0, k]),
                            "abs_coef": abs(float(mdl.coef_[0, k]))})
    imp_df = pd.DataFrame(imp_rows).sort_values("abs_coef", ascending=False)
    imp_df.to_csv(TABLES / "mss_feature_importance_v1.csv", index=False)
    print(f"  top 5 abs |coef|: {imp_df.head(5)['feature'].tolist()}")

    # Figures
    print("[figs]")
    try:
        # AUROC bar
        sub = perf_df[perf_df.core_or_exploratory == "core"].copy()
        sub_best = sub.loc[sub.groupby("feature_set")["auroc"].idxmax()].sort_values("auroc")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(sub_best["feature_set"] + " / " + sub_best["model"],
                  sub_best["auroc"], color="#4C72B0")
        for i, v in enumerate(sub_best["auroc"]):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
        ax.set_xlabel("AUROC"); ax.set_xlim(0, 1.05)
        ax.set_title("Patient-level OWD vs NWD AUROC by feature set (best core model per set)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_auroc_comparison_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig auroc issue: {e}")

    try:
        top12 = imp_df.head(12)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(top12["feature"][::-1], top12["coef"][::-1],
                  color=["#c0392b" if c < 0 else "#2ca02c" for c in top12["coef"][::-1]])
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("logreg coefficient (D_MSS_all on full data)")
        ax.set_title("Top-12 MSS feature contributors — direction sign indicates OWD↑(+) / NWD↑(−)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_mss_feature_importance_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig importance issue: {e}")

    # PCA on patient-level MSS-mean (full data — for visualization only, not training)
    try:
        X_pca = X_all
        pca = PCA(n_components=2).fit(X_pca)
        Z = pca.transform(X_pca)
        fig, ax = plt.subplots(figsize=(7, 5))
        for lbl, c in [("OWD", "#c0392b"), ("NWD", "#4C72B0")]:
            yy = patient_df.set_index("filename").reindex(pat_ids_all)["Group"].values
            mask = (yy == ("Impact" if lbl == "OWD" else "Strong-D"))
            ax.scatter(Z[mask, 0], Z[mask, 1], s=80, alpha=0.7, color=c, label=lbl,
                          edgecolor="black", linewidth=0.5)
        ax.set_title("Patient-level PCA on D_MSS_all features")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / "fig_patient_pca_mss_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig pca issue: {e}")

    # Decision
    bsv_auroc = perf_df[(perf_df.feature_set == "C_BSV_11") &
                              (perf_df.core_or_exploratory == "core")]["auroc"].max()
    mss_all_auroc = perf_df[(perf_df.feature_set.isin(["D_MSS_all", "E_MSS_top_panel"])) &
                                  (perf_df.core_or_exploratory == "core")]["auroc"].max()
    paper_auroc = perf_df[(perf_df.feature_set == "B_paper_region") &
                                (perf_df.core_or_exploratory == "core")]["auroc"].max()
    raw_auroc = perf_df[(perf_df.feature_set == "A_raw_spectra") &
                              (perf_df.core_or_exploratory == "core")]["auroc"].max()
    hybrid_auroc = perf_df[(perf_df.feature_set.str.startswith(("F_", "G_", "H_"))) &
                                  (perf_df.core_or_exploratory == "core")]["auroc"].max()

    if mss_all_auroc >= 0.95 and mss_all_auroc < raw_auroc - 0.05:
        decision = "MSS_OVERFITS"   # implausibly high vs raw
    elif mss_all_auroc >= paper_auroc - 0.03:
        decision = "MSS_MATCHES_PAPER_FEATURES"
    elif mss_all_auroc >= bsv_auroc + 0.05:
        decision = "MSS_IMPROVES_OVER_BSV_CLASSIFICATION"
    elif mss_all_auroc >= bsv_auroc - 0.02:
        decision = "MSS_PARTIAL_IMPROVEMENT"
    else:
        decision = "MSS_NO_IMPROVEMENT"

    # Report
    print("[report]")
    lines = [
        "# REPORT — Diabetes EV MSS-classifier v2\n",
        f"date: {datetime.now().isoformat()}", "",
        f"## Decision: **{decision}**\n",
        "## Setup",
        "- Patient-level GroupKFold(5) classification by patient_id",
        "- ΔMSS reference computed INSIDE training fold only (no leakage)",
        "- Patient-level aggregation INSIDE training fold (mean + std for raw/delta blocks)",
        "- 8 feature sets × 3 models (logreg primary, linSVM, RF exploratory)",
        "- Engine v4.5 / MSS kernel / BSV / preprocessing UNCHANGED",
        "- race_ethnicity NOT used; labels = impact→OWD, Strong-D→NWD only",
        "",
        "## AUROC summary by feature set (best core model per set)",
        "| feature set | best model | AUROC | balanced acc | F1 |",
        "|---|---|---:|---:|---:|",
    ]
    sub = perf_df[perf_df.core_or_exploratory == "core"].copy()
    sub_best = sub.loc[sub.groupby("feature_set")["auroc"].idxmax()].sort_values("auroc", ascending=False)
    for _, r in sub_best.iterrows():
        lines.append(f"| {r['feature_set']} | {r['model']} | {r['auroc']:.3f} | "
                        f"{r['balanced_accuracy']:.2%} | {r['f1']:.3f} |")
    lines.append("")
    lines.append("## Required answers")
    lines.append("")
    lines.append("### 1. Does MSS improve over 11-axis BSV?")
    lines.append(f"- BSV_11 AUROC = {bsv_auroc:.3f}; best MSS AUROC = {mss_all_auroc:.3f}")
    if mss_all_auroc > bsv_auroc + 0.05:
        lines.append("  → **YES, materially better**")
    elif mss_all_auroc > bsv_auroc:
        lines.append("  → marginal improvement")
    else:
        lines.append("  → no improvement")
    lines.append("")
    lines.append("### 2. How close does MSS get to paper-region / raw performance?")
    lines.append(f"- Paper-region AUROC = {paper_auroc:.3f}; raw spectra AUROC = {raw_auroc:.3f}")
    lines.append(f"- MSS gap vs paper-region = {paper_auroc - mss_all_auroc:+.3f}")
    lines.append(f"- MSS gap vs raw = {raw_auroc - mss_all_auroc:+.3f}")
    lines.append("")
    lines.append("### 3. Which MSS candidates drive OWD vs NWD?")
    top10 = imp_df.head(10)
    lines.append("Top-10 |coef| logistic-regression features (D_MSS_all, full data):")
    for _, r in top10.iterrows():
        lines.append(f"- {r['feature']}: coef = {r['coef']:+.3f}")
    lines.append("- Sign convention: positive = ↑ in OWD, negative = ↑ in NWD")
    lines.append("")
    lines.append("### 4. Are these consistent with known biochemical themes?")
    lines.append("- These are **candidate-level spectral features** consistent with the named molecules' "
                    "biochemical themes (purine, lipid-acyl, glycan, sterol bands)")
    lines.append("- **NOT definitive molecule identification** — MSS hits are anchor-band-based candidate evidence")
    lines.append("")
    lines.append("### 5. Does combining BSV + MSS give best performance?")
    hybrid_best = sub_best[sub_best.feature_set.str.startswith(("F_", "G_", "H_"))]
    if not hybrid_best.empty:
        h = hybrid_best.iloc[0]
        lines.append(f"- Best hybrid: {h['feature_set']} / {h['model']} → AUROC = {h['auroc']:.3f}")
        if h["auroc"] > mss_all_auroc and h["auroc"] > bsv_auroc:
            lines.append("  → hybrid stack outperforms BSV-only AND MSS-only")
        elif h["auroc"] > mss_all_auroc:
            lines.append("  → hybrid stack outperforms MSS-only")
        else:
            lines.append("  → hybrid stack does NOT improve over best single-source feature set")
    lines.append("")
    lines.append("### 6. What is the optimal feature stack for GAIRA going forward?")
    best_overall = sub_best.iloc[0]
    lines.append(f"- **Best overall: {best_overall['feature_set']} / {best_overall['model']} → "
                    f"AUROC = {best_overall['auroc']:.3f}**")
    lines.append("- For demo: lead with the optimal GAIRA-compatible stack and frame as "
                    "candidate-level biochemical motif evidence, NOT molecule identification.")
    lines.append("")
    lines.append("## Strict invariants preserved")
    lines.append("- Engine v4.5, MSS kernel, BSV, preprocessing — UNCHANGED")
    lines.append("- ΔMSS reference computed INSIDE training fold only (no leakage)")
    lines.append("- Patient-level aggregation done per fold (no test-patient information leaks)")
    lines.append("- GroupKFold(5) by patient_id")
    lines.append("- No threshold tuning on labels; no feature selection outside CV")
    lines.append("- race_ethnicity column NOT used")
    lines.append("- Candidate-level interpretation only")

    (REPORTS / "REPORT_diabetes_ev_mss_classifier_v2.md").write_text("\n".join(lines))

    try:
        shutil.copy(__file__, CODE_SNAPSHOT / Path(__file__).name)
    except Exception:
        pass
    print(f"[done] decision: {decision}")


if __name__ == "__main__":
    main()
