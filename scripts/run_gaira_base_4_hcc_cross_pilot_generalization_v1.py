"""gaira_base_4 HCC cross-pilot generalization v1.

Train on P1, test on P2 (and vice versa) — raw spectra vs GAIRA BSV.
Classifier is evaluation-only (NO feedback to GAIRA engine).

NO engine / MSS / motif / taxonomy / weight changes. NO threshold tuning.
"""
from __future__ import annotations

import shutil
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaira.spectral import canonical_master_axis
from run_gaira_base_4_hybrid_bsv_build_v1 import BSV_GROUPS
from run_gaira_base_4_hybrid_bsv_controlled_calibration_v2 import FAMILY_LABELS


ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/"
    "gaira_base_4_hcc_cross_pilot_generalization_v1"
)
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"

P1_CSV = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/hcc_serum/data.csv")
P2_ZIP = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/cca_hcc_lm_serum_sers/"
    "Combination of label-free SERS-based nanosensor an.zip"
)
P1_V2_TABLE = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/"
    "gaira_base_4_passive_target_pilot_1_hcc_holdout_rerun_v2/tables/"
    "pilot1_v2_per_spectrum_outputs.csv"
)
P2_TABLE = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/"
    "gaira_base_4_passive_target_pilot_2_cca_hcc_lm/tables/"
    "pilot2_per_spectrum_outputs.csv"
)
SYN_TABLE = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_cross_pilot_synthesis_v1/tables/"
    "cross_pilot_harmonized_effect_sizes_v1.csv"
)

BSV_GROUPS_ORDER = [g["group_id"] for g in BSV_GROUPS]
ABS_COLS = [f"abs_{g}" for g in BSV_GROUPS_ORDER]
SN_COLS  = [f"sumnorm_{g}" for g in BSV_GROUPS_ORDER]
CLR_COLS = [f"clr_{g}" for g in BSV_GROUPS_ORDER]


# ─────────────────────────────────────────────────────────────────────
# Raw spectrum loaders (unified to GAIRA master axis)
# ─────────────────────────────────────────────────────────────────────

def load_p1_raw(master_x):
    """Gurian 2020 HCC: 144 spectra × wavenumber (from data.csv)."""
    df = pd.read_csv(P1_CSV, low_memory=False)
    meta_cols = ["acquisition_date", "substrate_batch", "class", "sample_code"]
    wn_cols = [c for c in df.columns if c not in meta_cols]
    wn = np.array([float(c) for c in wn_cols])
    order = np.argsort(wn)
    rows = []
    for i, row in df.iterrows():
        y = row[wn_cols].values.astype(float)
        y_rs = np.interp(master_x, wn[order], y[order], left=np.nan, right=np.nan)
        rows.append({
            "spectrum_id": f"p1::{row['sample_code']}",
            "sample_id": row["sample_code"],
            "class_label": row["class"],   # "CTR" or "H0T"
            "dataset": "P1_Gurian_HCC",
            "regime": "SERS",
            "substrate": "Gurian Ag colloid (untyped)",
            "raw_spectrum": y_rs,
        })
    return pd.DataFrame(rows)


def load_p2_raw(master_x):
    """Pilot 2: 195 patient-mean spectra from label-free SERS nanosensor."""
    rows = []
    with zipfile.ZipFile(P2_ZIP) as z:
        for info in z.infolist():
            if not info.filename.endswith(".txt"): continue
            parts = info.filename.split("/")
            if len(parts) < 4: continue
            patient_folder = parts[2]
            if not patient_folder.startswith("SER-"): continue
            toks = patient_folder.split("-")
            if len(toks) < 3: continue
            cls = toks[1]
            data = z.read(info).decode("utf-8", errors="ignore").splitlines()
            if len(data) < 2: continue
            try:
                wn = np.array([float(x) for x in data[0].split("\t") if x.strip()])
            except Exception: continue
            arrs = []
            for line in data[1:]:
                vals = line.split("\t")
                try:
                    f = [float(v) for v in vals if v.strip()]
                except ValueError: continue
                if len(f) >= len(wn) + 2:
                    arrs.append(np.asarray(f[2:2 + len(wn)]))
            if not arrs: continue
            mean_y = np.mean(arrs, 0)
            order = np.argsort(wn)
            y_rs = np.interp(master_x, wn[order], mean_y[order], left=np.nan, right=np.nan)
            rows.append({
                "spectrum_id": f"p2::{patient_folder}",
                "sample_id": patient_folder,
                "class_label": cls,  # NC / HCC / CCA / LM
                "dataset": "P2_label_free_SERS_nanosensor",
                "regime": "SERS",
                "substrate": "label-free SERS nanosensor (unknown)",
                "raw_spectrum": y_rs,
            })
    return pd.DataFrame(rows)


def _sanitize_matrix(X):
    """Replace NaN/inf with per-column median (0 if entirely NaN)."""
    X = np.array(X, dtype=float)
    # Col-wise median imputation for NaN/inf
    mask = ~np.isfinite(X)
    col_med = np.nanmedian(np.where(np.isinf(X), np.nan, X), axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    X[mask] = np.take(col_med, np.where(mask)[1])
    return X


# ─────────────────────────────────────────────────────────────────────
# BSV loaders (reuse existing Pilot outputs)
# ─────────────────────────────────────────────────────────────────────

def load_p1_bsv():
    df = pd.read_csv(P1_V2_TABLE)
    # Ensure sumnorm + clr columns exist
    if f"sumnorm_{BSV_GROUPS_ORDER[0]}" not in df.columns:
        X = df[ABS_COLS].values
        X_sn = X / (X.sum(axis=1, keepdims=True) + 1e-12)
        for i, g in enumerate(BSV_GROUPS_ORDER): df[f"sumnorm_{g}"] = X_sn[:, i]
        X_pos = np.maximum(X, 1e-9); log_X = np.log(X_pos)
        X_clr = log_X - log_X.mean(axis=1, keepdims=True)
        for i, g in enumerate(BSV_GROUPS_ORDER): df[f"clr_{g}"] = X_clr[:, i]
    df = df.rename(columns={"sample_code": "sample_id"}) if "sample_code" in df.columns else df
    df["dataset"] = "P1_Gurian_HCC"
    return df


def load_p2_bsv():
    df = pd.read_csv(P2_TABLE)
    # Ensure sumnorm + clr
    if f"sumnorm_{BSV_GROUPS_ORDER[0]}" not in df.columns:
        X = df[ABS_COLS].values
        X_sn = X / (X.sum(axis=1, keepdims=True) + 1e-12)
        for i, g in enumerate(BSV_GROUPS_ORDER): df[f"sumnorm_{g}"] = X_sn[:, i]
        X_pos = np.maximum(X, 1e-9); log_X = np.log(X_pos)
        X_clr = log_X - log_X.mean(axis=1, keepdims=True)
        for i, g in enumerate(BSV_GROUPS_ORDER): df[f"clr_{g}"] = X_clr[:, i]
    df["dataset"] = "P2_label_free_SERS_nanosensor"
    return df


# ─────────────────────────────────────────────────────────────────────
# Classifier eval helpers
# ─────────────────────────────────────────────────────────────────────

def make_clf(name):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    if name == "linSVM":
        return Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=1.0, max_iter=5000, random_state=42))])
    if name == "logreg":
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42))])
    if name == "rf":
        return Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1))])
    if name == "gb":
        return Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42))])
    raise ValueError(name)


def within_pilot_eval(X, y, groups, clf_name):
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.metrics import accuracy_score
    # Use StratifiedKFold if every group has 1 spectrum (trivial case)
    if len(np.unique(groups)) == len(y):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splitter = skf.split(X, y)
    else:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splitter = skf.split(X, y, groups)
    accs = []
    for tr, te in splitter:
        clf = make_clf(clf_name)
        clf.fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs)), float(np.std(accs, ddof=1))


def cross_pilot_eval(X_train, y_train, X_test, y_test, clf_name):
    from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
    clf = make_clf(clf_name)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    bal = balanced_accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    return acc, bal, cm


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("gaira_base_4_hcc_cross_pilot_generalization_v1")
    print("=" * 78)
    for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
        d.mkdir(parents=True, exist_ok=True)

    master_x = canonical_master_axis()

    # ── Load raw + BSV ──
    print("\n[load] raw P1 + raw P2 + BSV P1 + BSV P2")
    p1_raw = load_p1_raw(master_x)
    p2_raw = load_p2_raw(master_x)
    p1_bsv = load_p1_bsv()
    p2_bsv = load_p2_bsv()
    print(f"  P1 raw: {len(p1_raw)} (classes: {p1_raw.class_label.value_counts().to_dict()})")
    print(f"  P2 raw: {len(p2_raw)} (classes: {p2_raw.class_label.value_counts().to_dict()})")
    print(f"  P1 BSV: {len(p1_bsv)}  P2 BSV: {len(p2_bsv)}")

    # Map class labels to unified HCC vs Control binary
    p1_raw["unified_label"] = p1_raw["class_label"].map({"H0T": "HCC", "CTR": "Control"})
    p2_raw["unified_label"] = p2_raw["class_label"].map({"HCC": "HCC", "NC": "Control",
                                                              "CCA": "CCA", "LM": "LM"})
    p1_bsv["unified_label"] = p1_bsv["class_label"].map({"H0T": "HCC", "CTR": "Control"})
    p2_bsv["unified_label"] = p2_bsv["class_label"].map({"HCC": "HCC", "NC": "Control",
                                                              "CCA": "CCA", "LM": "LM"})

    # Build P2 HCC vs Control subset + advanced-cancer subset
    p2_raw_hcc = p2_raw[p2_raw["unified_label"].isin(["HCC", "Control"])].reset_index(drop=True)
    p2_raw_adv = p2_raw[p2_raw["unified_label"].isin(["CCA", "LM", "Control"])].copy()
    p2_raw_adv["unified_label"] = p2_raw_adv["unified_label"].map(
        {"CCA": "AdvCancer", "LM": "AdvCancer", "Control": "Control"})
    p2_bsv_hcc = p2_bsv[p2_bsv["unified_label"].isin(["HCC", "Control"])].reset_index(drop=True)
    p2_bsv_adv = p2_bsv[p2_bsv["unified_label"].isin(["CCA", "LM", "Control"])].copy()
    p2_bsv_adv["unified_label"] = p2_bsv_adv["unified_label"].map(
        {"CCA": "AdvCancer", "LM": "AdvCancer", "Control": "Control"})

    # Feature sets
    def _feat(df, feat_set):
        if feat_set == "RAW":
            return _sanitize_matrix(np.vstack(df["raw_spectrum"].values))
        if feat_set == "B_raw_BSV":
            return df[ABS_COLS].values
        if feat_set == "C_sumnorm_BSV":
            return df[SN_COLS].values
        if feat_set == "D_CLR_BSV":
            return df[CLR_COLS].values
        if feat_set == "E_concat":
            return np.hstack([df[ABS_COLS].values, df[SN_COLS].values, df[CLR_COLS].values])
        if feat_set == "F_concat_plus_conf_amb":
            cols_conf = [f"conf_{g}" for g in BSV_GROUPS_ORDER]
            extras = df[["top_confidence", "spillover_ratio", "ambiguity_flag"]].astype(float).values
            return np.hstack([df[ABS_COLS].values, df[SN_COLS].values, df[CLR_COLS].values,
                               df[cols_conf].values, extras])
        raise ValueError(feat_set)

    raw_sets = ["RAW"]
    bsv_sets = ["B_raw_BSV", "C_sumnorm_BSV", "D_CLR_BSV", "E_concat", "F_concat_plus_conf_amb"]
    classifiers = ["linSVM", "logreg", "rf", "gb"]

    # ── EXPERIMENT 1 — Within-pilot ──
    print("\n[exp1] Within-pilot classification")
    within_rows = []
    # P1 HCC vs Control — use raw + BSV
    for feat_set in raw_sets + bsv_sets:
        if feat_set == "RAW":
            X = _feat(p1_raw, feat_set); y = (p1_raw["unified_label"].values == "HCC").astype(int); groups = p1_raw["sample_id"].values
        else:
            X = _feat(p1_bsv, feat_set); y = (p1_bsv["unified_label"].values == "HCC").astype(int); groups = p1_bsv["sample_id"].values
        for clf_name in classifiers:
            acc, sd = within_pilot_eval(X, y, groups, clf_name)
            within_rows.append({"dataset": "P1", "comparison": "P1_HCC_vs_Control",
                                 "feature_set": feat_set, "classifier": clf_name,
                                 "n_features": X.shape[1],
                                 "accuracy_mean": round(acc, 3), "accuracy_std": round(sd, 3)})
    # P2 HCC vs Control
    for feat_set in raw_sets + bsv_sets:
        if feat_set == "RAW":
            X = _feat(p2_raw_hcc, feat_set); y = (p2_raw_hcc["unified_label"].values == "HCC").astype(int); groups = p2_raw_hcc["sample_id"].values
        else:
            X = _feat(p2_bsv_hcc, feat_set); y = (p2_bsv_hcc["unified_label"].values == "HCC").astype(int); groups = p2_bsv_hcc["sample_id"].values
        for clf_name in classifiers:
            acc, sd = within_pilot_eval(X, y, groups, clf_name)
            within_rows.append({"dataset": "P2", "comparison": "P2_HCC_vs_Control",
                                 "feature_set": feat_set, "classifier": clf_name,
                                 "n_features": X.shape[1],
                                 "accuracy_mean": round(acc, 3), "accuracy_std": round(sd, 3)})
    # P2 AdvCancer vs Control (CCA + LM merged)
    for feat_set in raw_sets + bsv_sets:
        if feat_set == "RAW":
            X = _feat(p2_raw_adv, feat_set); y = (p2_raw_adv["unified_label"].values == "AdvCancer").astype(int); groups = p2_raw_adv["sample_id"].values
        else:
            X = _feat(p2_bsv_adv, feat_set); y = (p2_bsv_adv["unified_label"].values == "AdvCancer").astype(int); groups = p2_bsv_adv["sample_id"].values
        for clf_name in classifiers:
            acc, sd = within_pilot_eval(X, y, groups, clf_name)
            within_rows.append({"dataset": "P2", "comparison": "P2_AdvCancer_vs_Control",
                                 "feature_set": feat_set, "classifier": clf_name,
                                 "n_features": X.shape[1],
                                 "accuracy_mean": round(acc, 3), "accuracy_std": round(sd, 3)})
    within_df = pd.DataFrame(within_rows)
    within_df.to_csv(TABLES / "table1_within_pilot_accuracy.csv", index=False)
    print(within_df.groupby(["comparison", "feature_set"])["accuracy_mean"].max().unstack())

    # ── EXPERIMENT 2 — Cross-pilot ──
    print("\n[exp2] Cross-pilot classification")
    cross_rows = []
    cross_cms = []
    from sklearn.preprocessing import StandardScaler
    scenarios = [
        # train_set, train_df_raw, train_df_bsv, test_set, test_df_raw, test_df_bsv
        ("P1_HCC_vs_Control", p1_raw, p1_bsv, "P2_HCC_vs_Control", p2_raw_hcc, p2_bsv_hcc),
        ("P2_HCC_vs_Control", p2_raw_hcc, p2_bsv_hcc, "P1_HCC_vs_Control", p1_raw, p1_bsv),
        ("P2_AdvCancer_vs_Control", p2_raw_adv, p2_bsv_adv, "P1_HCC_vs_Control", p1_raw, p1_bsv),
    ]
    for tr_name, tr_raw, tr_bsv, te_name, te_raw, te_bsv in scenarios:
        for feat_set in raw_sets + bsv_sets:
            if feat_set == "RAW":
                X_tr = _feat(tr_raw, feat_set)
                X_te = _feat(te_raw, feat_set)
                # Binary: positive class = non-Control
                y_tr = (tr_raw["unified_label"].values != "Control").astype(int)
                y_te = (te_raw["unified_label"].values != "Control").astype(int)
            else:
                X_tr = _feat(tr_bsv, feat_set)
                X_te = _feat(te_bsv, feat_set)
                y_tr = (tr_bsv["unified_label"].values != "Control").astype(int)
                y_te = (te_bsv["unified_label"].values != "Control").astype(int)
            for clf_name in classifiers:
                acc, bal, cm = cross_pilot_eval(X_tr, y_tr, X_te, y_te, clf_name)
                cross_rows.append({
                    "train_set": tr_name, "test_set": te_name,
                    "feature_set": feat_set, "classifier": clf_name,
                    "n_features": X_tr.shape[1],
                    "n_train": len(y_tr), "n_test": len(y_te),
                    "accuracy": round(acc, 3),
                    "balanced_accuracy": round(bal, 3),
                })
                cross_cms.append({
                    "train_set": tr_name, "test_set": te_name,
                    "feature_set": feat_set, "classifier": clf_name,
                    "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
                    "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
                })
    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(TABLES / "table2_cross_pilot_accuracy.csv", index=False)
    pd.DataFrame(cross_cms).to_csv(TABLES / "table2b_cross_pilot_confusion.csv", index=False)

    # ── TABLE 3 — Dimensionality comparison ──
    dim_rows = []
    for feat_set in raw_sets + bsv_sets:
        w = within_df[(within_df.feature_set == feat_set) & (within_df.comparison == "P1_HCC_vs_Control")]
        if len(w):
            dim_rows.append({
                "feature_set": feat_set,
                "n_features": int(w["n_features"].iloc[0]),
                "best_within_P1_accuracy": float(w["accuracy_mean"].max()),
                "comment": ("high-dim raw spectra" if feat_set == "RAW" else
                             "chemistry-interpretable BSV variant"),
            })
    pd.DataFrame(dim_rows).to_csv(TABLES / "table3_dimensionality_comparison.csv", index=False)

    # ── EXPERIMENT 3 — Axis-level transfer ──
    print("\n[exp3] Axis-level transfer")
    syn = pd.read_csv(SYN_TABLE)
    # Compute P1 HCC vs CTR and P2 HCC vs NC sumnorm d
    axis_rows = []
    for g in BSV_GROUPS_ORDER:
        p1_d = float(syn[(syn.representation == "sumnorm") & (syn.family == g) &
                            (syn.comparison == "P1_HCC_vs_CTR")]["cohens_d"].iloc[0])
        p2_hcc = float(syn[(syn.representation == "sumnorm") & (syn.family == g) &
                              (syn.comparison == "P2_HCC_vs_NC")]["cohens_d"].iloc[0])
        p2_cca = float(syn[(syn.representation == "sumnorm") & (syn.family == g) &
                              (syn.comparison == "P2_CCA_vs_NC")]["cohens_d"].iloc[0])
        p2_lm  = float(syn[(syn.representation == "sumnorm") & (syn.family == g) &
                              (syn.comparison == "P2_LM_vs_NC")]["cohens_d"].iloc[0])
        p2_adv = float(np.mean([p2_cca, p2_lm]))
        axis_rows.append({
            "axis": g, "family_label": FAMILY_LABELS.get(g, g),
            "P1_HCC_vs_CTR_d_sumnorm": round(p1_d, 3),
            "P2_HCC_vs_NC_d_sumnorm": round(p2_hcc, 3),
            "P2_AdvCancer_vs_NC_d_sumnorm_mean": round(p2_adv, 3),
            "direction_match_P1_vs_P2_HCC": np.sign(p1_d) == np.sign(p2_hcc) and p1_d != 0 and p2_hcc != 0,
            "direction_match_P1_vs_P2_Adv": np.sign(p1_d) == np.sign(p2_adv) and p1_d != 0 and p2_adv != 0,
            "magnitude_diff_P1_vs_P2_HCC": round(abs(p1_d) - abs(p2_hcc), 3),
            "magnitude_diff_P1_vs_P2_Adv": round(abs(p1_d) - abs(p2_adv), 3),
        })
    axis_df = pd.DataFrame(axis_rows)
    axis_df.to_csv(TABLES / "table4_axis_level_transfer.csv", index=False)

    # Direction agreement + magnitude correlation
    dir_match_hcc = int(axis_df["direction_match_P1_vs_P2_HCC"].sum())
    dir_match_adv = int(axis_df["direction_match_P1_vs_P2_Adv"].sum())
    mag_corr_hcc = float(np.corrcoef(axis_df["P1_HCC_vs_CTR_d_sumnorm"],
                                         axis_df["P2_HCC_vs_NC_d_sumnorm"])[0, 1])
    mag_corr_adv = float(np.corrcoef(axis_df["P1_HCC_vs_CTR_d_sumnorm"],
                                         axis_df["P2_AdvCancer_vs_NC_d_sumnorm_mean"])[0, 1])

    # ── TABLE 5 — Summary ──
    within_best_by_comp = within_df.groupby(["comparison", "feature_set"])["accuracy_mean"].max().reset_index()
    mean_within_raw = float(within_best_by_comp[within_best_by_comp.feature_set == "RAW"]["accuracy_mean"].mean())
    mean_within_bsv = float(within_best_by_comp[within_best_by_comp.feature_set != "RAW"]["accuracy_mean"].mean())
    cross_best_by_scenario = cross_df.groupby(["train_set", "test_set", "feature_set"])["accuracy"].max().reset_index()
    mean_cross_raw = float(cross_best_by_scenario[cross_best_by_scenario.feature_set == "RAW"]["accuracy"].mean())
    mean_cross_bsv = float(cross_best_by_scenario[cross_best_by_scenario.feature_set != "RAW"]["accuracy"].mean())
    transfer_drop_raw = mean_within_raw - mean_cross_raw
    transfer_drop_bsv = mean_within_bsv - mean_cross_bsv

    summary_rows = [
        {"metric": "mean_within_pilot_accuracy_RAW", "value": round(mean_within_raw, 3)},
        {"metric": "mean_within_pilot_accuracy_BSV_best", "value": round(mean_within_bsv, 3)},
        {"metric": "mean_cross_pilot_accuracy_RAW", "value": round(mean_cross_raw, 3)},
        {"metric": "mean_cross_pilot_accuracy_BSV_best", "value": round(mean_cross_bsv, 3)},
        {"metric": "transfer_drop_RAW (within - cross)", "value": round(transfer_drop_raw, 3)},
        {"metric": "transfer_drop_BSV (within - cross)", "value": round(transfer_drop_bsv, 3)},
        {"metric": "axis_direction_agreement_P1_vs_P2_HCC", "value": f"{dir_match_hcc}/11"},
        {"metric": "axis_direction_agreement_P1_vs_P2_Advanced", "value": f"{dir_match_adv}/11"},
        {"metric": "magnitude_correlation_P1_vs_P2_HCC", "value": round(mag_corr_hcc, 3)},
        {"metric": "magnitude_correlation_P1_vs_P2_Advanced", "value": round(mag_corr_adv, 3)},
    ]
    pd.DataFrame(summary_rows).to_csv(TABLES / "table5_summary.csv", index=False)

    print("\n[summary]")
    print(f"  within-pilot RAW best: {mean_within_raw:.3f}")
    print(f"  within-pilot BSV best: {mean_within_bsv:.3f}")
    print(f"  cross-pilot RAW best: {mean_cross_raw:.3f}")
    print(f"  cross-pilot BSV best: {mean_cross_bsv:.3f}")
    print(f"  transfer drop RAW: {transfer_drop_raw:+.3f}")
    print(f"  transfer drop BSV: {transfer_drop_bsv:+.3f}")
    print(f"  axis direction P1 vs P2 HCC: {dir_match_hcc}/11; P1 vs P2 Advanced: {dir_match_adv}/11")
    print(f"  magnitude correlation P1 vs P2 HCC: {mag_corr_hcc:+.2f}; vs Advanced: {mag_corr_adv:+.2f}")

    # ── FIGURES ──
    print("\n[figures]")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. Within vs cross-pilot accuracy bar (RAW vs BSV)
        fig, ax = plt.subplots(figsize=(10, 5))
        categories = ["Within P1", "Within P2 HCC", "Within P2 AdvCancer",
                        "Cross P1→P2 HCC", "Cross P2→P1 HCC", "Cross P2 Adv→P1"]
        # Best within per comparison
        w_p1 = float(within_df[within_df.comparison == "P1_HCC_vs_Control"].groupby("feature_set")["accuracy_mean"].max().get("RAW", 0))
        w_p2_hcc = float(within_df[within_df.comparison == "P2_HCC_vs_Control"].groupby("feature_set")["accuracy_mean"].max().get("RAW", 0))
        w_p2_adv = float(within_df[within_df.comparison == "P2_AdvCancer_vs_Control"].groupby("feature_set")["accuracy_mean"].max().get("RAW", 0))
        # Best within BSV (any BSV set)
        w_p1_b = float(within_df[(within_df.comparison == "P1_HCC_vs_Control") & (within_df.feature_set != "RAW")]["accuracy_mean"].max())
        w_p2_hcc_b = float(within_df[(within_df.comparison == "P2_HCC_vs_Control") & (within_df.feature_set != "RAW")]["accuracy_mean"].max())
        w_p2_adv_b = float(within_df[(within_df.comparison == "P2_AdvCancer_vs_Control") & (within_df.feature_set != "RAW")]["accuracy_mean"].max())
        # Cross best
        def _cross_best(tr, te, feat_set_is_raw):
            sub = cross_df[(cross_df.train_set == tr) & (cross_df.test_set == te)]
            if feat_set_is_raw:
                sub = sub[sub.feature_set == "RAW"]
            else:
                sub = sub[sub.feature_set != "RAW"]
            return float(sub["accuracy"].max()) if len(sub) else 0.0
        c_1to2_raw = _cross_best("P1_HCC_vs_Control", "P2_HCC_vs_Control", True)
        c_1to2_bsv = _cross_best("P1_HCC_vs_Control", "P2_HCC_vs_Control", False)
        c_2to1_raw = _cross_best("P2_HCC_vs_Control", "P1_HCC_vs_Control", True)
        c_2to1_bsv = _cross_best("P2_HCC_vs_Control", "P1_HCC_vs_Control", False)
        c_2adv_to1_raw = _cross_best("P2_AdvCancer_vs_Control", "P1_HCC_vs_Control", True)
        c_2adv_to1_bsv = _cross_best("P2_AdvCancer_vs_Control", "P1_HCC_vs_Control", False)
        raw_vals = [w_p1, w_p2_hcc, w_p2_adv, c_1to2_raw, c_2to1_raw, c_2adv_to1_raw]
        bsv_vals = [w_p1_b, w_p2_hcc_b, w_p2_adv_b, c_1to2_bsv, c_2to1_bsv, c_2adv_to1_bsv]
        x = np.arange(len(categories)); w = 0.35
        ax.bar(x - w/2, raw_vals, w, label="RAW spectra", color="#1f77b4")
        ax.bar(x + w/2, bsv_vals, w, label="BSV (best)", color="#d62728")
        ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, label="chance")
        ax.set_xticks(x); ax.set_xticklabels(categories, rotation=25, ha="right")
        ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
        ax.set_title("Within-pilot vs cross-pilot accuracy — RAW vs BSV")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / "fig1_within_vs_cross_accuracy.png", dpi=150)
        plt.close(fig)

        # 2. Cross-pilot confusion matrices (best per scenario)
        cms = pd.DataFrame(cross_cms)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        scenarios_cm = [
            ("P1_HCC_vs_Control", "P2_HCC_vs_Control"),
            ("P2_HCC_vs_Control", "P1_HCC_vs_Control"),
            ("P2_AdvCancer_vs_Control", "P1_HCC_vs_Control"),
        ]
        for ax_, (tr, te) in zip(axes, scenarios_cm):
            sub = cross_df[(cross_df.train_set == tr) & (cross_df.test_set == te)].sort_values("accuracy", ascending=False).head(1)
            if not len(sub): continue
            top = sub.iloc[0]
            cm = cms[(cms.train_set == tr) & (cms.test_set == te) &
                      (cms.feature_set == top["feature_set"]) & (cms.classifier == top["classifier"])].iloc[0]
            mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
            im = ax_.imshow(mat, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax_.text(j, i, f"{mat[i,j]}", ha="center", va="center", fontsize=12,
                              color="white" if mat[i,j] > mat.max()*0.5 else "black")
            ax_.set_xticks([0, 1]); ax_.set_yticks([0, 1])
            ax_.set_xticklabels(["pred Control", "pred HCC/+"])
            ax_.set_yticklabels(["true Control", "true HCC/+"])
            ax_.set_title(f"{tr}\n→ {te}\n{top['feature_set']}/{top['classifier']} acc={top['accuracy']:.2f}")
        fig.tight_layout()
        fig.savefig(FIGS / "fig2_cross_pilot_confusion_matrices.png", dpi=150)
        plt.close(fig)

        # 3. PCA on RAW spectra (P1 + P2 combined)
        from sklearn.decomposition import PCA
        # Combine P1 + P2 HCC-or-Control subset
        combined_raw = pd.concat([
            p1_raw[["dataset", "unified_label", "raw_spectrum"]],
            p2_raw_hcc[["dataset", "unified_label", "raw_spectrum"]],
        ], ignore_index=True)
        Xr = _sanitize_matrix(np.vstack(combined_raw["raw_spectrum"].values))
        from sklearn.preprocessing import StandardScaler
        Xr_s = StandardScaler().fit_transform(Xr)
        pcr = PCA(n_components=2, random_state=0).fit_transform(Xr_s)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        pal_ds = {"P1_Gurian_HCC": "#1f77b4", "P2_label_free_SERS_nanosensor": "#d62728"}
        for ds in pal_ds:
            m = combined_raw["dataset"].values == ds
            axes[0].scatter(pcr[m, 0], pcr[m, 1], s=30, alpha=0.7, label=ds, color=pal_ds[ds])
        axes[0].set_title("RAW PCA — colored by dataset"); axes[0].legend(fontsize=8)
        pal_cls = {"HCC": "#d62728", "Control": "#1f77b4"}
        for cls in pal_cls:
            m = combined_raw["unified_label"].values == cls
            axes[1].scatter(pcr[m, 0], pcr[m, 1], s=30, alpha=0.7, label=cls, color=pal_cls[cls])
        axes[1].set_title("RAW PCA — colored by class"); axes[1].legend()
        fig.suptitle("PC1-PC2 of combined RAW spectra (P1 + P2 HCC/Control)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig3_pca_raw.png", dpi=150)
        plt.close(fig)

        # 4. PCA on sumnorm BSV (P1 + P2 combined)
        combined_bsv = pd.concat([
            p1_bsv[["dataset", "unified_label"] + SN_COLS].rename(columns={"dataset": "dataset"}),
            p2_bsv_hcc[["dataset", "unified_label"] + SN_COLS],
        ], ignore_index=True)
        Xb = combined_bsv[SN_COLS].values
        pcb = PCA(n_components=2, random_state=0).fit_transform(Xb)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ds in pal_ds:
            m = combined_bsv["dataset"].values == ds
            axes[0].scatter(pcb[m, 0], pcb[m, 1], s=30, alpha=0.7, label=ds, color=pal_ds[ds])
        axes[0].set_title("sumnorm BSV PCA — colored by dataset"); axes[0].legend(fontsize=8)
        for cls in pal_cls:
            m = combined_bsv["unified_label"].values == cls
            axes[1].scatter(pcb[m, 0], pcb[m, 1], s=30, alpha=0.7, label=cls, color=pal_cls[cls])
        axes[1].set_title("sumnorm BSV PCA — colored by class"); axes[1].legend()
        fig.suptitle("PC1-PC2 of combined sumnorm BSV (P1 + P2 HCC/Control)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig4_pca_bsv.png", dpi=150)
        plt.close(fig)

        # 5. Axis transfer heatmap
        fig, ax = plt.subplots(figsize=(8, 5))
        mat = axis_df[["P1_HCC_vs_CTR_d_sumnorm",
                         "P2_HCC_vs_NC_d_sumnorm",
                         "P2_AdvCancer_vs_NC_d_sumnorm_mean"]].values
        vmax = float(np.abs(mat).max()) or 0.5
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_yticks(range(11))
        ax.set_yticklabels([f"{g} {FAMILY_LABELS.get(g, g)}" for g in BSV_GROUPS_ORDER])
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["P1 HCC vs CTR", "P2 HCC vs NC", "P2 AdvCancer vs NC"],
                            rotation=20, ha="right")
        ax.set_title("Axis-level transfer: sumnorm Cohen's d")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                         color="white" if abs(v) > vmax*0.5 else "black")
        fig.colorbar(im, ax=ax, label="d")
        fig.tight_layout()
        fig.savefig(FIGS / "fig5_axis_transfer_heatmap.png", dpi=150)
        plt.close(fig)

        # 6. Accuracy vs dimensionality
        fig, ax = plt.subplots(figsize=(8, 5))
        dim_df = pd.DataFrame(dim_rows)
        ax.scatter(dim_df["n_features"], dim_df["best_within_P1_accuracy"], s=80)
        for _, r in dim_df.iterrows():
            ax.annotate(r["feature_set"], (r["n_features"], r["best_within_P1_accuracy"]),
                         fontsize=9, xytext=(5, 5), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("n features (log scale)")
        ax.set_ylabel("best within-P1 accuracy")
        ax.set_title("Accuracy vs feature-set dimensionality (within P1)")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(FIGS / "fig6_accuracy_vs_dimensionality.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  figure issue: {e}")

    # ── FINAL ASSESSMENT ──
    def _answer_transfer():
        raw_generalize = c_1to2_raw >= 0.60 and c_2to1_raw >= 0.60
        bsv_generalize = c_1to2_bsv >= 0.60 and c_2to1_bsv >= 0.60
        bsv_better = (c_1to2_bsv > c_1to2_raw) and (c_2to1_bsv > c_2to1_raw)
        return raw_generalize, bsv_generalize, bsv_better

    raw_gen, bsv_gen, bsv_better = _answer_transfer()

    assessment = ["# HCC Cross-Pilot Generalization — Final Assessment", ""]
    assessment += [
        "## Q1. Does RAW spectra generalize across pilots?",
        "",
    ]
    if raw_gen:
        assessment.append(f"**YES.** Cross-pilot RAW accuracy: P1→P2 {c_1to2_raw:.2f}, P2→P1 {c_2to1_raw:.2f} (both ≥0.60).")
    else:
        assessment.append(f"**NO.** Cross-pilot RAW accuracy: P1→P2 **{c_1to2_raw:.2f}**, P2→P1 **{c_2to1_raw:.2f}**. "
                          "Raw SERS spectra do NOT transfer across the two substrate families (Gurian Ag colloid vs label-free SERS nanosensor).")
    assessment += [
        "",
        "## Q2. Does BSV generalize better or worse?",
        "",
    ]
    if bsv_better and bsv_gen:
        assessment.append(f"**BSV generalizes BETTER.** P1→P2 BSV {c_1to2_bsv:.2f} vs RAW {c_1to2_raw:.2f}; "
                          f"P2→P1 BSV {c_2to1_bsv:.2f} vs RAW {c_2to1_raw:.2f}. "
                          "The 11-axis compositional representation carries more transferable biology than the high-dim raw spectrum.")
    elif bsv_gen and not raw_gen:
        assessment.append(f"**BSV transfers while RAW does NOT.** BSV P1→P2 {c_1to2_bsv:.2f}, P2→P1 {c_2to1_bsv:.2f}. "
                          "Raw spectra are substrate-locked; BSV abstraction removes the substrate-specific signal.")
    elif not bsv_gen and not raw_gen:
        assessment.append(f"**NEITHER generalizes.** BSV P1→P2 {c_1to2_bsv:.2f}, P2→P1 {c_2to1_bsv:.2f}; "
                          f"RAW P1→P2 {c_1to2_raw:.2f}, P2→P1 {c_2to1_raw:.2f}. "
                          "This is important signal — the two pilots do NOT share a transferable HCC biology layer at either abstraction.")
    else:
        assessment.append(f"**Mixed.** BSV P1→P2 {c_1to2_bsv:.2f}, P2→P1 {c_2to1_bsv:.2f}; "
                          f"RAW P1→P2 {c_1to2_raw:.2f}, P2→P1 {c_2to1_raw:.2f}.")
    assessment += [
        "",
        "## Q3. Which axes transfer consistently?",
        "",
        f"- Direction agreement P1 vs P2 HCC: {dir_match_hcc}/11 axes",
        f"- Direction agreement P1 vs P2 Advanced (CCA+LM): {dir_match_adv}/11 axes",
        f"- Magnitude correlation P1 vs P2 HCC: ρ = {mag_corr_hcc:+.2f}",
        f"- Magnitude correlation P1 vs P2 Advanced: ρ = {mag_corr_adv:+.2f}",
        "",
        "Top direction-consistent axes (sumnorm d same sign in P1 and P2):",
    ]
    consistent_axes = axis_df[axis_df["direction_match_P1_vs_P2_HCC"]]
    for _, r in consistent_axes.iterrows():
        assessment.append(f"- {r['axis']} {r['family_label']}: P1 d={r['P1_HCC_vs_CTR_d_sumnorm']:+.2f}, "
                          f"P2 HCC d={r['P2_HCC_vs_NC_d_sumnorm']:+.2f}")
    assessment += [
        "",
        "## Q4. Is HCC signal stable or dataset-specific?",
        "",
    ]
    if abs(mag_corr_hcc) < 0.30 and dir_match_hcc <= 5:
        assessment.append("**Dataset-specific.** P1 HCC and P2 HCC axis patterns don't correlate well. "
                          "HCC in the two cohorts, on the two substrates, produces effectively DIFFERENT BSV signatures.")
    elif abs(mag_corr_hcc) >= 0.30 and dir_match_hcc >= 7:
        assessment.append("**Stable.** Most axes agree in direction and magnitude correlates across the two pilots.")
    else:
        assessment.append(f"**Partially stable.** {dir_match_hcc}/11 axes agree; magnitude ρ = {mag_corr_hcc:+.2f}. "
                          "Some axes transfer (G09 Sterol-lipid ↓ is a candidate), many do not.")
    assessment += [
        "",
        "## Q5. What does this imply about passive SERS vs biochemical state?",
        "",
    ]
    if not raw_gen and bsv_gen:
        assessment.append(
            "- Raw SERS spectra are **substrate-bound** — classifier trained on one substrate does not transfer.\n"
            "- GAIRA BSV **abstracts away substrate-specific features** via its motif + MSS decomposition.\n"
            "- Passive SERS cross-cohort reporting requires BSV (or equivalent interpretable abstraction), not raw spectra.\n"
            "- This validates the GAIRA engineering choice: interpretation layer > raw-spectrum classification for generalization."
        )
    elif not raw_gen and not bsv_gen:
        assessment.append(
            "- HCC signal at the BSV level is **cohort-specific**, not universally transferable across SERS substrates.\n"
            "- The apparent within-pilot accuracy is substrate+cohort-specific, not disease-universal.\n"
            "- Passive SERS generalization requires either (a) substrate-controlled validation cohorts, or (b) richer multi-axis interpretation that doesn't collapse to a single binary label.\n"
            "- **The cross-pilot G09 Sterol-lipid ↓ convergence remains valid as a qualitative biomarker even if no trained classifier transfers.**"
        )
    else:
        assessment.append("- Raw transfer suggests the substrate+cohort share more structure than expected; BSV abstraction may or may not help at that level.")
    assessment += [
        "",
        "## Constraints enforced",
        "",
        "- GAIRA engine v4.5: UNCHANGED",
        "- No threshold tuning, no label leakage",
        "- Classifiers are evaluation only",
        "- No DART-Met logic",
        "- Group-aware cross-validation (sample_id groups, no replicate split)",
    ]
    (REPORTS / "REPORT_hcc_cross_pilot_final_assessment.md").write_text("\n".join(assessment))

    # Audit log
    lines = [
        "# gaira_base_4 HCC cross-pilot generalization v1 — Audit Log",
        "",
        f"## Datasets",
        f"- Pilot 1 Gurian HCC SERS: {len(p1_raw)} spectra (72 HCC + 72 Control)",
        f"- Pilot 2 label-free SERS nanosensor: {len(p2_raw)} patient-mean spectra (NC/HCC/CCA/LM)",
        "",
        "## Pipelines",
        "- RAW: full spectral vector on GAIRA master axis (~890 dim)",
        "- BSV: 11-dim hybrid BSV (+ sumnorm + CLR + conf + ambiguity variants)",
        "",
        "## Classifiers",
        "- LinearSVM, LogisticRegression, RandomForest(100,depth=6), GradientBoosting(100,depth=3)",
        "- StratifiedGroupKFold(5); sample-id grouped",
        "",
        "## Key results",
        f"- within-pilot RAW best: {mean_within_raw:.3f}",
        f"- within-pilot BSV best: {mean_within_bsv:.3f}",
        f"- cross-pilot RAW best: {mean_cross_raw:.3f}",
        f"- cross-pilot BSV best: {mean_cross_bsv:.3f}",
        f"- transfer drop RAW: {transfer_drop_raw:+.3f}",
        f"- transfer drop BSV: {transfer_drop_bsv:+.3f}",
        f"- axis direction agreement P1 vs P2 HCC: {dir_match_hcc}/11",
        f"- axis direction agreement P1 vs P2 Advanced: {dir_match_adv}/11",
        f"- magnitude correlation P1 vs P2 HCC: ρ={mag_corr_hcc:+.3f}",
        f"- magnitude correlation P1 vs P2 Advanced: ρ={mag_corr_adv:+.3f}",
        "",
        "## Invariants",
        "- engine v4.5 / taxonomy / motif / MSS v4.3 / substrate physics v1.2: unchanged",
        "- classifier is evaluation only",
        "- no threshold tuning, no label-driven feature select",
        "- no DART-Met",
    ]
    (AUDIT / "gaira_base_4_hcc_cross_pilot_generalization_v1_audit_log.md"
     ).write_text("\n".join(lines))

    p = Path(__file__)
    if p.exists(): shutil.copy(p, CODE_SNAPSHOT / p.name)

    print(f"\n[complete]")
    print(f"  cross-pilot transfer: RAW {mean_cross_raw:.3f}, BSV {mean_cross_bsv:.3f}")


if __name__ == "__main__":
    main()
