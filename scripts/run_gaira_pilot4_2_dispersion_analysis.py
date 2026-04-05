from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelBinarizer, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaira.demo.gaira_experiment_runner_utils import load_architecture_registries, load_query_dataframe
from gaira.demo.gaira_pilot_utils import build_pdf_report
from gaira.demo.raw_bsv_pilot_utils import decode_and_align
from scripts.run_gaira_pilot4_cca_hcc_lm_serum_sers import (
    ARCH_DIR,
    CLASS_COLORS,
    DISPLAY_ORDER,
    FAMILY_ORDER,
    FIXED_RADAR_AXES,
    PHASE1_DIR,
    ROOT,
    _broad_label,
    _display_label,
    _extract_sample_id,
)
from scripts.run_gaira_pilot4_1_cca_hcc_lm_serum_patient_level import _load_query_df


PILOT4_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot4_cca_hcc_lm_serum_sers"
)
PILOT41_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot4_1_cca_hcc_lm_serum_patient_level"
)
OUTPUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot4_2_dispersion_analysis"
)
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORT_DIR = OUTPUT_ROOT / "report"


def _ensure_dirs() -> None:
    for path in [OUTPUT_ROOT, TABLES_DIR, FIGURES_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _df_to_md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _nn_purity(X: np.ndarray, labels: np.ndarray, *, n_neighbors: int = 5) -> float:
    n_use = min(n_neighbors + 1, len(X))
    if n_use <= 1:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=n_use)
    nn.fit(X)
    idx = nn.kneighbors(X, return_distance=False)
    scores = []
    for i in range(len(X)):
        neigh = [j for j in idx[i] if j != i][:n_neighbors]
        if not neigh:
            continue
        scores.append(float(np.mean(labels[neigh] == labels[i])))
    return float(np.mean(scores)) if scores else float("nan")


def _representation_metrics(X: np.ndarray, labels_4: np.ndarray, labels_binary: np.ndarray) -> dict[str, float]:
    return {
        "silhouette_4class": float(silhouette_score(X, labels_4)),
        "silhouette_healthy_vs_cancer": float(silhouette_score(X, labels_binary)),
        "nearest_neighbor_purity_4class": _nn_purity(X, labels_4),
    }


def _axis_entropy(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-12, None)
    normalized = clipped / clipped.sum(axis=1, keepdims=True)
    return -(normalized * np.log(normalized)).sum(axis=1)


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((len(x) - 1) * vx + (len(y) - 1) * vy) / max(len(x) + len(y) - 2, 1)
    if pooled <= 1e-12:
        return 0.0
    return float((x.mean() - y.mean()) / math.sqrt(pooled))


def _kw_pvalue(groups: list[np.ndarray]) -> float:
    valid = [g for g in groups if len(g) > 0]
    if len(valid) < 2:
        return float("nan")
    return float(kruskal(*valid).pvalue)


def _multiclass_cv_metrics(X: np.ndarray, labels: np.ndarray) -> tuple[float, float, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=5000)
    pred = cross_val_predict(model, X, labels, cv=cv, method="predict")
    proba = cross_val_predict(model, X, labels, cv=cv, method="predict_proba")
    acc = float(accuracy_score(labels, pred))
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(labels)
    macro_auc = float(roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")) if y_bin.shape[1] > 1 else float("nan")
    conf = confusion_matrix(labels, pred, labels=list(DISPLAY_ORDER))
    conf_df = pd.DataFrame(conf, index=DISPLAY_ORDER, columns=DISPLAY_ORDER)
    return acc, macro_auc, conf_df


def _binary_cv_metrics(X: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=5000)
    pred = cross_val_predict(model, X, labels, cv=cv, method="predict")
    proba = cross_val_predict(model, X, labels, cv=cv, method="predict_proba")[:, 1]
    acc = float(accuracy_score(labels, pred))
    auc = float(roc_auc_score(labels, proba))
    return acc, auc


def _plot_pca(df: pd.DataFrame, hue_col: str, path: Path, title: str) -> None:
    plt.figure(figsize=(8.0, 6.0))
    for label, sub in df.groupby(hue_col, sort=False):
        color = CLASS_COLORS.get(str(label), "#355070")
        plt.scatter(sub["pc1"], sub["pc2"], s=32, alpha=0.8, color=color, label=str(label), edgecolors="none")
    plt.xlabel(f"PC1 ({df['pc1_var'].iloc[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({df['pc2_var'].iloc[0]*100:.1f}%)")
    plt.title(title)
    plt.grid(alpha=0.18, linewidth=0.6)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _pca_df(X: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    out = meta.copy()
    out["pc1"] = coords[:, 0]
    out["pc2"] = coords[:, 1]
    out["pc1_var"] = float(pca.explained_variance_ratio_[0])
    out["pc2_var"] = float(pca.explained_variance_ratio_[1])
    return out


def _feature_importance_summary(feature_names: list[str], coef: np.ndarray) -> str:
    order = np.argsort(np.abs(coef))[::-1][:5]
    return ", ".join([f"{feature_names[i]} ({coef[i]:.3f})" for i in order])


def main() -> None:
    _ensure_dirs()

    bsv_df = pd.read_csv(PILOT4_ROOT / "tables" / "per_spectrum_bsv.csv")
    delta_df = pd.read_csv(PILOT4_ROOT / "tables" / "per_spectrum_delta_bsv.csv")
    family_long = pd.read_csv(PILOT4_ROOT / "tables" / "per_spectrum_family.csv")
    patient_mean_bsv = pd.read_csv(PILOT41_ROOT / "tables" / "patient_level_bsv.csv")
    patient_mean_family = pd.read_csv(PILOT41_ROOT / "tables" / "patient_level_family.csv")
    p4_baseline = pd.read_csv(PILOT4_ROOT / "tables" / "paper_style_baseline_metrics.csv")
    p41_baseline = pd.read_csv(PILOT41_ROOT / "tables" / "patient_level_paper_style_metrics.csv")

    bsv_df["sample_id"] = [_extract_sample_id(k, k) for k in bsv_df["sample_key"].astype(str)]
    delta_df["sample_id"] = [_extract_sample_id(k, k) for k in delta_df["sample_key"].astype(str)]
    bsv_df["class_label_display"] = bsv_df["class_label"].map(_display_label)
    bsv_df["broad_label"] = bsv_df["class_label_display"].map(_broad_label)
    if "class_label" in delta_df.columns:
        delta_df["class_label_display"] = delta_df["class_label"].map(_display_label)
    else:
        delta_df["class_label_display"] = delta_df["class_label_display"].map(_display_label)
    delta_df["broad_label"] = delta_df["class_label_display"].map(_broad_label)
    family_long["class_label_display"] = family_long["class_label_display"].map(_display_label)
    family_long["broad_label"] = family_long["class_label_display"].map(_broad_label)

    axes = FIXED_RADAR_AXES
    for axis in axes:
        if axis not in bsv_df.columns:
            bsv_df[axis] = 0.0
        if axis not in delta_df.columns:
            delta_df[axis] = 0.0

    bsv_df["axis_entropy"] = _axis_entropy(bsv_df[axes].to_numpy(dtype=float))
    family_top = family_long.groupby("sample_key", as_index=False)["family_fraction"].max().rename(columns={"family_fraction": "top1_dominance"})
    family_entropy = family_long.groupby("sample_key").apply(
        lambda df: float(-(df["family_fraction"].clip(lower=1e-12) * np.log(df["family_fraction"].clip(lower=1e-12))).sum())
    ).rename("family_entropy").reset_index()

    query_df = _load_query_df()
    wavenumbers, spectra_matrix = decode_and_align(query_df)
    spectra_matrix = np.asarray(spectra_matrix, dtype=float)
    pca3 = PCA(n_components=3, random_state=42)
    spectral_scores = pca3.fit_transform(StandardScaler().fit_transform(spectra_matrix))
    spectral_score_df = pd.DataFrame(
        {
            "sample_key": query_df["sample_key"].astype(str).values,
            "sample_id": query_df["sample_id"].astype(str).values,
            "class_label_display": query_df["class_label_display"].astype(str).values,
            "broad_label": query_df["broad_label"].astype(str).values,
            "spectral_pc1": spectral_scores[:, 0],
            "spectral_pc2": spectral_scores[:, 1],
            "spectral_pc3": spectral_scores[:, 2],
        }
    )

    bsv_disp = bsv_df.groupby(["sample_id", "class_label_display", "broad_label"], as_index=False).agg(
        **{f"{axis}_var": (axis, "var") for axis in axes},
        **{f"{axis}_mad": (axis, lambda s: float(np.mean(np.abs(s - s.mean())))) for axis in axes},
    ).fillna(0.0)
    bsv_disp["bsv_total_variance"] = bsv_disp[[f"{axis}_var" for axis in axes]].sum(axis=1)

    delta_disp = delta_df.groupby(["sample_id", "class_label_display", "broad_label"], as_index=False).agg(
        **{f"delta_{axis}_var": (axis, "var") for axis in axes},
        **{f"delta_{axis}_mad": (axis, lambda s: float(np.mean(np.abs(s - s.mean())))) for axis in axes},
    ).fillna(0.0)
    delta_disp["delta_bsv_total_variance"] = delta_disp[[f"delta_{axis}_var" for axis in axes]].sum(axis=1)

    family_wide = family_long.pivot_table(
        index=["sample_key", "sample_id", "class_label_display", "broad_label"],
        columns="family",
        values="family_fraction",
        aggfunc="mean",
        fill_value=0.0,
    ).reset_index()
    for family in FAMILY_ORDER:
        if family not in family_wide.columns:
            family_wide[family] = 0.0
    family_disp = family_wide.groupby(["sample_id", "class_label_display", "broad_label"], as_index=False).agg(
        **{f"{family}_var": (family, "var") for family in FAMILY_ORDER},
    ).fillna(0.0)
    family_entropy_sample = family_entropy.merge(family_wide[["sample_key", "sample_id"]], on="sample_key", how="left")
    family_entropy_agg = family_entropy_sample.groupby("sample_id", as_index=False).agg(
        family_entropy_mean=("family_entropy", "mean"),
        family_entropy_var=("family_entropy", "var"),
    ).fillna(0.0)

    spectral_disp = spectral_score_df.groupby(["sample_id", "class_label_display", "broad_label"], as_index=False).agg(
        spectral_pc1_var=("spectral_pc1", "var"),
        spectral_pc2_var=("spectral_pc2", "var"),
        spectral_pc3_var=("spectral_pc3", "var"),
    ).fillna(0.0)
    spectral_disp["spectral_total_variance"] = spectral_disp[["spectral_pc1_var", "spectral_pc2_var", "spectral_pc3_var"]].sum(axis=1)

    sample_meta = bsv_df.groupby("sample_id", as_index=False).agg(
        class_label_display=("class_label_display", "first"),
        broad_label=("broad_label", "first"),
        spectrum_count=("sample_key", "count"),
        axis_entropy_mean=("axis_entropy", "mean"),
        axis_entropy_var=("axis_entropy", "var"),
    ).fillna(0.0)
    top1_agg = family_top.merge(family_wide[["sample_key", "sample_id"]], on="sample_key", how="left").groupby("sample_id", as_index=False).agg(
        top1_dominance_mean=("top1_dominance", "mean"),
        top1_dominance_var=("top1_dominance", "var"),
    ).fillna(0.0)

    dispersion_df = (
        sample_meta
        .merge(bsv_disp, on=["sample_id", "class_label_display", "broad_label"], how="left")
        .merge(delta_disp, on=["sample_id", "class_label_display", "broad_label"], how="left")
        .merge(family_disp, on=["sample_id", "class_label_display", "broad_label"], how="left")
        .merge(family_entropy_agg, on="sample_id", how="left")
        .merge(spectral_disp, on=["sample_id", "class_label_display", "broad_label"], how="left")
        .merge(top1_agg, on="sample_id", how="left")
        .fillna(0.0)
        .sort_values(["class_label_display", "sample_id"])
        .reset_index(drop=True)
    )
    dispersion_df.to_csv(TABLES_DIR / "patient_level_dispersion_features.csv", index=False)

    feature_cols = [
        c for c in dispersion_df.columns
        if c not in {"sample_id", "class_label_display", "broad_label"}
    ]

    class_rows = []
    labels = dispersion_df["class_label_display"].astype(str).to_numpy()
    unique_labels = DISPLAY_ORDER
    for feat in feature_cols:
        groups = [dispersion_df.loc[dispersion_df["class_label_display"] == label, feat].to_numpy(dtype=float) for label in unique_labels]
        pval = _kw_pvalue(groups)
        best_pair = ""
        best_d = float("-inf")
        for i, left in enumerate(unique_labels):
            for right in unique_labels[i + 1 :]:
                xl = dispersion_df.loc[dispersion_df["class_label_display"] == left, feat].to_numpy(dtype=float)
                xr = dispersion_df.loc[dispersion_df["class_label_display"] == right, feat].to_numpy(dtype=float)
                d = abs(_cohens_d(xl, xr))
                if np.isnan(d):
                    continue
                if d > best_d:
                    best_d = d
                    best_pair = f"{left} vs {right}"
        row = {
            "feature_name": feat,
            "kw_pvalue": pval,
            "max_abs_cohens_d": best_d if best_d != float("-inf") else float("nan"),
            "max_effect_pair": best_pair,
        }
        for label in unique_labels:
            row[f"mean_{label}"] = float(dispersion_df.loc[dispersion_df["class_label_display"] == label, feat].mean())
        class_rows.append(row)
    comparison_df = pd.DataFrame(class_rows).sort_values(["max_abs_cohens_d", "kw_pvalue"], ascending=[False, True]).reset_index(drop=True)
    comparison_df.to_csv(TABLES_DIR / "dispersion_class_comparison.csv", index=False)

    scaler = StandardScaler()
    X_disp = scaler.fit_transform(dispersion_df[feature_cols].to_numpy(dtype=float))
    disp_metrics = _representation_metrics(X_disp, labels, dispersion_df["broad_label"].astype(str).to_numpy())
    disp_pca_df = _pca_df(dispersion_df[feature_cols].to_numpy(dtype=float), dispersion_df[["sample_id", "class_label_display", "broad_label"]])
    _plot_pca(disp_pca_df, "class_label_display", FIGURES_DIR / "dispersion_pca_4class.png", "Dispersion Features PCA")

    p41_geom = pd.read_csv(PILOT41_ROOT / "tables" / "patient_level_geometry_comparison.csv")
    geometry_comp = pd.concat(
        [
            p41_geom.assign(representation_source="mean_level"),
            pd.DataFrame([{"space_name": "dispersion_only", "representation_source": "dispersion_only", **disp_metrics}]),
        ],
        ignore_index=True,
        sort=False,
    )
    geometry_comp.to_csv(TABLES_DIR / "dispersion_geometry_comparison.csv", index=False)

    y_binary = (dispersion_df["broad_label"].astype(str) == "Cancer").astype(int).to_numpy()
    binary_acc, binary_auc = _binary_cv_metrics(X_disp, y_binary)
    multi_acc, multi_auc, multi_conf = _multiclass_cv_metrics(X_disp, labels)
    disp_class_df = pd.DataFrame(
        [
            {"task": "healthy_vs_cancer", "cv_accuracy": binary_acc, "macro_auc": binary_auc},
            {"task": "four_class", "cv_accuracy": multi_acc, "macro_auc": multi_auc},
            {
                "task": "pilot4_per_spectrum_baseline",
                "cv_accuracy": float(p4_baseline[p4_baseline["analysis_name"] == "lda_cv_accuracy"]["metric_value"].iloc[0]),
                "macro_auc": float(p4_baseline[p4_baseline["analysis_name"] == "lda_macro_auc"]["metric_value"].iloc[0]),
            },
            {
                "task": "pilot4_1_patient_mean_baseline",
                "cv_accuracy": float(p41_baseline[p41_baseline["analysis_name"] == "patient_level_lda_cv_accuracy"]["metric_value"].iloc[0]),
                "macro_auc": float(p41_baseline[p41_baseline["analysis_name"] == "patient_level_lda_macro_auc"]["metric_value"].iloc[0]),
            },
        ]
    )
    disp_class_df.to_csv(TABLES_DIR / "dispersion_classification_metrics.csv", index=False)
    multi_conf.to_csv(TABLES_DIR / "dispersion_confusion_matrix.csv")

    top_feats = comparison_df.head(12).copy()
    interp_rows = []
    for row in top_feats.itertuples(index=False):
        feat = row.feature_name
        if "nucleic_acid" in feat:
            theme = "nucleic acid / stress"
        elif "small_molecule_metabolite" in feat or "purine" in feat or "family_entropy" in feat:
            theme = "small molecule / purine"
        elif "protein_peptide" in feat:
            theme = "protein / amide"
        elif "lipid_membrane" in feat:
            theme = "lipid / membrane"
        elif "substrate_adsorption_bias" in feat or "top1_dominance" in feat:
            theme = "adsorption / dominance"
        elif "guanidine_like" in feat or "methylated_purine_like" in feat or "purine_core_like" in feat:
            theme = "small molecule / purine"
        elif "spectral_pc" in feat:
            theme = "spectral heterogeneity"
        else:
            theme = "mixed"
        interp_rows.append(
            {
                "feature_name": feat,
                "max_abs_cohens_d": float(row.max_abs_cohens_d),
                "max_effect_pair": row.max_effect_pair,
                "biochemical_theme": theme,
                "interpretation": (
                    "supports higher within-sample heterogeneity"
                    if any(x in feat for x in ["var", "mad", "entropy"])
                    else "mixed dispersion signal"
                ),
            }
        )
    interp_df = pd.DataFrame(interp_rows)
    interp_df.to_csv(TABLES_DIR / "dispersion_biochemical_interpretation.csv", index=False)

    mean_bsv = patient_mean_bsv.drop(columns=["sample_id", "class_label_display", "broad_label"]).fillna(0.0)
    mean_family = patient_mean_family.drop(columns=["sample_id", "class_label_display", "broad_label"]).fillna(0.0)
    mean_family = mean_family[[c for c in FAMILY_ORDER if c in mean_family.columns]]
    mean_features = pd.concat([mean_bsv.reset_index(drop=True), mean_family.reset_index(drop=True)], axis=1).fillna(0.0)
    disp_features = dispersion_df[feature_cols].reset_index(drop=True).fillna(0.0)
    combined_features = pd.concat([mean_features, disp_features], axis=1).fillna(0.0)

    def eval_multi(X: pd.DataFrame, name: str) -> dict[str, object]:
        Xs = StandardScaler().fit_transform(X.to_numpy(dtype=float))
        acc, auc, _ = _multiclass_cv_metrics(Xs, labels)
        model = LogisticRegression(max_iter=5000).fit(Xs, labels)
        coef = np.mean(np.abs(model.coef_), axis=0)
        return {
            "feature_set": name,
            "cv_accuracy": acc,
            "macro_auc": auc,
            "top_features": _feature_importance_summary(list(X.columns), coef),
        }

    mean_vs_disp_df = pd.DataFrame(
        [
            eval_multi(mean_features, "mean_only"),
            eval_multi(disp_features, "dispersion_only"),
            eval_multi(combined_features, "mean_plus_dispersion"),
        ]
    )
    mean_vs_disp_df.to_csv(TABLES_DIR / "mean_vs_dispersion_comparison.csv", index=False)

    best_disp_acc = float(mean_vs_disp_df[mean_vs_disp_df["feature_set"] == "dispersion_only"]["cv_accuracy"].iloc[0])
    best_mean_acc = float(mean_vs_disp_df[mean_vs_disp_df["feature_set"] == "mean_only"]["cv_accuracy"].iloc[0])
    best_combined_acc = float(mean_vs_disp_df[mean_vs_disp_df["feature_set"] == "mean_plus_dispersion"]["cv_accuracy"].iloc[0])
    if best_disp_acc > best_mean_acc and best_disp_acc > 0.75:
        signal_label = "primary signal"
    elif best_combined_acc > best_mean_acc or best_disp_acc > 0.55:
        signal_label = "partial signal"
    else:
        signal_label = "noise"

    decision_lines = [
        "# Pilot4.2 Dispersion Decision",
        "",
        f"1. Does within-patient dispersion carry disease signal? `{'yes' if best_disp_acc > 0.5 else 'weakly or no'}`",
        f"2. Is dispersion stronger than mean for any task? `{'yes' if best_disp_acc > best_mean_acc else 'no'}`",
        f"3. Does it improve interpretation or classification? `{'classification and interpretation' if best_combined_acc > best_mean_acc else 'interpretation only or weakly'}`",
        f"4. Is dispersion noise, partial signal, or primary signal? `{signal_label}`",
    ]
    (REPORT_DIR / "pilot4_2_dispersion_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# GAIRAv3 Pilot4.2 Dispersion Report",
        "",
        "## 1. Dispersion feature construction",
        f"- per-sample dispersion table rows: `{len(dispersion_df)}`",
        f"- feature count: `{len(feature_cols)}`",
        "",
        "## 2. Class comparison",
        _df_to_md(comparison_df.head(20)),
        "",
        "## 3. Geometry in dispersion space",
        _df_to_md(geometry_comp),
        "",
        "## 4. Classification using dispersion",
        _df_to_md(disp_class_df),
        "",
        "## 5. Biological interpretation of dispersion",
        _df_to_md(interp_df),
        "",
        "## 6. Mean vs dispersion complementarity",
        _df_to_md(mean_vs_disp_df),
        "",
        "## 7. Final conclusion",
        *decision_lines[2:],
    ]
    report_md = REPORT_DIR / "GAIRAv3_Pilot4_2_Dispersion_Report.md"
    report_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    build_pdf_report(report_md, [FIGURES_DIR / "dispersion_pca_4class.png"], REPORT_DIR / "GAIRAv3_Pilot4_2_Dispersion_Report.pdf")


if __name__ == "__main__":
    main()
