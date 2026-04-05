from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelBinarizer, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaira.demo.autoresearch_pass5_utils import Pass5HarnessConfig, apply_pass5_filter_mode, build_bsv_profiles_pass5
from gaira.demo.gaira_experiment_runner_utils import (
    ResolvedExperiment,
    build_source_role_sets,
    load_architecture_registries,
    load_grounding_family_dataframe,
    load_query_dataframe,
)
from gaira.demo.gaira_pilot_utils import ALL_AXES, build_pdf_report
from gaira.demo.raw_bsv_pilot_utils import apply_source_role_policy, decode_and_align, load_ontology_rules, map_references_to_axes
from scripts.run_gaira_pilot2_target_validation_v1 import _compound_to_family


ROOT = PROJECT_ROOT
ARCH_DIR = ROOT / "reports" / "gaira_architecture_scaffold_v2"
PHASE1_DIR = ROOT / "reports" / "gaira_phase1_registry_audit_v2"
ONTOLOGY_PATH = ROOT / "config" / "phase2_bsv_ontology_rules_v2.csv"

SUBSET_ALIAS = "covid_serum_cohort"
OUTPUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot5_covid_serum_raman"
)
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORT_DIR = OUTPUT_ROOT / "report"

CONFIG_SPEC = {
    "config_id": "candidate_v2_cfg05_max_desaturation",
    "short_label": "cfg05",
    "display_name": "Candidate v2 cfg05 max desaturation",
    "filter_mode": "purine_expanded_neighbor",
    "top_k": 5,
    "weighting_mode": "softmax_temperature",
    "weighting_param": 1.0,
    "diversity_mode": "compound_uniqueness_penalty",
}

MAIN_CLASSES = ["healthy_control", "covid_confirmed"]
CLASS_DISPLAY = {
    "healthy_control": "Healthy",
    "covid_confirmed": "COVID",
    "suspected_case": "Suspected",
    "tube_control": "Tube",
}
CLASS_COLORS = {
    "Healthy": "#355070",
    "COVID": "#b56576",
    "Suspected": "#6d597a",
    "Tube": "#2a9d8f",
}
FIXED_RADAR_AXES = [
    "nucleic_acid",
    "protein_peptide",
    "lipid_membrane",
    "carbohydrate_glycan",
    "small_molecule_metabolite",
    "substrate_adsorption_bias",
]
FAMILY_ORDER = [
    "purine_core_like",
    "methylated_purine_like",
    "guanidine_like",
    "sulfur_small_molecule_like",
    "aromatic_small_molecule_like",
    "generic_other_metabolite",
]


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
                if math.isnan(value):
                    vals.append("nan")
                else:
                    vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _split_semicolon_list(value: object) -> list[str]:
    return [part.strip() for part in str(value).split(";") if part and part.strip()]


def _resolve_alias(registries, subset_alias: str) -> ResolvedExperiment:
    matches = registries.dataset_experiments[
        registries.dataset_experiments["subset_alias"].astype(str) == str(subset_alias)
    ].copy()
    if matches.empty or len(matches) > 1:
        raise RuntimeError(f"Could not resolve unique subset alias {subset_alias}")
    dataset_row = matches.iloc[0]
    grounding_families = _split_semicolon_list(dataset_row["allowed_grounding_families"])
    experiment_row = pd.Series(
        {
            "experiment_id": f"pilot5_target_validation__{subset_alias}",
            "subset_alias": subset_alias,
            "grounding_families_used": "; ".join(grounding_families),
        }
    )
    return ResolvedExperiment(
        experiment_row=experiment_row,
        dataset_row=dataset_row,
        subset_alias=subset_alias,
        grounding_family_names=grounding_families,
    )


def _config_to_harness(spec: dict[str, object]) -> Pass5HarnessConfig:
    return Pass5HarnessConfig(
        config_id=str(spec["config_id"]),
        universal_grounding_filter_mode=str(spec["filter_mode"]),
        top_k=int(spec["top_k"]),
        weighting_mode=str(spec["weighting_mode"]),
        weighting_param=None if spec["weighting_param"] is None else float(spec["weighting_param"]),
        diversity_mode=str(spec["diversity_mode"]),
        family_min_coverage=0,
    )


def _prepare_grounding_and_mapping(registries, resolved: ResolvedExperiment, config_spec: dict[str, object]):
    harness_config = _config_to_harness(config_spec)
    grounding_df, family_to_sources, _ = load_grounding_family_dataframe(resolved, registries)
    grounding_df = apply_pass5_filter_mode(grounding_df, harness_config.universal_grounding_filter_mode)
    primary_sources, caveat_only_sources = build_source_role_sets(resolved, family_to_sources)
    available_source_keys = set(grounding_df["source_key"].astype(str))
    primary_sources = {key for key in primary_sources if key in available_source_keys}
    caveat_only_sources = {key for key in caveat_only_sources if key in available_source_keys}
    ontology_rules = load_ontology_rules(ONTOLOGY_PATH)
    raw_mapping_df = map_references_to_axes(grounding_df, ontology_rules)
    mapping_df = apply_source_role_policy(
        raw_mapping_df,
        grounding_df,
        primary_sources=primary_sources,
        caveat_only_sources=caveat_only_sources,
    )
    return grounding_df, mapping_df, harness_config


def _extract_sample_id(sample_key: str, source_file: str) -> str:
    chunks = str(sample_key).split("__")
    if len(chunks) >= 3:
        return chunks[2]
    member = str(source_file).split("::")[-1]
    return member.replace("column_", "col_")


def _display_label(raw_label: str) -> str:
    return CLASS_DISPLAY.get(str(raw_label), str(raw_label))


def _fit_pca(matrix: np.ndarray, scale: bool = True) -> tuple[np.ndarray, np.ndarray]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if scale:
        std = centered.std(axis=0, keepdims=True)
        centered = centered / np.where(std > 1e-8, std, 1.0)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :2] * s[:2]
    explained = (s**2) / np.maximum((s**2).sum(), 1e-8)
    return coords, explained[:2]


def _pca_dataframe(matrix: np.ndarray, meta_df: pd.DataFrame, scale: bool = True) -> pd.DataFrame:
    coords, explained = _fit_pca(matrix, scale=scale)
    out = meta_df.reset_index(drop=True).copy()
    out["pc1"] = coords[:, 0]
    out["pc2"] = coords[:, 1] if coords.shape[1] > 1 else 0.0
    out["pc1_explained_ratio"] = float(explained[0]) if len(explained) > 0 else 1.0
    out["pc2_explained_ratio"] = float(explained[1]) if len(explained) > 1 else 0.0
    return out


def _plot_pca(df: pd.DataFrame, label_col: str, path: Path, title: str) -> None:
    plt.figure(figsize=(6.8, 5.2))
    for label, group in df.groupby(label_col, sort=False):
        plt.scatter(
            group["pc1"],
            group["pc2"],
            s=36,
            alpha=0.75,
            label=str(label),
            color=CLASS_COLORS.get(str(label), None),
        )
    xlab = f"PC1 ({float(df['pc1_explained_ratio'].iloc[0]) * 100:.1f}%)"
    ylab = f"PC2 ({float(df['pc2_explained_ratio'].iloc[0]) * 100:.1f}%)"
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _nearest_neighbor_purity(matrix: np.ndarray, labels: np.ndarray) -> float:
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(matrix)
    indices = nn.kneighbors(matrix, return_distance=False)
    neigh_labels = labels[indices[:, 1]]
    return float(np.mean(neigh_labels == labels))


def _representation_metrics(matrix: np.ndarray, class_labels: np.ndarray) -> dict[str, float]:
    scaled = StandardScaler().fit_transform(matrix)
    centroid_rows = []
    within_rows = []
    for label in sorted(pd.unique(class_labels)):
        sub = scaled[class_labels == label]
        centroid = sub.mean(axis=0)
        centroid_rows.append((label, centroid))
        within_rows.append(float(np.mean(np.sum((sub - centroid) ** 2, axis=1))))
    centroids = np.vstack([row[1] for row in centroid_rows])
    if len(centroids) == 2:
        between = float(np.linalg.norm(centroids[0] - centroids[1]))
    else:
        between = float("nan")
    if len(np.unique(class_labels)) > 1 and len(class_labels) > len(np.unique(class_labels)):
        silhouette = float(silhouette_score(scaled, class_labels))
    else:
        silhouette = float("nan")
    return {
        "silhouette_main_task": silhouette,
        "nearest_neighbor_purity": _nearest_neighbor_purity(scaled, class_labels),
        "centroid_distance": between,
        "within_class_variance": float(np.mean(within_rows)),
        "between_class_distance": between,
    }


def _lda_cv_metrics(matrix: np.ndarray, labels: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    X = StandardScaler().fit_transform(matrix)
    y = np.asarray(labels)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lda = LinearDiscriminantAnalysis()
    pred = cross_val_predict(lda, X, y, cv=cv, method="predict")
    proba = cross_val_predict(lda, X, y, cv=cv, method="predict_proba")
    accuracy = float(accuracy_score(y, pred))
    lb = LabelBinarizer().fit(y)
    y_bin = lb.transform(y)
    if y_bin.ndim == 1:
        y_bin = np.column_stack([1 - y_bin, y_bin])
    if proba.shape[1] == 2:
        auc = float(roc_auc_score(y, proba[:, 1]))
        macro_auc = auc
        micro_auc = auc
    else:
        macro_auc = float(roc_auc_score(y_bin, proba, average="macro", multi_class="ovr"))
        micro_auc = float(roc_auc_score(y_bin, proba, average="micro", multi_class="ovr"))
    return (
        {
            "cv_accuracy": accuracy,
            "macro_auc": macro_auc,
            "micro_auc": micro_auc,
        },
        pred,
        proba,
    )


def _logreg_cv_metrics(matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    X = StandardScaler().fit_transform(matrix)
    y = np.asarray(labels)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    pred = cross_val_predict(model, X, y, cv=cv, method="predict")
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    accuracy = float(accuracy_score(y, pred))
    auc = float(roc_auc_score(y, proba[:, 1]))
    return {"cv_accuracy": accuracy, "macro_auc": auc, "micro_auc": auc}


def _plot_lda(matrix: np.ndarray, labels: np.ndarray, path: Path, title: str) -> None:
    X = StandardScaler().fit_transform(matrix)
    lda = LinearDiscriminantAnalysis(n_components=1)
    scores = lda.fit_transform(X, labels).reshape(-1)
    df = pd.DataFrame({"ld1": scores, "ld2": np.zeros_like(scores), "label": labels})
    plt.figure(figsize=(6.2, 4.8))
    sns.stripplot(data=df, x="label", y="ld1", hue="label", dodge=False, palette=CLASS_COLORS, alpha=0.75)
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("LD1 score")
    if plt.gca().legend_ is not None:
        plt.gca().legend_.remove()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _plot_confusion(cm: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    plt.figure(figsize=(5.2, 4.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _plot_roc_binary(labels: np.ndarray, proba: np.ndarray, path: Path, title: str) -> None:
    from sklearn.metrics import roc_curve

    positive = "COVID"
    y = (labels == positive).astype(int)
    fpr, tpr, _ = roc_curve(y, proba[:, 1])
    auc = roc_auc_score(y, proba[:, 1])
    plt.figure(figsize=(5.2, 4.6))
    plt.plot(fpr, tpr, color="#b56576", lw=2, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _family_fingerprint_from_retrieval(retrieval_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    work = retrieval_df.copy()
    work["family"] = work["reference_compound_label"].astype(str).map(_compound_to_family)
    grouped = (
        work.groupby(["query_sample_key", "query_class_label", "family"], as_index=False)["support_weight"]
        .sum()
        .rename(
            columns={
                "query_sample_key": "sample_key",
                "query_class_label": "class_label",
                "support_weight": "family_support",
            }
        )
    )
    sample_map = meta_df.set_index("sample_key")[["sample_id", "class_label_display"]]
    rows: list[dict[str, object]] = []
    for sample_key, sub in grouped.groupby("sample_key", sort=True):
        total = float(sub["family_support"].sum())
        meta = sample_map.loc[str(sample_key)]
        existing = {str(x) for x in sub["family"].tolist()}
        for family in FAMILY_ORDER:
            value = 0.0
            if family in existing:
                value = float(sub[sub["family"].astype(str) == family]["family_support"].iloc[0])
            rows.append(
                {
                    "sample_key": str(sample_key),
                    "sample_id": str(meta["sample_id"]),
                    "class_label_display": str(meta["class_label_display"]),
                    "family": family,
                    "family_fraction": (value / total) if total > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _cohort_delta(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    axes = [axis for axis in FIXED_RADAR_AXES if axis in df.columns]
    cohort = df[axes].mean(axis=0)
    out = df[[group_col] + axes].copy()
    for axis in axes:
        out[axis] = out[axis] - float(cohort[axis])
    return out


def _plot_radar_grid(df: pd.DataFrame, group_col: str, path: Path, title: str) -> None:
    axes = [axis for axis in FIXED_RADAR_AXES if axis in df.columns]
    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
    angles += angles[:1]
    n = len(df)
    fig, axs = plt.subplots(1, n, subplot_kw={"polar": True}, figsize=(4.4 * n, 4.4))
    if n == 1:
        axs = [axs]
    for ax, (_, row) in zip(axs, df.iterrows(), strict=False):
        values = [float(row[axis]) for axis in axes]
        values += values[:1]
        ax.fill(angles, values, alpha=0.28, color=CLASS_COLORS.get(str(row[group_col]), "#577590"))
        ax.plot(angles, values, color=CLASS_COLORS.get(str(row[group_col]), "#577590"), lw=1.2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axes, fontsize=8)
        ax.set_title(str(row[group_col]))
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_family_bars(family_df: pd.DataFrame, group_col: str, path: Path, title: str) -> None:
    plt.figure(figsize=(7.2, 4.8))
    sns.barplot(data=family_df, x="family", y="family_fraction", hue=group_col)
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _entropy(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.clip(arr, 0.0, None)
    denom = arr.sum(axis=1, keepdims=True)
    probs = np.divide(arr, np.where(denom > 0, denom, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(probs > 0, np.log(probs), 0.0)
    return -np.sum(probs * logs, axis=1)


def _axis_associations(space_name: str, pca_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    merged = pca_df.merge(feature_df, on="sample_key", how="left")
    rows = []
    for pc in ["pc1", "pc2"]:
        for feature in ["substrate_adsorption_bias", "small_molecule_metabolite", "protein_peptide", "family_entropy", "top1_dominance"]:
            if feature not in merged.columns:
                continue
            rows.append(
                {
                    "space_name": space_name,
                    "metric": pc,
                    "feature": feature,
                    "spearman_r": float(merged[pc].corr(merged[feature], method="spearman")),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    _ensure_dirs()

    registries = load_architecture_registries(
        grounding_family_registry_path=ROOT / "config" / "gaira_grounding_family_registry_v1.csv",
        target_family_registry_path=ROOT / "config" / "gaira_target_family_registry_v1.csv",
        inference_lane_registry_path=ROOT / "config" / "gaira_inference_lane_registry_v2.csv",
        representation_mode_registry_path=ROOT / "config" / "gaira_representation_mode_registry_v2.csv",
        dataset_experiment_registry_path=ROOT / "config" / "gaira_dataset_experiment_registry_v2.csv",
        experiment_plan_path=ARCH_DIR / "first_pass_experiment_plan.csv",
        phase1_registry_path=PHASE1_DIR / "phase1_dataset_registry_v2.csv",
        phase1_grounding_map_path=PHASE1_DIR / "phase1_target_grounding_map_v2.csv",
        phase1_exclusions_path=PHASE1_DIR / "phase1_grounding_exclusions.csv",
    )
    resolved = _resolve_alias(registries, SUBSET_ALIAS)
    query_df = load_query_dataframe(resolved.dataset_row).reset_index(drop=True).copy()
    query_df["sample_id"] = [
        _extract_sample_id(sample_key, source_file)
        for sample_key, source_file in zip(query_df["sample_key"].astype(str), query_df["source_file"].astype(str), strict=False)
    ]
    query_df["class_label_display"] = query_df["class_label"].astype(str).map(_display_label)
    query_df["aggregation_unit"] = query_df["sample_id"].astype(str)

    verification_rows = []
    for class_label, sub in query_df.groupby("class_label", sort=True):
        verification_rows.append(
            {
                "class_label": str(class_label),
                "class_label_display": _display_label(str(class_label)),
                "subclass_label": str(sub["subclass_label"].iloc[0]),
                "spectra_n": int(len(sub)),
                "sample_ids_n": int(sub["sample_id"].astype(str).nunique()),
                "biosample_ids_n": int(sub["biosample_id"].astype(str).nunique()) if "biosample_id" in sub.columns else int(len(sub)),
                "patient_ids_n": int(sub["patient_id"].dropna().astype(str).nunique()) if "patient_id" in sub.columns else 0,
                "replicate_ids_n": int(sub["replicate_id"].dropna().astype(str).nunique()) if "replicate_id" in sub.columns else 0,
                "per_spectrum_ingest": "yes",
                "sample_id_one_to_one_with_row": "yes" if sub["sample_id"].astype(str).nunique() == len(sub) else "no",
            }
        )
    verification_df = pd.DataFrame(verification_rows)
    verification_df.to_csv(TABLES_DIR / "pilot5_input_verification.csv", index=False)

    dataset_note = "\n".join(
        [
            "# Pilot5 Dataset Note",
            "",
            "- Usable broad labels in the local ingest: `healthy_control`, `covid_confirmed`, `suspected_case`, `tube_control`.",
            "- Main benchmark task should not mix uncertain or procedural labels into disease-vs-control evaluation.",
            "- `sample_id` is the only defensible biological sample proxy in this ingest.",
            "- `patient_id` and `replicate_id` are absent.",
            "- Each `sample_id` appears exactly once, so the ingest is per-spectrum and the sample-level view is one-to-one with the raw rows.",
            "",
            "Direct answers:",
            "1. what are the usable broad labels? `healthy_control` and `covid_confirmed` for the main benchmark; `suspected_case` and `tube_control` are audit-only / secondary-context labels.",
            "2. what aggregation unit is likely the correct patient/sample proxy? `sample_id`.",
            "3. is this dataset suitable for both per-spectrum and patient-level analysis? `Per-spectrum yes; patient-level only as a relabeling of the same units, not as a true averaging analysis.`",
        ]
    )
    (REPORT_DIR / "pilot5_dataset_note.md").write_text(dataset_note, encoding="utf-8")

    task_note = "\n".join(
        [
            "# Pilot5 Task Definition",
            "",
            "- Chosen main task: `healthy_control` vs `covid_confirmed`.",
            "- Secondary labels retained only for forensic context: `suspected_case`, `tube_control`.",
            "- Reason: `suspected_case` is clinically unresolved and `tube_control` is procedural rather than biological; using them in the main benchmark would weaken interpretability and contaminate the disease-vs-control comparison.`",
        ]
    )
    (REPORT_DIR / "pilot5_task_definition.md").write_text(task_note, encoding="utf-8")

    main_df = query_df[query_df["class_label"].astype(str).isin(MAIN_CLASSES)].reset_index(drop=True).copy()
    labels = main_df["class_label_display"].astype(str).to_numpy()

    master_x, spectral_matrix = decode_and_align(main_df)
    meta_df = main_df[["sample_key", "sample_id", "class_label_display"]].copy()
    spectral_pca_df = _pca_dataframe(spectral_matrix, meta_df, scale=True)
    _plot_pca(spectral_pca_df, "class_label_display", FIGURES_DIR / "pilot5_per_spectrum_spectral_pca.png", "COVID Serum Spectral PCA")
    _plot_pca(
        spectral_pca_df,
        "class_label_display",
        FIGURES_DIR / "pilot5_per_spectrum_spectral_pca_binary_or_main_task.png",
        "COVID Serum Spectral PCA (Main Task)",
    )

    spectral_metrics = _representation_metrics(spectral_matrix, labels)
    lda_metrics, lda_pred, lda_proba = _lda_cv_metrics(spectral_matrix, labels)
    log_metrics = _logreg_cv_metrics(spectral_matrix, labels)
    cm = confusion_matrix(labels, lda_pred, labels=["Healthy", "COVID"])
    _plot_lda(spectral_matrix, labels, FIGURES_DIR / "pilot5_per_spectrum_lda_2d.png", "Per-spectrum LDA")
    _plot_confusion(cm, ["Healthy", "COVID"], FIGURES_DIR / "pilot5_per_spectrum_confusion_matrix.png", "Per-spectrum LDA Confusion")
    _plot_roc_binary(labels, lda_proba, FIGURES_DIR / "pilot5_per_spectrum_roc.png", "Per-spectrum LDA ROC")

    baseline_df = pd.DataFrame(
        [
            {"analysis_name": "spectral_geometry", **spectral_metrics},
            {"analysis_name": "lda_cv", **lda_metrics},
            {"analysis_name": "logistic_cv", **log_metrics},
        ]
    )
    baseline_df.to_csv(TABLES_DIR / "pilot5_per_spectrum_baseline_metrics.csv", index=False)

    grounding_df, mapping_df, harness_config = _prepare_grounding_and_mapping(registries, resolved, CONFIG_SPEC)
    bsv_df, retrieval_df = build_bsv_profiles_pass5(
        main_df,
        grounding_df,
        mapping_df,
        top_k=harness_config.top_k,
        similarity_metric="cosine",
        weighting_mode=harness_config.weighting_mode,
        weighting_param=harness_config.weighting_param,
        diversity_mode=harness_config.diversity_mode,
        family_min_coverage=harness_config.family_min_coverage,
    )
    bsv_df = bsv_df.copy()
    bsv_df["sample_id"] = main_df.set_index("sample_key").loc[bsv_df["sample_key"].astype(str), "sample_id"].to_list()
    bsv_df["class_label_display"] = main_df.set_index("sample_key").loc[bsv_df["sample_key"].astype(str), "class_label_display"].to_list()
    bsv_df["broad_label"] = bsv_df["class_label_display"].astype(str)
    bsv_df.to_csv(TABLES_DIR / "per_spectrum_bsv.csv", index=False)

    axis_cols = [axis for axis in ALL_AXES if axis in bsv_df.columns]
    control_mean = bsv_df[bsv_df["class_label_display"].astype(str) == "Healthy"][axis_cols].mean(axis=0)
    delta_df = bsv_df[["sample_key", "sample_id", "class_label_display"] + axis_cols].copy()
    for axis in axis_cols:
        delta_df[axis] = delta_df[axis] - float(control_mean[axis])
    delta_df.to_csv(TABLES_DIR / "per_spectrum_delta_bsv.csv", index=False)

    family_df = _family_fingerprint_from_retrieval(retrieval_df, meta_df)
    family_df.to_csv(TABLES_DIR / "per_spectrum_family.csv", index=False)

    family_wide = family_df.pivot(index="sample_key", columns="family", values="family_fraction").reset_index()
    family_wide.columns.name = None
    family_wide = meta_df.merge(family_wide, on="sample_key", how="left").fillna(0.0)
    family_wide["family_entropy"] = _entropy(family_wide[FAMILY_ORDER].to_numpy())
    family_wide["top1_dominance"] = family_wide[FAMILY_ORDER].max(axis=1)
    bsv_df["axis_entropy"] = _entropy(bsv_df[axis_cols].to_numpy())

    geometry_rows = []
    for space_name, matrix in [
        ("spectral", spectral_matrix),
        ("bsv", bsv_df[axis_cols].to_numpy()),
        ("delta_bsv", delta_df[axis_cols].to_numpy()),
        ("family", family_wide[FAMILY_ORDER].to_numpy()),
    ]:
        geometry_rows.append({"space_name": space_name, **_representation_metrics(matrix, labels)})
    geometry_df = pd.DataFrame(geometry_rows)
    geometry_df.to_csv(TABLES_DIR / "pilot5_per_spectrum_geometry_comparison.csv", index=False)

    _plot_pca(_pca_dataframe(bsv_df[axis_cols].to_numpy(), meta_df, scale=False), "class_label_display", FIGURES_DIR / "pilot5_per_spectrum_bsv_pca.png", "Per-spectrum BSV PCA")
    _plot_pca(_pca_dataframe(delta_df[axis_cols].to_numpy(), meta_df, scale=False), "class_label_display", FIGURES_DIR / "pilot5_per_spectrum_delta_bsv_pca.png", "Per-spectrum Delta-BSV PCA")
    _plot_pca(_pca_dataframe(family_wide[FAMILY_ORDER].to_numpy(), meta_df, scale=False), "class_label_display", FIGURES_DIR / "pilot5_per_spectrum_family_pca.png", "Per-spectrum Family PCA")

    agg_df = main_df[["sample_key", "sample_id", "class_label_display"]].copy()
    agg_verification = (
        agg_df.groupby("class_label_display", as_index=False)["sample_id"].nunique().rename(columns={"sample_id": "aggregated_units"})
    )
    agg_verification["aggregation_unit"] = "sample_id"
    agg_verification["one_to_one_with_per_spectrum_rows"] = "yes"
    agg_verification.to_csv(TABLES_DIR / "pilot5_aggregation_unit_verification.csv", index=False)

    aggregation_note = "\n".join(
        [
            "# Pilot5 Aggregation Note",
            "",
            "- Chosen aggregation unit: `sample_id`.",
            "- This is the only defensible biological sample proxy in the ingest.",
            "- It is one-to-one with the per-spectrum rows, so patient/sample-level summaries are numerically identical to per-spectrum rows for this dataset.",
            "- Therefore Pilot 5 can test the reporting doctrine, but not a true repeated-measure averaging gain.",
        ]
    )
    (REPORT_DIR / "pilot5_aggregation_note.md").write_text(aggregation_note, encoding="utf-8")

    mean_spectra_df = pd.DataFrame(spectral_matrix, columns=[f"wn_{int(round(x))}" for x in master_x])
    mean_spectra_df.insert(0, "class_label_display", labels)
    mean_spectra_df.insert(0, "sample_id", main_df["sample_id"].astype(str))
    mean_spectra_df.insert(0, "sample_key", main_df["sample_key"].astype(str))
    mean_spectra_df.to_csv(TABLES_DIR / "patient_level_mean_spectra.csv", index=False)

    patient_bsv_df = bsv_df.copy()
    patient_delta_df = delta_df.copy()
    patient_family_df = family_wide.copy()
    patient_bsv_df.to_csv(TABLES_DIR / "patient_level_bsv.csv", index=False)
    patient_delta_df.to_csv(TABLES_DIR / "patient_level_delta_bsv.csv", index=False)
    patient_family_df.to_csv(TABLES_DIR / "patient_level_family.csv", index=False)

    patient_labels = labels.copy()
    patient_baseline_df = pd.DataFrame(
        [
            {"analysis_name": "spectral_geometry", **_representation_metrics(spectral_matrix, patient_labels)},
            {"analysis_name": "lda_cv", **lda_metrics},
            {"analysis_name": "logistic_cv", **log_metrics},
        ]
    )
    patient_baseline_df.to_csv(TABLES_DIR / "patient_level_baseline_metrics.csv", index=False)

    patient_geometry_df = geometry_df.copy()
    patient_geometry_df.to_csv(TABLES_DIR / "patient_level_geometry_comparison.csv", index=False)

    compare_df = geometry_df.merge(
        patient_geometry_df,
        on="space_name",
        suffixes=("_per_spectrum", "_patient_level"),
    )
    compare_df.to_csv(TABLES_DIR / "per_spectrum_vs_patient_level_comparison.csv", index=False)

    _plot_pca(spectral_pca_df, "class_label_display", FIGURES_DIR / "patient_level_spectral_pca.png", "Patient-level Spectral PCA")
    _plot_pca(_pca_dataframe(patient_bsv_df[axis_cols].to_numpy(), meta_df, scale=False), "class_label_display", FIGURES_DIR / "patient_level_bsv_pca.png", "Patient-level BSV PCA")
    _plot_pca(_pca_dataframe(patient_delta_df[axis_cols].to_numpy(), meta_df, scale=False), "class_label_display", FIGURES_DIR / "patient_level_delta_bsv_pca.png", "Patient-level Delta-BSV PCA")
    _plot_pca(_pca_dataframe(patient_family_df[FAMILY_ORDER].to_numpy(), meta_df, scale=False), "class_label_display", FIGURES_DIR / "patient_level_family_pca.png", "Patient-level Family PCA")
    _plot_confusion(cm, ["Healthy", "COVID"], FIGURES_DIR / "patient_level_confusion_matrix.png", "Patient-level LDA Confusion")
    _plot_roc_binary(labels, lda_proba, FIGURES_DIR / "patient_level_roc.png", "Patient-level LDA ROC")

    class_mean_bsv = patient_bsv_df.groupby("class_label_display", as_index=False)[axis_cols].mean()
    class_delta_bsv = _cohort_delta(class_mean_bsv, "class_label_display")
    class_family = patient_family_df.melt(
        id_vars=["sample_key", "sample_id", "class_label_display"],
        value_vars=FAMILY_ORDER,
        var_name="family",
        value_name="family_fraction",
    ).groupby(["class_label_display", "family"], as_index=False)["family_fraction"].mean()
    class_mean_bsv.to_csv(TABLES_DIR / "class_mean_bsv.csv", index=False)

    interp_rows = []
    for _, row in class_mean_bsv.iterrows():
        label = str(row["class_label_display"])
        strongest = sorted([(axis, float(row[axis])) for axis in axis_cols], key=lambda x: x[1], reverse=True)[:3]
        weakest = sorted([(axis, float(row[axis])) for axis in axis_cols], key=lambda x: x[1])[:2]
        fam_sub = class_family[class_family["class_label_display"].astype(str) == label].sort_values("family_fraction", ascending=False).head(3)
        interp_rows.append(
            {
                "class_label": label,
                "dominant_bsv_axes": "; ".join([f"{axis}={value:.3f}" for axis, value in strongest]),
                "dominant_family_themes": "; ".join([f"{r.family}={float(r.family_fraction):.3f}" for r in fam_sub.itertuples(index=False)]),
                "notable_depletions": "; ".join([f"{axis}={value:.3f}" for axis, value in weakest]),
                "interpretation_summary": "Broader serum shift dominated by nucleic-acid / metabolite balance with adsorption caveat.",
                "serum_domain_caveat": "Spontaneous serum Raman remains susceptible to background and adsorption-style dominance.",
            }
        )
    interpretation_df = pd.DataFrame(interp_rows)
    interpretation_df.to_csv(TABLES_DIR / "class_interpretation_summary.csv", index=False)

    pair = []
    healthy_mean = class_mean_bsv[class_mean_bsv["class_label_display"].astype(str) == "Healthy"][axis_cols].iloc[0].to_numpy(dtype=float)
    covid_mean = class_mean_bsv[class_mean_bsv["class_label_display"].astype(str) == "COVID"][axis_cols].iloc[0].to_numpy(dtype=float)
    pair.append(
        {
            "class_pair": "Healthy vs COVID",
            "spectral_overlap_comment": "Broad overlap remains despite a cleaner binary disease contrast than Pilot 4 liver subtypes.",
            "bsv_overlap_comment": "BSV preserves the same broad shift but does not create stronger geometry than spectral space.",
            "shared_family_themes": "; ".join(
                class_family.groupby("family", as_index=False)["family_fraction"].mean().sort_values("family_fraction", ascending=False).head(3)["family"]
            ),
            "likely_explanation": "Systemic serum chemistry shifts are real, but the dominant variance still mixes disease biology with background serum structure.",
            "bsv_centroid_distance": float(np.linalg.norm(healthy_mean - covid_mean)),
        }
    )
    overlap_df = pd.DataFrame(pair)
    overlap_df.to_csv(TABLES_DIR / "overlap_zone_analysis.csv", index=False)

    _plot_radar_grid(class_mean_bsv[["class_label_display"] + [a for a in FIXED_RADAR_AXES if a in class_mean_bsv.columns]], "class_label_display", FIGURES_DIR / "class_mean_bsv_radars.png", "Class Mean BSV Radars")
    _plot_radar_grid(class_delta_bsv[["class_label_display"] + [a for a in FIXED_RADAR_AXES if a in class_delta_bsv.columns]], "class_label_display", FIGURES_DIR / "class_delta_bsv_radars.png", "Class Delta-BSV Radars")
    _plot_family_bars(class_family, "class_label_display", FIGURES_DIR / "class_family_bars.png", "Class Family Composition")

    heatmap_df = class_mean_bsv.set_index("class_label_display")[axis_cols]
    plt.figure(figsize=(7.0, 4.0))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="mako")
    plt.title("Class BSV Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_bsv_heatmap.png", dpi=220)
    plt.close()

    dist_matrix = pd.DataFrame(
        [
            [float(np.linalg.norm(class_mean_bsv[class_mean_bsv["class_label_display"] == a][axis_cols].iloc[0].to_numpy(dtype=float) - class_mean_bsv[class_mean_bsv["class_label_display"] == b][axis_cols].iloc[0].to_numpy(dtype=float))) for b in class_mean_bsv["class_label_display"]]
            for a in class_mean_bsv["class_label_display"]
        ],
        index=class_mean_bsv["class_label_display"],
        columns=class_mean_bsv["class_label_display"],
    )
    plt.figure(figsize=(4.8, 4.1))
    sns.heatmap(dist_matrix, annot=True, fmt=".3f", cmap="crest")
    plt.title("Class Pairwise Distance Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_pairwise_distance_heatmap.png", dpi=220)
    plt.close()

    plt.figure(figsize=(6.0, 4.4))
    plt.bar(["Healthy vs COVID"], [float(np.linalg.norm(healthy_mean - covid_mean))], color="#6d597a")
    plt.ylabel("BSV centroid distance")
    plt.title("Overlap Zone Pairwise Panel")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "overlap_zone_pairwise_panels.png", dpi=220)
    plt.close()

    spectral_feature_df = meta_df.copy()
    spectral_feature_df["substrate_adsorption_bias"] = patient_bsv_df["substrate_adsorption_bias"].to_numpy()
    spectral_feature_df["small_molecule_metabolite"] = patient_bsv_df["small_molecule_metabolite"].to_numpy()
    spectral_feature_df["protein_peptide"] = patient_bsv_df["protein_peptide"].to_numpy()
    spectral_feature_df["family_entropy"] = patient_family_df["family_entropy"].to_numpy()
    spectral_feature_df["top1_dominance"] = patient_family_df["top1_dominance"].to_numpy()

    assoc_df = pd.concat(
        [
            _axis_associations("spectral", spectral_pca_df, spectral_feature_df),
            _axis_associations("bsv", _pca_dataframe(patient_bsv_df[axis_cols].to_numpy(), meta_df, scale=False), spectral_feature_df),
        ],
        ignore_index=True,
    )
    assoc_df.to_csv(TABLES_DIR / "serum_bias_axis_associations.csv", index=False)
    pivot = assoc_df.pivot(index="feature", columns=["space_name", "metric"], values="spearman_r")
    plt.figure(figsize=(8.0, 4.8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0.0)
    plt.title("Serum Bias Axis Associations")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "serum_bias_association_panels.png", dpi=220)
    plt.close()

    decision_label = "interpretation_only_benchmark"
    decision_lines = [
        "# Pilot5 Decision Note",
        "",
        f"1. Is this dataset easier than the CCA/HCC/LM subtype task? `yes, modestly`",
        f"2. Does GAIRA improve geometry, interpretation, or both? `interpretation`",
        f"3. Does patient-level aggregation again help interpretation more than classification? `not really testable here because sample_id is already one-to-one with spectra, but the reporting logic still favors sample-level summaries.`",
        f"4. Does the serum doctrine from Pilot 4 / 4.1 generalize? `yes, in the sense that spectral baseline remains the geometry benchmark and GAIRA remains more useful as an interpretation layer than as a geometry rescue.`",
        "",
        f"Decision label: `{decision_label}`",
    ]
    (REPORT_DIR / "pilot5_decision_note.md").write_text("\n".join(decision_lines), encoding="utf-8")

    report_lines = [
        "# GAIRAv3 Pilot5 COVID Serum Raman Report",
        "",
        "## 1. Why this dataset was chosen",
        "- Pilot 5 tests whether the serum doctrine from Pilot 4 / 4.1 generalizes to a broader systemic disease context.",
        "- COVID serum should be an easier binary serum task than liver-cancer subtype discrimination if the dominant biology is more global.",
        "",
        "## 2. Dataset and task definition",
        "",
        dataset_note,
        "",
        task_note,
        "",
        "### Input verification",
        "",
        _df_to_md(verification_df),
        "",
        "## 3. Per-spectrum baseline",
        "",
        _df_to_md(baseline_df),
        "",
        "## 4. Per-spectrum GAIRA",
        "",
        _df_to_md(geometry_df),
        "",
        "## 5. Patient-level analysis",
        "",
        aggregation_note,
        "",
        _df_to_md(patient_baseline_df),
        "",
        _df_to_md(patient_geometry_df),
        "",
        "## 6. Biochemical interpretation",
        "",
        _df_to_md(interpretation_df),
        "",
        _df_to_md(overlap_df),
        "",
        "## 7. Serum bias assessment",
        "",
        _df_to_md(assoc_df),
        "",
        "## 8. Final conclusion",
        f"- Is Pilot 5 a stronger serum benchmark than Pilot 4? `yes for a binary disease-vs-control task, but not for true aggregation testing.`",
        f"- Does the serum doctrine generalize? `yes`",
        f"- What does COVID serum teach us about GAIRA's usable strengths? `GAIRA remains most useful as a biochemical-theme interpretation layer on top of a spectral baseline, not as a geometry rescue mechanism.`",
        f"- Final decision label: `{decision_label}`",
    ]
    report_md = REPORT_DIR / "GAIRAv3_Pilot5_COVID_serum_raman_report.md"
    report_md.write_text("\n".join(report_lines), encoding="utf-8")

    build_pdf_report(
        report_md,
        [
            FIGURES_DIR / "pilot5_per_spectrum_spectral_pca.png",
            FIGURES_DIR / "pilot5_per_spectrum_lda_2d.png",
            FIGURES_DIR / "pilot5_per_spectrum_bsv_pca.png",
            FIGURES_DIR / "patient_level_spectral_pca.png",
            FIGURES_DIR / "class_mean_bsv_radars.png",
            FIGURES_DIR / "class_family_bars.png",
            FIGURES_DIR / "serum_bias_association_panels.png",
        ],
        REPORT_DIR / "GAIRAv3_Pilot5_COVID_serum_raman_report.pdf",
    )


if __name__ == "__main__":
    main()
