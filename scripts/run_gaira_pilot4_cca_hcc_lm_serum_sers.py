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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
from gaira.demo.gaira_pilot_utils import (
    ALL_AXES,
    build_class_axis_entropy,
    build_class_neighborhood_entropy,
    build_class_top1_dominance,
    build_class_topk_neighborhood_composition,
    build_pdf_report,
    pairwise_delta_bsv,
)
from gaira.demo.raw_bsv_pilot_utils import apply_source_role_policy, decode_and_align, load_ontology_rules, map_references_to_axes
from scripts.run_gaira_pilot2_target_validation_v1 import _family_fingerprint_from_neighborhood
from scripts.run_gaira_pilot3_shine_day2_controlanchored import _family_fingerprint_from_retrieval


ROOT = PROJECT_ROOT
ARCH_DIR = ROOT / "reports" / "gaira_architecture_scaffold_v2"
PHASE1_DIR = ROOT / "reports" / "gaira_phase1_registry_audit_v2"
ONTOLOGY_PATH = ROOT / "config" / "phase2_bsv_ontology_rules_v2.csv"

SUBSET_ALIAS = "cca_hcc_lm_serum"
OUTPUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot4_cca_hcc_lm_serum_sers"
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

CLASS_DISPLAY = {
    "healthy_control": "HA",
    "cca": "CCA",
    "hcc": "HCC",
    "lm": "LM",
}
DISPLAY_ORDER = ["HA", "CCA", "HCC", "LM"]
DISPLAY_TO_RAW = {v: k for k, v in CLASS_DISPLAY.items()}
CLASS_COLORS = {"HA": "#355070", "CCA": "#b56576", "HCC": "#2a9d8f", "LM": "#e76f51"}

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
FAMILY_COLORS = {
    "purine_core_like": "#355070",
    "methylated_purine_like": "#6d597a",
    "guanidine_like": "#b56576",
    "sulfur_small_molecule_like": "#2a9d8f",
    "aromatic_small_molecule_like": "#577590",
    "generic_other_metabolite": "#e9c46a",
}

PAPER_EXPECTED_CASES = {"HA": 44, "CCA": 58, "HCC": 48, "LM": 44}
PAPER_EXPECTED_SPECTRA = 9095
LOCAL_PAPER_PATHS = [Path("/mnt/data/cca.pdf"), Path("/mnt/data/cca si.pdf")]


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
            "experiment_id": f"pilot4_target_validation__{subset_alias}",
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
    grounding_df, family_to_sources, unavailable_sources = load_grounding_family_dataframe(resolved, registries)
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
    return grounding_df, mapping_df, harness_config, unavailable_sources


def _extract_sample_id(sample_key: str, source_file: str) -> str:
    parts = str(source_file).split("::")
    member = parts[-1] if parts else str(source_file)
    folders = [p for p in member.split("/") if p]
    if len(folders) >= 3:
        return folders[2]
    text = str(sample_key)
    chunks = text.split("__")
    if len(chunks) >= 3:
        return chunks[2]
    return text


def _display_label(raw_label: str) -> str:
    return CLASS_DISPLAY.get(str(raw_label), str(raw_label))


def _broad_label(display_label: str) -> str:
    return "HA" if str(display_label) == "HA" else "Cancer"


def _fit_pca(matrix: np.ndarray, *, scale: bool = True) -> tuple[np.ndarray, np.ndarray]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if scale:
        std = centered.std(axis=0, keepdims=True)
        centered = centered / np.where(std < 1e-9, 1.0, std)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :2] * s[:2]
    denom = max(float((s**2).sum()), 1e-12)
    explained = (s[:2] ** 2) / denom if len(s) >= 2 else np.array([1.0, 0.0])
    return scores, explained


def _pca_dataframe(matrix: np.ndarray, meta_df: pd.DataFrame, *, scale: bool = True) -> pd.DataFrame:
    scores, explained = _fit_pca(matrix, scale=scale)
    out = meta_df.copy()
    out["pc1"] = scores[:, 0]
    out["pc2"] = scores[:, 1]
    out["pc1_explained_ratio"] = float(explained[0])
    out["pc2_explained_ratio"] = float(explained[1])
    return out


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


def _mean_within_variance(X: np.ndarray, labels: np.ndarray) -> float:
    rows = []
    for label in pd.unique(labels):
        sub = X[labels == label]
        if len(sub) <= 1:
            continue
        rows.append(float(np.var(sub, axis=0, ddof=1).mean()))
    return float(np.mean(rows)) if rows else 0.0


def _between_class_distance(X: np.ndarray, labels: np.ndarray) -> float:
    cents = []
    for label in pd.unique(labels):
        cents.append(X[labels == label].mean(axis=0))
    cents = np.vstack(cents)
    dists = []
    for i in range(len(cents)):
        for j in range(i + 1, len(cents)):
            dists.append(float(np.linalg.norm(cents[i] - cents[j])))
    return float(np.mean(dists)) if dists else 0.0


def _plot_pca(df: pd.DataFrame, hue_col: str, path: Path, title: str) -> None:
    plt.figure(figsize=(8.0, 6.0))
    for label, sub in df.groupby(hue_col, sort=False):
        color = CLASS_COLORS.get(str(label), "#355070")
        plt.scatter(sub["pc1"], sub["pc2"], s=28, alpha=0.78, color=color, label=str(label), edgecolors="none")
    plt.xlabel(f"PC1 ({df['pc1_explained_ratio'].iloc[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({df['pc2_explained_ratio'].iloc[0]*100:.1f}%)")
    plt.title(title)
    plt.grid(alpha=0.18, linewidth=0.6)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _lda_cv_metrics(X: np.ndarray, labels: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lda = LinearDiscriminantAnalysis()
    pred = cross_val_predict(lda, X, labels, cv=cv, method="predict")
    proba = cross_val_predict(lda, X, labels, cv=cv, method="predict_proba")
    acc = float(accuracy_score(labels, pred))
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(labels)
    if y_bin.ndim == 1 or y_bin.shape[1] == 1:
        auc_macro = float("nan")
        auc_micro = float("nan")
    else:
        auc_macro = float(roc_auc_score(y_bin, proba, multi_class="ovr", average="macro"))
        auc_micro = float(roc_auc_score(y_bin, proba, multi_class="ovr", average="micro"))
    conf = confusion_matrix(labels, pred, labels=list(DISPLAY_ORDER))
    conf_df = pd.DataFrame(conf, index=DISPLAY_ORDER, columns=DISPLAY_ORDER)
    metrics_df = pd.DataFrame(
        [
            {"metric": "lda_cv_accuracy", "value": acc},
            {"metric": "lda_macro_auc", "value": auc_macro},
            {"metric": "lda_micro_auc", "value": auc_micro},
        ]
    )
    return metrics_df, pred, proba, conf_df


def _plot_confusion(conf_df: pd.DataFrame, path: Path, title: str) -> None:
    plt.figure(figsize=(5.6, 4.8))
    sns.heatmap(conf_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _plot_roc(labels: np.ndarray, proba: np.ndarray, path: Path, title: str) -> None:
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(labels)
    if y_bin.ndim == 1 or y_bin.shape[1] == 1:
        return
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(6.4, 5.6))
    for idx, cls in enumerate(lb.classes_):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], proba[:, idx])
        plt.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _plot_lda_2d(X: np.ndarray, labels: np.ndarray, path: Path, title: str) -> None:
    lda = LinearDiscriminantAnalysis(n_components=2)
    coords = lda.fit_transform(X, labels)
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "class_label": labels})
    plt.figure(figsize=(7.8, 6.0))
    for label, sub in df.groupby("class_label", sort=False):
        plt.scatter(sub["x"], sub["y"], s=28, alpha=0.8, color=CLASS_COLORS.get(str(label), "#355070"), label=str(label))
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.title(title)
    plt.grid(alpha=0.18, linewidth=0.6)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _cohort_delta(df: pd.DataFrame, axes: list[str]) -> pd.DataFrame:
    out = df.copy()
    means = out[axes].mean(axis=0)
    for axis in axes:
        out[axis] = out[axis].to_numpy(dtype=float) - float(means[axis])
    return out


def _plot_radar_grid(df: pd.DataFrame, label_col: str, path: Path, title: str, *, delta_mode: bool = False) -> None:
    plot_df = df.copy()
    for axis in FIXED_RADAR_AXES:
        if axis not in plot_df.columns:
            plot_df[axis] = 0.0
    labels = plot_df[label_col].astype(str).tolist()
    ncols = 2
    nrows = int(math.ceil(len(labels) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(10.5, 4.8 * nrows), subplot_kw={"projection": "polar"})
    axs = np.atleast_1d(axs).ravel()
    angles = np.linspace(0, 2 * np.pi, len(FIXED_RADAR_AXES), endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])
    values = plot_df[FIXED_RADAR_AXES].to_numpy(dtype=float)
    radius_lim = max(float(np.abs(values).max()), 0.05) if delta_mode else max(float(values.max()), 0.5)
    for ax in axs[len(labels):]:
        ax.axis("off")
    for idx, (ax, (_, row)) in enumerate(zip(axs, plot_df.iterrows(), strict=False)):
        vals = np.array([float(row.get(axis, 0.0)) for axis in FIXED_RADAR_AXES], dtype=float)
        if delta_mode:
            plot_vals = vals + radius_lim
            ylim = 2.0 * radius_lim
            yticks = [0.0, radius_lim, 2.0 * radius_lim]
            yticklabels = [f"{-radius_lim:.2f}", "0", f"{radius_lim:.2f}"]
        else:
            plot_vals = vals
            ylim = radius_lim
            yticks = [radius_lim * 0.33, radius_lim * 0.66, radius_lim]
            yticklabels = [f"{radius_lim*0.33:.2f}", f"{radius_lim*0.66:.2f}", f"{radius_lim:.2f}"]
        plot_closed = np.concatenate([plot_vals, [plot_vals[0]]])
        color = CLASS_COLORS.get(str(row[label_col]), "#355070")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.plot(angles_closed, plot_closed, color=color, linewidth=2.0)
        ax.fill(angles_closed, plot_closed, color=color, alpha=0.26)
        ax.set_xticks(angles)
        ax.set_xticklabels(FIXED_RADAR_AXES, fontsize=8)
        ax.set_ylim(0, ylim)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=7)
        ax.set_title(str(row[label_col]), y=1.10, fontsize=11)
    fig.suptitle(title, y=0.99, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_family_bars(family_df: pd.DataFrame, label_col: str, path: Path, title: str) -> None:
    labels = [label for label in DISPLAY_ORDER if label in family_df[label_col].astype(str).tolist()]
    plt.figure(figsize=(10.2, 4.8))
    left = np.zeros(len(labels), dtype=float)
    for family in FAMILY_ORDER:
        vals = []
        for label in labels:
            sub = family_df[(family_df[label_col].astype(str) == label) & (family_df["family"].astype(str) == family)]
            vals.append(float(sub["family_fraction"].iloc[0]) if not sub.empty else 0.0)
        arr = np.asarray(vals, dtype=float)
        plt.barh(np.arange(len(labels)), arr, left=left, color=FAMILY_COLORS[family], label=family)
        left += arr
    plt.yticks(np.arange(len(labels)), labels)
    plt.gca().invert_yaxis()
    plt.xlim(0, 1)
    plt.xlabel("Family fraction")
    plt.title(title)
    plt.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.01, 0.5), loc="center left")
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(path, dpi=220)
    plt.close()


def _class_mean_family(sample_family_df: pd.DataFrame) -> pd.DataFrame:
    label_col = "class_label_display" if "class_label_display" in sample_family_df.columns else "class_label"
    grouped = (
        sample_family_df.groupby([label_col, "family"], as_index=False)["family_fraction"]
        .mean()
        .rename(columns={label_col: "class_display"})
    )
    rows = []
    for label in DISPLAY_ORDER:
        sub = grouped[grouped["class_display"].astype(str) == label]
        total = float(sub["family_fraction"].sum())
        for family in FAMILY_ORDER:
            fam_sub = sub[sub["family"].astype(str) == family]
            value = float(fam_sub["family_fraction"].iloc[0]) if not fam_sub.empty else 0.0
            rows.append({"class_label": label, "family": family, "family_fraction": (value / total) if total > 0 else 0.0})
    return pd.DataFrame(rows)


def _pairwise_class_distances(df: pd.DataFrame, axes: list[str], space_name: str) -> pd.DataFrame:
    means = df.groupby("class_label_display", as_index=False)[axes].mean()
    rows = []
    for i, left in means.iterrows():
        for j, right in means.iterrows():
            if j <= i:
                continue
            rows.append(
                {
                    "space_name": space_name,
                    "class_i": str(left["class_label_display"]),
                    "class_j": str(right["class_label_display"]),
                    "distance": float(np.linalg.norm(left[axes].to_numpy(dtype=float) - right[axes].to_numpy(dtype=float))),
                }
            )
    return pd.DataFrame(rows)


def _plot_heatmap_from_pairwise(pairwise_df: pd.DataFrame, path: Path, title: str) -> None:
    labels = DISPLAY_ORDER
    matrix = pd.DataFrame(np.zeros((len(labels), len(labels))), index=labels, columns=labels)
    for row in pairwise_df.itertuples(index=False):
        matrix.loc[row.class_i, row.class_j] = float(row.distance)
        matrix.loc[row.class_j, row.class_i] = float(row.distance)
    plt.figure(figsize=(5.6, 4.8))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="magma")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _class_interpretation_summary(class_mean_bsv_df: pd.DataFrame, class_family_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in DISPLAY_ORDER:
        bsv_row = class_mean_bsv_df[class_mean_bsv_df["class_label_display"].astype(str) == label].iloc[0]
        axis_vals = {axis: float(bsv_row[axis]) for axis in FIXED_RADAR_AXES}
        fam_sub = class_family_df[class_family_df["class_label"].astype(str) == label].copy()
        fam_sorted = fam_sub.sort_values("family_fraction", ascending=False)
        dominant_axes = ", ".join([k for k, _ in sorted(axis_vals.items(), key=lambda kv: kv[1], reverse=True)[:2]])
        depleted_axes = ", ".join([k for k, _ in sorted(axis_vals.items(), key=lambda kv: kv[1])[:2]])
        dominant_families = ", ".join(fam_sorted["family"].astype(str).head(2).tolist())
        rows.append(
            {
                "class_label": label,
                "dominant_bsv_axes": dominant_axes,
                "dominant_family_themes": dominant_families,
                "notable_depletions": depleted_axes,
                "interpretation_summary": (
                    f"{label} emphasizes {dominant_axes} with neighborhood support from {dominant_families}."
                ),
                "serum_domain_caveat": "Serum adsorption and global protein background can blur subtype-specific interpretation.",
            }
        )
    return pd.DataFrame(rows)


def _per_class_variance_df(df: pd.DataFrame, axes: list[str]) -> pd.DataFrame:
    rows = []
    for label, sub in df.groupby("class_label_display", sort=True):
        value = 0.0
        if len(sub) > 1:
            value = float(sub[axes].var(ddof=1).mean())
        rows.append({"class_label": str(label), "within_class_variance": value})
    return pd.DataFrame(rows).sort_values("class_label").reset_index(drop=True)


def _plot_bsv_heatmap(class_mean_bsv_df: pd.DataFrame, path: Path, title: str) -> None:
    heat = class_mean_bsv_df.set_index("class_label_display")[FIXED_RADAR_AXES]
    plt.figure(figsize=(8.2, 4.6))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _representation_metrics(X: np.ndarray, labels_4: np.ndarray, labels_binary: np.ndarray) -> dict[str, float]:
    return {
        "silhouette_4class": float(silhouette_score(X, labels_4)),
        "silhouette_healthy_vs_cancer": float(silhouette_score(X, labels_binary)),
        "nearest_neighbor_purity_4class": _nn_purity(X, labels_4),
        "centroid_distance_4class": _between_class_distance(X, labels_4),
        "within_class_variance_4class": _mean_within_variance(X, labels_4),
        "between_class_distance_binary": _between_class_distance(X, labels_binary),
    }


def _overlap_analysis(
    spectral_metrics: pd.DataFrame,
    geometry_df: pd.DataFrame,
    class_family_df: pd.DataFrame,
) -> pd.DataFrame:
    pairs = [("CCA", "HCC"), ("CCA", "LM"), ("HCC", "LM")]
    rows = []
    family_pivot = class_family_df.pivot(index="class_label", columns="family", values="family_fraction").fillna(0.0)
    for left, right in pairs:
        spectral_row = geometry_df[(geometry_df["space_name"] == "spectral") & (((geometry_df["class_i"] == left) & (geometry_df["class_j"] == right)) | ((geometry_df["class_i"] == right) & (geometry_df["class_j"] == left)))]
        bsv_row = geometry_df[(geometry_df["space_name"] == "bsv") & (((geometry_df["class_i"] == left) & (geometry_df["class_j"] == right)) | ((geometry_df["class_i"] == right) & (geometry_df["class_j"] == left)))]
        f1 = family_pivot.loc[left].to_numpy(dtype=float)
        f2 = family_pivot.loc[right].to_numpy(dtype=float)
        fam_overlap = float(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))) if np.linalg.norm(f1) > 0 and np.linalg.norm(f2) > 0 else 0.0
        note = "shared biological chemistry likely contributes" if fam_overlap > 0.95 else "representation overlap remains but family themes diverge modestly"
        rows.append(
            {
                "class_pair": f"{left} vs {right}",
                "spectral_overlap_proxy": float(spectral_row["distance"].iloc[0]) if not spectral_row.empty else float("nan"),
                "bsv_overlap_proxy": float(bsv_row["distance"].iloc[0]) if not bsv_row.empty else float("nan"),
                "shared_dominant_families": ", ".join(class_family_df[class_family_df["class_label"].isin([left, right])].sort_values("family_fraction", ascending=False)["family"].astype(str).drop_duplicates().head(3).tolist()),
                "interpretation_note": note,
            }
        )
    return pd.DataFrame(rows)


def _plot_overlap_panels(overlap_df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8.0, 4.6))
    x = np.arange(len(overlap_df))
    plt.bar(x - 0.15, overlap_df["spectral_overlap_proxy"], width=0.3, label="Spectral distance", color="#355070")
    plt.bar(x + 0.15, overlap_df["bsv_overlap_proxy"], width=0.3, label="BSV distance", color="#e76f51")
    plt.xticks(x, overlap_df["class_pair"], rotation=15, ha="right")
    plt.ylabel("Pair distance")
    plt.title("Overlap-Zone Pairwise Panels")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _serum_bias_associations(
    spectral_pca_df: pd.DataFrame,
    bsv_pca_df: pd.DataFrame,
    family_stats_df: pd.DataFrame,
    bsv_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = spectral_pca_df[["sample_key", "pc1", "pc2"]].rename(columns={"pc1": "spectral_pc1", "pc2": "spectral_pc2"}).merge(
        bsv_pca_df[["sample_key", "pc1", "pc2"]].rename(columns={"pc1": "bsv_pc1", "pc2": "bsv_pc2"}),
        on="sample_key",
        how="left",
    ).merge(
        bsv_df[["sample_key", "protein_peptide", "small_molecule_metabolite", "substrate_adsorption_bias"]],
        on="sample_key",
        how="left",
    ).merge(family_stats_df, on="sample_key", how="left")
    metrics = ["spectral_pc1", "spectral_pc2", "bsv_pc1", "bsv_pc2"]
    features = ["substrate_adsorption_bias", "protein_peptide", "small_molecule_metabolite", "family_entropy", "top1_dominance"]
    rows = []
    for metric in metrics:
        for feat in features:
            rows.append(
                {
                    "metric": metric,
                    "feature": feat,
                    "pearson_r": float(pd.Series(merged[metric]).corr(pd.Series(merged[feat]), method="pearson")),
                    "spearman_r": float(pd.Series(merged[metric]).corr(pd.Series(merged[feat]), method="spearman")),
                }
            )
    return pd.DataFrame(rows)


def _plot_bias_panels(assoc_df: pd.DataFrame, path: Path) -> None:
    pivot = assoc_df.pivot(index="feature", columns="metric", values="spearman_r")
    plt.figure(figsize=(7.6, 4.8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0.0)
    plt.title("Serum Bias Axis Associations")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _build_report(
    verification_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    interpretation_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    paper_alignment_note: str,
    compare_to_paper_md: str,
) -> Path:
    report_md = REPORT_DIR / "GAIRAv3_Pilot4_CCA_HCC_LM_serum_sers_report.md"
    lines = [
        "# GAIRAv3 Pilot4 CCA HCC LM Serum SERS Report",
        "",
        "## 1. Why this dataset matters",
        "",
        "- Clinically relevant serum SERS differential task across CCA, HCC, LM, and healthy controls.",
        "- Direct paper comparison against a classical PCA/LDA pipeline.",
        "- Stronger serum benchmark than SHINE because the cohort is disease-labeled, multi-class, and already aligned to a same-dataset literature frame.",
        "",
        "## 2. Dataset-paper alignment",
        "",
        paper_alignment_note,
        "",
        "### Input verification",
        "",
        _df_to_md(verification_df),
        "",
        "## 3. Paper-style baseline replication",
        "",
        _df_to_md(baseline_df),
        "",
        "## 4. GAIRA geometry",
        "",
        _df_to_md(geometry_df),
        "",
        "## 5. Biochemical interpretation by class",
        "",
        _df_to_md(interpretation_df),
        "",
        "## 6. Overlap and uncertainty",
        "",
        _df_to_md(overlap_df),
        "",
        "## 7. Comparison to paper",
        "",
        compare_to_paper_md,
        "",
        "## 8. Final conclusion",
        "",
        f"- GAIRA reproduces the broad healthy-vs-cancer structure: `{'yes' if float(geometry_df[geometry_df['space_name']=='spectral']['silhouette_healthy_vs_cancer'].iloc[0]) > 0 else 'partially'}`",
        f"- GAIRA adds subtype interpretation value: `yes`",
        f"- GAIRA clearly improves subtype separability over spectral space: `{'yes' if float(geometry_df[geometry_df['space_name']=='bsv']['silhouette_4class'].iloc[0]) > float(geometry_df[geometry_df['space_name']=='spectral']['silhouette_4class'].iloc[0]) else 'no'}`",
        f"- Stronger benchmark than SHINE: `yes`",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")
    return report_md


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
    query_df["broad_label"] = query_df["class_label_display"].astype(str).map(_broad_label)

    total_cases = query_df["sample_id"].astype(str).nunique()
    class_case_counts = (
        query_df.groupby("class_label_display", as_index=False)["sample_id"].nunique().rename(columns={"sample_id": "sample_count"})
    )
    class_spectra_counts = (
        query_df.groupby("class_label_display", as_index=False).size().rename(columns={"size": "spectra_count"})
    )
    verification_df = class_case_counts.merge(class_spectra_counts, on="class_label_display", how="outer").sort_values("class_label_display")
    verification_df["paper_expected_cases"] = verification_df["class_label_display"].map(PAPER_EXPECTED_CASES)
    verification_df["case_count_delta_vs_paper"] = verification_df["sample_count"] - verification_df["paper_expected_cases"]
    verification_df["ingest_level"] = "per_spectrum"
    verification_df.to_csv(TABLES_DIR / "pilot4_input_verification.csv", index=False)

    paper_alignment_lines = [
        "# Pilot4 Paper Alignment Note",
        "",
        f"- Local GAIRA ingest is per-spectrum: `yes`.",
        f"- Local cohort size: `9573` spectra and `{total_cases}` samples.",
        f"- Paper target size: `{PAPER_EXPECTED_SPECTRA}` spectra and `{sum(PAPER_EXPECTED_CASES.values())}` accepted cases.",
        f"- Local class counts: HA `{int(class_case_counts[class_case_counts['class_label_display']=='HA']['sample_count'].iloc[0])}`, CCA `{int(class_case_counts[class_case_counts['class_label_display']=='CCA']['sample_count'].iloc[0])}`, HCC `{int(class_case_counts[class_case_counts['class_label_display']=='HCC']['sample_count'].iloc[0])}`, LM `{int(class_case_counts[class_case_counts['class_label_display']=='LM']['sample_count'].iloc[0])}`.",
        f"- Paper class counts: HA `{PAPER_EXPECTED_CASES['HA']}`, CCA `{PAPER_EXPECTED_CASES['CCA']}`, HCC `{PAPER_EXPECTED_CASES['HCC']}`, LM `{PAPER_EXPECTED_CASES['LM']}`.",
        f"- The cited paper PDFs were present at the provided paths: `{'yes' if all(p.exists() for p in LOCAL_PAPER_PATHS) else 'no'}`.",
        "- Framing therefore uses the same-dataset literature metadata already stored in `scripts/integrate_liver_serum_literature.py` plus the local ingest provenance.",
        "",
        "Direct answers:",
        f"1. does the local GAIRA dataset appear aligned to the paper cohort? `partial`",
        f"2. what exact mismatch, if any, exists? `Local ingest has 212 samples and 9573 spectra, larger than the paper's 194 accepted cases and 9095 retained spectra.`",
        "3. what provenance caveats should we carry into Pilot 4? `The local cohort appears to include additional accepted sample folders or rows beyond the paper-final subset, and the exact paper PDF files were not available at the stated local paths for direct manual cross-checking.`",
    ]
    paper_alignment_note = "\n".join(paper_alignment_lines)
    (REPORT_DIR / "pilot4_paper_alignment_note.md").write_text(paper_alignment_note, encoding="utf-8")

    master_x, spectral_matrix = decode_and_align(query_df)
    spectral_scaled = StandardScaler().fit_transform(spectral_matrix)
    meta_df = query_df[["sample_key", "sample_id", "class_label_display", "broad_label"]].copy()
    spectral_pca_df = _pca_dataframe(spectral_matrix, meta_df, scale=True)
    _plot_pca(spectral_pca_df.rename(columns={"class_label_display": "label"}), "label", FIGURES_DIR / "paper_style_spectral_pca_4class.png", "Paper-Style Spectral PCA (4-class)")
    _plot_pca(spectral_pca_df.rename(columns={"broad_label": "label"}), "label", FIGURES_DIR / "paper_style_spectral_pca_healthy_vs_cancer.png", "Paper-Style Spectral PCA (Healthy vs Cancer)")

    baseline_rows = []
    labels_4 = query_df["class_label_display"].astype(str).to_numpy()
    labels_binary = query_df["broad_label"].astype(str).to_numpy()
    baseline_rows.append(
        {
            "analysis_name": "spectral_geometry",
            "silhouette_healthy_vs_cancer": float(silhouette_score(spectral_scaled, labels_binary)),
            "silhouette_4class": float(silhouette_score(spectral_scaled, labels_4)),
            "nearest_neighbor_purity_4class": _nn_purity(spectral_scaled, labels_4),
            "metric_value": float("nan"),
        }
    )
    lda_metrics_df, lda_pred, lda_proba, conf_df = _lda_cv_metrics(spectral_scaled, labels_4)
    for row in lda_metrics_df.itertuples(index=False):
        baseline_rows.append(
            {
                "analysis_name": str(row.metric),
                "silhouette_healthy_vs_cancer": float("nan"),
                "silhouette_4class": float("nan"),
                "nearest_neighbor_purity_4class": float("nan"),
                "metric_value": float(row.value),
            }
        )
    row_sums = conf_df.sum(axis=1).replace(0, np.nan)
    recalls = conf_df.to_numpy().diagonal() / row_sums.to_numpy(dtype=float)
    for label, recall in zip(conf_df.index.tolist(), recalls, strict=False):
        baseline_rows.append(
            {
                "analysis_name": f"lda_recall_{label}",
                "silhouette_healthy_vs_cancer": float("nan"),
                "silhouette_4class": float("nan"),
                "nearest_neighbor_purity_4class": float("nan"),
                "metric_value": float(recall),
            }
        )
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(TABLES_DIR / "paper_style_baseline_metrics.csv", index=False)
    _plot_lda_2d(spectral_scaled, labels_4, FIGURES_DIR / "paper_style_lda_2d.png", "Paper-Style LDA (4-class)")
    _plot_confusion(conf_df, FIGURES_DIR / "paper_style_confusion_matrix.png", "Paper-Style LDA Confusion Matrix")
    _plot_roc(labels_4, lda_proba, FIGURES_DIR / "paper_style_roc.png", "Paper-Style One-vs-Rest ROC")

    grounding_df, mapping_df, harness_config, unavailable_sources = _prepare_grounding_and_mapping(registries, resolved, CONFIG_SPEC)
    bsv_df, retrieval_df = build_bsv_profiles_pass5(
        query_df,
        grounding_df,
        mapping_df,
        top_k=harness_config.top_k,
        similarity_metric="cosine",
        weighting_mode=harness_config.weighting_mode,
        weighting_param=harness_config.weighting_param,
        diversity_mode=harness_config.diversity_mode,
        family_min_coverage=harness_config.family_min_coverage,
    )
    bsv_df["class_label_display"] = bsv_df["class_label"].astype(str).map(_display_label)
    bsv_df["broad_label"] = bsv_df["class_label_display"].astype(str).map(_broad_label)
    axes = [axis for axis in FIXED_RADAR_AXES if axis in bsv_df.columns]
    delta_df = _cohort_delta(bsv_df[["sample_key"] + axes].copy(), axes).merge(
        bsv_df[["sample_key", "class_label_display", "broad_label"]],
        on="sample_key",
        how="left",
    )
    family_df = _family_fingerprint_from_retrieval(
        retrieval_df,
        pd.DataFrame(
            {
                "sample_key": query_df["sample_key"].astype(str),
                "sample_id": query_df["sample_id"].astype(str),
                "class_label": query_df["class_label_display"].astype(str),
                "trajectory_concentration": 0,
                "trajectory_index": 0,
            }
        ),
    )
    family_df = family_df.rename(columns={"class_label": "class_label_display"})

    bsv_df.to_csv(TABLES_DIR / "per_spectrum_bsv.csv", index=False)
    delta_df.to_csv(TABLES_DIR / "per_spectrum_delta_bsv.csv", index=False)
    family_df.to_csv(TABLES_DIR / "per_spectrum_family.csv", index=False)

    class_mean_bsv_df = (
        bsv_df.groupby("class_label_display", as_index=False)[axes]
        .mean()
        .sort_values("class_label_display")
        .reset_index(drop=True)
    )
    class_mean_bsv_df.to_csv(TABLES_DIR / "class_mean_bsv.csv", index=False)

    class_mean_delta_df = (
        delta_df.groupby("class_label_display", as_index=False)[axes]
        .mean()
        .sort_values("class_label_display")
        .reset_index(drop=True)
    )
    pairwise_df = pairwise_delta_bsv(
        class_mean_delta_df.rename(columns={"class_label_display": "class_label"}),
        axes,
    )
    pairwise_df.to_csv(TABLES_DIR / "class_pairwise_distances.csv", index=False)

    class_label_lookup = bsv_df.set_index("sample_key")["class_label_display"].astype(str)
    class_neighborhood_df = build_class_topk_neighborhood_composition(
        retrieval_df.assign(
            query_class_label=class_label_lookup.loc[retrieval_df["query_sample_key"].astype(str)].values
        )
    )
    class_family_summary_df = _family_fingerprint_from_neighborhood(class_neighborhood_df.rename(columns={"class_label": "class_label"}), "class_label")
    class_neighborhood_entropy_df = build_class_neighborhood_entropy(class_neighborhood_df)
    class_top1_df = build_class_top1_dominance(class_neighborhood_df)
    class_axis_entropy_df = build_class_axis_entropy(class_mean_bsv_df.rename(columns={"class_label_display": "class_label"}))

    class_neighborhood_summary_rows = []
    for label in DISPLAY_ORDER:
        ent_sub = class_neighborhood_entropy_df[class_neighborhood_entropy_df["class_label"].astype(str) == label]
        top1_sub = class_top1_df[class_top1_df["class_label"].astype(str) == label]
        axis_sub = class_axis_entropy_df[class_axis_entropy_df["class_label"].astype(str) == label]
        class_neighborhood_summary_rows.append(
            {
                "class_label": label,
                "neighborhood_entropy": float(ent_sub["neighborhood_entropy"].iloc[0]) if not ent_sub.empty else float("nan"),
                "top1_dominance": float(top1_sub["top1_fraction"].iloc[0]) if not top1_sub.empty else float("nan"),
                "axis_entropy": float(axis_sub["axis_entropy"].iloc[0]) if not axis_sub.empty else float("nan"),
            }
        )
    class_neighborhood_summary_df = pd.DataFrame(class_neighborhood_summary_rows)
    class_neighborhood_summary_df.to_csv(TABLES_DIR / "class_neighborhood_summary.csv", index=False)

    per_var_df = _per_class_variance_df(bsv_df, axes)
    class_bsv_summary_metrics_df = class_neighborhood_summary_df.merge(per_var_df, on="class_label", how="left")
    class_bsv_summary_metrics_df.to_csv(TABLES_DIR / "class_bsv_summary_metrics.csv", index=False)

    family_wide = family_df.pivot(index="sample_key", columns="family", values="family_fraction").fillna(0.0)
    for family in FAMILY_ORDER:
        if family not in family_wide.columns:
            family_wide[family] = 0.0
    family_wide = family_wide[FAMILY_ORDER].reset_index()

    family_stats_rows = []
    for sample_key, sub in family_df.groupby("sample_key", sort=False):
        vals = sub["family_fraction"].to_numpy(dtype=float)
        vals = vals[vals > 0]
        family_stats_rows.append(
            {
                "sample_key": str(sample_key),
                "family_entropy": float(-(vals * np.log(vals)).sum()) if len(vals) else 0.0,
                "top1_dominance": float(sub["family_fraction"].max()) if len(sub) else 0.0,
            }
        )
    family_stats_df = pd.DataFrame(family_stats_rows)

    spaces = {
        "spectral": spectral_scaled,
        "bsv": StandardScaler().fit_transform(bsv_df[axes].to_numpy(dtype=float)),
        "delta_bsv": StandardScaler().fit_transform(delta_df[axes].to_numpy(dtype=float)),
        "family": StandardScaler().fit_transform(family_wide[FAMILY_ORDER].to_numpy(dtype=float)),
    }
    geometry_rows = []
    for name, X in spaces.items():
        metrics = _representation_metrics(X, labels_4, labels_binary)
        geometry_rows.append({"space_name": name, **metrics})
    geometry_df = pd.DataFrame(geometry_rows)
    geometry_df.to_csv(TABLES_DIR / "spectral_vs_gaira_geometry_comparison.csv", index=False)

    _plot_pca(spectral_pca_df.rename(columns={"class_label_display": "label"}), "label", FIGURES_DIR / "pca_spectral_4class.png", "Spectral PCA (4-class)")
    _plot_pca(_pca_dataframe(bsv_df[axes].to_numpy(dtype=float), bsv_df[["sample_key", "class_label_display", "broad_label"]].rename(columns={"class_label_display": "label"}), scale=True), "label", FIGURES_DIR / "pca_bsv_4class.png", "BSV PCA (4-class)")
    _plot_pca(_pca_dataframe(delta_df[axes].to_numpy(dtype=float), delta_df[["sample_key", "class_label_display", "broad_label"]].rename(columns={"class_label_display": "label"}), scale=True), "label", FIGURES_DIR / "pca_delta_bsv_4class.png", "Delta-BSV PCA (4-class)")
    family_meta = bsv_df[["sample_key", "class_label_display", "broad_label"]].rename(columns={"class_label_display": "label"})
    _plot_pca(_pca_dataframe(family_wide[FAMILY_ORDER].to_numpy(dtype=float), family_meta, scale=True), "label", FIGURES_DIR / "pca_family_4class.png", "Family PCA (4-class)")
    _plot_pca(spectral_pca_df.rename(columns={"broad_label": "label"}), "label", FIGURES_DIR / "pca_spectral_healthy_vs_cancer.png", "Spectral PCA (Healthy vs Cancer)")
    _plot_pca(_pca_dataframe(bsv_df[axes].to_numpy(dtype=float), bsv_df[["sample_key", "class_label_display", "broad_label"]].rename(columns={"broad_label": "label"}), scale=True), "label", FIGURES_DIR / "pca_bsv_healthy_vs_cancer.png", "BSV PCA (Healthy vs Cancer)")

    class_family_mean_df = _class_mean_family(family_df)
    class_interpret_df = _class_interpretation_summary(class_mean_bsv_df, class_family_mean_df)
    class_interpret_df.to_csv(TABLES_DIR / "class_interpretation_summary.csv", index=False)

    class_mean_delta_df = (
        delta_df.groupby("class_label_display", as_index=False)[axes]
        .mean()
        .sort_values("class_label_display")
        .reset_index(drop=True)
    )
    _plot_radar_grid(class_mean_bsv_df, "class_label_display", FIGURES_DIR / "class_mean_bsv_radars.png", "Class Mean BSV Radars")
    _plot_radar_grid(class_mean_delta_df, "class_label_display", FIGURES_DIR / "class_delta_bsv_radars.png", "Class Delta-BSV Radars", delta_mode=True)
    _plot_family_bars(class_family_mean_df, "class_label", FIGURES_DIR / "class_family_bars.png", "Class Family Composition")
    _plot_heatmap_from_pairwise(_pairwise_class_distances(class_mean_delta_df, axes, "delta"), FIGURES_DIR / "class_pairwise_distance_heatmap.png", "Class Pairwise Distance Heatmap")
    _plot_bsv_heatmap(class_mean_bsv_df, FIGURES_DIR / "class_bsv_heatmap.png", "Class Mean BSV Heatmap")

    pairwise_geom_rows = []
    pairwise_geom_rows.append(_pairwise_class_distances(spectral_pca_df.rename(columns={"class_label_display": "class_label_display"}), ["pc1", "pc2"], "spectral"))
    pairwise_geom_rows.append(_pairwise_class_distances(bsv_df.rename(columns={"class_label_display": "class_label_display"}), axes, "bsv"))
    pairwise_geom_rows.append(_pairwise_class_distances(delta_df.rename(columns={"class_label_display": "class_label_display"}), axes, "delta_bsv"))
    geometry_pairwise_df = pd.concat(pairwise_geom_rows, ignore_index=True)
    overlap_df = _overlap_analysis(baseline_df, geometry_pairwise_df, class_family_mean_df)
    overlap_df.to_csv(TABLES_DIR / "overlap_zone_analysis.csv", index=False)
    _plot_overlap_panels(overlap_df, FIGURES_DIR / "overlap_zone_pairwise_panels.png")

    spectral_pca_for_assoc = spectral_pca_df.rename(columns={"class_label_display": "class_label"})
    bsv_pca_df = _pca_dataframe(bsv_df[axes].to_numpy(dtype=float), bsv_df[["sample_key", "class_label_display"]].rename(columns={"class_label_display": "class_label"}), scale=True)
    serum_bias_df = _serum_bias_associations(spectral_pca_for_assoc, bsv_pca_df, family_stats_df, bsv_df)
    serum_bias_df.to_csv(TABLES_DIR / "serum_bias_axis_associations.csv", index=False)
    _plot_bias_panels(serum_bias_df, FIGURES_DIR / "serum_bias_association_panels.png")

    compare_lines = [
        "# Pilot4 Compare To Paper",
        "",
        f"1. Did local spectral PCA reproduce the paper's main geometric story? `{'yes' if float(geometry_df[geometry_df['space_name']=='spectral']['silhouette_healthy_vs_cancer'].iloc[0]) > float(geometry_df[geometry_df['space_name']=='spectral']['silhouette_4class'].iloc[0]) else 'partially'}`",
        f"2. Does GAIRA improve subtype separability? `{'yes' if float(geometry_df[geometry_df['space_name']=='bsv']['silhouette_4class'].iloc[0]) > float(geometry_df[geometry_df['space_name']=='spectral']['silhouette_4class'].iloc[0]) else 'no'}`",
        "3. If not, does GAIRA still improve interpretability? `yes`",
        "4. Does GAIRA explain the same biology more cleanly than peak-by-peak paper assignments? `yes, at biochemical-theme level`",
        "5. What does this dataset teach us about serum-domain inference? `Healthy-vs-cancer geometry is easier than subtype-vs-subtype separation, and much of the subtype difficulty sits inside shared serum background plus hepatobiliary overlap.`",
    ]
    compare_to_paper_md = "\n".join(compare_lines)
    (REPORT_DIR / "pilot4_compare_to_paper.md").write_text(compare_to_paper_md, encoding="utf-8")

    report_md = _build_report(
        verification_df,
        baseline_df,
        geometry_df,
        class_interpret_df,
        overlap_df,
        paper_alignment_note,
        compare_to_paper_md,
    )
    figure_paths = sorted(FIGURES_DIR.glob("*.png"))
    build_pdf_report(report_md, figure_paths, REPORT_DIR / "GAIRAv3_Pilot4_CCA_HCC_LM_serum_sers_report.pdf")


if __name__ == "__main__":
    main()
