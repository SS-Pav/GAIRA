from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gaira.autoresearch_storage import (
    DEFAULT_STORAGE_CONFIG_PATH,
    initialize_autoresearch_sprint,
    load_autoresearch_storage_config,
)
from gaira.demo.autoresearch_pass5_utils import (
    Pass5HarnessConfig,
    apply_pass5_filter_mode,
    build_bsv_profiles_pass5,
)
from gaira.demo.gaira_experiment_runner_utils import (
    ResolvedExperiment,
    build_source_role_sets,
    load_architecture_registries,
    load_grounding_family_dataframe,
    load_query_dataframe,
)
from gaira.demo.raw_bsv_pilot_utils import (
    ALL_AXES,
    apply_source_role_policy,
    build_group_mean_query_df,
    decode_and_align,
    load_ontology_rules,
    map_references_to_axes,
)


ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = ROOT / "reports" / "gaira_architecture_scaffold_v2"
PHASE1_DIR = ROOT / "reports" / "gaira_phase1_registry_audit_v2"
ONTOLOGY_PATH = ROOT / "config" / "phase2_bsv_ontology_rules_v2.csv"
V4_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot1a_celltype_probe1_v4"
)
PASS5_TABLE = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pass5_saturation_fix/tables/calibration_results_ranked.csv"
)
SPRINT_SUBDIR = "pilot1a_celltype_probe1_v5"
SUBSET_ALIAS = "small2023_cellline"

CONFIG_SPECS = [
    {
        "config_id": "baseline_v1_locked_purine",
        "short_label": "baseline",
        "display_name": "Baseline v1 locked purine",
        "filter_mode": "purine_focused_universal",
        "top_k": 5,
        "weighting_mode": "uniform_weighting",
        "weighting_param": None,
        "diversity_mode": "none",
    },
    {
        "config_id": "candidate_v2_cfg05_max_desaturation",
        "short_label": "cfg05",
        "display_name": "Candidate v2 cfg05 max desaturation",
        "filter_mode": "purine_expanded_neighbor",
        "top_k": 5,
        "weighting_mode": "softmax_temperature",
        "weighting_param": 1.0,
        "diversity_mode": "compound_uniqueness_penalty",
    },
    {
        "config_id": "candidate_v2_cfg08_balanced_update",
        "short_label": "cfg08",
        "display_name": "Candidate v2 cfg08 balanced update",
        "filter_mode": "balanced_metabolite_subset",
        "top_k": 8,
        "weighting_mode": "rank_decay_weighting",
        "weighting_param": 0.75,
        "diversity_mode": "family_balance_penalty",
    },
]

REUSED_FILES = [
    "per_spectrum_bsv.csv",
    "class_mean_bsv.csv",
    "pairwise_delta_bsv.csv",
    "intra_class_bsv_variance.csv",
    "inter_class_bsv_distance.csv",
    "class_topk_neighborhood_composition.csv",
    "class_neighborhood_entropy.csv",
    "class_top1_dominance.csv",
    "class_axis_entropy.csv",
    "retrieval_hit_summary_by_class.csv",
    "per_spectrum_retrieval_hits.csv",
    "pca_coordinates_spectral.csv",
    "pca_coordinates_bsv.csv",
    "pca_coordinates_bsv_class_mean.csv",
    "config_within_between_summary.csv",
]

ROOT_FIGURES_TO_COPY = [
    "pca_spectral_original_dataset.png",
    "pca_bsv_baseline_v1_locked_purine.png",
    "pca_bsv_candidate_v2_cfg05_max_desaturation.png",
    "pca_bsv_candidate_v2_cfg08_balanced_update.png",
    "pca_bsv_class_mean_baseline_v1_locked_purine.png",
    "pca_bsv_class_mean_candidate_v2_cfg05_max_desaturation.png",
    "pca_bsv_class_mean_candidate_v2_cfg08_balanced_update.png",
    "pca_bsv_class_mean_overlay.png",
    "pilot1a_within_between_comparison.png",
    "pilot1a_entropy_dominance_comparison.png",
    "pilot1a_config_tradeoff_summary.png",
]

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

CLASS_COLORS = {
    "Hec": "#4c78a8",
    "Hela": "#f58518",
    "Ht": "#54a24b",
    "Mef": "#e45756",
    "Thp": "#72b7b2",
}
CONFIG_COLORS = {"baseline": "#577590", "cfg05": "#f3722c", "cfg08": "#43aa8b"}
FAMILY_COLORS = {
    "purine_core_like": "#355070",
    "methylated_purine_like": "#6d597a",
    "guanidine_like": "#b56576",
    "sulfur_small_molecule_like": "#2a9d8f",
    "aromatic_small_molecule_like": "#577590",
    "generic_other_metabolite": "#e9c46a",
}


def _axes_present(df: pd.DataFrame) -> list[str]:
    return [axis for axis in ALL_AXES if axis in df.columns]


def _require_v4_inputs() -> None:
    if not V4_ROOT.exists():
        raise RuntimeError(f"Missing v4 root: {V4_ROOT}")
    for spec in CONFIG_SPECS:
        run_dir = V4_ROOT / "runs" / str(spec["config_id"])
        for name in REUSED_FILES:
            path = run_dir / name
            if not path.exists() or path.stat().st_size <= 0:
                raise RuntimeError(f"Missing v4 artifact: {path}")
            df = pd.read_csv(path)
            if df.empty:
                raise RuntimeError(f"Empty v4 artifact: {path}")


def _copy_reused_outputs(sprint_root: Path, figures_dir: Path) -> None:
    for spec in CONFIG_SPECS:
        src_run = V4_ROOT / "runs" / str(spec["config_id"])
        dst_run = sprint_root / "runs" / str(spec["config_id"])
        dst_run.mkdir(parents=True, exist_ok=True)
        (dst_run / "tables").mkdir(exist_ok=True)
        (dst_run / "report").mkdir(exist_ok=True)
        for name in REUSED_FILES:
            shutil.copy2(src_run / name, dst_run / name)
            shutil.copy2(src_run / name, dst_run / "tables" / name)
        run_cfg = src_run / "report" / "run_config.json"
        if run_cfg.exists():
            shutil.copy2(run_cfg, dst_run / "report" / "run_config.json")
    figures_dir.mkdir(parents=True, exist_ok=True)
    for name in ROOT_FIGURES_TO_COPY:
        src = V4_ROOT / "figures" / name
        if src.exists():
            shutil.copy2(src, figures_dir / name)


def _resolve_alias(registries, subset_alias: str) -> ResolvedExperiment:
    matches = registries.dataset_experiments[
        registries.dataset_experiments["subset_alias"].astype(str) == str(subset_alias)
    ].copy()
    if matches.empty or len(matches) > 1:
        raise RuntimeError(f"Could not resolve unique subset alias {subset_alias}")
    dataset_row = matches.iloc[0]
    experiment_row = pd.Series(
        {
            "experiment_id": f"pilot1a_probe1_v5__{subset_alias}",
            "subset_alias": subset_alias,
            "grounding_families_used": "universal_biochemical_grounding",
        }
    )
    return ResolvedExperiment(
        experiment_row=experiment_row,
        dataset_row=dataset_row,
        subset_alias=subset_alias,
        grounding_family_names=["universal_biochemical_grounding"],
    )


def _read_run_df(sprint_root: Path, config_id: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(sprint_root / "runs" / config_id / filename)


def _build_delta_tables(class_mean_df: pd.DataFrame, per_spectrum_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    axes = _axes_present(class_mean_df)
    cohort_mean = per_spectrum_df[axes].mean(axis=0)
    delta_class = class_mean_df[["sample_key", "dataset_id", "subset_id", "class_label"]].copy()
    delta_spec = per_spectrum_df[["sample_key", "dataset_id", "subset_id", "class_label"]].copy()
    for axis in axes:
        delta_class[axis] = class_mean_df[axis].to_numpy(dtype=float) - float(cohort_mean[axis])
        delta_spec[axis] = per_spectrum_df[axis].to_numpy(dtype=float) - float(cohort_mean[axis])
    return delta_class, delta_spec


def _compound_to_fine_family(name: str) -> str:
    lower = str(name).strip().lower()
    if any(token in lower for token in ["3-methyladenine", "methyladenine"]):
        return "methylated_purine_like"
    if any(token in lower for token in ["guanidine", "guanidino"]):
        return "guanidine_like"
    if any(token in lower for token in ["cyste", "glutath", "methion", "seleno", "sulfoximine", "sulfur"]):
        return "sulfur_small_molecule_like"
    if any(token in lower for token in ["tyr", "trypt", "phenyl", "indole", "dopamine", "3-methoxytyramine"]):
        return "aromatic_small_molecule_like"
    if any(token in lower for token in ["adenine", "xanth", "hypox", "uric", "urate", "inos", "purine"]):
        return "purine_core_like"
    return "generic_other_metabolite"


def _build_family_fingerprint(class_neighborhood_df: pd.DataFrame) -> pd.DataFrame:
    df = class_neighborhood_df.copy()
    df["family"] = df["compound_label"].astype(str).map(_compound_to_fine_family)
    grouped = df.groupby(["class_label", "family"], as_index=False)["support_fraction"].sum()
    rows = []
    for class_label in sorted(df["class_label"].astype(str).unique().tolist()):
        sub = grouped[grouped["class_label"].astype(str) == class_label].copy()
        total = float(sub["support_fraction"].sum())
        existing = {str(x) for x in sub["family"].tolist()}
        for family in FAMILY_ORDER:
            if family not in existing:
                rows.append({"class_label": class_label, "family": family, "family_fraction": 0.0})
        if total > 0:
            sub["family_fraction"] = sub["support_fraction"] / total
        else:
            sub["family_fraction"] = 0.0
        rows.extend(sub[["class_label", "family", "family_fraction"]].to_dict("records"))
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["class_label", "family", "family_fraction"])
    return out.sort_values(["class_label", "family"]).reset_index(drop=True)


def _fit_pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :2] * s[:2]
    denom = max(float((s**2).sum()), 1e-12)
    explained = (s[:2] ** 2) / denom if len(s) >= 2 else np.array([1.0, 0.0])
    return scores, explained


def _pca_dataframe(df: pd.DataFrame, axes: list[str], *, scale: bool = False) -> pd.DataFrame:
    matrix = df[axes].to_numpy(dtype=float)
    if scale:
        std = matrix.std(axis=0, keepdims=True)
        std = np.where(std < 1e-9, 1.0, std)
        matrix = matrix / std
    scores, explained = _fit_pca(matrix)
    out = df[["class_label"]].copy()
    if "sample_key" in df.columns:
        out["sample_key"] = df["sample_key"].astype(str)
    out["pc1"] = scores[:, 0]
    out["pc2"] = scores[:, 1] if scores.shape[1] > 1 else 0.0
    out["pc1_explained_ratio"] = float(explained[0])
    out["pc2_explained_ratio"] = float(explained[1]) if len(explained) > 1 else 0.0
    return out


def _build_residual_query_df(query_df: pd.DataFrame) -> pd.DataFrame:
    master_x, matrix = decode_and_align(query_df)
    class_mean_df = build_group_mean_query_df(query_df, group_col="class_label")
    _, class_mean_matrix = decode_and_align(class_mean_df)
    class_order = class_mean_df["class_label"].astype(str).tolist()
    class_to_mean = {label: class_mean_matrix[i] for i, label in enumerate(class_order)}
    residuals = []
    work = query_df.reset_index(drop=True).copy()
    for idx, row in work.iterrows():
        label = str(row["class_label"])
        residuals.append(matrix[idx] - class_to_mean[label])
    residual_matrix = np.vstack(residuals).astype(float)
    out = work.copy()
    out["wavenumbers_json"] = json.dumps(master_x.astype(float).tolist())
    out["intensity_json"] = [json.dumps(vec.astype(float).tolist()) for vec in residual_matrix]
    return out


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
    original_grounding = list(resolved.grounding_family_names)
    object.__setattr__(resolved, "grounding_family_names", ["universal_biochemical_grounding"])
    try:
        grounding_df, family_to_sources, _ = load_grounding_family_dataframe(resolved, registries)
    finally:
        object.__setattr__(resolved, "grounding_family_names", original_grounding)
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


def _plot_scatter_pca(pca_df: pd.DataFrame, output_path: Path, title: str, *, annotate: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    labels = sorted(pca_df["class_label"].astype(str).unique().tolist())
    for label in labels:
        sub = pca_df[pca_df["class_label"].astype(str) == label].copy()
        ax.scatter(
            sub["pc1"].to_numpy(dtype=float),
            sub["pc2"].to_numpy(dtype=float),
            s=30 if not annotate else 70,
            alpha=0.78,
            color=CLASS_COLORS.get(label, "#666666"),
            label=label,
            edgecolors="white",
            linewidths=0.4,
        )
    if annotate:
        for i, row in enumerate(pca_df.itertuples(index=False)):
            ax.annotate(
                str(row.class_label),
                (float(row.pc1), float(row.pc2)),
                xytext=(6, 6 + (i % 3) * 3),
                textcoords="offset points",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
            )
    ax.set_xlabel(f"PC1 ({float(pca_df['pc1_explained_ratio'].iloc[0]) * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({float(pca_df['pc2_explained_ratio'].iloc[0]) * 100:.1f}% var)")
    ax.set_title(title)
    ax.grid(True, alpha=0.20, linewidth=0.6)
    ax.axhline(0.0, color="#bbbbbb", linewidth=0.7)
    ax.axvline(0.0, color="#bbbbbb", linewidth=0.7)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="Class")
    fig.tight_layout(rect=[0.0, 0.0, 0.83, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_fixed_radar(df: pd.DataFrame, output_path: Path, title: str, *, delta_mode: bool = False) -> None:
    labels = sorted(df["class_label"].astype(str).tolist())
    ncols = 2
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(10.8, 5.0 * nrows),
        subplot_kw={"projection": "polar"},
    )
    axs = np.atleast_1d(axs).ravel()
    angles = np.linspace(0, 2 * np.pi, len(FIXED_RADAR_AXES), endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])
    if delta_mode:
        radius_lim = max(float(np.abs(df[FIXED_RADAR_AXES].to_numpy(dtype=float)).max()), 0.05)
    else:
        radius_lim = max(float(df[FIXED_RADAR_AXES].to_numpy(dtype=float).max()), 0.5)
    for ax in axs[len(labels) :]:
        ax.axis("off")
    for ax, label in zip(axs, labels, strict=False):
        row = df[df["class_label"].astype(str) == label].iloc[0]
        vals = np.array([float(row.get(axis, 0.0)) for axis in FIXED_RADAR_AXES], dtype=float)
        plot_vals = vals + radius_lim if delta_mode else vals
        vals_closed = np.concatenate([plot_vals, [plot_vals[0]]])
        color = CLASS_COLORS.get(label, "#666666")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.plot(angles_closed, vals_closed, color=color, linewidth=2.4)
        ax.fill(angles_closed, vals_closed, color=color, alpha=0.28)
        ax.scatter(angles, plot_vals, color=color, s=16, zorder=3)
        ax.set_xticks(angles)
        ax.set_xticklabels(FIXED_RADAR_AXES, fontsize=8)
        ax.tick_params(axis="x", pad=9)
        if delta_mode:
            ax.set_ylim(0.0, 2.0 * radius_lim)
            ax.set_yticks([0.0, radius_lim, 2.0 * radius_lim])
            ax.set_yticklabels([f"{-radius_lim:.2f}", "0", f"{radius_lim:.2f}"], fontsize=7)
        else:
            ax.set_ylim(0.0, radius_lim)
            ax.set_yticks([radius_lim * 0.33, radius_lim * 0.66, radius_lim])
            ax.set_yticklabels([f"{radius_lim*0.33:.2f}", f"{radius_lim*0.66:.2f}", f"{radius_lim:.2f}"], fontsize=7)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_title(label, y=1.12, fontsize=11, fontweight="bold")
    fig.suptitle(title, fontsize=15, y=0.99)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_family_bars(family_df: pd.DataFrame, output_path: Path, title: str) -> None:
    classes = sorted(family_df["class_label"].astype(str).unique().tolist())
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    left = np.zeros(len(classes), dtype=float)
    for family in FAMILY_ORDER:
        vals = []
        for label in classes:
            sub = family_df[
                (family_df["class_label"].astype(str) == label)
                & (family_df["family"].astype(str) == family)
            ]
            vals.append(float(sub["family_fraction"].iloc[0]) if not sub.empty else 0.0)
        vals_arr = np.asarray(vals, dtype=float)
        ax.barh(
            np.arange(len(classes)),
            vals_arr,
            left=left,
            color=FAMILY_COLORS[family],
            label=family,
            alpha=0.9,
        )
        left += vals_arr
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Family fraction")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="Family")
    ax.grid(True, axis="x", alpha=0.20, linewidth=0.6)
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _classification_metrics(features_df: pd.DataFrame, axes: list[str]) -> tuple[dict[str, float], np.ndarray, list[str]]:
    X = features_df[axes].to_numpy(dtype=float)
    y = features_df["class_label"].astype(str).to_numpy()
    labels = sorted(np.unique(y).tolist())
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    preds = cross_val_predict(model, X, y, cv=splitter)
    acc = float(accuracy_score(y, preds))
    cm = confusion_matrix(y, preds, labels=labels)
    row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm = cm / row_sum
    return {"accuracy": acc}, cm_norm, labels


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_confusion_grid(confusion_results: list[tuple[str, np.ndarray, list[str]]], output_path: Path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for ax, (short_label, cm, labels) in zip(axs, confusion_results, strict=False):
        im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(short_label)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axs, shrink=0.76)
    fig.suptitle("Delta-BSV Confusion Matrices", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_tradeoff(comparator_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.1))
    for row in comparator_df.itertuples(index=False):
        ax.scatter(
            float(row.mean_neighborhood_entropy),
            float(row.mean_inter_class_distance_delta),
            s=250 * float(row.classification_accuracy_delta),
            color=CONFIG_COLORS.get(str(row.config_short_label), "#666666"),
            alpha=0.82,
            edgecolors="white",
            linewidths=0.8,
        )
        ax.annotate(
            f"{row.config_short_label}\nacc={row.classification_accuracy_delta:.2f}",
            (float(row.mean_neighborhood_entropy), float(row.mean_inter_class_distance_delta)),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("Mean neighborhood entropy")
    ax.set_ylabel("Mean inter-class distance (delta space)")
    ax.set_title("Pilot 1a v5 Tradeoff: entropy vs separation vs accuracy")
    ax.grid(True, alpha=0.20, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _build_report(report_path: Path, comparator_df: pd.DataFrame, pass5_ranked_df: pd.DataFrame) -> None:
    baseline = comparator_df[comparator_df["config_short_label"] == "baseline"].iloc[0]
    cfg05 = comparator_df[comparator_df["config_short_label"] == "cfg05"].iloc[0]
    cfg08 = comparator_df[comparator_df["config_short_label"] == "cfg08"].iloc[0]
    baseline_val = float(pass5_ranked_df[pass5_ranked_df["config_id"] == "cfg02"]["validation_score"].iloc[0])
    cfg05_val = float(pass5_ranked_df[pass5_ranked_df["config_id"] == "cfg05"]["validation_score"].iloc[0])
    cfg08_val = float(pass5_ranked_df[pass5_ranked_df["config_id"] == "cfg08"]["validation_score"].iloc[0])
    lines = [
        "# GAIRAv3 Pilot 1a Celltype Probe1 v5 Report",
        "",
        "## 1. Comparison baseline vs cfg05 vs cfg08",
        "- v5 upgrades the fingerprint from absolute BSV to a delta-first stack: delta-BSV, fine-grained neighborhood family mix, and residual-BSV projection.",
        "- Existing v3/v4 outputs were reused. Core spectra and baseline BSV tables were not recomputed.",
        "",
        "## 2. Delta-first advantage",
        "- Absolute BSV still describes broad biochemical position.",
        "- Delta-BSV makes relative enrichment and depletion easier to read at the class level.",
        "- Fine-grained family composition makes the local chemistry neighborhood more interpretable than coarse Tier-1 axes alone.",
        "- Residual-BSV PCA gives a lightweight view of subtle within-class structure after subtracting the class mean spectrum.",
        "",
        "## 3. Config comparison",
        f"- Baseline: delta accuracy `{baseline['classification_accuracy_delta']:.4f}`, absolute accuracy `{baseline['classification_accuracy_absolute']:.4f}`, top1 dominance `{baseline['mean_top1_dominance']:.4f}`.",
        f"- cfg05: delta accuracy `{cfg05['classification_accuracy_delta']:.4f}`, absolute accuracy `{cfg05['classification_accuracy_absolute']:.4f}`, top1 dominance `{cfg05['mean_top1_dominance']:.4f}`.",
        f"- cfg08: delta accuracy `{cfg08['classification_accuracy_delta']:.4f}`, absolute accuracy `{cfg08['classification_accuracy_absolute']:.4f}`, top1 dominance `{cfg08['mean_top1_dominance']:.4f}`.",
        "- Delta accuracy matches absolute-space accuracy here because delta-BSV is a cohort-centered translation of the same BSV support vectors. The main v5 gain is interpretability: delta-BSV, family composition, and residual projection make the class fingerprint easier to read.",
        "- In the v5 comparator, cfg08 has the strongest class-label recovery, the lowest neighborhood dominance, and the broadest family-level fingerprint among the three fixed configs.",
        "",
        "## 4. Recommendation",
        f"- Pass 5 validation note: baseline `{baseline_val:.4f}`, cfg05 `{cfg05_val:.4f}`, cfg08 `{cfg08_val:.4f}`.",
        "- cfg08 is the recommended working default for Pilot 1b under the v5 fingerprint definition.",
        "- Baseline remains the narrow purine-heavy reference.",
        "- cfg05 remains the purine-anchored desaturation comparator when a tighter chemistry vocabulary is preferred.",
        "- cfg08 should still be interpreted cautiously: it broadens the vocabulary substantially, but in v5 that broader fingerprint is accompanied by the strongest class separability rather than a loss of structure.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_pdf(report_md: Path, figure_paths: list[Path], output_path: Path) -> None:
    text = report_md.read_text(encoding="utf-8")
    wrapped_lines: list[str] = []
    for raw in text.splitlines():
        if raw.startswith("#"):
            wrapped_lines.append(raw)
        elif raw.strip():
            wrapped_lines.extend(textwrap.wrap(raw, width=96))
        else:
            wrapped_lines.append("")
    with PdfPages(output_path) as pdf:
        chunk_size = 34
        for i in range(0, len(wrapped_lines), chunk_size):
            fig = plt.figure(figsize=(8.27, 11.69))
            y = 0.965
            for line in wrapped_lines[i : i + chunk_size]:
                size = 12 if line.startswith("# ") else 10 if line.startswith("## ") else 8.6
                weight = "bold" if line.startswith("#") else "normal"
                fig.text(0.06, y, line, ha="left", va="top", fontsize=size, fontweight=weight, family="DejaVu Sans Mono")
                y -= 0.026 if line.startswith("#") else 0.023
            pdf.savefig(fig)
            plt.close(fig)
        for path in figure_paths:
            img = plt.imread(path)
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_axes([0.03, 0.06, 0.94, 0.88])
            ax.imshow(img)
            ax.axis("off")
            fig.suptitle(path.name, fontsize=12, y=0.98)
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    _require_v4_inputs()
    storage_cfg = load_autoresearch_storage_config(DEFAULT_STORAGE_CONFIG_PATH)
    sprint_paths = initialize_autoresearch_sprint(
        DEFAULT_STORAGE_CONFIG_PATH,
        sprint_id=f"{storage_cfg.sprint_id}/{SPRINT_SUBDIR}",
    )
    _copy_reused_outputs(sprint_paths.sprint_root, sprint_paths.figures_dir)
    pass5_ranked_df = pd.read_csv(PASS5_TABLE)

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
    query_df = load_query_dataframe(resolved.dataset_row)

    comparator_rows = []
    classifier_rows = []
    confusion_results = []

    for spec in CONFIG_SPECS:
        config_id = str(spec["config_id"])
        short_label = str(spec["short_label"])
        run_dir = sprint_paths.sprint_root / "runs" / config_id
        tables_dir = run_dir / "tables"
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        per_spectrum_bsv_df = _read_run_df(sprint_paths.sprint_root, config_id, "per_spectrum_bsv.csv")
        class_mean_bsv_df = _read_run_df(sprint_paths.sprint_root, config_id, "class_mean_bsv.csv")
        class_neighborhood_df = _read_run_df(sprint_paths.sprint_root, config_id, "class_topk_neighborhood_composition.csv")
        intra_df = _read_run_df(sprint_paths.sprint_root, config_id, "intra_class_bsv_variance.csv")
        inter_df = _read_run_df(sprint_paths.sprint_root, config_id, "inter_class_bsv_distance.csv")
        entropy_df = _read_run_df(sprint_paths.sprint_root, config_id, "class_neighborhood_entropy.csv")
        top1_df = _read_run_df(sprint_paths.sprint_root, config_id, "class_top1_dominance.csv")
        axis_entropy_df = _read_run_df(sprint_paths.sprint_root, config_id, "class_axis_entropy.csv")

        delta_class_df, delta_spec_df = _build_delta_tables(class_mean_bsv_df, per_spectrum_bsv_df)
        delta_class_df.to_csv(run_dir / "delta_class_mean_bsv.csv", index=False)
        delta_spec_df.to_csv(run_dir / "delta_per_spectrum_bsv.csv", index=False)
        delta_class_df.to_csv(tables_dir / "delta_class_mean_bsv.csv", index=False)
        delta_spec_df.to_csv(tables_dir / "delta_per_spectrum_bsv.csv", index=False)

        family_df = _build_family_fingerprint(class_neighborhood_df)
        family_df.to_csv(run_dir / "class_family_fingerprint.csv", index=False)
        family_df.to_csv(tables_dir / "class_family_fingerprint.csv", index=False)

        delta_axes = _axes_present(delta_spec_df)
        delta_pca_df = _pca_dataframe(delta_spec_df, delta_axes, scale=False)
        delta_class_pca_df = _pca_dataframe(delta_class_df, _axes_present(delta_class_df), scale=False)
        delta_pca_df.to_csv(run_dir / "pca_delta_bsv.csv", index=False)
        delta_class_pca_df.to_csv(run_dir / "pca_delta_bsv_class_mean.csv", index=False)
        delta_pca_df.to_csv(tables_dir / "pca_delta_bsv.csv", index=False)
        delta_class_pca_df.to_csv(tables_dir / "pca_delta_bsv_class_mean.csv", index=False)

        _plot_scatter_pca(delta_pca_df, sprint_paths.figures_dir / f"pca_delta_bsv_{config_id}.png", f"Delta-BSV PCA: {short_label}")
        _plot_scatter_pca(
            delta_class_pca_df,
            sprint_paths.figures_dir / f"pca_delta_bsv_class_mean_{config_id}.png",
            f"Delta-BSV Class-Mean PCA: {short_label}",
            annotate=True,
        )

        grounding_df, mapping_df, harness_config = _prepare_grounding_and_mapping(registries, resolved, spec)
        residual_query_df = _build_residual_query_df(query_df)
        residual_bsv_df, _ = build_bsv_profiles_pass5(
            residual_query_df,
            grounding_df,
            mapping_df,
            top_k=harness_config.top_k,
            similarity_metric="cosine",
            weighting_mode=harness_config.weighting_mode,
            weighting_param=harness_config.weighting_param,
            diversity_mode=harness_config.diversity_mode,
            family_min_coverage=harness_config.family_min_coverage,
        )
        residual_axes = _axes_present(residual_bsv_df)
        residual_cohort = residual_bsv_df[residual_axes].mean(axis=0)
        delta_residual_bsv_df = residual_bsv_df[["sample_key", "dataset_id", "subset_id", "class_label"]].copy()
        for axis in residual_axes:
            delta_residual_bsv_df[axis] = residual_bsv_df[axis].to_numpy(dtype=float) - float(residual_cohort[axis])
        delta_residual_bsv_df.to_csv(run_dir / "delta_residual_bsv.csv", index=False)
        delta_residual_bsv_df.to_csv(tables_dir / "delta_residual_bsv.csv", index=False)
        residual_pca_df = _pca_dataframe(delta_residual_bsv_df, residual_axes, scale=False)
        residual_pca_df.to_csv(run_dir / "pca_residual_bsv.csv", index=False)
        residual_pca_df.to_csv(tables_dir / "pca_residual_bsv.csv", index=False)
        _plot_scatter_pca(
            residual_pca_df,
            sprint_paths.figures_dir / f"pca_residual_bsv_{config_id}.png",
            f"Residual-BSV PCA: {short_label}",
        )

        _plot_fixed_radar(
            class_mean_bsv_df,
            sprint_paths.figures_dir / f"radar_absolute_fixed_{config_id}.png",
            f"Absolute Fixed-Axis Radar: {short_label}",
            delta_mode=False,
        )
        _plot_fixed_radar(
            delta_class_df,
            sprint_paths.figures_dir / f"radar_delta_fixed_{config_id}.png",
            f"Delta Fixed-Axis Radar: {short_label}",
            delta_mode=True,
        )
        _plot_family_bars(
            family_df,
            sprint_paths.figures_dir / f"neighborhood_family_bars_{config_id}.png",
            f"Neighborhood Family Distribution: {short_label}",
        )

        abs_metrics, _, _ = _classification_metrics(per_spectrum_bsv_df, _axes_present(per_spectrum_bsv_df))
        delta_metrics, cm_delta, labels = _classification_metrics(delta_spec_df, delta_axes)
        cls_metrics_df = pd.DataFrame(
            [
                {
                    "config_id": config_id,
                    "config_short_label": short_label,
                    "classification_accuracy_absolute": abs_metrics["accuracy"],
                    "classification_accuracy_delta": delta_metrics["accuracy"],
                }
            ]
        )
        cls_metrics_df.to_csv(run_dir / "fingerprint_classification_metrics.csv", index=False)
        cls_metrics_df.to_csv(tables_dir / "fingerprint_classification_metrics.csv", index=False)
        _plot_confusion_matrix(
            cm_delta,
            labels,
            sprint_paths.figures_dir / f"confusion_matrix_{config_id}.png",
            f"Delta-BSV confusion matrix: {short_label}",
        )
        classifier_rows.append(cls_metrics_df.iloc[0].to_dict())
        confusion_results.append((short_label, cm_delta, labels))

        intra_mean = float(intra_df.drop(columns=["class_label"]).mean(axis=1).mean())
        nonident = inter_df[inter_df["class_label_a"] != inter_df["class_label_b"]].copy()
        mean_inter_delta = float(nonident["euclidean_distance"].mean())
        family_pivot = (
            family_df.pivot(index="class_label", columns="family", values="family_fraction")
            .reindex(sorted(class_mean_bsv_df["class_label"].astype(str).tolist()))
            .reindex(FAMILY_ORDER, axis=1)
            .fillna(0.0)
        )
        comparator_rows.append(
            {
                "config_id": config_id,
                "config_short_label": short_label,
                "mean_intra_class_variance_delta": intra_mean,
                "mean_inter_class_distance_delta": mean_inter_delta,
                "mean_neighborhood_entropy": float(entropy_df["neighborhood_entropy"].mean()),
                "mean_top1_dominance": float(top1_df["top1_fraction"].mean()),
                "mean_axis_entropy": float(axis_entropy_df["axis_entropy"].mean()),
                "classification_accuracy_absolute": float(abs_metrics["accuracy"]),
                "classification_accuracy_delta": float(delta_metrics["accuracy"]),
                "mean_family_distance": float(np.mean([
                    np.linalg.norm(family_pivot.iloc[i].to_numpy(dtype=float) - family_pivot.iloc[j].to_numpy(dtype=float))
                    for i in range(len(family_pivot)) for j in range(i + 1, len(family_pivot))
                ])) if len(family_pivot) > 1 else 0.0,
            }
        )

    classification_df = pd.DataFrame(classifier_rows)
    comparator_df = pd.DataFrame(comparator_rows)
    classification_df.to_csv(sprint_paths.tables_dir / "fingerprint_classification_metrics.csv", index=False)
    comparator_df.to_csv(sprint_paths.tables_dir / "pilot1a_v5_comparator.csv", index=False)
    _plot_confusion_grid(confusion_results, sprint_paths.figures_dir / "confusion_matrix.png")
    _plot_tradeoff(comparator_df, sprint_paths.figures_dir / "tradeoff_plot.png")

    report_md = sprint_paths.report_dir / "GAIRAv3_Pilot1a_celltype_probe1_v5_report.md"
    _build_report(report_md, comparator_df, pass5_ranked_df)

    figure_paths = [
        sprint_paths.figures_dir / "pca_spectral_original_dataset.png",
        sprint_paths.figures_dir / "pca_bsv_baseline_v1_locked_purine.png",
        sprint_paths.figures_dir / "pca_bsv_candidate_v2_cfg05_max_desaturation.png",
        sprint_paths.figures_dir / "pca_bsv_candidate_v2_cfg08_balanced_update.png",
        sprint_paths.figures_dir / "pca_bsv_class_mean_overlay.png",
        sprint_paths.figures_dir / "confusion_matrix.png",
        sprint_paths.figures_dir / "tradeoff_plot.png",
    ]
    for spec in CONFIG_SPECS:
        config_id = str(spec["config_id"])
        figure_paths.extend(
            [
                sprint_paths.figures_dir / f"pca_delta_bsv_{config_id}.png",
                sprint_paths.figures_dir / f"pca_delta_bsv_class_mean_{config_id}.png",
                sprint_paths.figures_dir / f"pca_residual_bsv_{config_id}.png",
                sprint_paths.figures_dir / f"radar_absolute_fixed_{config_id}.png",
                sprint_paths.figures_dir / f"radar_delta_fixed_{config_id}.png",
                sprint_paths.figures_dir / f"neighborhood_family_bars_{config_id}.png",
                sprint_paths.figures_dir / f"confusion_matrix_{config_id}.png",
            ]
        )
    _build_pdf(report_md, figure_paths, sprint_paths.report_dir / "GAIRAv3_Pilot1a_celltype_probe1_v5_report.pdf")
    print(comparator_df.to_string(index=False))
    print("\nclassification_accuracy_per_config")
    print(classification_df.to_string(index=False))
    best_cfg = comparator_df.sort_values(
        ["classification_accuracy_delta", "mean_top1_dominance"],
        ascending=[False, True],
    ).iloc[0]
    print(f"\nfinal_recommendation: {best_cfg['config_id']}")


if __name__ == "__main__":
    main()
