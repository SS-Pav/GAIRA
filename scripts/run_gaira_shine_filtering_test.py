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
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaira.demo.autoresearch_pass5_utils import build_bsv_profiles_pass5
from gaira.demo.gaira_experiment_runner_utils import load_architecture_registries
from gaira.demo.gaira_pilot_utils import build_pdf_report
from scripts.run_gaira_pilot3_shine_day2_controlanchored import _family_fingerprint_from_retrieval
from scripts.run_gaira_pilot3_shine_ev_sers_fullspectra import (
    ARCH_DIR,
    CONFIG_SPEC,
    FAMILY_ORDER,
    FIXED_RADAR_AXES,
    PHASE1_DIR,
    ROOT,
    _cohort_delta,
    _fit_pca,
    _prepare_grounding_and_mapping,
    _resolve_alias,
)
from scripts.run_gaira_shine_bsv_extension_test import _build_set9_matched_subsets
from scripts.run_gaira_shine_fig4_replication_and_bsv import (
    CONDITION_ORDER,
    CONCENTRATION_VALUES,
    DATA_ROOT,
    RANGE_WAVENUMBERS,
    _build_raw_sampled_subset,
)


OUTPUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/shine_filtering_test"
)
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORT_DIR = OUTPUT_ROOT / "report"

SET10_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/shine_set10_day2_spectral_axis_to_bsv"
)
EXT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/shine_bsv_extension_test"
)
RAW_FIG4_ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/shine_ev_sers/SERS-Hepatotoxicity_DATA_CODE_FIGURE/Figure4")
SUBSET_ALIAS = "shine_ev_stress"
SET10_LABEL = "Set10_D2_raw_sampled"
WINNING_AXIS = "linear_regression"
DAY_LABEL = "D2"
COND_CLASS_LABELS = [f"{DAY_LABEL}_{cond}" for cond in CONDITION_ORDER]
COND_COLORS = {"C0": "#355070", "C10": "#b56576", "C20": "#2a9d8f", "C40": "#e76f51"}


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


def _load_set10_base() -> dict[str, pd.DataFrame]:
    matrix_df = pd.read_csv(SET10_ROOT / "tables" / "set10_day2_spectral_matrix.csv")
    meta_df = pd.read_csv(SET10_ROOT / "tables" / "set10_day2_metadata.csv")
    bsv_df = pd.read_csv(SET10_ROOT / "tables" / "set10_day2_bsv.csv")
    delta_df = pd.read_csv(SET10_ROOT / "tables" / "set10_day2_delta_bsv.csv")
    family_df = pd.read_csv(SET10_ROOT / "tables" / "set10_day2_family.csv")
    ext_df = pd.read_csv(EXT_ROOT / "tables" / "set10_day2_extension_features.csv")
    axis_scores = pd.read_csv(SET10_ROOT / "tables" / "spectral_response_axis_scores.csv")
    axis_scores = axis_scores[axis_scores["axis_name"].astype(str) == WINNING_AXIS].copy().reset_index(drop=True)
    axis_scores["sample_key"] = meta_df["sample_key"].astype(str).tolist()
    return {
        "matrix_df": matrix_df,
        "meta_df": meta_df,
        "bsv_df": bsv_df,
        "delta_df": delta_df,
        "family_df": family_df,
        "ext_df": ext_df,
        "axis_scores": axis_scores,
    }


def _spectral_cols(matrix_df: pd.DataFrame) -> list[str]:
    return [c for c in matrix_df.columns if str(c).startswith("wn_")]


def _concentrations_from_class(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.split("_").str[1].str.replace("C", "", regex=False).astype(int).to_numpy(dtype=int)


def _family_summary(family_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_key, sub in family_df.groupby("sample_key"):
        vals = sub["family_fraction"].to_numpy(dtype=float)
        vals = vals[vals > 0]
        rows.append(
            {
                "sample_key": str(sample_key),
                "top1_dominance": float(sub["family_fraction"].max()) if len(sub) else 0.0,
                "family_entropy": float(-(vals * np.log(vals)).sum()) if len(vals) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _fit_pca_scores(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _fit_pca(matrix, scale=True)


def _pca_metrics_from_mask(
    matrix_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    mask: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    spectral_cols = _spectral_cols(matrix_df)
    sub_matrix = matrix_df.loc[mask, spectral_cols].to_numpy(dtype=float)
    sub_meta = meta_df.loc[mask].copy().reset_index(drop=True)
    scores, explained = _fit_pca_scores(sub_matrix)
    score_df = sub_meta.copy()
    score_df["pc1"] = scores[:, 0]
    score_df["pc2"] = scores[:, 1]
    score_df["pc1_explained_ratio"] = float(explained[0])
    score_df["pc2_explained_ratio"] = float(explained[1])
    silhouette = float(silhouette_score(score_df[["pc1", "pc2"]].to_numpy(dtype=float), score_df["class_label"].astype(str)))
    means = (
        score_df.groupby("class_label", as_index=False)[["pc1", "pc2"]]
        .mean()
        .assign(concentration=lambda f: _concentrations_from_class(f["class_label"]))
        .sort_values("concentration")
        .reset_index(drop=True)
    )
    mean_pc1_order = float(means["concentration"].corr(means["pc1"], method="spearman"))
    pairwise = []
    arr = means[["pc1", "pc2"]].to_numpy(dtype=float)
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            pairwise.append(float(np.linalg.norm(arr[i] - arr[j])))
    adjacent = []
    for i in range(1, len(arr)):
        adjacent.append(float(np.linalg.norm(arr[i] - arr[i - 1])))
    compactness_note = "tighter" if silhouette > -0.02 else "still overlapping"
    return score_df, {
        "silhouette_by_concentration": silhouette,
        "mean_centroid_distance": float(np.mean(pairwise)) if pairwise else 0.0,
        "condition_mean_ordering_spearman": mean_pc1_order,
        "pc1_explained_ratio": float(explained[0]),
        "pc2_explained_ratio": float(explained[1]),
        "mean_adjacent_distance": float(np.mean(adjacent)) if adjacent else 0.0,
        "visual_compactness_note": compactness_note,
    }


def _plot_pca(score_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    for cond in CONDITION_ORDER:
        class_label = f"D2_{cond}"
        sub = score_df[score_df["class_label"].astype(str) == class_label].copy()
        ax.scatter(
            sub["pc1"].to_numpy(dtype=float),
            sub["pc2"].to_numpy(dtype=float),
            s=20,
            alpha=0.55,
            color=COND_COLORS[cond],
            label=f"{CONCENTRATION_VALUES[cond]} mM",
        )
    ax.set_xlabel(f"PC1 ({score_df['pc1_explained_ratio'].iloc[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({score_df['pc2_explained_ratio'].iloc[0]*100:.1f}%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _filter_candidates(
    matrix_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    bsv_df: pd.DataFrame,
    family_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    spectral_cols = _spectral_cols(matrix_df)
    X = StandardScaler().fit_transform(matrix_df[spectral_cols].to_numpy(dtype=float))
    pca_scores, _ = _fit_pca(X, scale=False)
    pc5 = pca_scores[:, :5]
    class_labels = meta_df["class_label"].astype(str).to_numpy()

    family_summary = _family_summary(family_df)
    diag_df = (
        bsv_df.merge(family_summary, on="sample_key", how="left")
        .assign(
            biological_signal=lambda f: f[
                ["nucleic_acid", "protein_peptide", "lipid_membrane", "carbohydrate_glycan", "small_molecule_metabolite"]
            ].sum(axis=1)
        )
    )

    definitions = [
        {
            "filter_name": "pca_cluster_dominant",
            "rule_definition": "KMeans(k=3) on standardized spectral PC1-5; within each concentration retain the dominant cluster.",
        },
        {
            "filter_name": "spectral_outlier",
            "rule_definition": "Within each concentration, retain spectra whose PC1-5 distance to condition centroid is <= 90th percentile.",
        },
        {
            "filter_name": "substrate_bias",
            "rule_definition": "Remove spectra with substrate_adsorption_bias > global 85th percentile, biological_signal < 25th percentile, and top1_dominance > 75th percentile.",
        },
        {
            "filter_name": "combined",
            "rule_definition": "Intersection of pca_cluster_dominant, spectral_outlier, and substrate_bias keeps.",
        },
        {
            "filter_name": "paper_aligned_cluster_count",
            "rule_definition": "Template-count candidate based on archive clustered Set10 D2 counts. On the validated raw-sampled subset it becomes a no-op because retained raw counts already do not exceed the bundled clustered counts.",
        },
    ]
    defs_df = pd.DataFrame(definitions)
    defs_df.to_csv(TABLES_DIR / "filter_candidate_definitions.csv", index=False)

    masks: dict[str, np.ndarray] = {}

    km = KMeans(n_clusters=3, random_state=0, n_init=20)
    labels = km.fit_predict(pc5)
    dominant_keep = np.zeros(len(meta_df), dtype=bool)
    for class_label in sorted(set(class_labels)):
        idx = np.where(class_labels == class_label)[0]
        label_counts = pd.Series(labels[idx]).value_counts().sort_values(ascending=False)
        dominant = int(label_counts.index[0])
        dominant_keep[idx] = labels[idx] == dominant
    masks["pca_cluster_dominant"] = dominant_keep

    outlier_keep = np.zeros(len(meta_df), dtype=bool)
    for class_label in sorted(set(class_labels)):
        idx = np.where(class_labels == class_label)[0]
        points = pc5[idx]
        center = points.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(points - center, axis=1)
        cutoff = float(np.quantile(dist, 0.90))
        outlier_keep[idx] = dist <= cutoff
    masks["spectral_outlier"] = outlier_keep

    substrate_q = float(diag_df["substrate_adsorption_bias"].quantile(0.85))
    bio_q = float(diag_df["biological_signal"].quantile(0.25))
    top1_q = float(diag_df["top1_dominance"].quantile(0.75))
    bad = (
        (diag_df["substrate_adsorption_bias"].to_numpy(dtype=float) > substrate_q)
        & (diag_df["biological_signal"].to_numpy(dtype=float) < bio_q)
        & (diag_df["top1_dominance"].to_numpy(dtype=float) > top1_q)
    )
    masks["substrate_bias"] = ~bad

    masks["combined"] = masks["pca_cluster_dominant"] & masks["spectral_outlier"] & masks["substrate_bias"]
    masks["paper_aligned_cluster_count"] = np.ones(len(meta_df), dtype=bool)

    rows = []
    total_n = len(meta_df)
    for name, mask in masks.items():
        kept = meta_df.loc[mask].copy()
        row = {
            "filter_name": name,
            "n_retained_total": int(mask.sum()),
            "retention_fraction_total": float(mask.mean()),
        }
        for cond in CONDITION_ORDER:
            class_label = f"D2_{cond}"
            n_cond_total = int((meta_df["class_label"].astype(str) == class_label).sum())
            n_cond_keep = int((kept["class_label"].astype(str) == class_label).sum())
            row[f"n_retained_{cond}"] = n_cond_keep
            row[f"retention_fraction_{cond}"] = float(n_cond_keep / n_cond_total) if n_cond_total else 0.0
        rows.append(row)
    retention_df = pd.DataFrame(rows).sort_values("filter_name").reset_index(drop=True)
    retention_df.to_csv(TABLES_DIR / "filter_retention_summary.csv", index=False)

    lines = [
        "# Filter Candidate Design",
        "",
        "- All filter candidates operate only on the validated Set10_D2_raw_sampled subset.",
        "- They are temporary diagnostic layers and do not modify the locked cfg05 base pipeline.",
        "",
        _df_to_md(defs_df),
        "",
        "Retention summary:",
        "",
        _df_to_md(retention_df),
    ]
    (REPORT_DIR / "filter_candidate_design.md").write_text("\n".join(lines), encoding="utf-8")
    return defs_df, retention_df, masks


def _evaluate_filter_candidates(
    matrix_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    masks: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, str, dict[str, pd.DataFrame]]:
    baseline_scores, baseline_metrics = _pca_metrics_from_mask(matrix_df, meta_df, np.ones(len(meta_df), dtype=bool))
    score_frames = {"unfiltered": baseline_scores}
    rows = [
        {
            "filter_name": "unfiltered",
            **baseline_metrics,
            "retention_fraction_total": 1.0,
        }
    ]
    _plot_pca(baseline_scores, FIGURES_DIR / "filter_unfiltered_spectral_pca.png", "Set10 D2 spectral PCA: unfiltered")
    for name, mask in masks.items():
        score_df, metrics = _pca_metrics_from_mask(matrix_df, meta_df, mask)
        score_frames[name] = score_df
        rows.append(
            {
                "filter_name": name,
                **metrics,
                "retention_fraction_total": float(mask.mean()),
            }
        )
        _plot_pca(score_df, FIGURES_DIR / f"filter_{name}_spectral_pca.png", f"Set10 D2 spectral PCA: {name}")
    metrics_df = pd.DataFrame(rows)
    metrics_df["silhouette_improvement_vs_unfiltered"] = (
        metrics_df["silhouette_by_concentration"] - float(baseline_metrics["silhouette_by_concentration"])
    )
    metrics_df["selection_score"] = (
        metrics_df["silhouette_improvement_vs_unfiltered"]
        + 0.15 * metrics_df["condition_mean_ordering_spearman"].abs()
        + 0.05 * metrics_df["retention_fraction_total"]
    )
    metrics_df.to_csv(TABLES_DIR / "filter_pca_replication_metrics.csv", index=False)
    filt_df = metrics_df[metrics_df["filter_name"].astype(str) != "unfiltered"].copy()
    effective_df = filt_df[
        (filt_df["retention_fraction_total"] < 0.999)
        | (filt_df["silhouette_improvement_vs_unfiltered"].abs() > 1e-6)
    ].copy()
    candidate_df = effective_df if not effective_df.empty else filt_df
    best = candidate_df.sort_values(
        ["selection_score", "silhouette_by_concentration", "retention_fraction_total"],
        ascending=[False, False, False],
    ).iloc[0]
    best_name = str(best["filter_name"])
    improvement = float(best["silhouette_improvement_vs_unfiltered"])
    meaningful = "meaningful" if improvement >= 0.05 else "cosmetic"
    lines = [
        "# Best Filter Decision",
        "",
        f"- winning_filter: `{best_name}`",
        f"- spectral silhouette improvement vs unfiltered: `{improvement:.4f}`",
        f"- condition-mean ordering Spearman: `{best['condition_mean_ordering_spearman']:.4f}`",
        f"- retention fraction: `{best['retention_fraction_total']:.4f}`",
        "",
        "Direct answers:",
        f"1. Which filter wins? `{best_name}`",
        f"2. How much does it improve spectral PCA relative to unfiltered Set10_D2_raw_sampled? `{improvement:.4f}` silhouette units",
        f"3. Is the improvement meaningful or cosmetic? `{meaningful}`",
    ]
    (REPORT_DIR / "best_filter_decision.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics_df, best_name, score_frames


def _build_query_df(matrix_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    spectral_cols = _spectral_cols(matrix_df)
    rows = []
    for i, row in enumerate(meta_df.itertuples(index=False)):
        vec = matrix_df.loc[i, spectral_cols].to_numpy(dtype=float)
        subclass_label = "Set9" if "set9" in str(row.sample_key).lower() else "Set10"
        rows.append(
            {
                "sample_key": str(row.sample_key),
                "dataset_id": "shine_ev_sers",
                "subclass_label": subclass_label,
                "class_label": str(row.class_label),
                "source_file": str(row.source_file),
                "wavenumbers_json": json.dumps([int(c.replace("wn_", "")) for c in spectral_cols]),
                "intensity_json": json.dumps(vec.tolist()),
            }
        )
    return pd.DataFrame(rows)


def _run_bsv_from_matrix(matrix_df: pd.DataFrame, meta_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    query_df = _build_query_df(matrix_df, meta_df)
    grounding_df, mapping_df, harness_config, _ = _prepare_grounding_and_mapping(registries, resolved, CONFIG_SPEC)
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
    meta = meta_df.copy()
    meta["sample_id"] = meta["sample_key"].astype(str)
    meta["trajectory_index"] = meta["trajectory_concentration"].astype(int)
    bsv_df = bsv_df.merge(meta[["sample_key", "class_label", "trajectory_concentration", "source_file", "sample_id", "trajectory_index"]], on=["sample_key", "class_label"], how="left")
    axes = [axis for axis in FIXED_RADAR_AXES if axis in bsv_df.columns]
    delta_df = _cohort_delta(bsv_df[["sample_key"] + axes].copy(), axes).merge(
        bsv_df[["sample_key", "class_label", "trajectory_concentration", "source_file"]],
        on="sample_key",
        how="left",
    )
    family_df = _family_fingerprint_from_retrieval(
        retrieval_df,
        meta[["sample_key", "sample_id", "class_label", "trajectory_concentration", "trajectory_index"]],
    )
    return bsv_df, delta_df, family_df


def _family_wide(family_df: pd.DataFrame, sample_keys: pd.Series) -> pd.DataFrame:
    wide = family_df.pivot(index="sample_key", columns="family", values="family_fraction")
    wide = wide.reindex(sample_keys.astype(str)).fillna(0.0)
    for family in FAMILY_ORDER:
        if family not in wide.columns:
            wide[family] = 0.0
    return wide[FAMILY_ORDER].reset_index()


def _explainability(
    target_scores: np.ndarray,
    bsv_df: pd.DataFrame,
    family_df: pd.DataFrame,
    ext_df: pd.DataFrame | None,
) -> pd.DataFrame:
    axes = [axis for axis in FIXED_RADAR_AXES if axis in bsv_df.columns]
    family_wide = _family_wide(family_df, bsv_df["sample_key"])
    merged = bsv_df[["sample_key"] + axes].copy().merge(family_wide, left_index=False, right_index=False, on="sample_key", how="left")
    ext_cols: list[str] = []
    if ext_df is not None:
        ext_cols = [c for c in ext_df.columns if c not in {"sample_key", "class_label", "trajectory_concentration", "source_file"}]
        merged = merged.merge(ext_df[["sample_key"] + ext_cols], on="sample_key", how="left")
    merged = merged.fillna(0.0)
    specs = {
        "bsv_only": axes,
        "family_only": FAMILY_ORDER,
        "bsv_plus_family": axes + FAMILY_ORDER,
    }
    if ext_cols:
        specs["bsv_plus_family_plus_extension"] = axes + FAMILY_ORDER + ext_cols
    rows = []
    for name, cols in specs.items():
        X = StandardScaler().fit_transform(merged[cols].to_numpy(dtype=float))
        model = LinearRegression().fit(X, target_scores)
        pred = model.predict(X)
        rows.append(
            {
                "model_name": name,
                "r2": float(r2_score(target_scores, pred)),
                "spearman_r": float(pd.Series(target_scores).corr(pd.Series(pred), method="spearman")),
                "pearson_r": float(pd.Series(target_scores).corr(pd.Series(pred), method="pearson")),
                "rmse": float(np.sqrt(np.mean((target_scores - pred) ** 2))),
            }
        )
    return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)


def _pca_metrics_representation(df: pd.DataFrame, axes: list[str]) -> dict[str, float]:
    scores, explained = _fit_pca(df[axes].to_numpy(dtype=float), scale=True)
    score_df = df[["class_label", "trajectory_concentration"]].copy()
    score_df["pc1"] = scores[:, 0]
    score_df["pc2"] = scores[:, 1]
    silhouette = float(silhouette_score(score_df[["pc1", "pc2"]].to_numpy(dtype=float), score_df["class_label"].astype(str)))
    means = score_df.groupby("trajectory_concentration", as_index=False)["pc1"].mean().sort_values("trajectory_concentration")
    return {
        "silhouette": silhouette,
        "ordering_spearman": float(means["trajectory_concentration"].corr(means["pc1"], method="spearman")),
        "pc1_explained_ratio": float(explained[0]),
        "pc2_explained_ratio": float(explained[1]),
        "score_df": score_df,
    }


def _plot_two_pca(score_a: pd.DataFrame, score_b: pd.DataFrame, output_path: Path, title_a: str, title_b: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), squeeze=False)
    for ax, score_df, title in [(axes[0, 0], score_a, title_a), (axes[0, 1], score_b, title_b)]:
        for cond in CONDITION_ORDER:
            class_label = f"D2_{cond}"
            sub = score_df[score_df["class_label"].astype(str) == class_label].copy()
            ax.scatter(
                sub["pc1"].to_numpy(dtype=float),
                sub["pc2"].to_numpy(dtype=float),
                s=18,
                alpha=0.55,
                color=COND_COLORS[cond],
                label=f"{CONCENTRATION_VALUES[cond]} mM",
            )
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="center right")
    fig.tight_layout(rect=[0.0, 0.0, 0.92, 1.0])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_explainability_compare(unfiltered_df: pd.DataFrame, filtered_df: pd.DataFrame, output_path: Path) -> None:
    merged = unfiltered_df.merge(filtered_df, on="model_name", suffixes=("_unfiltered", "_filtered"))
    x = np.arange(len(merged))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.bar(x - width / 2, merged["r2_unfiltered"], width=width, label="unfiltered", color="#6d597a")
    ax.bar(x + width / 2, merged["r2_filtered"], width=width, label="filtered", color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["model_name"], rotation=20)
    ax.set_ylabel("R^2")
    ax.set_title("Filtered vs unfiltered explainability")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _diagnostics_table(
    retained_mask: np.ndarray,
    meta_df: pd.DataFrame,
    bsv_df: pd.DataFrame,
    family_df: pd.DataFrame,
    ext_df: pd.DataFrame,
) -> pd.DataFrame:
    family_summary = _family_summary(family_df)
    merged = bsv_df.merge(family_summary, on="sample_key", how="left").merge(
        ext_df[["sample_key", "guanidino_response", "aromatic_stress_response"]], on="sample_key", how="left"
    )
    merged["retention_status"] = np.where(retained_mask, "retained", "removed")
    features = [
        "substrate_adsorption_bias",
        "top1_dominance",
        "family_entropy",
        "nucleic_acid",
        "small_molecule_metabolite",
        "guanidino_response",
        "aromatic_stress_response",
    ]
    rows = []
    for feat in features:
        ret = merged.loc[merged["retention_status"] == "retained", feat].to_numpy(dtype=float)
        rem = merged.loc[merged["retention_status"] == "removed", feat].to_numpy(dtype=float)
        rows.append(
            {
                "feature_name": feat,
                "mean_retained": float(np.nanmean(ret)) if len(ret) else np.nan,
                "mean_removed": float(np.nanmean(rem)) if len(rem) else np.nan,
                "delta_removed_minus_retained": float(np.nanmean(rem) - np.nanmean(ret)) if len(ret) and len(rem) else np.nan,
            }
        )
    diag_df = pd.DataFrame(rows)
    diag_df.to_csv(TABLES_DIR / "retained_vs_removed_diagnostics.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.barh(diag_df["feature_name"], diag_df["delta_removed_minus_retained"], color="#b56576")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Removed minus retained mean")
    ax.set_title("Retained vs removed diagnostics")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "retained_vs_removed_bsv_diagnostics.png", dpi=240)
    plt.close(fig)
    return diag_df


def _reference_counts_set10() -> dict[str, int]:
    mat = loadmat(DATA_ROOT / "RawDataset119.mat")["clustered"]
    counts = {}
    for idx, cond in zip([9, 10, 11, 12], CONDITION_ORDER, strict=False):
        arr = np.asarray(mat[0, idx - 1])
        counts[f"D2_{cond}"] = int(arr.shape[1])
    return counts


def _transfer_check_set9(best_filter: str) -> pd.DataFrame:
    if best_filter == "paper_aligned_cluster_count":
        out = pd.DataFrame([{"subset_label": "set9_d2_matched", "spectral_silhouette_unfiltered": np.nan, "spectral_silhouette_filtered": np.nan, "bsv_r2_unfiltered": np.nan, "bsv_r2_filtered": np.nan}])
        out.to_csv(TABLES_DIR / "set9_filter_transfer_metrics.csv", index=False)
        return out

    set10_counts = _reference_counts_set10()
    set9_d2_df, _ = _build_set9_matched_subsets(set10_counts)
    fcols = [c for c in set9_d2_df.columns if str(c).startswith("f")]
    matrix_df = set9_d2_df[["sample_key"] + fcols].copy()
    matrix_df = matrix_df.rename(columns={c: f"wn_{int(round(float(RANGE_WAVENUMBERS[int(c[1:])])))}" for c in fcols})
    meta_df = set9_d2_df[["sample_key", "class_label", "trajectory_concentration", "source_file"]].copy()
    meta_df["trajectory_concentration"] = meta_df["trajectory_concentration"].astype(int)

    unfiltered_bsv = pd.read_csv(EXT_ROOT / "tables" / "set9_d2_bsv.csv")
    unfiltered_family = pd.read_csv(EXT_ROOT / "tables" / "set9_d2_family.csv")
    unfiltered_ext = pd.read_csv(EXT_ROOT / "tables" / "set9_d2_extension_features.csv")
    family_summary = _family_summary(unfiltered_family)
    diag_df = (
        unfiltered_bsv.merge(family_summary, on="sample_key", how="left")
        .assign(
            biological_signal=lambda f: f[
                ["nucleic_acid", "protein_peptide", "lipid_membrane", "carbohydrate_glycan", "small_molecule_metabolite"]
            ].sum(axis=1)
        )
    )
    X = StandardScaler().fit_transform(matrix_df[_spectral_cols(matrix_df)].to_numpy(dtype=float))
    pca_scores, _ = _fit_pca(X, scale=False)
    pc5 = pca_scores[:, :5]
    class_labels = meta_df["class_label"].astype(str).to_numpy()
    if best_filter == "pca_cluster_dominant":
        labels = KMeans(n_clusters=3, random_state=0, n_init=20).fit_predict(pc5)
        keep = np.zeros(len(meta_df), dtype=bool)
        for class_label in sorted(set(class_labels)):
            idx = np.where(class_labels == class_label)[0]
            dominant = pd.Series(labels[idx]).value_counts().sort_values(ascending=False).index[0]
            keep[idx] = labels[idx] == dominant
    elif best_filter == "spectral_outlier":
        keep = np.zeros(len(meta_df), dtype=bool)
        for class_label in sorted(set(class_labels)):
            idx = np.where(class_labels == class_label)[0]
            points = pc5[idx]
            dist = np.linalg.norm(points - points.mean(axis=0, keepdims=True), axis=1)
            keep[idx] = dist <= float(np.quantile(dist, 0.90))
    elif best_filter == "substrate_bias":
        substrate_q = float(diag_df["substrate_adsorption_bias"].quantile(0.85))
        bio_q = float(diag_df["biological_signal"].quantile(0.25))
        top1_q = float(diag_df["top1_dominance"].quantile(0.75))
        bad = (
            (diag_df["substrate_adsorption_bias"].to_numpy(dtype=float) > substrate_q)
            & (diag_df["biological_signal"].to_numpy(dtype=float) < bio_q)
            & (diag_df["top1_dominance"].to_numpy(dtype=float) > top1_q)
        )
        keep = ~bad
    else:
        labels = KMeans(n_clusters=3, random_state=0, n_init=20).fit_predict(pc5)
        keep_a = np.zeros(len(meta_df), dtype=bool)
        for class_label in sorted(set(class_labels)):
            idx = np.where(class_labels == class_label)[0]
            dominant = pd.Series(labels[idx]).value_counts().sort_values(ascending=False).index[0]
            keep_a[idx] = labels[idx] == dominant
        keep_b = np.zeros(len(meta_df), dtype=bool)
        for class_label in sorted(set(class_labels)):
            idx = np.where(class_labels == class_label)[0]
            points = pc5[idx]
            dist = np.linalg.norm(points - points.mean(axis=0, keepdims=True), axis=1)
            keep_b[idx] = dist <= float(np.quantile(dist, 0.90))
        substrate_q = float(diag_df["substrate_adsorption_bias"].quantile(0.85))
        bio_q = float(diag_df["biological_signal"].quantile(0.25))
        top1_q = float(diag_df["top1_dominance"].quantile(0.75))
        bad = (
            (diag_df["substrate_adsorption_bias"].to_numpy(dtype=float) > substrate_q)
            & (diag_df["biological_signal"].to_numpy(dtype=float) < bio_q)
            & (diag_df["top1_dominance"].to_numpy(dtype=float) > top1_q)
        )
        keep = keep_a & keep_b & (~bad)

    unfiltered_scores, unfiltered_metrics = _pca_metrics_from_mask(matrix_df, meta_df, np.ones(len(meta_df), dtype=bool))
    filtered_scores, filtered_metrics = _pca_metrics_from_mask(matrix_df, meta_df, keep)
    target_unfiltered = _concentrations_from_class(meta_df["class_label"])
    target_scores = LinearRegression().fit(StandardScaler().fit_transform(matrix_df[_spectral_cols(matrix_df)].to_numpy(dtype=float)), target_unfiltered).predict(
        StandardScaler().fit_transform(matrix_df[_spectral_cols(matrix_df)].to_numpy(dtype=float))
    )
    filtered_bsv, _, filtered_family = _run_bsv_from_matrix(matrix_df.loc[keep].reset_index(drop=True), meta_df.loc[keep].reset_index(drop=True))
    filtered_ext = unfiltered_ext[unfiltered_ext["sample_key"].astype(str).isin(meta_df.loc[keep, "sample_key"].astype(str))].copy().reset_index(drop=True)
    explain_unfiltered = _explainability(target_scores, unfiltered_bsv, unfiltered_family, unfiltered_ext)
    explain_filtered = _explainability(target_scores[keep], filtered_bsv, filtered_family, filtered_ext)
    rows = [
        {
            "subset_label": "set9_d2_matched",
            "spectral_silhouette_unfiltered": float(unfiltered_metrics["silhouette_by_concentration"]),
            "spectral_silhouette_filtered": float(filtered_metrics["silhouette_by_concentration"]),
            "bsv_r2_unfiltered": float(
                explain_unfiltered.loc[explain_unfiltered["model_name"].astype(str) == "bsv_plus_family", "r2"].iloc[0]
            ),
            "bsv_r2_filtered": float(
                explain_filtered.loc[explain_filtered["model_name"].astype(str) == "bsv_plus_family", "r2"].iloc[0]
            ),
        }
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "set9_filter_transfer_metrics.csv", index=False)
    return out


def main() -> None:
    _ensure_dirs()
    data = _load_set10_base()
    matrix_df = data["matrix_df"]
    meta_df = data["meta_df"]
    bsv_df = data["bsv_df"]
    delta_df = data["delta_df"]
    family_df = data["family_df"]
    ext_df = data["ext_df"]
    axis_scores_df = data["axis_scores"]
    target_scores = axis_scores_df["axis_score"].to_numpy(dtype=float)

    defs_df, retention_df, masks = _filter_candidates(matrix_df, meta_df, bsv_df, family_df)
    filter_metrics_df, best_filter, score_frames = _evaluate_filter_candidates(matrix_df, meta_df, masks)
    keep_mask = masks[best_filter]

    # Unfiltered existing artifacts
    unfiltered_bsv = bsv_df.copy()
    unfiltered_delta = delta_df.copy()
    unfiltered_family = family_df.copy()
    unfiltered_ext = ext_df.copy()
    unfiltered_bsv.to_csv(TABLES_DIR / "unfiltered_set10_day2_bsv.csv", index=False)
    unfiltered_delta.to_csv(TABLES_DIR / "unfiltered_set10_day2_delta_bsv.csv", index=False)
    unfiltered_family.to_csv(TABLES_DIR / "unfiltered_set10_day2_family.csv", index=False)

    # Filtered rerun
    filt_matrix_df = matrix_df.loc[keep_mask].reset_index(drop=True)
    filt_meta_df = meta_df.loc[keep_mask].reset_index(drop=True)
    filtered_bsv, filtered_delta, filtered_family = _run_bsv_from_matrix(filt_matrix_df, filt_meta_df)
    filtered_ext = ext_df[ext_df["sample_key"].astype(str).isin(filt_meta_df["sample_key"].astype(str))].copy().reset_index(drop=True)
    filtered_bsv.to_csv(TABLES_DIR / "filtered_set10_day2_bsv.csv", index=False)
    filtered_delta.to_csv(TABLES_DIR / "filtered_set10_day2_delta_bsv.csv", index=False)
    filtered_family.to_csv(TABLES_DIR / "filtered_set10_day2_family.csv", index=False)

    # Representation comparison
    spectral_unf = score_frames["unfiltered"]
    spectral_flt = score_frames[best_filter]
    _plot_two_pca(
        spectral_unf,
        spectral_flt,
        FIGURES_DIR / "filtered_vs_unfiltered_spectral_pca_comparison.png",
        "Unfiltered spectral PCA",
        f"Filtered spectral PCA: {best_filter}",
    )

    bsv_axes_unf = [axis for axis in FIXED_RADAR_AXES if axis in unfiltered_bsv.columns]
    bsv_axes_flt = [axis for axis in FIXED_RADAR_AXES if axis in filtered_bsv.columns]
    bsv_unf_metrics = _pca_metrics_representation(unfiltered_bsv, bsv_axes_unf)
    bsv_flt_metrics = _pca_metrics_representation(filtered_bsv, bsv_axes_flt)
    _plot_two_pca(
        bsv_unf_metrics["score_df"],
        bsv_flt_metrics["score_df"],
        FIGURES_DIR / "filtered_vs_unfiltered_bsv_pca_comparison.png",
        "Unfiltered BSV PCA",
        "Filtered BSV PCA",
    )

    delta_axes_unf = [axis for axis in FIXED_RADAR_AXES if axis in unfiltered_delta.columns]
    delta_axes_flt = [axis for axis in FIXED_RADAR_AXES if axis in filtered_delta.columns]
    delta_unf_metrics = _pca_metrics_representation(unfiltered_delta, delta_axes_unf)
    delta_flt_metrics = _pca_metrics_representation(filtered_delta, delta_axes_flt)
    _plot_two_pca(
        delta_unf_metrics["score_df"],
        delta_flt_metrics["score_df"],
        FIGURES_DIR / "filtered_vs_unfiltered_delta_bsv_pca_comparison.png",
        "Unfiltered delta-BSV PCA",
        "Filtered delta-BSV PCA",
    )

    explain_unfiltered = _explainability(target_scores, unfiltered_bsv, unfiltered_family, unfiltered_ext)
    filtered_target = axis_scores_df[axis_scores_df["sample_key"].astype(str).isin(filt_meta_df["sample_key"].astype(str))]["axis_score"].to_numpy(dtype=float)
    explain_filtered = _explainability(filtered_target, filtered_bsv, filtered_family, filtered_ext)
    _plot_explainability_compare(
        explain_unfiltered,
        explain_filtered,
        FIGURES_DIR / "filtered_vs_unfiltered_explainability.png",
    )

    compare_rows = [
        {
            "representation_name": "spectral",
            "unfiltered_silhouette": float(filter_metrics_df.loc[filter_metrics_df["filter_name"] == "unfiltered", "silhouette_by_concentration"].iloc[0]),
            "filtered_silhouette": float(filter_metrics_df.loc[filter_metrics_df["filter_name"] == best_filter, "silhouette_by_concentration"].iloc[0]),
            "unfiltered_ordering_spearman": float(filter_metrics_df.loc[filter_metrics_df["filter_name"] == "unfiltered", "condition_mean_ordering_spearman"].iloc[0]),
            "filtered_ordering_spearman": float(filter_metrics_df.loc[filter_metrics_df["filter_name"] == best_filter, "condition_mean_ordering_spearman"].iloc[0]),
        },
        {
            "representation_name": "bsv",
            "unfiltered_silhouette": float(bsv_unf_metrics["silhouette"]),
            "filtered_silhouette": float(bsv_flt_metrics["silhouette"]),
            "unfiltered_ordering_spearman": float(bsv_unf_metrics["ordering_spearman"]),
            "filtered_ordering_spearman": float(bsv_flt_metrics["ordering_spearman"]),
        },
        {
            "representation_name": "delta_bsv",
            "unfiltered_silhouette": float(delta_unf_metrics["silhouette"]),
            "filtered_silhouette": float(delta_flt_metrics["silhouette"]),
            "unfiltered_ordering_spearman": float(delta_unf_metrics["ordering_spearman"]),
            "filtered_ordering_spearman": float(delta_flt_metrics["ordering_spearman"]),
        },
    ]
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(TABLES_DIR / "filtered_vs_unfiltered_bsv_comparison.csv", index=False)

    explain_compare = explain_unfiltered.merge(explain_filtered, on="model_name", suffixes=("_unfiltered", "_filtered"))
    explain_compare.to_csv(TABLES_DIR / "filtered_vs_unfiltered_explainability.csv", index=False)

    diag_df = _diagnostics_table(keep_mask, meta_df, unfiltered_bsv, unfiltered_family, unfiltered_ext)
    set9_transfer_df = _transfer_check_set9(best_filter)

    spectral_improvement = float(filter_metrics_df.loc[filter_metrics_df["filter_name"] == best_filter, "silhouette_improvement_vs_unfiltered"].iloc[0])
    bsv_r2_unfiltered = float(
        explain_compare.loc[explain_compare["model_name"] == "bsv_plus_family", "r2_unfiltered"].iloc[0]
    )
    bsv_r2_filtered = float(
        explain_compare.loc[explain_compare["model_name"] == "bsv_plus_family", "r2_filtered"].iloc[0]
    )
    ext_r2_unfiltered = float(
        explain_compare.loc[explain_compare["model_name"] == "bsv_plus_family_plus_extension", "r2_unfiltered"].iloc[0]
    )
    ext_r2_filtered = float(
        explain_compare.loc[explain_compare["model_name"] == "bsv_plus_family_plus_extension", "r2_filtered"].iloc[0]
    )

    if spectral_improvement >= 0.05 and (bsv_r2_filtered - bsv_r2_unfiltered >= 0.03 or ext_r2_filtered - ext_r2_unfiltered >= 0.03):
        decision_label = "spectral_and_bsv_help"
    elif spectral_improvement >= 0.05:
        decision_label = "spectral_only_help"
    else:
        decision_label = "no_help"

    lines = [
        "# Filtering Experiment Decision",
        "",
        f"- winning_filter: `{best_filter}`",
        f"- decision_label: `{decision_label}`",
        f"- spectral silhouette improvement: `{spectral_improvement:.4f}`",
        f"- BSV explainability change: `{(bsv_r2_filtered - bsv_r2_unfiltered):.4f}`",
        f"- BSV+family+extension explainability change: `{(ext_r2_filtered - ext_r2_unfiltered):.4f}`",
        "",
        "Direct answers:",
        f"1. Can a reproducible filtering layer recover more paper-like SHINE PCA structure? `{'yes' if spectral_improvement >= 0.05 else 'not meaningfully'}`",
        f"2. Does filtering help BSV capture the SHINE Day2 signal better? `{'yes' if (bsv_r2_filtered - bsv_r2_unfiltered >= 0.03 or ext_r2_filtered - ext_r2_unfiltered >= 0.03) else 'no clear improvement'}`",
        f"3. Is SHINE failure mostly due to unfiltered raw heterogeneity? `{'partly' if decision_label != 'no_help' else 'not supported by this experiment'}`",
        f"4. Should GAIRA keep: `{('a SHINE-specific temporary filter lane' if decision_label != 'no_help' else 'no SHINE filter')}`",
    ]
    (REPORT_DIR / "filtering_experiment_decision.md").write_text("\n".join(lines), encoding="utf-8")

    report_lines = [
        "# GAIRAv3 SHINE Filtering Test Report",
        "",
        "## 1. Why this experiment was needed",
        "",
        "- The paper-side SHINE structure likely depends on filtered or clustered spectra rather than the full raw measurement distribution.",
        "- This experiment tests whether a temporary reproducible filtering layer can recover that structure and improve BSV readout on the validated Set10 Day2 subset.",
        "",
        "## 2. Filter candidates",
        "",
        _df_to_md(defs_df),
        "",
        "Retention summary:",
        "",
        _df_to_md(retention_df),
        "",
        "## 3. Best spectral filter",
        "",
        _df_to_md(filter_metrics_df),
        "",
        (REPORT_DIR / "best_filter_decision.md").read_text(encoding="utf-8"),
        "",
        "## 4. BSV before vs after filtering",
        "",
        _df_to_md(compare_df),
        "",
        "Explainability comparison:",
        "",
        _df_to_md(explain_compare),
        "",
        "## 5. Removed vs retained biology",
        "",
        _df_to_md(diag_df),
        "",
        "## 6. Final conclusion",
        "",
        (REPORT_DIR / "filtering_experiment_decision.md").read_text(encoding="utf-8"),
        "",
        "Optional Set9 transfer check:",
        "",
        _df_to_md(set9_transfer_df),
    ]
    report_md = REPORT_DIR / "GAIRAv3_SHINE_filtering_test_report.md"
    report_md.write_text("\n".join(report_lines), encoding="utf-8")
    figure_paths = [
        FIGURES_DIR / f"filter_{best_filter}_spectral_pca.png",
        FIGURES_DIR / "filtered_vs_unfiltered_spectral_pca_comparison.png",
        FIGURES_DIR / "filtered_vs_unfiltered_bsv_pca_comparison.png",
        FIGURES_DIR / "filtered_vs_unfiltered_delta_bsv_pca_comparison.png",
        FIGURES_DIR / "filtered_vs_unfiltered_explainability.png",
        FIGURES_DIR / "retained_vs_removed_bsv_diagnostics.png",
    ]
    build_pdf_report(report_md, [p for p in figure_paths if p.exists()], REPORT_DIR / "GAIRAv3_SHINE_filtering_test_report.pdf")


if __name__ == "__main__":
    main()
