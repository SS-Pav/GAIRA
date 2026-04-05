from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from gaira.autoresearch_storage import (
    DEFAULT_STORAGE_CONFIG_PATH,
    initialize_autoresearch_sprint,
    load_autoresearch_storage_config,
)
from gaira.demo.autoresearch_pass5_utils import (
    Pass5HarnessConfig,
    apply_pass5_filter_mode,
    build_bsv_profiles_pass5,
    classify_compound_family,
)
from gaira.demo.gaira_experiment_runner_utils import (
    ResolvedExperiment,
    build_source_role_sets,
    load_architecture_registries,
    load_grounding_family_dataframe,
    load_query_dataframe,
    retrieval_hit_summary,
)
from gaira.demo.gaira_pilot_utils import (
    build_class_axis_entropy,
    build_class_neighborhood_entropy,
    build_class_top1_dominance,
    build_class_topk_neighborhood_composition,
    compute_stability_tables,
    pairwise_delta_bsv,
)
from gaira.demo.raw_bsv_pilot_utils import (
    ALL_AXES,
    apply_source_role_policy,
    build_group_mean_query_df,
    compute_local_pca,
    load_ontology_rules,
    map_references_to_axes,
)


ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = ROOT / "reports" / "gaira_architecture_scaffold_v2"
PHASE1_DIR = ROOT / "reports" / "gaira_phase1_registry_audit_v2"
PASS5_TABLE = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pass5_saturation_fix/tables/calibration_results_ranked.csv"
)
SPRINT_SUBDIR = "pilot1a_celltype_probe1_v2"
ONTOLOGY_PATH = ROOT / "config" / "phase2_bsv_ontology_rules_v2.csv"
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


def _resolve_alias(registries, subset_alias: str) -> ResolvedExperiment:
    matches = registries.dataset_experiments[
        registries.dataset_experiments["subset_alias"].astype(str) == str(subset_alias)
    ].copy()
    if matches.empty or len(matches) > 1:
        raise RuntimeError(f"Could not resolve unique subset alias {subset_alias}")
    dataset_row = matches.iloc[0]
    experiment_row = pd.Series(
        {
            "experiment_id": f"pilot1a_probe1_v2__{subset_alias}",
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


def _pca_from_matrix(
    matrix: np.ndarray,
    *,
    labels: list[str],
    sample_keys: list[str],
    n_components: int = 3,
) -> pd.DataFrame:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    max_components = min(n_components, u.shape[1], matrix.shape[0])
    coords = u[:, :max_components] * s[:max_components]
    explained = (s**2) / np.maximum((s**2).sum(), 1e-12)
    df = pd.DataFrame(
        {
            "sample_key": sample_keys,
            "class_label": labels,
        }
    )
    for i in range(max_components):
        df[f"pc{i+1}"] = coords[:, i]
        df[f"pc{i+1}_explained_ratio"] = float(explained[i])
    return df


def _axes_present(df: pd.DataFrame) -> list[str]:
    return [axis for axis in ALL_AXES if axis in df.columns]


def _draw_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    output_path: Path,
    title: str,
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(7.2, 0.8 * len(col_labels) + 2.5), max(4.6, 0.55 * len(row_labels) + 2.4)))
    if center is None:
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        lim = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))), 1e-9)
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_spectral_pca(pca_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.3))
    labels = sorted(pca_df["class_label"].astype(str).unique().tolist())
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(labels))))
    for color, label in zip(colors, labels, strict=False):
        sub = pca_df[pca_df["class_label"].astype(str) == label].copy()
        ax.scatter(
            sub["pc1"].to_numpy(dtype=float),
            sub["pc2"].to_numpy(dtype=float),
            s=28,
            alpha=0.85,
            color=color,
            label=label,
            edgecolors="none",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Spectral PCA: Original Dataset")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_bsv_pca(pca_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.3))
    labels = sorted(pca_df["class_label"].astype(str).unique().tolist())
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(labels))))
    for color, label in zip(colors, labels, strict=False):
        sub = pca_df[pca_df["class_label"].astype(str) == label].copy()
        ax.scatter(
            sub["pc1"].to_numpy(dtype=float),
            sub["pc2"].to_numpy(dtype=float),
            s=28,
            alpha=0.85,
            color=color,
            label=label,
            edgecolors="none",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_bsv_class_mean_pca(pca_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    xvals = pca_df["pc1"].to_numpy(dtype=float)
    yvals = pca_df["pc2"].to_numpy(dtype=float)
    ax.scatter(
        xvals,
        yvals,
        s=55,
        color="#1f77b4",
        alpha=0.9,
    )
    for row in pca_df.itertuples(index=False):
        ax.text(float(row.pc1) + 0.005, float(row.pc2) + 0.005, str(row.class_label), fontsize=9)
    xrange = max(float(xvals.max() - xvals.min()), 0.05)
    yrange = max(float(yvals.max() - yvals.min()), 0.05)
    ax.set_xlim(float(xvals.min()) - 0.15 * xrange, float(xvals.max()) + 0.22 * xrange)
    ax.set_ylim(float(yvals.min()) - 0.15 * yrange, float(yvals.max()) + 0.22 * yrange)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_bsv_class_mean_overlay(output_path: Path, overlay_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    colors = {"baseline": "#577590", "cfg05": "#f3722c", "cfg08": "#43aa8b"}
    markers = {"baseline": "o", "cfg05": "s", "cfg08": "^"}
    all_x = []
    all_y = []
    for short_label in ["baseline", "cfg05", "cfg08"]:
        sub = overlay_df[overlay_df["config_short_label"] == short_label].copy()
        xvals = sub["pc1"].to_numpy(dtype=float)
        yvals = sub["pc2"].to_numpy(dtype=float)
        all_x.extend(xvals.tolist())
        all_y.extend(yvals.tolist())
        ax.scatter(
            xvals,
            yvals,
            s=62,
            color=colors[short_label],
            marker=markers[short_label],
            alpha=0.88,
            label=short_label,
        )
        for row in sub.itertuples(index=False):
            ax.text(float(row.pc1) + 0.004, float(row.pc2) + 0.004, f"{row.class_label}-{row.config_short_label}", fontsize=7)
    xarr = np.asarray(all_x, dtype=float)
    yarr = np.asarray(all_y, dtype=float)
    xrange = max(float(xarr.max() - xarr.min()), 0.05)
    yrange = max(float(yarr.max() - yarr.min()), 0.05)
    ax.set_xlim(float(xarr.min()) - 0.15 * xrange, float(xarr.max()) + 0.28 * xrange)
    ax.set_ylim(float(yarr.min()) - 0.15 * yrange, float(yarr.max()) + 0.22 * yrange)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Class-Mean BSV PCA Overlay")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_bsv_heatmap(class_mean_bsv: pd.DataFrame, output_path: Path, title: str) -> None:
    axes = _axes_present(class_mean_bsv)
    heat = class_mean_bsv.set_index("class_label")[axes]
    _draw_heatmap(
        heat.to_numpy(dtype=float),
        heat.index.astype(str).tolist(),
        axes,
        output_path,
        title,
        cmap="viridis",
        vmin=0.0,
        vmax=max(1.0, float(np.nanmax(heat.to_numpy(dtype=float)))),
    )


def _plot_pairwise_delta_heatmap(pairwise_delta_df: pd.DataFrame, axis: str, output_path: Path) -> None:
    heat = pairwise_delta_df.pivot(index="group_label", columns="reference_group", values=axis)
    heat = heat.reindex(sorted(heat.index), axis=0).reindex(sorted(heat.columns), axis=1)
    _draw_heatmap(
        heat.to_numpy(dtype=float),
        heat.index.astype(str).tolist(),
        heat.columns.astype(str).tolist(),
        output_path,
        f"Pairwise Delta BSV Heatmap: {axis}",
        cmap="coolwarm",
        center=0.0,
    )


def _plot_radar_grid(class_mean_bsv: pd.DataFrame, output_path: Path, title: str) -> None:
    axes = _axes_present(class_mean_bsv)
    labels = sorted(class_mean_bsv["class_label"].astype(str).tolist())
    ncols = 3
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.8 * ncols, 4.4 * nrows),
        subplot_kw={"projection": "polar"},
    )
    axs = np.atleast_1d(axs).ravel()
    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    color = "#1f77b4"
    for ax in axs[len(labels) :]:
        ax.axis("off")
    for ax, label in zip(axs, labels, strict=False):
        row = class_mean_bsv[class_mean_bsv["class_label"].astype(str) == label].iloc[0]
        values = [float(row[axis]) for axis in axes]
        values_closed = values + values[:1]
        ax.plot(angles_closed, values_closed, color=color, linewidth=2.2)
        ax.fill(angles_closed, values_closed, color=color, alpha=0.28)
        ax.set_xticks(angles)
        ax.set_xticklabels(axes, fontsize=7)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=6)
        ax.set_title(label, fontsize=10, pad=12)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_neighborhood_grid(class_neighborhood_df: pd.DataFrame, output_path: Path, title: str, top_n: int = 8) -> None:
    labels = sorted(class_neighborhood_df["class_label"].astype(str).unique().tolist())
    ncols = 2
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.0, 3.4 * nrows))
    axs = np.atleast_1d(axs).ravel()
    for ax in axs[len(labels) :]:
        ax.axis("off")
    for ax, label in zip(axs, labels, strict=False):
        sub = class_neighborhood_df[class_neighborhood_df["class_label"].astype(str) == label].copy()
        sub = sub.sort_values("support_fraction", ascending=False).head(top_n)
        y = np.arange(len(sub))
        ax.barh(y, sub["support_fraction"], color="#2a9d8f")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["compound_label"].astype(str).tolist(), fontsize=8)
        ax.invert_yaxis()
        ax.set_title(label, fontsize=10)
        ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _stability_summary(config_name: str, intra_df: pd.DataFrame, inter_df: pd.DataFrame) -> pd.DataFrame:
    axes = _axes_present(intra_df)
    within = intra_df.copy()
    within["mean_variance"] = within[axes].mean(axis=1)
    nonident = inter_df[inter_df["class_label_a"] != inter_df["class_label_b"]].copy()
    nonident["pair_key"] = nonident.apply(
        lambda r: "::".join(sorted([str(r["class_label_a"]), str(r["class_label_b"])])),
        axis=1,
    )
    unique_pairs = nonident.drop_duplicates("pair_key").copy()
    farthest = unique_pairs.sort_values("euclidean_distance", ascending=False).head(3)
    closest = unique_pairs.sort_values("euclidean_distance", ascending=True).head(3)
    return pd.DataFrame(
        [
            {
                "config_name": config_name,
                "mean_intra_class_bsv_variance": float(within["mean_variance"].mean()),
                "median_intra_class_bsv_variance": float(within["mean_variance"].median()),
                "mean_inter_class_bsv_distance": float(unique_pairs["euclidean_distance"].mean()),
                "median_inter_class_bsv_distance": float(unique_pairs["euclidean_distance"].median()),
                "within_between_ratio": float(within["mean_variance"].mean() / max(unique_pairs["euclidean_distance"].mean(), 1e-12)),
                "top_3_farthest_class_pairs": "; ".join(
                    [f"{r.class_label_a} vs {r.class_label_b} ({float(r.euclidean_distance):.4f})" for r in farthest.itertuples(index=False)]
                ),
                "top_3_closest_nonidentical_class_pairs": "; ".join(
                    [f"{r.class_label_a} vs {r.class_label_b} ({float(r.euclidean_distance):.4f})" for r in closest.itertuples(index=False)]
                ),
            }
        ]
    )


def _dominant_family_summary(class_neighborhood_df: pd.DataFrame) -> str:
    top = (
        class_neighborhood_df.sort_values(["class_label", "support_fraction"], ascending=[True, False])
        .groupby("class_label", sort=True)
        .head(1)
        .copy()
    )
    top["family"] = top["compound_label"].astype(str).map(lambda x: classify_compound_family(x, ""))
    grouped = top.groupby("family", sort=True).size().sort_values(ascending=False)
    return "; ".join([f"{family}:{int(count)}" for family, count in grouped.items()])


def _interpretation_note(mean_neighborhood_entropy: float, mean_top1_dominance: float, within_between_ratio: float) -> str:
    if mean_neighborhood_entropy < 1.0 and mean_top1_dominance > 0.7:
        return "narrow and dominance-heavy"
    if mean_neighborhood_entropy > 1.6 and within_between_ratio > 0.008:
        return "broadened but more diffuse"
    if mean_neighborhood_entropy > 1.1 and mean_top1_dominance < 0.65:
        return "broadened with usable structure"
    return "intermediate tradeoff"


def _plot_within_between(output_path: Path, comparator_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    x = np.arange(len(comparator_df))
    width = 0.36
    ax.bar(x - width / 2, comparator_df["mean_intra_class_bsv_variance"], width=width, label="mean intra variance", color="#577590")
    ax.bar(x + width / 2, comparator_df["mean_inter_class_bsv_distance"], width=width, label="mean inter distance", color="#43aa8b")
    ax.set_xticks(x)
    ax.set_xticklabels(comparator_df["config_short_label"].tolist())
    ax.set_title("Pilot 1a Within-class vs Between-class Comparison")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.8, 1.0])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_entropy_dominance(output_path: Path, comparator_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.5))
    x = np.arange(len(comparator_df))
    width = 0.26
    ax.bar(x - width, comparator_df["mean_neighborhood_entropy"], width=width, label="mean neighborhood entropy", color="#2a9d8f")
    ax.bar(x, comparator_df["mean_axis_entropy"], width=width, label="mean axis entropy", color="#e9c46a")
    ax.bar(x + width, comparator_df["mean_top1_dominance"], width=width, label="mean top1 dominance", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(comparator_df["config_short_label"].tolist())
    ax.set_title("Pilot 1a Entropy and Dominance Comparison")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_tradeoff(output_path: Path, comparator_df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.8))
    metrics = [
        ("within_between_ratio", "Within/Between Ratio"),
        ("mean_neighborhood_entropy", "Fingerprint Breadth"),
        ("mean_top1_dominance", "Neighborhood Dominance"),
    ]
    colors = ["#577590", "#43aa8b", "#e76f51"]
    for ax, (col, title), color in zip(axs, metrics, colors, strict=False):
        ax.bar(comparator_df["config_short_label"].tolist(), comparator_df[col].to_numpy(dtype=float), color=color)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    fig.suptitle("Pilot 1a Config Tradeoff Summary")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _run_config(*, registries, resolved: ResolvedExperiment, config_spec: dict[str, object], sprint_root: Path, spectral_pca_df: pd.DataFrame) -> dict[str, object]:
    harness_config = _config_to_harness(config_spec)
    query_df = load_query_dataframe(resolved.dataset_row)

    original_grounding = list(resolved.grounding_family_names)
    object.__setattr__(resolved, "grounding_family_names", ["universal_biochemical_grounding"])
    try:
        grounding_df, family_to_sources, unavailable_sources = load_grounding_family_dataframe(resolved, registries)
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

    per_spectrum_bsv_df, per_spectrum_retrieval_df = build_bsv_profiles_pass5(
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
    class_mean_query_df = build_group_mean_query_df(query_df, group_col="class_label")
    class_mean_bsv_df, class_mean_retrieval_df = build_bsv_profiles_pass5(
        class_mean_query_df,
        grounding_df,
        mapping_df,
        top_k=harness_config.top_k,
        similarity_metric="cosine",
        weighting_mode=harness_config.weighting_mode,
        weighting_param=harness_config.weighting_param,
        diversity_mode=harness_config.diversity_mode,
        family_min_coverage=harness_config.family_min_coverage,
    )

    pairwise_delta_df = pairwise_delta_bsv(class_mean_bsv_df)
    intra_class_variance_df, inter_class_distance_df = compute_stability_tables(per_spectrum_bsv_df, class_mean_bsv_df)
    class_neighborhood_df = build_class_topk_neighborhood_composition(per_spectrum_retrieval_df)
    class_neighborhood_entropy_df = build_class_neighborhood_entropy(class_neighborhood_df)
    class_top1_dominance_df = build_class_top1_dominance(class_neighborhood_df)
    class_axis_entropy_df = build_class_axis_entropy(class_mean_bsv_df)
    retrieval_summary_by_class_df = retrieval_hit_summary(class_mean_retrieval_df)
    within_between_summary_df = _stability_summary(str(config_spec["config_id"]), intra_class_variance_df, inter_class_distance_df)

    bsv_axes = _axes_present(per_spectrum_bsv_df)
    bsv_matrix = per_spectrum_bsv_df[bsv_axes].to_numpy(dtype=float)
    bsv_pca_df = _pca_from_matrix(
        bsv_matrix,
        labels=per_spectrum_bsv_df["class_label"].astype(str).tolist(),
        sample_keys=per_spectrum_bsv_df["sample_key"].astype(str).tolist(),
    )
    class_mean_matrix = class_mean_bsv_df[bsv_axes].to_numpy(dtype=float)
    bsv_class_mean_pca_df = _pca_from_matrix(
        class_mean_matrix,
        labels=class_mean_bsv_df["class_label"].astype(str).tolist(),
        sample_keys=class_mean_bsv_df["sample_key"].astype(str).tolist(),
    )

    run_dir = sprint_root / "runs" / str(config_spec["config_id"])
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    report_dir = run_dir / "report"
    for directory in [tables_dir, figures_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    per_spectrum_bsv_df.to_csv(tables_dir / "per_spectrum_bsv.csv", index=False)
    class_mean_bsv_df.to_csv(tables_dir / "class_mean_bsv.csv", index=False)
    pairwise_delta_df.to_csv(tables_dir / "pairwise_delta_bsv.csv", index=False)
    intra_class_variance_df.to_csv(tables_dir / "intra_class_bsv_variance.csv", index=False)
    inter_class_distance_df.to_csv(tables_dir / "inter_class_bsv_distance.csv", index=False)
    class_neighborhood_df.to_csv(tables_dir / "class_topk_neighborhood_composition.csv", index=False)
    class_neighborhood_entropy_df.to_csv(tables_dir / "class_neighborhood_entropy.csv", index=False)
    class_top1_dominance_df.to_csv(tables_dir / "class_top1_dominance.csv", index=False)
    class_axis_entropy_df.to_csv(tables_dir / "class_axis_entropy.csv", index=False)
    retrieval_summary_by_class_df.to_csv(tables_dir / "retrieval_hit_summary_by_class.csv", index=False)
    per_spectrum_retrieval_df.to_csv(tables_dir / "per_spectrum_retrieval_hits.csv", index=False)
    spectral_pca_df.to_csv(tables_dir / "pca_coordinates_spectral.csv", index=False)
    bsv_pca_df.to_csv(tables_dir / "pca_coordinates_bsv.csv", index=False)
    bsv_class_mean_pca_df.to_csv(tables_dir / "pca_coordinates_bsv_class_mean.csv", index=False)
    within_between_summary_df.to_csv(tables_dir / "config_within_between_summary.csv", index=False)

    _plot_bsv_heatmap(class_mean_bsv_df, figures_dir / f"class_mean_bsv_heatmap_{config_spec['short_label']}.png", f"Class Mean BSV Heatmap ({config_spec['short_label']})")
    _plot_pairwise_delta_heatmap(pairwise_delta_df, "small_molecule_metabolite", figures_dir / f"pairwise_delta_bsv_heatmap_small_molecule_metabolite_{config_spec['short_label']}.png")
    _plot_bsv_pca(bsv_pca_df, figures_dir / f"pca_bsv_{config_spec['short_label']}.png", f"BSV PCA ({config_spec['short_label']})")
    _plot_bsv_class_mean_pca(bsv_class_mean_pca_df, figures_dir / f"pca_bsv_class_mean_{config_spec['short_label']}.png", f"BSV PCA Class Means ({config_spec['short_label']})")
    _plot_radar_grid(class_mean_bsv_df, figures_dir / f"radar_fingerprint_grid_{config_spec['short_label']}.png", f"Radar Fingerprint Grid ({config_spec['short_label']})")
    _plot_neighborhood_grid(class_neighborhood_df, figures_dir / f"neighborhood_grid_{config_spec['short_label']}.png", f"Neighborhood Grid ({config_spec['short_label']})")

    (report_dir / "run_config.json").write_text(
        json.dumps(
            {
                "representation_mode": "raw_direct_bsv_input",
                "grounding_mode": "universal_only",
                "universal_grounding_filter_mode": config_spec["filter_mode"],
                "aggregation_mode": "class_mean_spectrum_then_bsv",
                "ontology_mode": "tier1_plus_subclass",
                "similarity_metric": "cosine",
                "plausibility_scoring_mode": "baseline_plausibility",
                "pca_grouping_mode": "class_label_groups",
                "top_k": config_spec["top_k"],
                "weighting_mode": config_spec["weighting_mode"],
                "weighting_param": config_spec["weighting_param"],
                "diversity_mode": config_spec["diversity_mode"],
                "available_sources": sorted(grounding_df["source_key"].astype(str).unique().tolist()),
                "unavailable_sources": unavailable_sources,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "config_spec": config_spec,
        "class_mean_bsv_df": class_mean_bsv_df,
        "class_neighborhood_df": class_neighborhood_df,
        "class_neighborhood_entropy_df": class_neighborhood_entropy_df,
        "class_top1_dominance_df": class_top1_dominance_df,
        "class_axis_entropy_df": class_axis_entropy_df,
        "within_between_summary_df": within_between_summary_df,
        "bsv_class_mean_pca_df": bsv_class_mean_pca_df,
    }


def _build_report(report_path: Path, comparator_df: pd.DataFrame, pass5_ranked_df: pd.DataFrame) -> None:
    baseline = comparator_df[comparator_df["config_short_label"] == "baseline"].iloc[0]
    cfg05 = comparator_df[comparator_df["config_short_label"] == "cfg05"].iloc[0]
    cfg08 = comparator_df[comparator_df["config_short_label"] == "cfg08"].iloc[0]
    lines = [
        "# GAIRAv3 Pilot 1a Celltype Probe1 v2 Report",
        "",
        "## 1. Overview",
        "- This rerun is cell-type-only on Probe 1 and compares the three fixed deterministic configurations directly.",
        "- The original spectral PCA is identical across configs because the configs do not alter the spectra. It is shown once as the canonical dataset geometry.",
        "- The BSV-space PCAs are config-specific because the fingerprint geometry does change with retrieval and weighting.",
        "",
        "## 2. Cell-type Fingerprints by Config",
        f"- `baseline`: narrow profile, mean neighborhood entropy `{baseline['mean_neighborhood_entropy']:.4f}`, mean top1 dominance `{baseline['mean_top1_dominance']:.4f}`.",
        f"- `cfg05`: broadened profile, mean neighborhood entropy `{cfg05['mean_neighborhood_entropy']:.4f}`, mean top1 dominance `{cfg05['mean_top1_dominance']:.4f}`.",
        f"- `cfg08`: broadest profile, mean neighborhood entropy `{cfg08['mean_neighborhood_entropy']:.4f}`, mean top1 dominance `{cfg08['mean_top1_dominance']:.4f}`.",
        "- The relevant question is not just breadth. It is whether broadening keeps class-specific biochemical identity rather than making the map diffuse.",
        "",
        "## 3. Stability Analysis",
        f"- `baseline`: mean intra `{baseline['mean_intra_class_bsv_variance']:.6f}`, mean inter `{baseline['mean_inter_class_bsv_distance']:.6f}`, ratio `{baseline['within_between_ratio']:.6f}`.",
        f"- `cfg05`: mean intra `{cfg05['mean_intra_class_bsv_variance']:.6f}`, mean inter `{cfg05['mean_inter_class_bsv_distance']:.6f}`, ratio `{cfg05['within_between_ratio']:.6f}`.",
        f"- `cfg08`: mean intra `{cfg08['mean_intra_class_bsv_variance']:.6f}`, mean inter `{cfg08['mean_inter_class_bsv_distance']:.6f}`, ratio `{cfg08['within_between_ratio']:.6f}`.",
        "- Smaller absolute between-class distance does not automatically mean worse biology. If the BSV geometry becomes more multi-axis, some compression of raw Euclidean gaps is expected.",
        "- The main question is whether within-class structure stays tight while class identity remains visually separable in BSV PCA space.",
        "",
        "## 4. Neighborhood Behavior",
        f"- `baseline` dominant family summary: `{baseline['dominant_compound_family_summary']}`.",
        f"- `cfg05` dominant family summary: `{cfg05['dominant_compound_family_summary']}`.",
        f"- `cfg08` dominant family summary: `{cfg08['dominant_compound_family_summary']}`.",
        "- Baseline still shows large raw between-class separation partly because of narrow purine-dominant geometry.",
        "- `cfg05` keeps the same purine-core neighborhood family but redistributes support enough to broaden the fingerprint.",
        "- `cfg08` broadens further, but it shifts the chemistry vocabulary much more aggressively and risks becoming less cell-type-specific.",
        "",
        "## 5. Comparator Interpretation",
        f"- `cfg05` note: `{cfg05['concise_interpretation_note']}`.",
        f"- `cfg08` note: `{cfg08['concise_interpretation_note']}`.",
        f"- Pass 5 validation note: baseline `{float(pass5_ranked_df[pass5_ranked_df['config_id']=='cfg02']['validation_score'].iloc[0]):.4f}`, cfg05 `{float(pass5_ranked_df[pass5_ranked_df['config_id']=='cfg05']['validation_score'].iloc[0]):.4f}`, cfg08 `{float(pass5_ranked_df[pass5_ranked_df['config_id']=='cfg08']['validation_score'].iloc[0]):.4f}`.",
        "- `cfg05` gives the most convincing compromise here: broader fingerprints, lower dominance, and a config-specific BSV geometry that still looks interpretable rather than diffuse.",
        "- `cfg08` remains useful as a broadness comparator, but on this cell-type-only rerun it pushes farther toward diffuse chemistry than is helpful for the next pilot step.",
        "",
        "## 6. Recommendation",
        "- Recommended config for Pilot 1b: `cfg05`.",
        "- Keep `baseline` as the narrow reference and `cfg08` as the broadness stress-test comparator.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_pdf(report_md: Path, figure_paths: list[Path], output_path: Path) -> None:
    text = report_md.read_text(encoding="utf-8")
    lines: list[str] = []
    for raw in text.splitlines():
        if raw.startswith("#"):
            lines.append(raw)
        elif raw.strip():
            lines.extend(textwrap.wrap(raw, width=96))
        else:
            lines.append("")
    with PdfPages(output_path) as pdf:
        chunk_size = 34
        for i in range(0, len(lines), chunk_size):
            fig = plt.figure(figsize=(8.27, 11.69))
            y = 0.96
            for line in lines[i : i + chunk_size]:
                size = 12 if line.startswith("# ") else 10 if line.startswith("## ") else 8.6
                weight = "bold" if line.startswith("#") else "normal"
                fig.text(0.06, y, line, ha="left", va="top", fontsize=size, fontweight=weight, family="DejaVu Sans Mono")
                y -= 0.026 if line.startswith("#") else 0.023
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        for path in figure_paths:
            img = plt.imread(path)
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_axes([0.03, 0.06, 0.94, 0.88])
            ax.imshow(img)
            ax.axis("off")
            fig.suptitle(path.name, fontsize=12, y=0.98)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    storage_cfg = load_autoresearch_storage_config(DEFAULT_STORAGE_CONFIG_PATH)
    sprint_paths = initialize_autoresearch_sprint(
        DEFAULT_STORAGE_CONFIG_PATH,
        sprint_id=f"{storage_cfg.sprint_id}/{SPRINT_SUBDIR}",
    )
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
    pass5_ranked_df = pd.read_csv(PASS5_TABLE)

    query_df = load_query_dataframe(resolved.dataset_row)
    _, _, spectral_pca_df = compute_local_pca(query_df, n_components=3)
    _plot_spectral_pca(spectral_pca_df, sprint_paths.figures_dir / "pca_spectral_original_dataset.png")

    config_outputs = []
    for spec in CONFIG_SPECS:
        print(f"START config={spec['config_id']}", flush=True)
        config_outputs.append(_run_config(registries=registries, resolved=resolved, config_spec=spec, sprint_root=sprint_paths.sprint_root, spectral_pca_df=spectral_pca_df))
        print(f"DONE config={spec['config_id']}", flush=True)

    overlay_rows = []
    comparator_rows = []
    for outputs in config_outputs:
        spec = outputs["config_spec"]
        within = outputs["within_between_summary_df"].iloc[0]
        entropy_df = outputs["class_neighborhood_entropy_df"]
        top1_df = outputs["class_top1_dominance_df"]
        axis_df = outputs["class_axis_entropy_df"]
        comparator_rows.append(
            {
                "config_id": spec["config_id"],
                "config_short_label": spec["short_label"],
                "display_name": spec["display_name"],
                "mean_intra_class_bsv_variance": float(within["mean_intra_class_bsv_variance"]),
                "mean_inter_class_bsv_distance": float(within["mean_inter_class_bsv_distance"]),
                "within_between_ratio": float(within["within_between_ratio"]),
                "mean_neighborhood_entropy": float(entropy_df["neighborhood_entropy"].mean()),
                "mean_top1_dominance": float(top1_df["top1_fraction"].mean()),
                "mean_axis_entropy": float(axis_df["axis_entropy"].mean()),
                "dominant_compound_family_summary": _dominant_family_summary(outputs["class_neighborhood_df"]),
                "concise_interpretation_note": _interpretation_note(
                    float(entropy_df["neighborhood_entropy"].mean()),
                    float(top1_df["top1_fraction"].mean()),
                    float(within["within_between_ratio"]),
                ),
            }
        )
        overlay = outputs["bsv_class_mean_pca_df"].copy()
        overlay["config_short_label"] = str(spec["short_label"])
        overlay_rows.append(overlay)

    comparator_df = pd.DataFrame(comparator_rows)
    comparator_df.to_csv(sprint_paths.tables_dir / "pilot1a_probe1_v2_config_comparison.csv", index=False)

    overlay_df = pd.concat(overlay_rows, ignore_index=True)
    _plot_bsv_class_mean_overlay(sprint_paths.figures_dir / "pca_bsv_class_mean_overlay.png", overlay_df)
    _plot_within_between(sprint_paths.figures_dir / "pilot1a_within_between_comparison.png", comparator_df)
    _plot_entropy_dominance(sprint_paths.figures_dir / "pilot1a_entropy_dominance_comparison.png", comparator_df)
    _plot_tradeoff(sprint_paths.figures_dir / "pilot1a_config_tradeoff_summary.png", comparator_df)

    report_md = sprint_paths.report_dir / "GAIRAv3_Pilot1a_celltype_probe1_v2_report.md"
    _build_report(report_md, comparator_df, pass5_ranked_df)

    figure_paths = [sprint_paths.figures_dir / "pca_spectral_original_dataset.png"]
    for spec in CONFIG_SPECS:
        figure_paths.extend(
            [
                sprint_paths.sprint_root / "runs" / str(spec["config_id"]) / "figures" / f"class_mean_bsv_heatmap_{spec['short_label']}.png",
                sprint_paths.sprint_root / "runs" / str(spec["config_id"]) / "figures" / f"pairwise_delta_bsv_heatmap_small_molecule_metabolite_{spec['short_label']}.png",
                sprint_paths.sprint_root / "runs" / str(spec["config_id"]) / "figures" / f"pca_bsv_{spec['short_label']}.png",
                sprint_paths.sprint_root / "runs" / str(spec["config_id"]) / "figures" / f"pca_bsv_class_mean_{spec['short_label']}.png",
                sprint_paths.sprint_root / "runs" / str(spec["config_id"]) / "figures" / f"radar_fingerprint_grid_{spec['short_label']}.png",
                sprint_paths.sprint_root / "runs" / str(spec["config_id"]) / "figures" / f"neighborhood_grid_{spec['short_label']}.png",
            ]
        )
    figure_paths.extend(
        [
            sprint_paths.figures_dir / "pca_bsv_class_mean_overlay.png",
            sprint_paths.figures_dir / "pilot1a_within_between_comparison.png",
            sprint_paths.figures_dir / "pilot1a_entropy_dominance_comparison.png",
            sprint_paths.figures_dir / "pilot1a_config_tradeoff_summary.png",
        ]
    )
    _build_pdf(report_md, figure_paths, sprint_paths.report_dir / "GAIRAv3_Pilot1a_celltype_probe1_v2_report.pdf")
    print(f"Wrote Pilot 1a v2 outputs under {sprint_paths.sprint_root}")


if __name__ == "__main__":
    main()
