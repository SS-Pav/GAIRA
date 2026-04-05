from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    build_pdf_report,
    compute_stability_tables,
    pairwise_delta_bsv,
    plot_bsv_heatmap,
    plot_pairwise_delta_heatmap,
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
SPRINT_SUBDIR = "pilot1a_celltype_probe1_clean"
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
            "experiment_id": f"pilot1a_clean__{subset_alias}",
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


def _plot_pca_by_class(pca_df: pd.DataFrame, output_path: Path, title: str) -> None:
    work = pca_df.copy()
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    labels = sorted(work["class_label"].astype(str).unique().tolist())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 1)))
    for color, label in zip(colors, labels, strict=False):
        sub = work[work["class_label"].astype(str) == label].copy()
        ax.scatter(
            sub["pc1"].to_numpy(dtype=float),
            sub["pc2"].to_numpy(dtype=float),
            s=28,
            alpha=0.85,
            label=label,
            color=color,
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


def _stability_summary(
    config_name: str,
    intra_df: pd.DataFrame,
    inter_df: pd.DataFrame,
) -> pd.DataFrame:
    axes = [axis for axis in ALL_AXES if axis in intra_df.columns]
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
                "within_between_ratio": float(
                    within["mean_variance"].mean() / max(unique_pairs["euclidean_distance"].mean(), 1e-12)
                ),
                "top_3_farthest_class_pairs": "; ".join(
                    [
                        f"{r.class_label_a} vs {r.class_label_b} ({float(r.euclidean_distance):.4f})"
                        for r in farthest.itertuples(index=False)
                    ]
                ),
                "top_3_closest_nonidentical_class_pairs": "; ".join(
                    [
                        f"{r.class_label_a} vs {r.class_label_b} ({float(r.euclidean_distance):.4f})"
                        for r in closest.itertuples(index=False)
                    ]
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
    if mean_neighborhood_entropy > 1.6 and within_between_ratio > 0.01:
        return "broadened but tending diffuse"
    if mean_neighborhood_entropy > 1.1 and mean_top1_dominance < 0.65:
        return "broadened with usable structure"
    return "intermediate tradeoff"


def _plot_radar_grid(class_mean_bsv: pd.DataFrame, output_path: Path, title: str) -> None:
    axes = [axis for axis in ALL_AXES if axis in class_mean_bsv.columns]
    labels = sorted(class_mean_bsv["class_label"].astype(str).tolist())
    ncols = 3
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.6 * ncols, 4.4 * nrows),
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
        ax.plot(angles_closed, values_closed, color=color, linewidth=1.8)
        ax.fill(angles_closed, values_closed, color=color, alpha=0.22)
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


def _plot_within_between(output_path: Path, comparator_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
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
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
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
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 4.6))
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


def _run_config(*, registries, resolved: ResolvedExperiment, config_spec: dict[str, object], sprint_root: Path) -> dict[str, object]:
    harness_config = _config_to_harness(config_spec)
    query_df = load_query_dataframe(resolved.dataset_row)
    _, _, pca_df = compute_local_pca(query_df, n_components=3)

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
    within_between_summary_df = _stability_summary(
        str(config_spec["config_id"]),
        intra_class_variance_df,
        inter_class_distance_df,
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
    pca_df.to_csv(tables_dir / "pca_coordinates.csv", index=False)
    within_between_summary_df.to_csv(tables_dir / "config_within_between_summary.csv", index=False)

    plot_bsv_heatmap(
        class_mean_bsv_df,
        figures_dir / f"class_mean_bsv_heatmap_{config_spec['short_label']}.png",
        f"Class Mean BSV Heatmap ({config_spec['short_label']})",
    )
    plot_pairwise_delta_heatmap(
        pairwise_delta_df,
        "small_molecule_metabolite",
        figures_dir / f"pairwise_delta_bsv_heatmap_small_molecule_metabolite_{config_spec['short_label']}.png",
    )
    _plot_pca_by_class(
        pca_df,
        figures_dir / f"pca_by_class_{config_spec['short_label']}.png",
        f"PCA by Class ({config_spec['short_label']})",
    )
    _plot_radar_grid(
        class_mean_bsv_df,
        figures_dir / f"radar_fingerprint_grid_{config_spec['short_label']}.png",
        f"Radar Fingerprint Grid ({config_spec['short_label']})",
    )
    _plot_neighborhood_grid(
        class_neighborhood_df,
        figures_dir / f"neighborhood_grid_{config_spec['short_label']}.png",
        f"Neighborhood Grid ({config_spec['short_label']})",
    )

    config_summary = {
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
    }
    (report_dir / "run_config.json").write_text(json.dumps(config_summary, indent=2) + "\n", encoding="utf-8")

    return {
        "config_spec": config_spec,
        "run_dir": run_dir,
        "class_mean_bsv_df": class_mean_bsv_df,
        "intra_class_variance_df": intra_class_variance_df,
        "inter_class_distance_df": inter_class_distance_df,
        "class_neighborhood_df": class_neighborhood_df,
        "class_neighborhood_entropy_df": class_neighborhood_entropy_df,
        "class_top1_dominance_df": class_top1_dominance_df,
        "class_axis_entropy_df": class_axis_entropy_df,
        "within_between_summary_df": within_between_summary_df,
    }


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

    config_outputs = []
    for config_spec in CONFIG_SPECS:
        print(f"START config={config_spec['config_id']}", flush=True)
        config_outputs.append(_run_config(registries=registries, resolved=resolved, config_spec=config_spec, sprint_root=sprint_paths.sprint_root))
        print(f"DONE config={config_spec['config_id']}", flush=True)

    comparator_rows = []
    for outputs in config_outputs:
        spec = outputs["config_spec"]
        within_summary = outputs["within_between_summary_df"].iloc[0]
        entropy_df = outputs["class_neighborhood_entropy_df"]
        top1_df = outputs["class_top1_dominance_df"]
        axis_df = outputs["class_axis_entropy_df"]
        comparator_rows.append(
            {
                "config_id": spec["config_id"],
                "config_short_label": spec["short_label"],
                "display_name": spec["display_name"],
                "mean_intra_class_bsv_variance": float(within_summary["mean_intra_class_bsv_variance"]),
                "mean_inter_class_bsv_distance": float(within_summary["mean_inter_class_bsv_distance"]),
                "within_between_ratio": float(within_summary["within_between_ratio"]),
                "mean_neighborhood_entropy": float(entropy_df["neighborhood_entropy"].mean()),
                "mean_top1_dominance": float(top1_df["top1_fraction"].mean()),
                "mean_axis_entropy": float(axis_df["axis_entropy"].mean()),
                "dominant_compound_family_summary": _dominant_family_summary(outputs["class_neighborhood_df"]),
                "concise_interpretation_note": _interpretation_note(
                    float(entropy_df["neighborhood_entropy"].mean()),
                    float(top1_df["top1_fraction"].mean()),
                    float(within_summary["within_between_ratio"]),
                ),
                "top_3_farthest_class_pairs": str(within_summary["top_3_farthest_class_pairs"]),
                "top_3_closest_nonidentical_class_pairs": str(within_summary["top_3_closest_nonidentical_class_pairs"]),
            }
        )
    comparator_df = pd.DataFrame(comparator_rows)
    comparator_df.to_csv(sprint_paths.tables_dir / "pilot1a_probe1_config_comparison.csv", index=False)

    _plot_within_between(
        sprint_paths.figures_dir / "pilot1a_within_between_comparison.png",
        comparator_df,
    )
    _plot_entropy_dominance(
        sprint_paths.figures_dir / "pilot1a_entropy_dominance_comparison.png",
        comparator_df,
    )
    _plot_tradeoff(
        sprint_paths.figures_dir / "pilot1a_config_tradeoff_summary.png",
        comparator_df,
    )

    baseline = comparator_df[comparator_df["config_short_label"] == "baseline"].iloc[0]
    cfg05 = comparator_df[comparator_df["config_short_label"] == "cfg05"].iloc[0]
    cfg08 = comparator_df[comparator_df["config_short_label"] == "cfg08"].iloc[0]
    validation_lookup = {
        "baseline": "cfg02",
        "cfg05": "cfg05",
        "cfg08": "cfg08",
    }
    lines = [
        "# GAIRAv3 Pilot 1a Cell-type Probe1 Clean Report",
        "",
        "## 1. Overview",
        "- This is the cleaned Pilot 1a cell-type fingerprint study on Probe 1.",
        "- The three compared configurations are fixed deterministic comparators, not an autoresearch sweep.",
        "- Fingerprint here means the combined object of class-level BSV profile, class-level neighborhood composition, and within-class vs between-class stability.",
        "",
        "## 2. Cell-type Fingerprints by Config",
        f"- `baseline`: narrow profile with mean neighborhood entropy `{baseline['mean_neighborhood_entropy']:.4f}` and mean top1 dominance `{baseline['mean_top1_dominance']:.4f}`.",
        f"- `cfg05`: broadened profile with mean neighborhood entropy `{cfg05['mean_neighborhood_entropy']:.4f}` and mean top1 dominance `{cfg05['mean_top1_dominance']:.4f}`.",
        f"- `cfg08`: broadest profile with mean neighborhood entropy `{cfg08['mean_neighborhood_entropy']:.4f}` and mean top1 dominance `{cfg08['mean_top1_dominance']:.4f}`.",
        "- Broader does not automatically mean better. The key question is whether broadening preserves class structure rather than smearing it.",
        "",
        "## 3. Stability Analysis",
        f"- `baseline`: mean intra `{baseline['mean_intra_class_bsv_variance']:.6f}`, mean inter `{baseline['mean_inter_class_bsv_distance']:.6f}`, ratio `{baseline['within_between_ratio']:.6f}`.",
        f"- `cfg05`: mean intra `{cfg05['mean_intra_class_bsv_variance']:.6f}`, mean inter `{cfg05['mean_inter_class_bsv_distance']:.6f}`, ratio `{cfg05['within_between_ratio']:.6f}`.",
        f"- `cfg08`: mean intra `{cfg08['mean_intra_class_bsv_variance']:.6f}`, mean inter `{cfg08['mean_inter_class_bsv_distance']:.6f}`, ratio `{cfg08['within_between_ratio']:.6f}`.",
        "- Desired behavior here is small within-class variance and larger between-class distance, not just entropy inflation.",
        "",
        "## 4. Neighborhood Behavior",
        f"- `baseline` dominant family summary: `{baseline['dominant_compound_family_summary']}`.",
        f"- `cfg05` dominant family summary: `{cfg05['dominant_compound_family_summary']}`.",
        f"- `cfg08` dominant family summary: `{cfg08['dominant_compound_family_summary']}`.",
        "- The classes can differ either by strong single-neighborhood dominance or by shifted ratios within a broader chemistry region.",
        "",
        "## 5. Comparator Interpretation",
        f"- `cfg05` interpretation: {cfg05['concise_interpretation_note']}.",
        f"- `cfg08` interpretation: {cfg08['concise_interpretation_note']}.",
        f"- Pass 5 validation note: baseline `{float(pass5_ranked_df[pass5_ranked_df['config_id']=='cfg02']['validation_score'].iloc[0]):.4f}`, cfg05 `{float(pass5_ranked_df[pass5_ranked_df['config_id']=='cfg05']['validation_score'].iloc[0]):.4f}`, cfg08 `{float(pass5_ranked_df[pass5_ranked_df['config_id']=='cfg08']['validation_score'].iloc[0]):.4f}`.",
        "- `cfg05` broadens without obvious collapse and preserves the cleanest within-vs-between structure among the broadened options.",
        "- `cfg08` broadens further, but it also shifts the chemistry vocabulary more aggressively and gives up more class-specific shape in the cell-type panel.",
        "",
        "## 6. Decision for Moving to Pilot 1b",
    ]
    if float(cfg05["within_between_ratio"]) < float(cfg08["within_between_ratio"]):
        lines.append("- Recommended working default for Pilot 1b: `cfg05`.")
        lines.append("- Keep `cfg08` as a breadth comparator, but not as the default for the cleaned fingerprint-first sequence.")
    else:
        lines.append("- Recommended working default for Pilot 1b: `cfg08`.")
        lines.append("- Keep `cfg05` as a comparator reference.")
    report_md = sprint_paths.report_dir / "GAIRAv3_Pilot1a_celltype_probe1_clean_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    figure_paths = []
    for spec in CONFIG_SPECS:
        figure_paths.extend(
            [
                sprint_paths.sprint_root / "runs" / str(spec["config_id"]) / "figures" / f"pca_by_class_{spec['short_label']}.png",
                sprint_paths.sprint_root / "runs" / str(spec["config_id"]) / "figures" / f"class_mean_bsv_heatmap_{spec['short_label']}.png",
                sprint_paths.sprint_root / "runs" / str(spec["config_id"]) / "figures" / f"pairwise_delta_bsv_heatmap_small_molecule_metabolite_{spec['short_label']}.png",
                sprint_paths.sprint_root / "runs" / str(spec["config_id"]) / "figures" / f"radar_fingerprint_grid_{spec['short_label']}.png",
                sprint_paths.sprint_root / "runs" / str(spec["config_id"]) / "figures" / f"neighborhood_grid_{spec['short_label']}.png",
            ]
        )
    figure_paths.extend(
        [
            sprint_paths.figures_dir / "pilot1a_within_between_comparison.png",
            sprint_paths.figures_dir / "pilot1a_entropy_dominance_comparison.png",
            sprint_paths.figures_dir / "pilot1a_config_tradeoff_summary.png",
        ]
    )
    build_pdf_report(
        report_md,
        figure_paths,
        sprint_paths.report_dir / "GAIRAv3_Pilot1a_celltype_probe1_clean_report.pdf",
    )
    print(f"Wrote Pilot 1a clean outputs under {sprint_paths.sprint_root}")


if __name__ == "__main__":
    main()
