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
    compute_mixture_metrics,
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
    build_mixture_progression_summary,
    build_pdf_report,
    compute_stability_tables,
    infer_mixture_order,
    pairwise_delta_bsv,
)
from gaira.demo.raw_bsv_pilot_utils import (
    ALL_AXES,
    apply_source_role_policy,
    build_group_mean_query_df,
    compute_local_pca,
    group_mean_bsv,
    load_ontology_rules,
    map_references_to_axes,
)


ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = ROOT / "reports" / "gaira_architecture_scaffold_v2"
PHASE1_DIR = ROOT / "reports" / "gaira_phase1_registry_audit_v2"
PASS5_TABLE = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pass5_saturation_fix/tables/calibration_results_ranked.csv"
)
SPRINT_SUBDIR = "pilot1_comparator_cfg05_cfg08"
ONTOLOGY_PATH = ROOT / "config" / "phase2_bsv_ontology_rules_v2.csv"

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
        "family_min_coverage": 0,
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
        "family_min_coverage": 0,
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
        "family_min_coverage": 0,
    },
]

PANEL_SPECS = [
    {"panel_name": "small2023_cellline", "subset_alias": "small2023_cellline", "kind": "cellline"},
    {"panel_name": "small2023_mixture_probe1", "subset_alias": "small2023_mixture_probe1", "kind": "mixture"},
    {"panel_name": "small2023_mixture_probe2", "subset_alias": "small2023_mixture_probe2", "kind": "mixture_holdout"},
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
            "experiment_id": f"pilot1_comparator__{subset_alias}",
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
        family_min_coverage=int(spec["family_min_coverage"]),
    )


def _run_panel(
    *,
    registries,
    resolved: ResolvedExperiment,
    config_spec: dict[str, object],
    sprint_root: Path,
) -> dict[str, object]:
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

    if "c00" in class_mean_bsv_df["class_label"].astype(str).tolist():
        ordered = infer_mixture_order(class_mean_bsv_df["class_label"].astype(str).tolist())
        class_mean_bsv_df["class_label"] = class_mean_bsv_df["class_label"].astype(str)
        class_mean_bsv_df["class_order"] = class_mean_bsv_df["class_label"].map(
            {label: i for i, label in enumerate(ordered)}
        )
        class_mean_bsv_df = (
            class_mean_bsv_df.sort_values("class_order").drop(columns=["class_order"]).reset_index(drop=True)
        )

    pairwise_delta_df = pairwise_delta_bsv(class_mean_bsv_df)
    retrieval_summary_by_class_df = retrieval_hit_summary(class_mean_retrieval_df)
    intra_class_variance_df, inter_class_distance_df = compute_stability_tables(
        per_spectrum_bsv_df, class_mean_bsv_df
    )
    class_neighborhood_df = build_class_topk_neighborhood_composition(per_spectrum_retrieval_df)
    class_neighborhood_entropy_df = build_class_neighborhood_entropy(class_neighborhood_df)
    class_top1_dominance_df = build_class_top1_dominance(class_neighborhood_df)
    class_axis_entropy_df = build_class_axis_entropy(class_mean_bsv_df)

    progression_df = None
    mixture_metrics = None
    if resolved.subset_alias.startswith("small2023_mixture_"):
        progression_df = build_mixture_progression_summary(class_mean_bsv_df, class_neighborhood_df)
        mixture_metrics = compute_mixture_metrics(class_mean_bsv_df, per_spectrum_retrieval_df)

    run_dir = sprint_root / "runs" / str(config_spec["config_id"]) / resolved.subset_alias
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

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
    raw_mapping_df.to_csv(tables_dir / "ontology_mapping_applied.csv", index=False)
    if progression_df is not None:
        progression_df.to_csv(tables_dir / "mixture_progression_summary.csv", index=False)

    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "subset_alias": resolved.subset_alias,
                "dataset_id": str(resolved.dataset_row["dataset_id"]),
                "subset_id": str(resolved.dataset_row["subset_id"]),
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
        "query_df": query_df,
        "pca_df": pca_df,
        "per_spectrum_bsv_df": per_spectrum_bsv_df,
        "class_mean_bsv_df": class_mean_bsv_df,
        "pairwise_delta_df": pairwise_delta_df,
        "intra_class_variance_df": intra_class_variance_df,
        "inter_class_distance_df": inter_class_distance_df,
        "class_neighborhood_df": class_neighborhood_df,
        "class_neighborhood_entropy_df": class_neighborhood_entropy_df,
        "class_top1_dominance_df": class_top1_dominance_df,
        "class_axis_entropy_df": class_axis_entropy_df,
        "retrieval_summary_by_class_df": retrieval_summary_by_class_df,
        "per_spectrum_retrieval_df": per_spectrum_retrieval_df,
        "progression_df": progression_df,
        "mixture_metrics": mixture_metrics,
        "run_dir": run_dir,
    }


def _plot_heatmap_matrix(ax, df: pd.DataFrame, title: str, class_order: list[str] | None = None) -> None:
    work = df.copy()
    work["class_label"] = work["class_label"].astype(str)
    if class_order is not None:
        work["__order"] = work["class_label"].map({label: i for i, label in enumerate(class_order)})
        work = work.sort_values("__order").drop(columns="__order")
    axes = [axis for axis in ALL_AXES if axis in work.columns]
    matrix = work.set_index("class_label")[axes].to_numpy(dtype=float)
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(float(matrix.max()), 1e-6))
    ax.set_xticks(np.arange(len(axes)))
    ax.set_xticklabels(axes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(work)))
    ax.set_yticklabels(work["class_label"].tolist(), fontsize=8)
    ax.set_title(title, fontsize=10)
    return im


def _plot_grouped_metric(
    output_path: Path,
    title: str,
    rows: pd.DataFrame,
    metric_col: str,
    class_col: str = "class_label",
) -> None:
    classes = sorted(rows[class_col].astype(str).unique().tolist(), key=lambda x: (int("".join(ch for ch in x if ch.isdigit()) or 10**9), x))
    configs = [spec["short_label"] for spec in CONFIG_SPECS]
    x = np.arange(len(classes))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for i, short_label in enumerate(configs):
        sub = rows[rows["config_short_label"] == short_label].copy()
        series = (
            sub.set_index(class_col)[metric_col].reindex(classes).fillna(0.0).to_numpy(dtype=float)
        )
        ax.bar(x + (i - 1) * width, series, width=width, label=short_label)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.8, 1.0])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_progression_comparison(output_path: Path, rows: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    for spec in CONFIG_SPECS:
        sub = rows[rows["config_short_label"] == spec["short_label"]].copy()
        sub = sub.sort_values("mixture_code_numeric")
        ax.plot(
            sub["class_label"].astype(str).tolist(),
            sub["toward_high_endpoint_score"].to_numpy(dtype=float),
            marker="o",
            linewidth=2.0,
            label=spec["short_label"],
        )
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.set_ylabel("Toward high-endpoint score")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.8, 1.0])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_probe1_bsv_comparison(output_path: Path, rows: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for axis_name, ax in zip(["small_molecule_metabolite", "nucleic_acid"], axes, strict=False):
        for spec in CONFIG_SPECS:
            sub = rows[rows["config_short_label"] == spec["short_label"]].copy()
            sub = sub.sort_values("mixture_code_numeric")
            ax.plot(
                sub["class_label"].astype(str).tolist(),
                sub[axis_name].to_numpy(dtype=float),
                marker="o",
                linewidth=2.0,
                label=spec["short_label"],
            )
        ax.set_title(axis_name.replace("_", " ").title())
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle("Pilot 1b Probe1 Class-Mean BSV Comparison")
    fig.tight_layout(rect=[0.0, 0.0, 0.88, 0.95])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_tradeoff_summary(output_path: Path, overview_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    panels = [
        ("pass5_validation_score", "Validation Note From Pass 5"),
        ("pilot1a_mean_neighborhood_entropy", "Pilot 1a Mean Neighborhood Entropy"),
        ("pilot1b_probe1_mixture_progression_score", "Pilot 1b Probe1 Progression"),
        ("pilot1b_probe2_mixture_progression_score", "Pilot 1b Probe2 Holdout Progression"),
    ]
    for ax, (col, title) in zip(axes.ravel(), panels, strict=False):
        sub = overview_df.copy()
        x = np.arange(len(sub))
        ax.bar(x, sub[col].to_numpy(dtype=float), color=["#577590", "#f3722c", "#43aa8b"])
        ax.set_xticks(x)
        ax.set_xticklabels(sub["config_short_label"].tolist(), rotation=0)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    fig.suptitle("Config Tradeoff Summary")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _summarize_cellline_rows(config_short_label: str, outputs: dict[str, object]) -> pd.DataFrame:
    class_mean_bsv_df = outputs["class_mean_bsv_df"].copy()
    class_mean_bsv_df["class_label"] = class_mean_bsv_df["class_label"].astype(str)
    entropy_df = outputs["class_neighborhood_entropy_df"].rename(columns={"neighborhood_entropy": "neighborhood_entropy"})
    top1_df = outputs["class_top1_dominance_df"].rename(columns={"top1_fraction": "top1_fraction"})
    axis_df = outputs["class_axis_entropy_df"].rename(columns={"axis_entropy": "axis_entropy"})
    retrieval_df = outputs["class_neighborhood_df"].copy()
    top_compounds = (
        retrieval_df.sort_values(["class_label", "support_fraction"], ascending=[True, False])
        .groupby("class_label", sort=True)
        .head(1)[["class_label", "compound_label", "support_fraction"]]
        .rename(columns={"compound_label": "top_compound_label", "support_fraction": "top_compound_fraction"})
    )
    merged = class_mean_bsv_df.merge(entropy_df, on="class_label").merge(top1_df, on="class_label").merge(axis_df, on="class_label").merge(top_compounds, on="class_label")
    merged.insert(0, "config_short_label", config_short_label)
    return merged


def _summarize_mixture_rows(config_short_label: str, outputs: dict[str, object]) -> pd.DataFrame:
    class_mean_bsv_df = outputs["class_mean_bsv_df"].copy()
    progression_df = outputs["progression_df"].copy()
    entropy_df = outputs["class_neighborhood_entropy_df"].copy()
    top1_df = outputs["class_top1_dominance_df"].copy()
    axis_df = outputs["class_axis_entropy_df"].copy()
    retrieval_df = outputs["class_neighborhood_df"].copy()
    top_compounds = (
        retrieval_df.sort_values(["class_label", "support_fraction"], ascending=[True, False])
        .groupby("class_label", sort=True)
        .head(1)[["class_label", "compound_label", "support_fraction"]]
        .rename(columns={"compound_label": "top_compound_label", "support_fraction": "top_compound_fraction"})
    )
    merged = (
        progression_df.merge(entropy_df, on="class_label")
        .merge(top1_df, on="class_label")
        .merge(axis_df, on="class_label")
        .merge(top_compounds, on="class_label")
        .merge(class_mean_bsv_df[["class_label"] + [a for a in ["small_molecule_metabolite", "nucleic_acid", "substrate_adsorption_bias"] if a in class_mean_bsv_df.columns]], on="class_label")
    )
    merged.insert(0, "config_short_label", config_short_label)
    return merged


def _build_overview_row(
    config_spec: dict[str, object],
    cellline_outputs: dict[str, object],
    probe1_outputs: dict[str, object],
    probe2_outputs: dict[str, object],
    pass5_ranked_df: pd.DataFrame,
) -> dict[str, object]:
    cell_entropy = float(cellline_outputs["class_neighborhood_entropy_df"]["neighborhood_entropy"].mean())
    cell_top1 = float(cellline_outputs["class_top1_dominance_df"]["top1_fraction"].mean())
    cell_axis = float(cellline_outputs["class_axis_entropy_df"]["axis_entropy"].mean())
    cell_var = float(cellline_outputs["intra_class_variance_df"].drop(columns=["class_label"]).mean(axis=1).mean())
    probe1_metrics = probe1_outputs["mixture_metrics"]
    probe2_metrics = probe2_outputs["mixture_metrics"]

    pass5_lookup = {
        "baseline_v1_locked_purine": "cfg02",
        "candidate_v2_cfg05_max_desaturation": "cfg05",
        "candidate_v2_cfg08_balanced_update": "cfg08",
    }
    pass5_row = pass5_ranked_df[pass5_ranked_df["config_id"] == pass5_lookup[str(config_spec["config_id"])]].iloc[0]
    return {
        "config_id": config_spec["config_id"],
        "config_short_label": config_spec["short_label"],
        "display_name": config_spec["display_name"],
        "pass5_validation_score": float(pass5_row["validation_score"]),
        "pilot1a_mean_neighborhood_entropy": cell_entropy,
        "pilot1a_mean_top1_dominance": cell_top1,
        "pilot1a_mean_axis_entropy": cell_axis,
        "pilot1a_mean_intra_class_variance": cell_var,
        "pilot1b_probe1_mixture_progression_score": float(probe1_metrics["mixture_progression_score"]),
        "pilot1b_probe1_diversity_score": float(probe1_metrics["diversity_score"]),
        "pilot1b_probe1_saturation_penalty": float(probe1_metrics["saturation_penalty"]),
        "pilot1b_probe1_mean_top1_dominance": float(probe1_metrics["mean_top1_dominance"]),
        "pilot1b_probe1_noncollapse_ratio": float(probe1_metrics["noncollapse_ratio"]),
        "pilot1b_probe2_mixture_progression_score": float(probe2_metrics["mixture_progression_score"]),
        "pilot1b_probe2_diversity_score": float(probe2_metrics["diversity_score"]),
        "pilot1b_probe2_saturation_penalty": float(probe2_metrics["saturation_penalty"]),
        "pilot1b_probe2_mean_top1_dominance": float(probe2_metrics["mean_top1_dominance"]),
        "pilot1b_probe2_noncollapse_ratio": float(probe2_metrics["noncollapse_ratio"]),
    }


def _build_report(
    sprint_root: Path,
    pilot1a_df: pd.DataFrame,
    probe1_df: pd.DataFrame,
    probe2_df: pd.DataFrame,
    overview_df: pd.DataFrame,
) -> Path:
    baseline = overview_df[overview_df["config_short_label"] == "baseline"].iloc[0]
    cfg05 = overview_df[overview_df["config_short_label"] == "cfg05"].iloc[0]
    cfg08 = overview_df[overview_df["config_short_label"] == "cfg08"].iloc[0]
    lines = [
        "# GAIRAv3 Pilot 1 Comparator: baseline vs cfg05 vs cfg08",
        "",
        "## Compared Configurations",
    ]
    for spec in CONFIG_SPECS:
        lines.append(
            f"- `{spec['short_label']}`: filter=`{spec['filter_mode']}`, top_k=`{spec['top_k']}`, weighting=`{spec['weighting_mode']}`, diversity=`{spec['diversity_mode']}`"
        )
    lines.extend(
        [
            "",
            "## 1. Pilot 1a: small2023_cellline",
            f"- baseline mean neighborhood entropy `{baseline['pilot1a_mean_neighborhood_entropy']:.4f}`, top1 dominance `{baseline['pilot1a_mean_top1_dominance']:.4f}`",
            f"- cfg05 mean neighborhood entropy `{cfg05['pilot1a_mean_neighborhood_entropy']:.4f}`, top1 dominance `{cfg05['pilot1a_mean_top1_dominance']:.4f}`",
            f"- cfg08 mean neighborhood entropy `{cfg08['pilot1a_mean_neighborhood_entropy']:.4f}`, top1 dominance `{cfg08['pilot1a_mean_top1_dominance']:.4f}`",
            "- Cellline interpretation should be treated as fingerprint broadening vs diffusion, not exact molecular assignment.",
            "",
            "## 2. Pilot 1b: small2023_mixture_probe1",
            f"- baseline progression `{baseline['pilot1b_probe1_mixture_progression_score']:.4f}`, noncollapse `{baseline['pilot1b_probe1_noncollapse_ratio']:.4f}`, top1 dominance `{baseline['pilot1b_probe1_mean_top1_dominance']:.4f}`",
            f"- cfg05 progression `{cfg05['pilot1b_probe1_mixture_progression_score']:.4f}`, noncollapse `{cfg05['pilot1b_probe1_noncollapse_ratio']:.4f}`, top1 dominance `{cfg05['pilot1b_probe1_mean_top1_dominance']:.4f}`",
            f"- cfg08 progression `{cfg08['pilot1b_probe1_mixture_progression_score']:.4f}`, noncollapse `{cfg08['pilot1b_probe1_noncollapse_ratio']:.4f}`, top1 dominance `{cfg08['pilot1b_probe1_mean_top1_dominance']:.4f}`",
            "",
            "## 3. Probe2 Holdout",
            f"- baseline progression `{baseline['pilot1b_probe2_mixture_progression_score']:.4f}`, noncollapse `{baseline['pilot1b_probe2_noncollapse_ratio']:.4f}`, top1 dominance `{baseline['pilot1b_probe2_mean_top1_dominance']:.4f}`",
            f"- cfg05 progression `{cfg05['pilot1b_probe2_mixture_progression_score']:.4f}`, noncollapse `{cfg05['pilot1b_probe2_noncollapse_ratio']:.4f}`, top1 dominance `{cfg05['pilot1b_probe2_mean_top1_dominance']:.4f}`",
            f"- cfg08 progression `{cfg08['pilot1b_probe2_mixture_progression_score']:.4f}`, noncollapse `{cfg08['pilot1b_probe2_noncollapse_ratio']:.4f}`, top1 dominance `{cfg08['pilot1b_probe2_mean_top1_dominance']:.4f}`",
            "",
            "## 4. Decision",
            f"- Pass 5 validation note: baseline validation `{baseline['pass5_validation_score']:.4f}`, cfg05 `{cfg05['pass5_validation_score']:.4f}`, cfg08 `{cfg08['pass5_validation_score']:.4f}`.",
            "- cfg05 is treated here as the anti-saturation upper-bound reference.",
            "- cfg08 is treated here as the balanced candidate that must preserve most of the desaturation gain without paying the full validation cost.",
        ]
    )
    if (
        float(cfg08["pilot1b_probe1_mixture_progression_score"]) > float(baseline["pilot1b_probe1_mixture_progression_score"])
        and float(cfg08["pilot1b_probe2_mixture_progression_score"]) > float(baseline["pilot1b_probe2_mixture_progression_score"])
        and float(cfg08["pass5_validation_score"]) >= float(baseline["pass5_validation_score"])
    ):
        lines.append("- Recommendation: lock `cfg08` for Pilot 2. It improves mixture behavior on Probe1 and Probe2 while staying compatible with the validation-side regime.")
    else:
        lines.append("- Recommendation: do not lock cfg08 yet; the comparator does not show enough retained benefit relative to cfg05 and baseline.")
    lines.append("")
    lines.append("## Appendix: Comparator Tables")
    lines.append(f"- Pilot 1a rows: `{len(pilot1a_df)}`")
    lines.append(f"- Pilot 1b Probe1 rows: `{len(probe1_df)}`")
    lines.append(f"- Pilot 1b Probe2 rows: `{len(probe2_df)}`")
    out = sprint_root / "report" / "GAIRAv3_Pilot1_comparator_cfg05_cfg08_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


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
    pass5_ranked_df = pd.read_csv(PASS5_TABLE)

    panel_outputs: dict[tuple[str, str], dict[str, object]] = {}
    for config_spec in CONFIG_SPECS:
        for panel_spec in PANEL_SPECS:
            print(
                f"START config={config_spec['config_id']} panel={panel_spec['panel_name']}",
                flush=True,
            )
            resolved = _resolve_alias(registries, panel_spec["subset_alias"])
            panel_outputs[(str(config_spec["config_id"]), str(panel_spec["panel_name"]))] = _run_panel(
                registries=registries,
                resolved=resolved,
                config_spec=config_spec,
                sprint_root=sprint_paths.sprint_root,
            )
            print(
                f"DONE config={config_spec['config_id']} panel={panel_spec['panel_name']}",
                flush=True,
            )

    pilot1a_rows = []
    probe1_rows = []
    probe2_rows = []
    overview_rows = []
    for config_spec in CONFIG_SPECS:
        config_id = str(config_spec["config_id"])
        short_label = str(config_spec["short_label"])
        cellline_outputs = panel_outputs[(config_id, "small2023_cellline")]
        probe1_outputs = panel_outputs[(config_id, "small2023_mixture_probe1")]
        probe2_outputs = panel_outputs[(config_id, "small2023_mixture_probe2")]
        pilot1a_rows.append(_summarize_cellline_rows(short_label, cellline_outputs))
        probe1_rows.append(_summarize_mixture_rows(short_label, probe1_outputs))
        probe2_rows.append(_summarize_mixture_rows(short_label, probe2_outputs))
        overview_rows.append(
            _build_overview_row(
                config_spec,
                cellline_outputs,
                probe1_outputs,
                probe2_outputs,
                pass5_ranked_df,
            )
        )

    pilot1a_df = pd.concat(pilot1a_rows, ignore_index=True)
    probe1_df = pd.concat(probe1_rows, ignore_index=True)
    probe2_df = pd.concat(probe2_rows, ignore_index=True)
    overview_df = pd.DataFrame(overview_rows)

    # Comparator tables.
    sprint_paths.tables_dir.mkdir(parents=True, exist_ok=True)
    pilot1a_df.to_csv(sprint_paths.tables_dir / "pilot1a_config_comparison.csv", index=False)
    probe1_df.to_csv(sprint_paths.tables_dir / "pilot1b_probe1_config_comparison.csv", index=False)
    probe2_df.to_csv(sprint_paths.tables_dir / "pilot1b_probe2_holdout_comparison.csv", index=False)
    overview_df.to_csv(sprint_paths.tables_dir / "config_summary_overview.csv", index=False)

    # Figures.
    sprint_paths.figures_dir.mkdir(parents=True, exist_ok=True)
    cell_order = sorted(
        pilot1a_df["class_label"].astype(str).unique().tolist(),
        key=lambda x: (int("".join(ch for ch in x if ch.isdigit()) or 10**9), x),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    ims = []
    for ax, spec in zip(axes, CONFIG_SPECS, strict=False):
        df = panel_outputs[(str(spec["config_id"]), "small2023_cellline")]["class_mean_bsv_df"]
        ims.append(_plot_heatmap_matrix(ax, df, spec["short_label"], cell_order))
    cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.8)
    cbar.ax.set_ylabel("BSV support")
    fig.suptitle("Pilot 1a BSV Heatmap Comparison")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(sprint_paths.figures_dir / "pilot1a_bsv_heatmap_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    _plot_grouped_metric(
        sprint_paths.figures_dir / "pilot1a_neighborhood_entropy_comparison.png",
        "Pilot 1a Neighborhood Entropy Comparison",
        pilot1a_df,
        "neighborhood_entropy",
    )
    _plot_grouped_metric(
        sprint_paths.figures_dir / "pilot1a_top1_dominance_comparison.png",
        "Pilot 1a Top1 Dominance Comparison",
        pilot1a_df,
        "top1_fraction",
    )

    _plot_progression_comparison(
        sprint_paths.figures_dir / "pilot1b_probe1_progression_comparison.png",
        probe1_df,
        "Pilot 1b Probe1 Progression Comparison",
    )
    _plot_grouped_metric(
        sprint_paths.figures_dir / "pilot1b_probe1_top1_dominance_comparison.png",
        "Pilot 1b Probe1 Top1 Dominance Comparison",
        probe1_df,
        "top1_fraction",
    )
    _plot_grouped_metric(
        sprint_paths.figures_dir / "pilot1b_probe1_entropy_comparison.png",
        "Pilot 1b Probe1 Neighborhood Entropy Comparison",
        probe1_df,
        "neighborhood_entropy",
    )
    _plot_probe1_bsv_comparison(
        sprint_paths.figures_dir / "pilot1b_probe1_class_mean_bsv_comparison.png",
        probe1_df,
    )

    _plot_progression_comparison(
        sprint_paths.figures_dir / "pilot1b_probe2_progression_comparison.png",
        probe2_df,
        "Pilot 1b Probe2 Holdout Progression Comparison",
    )
    _plot_grouped_metric(
        sprint_paths.figures_dir / "pilot1b_probe2_top1_dominance_comparison.png",
        "Pilot 1b Probe2 Top1 Dominance Comparison",
        probe2_df,
        "top1_fraction",
    )
    _plot_grouped_metric(
        sprint_paths.figures_dir / "pilot1b_probe2_entropy_comparison.png",
        "Pilot 1b Probe2 Neighborhood Entropy Comparison",
        probe2_df,
        "neighborhood_entropy",
    )
    _plot_tradeoff_summary(
        sprint_paths.figures_dir / "config_tradeoff_summary.png",
        overview_df,
    )

    report_md = _build_report(
        sprint_paths.sprint_root,
        pilot1a_df,
        probe1_df,
        probe2_df,
        overview_df,
    )
    figure_paths = [
        sprint_paths.figures_dir / "pilot1a_bsv_heatmap_comparison.png",
        sprint_paths.figures_dir / "pilot1a_neighborhood_entropy_comparison.png",
        sprint_paths.figures_dir / "pilot1a_top1_dominance_comparison.png",
        sprint_paths.figures_dir / "pilot1b_probe1_progression_comparison.png",
        sprint_paths.figures_dir / "pilot1b_probe1_top1_dominance_comparison.png",
        sprint_paths.figures_dir / "pilot1b_probe1_entropy_comparison.png",
        sprint_paths.figures_dir / "pilot1b_probe1_class_mean_bsv_comparison.png",
        sprint_paths.figures_dir / "pilot1b_probe2_progression_comparison.png",
        sprint_paths.figures_dir / "pilot1b_probe2_top1_dominance_comparison.png",
        sprint_paths.figures_dir / "pilot1b_probe2_entropy_comparison.png",
        sprint_paths.figures_dir / "config_tradeoff_summary.png",
    ]
    build_pdf_report(
        report_md,
        figure_paths,
        sprint_paths.report_dir / "GAIRAv3_Pilot1_comparator_cfg05_cfg08_report.pdf",
    )
    print(f"Wrote Pilot 1 comparator outputs under {sprint_paths.sprint_root}")


if __name__ == "__main__":
    main()
