from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gaira.autoresearch_storage import DEFAULT_STORAGE_CONFIG_PATH, initialize_autoresearch_sprint, load_autoresearch_storage_config
from gaira.demo.autoresearch_pass3_utils import apply_pass3_filter_mode
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
    plot_bsv_heatmap,
    plot_metric_bar,
    plot_mixture_progression,
    plot_neighborhood_composition_for_class,
    plot_pairwise_delta_heatmap,
    plot_pca_by_class,
    plot_radar_for_class,
)
from gaira.demo.raw_bsv_pilot_utils import (
    apply_source_role_policy,
    build_bsv_profiles,
    build_group_mean_query_df,
    compute_local_pca,
    load_ontology_rules,
    map_references_to_axes,
)


ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = ROOT / "reports" / "gaira_architecture_scaffold_v2"
PHASE1_DIR = ROOT / "reports" / "gaira_phase1_registry_audit_v2"

SUBSET_ALIAS = "small2023_mixture_probe1"
SPRINT_SUBDIR = "pilot1b_small2023_mixture_probe1_fingerprint"
ONTOLOGY_PATH = ROOT / "config" / "phase2_bsv_ontology_rules_v2.csv"
PROBE2_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot1b_small2023_mixture_fingerprint"
)


def _load_dataset_row(registries, subset_alias: str) -> pd.Series:
    matches = registries.dataset_experiments[registries.dataset_experiments["subset_alias"].astype(str) == subset_alias].copy()
    if matches.empty or len(matches) > 1:
        raise RuntimeError(f"Could not resolve unique subset alias {subset_alias}")
    return matches.iloc[0]


def _build_probe_comparison_table(
    *,
    probe1_class_mean_bsv: pd.DataFrame,
    probe1_entropy: pd.DataFrame,
    probe1_top1: pd.DataFrame,
    probe1_retrieval: pd.DataFrame,
    probe2_root: Path,
) -> pd.DataFrame:
    probe2_class_mean = pd.read_csv(probe2_root / "tables" / "class_mean_bsv.csv")
    probe2_entropy = pd.read_csv(probe2_root / "tables" / "class_neighborhood_entropy.csv")
    probe2_top1 = pd.read_csv(probe2_root / "tables" / "class_top1_dominance.csv")
    probe2_retrieval = pd.read_csv(probe2_root / "tables" / "retrieval_hit_summary_by_class.csv")

    def _top_compound(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for class_label, group in df.groupby("query_class_label", sort=True):
            top = group.sort_values("total_support_weight", ascending=False).iloc[0]
            rows.append(
                {
                    "class_label": str(class_label),
                    "top_compound": str(top["reference_compound_label"]),
                    "top_compound_support_weight": float(top["total_support_weight"]),
                }
            )
        return pd.DataFrame(rows)

    p1 = (
        probe1_class_mean_bsv[["class_label", "nucleic_acid", "small_molecule_metabolite", "substrate_adsorption_bias"]]
        .merge(probe1_entropy, on="class_label", how="left")
        .merge(probe1_top1, on="class_label", how="left")
        .merge(_top_compound(probe1_retrieval), on="class_label", how="left")
    )
    p2 = (
        probe2_class_mean[["class_label", "nucleic_acid", "small_molecule_metabolite", "substrate_adsorption_bias"]]
        .merge(probe2_entropy, on="class_label", how="left")
        .merge(probe2_top1, on="class_label", how="left")
        .merge(_top_compound(probe2_retrieval), on="class_label", how="left")
    )
    merged = p1.merge(p2, on="class_label", how="outer", suffixes=("_probe1", "_probe2"))

    def _collapse_flags(df: pd.DataFrame) -> tuple[dict[str, bool], bool]:
        work = df.set_index("class_label")
        baseline = work.loc["c00", ["nucleic_acid", "small_molecule_metabolite", "substrate_adsorption_bias"]].to_numpy(dtype=float)
        flags = {}
        for label in work.index.astype(str):
            vec = work.loc[label, ["nucleic_acid", "small_molecule_metabolite", "substrate_adsorption_bias"]].to_numpy(dtype=float)
            flags[label] = bool(((vec - baseline) ** 2).sum() ** 0.5 < 1e-9)
        c100_sep = not flags.get("c100", False)
        return flags, c100_sep

    p1_collapse, p1_c100_sep = _collapse_flags(probe1_class_mean_bsv)
    p2_collapse, p2_c100_sep = _collapse_flags(probe2_class_mean)
    merged["probe1_collapsed_like_c00_to_c50"] = merged["class_label"].astype(str).map(p1_collapse).fillna(False)
    merged["probe2_collapsed_like_c00_to_c50"] = merged["class_label"].astype(str).map(p2_collapse).fillna(False)
    merged["probe1_c100_separated"] = p1_c100_sep
    merged["probe2_c100_separated"] = p2_c100_sep
    return merged.sort_values("class_label").reset_index(drop=True)


def _build_probe1_markdown_report(
    output_path: Path,
    *,
    subset_alias: str,
    config_summary: dict[str, object],
    class_mean_bsv: pd.DataFrame,
    class_neighborhood_df: pd.DataFrame,
    class_neighborhood_entropy_df: pd.DataFrame,
    class_top1_dominance_df: pd.DataFrame,
    class_axis_entropy_df: pd.DataFrame,
    intra_class_variance_df: pd.DataFrame,
    inter_class_distance_df: pd.DataFrame,
    progression_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    axes = ["nucleic_acid", "small_molecule_metabolite", "substrate_adsorption_bias"]
    mean_variance = intra_class_variance_df.drop(columns=["class_label"]).mean(axis=1)
    top_distances = (
        inter_class_distance_df[inter_class_distance_df["class_label_a"] != inter_class_distance_df["class_label_b"]]
        .sort_values("euclidean_distance", ascending=False)
        .head(8)
    )
    ordered = infer_mixture_order(class_mean_bsv["class_label"].astype(str).tolist())
    lines = [
        "# GAIRAv3 Pilot 1b: small2023_mixture_probe1 Fingerprint Report",
        "",
        "## 1. Overview",
        "- This is a strict Probe1 mirror of Pilot 1b so Probe1 and Probe2 can be compared under the exact same locked baseline and fingerprint readout.",
        "- No chemistry, retrieval, ontology, or plotting logic was changed beyond the subset switch and the required comparison table.",
        f"- subset alias: `{subset_alias}`",
        *[f"- {key}: `{value}`" for key, value in config_summary.items()],
        "",
        "## 2. Mixture-class Fingerprints",
    ]
    for label in ordered:
        row = class_mean_bsv[class_mean_bsv["class_label"].astype(str) == label].iloc[0]
        hood = class_neighborhood_df[class_neighborhood_df["class_label"].astype(str) == label].sort_values(
            "support_fraction", ascending=False
        ).head(4)
        axis_entropy = float(class_axis_entropy_df[class_axis_entropy_df["class_label"].astype(str) == label]["axis_entropy"].iloc[0])
        hood_entropy = float(
            class_neighborhood_entropy_df[class_neighborhood_entropy_df["class_label"].astype(str) == label][
                "neighborhood_entropy"
            ].iloc[0]
        )
        top1 = float(
            class_top1_dominance_df[class_top1_dominance_df["class_label"].astype(str) == label]["top1_fraction"].iloc[0]
        )
        lines.extend(
            [
                f"### {label}",
                "- BSV profile: " + ", ".join([f"`{axis}={float(row[axis]):.3f}`" for axis in axes]),
                f"- Axis entropy: `{axis_entropy:.3f}`",
                f"- Neighborhood entropy: `{hood_entropy:.3f}`",
                f"- Top1 dominance: `{top1:.3f}`",
                "- Neighborhood composition: "
                + "; ".join([f"`{r.compound_label}` ({float(r.support_fraction):.3f})" for r in hood.itertuples(index=False)]),
                f"- Radar figure: `figures/radar_{label}.png`",
                f"- Neighborhood figure: `figures/neighborhood_{label}.png`",
                "",
            ]
        )
    lines.extend(
        [
            "## 3. Stability",
            f"- Mean intra-class variance across classes: `{float(mean_variance.mean()):.6f}`",
            "- Lowest-variance classes:",
        ]
    )
    tmp = intra_class_variance_df.copy()
    tmp["mean_variance"] = tmp.drop(columns=["class_label"]).mean(axis=1)
    for row in tmp.sort_values("mean_variance").head(6).itertuples(index=False):
        lines.append(f"- `{row.class_label}` mean variance `{float(row.mean_variance):.6f}`")
    lines.append("- Largest inter-class distances:")
    for row in top_distances.itertuples(index=False):
        lines.append(f"- `{row.class_label_a}` vs `{row.class_label_b}` distance `{float(row.euclidean_distance):.6f}`")
    lines.extend(
        [
            "",
            "## 4. Mixture Behavior",
            "- Mixture labels are treated as ordered only because the class codes explicitly encode `c00` ... `c100` levels.",
            "- Dominant-component alignment is judged only relative to the observed endpoint classes, not from hidden composition metadata.",
        ]
    )
    for row in progression_df.sort_values("mixture_code_numeric").itertuples(index=False):
        lines.append(
            f"- `{row.class_label}`: toward-high-endpoint score `{float(row.toward_high_endpoint_score):.3f}`, "
            f"`{row.low_endpoint_top_compound}` fraction `{float(row.low_endpoint_compound_fraction):.3f}`, "
            f"`{row.high_endpoint_top_compound}` fraction `{float(row.high_endpoint_compound_fraction):.3f}`"
        )
    probe1_collapse = bool(comparison_df[comparison_df["class_label"].isin(["c00", "c01", "c10", "c25", "c50"])]["probe1_collapsed_like_c00_to_c50"].all())
    probe2_collapse = bool(comparison_df[comparison_df["class_label"].isin(["c00", "c01", "c10", "c25", "c50"])]["probe2_collapsed_like_c00_to_c50"].all())
    probe1_c100_sep = bool(comparison_df["probe1_c100_separated"].iloc[0])
    probe2_c100_sep = bool(comparison_df["probe2_c100_separated"].iloc[0])
    lines.extend(
        [
            "",
            "## 5. Probe1 vs Probe2 Comparison",
            "- Comparison table: `tables/probe1_vs_probe2_comparison.csv`",
            f"- Probe1 `c00-c50` collapse present: `{probe1_collapse}`",
            f"- Probe2 `c00-c50` collapse present: `{probe2_collapse}`",
            f"- Probe1 `c100` separation present: `{probe1_c100_sep}`",
            f"- Probe2 `c100` separation present: `{probe2_c100_sep}`",
            "",
            "## 6. Interpretation",
            "- Interpretation remains conservative and probe-local.",
            "- Any apparent chemistry should be read as relative support shifts within the current purine-adjacent biochemical vocabulary, not as direct molecular calls.",
            "",
            "## 7. Direct Answers",
            f"1. Probe1 endpoint-heavy behavior matched Probe2: `{probe1_collapse and probe1_c100_sep}`",
            f"2. `c00-c50` collapse in Probe1: `{probe1_collapse}`",
            f"3. `c100` separated in Probe1: `{probe1_c100_sep}`",
            f"4. Probe1 and Probe2 are similar enough that the dominant pattern now looks more like chemistry/retrieval saturation than a probe artifact: `{probe1_collapse == probe2_collapse and probe1_c100_sep == probe2_c100_sep}`",
            f"5. Formal Pilot 1c cross-probe comparison justified next: `{probe1_collapse == probe2_collapse and probe1_c100_sep == probe2_c100_sep}`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    dataset_row = _load_dataset_row(registries, SUBSET_ALIAS)
    query_df = load_query_dataframe(dataset_row)
    resolved = ResolvedExperiment(
        experiment_row=dataset_row,
        dataset_row=dataset_row,
        subset_alias=SUBSET_ALIAS,
        grounding_family_names=["universal_biochemical_grounding"],
    )
    _, _, pca_df = compute_local_pca(query_df, n_components=3)

    grounding_df, family_to_sources, unavailable_sources = load_grounding_family_dataframe(resolved, registries)
    grounding_df = apply_pass3_filter_mode(grounding_df, "purine_focused_universal")

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

    per_spectrum_bsv_df, per_spectrum_retrieval_df = build_bsv_profiles(
        query_df,
        grounding_df,
        mapping_df,
        top_k=5,
        normalization_mode="per_spectrum_sum",
        similarity_metric="cosine",
    )
    class_mean_query_df = build_group_mean_query_df(query_df, group_col="class_label")
    class_mean_bsv_df, class_mean_retrieval_df = build_bsv_profiles(
        class_mean_query_df,
        grounding_df,
        mapping_df,
        top_k=5,
        normalization_mode="per_spectrum_sum",
        similarity_metric="cosine",
    )
    ordered_labels = infer_mixture_order(class_mean_bsv_df["class_label"].astype(str).tolist())
    class_mean_bsv_df["class_label"] = class_mean_bsv_df["class_label"].astype(str)
    class_mean_bsv_df["class_order"] = class_mean_bsv_df["class_label"].map({label: i for i, label in enumerate(ordered_labels)})
    class_mean_bsv_df = class_mean_bsv_df.sort_values("class_order").drop(columns=["class_order"]).reset_index(drop=True)

    pairwise_delta_df = pairwise_delta_bsv(class_mean_bsv_df)
    retrieval_summary_by_class_df = retrieval_hit_summary(class_mean_retrieval_df)
    intra_class_variance_df, inter_class_distance_df = compute_stability_tables(per_spectrum_bsv_df, class_mean_bsv_df)
    class_neighborhood_df = build_class_topk_neighborhood_composition(per_spectrum_retrieval_df)
    class_neighborhood_entropy_df = build_class_neighborhood_entropy(class_neighborhood_df)
    class_top1_dominance_df = build_class_top1_dominance(class_neighborhood_df)
    class_axis_entropy_df = build_class_axis_entropy(class_mean_bsv_df)
    progression_df = build_mixture_progression_summary(class_mean_bsv_df, class_neighborhood_df)
    comparison_df = _build_probe_comparison_table(
        probe1_class_mean_bsv=class_mean_bsv_df,
        probe1_entropy=class_neighborhood_entropy_df,
        probe1_top1=class_top1_dominance_df,
        probe1_retrieval=retrieval_summary_by_class_df,
        probe2_root=PROBE2_ROOT,
    )

    plot_pca_by_class(
        pca_df,
        sprint_paths.figures_dir / "pca_by_mixture_class.png",
        title="small2023_mixture_probe1 PCA of Canonical Spectra",
        legend_title="Mixture class",
    )
    plot_bsv_heatmap(
        class_mean_bsv_df,
        sprint_paths.figures_dir / "class_mean_bsv_heatmap.png",
        "small2023_mixture_probe1 Class Mean BSV",
    )
    plot_pairwise_delta_heatmap(
        pairwise_delta_df,
        axis="small_molecule_metabolite",
        output_path=sprint_paths.figures_dir / "pairwise_delta_bsv_heatmap_small_molecule_metabolite.png",
    )
    plot_mixture_progression(
        progression_df,
        sprint_paths.figures_dir / "mixture_progression_alignment.png",
        title="Probe1 Mixture Progression Relative to Endpoint Fingerprints",
    )
    for class_label in ordered_labels:
        plot_radar_for_class(class_mean_bsv_df, class_label, sprint_paths.figures_dir / f"radar_{class_label}.png")
        plot_neighborhood_composition_for_class(
            class_neighborhood_df,
            class_label,
            sprint_paths.figures_dir / f"neighborhood_{class_label}.png",
        )
    plot_metric_bar(
        class_neighborhood_entropy_df,
        "neighborhood_entropy",
        "Neighborhood Entropy by Mixture Class",
        sprint_paths.figures_dir / "neighborhood_entropy_comparison.png",
    )
    plot_metric_bar(
        class_top1_dominance_df,
        "top1_fraction",
        "Top1 Dominance by Mixture Class",
        sprint_paths.figures_dir / "top1_dominance_comparison.png",
    )

    pca_df.to_csv(sprint_paths.tables_dir / "pca_coordinates.csv", index=False)
    per_spectrum_bsv_df.to_csv(sprint_paths.tables_dir / "per_spectrum_bsv.csv", index=False)
    class_mean_bsv_df.to_csv(sprint_paths.tables_dir / "class_mean_bsv.csv", index=False)
    pairwise_delta_df.to_csv(sprint_paths.tables_dir / "pairwise_delta_bsv.csv", index=False)
    intra_class_variance_df.to_csv(sprint_paths.tables_dir / "intra_class_bsv_variance.csv", index=False)
    inter_class_distance_df.to_csv(sprint_paths.tables_dir / "inter_class_bsv_distance.csv", index=False)
    class_neighborhood_df.to_csv(sprint_paths.tables_dir / "class_topk_neighborhood_composition.csv", index=False)
    class_neighborhood_entropy_df.to_csv(sprint_paths.tables_dir / "class_neighborhood_entropy.csv", index=False)
    class_top1_dominance_df.to_csv(sprint_paths.tables_dir / "class_top1_dominance.csv", index=False)
    class_axis_entropy_df.to_csv(sprint_paths.tables_dir / "class_axis_entropy.csv", index=False)
    retrieval_summary_by_class_df.to_csv(sprint_paths.tables_dir / "retrieval_hit_summary_by_class.csv", index=False)
    per_spectrum_retrieval_df.to_csv(sprint_paths.tables_dir / "per_spectrum_retrieval_hits.csv", index=False)
    progression_df.to_csv(sprint_paths.tables_dir / "mixture_progression_summary.csv", index=False)
    comparison_df.to_csv(sprint_paths.tables_dir / "probe1_vs_probe2_comparison.csv", index=False)

    config_summary = {
        "representation_mode": "raw_direct_bsv_input",
        "grounding_mode": "universal_only",
        "universal_grounding_filter_mode": "purine_focused_universal",
        "aggregation_mode": "class_mean_spectrum_then_bsv",
        "ontology_mode": "tier1_plus_subclass",
        "similarity_metric": "cosine",
        "plausibility_scoring_mode": "baseline_plausibility",
        "pca_grouping_mode": "class_label_groups",
        "top_k": 5,
    }
    (sprint_paths.report_dir / "run_config.json").write_text(
        json.dumps(
            {
                "experiment_id": "pilot1b_small2023_mixture_probe1",
                "subset_alias": resolved.subset_alias,
                "dataset_id": str(resolved.dataset_row["dataset_id"]),
                "subset_id": str(resolved.dataset_row["subset_id"]),
                "available_sources": sorted(grounding_df["source_key"].astype(str).unique().tolist()),
                "unavailable_sources": unavailable_sources,
                **config_summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report_md = sprint_paths.report_dir / "GAIRAv3_Pilot1b_small2023_mixture_probe1_fingerprint_report.md"
    _build_probe1_markdown_report(
        report_md,
        subset_alias=resolved.subset_alias,
        config_summary=config_summary,
        class_mean_bsv=class_mean_bsv_df,
        class_neighborhood_df=class_neighborhood_df,
        class_neighborhood_entropy_df=class_neighborhood_entropy_df,
        class_top1_dominance_df=class_top1_dominance_df,
        class_axis_entropy_df=class_axis_entropy_df,
        intra_class_variance_df=intra_class_variance_df,
        inter_class_distance_df=inter_class_distance_df,
        progression_df=progression_df,
        comparison_df=comparison_df,
    )
    figure_paths = [
        sprint_paths.figures_dir / "pca_by_mixture_class.png",
        sprint_paths.figures_dir / "class_mean_bsv_heatmap.png",
        sprint_paths.figures_dir / "pairwise_delta_bsv_heatmap_small_molecule_metabolite.png",
        sprint_paths.figures_dir / "mixture_progression_alignment.png",
    ]
    for class_label in ordered_labels:
        figure_paths.append(sprint_paths.figures_dir / f"radar_{class_label}.png")
        figure_paths.append(sprint_paths.figures_dir / f"neighborhood_{class_label}.png")
    figure_paths.extend(
        [
            sprint_paths.figures_dir / "neighborhood_entropy_comparison.png",
            sprint_paths.figures_dir / "top1_dominance_comparison.png",
        ]
    )
    build_pdf_report(
        report_md,
        figure_paths,
        sprint_paths.report_dir / "GAIRAv3_Pilot1b_small2023_mixture_probe1_fingerprint_report.pdf",
    )
    print(f"Wrote Probe1 Pilot 1b mixture fingerprint outputs under {sprint_paths.sprint_root}")


if __name__ == "__main__":
    main()
