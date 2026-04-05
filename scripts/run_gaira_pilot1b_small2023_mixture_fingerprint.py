from __future__ import annotations

import json
from pathlib import Path

from gaira.autoresearch_storage import DEFAULT_STORAGE_CONFIG_PATH, initialize_autoresearch_sprint, load_autoresearch_storage_config
from gaira.demo.autoresearch_pass3_utils import apply_pass3_filter_mode
from gaira.demo.gaira_experiment_runner_utils import (
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
    build_mixture_fingerprint_markdown_report,
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

SUBSET_ALIAS = "small2023_mixture_probe2"
SPRINT_SUBDIR = "pilot1b_small2023_mixture_fingerprint"
ONTOLOGY_PATH = ROOT / "config" / "phase2_bsv_ontology_rules_v2.csv"


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
    matches = registries.dataset_experiments[
        registries.dataset_experiments["subset_alias"].astype(str) == SUBSET_ALIAS
    ].copy()
    if matches.empty or len(matches) > 1:
        raise RuntimeError(f"Could not resolve unique subset alias {SUBSET_ALIAS}")
    dataset_row = matches.iloc[0]
    query_df = load_query_dataframe(dataset_row)
    from gaira.demo.gaira_experiment_runner_utils import ResolvedExperiment

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

    plot_pca_by_class(
        pca_df,
        sprint_paths.figures_dir / "pca_by_mixture_class.png",
        title="small2023_mixture_probe2 PCA of Canonical Spectra",
        legend_title="Mixture class",
    )
    plot_bsv_heatmap(
        class_mean_bsv_df,
        sprint_paths.figures_dir / "class_mean_bsv_heatmap.png",
        "small2023_mixture_probe2 Class Mean BSV",
    )
    plot_pairwise_delta_heatmap(
        pairwise_delta_df,
        axis="small_molecule_metabolite",
        output_path=sprint_paths.figures_dir / "pairwise_delta_bsv_heatmap_small_molecule_metabolite.png",
    )
    for class_label in ordered_labels:
        plot_radar_for_class(
            class_mean_bsv_df,
            class_label,
            sprint_paths.figures_dir / f"radar_{class_label}.png",
        )
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
    plot_mixture_progression(
        progression_df,
        sprint_paths.figures_dir / "mixture_progression_alignment.png",
        title="Mixture Progression Relative to Endpoint Fingerprints",
    )

    pca_df.to_csv(sprint_paths.tables_dir / "pca_coordinates.csv", index=False)
    per_spectrum_bsv_df.to_csv(sprint_paths.tables_dir / "per_spectrum_bsv.csv", index=False)
    class_mean_bsv_df.to_csv(sprint_paths.tables_dir / "class_mean_bsv.csv", index=False)
    pairwise_delta_df.to_csv(sprint_paths.tables_dir / "pairwise_delta_bsv.csv", index=False)
    retrieval_summary_by_class_df.to_csv(sprint_paths.tables_dir / "retrieval_hit_summary_by_class.csv", index=False)
    intra_class_variance_df.to_csv(sprint_paths.tables_dir / "intra_class_bsv_variance.csv", index=False)
    inter_class_distance_df.to_csv(sprint_paths.tables_dir / "inter_class_bsv_distance.csv", index=False)
    class_neighborhood_df.to_csv(sprint_paths.tables_dir / "class_topk_neighborhood_composition.csv", index=False)
    class_neighborhood_entropy_df.to_csv(sprint_paths.tables_dir / "class_neighborhood_entropy.csv", index=False)
    class_top1_dominance_df.to_csv(sprint_paths.tables_dir / "class_top1_dominance.csv", index=False)
    class_axis_entropy_df.to_csv(sprint_paths.tables_dir / "class_axis_entropy.csv", index=False)
    retrieval_summary_by_class_df.to_csv(sprint_paths.tables_dir / "retrieval_hit_summary_by_class.csv", index=False)
    per_spectrum_retrieval_df.to_csv(sprint_paths.tables_dir / "per_spectrum_retrieval_hits.csv", index=False)
    progression_df.to_csv(sprint_paths.tables_dir / "mixture_progression_summary.csv", index=False)
    mapping_df.to_csv(sprint_paths.tables_dir / "ontology_mapping_applied.csv", index=False)

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
                "experiment_id": "pilot1b_small2023_mixture_probe2",
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

    report_md = build_mixture_fingerprint_markdown_report(
        sprint_paths,
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
        sprint_paths.report_dir / "GAIRAv3_Pilot1b_small2023_mixture_fingerprint_report.pdf",
    )
    print(f"Wrote Pilot 1b mixture fingerprint outputs under {sprint_paths.sprint_root}")


if __name__ == "__main__":
    main()
