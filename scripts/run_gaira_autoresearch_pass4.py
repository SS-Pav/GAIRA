from __future__ import annotations

from pathlib import Path
import traceback

import pandas as pd

from gaira.autoresearch_storage import DEFAULT_STORAGE_CONFIG_PATH, initialize_autoresearch_sprint, load_autoresearch_storage_config
from gaira.demo.autoresearch_pass3_utils import apply_pass3_filter_mode
from gaira.demo.autoresearch_pass4_utils import (
    Pass4HarnessConfig,
    apply_readout_mode,
    build_pass4_markdown_report,
    build_pass4_search_space,
    build_pdf_report,
    compute_axis_visibility_metrics,
    plot_axis_visibility_comparison,
    plot_pass4_best_vs_baseline,
    plot_pass4_leaderboard,
    save_pass4_summary_tables,
    score_pass4_outputs,
    write_pass4_run_artifacts,
)
from gaira.demo.gaira_experiment_runner_utils import (
    build_source_role_sets,
    load_architecture_registries,
    load_grounding_family_dataframe,
    load_query_dataframe,
    resolve_experiment,
    retrieval_hit_summary,
)
from gaira.demo.gaira_pilot_utils import compute_stability_tables, pairwise_delta_bsv
from gaira.demo.raw_bsv_pilot_utils import (
    apply_source_role_policy,
    build_bsv_profiles,
    build_group_mean_query_df,
    group_mean_bsv,
    load_ontology_rules,
    map_references_to_axes,
)


ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = ROOT / "reports" / "gaira_architecture_scaffold_v2"
PHASE1_DIR = ROOT / "reports" / "gaira_phase1_registry_audit_v2"


def main() -> None:
    storage_cfg = load_autoresearch_storage_config(DEFAULT_STORAGE_CONFIG_PATH)
    sprint_paths = initialize_autoresearch_sprint(
        DEFAULT_STORAGE_CONFIG_PATH,
        sprint_id=f"{storage_cfg.sprint_id}/pass4_readout_diagnostic",
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
    search_df = build_pass4_search_space()
    results_rows = []
    for _, row in search_df.iterrows():
        harness_config = Pass4HarnessConfig(
            experiment_id=str(row["experiment_id"]),
            subset_alias=str(row["subset_alias"]),
            panel_name=str(row["panel_name"]),
            readout_mode=str(row["readout_mode"]),
        )
        try:
            resolved = resolve_experiment(registries, harness_config.experiment_id)
            query_df = load_query_dataframe(resolved.dataset_row)

            original_grounding_names = resolved.grounding_family_names
            object.__setattr__(resolved, "grounding_family_names", ["universal_biochemical_grounding"])
            try:
                grounding_df, family_to_sources, _ = load_grounding_family_dataframe(resolved, registries)
            finally:
                object.__setattr__(resolved, "grounding_family_names", original_grounding_names)
            grounding_df = apply_pass3_filter_mode(grounding_df, "purine_focused_universal")

            primary_sources, caveat_only_sources = build_source_role_sets(resolved, family_to_sources)
            primary_sources = {key for key in primary_sources if key in set(grounding_df["source_key"].astype(str))}
            caveat_only_sources = {key for key in caveat_only_sources if key in set(grounding_df["source_key"].astype(str))}

            ontology_rules = load_ontology_rules(ROOT / "config" / "phase2_bsv_ontology_rules_v2.csv")
            raw_mapping_df = map_references_to_axes(grounding_df, ontology_rules)
            mapping_df = apply_source_role_policy(
                raw_mapping_df,
                grounding_df,
                primary_sources=primary_sources,
                caveat_only_sources=caveat_only_sources,
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
            class_mean_bsv_df = class_mean_bsv_df.sort_values("class_label").reset_index(drop=True)
            readout_bsv_df = apply_readout_mode(class_mean_bsv_df, readout_mode=harness_config.readout_mode)
            retrieval_summary_df = retrieval_hit_summary(class_mean_retrieval_df)

            delta_df = None
            inter_class_distance_df = None
            intra_class_variance_df = None
            if harness_config.panel_name == "small2023_cellline":
                per_spectrum_bsv_df, _ = build_bsv_profiles(
                    query_df,
                    grounding_df,
                    mapping_df,
                    top_k=5,
                    normalization_mode="per_spectrum_sum",
                    similarity_metric="cosine",
                )
                per_spectrum_readout_df = apply_readout_mode(per_spectrum_bsv_df, readout_mode=harness_config.readout_mode)
                intra_class_variance_df, inter_class_distance_df = compute_stability_tables(per_spectrum_readout_df, readout_bsv_df)
                delta_df = pairwise_delta_bsv(readout_bsv_df)
            else:
                delta_df = pairwise_delta_bsv(readout_bsv_df)

            scores = score_pass4_outputs(
                harness_config.panel_name,
                readout_bsv_df,
                retrieval_summary_df,
                delta_df,
                inter_class_distance_df,
            )
            run_dir = write_pass4_run_artifacts(
                sprint_paths,
                harness_config,
                class_mean_bsv_df=readout_bsv_df,
                retrieval_summary_df=retrieval_summary_df,
                delta_df=delta_df,
                inter_class_distance_df=inter_class_distance_df,
                intra_class_variance_df=intra_class_variance_df,
                scores=scores,
            )
            results_rows.append(
                {
                    **row.to_dict(),
                    **scores,
                    "status": "completed",
                    "error_message": "",
                    "run_id": harness_config.run_id,
                    "run_dir": str(run_dir),
                }
            )
        except Exception as exc:
            results_rows.append(
                {
                    **row.to_dict(),
                    "expected_axis_uplift_score": None,
                    "top_hit_plausibility_score": None,
                    "mean_primary_entropy": None,
                    "mean_secondary_axis_mass": None,
                    "mean_inter_class_distance": None,
                    "caveat_domination_penalty": None,
                    "matrix_collapse_penalty": None,
                    "overall_score": None,
                    "status": "failed",
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "run_id": harness_config.run_id,
                    "run_dir": "",
                }
            )

    results_df = pd.DataFrame(results_rows)
    completed_df = results_df[results_df["status"] == "completed"].copy()
    summary_tables = save_pass4_summary_tables(sprint_paths, search_df, results_df)
    figure_paths = [
        plot_pass4_leaderboard(completed_df, sprint_paths),
        plot_pass4_best_vs_baseline(summary_tables["compare"], sprint_paths, "cspp_metabolite_spike_validation", "pass4_best_vs_baseline_cspp.png"),
        plot_pass4_best_vs_baseline(summary_tables["compare"], sprint_paths, "serum_ag_uricase_validation", "pass4_best_vs_baseline_uricase.png"),
        plot_pass4_best_vs_baseline(summary_tables["compare"], sprint_paths, "small2023_cellline", "pass4_best_vs_baseline_small2023.png"),
        plot_axis_visibility_comparison(completed_df, sprint_paths),
    ]
    report_md = build_pass4_markdown_report(
        sprint_paths,
        completed_df,
        search_df,
        summary_tables["best"],
        summary_tables["compare"],
    )
    build_pdf_report(report_md, figure_paths, sprint_paths.report_dir / "GAIRA_autoresearch_v1_pass4_report.pdf")
    print(f"Wrote pass 4 autoresearch outputs under {sprint_paths.sprint_root}")


if __name__ == "__main__":
    main()
