from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import pandas as pd

from gaira.autoresearch_storage import DEFAULT_STORAGE_CONFIG_PATH, initialize_autoresearch_sprint, load_autoresearch_storage_config
from gaira.demo.autoresearch_pass2_utils import compute_pass2_outputs, score_pass2_outputs, write_pass2_run_artifacts
from gaira.demo.autoresearch_pass3_utils import (
    Pass3HarnessConfig,
    apply_pass3_filter_mode,
    build_pass3_markdown_report,
    build_pass3_search_space,
    build_pdf_report,
    plot_pass3_best_vs_pass2,
    plot_pass3_leaderboard,
    save_pass3_summary_tables,
)
from gaira.demo.gaira_experiment_runner_utils import (
    build_source_role_sets,
    load_architecture_registries,
    load_grounding_family_dataframe,
    load_query_dataframe,
    resolve_experiment,
    retrieval_hit_summary,
)
from gaira.demo.raw_bsv_pilot_utils import (
    apply_source_role_policy,
    build_bsv_profiles,
    build_group_mean_query_df,
    group_mean_bsv,
    load_ontology_rules,
    map_references_to_axes,
)
from gaira.demo.autoresearch_utils import comparator_map_for_alias, compute_delta_by_mapping, compute_panel_structural_summary


ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = ROOT / "reports" / "gaira_architecture_scaffold_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GAIRAv3 autoresearch v1 pass 3 fine refinement.")
    parser.add_argument("--storage-config-path", type=Path, default=DEFAULT_STORAGE_CONFIG_PATH)
    parser.add_argument("--sprint-subdir", type=str, default="pass3_fine_refine")
    parser.add_argument("--grounding-family-registry-path", type=Path, default=ROOT / "config" / "gaira_grounding_family_registry_v1.csv")
    parser.add_argument("--target-family-registry-path", type=Path, default=ROOT / "config" / "gaira_target_family_registry_v1.csv")
    parser.add_argument("--inference-lane-registry-path", type=Path, default=ROOT / "config" / "gaira_inference_lane_registry_v2.csv")
    parser.add_argument("--representation-mode-registry-path", type=Path, default=ROOT / "config" / "gaira_representation_mode_registry_v2.csv")
    parser.add_argument("--dataset-experiment-registry-path", type=Path, default=ROOT / "config" / "gaira_dataset_experiment_registry_v2.csv")
    parser.add_argument("--experiment-plan-path", type=Path, default=ARCH_DIR / "first_pass_experiment_plan.csv")
    parser.add_argument("--phase1-registry-path", type=Path, default=ROOT / "reports" / "gaira_phase1_registry_audit_v2" / "phase1_dataset_registry_v2.csv")
    parser.add_argument("--phase1-grounding-map-path", type=Path, default=ROOT / "reports" / "gaira_phase1_registry_audit_v2" / "phase1_target_grounding_map_v2.csv")
    parser.add_argument("--phase1-exclusions-path", type=Path, default=ROOT / "reports" / "gaira_phase1_registry_audit_v2" / "phase1_grounding_exclusions.csv")
    parser.add_argument("--pass2-best-config-path", type=Path, default=Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pass2_chemistry_focus/tables/best_config_by_panel.csv"))
    return parser.parse_args()


def compute_pass3_outputs(*, resolved, registries, project_root: Path, harness_config: Pass3HarnessConfig) -> dict[str, object]:
    query_df = load_query_dataframe(resolved.dataset_row)
    structural_summary_df = compute_panel_structural_summary(query_df)

    original_grounding_names = resolved.grounding_family_names
    object.__setattr__(resolved, "grounding_family_names", ["universal_biochemical_grounding"])
    try:
        grounding_df, family_to_sources, unavailable_sources = load_grounding_family_dataframe(resolved, registries)
    finally:
        object.__setattr__(resolved, "grounding_family_names", original_grounding_names)

    grounding_df = apply_pass3_filter_mode(grounding_df, harness_config.universal_grounding_filter_mode)
    primary_sources, caveat_only_sources = build_source_role_sets(resolved, family_to_sources)
    primary_sources = {key for key in primary_sources if key in set(grounding_df["source_key"].astype(str))}
    caveat_only_sources = {key for key in caveat_only_sources if key in set(grounding_df["source_key"].astype(str))}

    ontology_rules = load_ontology_rules(project_root / "config" / "phase2_bsv_ontology_rules_v2.csv")
    raw_mapping_df = map_references_to_axes(grounding_df, ontology_rules)
    mapping_df = apply_source_role_policy(
        raw_mapping_df,
        grounding_df,
        primary_sources=primary_sources,
        caveat_only_sources=caveat_only_sources,
    )
    query_input_df = build_group_mean_query_df(query_df, group_col="class_label")
    per_spectrum_df, retrieval_df = build_bsv_profiles(
        query_input_df,
        grounding_df,
        mapping_df,
        top_k=harness_config.top_k,
        normalization_mode="per_spectrum_sum",
        similarity_metric=harness_config.similarity_metric,
    )
    group_means_df = group_mean_bsv(per_spectrum_df, group_col="class_label")
    comparator_map = comparator_map_for_alias(harness_config.subset_alias, group_means_df)
    delta_df = compute_delta_by_mapping(group_means_df, comparator_map)
    retrieval_summary_df = retrieval_hit_summary(retrieval_df)
    filter_summary_df = (
        grounding_df.groupby(["dataset_id", "source_key", "class_label", "compound_label"], dropna=False)
        .size()
        .reset_index(name="reference_count")
        .sort_values(["dataset_id", "class_label", "compound_label"])
        .reset_index(drop=True)
    )
    return {
        "query_df": query_df,
        "structural_summary_df": structural_summary_df,
        "grounding_df": grounding_df,
        "mapping_df": mapping_df,
        "per_spectrum_df": per_spectrum_df,
        "group_means_df": group_means_df,
        "delta_df": delta_df,
        "retrieval_df": retrieval_df,
        "retrieval_summary_df": retrieval_summary_df,
        "filter_summary_df": filter_summary_df,
        "available_sources": sorted(grounding_df["source_key"].astype(str).unique().tolist()),
        "unavailable_sources": unavailable_sources,
        "primary_sources": sorted(primary_sources),
        "caveat_only_sources": sorted(caveat_only_sources),
    }


def main() -> None:
    args = parse_args()
    storage_cfg = load_autoresearch_storage_config(args.storage_config_path)
    sprint_id = f"{storage_cfg.sprint_id}/{args.sprint_subdir}"
    sprint_paths = initialize_autoresearch_sprint(args.storage_config_path, sprint_id=sprint_id)
    registries = load_architecture_registries(
        grounding_family_registry_path=args.grounding_family_registry_path,
        target_family_registry_path=args.target_family_registry_path,
        inference_lane_registry_path=args.inference_lane_registry_path,
        representation_mode_registry_path=args.representation_mode_registry_path,
        dataset_experiment_registry_path=args.dataset_experiment_registry_path,
        experiment_plan_path=args.experiment_plan_path,
        phase1_registry_path=args.phase1_registry_path,
        phase1_grounding_map_path=args.phase1_grounding_map_path,
        phase1_exclusions_path=args.phase1_exclusions_path,
    )
    search_df = build_pass3_search_space()
    results_rows = []
    for _, row in search_df.iterrows():
        harness_config = Pass3HarnessConfig(
            experiment_id=str(row["experiment_id"]),
            subset_alias=str(row["subset_alias"]),
            panel_name=str(row["panel_name"]),
            universal_grounding_filter_mode=str(row["universal_grounding_filter_mode"]),
            top_k=int(row["top_k"]),
        )
        try:
            resolved = resolve_experiment(registries, harness_config.experiment_id)
            outputs = compute_pass3_outputs(
                resolved=resolved,
                registries=registries,
                project_root=ROOT,
                harness_config=harness_config,
            )
            scores = score_pass2_outputs(harness_config, outputs)
            run_dir = write_pass2_run_artifacts(sprint_paths, harness_config, outputs, scores)
            results_rows.append(
                {
                    **row.to_dict(),
                    "run_id": harness_config.run_id,
                    **scores,
                    "status": "completed",
                    "error_message": "",
                    "run_dir": str(run_dir),
                }
            )
        except Exception as exc:
            results_rows.append(
                {
                    **row.to_dict(),
                    "run_id": harness_config.run_id,
                    "expected_axis_uplift_score": None,
                    "caveat_domination_penalty": None,
                    "matrix_collapse_penalty": None,
                    "top_hit_plausibility_score": None,
                    "stability_proxy_score": None,
                    "overall_score": None,
                    "status": "failed",
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "run_dir": "",
                }
            )

    results_df = pd.DataFrame(results_rows)
    completed_df = results_df[results_df["status"] == "completed"].copy()
    pass2_best_df = pd.read_csv(args.pass2_best_config_path)
    summary_tables = save_pass3_summary_tables(sprint_paths, search_df, results_df, pass2_best_df)
    figure_paths = []
    figure_paths.append(plot_pass3_leaderboard(completed_df, sprint_paths))
    figure_paths.append(
        plot_pass3_best_vs_pass2(
            summary_tables["best"],
            pass2_best_df,
            sprint_paths,
            "cspp_metabolite_spike_validation",
            "pass3_best_vs_pass2_cspp.png",
        )
    )
    figure_paths.append(
        plot_pass3_best_vs_pass2(
            summary_tables["best"],
            pass2_best_df,
            sprint_paths,
            "serum_ag_uricase_validation",
            "pass3_best_vs_pass2_uricase.png",
        )
    )
    report_md = build_pass3_markdown_report(
        sprint_paths,
        completed_df,
        search_df,
        summary_tables["best"],
        summary_tables["compare"],
    )
    build_pdf_report(report_md, figure_paths, sprint_paths.report_dir / "GAIRA_autoresearch_v1_pass3_report.pdf")
    print(f"Wrote pass 3 autoresearch outputs under {sprint_paths.sprint_root}")


if __name__ == "__main__":
    main()
