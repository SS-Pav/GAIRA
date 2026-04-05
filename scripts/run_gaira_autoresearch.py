from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import pandas as pd

from gaira.autoresearch_storage import DEFAULT_STORAGE_CONFIG_PATH, initialize_autoresearch_sprint
from gaira.demo.autoresearch_utils import (
    HarnessConfig,
    build_markdown_report,
    build_pdf_report,
    build_search_space,
    compute_run_outputs,
    plot_best_vs_baseline,
    plot_factor_ablation,
    plot_leaderboards,
    save_summary_tables,
    score_panel_outputs,
)
from gaira.demo.gaira_experiment_runner_utils import load_architecture_registries, resolve_experiment


ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = ROOT / "reports" / "gaira_architecture_scaffold_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first bounded GAIRAv3 autoresearch calibration sprint.")
    parser.add_argument("--storage-config-path", type=Path, default=DEFAULT_STORAGE_CONFIG_PATH)
    parser.add_argument("--sprint-id", type=str, default="")
    parser.add_argument("--grounding-family-registry-path", type=Path, default=ROOT / "config" / "gaira_grounding_family_registry_v1.csv")
    parser.add_argument("--target-family-registry-path", type=Path, default=ROOT / "config" / "gaira_target_family_registry_v1.csv")
    parser.add_argument("--inference-lane-registry-path", type=Path, default=ROOT / "config" / "gaira_inference_lane_registry_v2.csv")
    parser.add_argument("--representation-mode-registry-path", type=Path, default=ROOT / "config" / "gaira_representation_mode_registry_v2.csv")
    parser.add_argument("--dataset-experiment-registry-path", type=Path, default=ROOT / "config" / "gaira_dataset_experiment_registry_v2.csv")
    parser.add_argument("--experiment-plan-path", type=Path, default=ARCH_DIR / "first_pass_experiment_plan.csv")
    parser.add_argument("--phase1-registry-path", type=Path, default=ROOT / "reports" / "gaira_phase1_registry_audit_v2" / "phase1_dataset_registry_v2.csv")
    parser.add_argument("--phase1-grounding-map-path", type=Path, default=ROOT / "reports" / "gaira_phase1_registry_audit_v2" / "phase1_target_grounding_map_v2.csv")
    parser.add_argument("--phase1-exclusions-path", type=Path, default=ROOT / "reports" / "gaira_phase1_registry_audit_v2" / "phase1_grounding_exclusions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sprint_paths = initialize_autoresearch_sprint(args.storage_config_path, sprint_id=args.sprint_id.strip() or None)
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
    search_df = build_search_space()
    results_rows = []
    for _, row in search_df.iterrows():
        if row["status"] != "valid":
            results_rows.append(
                {
                    **row.to_dict(),
                    "run_id": None,
                    "expected_axis_uplift_score": None,
                    "caveat_domination_penalty": None,
                    "matrix_collapse_penalty": None,
                    "top_hit_plausibility_score": None,
                    "stability_proxy_score": None,
                    "overall_score": None,
                    "error_message": row["exclusion_reason"],
                }
            )
            continue
        harness_config = HarnessConfig(
            experiment_id=str(row["experiment_id"]),
            subset_alias=str(row["subset_alias"]),
            panel_name=str(row["panel_name"]),
            grounding_mode=str(row["grounding_mode"]),
            ontology_mode=str(row["ontology_mode"]),
            similarity_metric=str(row["similarity_metric"]),
            aggregation_mode=str(row["aggregation_mode"]),
            pca_grouping_mode=str(row["pca_grouping_mode"]),
        )
        try:
            resolved = resolve_experiment(registries, harness_config.experiment_id)
            outputs = compute_run_outputs(
                resolved=resolved,
                registries=registries,
                project_root=ROOT,
                harness_config=harness_config,
            )
            scores = score_panel_outputs(harness_config.panel_name, outputs)
            run_dir = write_run_artifacts_wrapper(sprint_paths, harness_config, outputs, scores)
            results_rows.append(
                {
                    **row.to_dict(),
                    "run_id": harness_config.run_id,
                    **scores,
                    "status": "completed",
                    "run_dir": str(run_dir),
                    "error_message": "",
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
                    "run_dir": "",
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )

    results_df = pd.DataFrame(results_rows)
    completed_df = results_df[results_df["status"] == "completed"].copy()
    summary_tables = save_summary_tables(sprint_paths, search_df, results_df)
    figure_paths = []
    figure_paths.extend(plot_leaderboards(completed_df, sprint_paths).values())
    figure_paths.append(plot_factor_ablation(completed_df, sprint_paths))
    figure_paths.append(plot_best_vs_baseline(completed_df, sprint_paths, "cspp_metabolite_spike_validation", "best_vs_baseline_cspp.png"))
    figure_paths.append(plot_best_vs_baseline(completed_df, sprint_paths, "serum_ag_uricase_validation", "best_vs_baseline_uricase.png"))
    report_md = build_markdown_report(sprint_paths, completed_df, search_df, summary_tables["best"])
    build_pdf_report(report_md, figure_paths, sprint_paths.report_dir / "GAIRA_autoresearch_v1_report.pdf")
    print(f"Wrote autoresearch sprint outputs under {sprint_paths.sprint_root}")


def write_run_artifacts_wrapper(sprint_paths, harness_config, outputs, scores):
    from gaira.demo.autoresearch_utils import write_run_artifacts

    return write_run_artifacts(sprint_paths, harness_config, outputs, scores)


if __name__ == "__main__":
    main()
