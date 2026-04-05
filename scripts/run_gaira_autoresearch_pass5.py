from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import traceback

import pandas as pd

from gaira.autoresearch_storage import DEFAULT_STORAGE_CONFIG_PATH, initialize_autoresearch_sprint, load_autoresearch_storage_config
from gaira.demo.autoresearch_utils import comparator_map_for_alias, compute_delta_by_mapping, score_panel_outputs
from gaira.demo.autoresearch_pass5_utils import (
    Pass5HarnessConfig,
    apply_pass5_filter_mode,
    build_bsv_profiles_pass5,
    build_pass5_markdown_report,
    build_pass5_search_space,
    build_pdf_report,
    compute_mixture_metrics,
    plot_pass5_figures,
    save_pass5_tables,
    write_pass5_run_artifacts,
)
from gaira.demo.gaira_experiment_runner_utils import (
    ResolvedExperiment,
    build_source_role_sets,
    load_architecture_registries,
    load_grounding_family_dataframe,
    load_query_dataframe,
    resolve_experiment,
    retrieval_hit_summary,
)
from gaira.demo.raw_bsv_pilot_utils import (
    apply_source_role_policy,
    build_group_mean_query_df,
    group_mean_bsv,
    load_ontology_rules,
    map_references_to_axes,
)


ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = ROOT / "reports" / "gaira_architecture_scaffold_v2"
PHASE1_DIR = ROOT / "reports" / "gaira_phase1_registry_audit_v2"
BASELINE_PROBE1_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot1b_small2023_mixture_probe1_fingerprint"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GAIRAv3 autoresearch v1 pass 5 saturation-fix harness.")
    parser.add_argument("--storage-config-path", type=Path, default=DEFAULT_STORAGE_CONFIG_PATH)
    parser.add_argument("--sprint-subdir", type=str, default="pass5_saturation_fix")
    parser.add_argument("--grounding-family-registry-path", type=Path, default=ROOT / "config" / "gaira_grounding_family_registry_v1.csv")
    parser.add_argument("--target-family-registry-path", type=Path, default=ROOT / "config" / "gaira_target_family_registry_v1.csv")
    parser.add_argument("--inference-lane-registry-path", type=Path, default=ROOT / "config" / "gaira_inference_lane_registry_v2.csv")
    parser.add_argument("--representation-mode-registry-path", type=Path, default=ROOT / "config" / "gaira_representation_mode_registry_v2.csv")
    parser.add_argument("--dataset-experiment-registry-path", type=Path, default=ROOT / "config" / "gaira_dataset_experiment_registry_v2.csv")
    parser.add_argument("--experiment-plan-path", type=Path, default=ARCH_DIR / "first_pass_experiment_plan.csv")
    parser.add_argument("--phase1-registry-path", type=Path, default=PHASE1_DIR / "phase1_dataset_registry_v2.csv")
    parser.add_argument("--phase1-grounding-map-path", type=Path, default=PHASE1_DIR / "phase1_target_grounding_map_v2.csv")
    parser.add_argument("--phase1-exclusions-path", type=Path, default=PHASE1_DIR / "phase1_grounding_exclusions.csv")
    return parser.parse_args()


def _load_dataset_row_by_alias(registries, subset_alias: str) -> pd.Series:
    matches = registries.dataset_experiments[registries.dataset_experiments["subset_alias"].astype(str) == subset_alias].copy()
    if matches.empty or len(matches) > 1:
        raise RuntimeError(f"Could not resolve unique subset alias {subset_alias}")
    return matches.iloc[0]


def _resolve_panel(registries, panel_name: str):
    if panel_name == "cspp_metabolite_spike_validation":
        return resolve_experiment(registries, "exp_diff_cspp_metabolite_spike")
    if panel_name == "serum_ag_uricase_validation":
        return resolve_experiment(registries, "exp_localdiff_serum_uricase")
    if panel_name == "small2023_mixture_probe1":
        dataset_row = _load_dataset_row_by_alias(registries, "small2023_mixture_probe1")
        return ResolvedExperiment(
            experiment_row=dataset_row,
            dataset_row=dataset_row,
            subset_alias="small2023_mixture_probe1",
            grounding_family_names=["universal_biochemical_grounding"],
        )
    if panel_name == "small2023_mixture_probe2":
        dataset_row = _load_dataset_row_by_alias(registries, "small2023_mixture_probe2")
        return ResolvedExperiment(
            experiment_row=dataset_row,
            dataset_row=dataset_row,
            subset_alias="small2023_mixture_probe2",
            grounding_family_names=["universal_biochemical_grounding"],
        )
    raise ValueError(f"Unsupported panel_name: {panel_name}")


def compute_pass5_panel_outputs(*, resolved, registries, harness_config: Pass5HarnessConfig, ontology_path: Path) -> dict[str, object]:
    query_df = load_query_dataframe(resolved.dataset_row)
    original_grounding_names = resolved.grounding_family_names
    object.__setattr__(resolved, "grounding_family_names", ["universal_biochemical_grounding"])
    try:
        grounding_df, family_to_sources, unavailable_sources = load_grounding_family_dataframe(resolved, registries)
    finally:
        object.__setattr__(resolved, "grounding_family_names", original_grounding_names)

    grounding_df = apply_pass5_filter_mode(grounding_df, harness_config.universal_grounding_filter_mode)
    primary_sources, caveat_only_sources = build_source_role_sets(resolved, family_to_sources)
    available_source_keys = set(grounding_df["source_key"].astype(str))
    primary_sources = {key for key in primary_sources if key in available_source_keys}
    caveat_only_sources = {key for key in caveat_only_sources if key in available_source_keys}
    ontology_rules = load_ontology_rules(ontology_path)
    raw_mapping_df = map_references_to_axes(grounding_df, ontology_rules)
    mapping_df = apply_source_role_policy(
        raw_mapping_df,
        grounding_df,
        primary_sources=primary_sources,
        caveat_only_sources=caveat_only_sources,
    )

    query_input_df = build_group_mean_query_df(query_df, group_col="class_label")
    per_spectrum_df, retrieval_df = build_bsv_profiles_pass5(
        query_input_df,
        grounding_df,
        mapping_df,
        top_k=harness_config.top_k,
        similarity_metric="cosine",
        weighting_mode=harness_config.weighting_mode,
        weighting_param=harness_config.weighting_param,
        diversity_mode=harness_config.diversity_mode,
        family_min_coverage=harness_config.family_min_coverage,
    )
    group_means_df = group_mean_bsv(per_spectrum_df, group_col="class_label")
    retrieval_summary_df = retrieval_hit_summary(retrieval_df)
    delta_df = None
    if resolved.subset_alias in {"cspp_metabolite_spike_validation", "serum_ag_uricase_validation"}:
        comparator_map = comparator_map_for_alias(resolved.subset_alias, group_means_df)
        delta_df = compute_delta_by_mapping(group_means_df, comparator_map)
    return {
        "query_df": query_df,
        "grounding_df": grounding_df,
        "mapping_df": mapping_df,
        "per_spectrum_df": per_spectrum_df,
        "group_means_df": group_means_df,
        "delta_df": delta_df if delta_df is not None else pd.DataFrame(),
        "retrieval_df": retrieval_df,
        "retrieval_summary_df": retrieval_summary_df,
        "available_sources": sorted(grounding_df["source_key"].astype(str).unique().tolist()),
        "unavailable_sources": unavailable_sources,
        "primary_sources": sorted(primary_sources),
        "caveat_only_sources": sorted(caveat_only_sources),
    }


def _baseline_probe1_metrics() -> pd.Series:
    class_mean = pd.read_csv(BASELINE_PROBE1_ROOT / "tables" / "class_mean_bsv.csv")
    progression = pd.read_csv(BASELINE_PROBE1_ROOT / "tables" / "mixture_progression_summary.csv")
    entropy = pd.read_csv(BASELINE_PROBE1_ROOT / "tables" / "class_neighborhood_entropy.csv")
    dominance = pd.read_csv(BASELINE_PROBE1_ROOT / "tables" / "class_top1_dominance.csv")
    axis_entropy = pd.read_csv(BASELINE_PROBE1_ROOT / "tables" / "class_axis_entropy.csv")
    order_numeric = pd.Series([int(str(x).replace("c", "")) for x in class_mean["class_label"]], dtype=float)
    small_spearman = float(order_numeric.corr(class_mean["small_molecule_metabolite"], method="spearman"))
    nucleic_inverse_spearman = float(-order_numeric.corr(class_mean["nucleic_acid"], method="spearman"))
    noncollapse_ratio = float(
        class_mean.drop(columns=["sample_key", "dataset_id", "subset_id", "class_label", "unmapped_support"])
        .round(8)
        .drop_duplicates()
        .shape[0]
        / len(class_mean)
    )
    adjacent_nonzero_ratio = float((progression["toward_high_endpoint_score"].diff().abs().fillna(0.0) > 1e-8).sum() / max(len(progression) - 1, 1))
    mean_top1 = float(dominance["top1_fraction"].mean())
    mean_neighborhood_entropy = float(entropy["neighborhood_entropy"].mean())
    mean_axis_entropy = float(axis_entropy["axis_entropy"].mean())
    diversity_score = 0.7 * (
        mean_neighborhood_entropy / max(1.0, math.log2(max(int(len(entropy)), 2)))
    ) + 0.3 * (
        mean_axis_entropy / max(1.0, math.log2(8))
    )
    mixture_progression_score = 0.9 * adjacent_nonzero_ratio + 0.9 * noncollapse_ratio + 0.7 * max(small_spearman, 0.0) + 0.5 * max(nucleic_inverse_spearman, 0.0)
    saturation_penalty = max(mean_top1 - 0.85, 0.0) * 2.0 + (1.0 - noncollapse_ratio) + (1.0 - adjacent_nonzero_ratio)
    return pd.Series(
        {
            "config_id": "locked_baseline_probe1",
            "mean_top1_dominance": mean_top1,
            "mean_neighborhood_entropy": mean_neighborhood_entropy,
            "mean_axis_entropy": mean_axis_entropy,
            "adjacent_nonzero_ratio": adjacent_nonzero_ratio,
            "noncollapse_ratio": noncollapse_ratio,
            "small_molecule_spearman": small_spearman,
            "nucleic_inverse_spearman": nucleic_inverse_spearman,
            "mixture_progression_score": mixture_progression_score,
            "diversity_score": diversity_score,
            "saturation_penalty": saturation_penalty,
            "progression_df": progression,
        }
    )


def main() -> None:
    args = parse_args()
    storage_cfg = load_autoresearch_storage_config(args.storage_config_path)
    sprint_paths = initialize_autoresearch_sprint(args.storage_config_path, sprint_id=f"{storage_cfg.sprint_id}/{args.sprint_subdir}")
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
    ontology_path = ROOT / "config" / "phase2_bsv_ontology_rules_v2.csv"
    search_df = build_pass5_search_space()

    panel_names = [
        "cspp_metabolite_spike_validation",
        "serum_ag_uricase_validation",
        "small2023_mixture_probe1",
    ]
    results_rows = []
    panel_rows = []
    mixture_metric_rows = []
    entropy_rows = []
    dominance_rows = []
    best_progression_df = None

    for _, row in search_df.iterrows():
        cfg = Pass5HarnessConfig(
            config_id=str(row["config_id"]),
            universal_grounding_filter_mode=str(row["universal_grounding_filter_mode"]),
            top_k=int(row["top_k"]),
            weighting_mode=str(row["weighting_mode"]),
            weighting_param=None if pd.isna(row["weighting_param"]) else float(row["weighting_param"]),
            diversity_mode=str(row["diversity_mode"]),
            family_min_coverage=int(row["family_min_coverage"]),
        )
        panel_scores = {}
        mixture_metrics_payload = None
        try:
            for panel_name in panel_names:
                resolved = _resolve_panel(registries, panel_name)
                outputs = compute_pass5_panel_outputs(
                    resolved=resolved,
                    registries=registries,
                    harness_config=cfg,
                    ontology_path=ontology_path,
                )
                write_pass5_run_artifacts(sprint_paths, cfg, panel_name, outputs)
                if panel_name in {"cspp_metabolite_spike_validation", "serum_ag_uricase_validation"}:
                    scores = score_panel_outputs(panel_name, outputs)
                    panel_scores[panel_name] = float(scores["overall_score"])
                    panel_rows.append(
                        {
                            "config_id": cfg.config_id,
                            "panel_name": panel_name,
                            "panel_score": float(scores["overall_score"]),
                            "expected_axis_uplift_score": float(scores["expected_axis_uplift_score"]),
                            "top_hit_plausibility_score": float(scores["top_hit_plausibility_score"]),
                            "matrix_collapse_penalty": float(scores["matrix_collapse_penalty"]),
                            "caveat_domination_penalty": float(scores["caveat_domination_penalty"]),
                        }
                    )
                else:
                    mixture_metrics = compute_mixture_metrics(outputs["group_means_df"], outputs["retrieval_df"])
                    mixture_metrics_payload = mixture_metrics
                    panel_scores[panel_name] = float(
                        mixture_metrics["mixture_progression_score"]
                        + mixture_metrics["diversity_score"]
                        - mixture_metrics["saturation_penalty"]
                    )
                    panel_rows.append(
                        {
                            "config_id": cfg.config_id,
                            "panel_name": panel_name,
                            "panel_score": panel_scores[panel_name],
                            "expected_axis_uplift_score": None,
                            "top_hit_plausibility_score": None,
                            "matrix_collapse_penalty": None,
                            "caveat_domination_penalty": None,
                        }
                    )
            validation_score = panel_scores["cspp_metabolite_spike_validation"] + panel_scores["serum_ag_uricase_validation"]
            mixture_panel_score = panel_scores["small2023_mixture_probe1"]
            overall_score = (
                validation_score
                + float(mixture_metrics_payload["mixture_progression_score"])
                + float(mixture_metrics_payload["diversity_score"])
                - float(mixture_metrics_payload["saturation_penalty"])
            )
            results_rows.append(
                {
                    **row.to_dict(),
                    "validation_score": validation_score,
                    "mixture_panel_score": mixture_panel_score,
                    "mixture_progression_score": float(mixture_metrics_payload["mixture_progression_score"]),
                    "diversity_score": float(mixture_metrics_payload["diversity_score"]),
                    "saturation_penalty": float(mixture_metrics_payload["saturation_penalty"]),
                    "mean_neighborhood_entropy": float(mixture_metrics_payload["mean_neighborhood_entropy"]),
                    "mean_axis_entropy": float(mixture_metrics_payload["mean_axis_entropy"]),
                    "mean_top1_dominance": float(mixture_metrics_payload["mean_top1_dominance"]),
                    "adjacent_nonzero_ratio": float(mixture_metrics_payload["adjacent_nonzero_ratio"]),
                    "noncollapse_ratio": float(mixture_metrics_payload["noncollapse_ratio"]),
                    "small_molecule_spearman": float(mixture_metrics_payload["small_molecule_spearman"]),
                    "nucleic_inverse_spearman": float(mixture_metrics_payload["nucleic_inverse_spearman"]),
                    "overall_score": overall_score,
                    "status": "completed",
                    "error_message": "",
                }
            )
            mixture_metric_rows.append(
                {
                    "config_id": cfg.config_id,
                    "mixture_progression_score": float(mixture_metrics_payload["mixture_progression_score"]),
                    "adjacent_nonzero_ratio": float(mixture_metrics_payload["adjacent_nonzero_ratio"]),
                    "noncollapse_ratio": float(mixture_metrics_payload["noncollapse_ratio"]),
                    "small_molecule_spearman": float(mixture_metrics_payload["small_molecule_spearman"]),
                    "nucleic_inverse_spearman": float(mixture_metrics_payload["nucleic_inverse_spearman"]),
                }
            )
            entropy_rows.append(
                {
                    "config_id": cfg.config_id,
                    "mean_neighborhood_entropy": float(mixture_metrics_payload["mean_neighborhood_entropy"]),
                    "mean_axis_entropy": float(mixture_metrics_payload["mean_axis_entropy"]),
                    "diversity_score": float(mixture_metrics_payload["diversity_score"]),
                }
            )
            dominance_rows.append(
                {
                    "config_id": cfg.config_id,
                    "mean_top1_dominance": float(mixture_metrics_payload["mean_top1_dominance"]),
                    "saturation_penalty": float(mixture_metrics_payload["saturation_penalty"]),
                }
            )
            if best_progression_df is None or overall_score > max(r["overall_score"] for r in results_rows[:-1]) if len(results_rows) > 1 else True:
                best_progression_df = mixture_metrics_payload["progression_df"]
        except Exception as exc:
            results_rows.append(
                {
                    **row.to_dict(),
                    "validation_score": None,
                    "mixture_panel_score": None,
                    "mixture_progression_score": None,
                    "diversity_score": None,
                    "saturation_penalty": None,
                    "mean_neighborhood_entropy": None,
                    "mean_axis_entropy": None,
                    "mean_top1_dominance": None,
                    "adjacent_nonzero_ratio": None,
                    "noncollapse_ratio": None,
                    "small_molecule_spearman": None,
                    "nucleic_inverse_spearman": None,
                    "overall_score": None,
                    "status": "failed",
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )

    results_df = pd.DataFrame(results_rows).sort_values("overall_score", ascending=False, na_position="last").reset_index(drop=True)
    panel_df = pd.DataFrame(panel_rows)
    best_by_panel_rows = []
    for panel_name in panel_names:
        sub = panel_df[panel_df["panel_name"] == panel_name].sort_values("panel_score", ascending=False)
        best_by_panel_rows.append(sub.iloc[0].to_dict())
    best_by_panel_df = pd.DataFrame(best_by_panel_rows)
    mixture_metrics_df = pd.DataFrame(mixture_metric_rows).merge(
        results_df[["config_id", "overall_score", "validation_score", "mixture_panel_score"]],
        on="config_id",
        how="left",
    )
    entropy_df = pd.DataFrame(entropy_rows)
    dominance_df = pd.DataFrame(dominance_rows)
    save_pass5_tables(sprint_paths, search_df, results_df, best_by_panel_df, mixture_metrics_df, entropy_df, dominance_df)

    baseline_mask = (
        (results_df["config_id"] == "cfg02")
        | (
            (results_df["universal_grounding_filter_mode"] == "purine_focused_universal")
            & (results_df["top_k"] == 5)
            & (results_df["weighting_mode"] == "uniform_weighting")
            & (results_df["diversity_mode"] == "none")
        )
    )
    baseline_candidates = results_df[baseline_mask & (results_df["status"] == "completed")].copy()
    if baseline_candidates.empty:
        raise RuntimeError("Could not resolve the locked baseline row inside pass 5 results.")
    baseline_row = baseline_candidates.iloc[0]
    best_row = results_df.iloc[0]

    # Post-search holdout evaluation on mixture_probe2 only.
    holdout_metrics = None
    try:
        best_cfg = Pass5HarnessConfig(
            config_id=str(best_row["config_id"]),
            universal_grounding_filter_mode=str(best_row["universal_grounding_filter_mode"]),
            top_k=int(best_row["top_k"]),
            weighting_mode=str(best_row["weighting_mode"]),
            weighting_param=None if pd.isna(best_row["weighting_param"]) else float(best_row["weighting_param"]),
            diversity_mode=str(best_row["diversity_mode"]),
            family_min_coverage=int(best_row["family_min_coverage"]),
        )
        holdout_resolved = _resolve_panel(registries, "small2023_mixture_probe2")
        holdout_outputs = compute_pass5_panel_outputs(
            resolved=holdout_resolved,
            registries=registries,
            harness_config=best_cfg,
            ontology_path=ontology_path,
        )
        holdout_metrics = compute_mixture_metrics(holdout_outputs["group_means_df"], holdout_outputs["retrieval_df"])
        holdout_score = (
            holdout_metrics["mixture_progression_score"]
            + holdout_metrics["diversity_score"]
            - holdout_metrics["saturation_penalty"]
        )
        pd.DataFrame(
            [
                {
                    "config_id": best_cfg.config_id,
                    "mixture_panel_score": holdout_score,
                    "mean_top1_dominance": holdout_metrics["mean_top1_dominance"],
                    "mean_neighborhood_entropy": holdout_metrics["mean_neighborhood_entropy"],
                    "mean_axis_entropy": holdout_metrics["mean_axis_entropy"],
                    "adjacent_nonzero_ratio": holdout_metrics["adjacent_nonzero_ratio"],
                    "noncollapse_ratio": holdout_metrics["noncollapse_ratio"],
                }
            ]
        ).to_csv(sprint_paths.tables_dir / "holdout_probe2_best_config_metrics.csv", index=False)
    except Exception:
        holdout_metrics = None

    baseline_progression_df = pd.read_csv(BASELINE_PROBE1_ROOT / "tables" / "mixture_progression_summary.csv")
    figure_paths = plot_pass5_figures(
        sprint_paths,
        results_df[results_df["status"] == "completed"].copy(),
        mixture_metrics_df,
        entropy_df,
        dominance_df,
        baseline_progression_df,
        best_progression_df if best_progression_df is not None else baseline_progression_df,
    )
    report_md = build_pass5_markdown_report(
        sprint_paths,
        results_df[results_df["status"] == "completed"].copy(),
        best_by_panel_df,
        baseline_row,
        best_row,
        None if holdout_metrics is None else {
            "mixture_panel_score": float(
                holdout_metrics["mixture_progression_score"] + holdout_metrics["diversity_score"] - holdout_metrics["saturation_penalty"]
            ),
            "mean_top1_dominance": float(holdout_metrics["mean_top1_dominance"]),
            "noncollapse_ratio": float(holdout_metrics["noncollapse_ratio"]),
        },
    )
    build_pdf_report(report_md, figure_paths, sprint_paths.report_dir / "GAIRA_autoresearch_v1_pass5_report.pdf")
    print(f"Wrote pass 5 autoresearch outputs under {sprint_paths.sprint_root}")


if __name__ == "__main__":
    main()
