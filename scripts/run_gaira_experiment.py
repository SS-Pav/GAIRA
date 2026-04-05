from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gaira.demo.gaira_experiment_runner_utils import (
    build_source_role_sets,
    load_architecture_registries,
    load_grounding_family_dataframe,
    load_query_dataframe,
    resolve_experiment,
    retrieval_hit_summary,
    write_run_note,
    write_run_snapshot,
)
from gaira.demo.raw_bsv_pilot_utils import (
    ALL_AXES,
    apply_source_role_policy,
    build_bsv_profiles,
    build_differential_query_df,
    delta_bsv,
    delta_from_residual_group_means,
    group_mean_bsv,
    load_ontology_rules,
    map_references_to_axes,
    resolve_reference_group,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE1_DIR = ROOT / "reports" / "gaira_phase1_registry_audit_v2"
DEFAULT_ARCHITECTURE_DIR = ROOT / "reports" / "gaira_architecture_scaffold_v1"
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "gaira_experiment_runner_v1"
DEFAULT_ONTOLOGY_RULES = ROOT / "config" / "phase2_bsv_ontology_rules_v2.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a GAIRAv3 registry-driven experiment.")
    parser.add_argument("--experiment-id", required=True, help="Experiment id from first_pass_experiment_plan.csv")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ontology-rules-path", type=Path, default=DEFAULT_ONTOLOGY_RULES)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--normalization-mode",
        choices=["raw_support", "per_spectrum_sum", "softmax_then_sum", "delta_zscore_placeholder"],
        default="per_spectrum_sum",
    )
    parser.add_argument(
        "--grounding-family-registry-path",
        type=Path,
        default=ROOT / "config" / "gaira_grounding_family_registry_v1.csv",
    )
    parser.add_argument(
        "--target-family-registry-path",
        type=Path,
        default=ROOT / "config" / "gaira_target_family_registry_v1.csv",
    )
    parser.add_argument(
        "--inference-lane-registry-path",
        type=Path,
        default=ROOT / "config" / "gaira_inference_lane_registry_v1.csv",
    )
    parser.add_argument(
        "--representation-mode-registry-path",
        type=Path,
        default=ROOT / "config" / "gaira_representation_mode_registry_v1.csv",
    )
    parser.add_argument(
        "--dataset-experiment-registry-path",
        type=Path,
        default=ROOT / "config" / "gaira_dataset_experiment_registry_v1.csv",
    )
    parser.add_argument(
        "--experiment-plan-path",
        type=Path,
        default=DEFAULT_ARCHITECTURE_DIR / "first_pass_experiment_plan.csv",
    )
    parser.add_argument(
        "--phase1-registry-path",
        type=Path,
        default=DEFAULT_PHASE1_DIR / "phase1_dataset_registry_v2.csv",
    )
    parser.add_argument(
        "--phase1-grounding-map-path",
        type=Path,
        default=DEFAULT_PHASE1_DIR / "phase1_target_grounding_map_v2.csv",
    )
    parser.add_argument(
        "--phase1-exclusions-path",
        type=Path,
        default=DEFAULT_PHASE1_DIR / "phase1_grounding_exclusions.csv",
    )
    parser.add_argument("--pca-components", type=int, default=3)
    return parser.parse_args()


def _run_bsv_for_query(
    *,
    query_df: pd.DataFrame,
    grounding_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    top_k: int,
    normalization_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_spectrum_df, retrieval_df = build_bsv_profiles(
        query_df,
        grounding_df,
        mapping_df,
        top_k=top_k,
        normalization_mode=normalization_mode,
    )
    group_means_df = group_mean_bsv(per_spectrum_df)
    return per_spectrum_df, group_means_df, retrieval_df


def _comparison_table(
    local_delta_df: pd.DataFrame,
    mean_delta_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for mode_name, df in [
        ("raw_direct_mean_background", mean_delta_df),
        ("local_structural_embedding_nearest_background", local_delta_df),
    ]:
        for _, row in df.iterrows():
            record = {"mode": mode_name, "comparison": str(row["comparison"]), "group_label": str(row["group_label"])}
            for axis in ALL_AXES + ["unmapped_support"]:
                if axis in row.index:
                    record[axis] = float(row[axis])
            rows.append(record)
    return pd.DataFrame(rows)


def _write_local_mode_note(
    output_path: Path,
    *,
    resolved,
    available_sources: list[str],
    unavailable_sources: list[str],
    local_delta_df: pd.DataFrame,
    mean_delta_df: pd.DataFrame,
    pair_df: pd.DataFrame,
) -> None:
    local_matrix = float(local_delta_df["matrix_background"].abs().mean()) if not local_delta_df.empty else float("nan")
    mean_matrix = float(mean_delta_df["matrix_background"].abs().mean()) if not mean_delta_df.empty else float("nan")
    local_metabolite = float(local_delta_df["small_molecule_metabolite"].mean()) if not local_delta_df.empty else float("nan")
    mean_metabolite = float(mean_delta_df["small_molecule_metabolite"].mean()) if not mean_delta_df.empty else float("nan")
    mean_dist = float(pair_df["pca_distance"].mean()) if "pca_distance" in pair_df.columns and not pair_df.empty else float("nan")
    improved_collapse = local_matrix < mean_matrix
    lines = [
        f"# GAIRA Experiment Run: {resolved.experiment_row['experiment_id']}",
        "",
        f"- question: {resolved.experiment_row['question_being_asked']}",
        f"- target family: `{resolved.experiment_row['target_family']}`",
        f"- inference lane: `{resolved.experiment_row['inference_lane']}`",
        f"- representation mode: `{resolved.experiment_row['representation_mode']}`",
        f"- grounding families used: `{'; '.join(resolved.grounding_family_names)}`",
        f"- baseline policy: `{resolved.experiment_row['baseline_policy']}`",
        f"- dataset/subset alias: `{resolved.subset_alias}`",
        f"- available grounding sources: `{'; '.join(available_sources)}`",
        f"- unavailable grounding sources: `{'; '.join(unavailable_sources) if unavailable_sources else 'none'}`",
        f"- matched pair mean PCA distance: `{mean_dist:.4f}`",
        "",
        "## Comparison summary",
        f"- raw_direct + mean_background mean |matrix_background delta|: `{mean_matrix:.4f}`",
        f"- local_structural_embedding + nearest_background mean |matrix_background delta|: `{local_matrix:.4f}`",
        f"- raw_direct + mean_background mean small_molecule_metabolite delta: `{mean_metabolite:.4f}`",
        f"- local_structural_embedding + nearest_background mean small_molecule_metabolite delta: `{local_metabolite:.4f}`",
        "",
        "## Interpretation",
        f"- Did local structure improve baseline matching? `{'yes' if pair_df['matched_background_sample_key'].ne('__mean_background__').all() else 'partially'}`",
        f"- Did dBSV become more chemically interpretable? `{'possibly' if local_metabolite > mean_metabolite else 'no clear gain'}`",
        f"- Did local structure reduce background collapse? `{'yes' if improved_collapse else 'no'}`",
        f"- Is this representation mode worth keeping? `{'yes, as a first-class baseline-selection mode' if improved_collapse or local_metabolite > mean_metabolite else 'yes, but only as an experimental scaffold for now'}`",
        "",
        "## What remains weak",
        "- PCA neighborhoods are only used for baseline matching. They are not biochemical attribution.",
        "- This first pass uses simple Euclidean nearest neighbors in panel-local PCA space and a panel-specific treated/background pairing rule.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
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
    resolved = resolve_experiment(registries, args.experiment_id)
    lane = str(resolved.experiment_row["inference_lane"])
    representation_mode = str(resolved.experiment_row["representation_mode"])
    if lane not in {"absolute_bsv", "differential_bsv"}:
        raise NotImplementedError(f"Inference lane {lane} is not implemented in this runner.")
    if representation_mode not in {"raw_direct", "local_structural_embedding"}:
        raise NotImplementedError(f"Representation mode {representation_mode} is not implemented in this runner.")

    output_dir = args.output_root / args.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    query_df = load_query_dataframe(resolved.dataset_row)
    grounding_df, family_to_sources, unavailable_sources = load_grounding_family_dataframe(resolved, registries)
    primary_sources, caveat_only_sources = build_source_role_sets(resolved, family_to_sources)
    available_sources = sorted(grounding_df["source_key"].astype(str).unique().tolist())

    ontology_rules = load_ontology_rules(args.ontology_rules_path)
    raw_mapping_df = map_references_to_axes(grounding_df, ontology_rules)
    mapping_df = apply_source_role_policy(
        raw_mapping_df,
        grounding_df,
        primary_sources=primary_sources,
        caveat_only_sources=caveat_only_sources,
    )
    if representation_mode == "raw_direct":
        per_spectrum_df, group_means_df, retrieval_df = _run_bsv_for_query(
            query_df=query_df,
            grounding_df=grounding_df,
            mapping_df=mapping_df,
            top_k=args.top_k,
            normalization_mode=args.normalization_mode,
        )
        retrieval_summary_df = retrieval_hit_summary(retrieval_df)

        reference_group = None
        if lane == "differential_bsv":
            reference_group = resolve_reference_group(query_df)
            delta_df = delta_bsv(group_means_df, reference_group=reference_group)
            delta_df.to_csv(output_dir / "delta_bsv.csv", index=False)

        per_spectrum_df.to_csv(output_dir / "per_spectrum_bsv.csv", index=False)
        group_means_df.to_csv(output_dir / "group_mean_bsv.csv", index=False)
        retrieval_df.to_csv(output_dir / "topk_retrieval_hits.csv", index=False)
        retrieval_summary_df.to_csv(output_dir / "retrieval_hit_summary.csv", index=False)
        mapping_df.to_csv(output_dir / "ontology_mapping_applied.csv", index=False)
        ontology_rules.to_csv(output_dir / "ontology_rules_used.csv", index=False)
        pd.DataFrame(
            {
                "axis_name": ALL_AXES,
                "present_in_output": [axis in per_spectrum_df.columns for axis in ALL_AXES],
            }
        ).to_csv(output_dir / "axis_inventory.csv", index=False)
        write_run_snapshot(
            output_dir / "run_config.json",
            resolved=resolved,
            available_sources=available_sources,
            unavailable_sources=unavailable_sources,
            primary_sources=primary_sources,
            caveat_only_sources=caveat_only_sources,
            normalization_mode=args.normalization_mode,
            top_k=args.top_k,
            reference_group=reference_group,
            ontology_rules_path=args.ontology_rules_path,
        )
        write_run_note(
            output_dir / "run_note.md",
            resolved=resolved,
            available_sources=available_sources,
            unavailable_sources=unavailable_sources,
            reference_group=reference_group,
        )
    else:
        residual_mean_df, mean_pair_df, _ = build_differential_query_df(
            query_df,
            baseline_policy="mean_background",
            n_pca_components=args.pca_components,
        )
        mean_per_df, mean_group_df, mean_retrieval_df = _run_bsv_for_query(
            query_df=residual_mean_df,
            grounding_df=grounding_df,
            mapping_df=mapping_df,
            top_k=args.top_k,
            normalization_mode=args.normalization_mode,
        )
        mean_delta_df = delta_from_residual_group_means(mean_group_df)

        residual_local_df, local_pair_df, pca_df = build_differential_query_df(
            query_df,
            baseline_policy="nearest_background",
            n_pca_components=args.pca_components,
        )
        local_per_df, local_group_df, local_retrieval_df = _run_bsv_for_query(
            query_df=residual_local_df,
            grounding_df=grounding_df,
            mapping_df=mapping_df,
            top_k=args.top_k,
            normalization_mode=args.normalization_mode,
        )
        local_delta_df = delta_from_residual_group_means(local_group_df)
        comparison_df = _comparison_table(local_delta_df, mean_delta_df)

        local_per_df.to_csv(output_dir / "per_spectrum_bsv.csv", index=False)
        local_group_df.to_csv(output_dir / "group_mean_bsv.csv", index=False)
        local_delta_df.to_csv(output_dir / "delta_bsv.csv", index=False)
        retrieval_hit_summary(local_retrieval_df).to_csv(output_dir / "retrieval_hit_summary.csv", index=False)
        local_retrieval_df.to_csv(output_dir / "topk_retrieval_hits.csv", index=False)
        local_pair_df.to_csv(output_dir / "matched_background_pairs.csv", index=False)
        pca_df.to_csv(output_dir / "local_pca_coordinates.csv", index=False)
        comparison_df.to_csv(output_dir / "representation_mode_comparison.csv", index=False)

        mean_per_df.to_csv(output_dir / "per_spectrum_bsv__raw_direct_mean_background.csv", index=False)
        mean_group_df.to_csv(output_dir / "group_mean_bsv__raw_direct_mean_background.csv", index=False)
        mean_delta_df.to_csv(output_dir / "delta_bsv__raw_direct_mean_background.csv", index=False)
        retrieval_hit_summary(mean_retrieval_df).to_csv(output_dir / "retrieval_hit_summary__raw_direct_mean_background.csv", index=False)
        mean_retrieval_df.to_csv(output_dir / "topk_retrieval_hits__raw_direct_mean_background.csv", index=False)
        mean_pair_df.to_csv(output_dir / "matched_background_pairs__raw_direct_mean_background.csv", index=False)

        local_per_df.to_csv(output_dir / "per_spectrum_bsv__local_structural_embedding.csv", index=False)
        local_group_df.to_csv(output_dir / "group_mean_bsv__local_structural_embedding.csv", index=False)
        local_delta_df.to_csv(output_dir / "delta_bsv__local_structural_embedding.csv", index=False)
        retrieval_hit_summary(local_retrieval_df).to_csv(output_dir / "retrieval_hit_summary__local_structural_embedding.csv", index=False)
        local_retrieval_df.to_csv(output_dir / "topk_retrieval_hits__local_structural_embedding.csv", index=False)

        mapping_df.to_csv(output_dir / "ontology_mapping_applied.csv", index=False)
        ontology_rules.to_csv(output_dir / "ontology_rules_used.csv", index=False)
        pd.DataFrame(
            {
                "axis_name": ALL_AXES,
                "present_in_output": [axis in local_per_df.columns for axis in ALL_AXES],
            }
        ).to_csv(output_dir / "axis_inventory.csv", index=False)
        write_run_snapshot(
            output_dir / "run_config.json",
            resolved=resolved,
            available_sources=available_sources,
            unavailable_sources=unavailable_sources,
            primary_sources=primary_sources,
            caveat_only_sources=caveat_only_sources,
            normalization_mode=args.normalization_mode,
            top_k=args.top_k,
            reference_group="matched_background_zero",
            ontology_rules_path=args.ontology_rules_path,
        )
        _write_local_mode_note(
            output_dir / "run_note.md",
            resolved=resolved,
            available_sources=available_sources,
            unavailable_sources=unavailable_sources,
            local_delta_df=local_delta_df,
            mean_delta_df=mean_delta_df,
            pair_df=local_pair_df,
        )

    pd.DataFrame(
        {
            "resolved_grounding_family": list(family_to_sources.keys()),
            "source_keys": ["; ".join(family_to_sources[key]) for key in family_to_sources],
        }
    ).to_csv(output_dir / "grounding_family_resolution.csv", index=False)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
