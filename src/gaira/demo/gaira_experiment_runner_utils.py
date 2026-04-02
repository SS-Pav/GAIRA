from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gaira.demo.raw_bsv_pilot_utils import (
    RegistryInputs,
    SourceSpec,
    _registry_source_string,
    apply_source_role_policy,
    derive_grounding_sources_for_alias,
    enforce_grounding_exclusions,
    get_registry_row_by_alias,
    load_biosample_subset,
    load_grounding_dataset,
    load_registry_inputs,
    parse_source_spec,
)


@dataclass
class ArchitectureRegistries:
    grounding_families: pd.DataFrame
    target_families: pd.DataFrame
    inference_lanes: pd.DataFrame
    representation_modes: pd.DataFrame
    dataset_experiments: pd.DataFrame
    experiment_plan: pd.DataFrame
    phase1_inputs: RegistryInputs


@dataclass
class ResolvedExperiment:
    experiment_row: pd.Series
    dataset_row: pd.Series
    subset_alias: str
    grounding_family_names: list[str]


def load_architecture_registries(
    *,
    grounding_family_registry_path: Path,
    target_family_registry_path: Path,
    inference_lane_registry_path: Path,
    representation_mode_registry_path: Path,
    dataset_experiment_registry_path: Path,
    experiment_plan_path: Path,
    phase1_registry_path: Path,
    phase1_grounding_map_path: Path,
    phase1_exclusions_path: Path,
) -> ArchitectureRegistries:
    paths = [
        grounding_family_registry_path,
        target_family_registry_path,
        inference_lane_registry_path,
        representation_mode_registry_path,
        dataset_experiment_registry_path,
        experiment_plan_path,
        phase1_registry_path,
        phase1_grounding_map_path,
        phase1_exclusions_path,
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required registry/config missing: {path}")
    return ArchitectureRegistries(
        grounding_families=pd.read_csv(grounding_family_registry_path),
        target_families=pd.read_csv(target_family_registry_path),
        inference_lanes=pd.read_csv(inference_lane_registry_path),
        representation_modes=pd.read_csv(representation_mode_registry_path),
        dataset_experiments=pd.read_csv(dataset_experiment_registry_path),
        experiment_plan=pd.read_csv(experiment_plan_path),
        phase1_inputs=load_registry_inputs(
            phase1_registry_path,
            phase1_grounding_map_path,
            phase1_exclusions_path,
        ),
    )


def resolve_experiment(registries: ArchitectureRegistries, experiment_id: str) -> ResolvedExperiment:
    matches = registries.experiment_plan[
        registries.experiment_plan["experiment_id"].astype(str) == str(experiment_id)
    ].copy()
    if matches.empty:
        raise KeyError(f"Unknown experiment_id: {experiment_id}")
    if len(matches) > 1:
        raise RuntimeError(f"Experiment id is not unique: {experiment_id}")
    experiment_row = matches.iloc[0]
    subset_alias = str(experiment_row["subset_alias"])
    dataset_row = get_registry_row_by_alias(registries.dataset_experiments, subset_alias)
    _validate_experiment_against_dataset_registry(experiment_row, dataset_row)
    grounding_family_names = _split_semicolon_list(experiment_row["grounding_families_used"])
    _validate_grounding_families(registries.grounding_families, grounding_family_names)
    return ResolvedExperiment(
        experiment_row=experiment_row,
        dataset_row=dataset_row,
        subset_alias=subset_alias,
        grounding_family_names=grounding_family_names,
    )


def _validate_experiment_against_dataset_registry(experiment_row: pd.Series, dataset_row: pd.Series) -> None:
    if str(experiment_row["target_family"]) != str(dataset_row["target_family"]):
        raise RuntimeError(
            f"Target family mismatch for {experiment_row['experiment_id']}: "
            f"plan={experiment_row['target_family']} registry={dataset_row['target_family']}"
        )
    inference_lane = str(experiment_row["inference_lane"])
    representation_mode = str(experiment_row["representation_mode"])
    allowed_lanes = _split_semicolon_list(dataset_row["allowed_inference_lanes"])
    allowed_modes = _split_semicolon_list(dataset_row["allowed_representation_modes"])
    allowed_grounding = _split_semicolon_list(dataset_row["allowed_grounding_families"])
    requested_grounding = _split_semicolon_list(experiment_row["grounding_families_used"])
    if inference_lane not in allowed_lanes:
        raise RuntimeError(f"Inference lane {inference_lane} is not allowed for {dataset_row['subset_alias']}")
    if representation_mode not in allowed_modes:
        raise RuntimeError(f"Representation mode {representation_mode} is not allowed for {dataset_row['subset_alias']}")
    disallowed = [family for family in requested_grounding if family not in allowed_grounding]
    if disallowed:
        raise RuntimeError(
            f"Grounding families not allowed for {dataset_row['subset_alias']}: {', '.join(disallowed)}"
        )


def _validate_grounding_families(grounding_family_df: pd.DataFrame, requested: list[str]) -> None:
    known = set(grounding_family_df["grounding_family"].astype(str))
    missing = [name for name in requested if name not in known]
    if missing:
        raise RuntimeError(f"Unknown grounding families requested: {', '.join(missing)}")


def _split_semicolon_list(value: object) -> list[str]:
    return [part.strip() for part in str(value).split(";") if part and part.strip()]


def load_query_dataframe(dataset_row: pd.Series) -> pd.DataFrame:
    return load_biosample_subset(str(dataset_row["dataset_id"]), str(dataset_row["subset_id"]))


def load_grounding_family_dataframe(
    resolved: ResolvedExperiment,
    registries: ArchitectureRegistries,
) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    allowed_sources = derive_grounding_sources_for_alias(resolved.subset_alias, registries.phase1_inputs)
    enforce_grounding_exclusions(resolved.subset_alias, allowed_sources, registries.phase1_inputs.exclusions)
    family_to_sources = _partition_sources_by_grounding_family(
        allowed_sources,
        registries.phase1_inputs.registry,
    )
    selected_sources = []
    for family_name in resolved.grounding_family_names:
        selected_sources.extend(family_to_sources.get(family_name, []))
    selected_sources = list(dict.fromkeys(selected_sources))
    if not selected_sources:
        raise RuntimeError(f"No grounding sources resolved for {resolved.experiment_row['experiment_id']}")

    frames = []
    available_sources = []
    unavailable_sources = []
    for source in selected_sources:
        spec = parse_source_spec(source, registries.phase1_inputs.registry)
        try:
            frames.append(_load_source_dataframe(spec))
            available_sources.append(spec.source_key)
        except RuntimeError:
            unavailable_sources.append(spec.source_key)
    if not frames:
        raise RuntimeError(f"No grounding sources available for {resolved.experiment_row['experiment_id']}")
    return pd.concat(frames, ignore_index=True), family_to_sources, unavailable_sources


def _partition_sources_by_grounding_family(
    allowed_sources: list[str],
    phase1_registry_df: pd.DataFrame,
) -> dict[str, list[str]]:
    by_family = {
        "universal_biochemical_grounding": [],
        "domain_specific_biochemical_grounding": [],
        "domain_specific_caveat_support_grounding": [],
    }
    for source in allowed_sources:
        spec = parse_source_spec(source, phase1_registry_df)
        source_role = _lookup_source_role(spec, phase1_registry_df)
        if source_role == "grounding_reference_universal_pure":
            by_family["universal_biochemical_grounding"].append(spec.source_key)
        elif source_role == "grounding_reference_serum_support":
            by_family["domain_specific_biochemical_grounding"].append(spec.source_key)
        elif source_role == "support_grounding_only_subset":
            by_family["domain_specific_caveat_support_grounding"].append(spec.source_key)
    return by_family


def _lookup_source_role(spec: SourceSpec, phase1_registry_df: pd.DataFrame) -> str:
    matches = phase1_registry_df[
        (phase1_registry_df["dataset_id"].astype(str) == spec.dataset_id)
        & (phase1_registry_df["subset_id"].astype(str) == spec.subset_id)
    ].copy()
    if matches.empty and spec.subset_id == "all":
        matches = phase1_registry_df[
            (phase1_registry_df["dataset_id"].astype(str) == spec.dataset_id)
            & (phase1_registry_df["subset_id"].astype(str) == "all")
        ].copy()
    if matches.empty:
        raise KeyError(f"Could not classify grounding source {spec.source_key}")
    return str(matches.iloc[0]["proposed_phase1_role"])


def _load_source_dataframe(spec: SourceSpec) -> pd.DataFrame:
    if spec.source_type == "grounding":
        df = load_grounding_dataset(spec.dataset_id)
    else:
        df = load_biosample_subset(spec.dataset_id, spec.subset_id)
        df["compound_label"] = df["class_label"].astype(str)
        df["grounding_role"] = "biosample_support_subset"
        df["experiment_family"] = spec.source_key
    df["source_key"] = spec.source_key
    return df


def build_source_role_sets(
    resolved: ResolvedExperiment,
    family_to_sources: dict[str, list[str]],
) -> tuple[set[str], set[str]]:
    primary = set()
    caveat_only = set()
    for family_name in resolved.grounding_family_names:
        if family_name == "universal_biochemical_grounding":
            primary.update(family_to_sources.get(family_name, []))
        elif family_name == "domain_specific_biochemical_grounding":
            primary.update(family_to_sources.get(family_name, []))
        elif family_name == "domain_specific_caveat_support_grounding":
            caveat_only.update(family_to_sources.get(family_name, []))
    return primary, caveat_only


def retrieval_hit_summary(retrieval_df: pd.DataFrame) -> pd.DataFrame:
    if retrieval_df.empty:
        return pd.DataFrame(
            columns=[
                "query_class_label",
                "reference_source_key",
                "reference_dataset_id",
                "reference_compound_label",
                "retrieval_count",
                "total_support_weight",
                "mean_similarity",
            ]
        )
    return (
        retrieval_df.groupby(
            [
                "query_class_label",
                "reference_source_key",
                "reference_dataset_id",
                "reference_compound_label",
            ],
            sort=False,
        )
        .agg(
            retrieval_count=("reference_sample_key", "size"),
            total_support_weight=("support_weight", "sum"),
            mean_similarity=("similarity", "mean"),
        )
        .reset_index()
        .sort_values(["query_class_label", "total_support_weight", "retrieval_count"], ascending=[True, False, False])
        .reset_index(drop=True)
    )


def write_run_snapshot(
    output_path: Path,
    *,
    resolved: ResolvedExperiment,
    available_sources: list[str],
    unavailable_sources: list[str],
    primary_sources: set[str],
    caveat_only_sources: set[str],
    normalization_mode: str,
    top_k: int,
    reference_group: str | None,
    ontology_rules_path: Path,
) -> None:
    payload = {
        "experiment_id": str(resolved.experiment_row["experiment_id"]),
        "dataset_id": str(resolved.dataset_row["dataset_id"]),
        "subset_id": str(resolved.dataset_row["subset_id"]),
        "subset_alias": resolved.subset_alias,
        "target_family": str(resolved.experiment_row["target_family"]),
        "inference_lane": str(resolved.experiment_row["inference_lane"]),
        "representation_mode": str(resolved.experiment_row["representation_mode"]),
        "grounding_families_used": resolved.grounding_family_names,
        "baseline_policy": str(resolved.experiment_row["baseline_policy"]),
        "normalization_mode": normalization_mode,
        "top_k": int(top_k),
        "reference_group": reference_group,
        "ontology_rules_path": str(ontology_rules_path),
        "available_grounding_sources": available_sources,
        "unavailable_grounding_sources": unavailable_sources,
        "primary_sources": sorted(primary_sources),
        "caveat_only_sources": sorted(caveat_only_sources),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def write_run_note(
    output_path: Path,
    *,
    resolved: ResolvedExperiment,
    available_sources: list[str],
    unavailable_sources: list[str],
    reference_group: str | None,
) -> None:
    lane = str(resolved.experiment_row["inference_lane"])
    baseline = str(resolved.experiment_row["baseline_policy"])
    lines = [
        f"# GAIRA Experiment Run: {resolved.experiment_row['experiment_id']}",
        "",
        f"- question: {resolved.experiment_row['question_being_asked']}",
        f"- target family: `{resolved.experiment_row['target_family']}`",
        f"- inference lane: `{lane}`",
        f"- representation mode: `{resolved.experiment_row['representation_mode']}`",
        f"- grounding families used: `{'; '.join(resolved.grounding_family_names)}`",
        f"- baseline policy: `{baseline}`",
        f"- dataset/subset alias: `{resolved.subset_alias}`",
        f"- available grounding sources: `{'; '.join(available_sources)}`",
        f"- unavailable grounding sources: `{'; '.join(unavailable_sources) if unavailable_sources else 'none'}`",
    ]
    if reference_group is not None:
        lines.append(f"- resolved comparator/background group: `{reference_group}`")
    lines.extend(
        [
            "",
            "## Execution note",
            "- This run is executed through the GAIRAv3 registry stack rather than an ad hoc experiment script.",
            "- `raw_direct` is fully implemented here. `local_structural_embedding` remains a scaffold path for later matched-background work.",
            "- Grounding exclusions remain hard-fail and are enforced before any retrieval begins.",
            "",
            "## What worked",
            "- Registry resolution across dataset alias, lane, representation, and grounding family completed end to end.",
            "- The run produced machine-readable per-spectrum and grouped outputs with a config snapshot.",
            "",
            "## What remains weak",
            "- `local_structural_embedding` is not executed in this runner yet.",
            "- The biochemical quality of results still depends on grounding coverage and ontology quality; the runner only standardizes execution.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def source_registry_inventory(phase1_registry_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in phase1_registry_df.iterrows():
        records.append(
            {
                "dataset_id": str(row["dataset_id"]),
                "subset_id": str(row["subset_id"]),
                "subset_alias": str(row["subset_alias"]),
                "proposed_phase1_role": str(row["proposed_phase1_role"]),
                "source_key": _registry_source_string(row),
            }
        )
    return pd.DataFrame(records)
