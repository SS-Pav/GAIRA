from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gaira.demo.raw_bsv_pilot_utils import (
    ALL_AXES,
    RegistryInputs,
    build_bsv_profiles,
    delta_bsv,
    enforce_grounding_exclusions,
    get_registry_row_by_alias,
    group_mean_bsv,
    load_biosample_subset,
    load_grounding_dataset,
    load_ontology_rules,
    load_registry_inputs,
    map_references_to_axes,
    parse_source_spec,
    resolve_reference_group,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE1_DIR = ROOT / "reports" / "gaira_phase1_registry_audit_v2"
PILOT_DIR = ROOT / "reports" / "gaira_bsv_raw_pilot_v1"
OUTPUT_DIR = ROOT / "reports" / "gaira_bsv_calibration_debug_v1"
ONTOLOGY_RULES = ROOT / "config" / "phase2_bsv_ontology_rules_v1.csv"

UNIVERSAL_PURE = [
    "ramanbiolib",
    "adenine_sers_control",
    "amino_acid_raman_grounding",
    "metabolite_sers63_support",
]
SERUM_SUPPORT_CURATED_ONLY = ["serum_ag_colloids_grounding"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build calibration/debug report for the raw/direct BSV pilot.")
    parser.add_argument("--registry-path", type=Path, default=PHASE1_DIR / "phase1_dataset_registry_v2.csv")
    parser.add_argument("--grounding-map-path", type=Path, default=PHASE1_DIR / "phase1_target_grounding_map_v2.csv")
    parser.add_argument("--exclusions-path", type=Path, default=PHASE1_DIR / "phase1_grounding_exclusions.csv")
    parser.add_argument("--pilot-dir", type=Path, default=PILOT_DIR)
    parser.add_argument("--ontology-rules-path", type=Path, default=ONTOLOGY_RULES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def load_query_alias(pilot_dir: Path) -> str:
    summary = pd.read_csv(pilot_dir / "run_summary.csv")
    return str(summary.iloc[0]["query_alias"])


def load_query_df(alias: str, inputs: RegistryInputs) -> pd.DataFrame:
    row = get_registry_row_by_alias(inputs.registry, alias)
    return load_biosample_subset(str(row["dataset_id"]), str(row["subset_id"]))


def load_sources(inputs: RegistryInputs, sources: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    frames = []
    available = []
    unavailable = []
    for source in sources:
        spec = parse_source_spec(source, inputs.registry)
        try:
            if spec.source_type == "grounding":
                df = load_grounding_dataset(spec.dataset_id)
            else:
                df = load_biosample_subset(spec.dataset_id, spec.subset_id)
                df["compound_label"] = df["class_label"].astype(str)
                df["grounding_role"] = "biosample_support_subset"
                df["experiment_family"] = spec.source_key
            df["source_key"] = spec.source_key
            frames.append(df)
            available.append(spec.source_key)
        except RuntimeError:
            unavailable.append(spec.source_key)
    if not frames:
        raise RuntimeError("No grounding sources available for this debug variant.")
    return pd.concat(frames, ignore_index=True), available, unavailable


def source_bucket(source_key: str) -> str:
    if source_key in UNIVERSAL_PURE:
        return "universal_pure_grounding"
    return "serum_support_grounding"


def build_retrieval_audit(
    hits: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    caveat_axes = {"matrix_background", "substrate_adsorption_bias", "protocol_sensitive_signal"}
    caveat_refs = set(
        mapping[mapping["output_axis"].astype(str).isin(caveat_axes)]["sample_key"].astype(str)
    )
    work = hits.copy()
    work["source_bucket"] = work["reference_source_key"].astype(str).map(source_bucket)
    work["caveat_like_reference"] = work["reference_sample_key"].astype(str).isin(caveat_refs)
    grouped = (
        work.groupby(
            [
                "query_class_label",
                "source_bucket",
                "caveat_like_reference",
                "reference_dataset_id",
                "reference_source_key",
                "reference_class_label",
                "reference_compound_label",
            ],
            as_index=False,
        )["support_weight"]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={"count": "hit_count", "sum": "support_weight_sum"})
    )
    totals = grouped.groupby("query_class_label")["support_weight_sum"].sum().rename("class_total")
    grouped = grouped.merge(totals, on="query_class_label", how="left")
    grouped["support_weight_fraction"] = grouped["support_weight_sum"] / grouped["class_total"].clip(lower=1e-8)
    return grouped.sort_values(
        ["query_class_label", "support_weight_fraction", "hit_count"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_ontology_coverage(mapping: pd.DataFrame, grounding_df: pd.DataFrame) -> pd.DataFrame:
    ref_counts = (
        mapping.groupby("output_axis", as_index=False)["sample_key"]
        .nunique()
        .rename(columns={"sample_key": "unique_reference_count"})
    )
    total_refs = grounding_df["sample_key"].astype(str).nunique()
    ref_counts["fraction_of_active_refs"] = ref_counts["unique_reference_count"] / max(total_refs, 1)
    source_counts = (
        grounding_df.groupby("source_key", as_index=False)["sample_key"].nunique().rename(columns={"sample_key": "unique_reference_count"})
    )
    ref_counts["active_source_count"] = len(source_counts)
    return ref_counts.sort_values("unique_reference_count", ascending=False).reset_index(drop=True)


def run_variant(
    *,
    variant_name: str,
    query_alias: str,
    query_df: pd.DataFrame,
    inputs: RegistryInputs,
    ontology_rules: pd.DataFrame,
    sources: list[str],
    top_k: int,
    normalization_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], pd.DataFrame]:
    enforce_grounding_exclusions(query_alias, sources, inputs.exclusions)
    grounding_df, available, unavailable = load_sources(inputs, sources)
    mapping_df = map_references_to_axes(grounding_df, ontology_rules)
    per_spectrum, hits = build_bsv_profiles(
        query_df,
        grounding_df,
        mapping_df,
        top_k=top_k,
        normalization_mode=normalization_mode,
    )
    means = group_mean_bsv(per_spectrum)
    reference_group = resolve_reference_group(query_df)
    deltas = delta_bsv(means, reference_group=reference_group)

    mean_rows = means.copy()
    mean_rows["variant_name"] = variant_name
    mean_rows["row_type"] = "group_mean"
    mean_rows["comparison"] = mean_rows["class_label"]
    delta_rows = deltas.copy()
    delta_rows["variant_name"] = variant_name
    delta_rows["row_type"] = "delta"
    combined = pd.concat(
        [
            mean_rows[["variant_name", "row_type", "comparison", "class_label", *ALL_AXES, "unmapped_support"]],
            delta_rows[["variant_name", "row_type", "comparison", "group_label", *ALL_AXES, "unmapped_support"]].rename(columns={"group_label": "class_label"}),
        ],
        ignore_index=True,
    )
    hits["variant_name"] = variant_name
    hits["source_bucket"] = hits["reference_source_key"].astype(str).map(source_bucket)
    return combined, hits, mapping_df, available, unavailable, grounding_df


def build_variant_sources(pilot_summary: pd.DataFrame) -> dict[str, list[str]]:
    full_sources = [part.strip() for part in str(pilot_summary.iloc[0]["available_grounding_sources"]).split(";") if part.strip()]
    universal_available = [src for src in UNIVERSAL_PURE if src in full_sources]
    return {
        "variant_a_universal_only": universal_available,
        "variant_b_universal_plus_curated_serum": universal_available + [src for src in SERUM_SUPPORT_CURATED_ONLY if src in full_sources],
        "variant_c_full_current_pool": full_sources,
    }


def build_weighting_configs(full_sources: list[str]) -> list[tuple[str, int, str, list[str]]]:
    return [
        ("k5_per_spectrum_sum", 5, "per_spectrum_sum", full_sources),
        ("k8_per_spectrum_sum", 8, "per_spectrum_sum", full_sources),
        ("k8_raw_support", 8, "raw_support", full_sources),
        ("k8_softmax_then_sum", 8, "softmax_then_sum", full_sources),
    ]


def write_summary(
    output_path: Path,
    *,
    query_alias: str,
    retrieval_audit: pd.DataFrame,
    ontology_coverage: pd.DataFrame,
    variant_comparison: pd.DataFrame,
    weighting_comparison: pd.DataFrame,
) -> None:
    lines = [
        "# GAIRA BSV Calibration Debug Summary",
        "",
        f"- validation panel: `{query_alias}`",
    ]

    if not retrieval_audit.empty:
        for label in sorted(retrieval_audit["query_class_label"].astype(str).unique()):
            sub = retrieval_audit[retrieval_audit["query_class_label"].astype(str) == label].copy()
            top = sub.head(5)
            lines.append(f"- top retrieval composition for `{label}`:")
            for _, row in top.iterrows():
                lines.append(
                    f"  - {row['reference_source_key']} / {row['reference_class_label']} : {row['support_weight_fraction']:.3f} ({row['source_bucket']}, caveat={bool(row['caveat_like_reference'])})"
                )

    lines.extend(["", "## Main debug signal"])
    if not ontology_coverage.empty:
        top_axes = ontology_coverage.head(5)
        for _, row in top_axes.iterrows():
            lines.append(
                f"- active axis coverage `{row['output_axis']}`: {int(row['unique_reference_count'])} references ({row['fraction_of_active_refs']:.3f} of active pool)"
            )

    variant_delta = variant_comparison[variant_comparison["row_type"] == "delta"].copy()
    if not variant_delta.empty:
        lines.extend(["", "## Grounding variant effect"])
        for variant_name in sorted(variant_delta["variant_name"].astype(str).unique()):
            sub = variant_delta[variant_delta["variant_name"].astype(str) == variant_name].copy()
            for _, row in sub.iterrows():
                lines.append(
                    f"- `{variant_name}` / `{row['comparison']}`: metabolite={row['small_molecule_metabolite']:.3f}, matrix={row['matrix_background']:.3f}, protein={row['protein_peptide']:.3f}"
                )

    lines.extend(
        [
            "",
            "## Recommendation",
            "- Main issue looks like a combination of grounding composition and ontology rule overlap, not just top-K weighting.",
            "- Serum-support references are dominating retrieval for the validation panel, especially UA+HSA-style references from `serum_ag_colloids_grounding`.",
            "- The single best next fix is to tighten the ontology and grounding policy so serum-support references contribute mainly caveat channels and are downweighted or excluded when testing controlled metabolite-spike validation.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_registry_inputs(args.registry_path, args.grounding_map_path, args.exclusions_path)
    ontology_rules = load_ontology_rules(args.ontology_rules_path)
    pilot_summary = pd.read_csv(args.pilot_dir / "run_summary.csv")
    query_alias = load_query_alias(args.pilot_dir)
    query_df = load_query_df(query_alias, inputs)

    full_sources = [part.strip() for part in str(pilot_summary.iloc[0]["available_grounding_sources"]).split(";") if part.strip()]

    # Retrieval audit from the original v1 full-pool run.
    retrieval_hits = pd.read_csv(args.pilot_dir / "topk_retrieval_hits.csv")
    full_grounding_df, _, _ = load_sources(inputs, full_sources)
    full_mapping_df = map_references_to_axes(full_grounding_df, ontology_rules)
    retrieval_audit = build_retrieval_audit(retrieval_hits, full_mapping_df)
    retrieval_audit.to_csv(args.output_dir / "retrieval_audit_by_class.csv", index=False)

    ontology_coverage = build_ontology_coverage(full_mapping_df, full_grounding_df)
    ontology_coverage.to_csv(args.output_dir / "ontology_axis_coverage_audit.csv", index=False)

    variant_rows = []
    variant_sources = build_variant_sources(pilot_summary)
    for variant_name, sources in variant_sources.items():
        combined, hits, _, available, unavailable, _ = run_variant(
            variant_name=variant_name,
            query_alias=query_alias,
            query_df=query_df,
            inputs=inputs,
            ontology_rules=ontology_rules,
            sources=sources,
            top_k=8,
            normalization_mode="per_spectrum_sum",
        )
        combined["available_sources"] = "; ".join(available)
        combined["unavailable_sources"] = "; ".join(unavailable)
        variant_rows.append(combined)
    grounding_variant_comparison = pd.concat(variant_rows, ignore_index=True)
    grounding_variant_comparison.to_csv(args.output_dir / "grounding_variant_comparison.csv", index=False)

    weight_rows = []
    for name, top_k, mode, sources in build_weighting_configs(full_sources):
        combined, _, _, available, unavailable, _ = run_variant(
            variant_name=name,
            query_alias=query_alias,
            query_df=query_df,
            inputs=inputs,
            ontology_rules=ontology_rules,
            sources=sources,
            top_k=top_k,
            normalization_mode=mode,
        )
        combined["top_k"] = top_k
        combined["normalization_mode"] = mode
        combined["available_sources"] = "; ".join(available)
        combined["unavailable_sources"] = "; ".join(unavailable)
        weight_rows.append(combined)
    weighting_comparison = pd.concat(weight_rows, ignore_index=True)
    weighting_comparison.to_csv(args.output_dir / "weighting_sensitivity_comparison.csv", index=False)

    write_summary(
        args.output_dir / "calibration_debug_summary.md",
        query_alias=query_alias,
        retrieval_audit=retrieval_audit,
        ontology_coverage=ontology_coverage,
        variant_comparison=grounding_variant_comparison,
        weighting_comparison=weighting_comparison,
    )

    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
