from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gaira.demo.raw_bsv_pilot_utils import (
    ALL_AXES,
    UNIVERSAL_PURE_SOURCE_KEYS,
    apply_source_role_policy,
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
OUTPUT_DIR = ROOT / "reports" / "gaira_bsv_rerun_strict_grounding_v1"
ONTOLOGY_RULES = ROOT / "config" / "phase2_bsv_ontology_rules_v1.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict-grounding rerun for the raw/direct GAIRA BSV pilot.")
    parser.add_argument("--registry-path", type=Path, default=PHASE1_DIR / "phase1_dataset_registry_v2.csv")
    parser.add_argument("--grounding-map-path", type=Path, default=PHASE1_DIR / "phase1_target_grounding_map_v2.csv")
    parser.add_argument("--exclusions-path", type=Path, default=PHASE1_DIR / "phase1_grounding_exclusions.csv")
    parser.add_argument("--pilot-dir", type=Path, default=PILOT_DIR)
    parser.add_argument("--ontology-rules-path", type=Path, default=ONTOLOGY_RULES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--normalization-mode", type=str, default="per_spectrum_sum")
    return parser.parse_args()


def load_query_alias(pilot_dir: Path) -> str:
    summary = pd.read_csv(pilot_dir / "run_summary.csv")
    return str(summary.iloc[0]["query_alias"])


def load_query_df(alias: str, registry_df: pd.DataFrame) -> pd.DataFrame:
    row = get_registry_row_by_alias(registry_df, alias)
    return load_biosample_subset(str(row["dataset_id"]), str(row["subset_id"]))


def load_sources(registry_df: pd.DataFrame, sources: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    frames = []
    available = []
    unavailable = []
    for source in sources:
        spec = parse_source_spec(source, registry_df)
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
        raise RuntimeError("No grounding sources available for strict rerun.")
    return pd.concat(frames, ignore_index=True), available, unavailable


def run_variant(
    *,
    variant_name: str,
    query_alias: str,
    query_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    exclusions_df: pd.DataFrame,
    ontology_rules: pd.DataFrame,
    sources: list[str],
    top_k: int,
    normalization_mode: str,
    primary_sources: set[str],
    caveat_only_sources: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enforce_grounding_exclusions(query_alias, sources, exclusions_df)
    grounding_df, available, unavailable = load_sources(registry_df, sources)
    raw_mapping = map_references_to_axes(grounding_df, ontology_rules)
    filtered_mapping = apply_source_role_policy(
        raw_mapping,
        grounding_df,
        primary_sources=primary_sources,
        caveat_only_sources=caveat_only_sources,
    )
    per_spectrum, hits = build_bsv_profiles(
        query_df,
        grounding_df,
        filtered_mapping,
        top_k=top_k,
        normalization_mode=normalization_mode,
    )
    means = group_mean_bsv(per_spectrum)
    ref_group = resolve_reference_group(query_df)
    deltas = delta_bsv(means, reference_group=ref_group)

    run_meta = pd.DataFrame(
        {
            "variant_name": [variant_name],
            "query_alias": [query_alias],
            "reference_group": [ref_group],
            "available_sources": ["; ".join(available)],
            "unavailable_sources": ["; ".join(unavailable)],
            "primary_sources": ["; ".join(sorted(primary_sources))],
            "caveat_only_sources": ["; ".join(sorted(caveat_only_sources))],
        }
    )
    return per_spectrum, means, deltas, pd.concat([filtered_mapping, run_meta.assign(sample_key="", output_axis="", axis_weight="", axis_kind="", source_key="", contribution_role="")], ignore_index=True)


def write_summary(
    output_path: Path,
    comparison_df: pd.DataFrame,
) -> None:
    lines = [
        "# GAIRA Strict Grounding Rerun Summary",
        "",
        "This rerun keeps the same validation panel but separates grounding roles:",
        "- universal pure grounding can contribute to primary biochemical axes",
        "- serum-support grounding can contribute only to caveat axes in the strict mixed variant",
        "",
    ]
    deltas = comparison_df[comparison_df["row_type"] == "delta"].copy()
    for variant_name in ["prior_full_pool", "strict_universal_only", "strict_universal_plus_serum_caveat_only"]:
        sub = deltas[deltas["variant_name"].astype(str) == variant_name]
        if sub.empty:
            continue
        lines.append(f"## {variant_name}")
        for _, row in sub.iterrows():
            lines.append(
                f"- `{row['comparison']}`: metabolite={row['small_molecule_metabolite']:.3f}, protein={row['protein_peptide']:.3f}, nucleic={row['nucleic_acid']:.3f}, carbohydrate={row['carbohydrate_glycan']:.3f}, matrix={row['matrix_background']:.3f}, substrate={row['substrate_adsorption_bias']:.3f}"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "- The strict rerun is more interpretable if caveat axes absorb serum-related nuisance while primary biochemical axes stop flipping in obviously wrong directions.",
            "- If metabolite movement still fails to increase for the metabolite-spike classes, the next bottleneck is ontology specificity, not serum-support leakage alone.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_registry_inputs(args.registry_path, args.grounding_map_path, args.exclusions_path)
    ontology_rules = load_ontology_rules(args.ontology_rules_path)
    query_alias = load_query_alias(args.pilot_dir)
    query_df = load_query_df(query_alias, inputs.registry)

    prior_group = pd.read_csv(args.pilot_dir / "group_mean_bsv.csv")
    prior_delta = pd.read_csv(args.pilot_dir / "delta_bsv.csv")
    prior_group["variant_name"] = "prior_full_pool"
    prior_group["row_type"] = "group_mean"
    prior_group["comparison"] = prior_group["class_label"]
    prior_delta["variant_name"] = "prior_full_pool"
    prior_delta["row_type"] = "delta"
    prior_delta["class_label"] = prior_delta["group_label"]
    prior_delta["comparison"] = prior_delta["comparison"]

    pilot_summary = pd.read_csv(args.pilot_dir / "run_summary.csv")
    full_sources = [part.strip() for part in str(pilot_summary.iloc[0]["available_grounding_sources"]).split(";") if part.strip()]
    universal_sources = [src for src in full_sources if src in UNIVERSAL_PURE_SOURCE_KEYS]
    serum_sources = [src for src in full_sources if src not in UNIVERSAL_PURE_SOURCE_KEYS]

    variants = [
        (
            "strict_universal_only",
            universal_sources,
            set(universal_sources),
            set(),
        ),
        (
            "strict_universal_plus_serum_caveat_only",
            universal_sources + serum_sources,
            set(universal_sources),
            set(serum_sources),
        ),
    ]

    comparison_rows = [
        prior_group[["variant_name", "row_type", "comparison", "class_label", *ALL_AXES, "unmapped_support"]],
        prior_delta[["variant_name", "row_type", "comparison", "class_label", *ALL_AXES, "unmapped_support"]],
    ]

    for variant_name, sources, primary_sources, caveat_only_sources in variants:
        per_spectrum, means, deltas, mapping_dump = run_variant(
            variant_name=variant_name,
            query_alias=query_alias,
            query_df=query_df,
            registry_df=inputs.registry,
            exclusions_df=inputs.exclusions,
            ontology_rules=ontology_rules,
            sources=sources,
            top_k=args.top_k,
            normalization_mode=args.normalization_mode,
            primary_sources=primary_sources,
            caveat_only_sources=caveat_only_sources,
        )
        per_spectrum.to_csv(args.output_dir / f"per_spectrum_bsv__{variant_name}.csv", index=False)
        means.to_csv(args.output_dir / f"group_mean_bsv__{variant_name}.csv", index=False)
        deltas.to_csv(args.output_dir / f"delta_bsv__{variant_name}.csv", index=False)
        mapping_dump.to_csv(args.output_dir / f"ontology_mapping__{variant_name}.csv", index=False)

        means["variant_name"] = variant_name
        means["row_type"] = "group_mean"
        means["comparison"] = means["class_label"]
        deltas["variant_name"] = variant_name
        deltas["row_type"] = "delta"
        deltas["class_label"] = deltas["group_label"]
        comparison_rows.append(means[["variant_name", "row_type", "comparison", "class_label", *ALL_AXES, "unmapped_support"]])
        comparison_rows.append(deltas[["variant_name", "row_type", "comparison", "class_label", *ALL_AXES, "unmapped_support"]])

    comparison_df = pd.concat(comparison_rows, ignore_index=True)
    comparison_df.to_csv(args.output_dir / "strict_grounding_comparison.csv", index=False)
    write_summary(args.output_dir / "strict_grounding_summary.md", comparison_df)

    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
