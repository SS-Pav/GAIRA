from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gaira.demo.raw_bsv_pilot_utils import (
    CAVEAT_AXES,
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
BASELINE_DIR = ROOT / "reports" / "gaira_bsv_metabolite_subclass_rerun_v1"
OUTPUT_DIR = ROOT / "reports" / "gaira_bsv_targeted_grounding_rerun_v1"
ONTOLOGY_RULES = ROOT / "config" / "phase2_bsv_ontology_rules_v2.csv"

TIER1_AXES = [
    "protein_peptide",
    "lipid_membrane",
    "nucleic_acid",
    "carbohydrate_glycan",
    "small_molecule_metabolite",
]
TIER2_AXES = [
    "purine_like_metabolite",
    "organic_acid_like",
    "aromatic_metabolite_like",
    "sulfur_containing_metabolite",
    "amino_acid_like_small_molecule",
]
ALL_OUTPUT_AXES = TIER1_AXES + TIER2_AXES + CAVEAT_AXES
TARGETED_SERUM_CLASS_LABELS = [
    "Hypox",
    "Ergo",
    "Xanth",
    "UA",
    "UAfree",
    "UAbound",
    "UAiso",
    "UA+HSA",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-targeted serum biochemical grounding rerun for GAIRA.")
    parser.add_argument("--registry-path", type=Path, default=PHASE1_DIR / "phase1_dataset_registry_v2.csv")
    parser.add_argument("--grounding-map-path", type=Path, default=PHASE1_DIR / "phase1_target_grounding_map_v2.csv")
    parser.add_argument("--exclusions-path", type=Path, default=PHASE1_DIR / "phase1_grounding_exclusions.csv")
    parser.add_argument("--pilot-dir", type=Path, default=PILOT_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
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


def load_pilot_universal_sources(pilot_dir: Path) -> list[str]:
    summary = pd.read_csv(pilot_dir / "run_summary.csv")
    full_sources = [part.strip() for part in str(summary.iloc[0]["available_grounding_sources"]).split(";") if part.strip()]
    return [src for src in full_sources if src in UNIVERSAL_PURE_SOURCE_KEYS]


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
        raise RuntimeError("No grounding sources available for targeted rerun.")
    return pd.concat(frames, ignore_index=True), available, unavailable


def build_variant(
    *,
    query_alias: str,
    query_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    exclusions_df: pd.DataFrame,
    ontology_rules: pd.DataFrame,
    sources: list[str],
    top_k: int,
    normalization_mode: str,
    primary_sources: set[str],
    targeted_source_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enforce_grounding_exclusions(query_alias, sources, exclusions_df)
    grounding_df, _, _ = load_sources(registry_df, sources)
    if targeted_source_key:
        keep_mask = (
            grounding_df["source_key"].astype(str) != targeted_source_key
        ) | grounding_df["class_label"].astype(str).isin(TARGETED_SERUM_CLASS_LABELS)
        grounding_df = grounding_df[keep_mask].copy()
    raw_mapping = map_references_to_axes(grounding_df, ontology_rules)
    filtered_mapping = apply_source_role_policy(
        raw_mapping,
        grounding_df,
        primary_sources=primary_sources,
        caveat_only_sources=set(),
        primary_axis_names=TIER1_AXES + TIER2_AXES,
        caveat_axis_names=CAVEAT_AXES,
    )
    per_spectrum, hits = build_bsv_profiles(
        query_df,
        grounding_df,
        filtered_mapping,
        top_k=top_k,
        normalization_mode=normalization_mode,
        axis_names=ALL_OUTPUT_AXES,
    )
    means = group_mean_bsv(per_spectrum, axis_names=ALL_OUTPUT_AXES)
    deltas = delta_bsv(means, reference_group=resolve_reference_group(query_df), axis_names=ALL_OUTPUT_AXES)
    return per_spectrum, means, deltas, hits


def summarize_hits(hits_df: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    grouped = (
        hits_df.groupby(
            ["query_class_label", "reference_dataset_id", "reference_class_label", "reference_compound_label"],
            dropna=False,
        )["support_weight"]
        .sum()
        .reset_index()
        .sort_values(["query_class_label", "support_weight"], ascending=[True, False])
    )
    grouped["rank_within_query_class"] = grouped.groupby("query_class_label").cumcount() + 1
    grouped = grouped[grouped["rank_within_query_class"] <= 10].copy()
    grouped.insert(0, "variant_name", variant_name)
    return grouped


def build_comparison_table(baseline_delta: pd.DataFrame, targeted_delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    axes = ALL_OUTPUT_AXES + ["unmapped_support"]
    for _, row in baseline_delta.iterrows():
        comparison = str(row["comparison"])
        target = targeted_delta[targeted_delta["comparison"].astype(str) == comparison]
        target_row = target.iloc[0] if not target.empty else None
        for axis in axes:
            base_val = float(row[axis]) if axis in row.index and pd.notna(row[axis]) else 0.0
            target_val = float(target_row[axis]) if target_row is not None and axis in target_row.index and pd.notna(target_row[axis]) else 0.0
            rows.append(
                {
                    "comparison": comparison,
                    "axis": axis,
                    "baseline_strict_universal_only": base_val,
                    "targeted_biochemical_extension": target_val,
                    "change_targeted_minus_baseline": target_val - base_val,
                }
            )
    return pd.DataFrame(rows)


def write_summary(
    output_path: Path,
    baseline_delta: pd.DataFrame,
    targeted_delta: pd.DataFrame,
    hit_summary: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    def row_for(df: pd.DataFrame, label: str) -> pd.Series | None:
        sub = df[df["comparison"].astype(str) == label]
        return None if sub.empty else sub.iloc[0]

    lines = [
        "# GAIRA Validation-Targeted Grounding Rerun Summary",
        "",
        "This rerun keeps the metabolite-subclass ontology and adds only explicit compound-controlled serum grounding rows for the current validation family.",
        "",
        "## Delta Comparison",
    ]
    for comparison in ["Erg-vs-Bkg", "Hyp-vs-Bkg"]:
        base = row_for(baseline_delta, comparison)
        targ = row_for(targeted_delta, comparison)
        if base is None or targ is None:
            continue
        lines.append(
            f"- `{comparison}` baseline: metabolite={base['small_molecule_metabolite']:.3f}, purine={base['purine_like_metabolite']:.3f}, sulfur={base['sulfur_containing_metabolite']:.3f}, matrix={base['matrix_background']:.3f}"
        )
        lines.append(
            f"- `{comparison}` targeted: metabolite={targ['small_molecule_metabolite']:.3f}, purine={targ['purine_like_metabolite']:.3f}, sulfur={targ['sulfur_containing_metabolite']:.3f}, matrix={targ['matrix_background']:.3f}"
        )

    lines.extend(["", "## Top Retrieval Hits"])
    for query_class in ["Bkg", "Erg", "Hyp"]:
        sub = hit_summary[hit_summary["query_class_label"].astype(str) == query_class].copy()
        if sub.empty:
            continue
        lines.append(f"- `{query_class}` top hits:")
        for _, row in sub.head(5).iterrows():
            lines.append(
                f"  `{row['reference_dataset_id']}::{row['reference_class_label']}` ({row['reference_compound_label']}) weight={row['support_weight']:.3f}"
            )

    hyp = row_for(targeted_delta, "Hyp-vs-Bkg")
    erg = row_for(targeted_delta, "Erg-vs-Bkg")
    lines.extend(["", "## Interpretation"])
    if hyp is not None:
        lines.append(
            f"- `Hyp` purine-like movement after targeted extension = {hyp['purine_like_metabolite']:.3f}."
        )
    if erg is not None:
        lines.append(
            f"- `Erg` sulfur-containing movement after targeted extension = {erg['sulfur_containing_metabolite']:.3f}."
        )
    if hyp is not None and erg is not None:
        lines.append(
            f"- Caveat leakage remains controlled if matrix stays near zero: `Hyp` matrix={hyp['matrix_background']:.3f}, `Erg` matrix={erg['matrix_background']:.3f}."
        )

    biggest = comparison_df.sort_values("change_targeted_minus_baseline", ascending=False).head(8)
    lines.extend(["", "## Biggest Changes vs Baseline"])
    for _, row in biggest.iterrows():
        lines.append(
            f"- `{row['comparison']}` / `{row['axis']}`: baseline={row['baseline_strict_universal_only']:.3f}, targeted={row['targeted_biochemical_extension']:.3f}, change={row['change_targeted_minus_baseline']:.3f}"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_registry_inputs(args.registry_path, args.grounding_map_path, args.exclusions_path)
    ontology_rules = load_ontology_rules(args.ontology_rules_path)
    query_alias = load_query_alias(args.pilot_dir)
    query_df = load_query_df(query_alias, inputs.registry)
    universal_sources = load_pilot_universal_sources(args.pilot_dir)

    baseline_per, baseline_means, baseline_delta, baseline_hits = build_variant(
        query_alias=query_alias,
        query_df=query_df,
        registry_df=inputs.registry,
        exclusions_df=inputs.exclusions,
        ontology_rules=ontology_rules,
        sources=universal_sources,
        top_k=args.top_k,
        normalization_mode=args.normalization_mode,
        primary_sources=set(universal_sources),
    )

    targeted_sources = universal_sources + ["serum_ag_colloids_grounding"]
    targeted_per, targeted_means, targeted_delta, targeted_hits = build_variant(
        query_alias=query_alias,
        query_df=query_df,
        registry_df=inputs.registry,
        exclusions_df=inputs.exclusions,
        ontology_rules=ontology_rules,
        sources=targeted_sources,
        top_k=args.top_k,
        normalization_mode=args.normalization_mode,
        primary_sources=set(universal_sources + ["serum_ag_colloids_grounding"]),
        targeted_source_key="serum_ag_colloids_grounding",
    )

    baseline_hit_summary = summarize_hits(baseline_hits, "strict_universal_only")
    targeted_hit_summary = summarize_hits(targeted_hits, "validation_targeted_biochemical_grounding")
    comparison_df = build_comparison_table(baseline_delta, targeted_delta)

    baseline_per.to_csv(args.output_dir / "per_spectrum_bsv__strict_universal_only.csv", index=False)
    baseline_means.to_csv(args.output_dir / "group_mean_bsv__strict_universal_only.csv", index=False)
    baseline_delta.to_csv(args.output_dir / "delta_bsv__strict_universal_only.csv", index=False)
    baseline_hits.to_csv(args.output_dir / "topk_hits__strict_universal_only.csv", index=False)

    targeted_per.to_csv(args.output_dir / "per_spectrum_bsv__validation_targeted_biochemical_grounding.csv", index=False)
    targeted_means.to_csv(args.output_dir / "group_mean_bsv__validation_targeted_biochemical_grounding.csv", index=False)
    targeted_delta.to_csv(args.output_dir / "delta_bsv__validation_targeted_biochemical_grounding.csv", index=False)
    targeted_hits.to_csv(args.output_dir / "topk_hits__validation_targeted_biochemical_grounding.csv", index=False)

    pd.concat([baseline_hit_summary, targeted_hit_summary], ignore_index=True).to_csv(
        args.output_dir / "top_retrieval_hit_summary.csv", index=False
    )
    comparison_df.to_csv(args.output_dir / "targeted_grounding_comparison.csv", index=False)
    write_summary(
        args.output_dir / "targeted_grounding_summary.md",
        baseline_delta,
        targeted_delta,
        targeted_hit_summary,
        comparison_df,
    )

    run_meta = pd.DataFrame(
        {
            "query_alias": [query_alias],
            "reference_group": [resolve_reference_group(query_df)],
            "normalization_mode": [args.normalization_mode],
            "top_k": [args.top_k],
            "baseline_sources": ["; ".join(universal_sources)],
            "targeted_added_labels": ["; ".join(TARGETED_SERUM_CLASS_LABELS)],
            "targeted_dataset": ["serum_ag_colloids_grounding"],
        }
    )
    run_meta.to_csv(args.output_dir / "run_summary.csv", index=False)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
