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
STRICT_DIR = ROOT / "reports" / "gaira_bsv_rerun_strict_grounding_v1"
OUTPUT_DIR = ROOT / "reports" / "gaira_bsv_metabolite_subclass_rerun_v1"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refined metabolite-subclass rerun for the GAIRA raw/direct BSV pilot.")
    parser.add_argument("--registry-path", type=Path, default=PHASE1_DIR / "phase1_dataset_registry_v2.csv")
    parser.add_argument("--grounding-map-path", type=Path, default=PHASE1_DIR / "phase1_target_grounding_map_v2.csv")
    parser.add_argument("--exclusions-path", type=Path, default=PHASE1_DIR / "phase1_grounding_exclusions.csv")
    parser.add_argument("--pilot-dir", type=Path, default=PILOT_DIR)
    parser.add_argument("--strict-dir", type=Path, default=STRICT_DIR)
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
        raise RuntimeError("No universal grounding sources available for metabolite subclass rerun.")
    return pd.concat(frames, ignore_index=True), available, unavailable


def load_universal_sources(pilot_dir: Path) -> list[str]:
    summary = pd.read_csv(pilot_dir / "run_summary.csv")
    full_sources = [part.strip() for part in str(summary.iloc[0]["available_grounding_sources"]).split(";") if part.strip()]
    return [src for src in full_sources if src in UNIVERSAL_PURE_SOURCE_KEYS]


def build_subclass_coverage_audit(
    grounding_df: pd.DataFrame,
    raw_mapping: pd.DataFrame,
) -> pd.DataFrame:
    merged = raw_mapping.merge(
        grounding_df[["sample_key", "dataset_id", "source_key", "class_label", "compound_label"]],
        on="sample_key",
        how="left",
    )
    subclass_rows = merged[merged["output_axis"].isin(TIER2_AXES)].copy()
    records = []
    for axis in TIER2_AXES:
        sub = subclass_rows[subclass_rows["output_axis"] == axis].copy()
        refs = sub["sample_key"].astype(str).nunique()
        datasets = sorted(sub["dataset_id"].dropna().astype(str).unique().tolist())
        labels = sorted(
            {
                *(sub["class_label"].dropna().astype(str).tolist()),
                *(sub["compound_label"].dropna().astype(str).tolist()),
            }
        )
        label_blob = " ".join(labels).lower()
        records.append(
            {
                "output_axis": axis,
                "reference_count": int(refs),
                "source_dataset_count": int(len(datasets)),
                "source_datasets": "; ".join(datasets),
                "example_labels": "; ".join(labels[:12]),
                "hyp_related_reference_present": bool(any(token in label_blob for token in ["hypox", "xanth", "urate", "uric"])),
                "erg_related_reference_present": bool(any(token in label_blob for token in ["erg", "ergoth", "thioneine"])),
            }
        )
    return pd.DataFrame(records)


def build_comparison_table(
    strict_comparison: pd.DataFrame,
    refined_delta: pd.DataFrame,
) -> pd.DataFrame:
    strict_delta = strict_comparison[
        (strict_comparison["variant_name"].astype(str) == "strict_universal_only")
        & (strict_comparison["row_type"].astype(str) == "delta")
    ].copy()
    long_rows = []
    axes = ALL_OUTPUT_AXES + ["unmapped_support"]
    for _, row in strict_delta.iterrows():
        comparison = str(row["comparison"])
        refined_row = refined_delta[refined_delta["comparison"].astype(str) == comparison]
        refined_record = refined_row.iloc[0] if not refined_row.empty else None
        for axis in axes:
            strict_value = float(row[axis]) if axis in row and pd.notna(row[axis]) else 0.0
            refined_value = float(refined_record[axis]) if refined_record is not None and axis in refined_record.index and pd.notna(refined_record[axis]) else 0.0
            long_rows.append(
                {
                    "comparison": comparison,
                    "axis": axis,
                    "strict_universal_only_value": strict_value,
                    "refined_subclass_value": refined_value,
                    "change_refined_minus_strict": refined_value - strict_value,
                }
            )
    return pd.DataFrame(long_rows)


def write_summary(
    output_path: Path,
    coverage_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    lines = [
        "# GAIRA Metabolite Subclass Rerun Summary",
        "",
        "This rerun keeps the strict universal-only biochemical grounding policy and replaces the coarse metabolite bucket with a minimal evidence-backed subclass layer.",
        "",
        "## Coverage Audit",
    ]
    for _, row in coverage_df.iterrows():
        lines.append(
            f"- `{row['output_axis']}`: {int(row['reference_count'])} mapped references from `{row['source_datasets'] or 'none'}`; Hyp-like present={bool(row['hyp_related_reference_present'])}; Erg-like present={bool(row['erg_related_reference_present'])}"
        )

    lines.extend(["", "## Delta BSV"])
    for _, row in delta_df.iterrows():
        lines.append(
            f"- `{row['comparison']}`: metabolite={row.get('small_molecule_metabolite', 0.0):.3f}, purine={row.get('purine_like_metabolite', 0.0):.3f}, sulfur={row.get('sulfur_containing_metabolite', 0.0):.3f}, aromatic={row.get('aromatic_metabolite_like', 0.0):.3f}, organic_acid={row.get('organic_acid_like', 0.0):.3f}, amino_acid_small={row.get('amino_acid_like_small_molecule', 0.0):.3f}"
        )

    hyp_row = delta_df[delta_df["comparison"].astype(str) == "Hyp-vs-Bkg"]
    erg_row = delta_df[delta_df["comparison"].astype(str) == "Erg-vs-Bkg"]
    lines.extend(["", "## Interpretation"])
    if not hyp_row.empty:
        hyp = hyp_row.iloc[0]
        lines.append(
            f"- `Hyp` purine-like movement = {hyp.get('purine_like_metabolite', 0.0):.3f}; broad metabolite movement = {hyp.get('small_molecule_metabolite', 0.0):.3f}."
        )
    if not erg_row.empty:
        erg = erg_row.iloc[0]
        lines.append(
            f"- `Erg` sulfur-like movement = {erg.get('sulfur_containing_metabolite', 0.0):.3f}; broad metabolite movement = {erg.get('small_molecule_metabolite', 0.0):.3f}."
        )

    lacking = coverage_df[
        ~coverage_df["hyp_related_reference_present"].astype(bool) | ~coverage_df["erg_related_reference_present"].astype(bool)
    ].copy()
    if not lacking.empty:
        lines.append(
            "- Universal grounding coverage does not contain explicit Erg or Hyp references in the refined subclass pool, so remaining failure is likely driven more by missing reference coverage than by axis naming alone."
        )

    changed = comparison_df.sort_values("change_refined_minus_strict", ascending=False).head(8)
    lines.extend(["", "## Biggest Axis Changes vs Strict Universal Only"])
    for _, row in changed.iterrows():
        lines.append(
            f"- `{row['comparison']}` / `{row['axis']}`: strict={row['strict_universal_only_value']:.3f}, refined={row['refined_subclass_value']:.3f}, change={row['change_refined_minus_strict']:.3f}"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_registry_inputs(args.registry_path, args.grounding_map_path, args.exclusions_path)
    ontology_rules = load_ontology_rules(args.ontology_rules_path)
    query_alias = load_query_alias(args.pilot_dir)
    query_df = load_query_df(query_alias, inputs.registry)

    universal_sources = load_universal_sources(args.pilot_dir)
    enforce_grounding_exclusions(query_alias, universal_sources, inputs.exclusions)
    grounding_df, available_sources, unavailable_sources = load_sources(inputs.registry, universal_sources)

    raw_mapping = map_references_to_axes(grounding_df, ontology_rules)
    subclass_coverage = build_subclass_coverage_audit(grounding_df, raw_mapping)
    filtered_mapping = apply_source_role_policy(
        raw_mapping,
        grounding_df,
        primary_sources=set(available_sources),
        caveat_only_sources=set(),
        primary_axis_names=TIER1_AXES + TIER2_AXES,
        caveat_axis_names=CAVEAT_AXES,
    )

    per_spectrum, hits = build_bsv_profiles(
        query_df,
        grounding_df,
        filtered_mapping,
        top_k=args.top_k,
        normalization_mode=args.normalization_mode,
        axis_names=ALL_OUTPUT_AXES,
    )
    means = group_mean_bsv(per_spectrum, axis_names=ALL_OUTPUT_AXES)
    reference_group = resolve_reference_group(query_df)
    deltas = delta_bsv(means, reference_group=reference_group, axis_names=ALL_OUTPUT_AXES)

    strict_comparison = pd.read_csv(args.strict_dir / "strict_grounding_comparison.csv")
    comparison_df = build_comparison_table(strict_comparison, deltas)

    per_spectrum.to_csv(args.output_dir / "per_spectrum_bsv_refined.csv", index=False)
    means.to_csv(args.output_dir / "group_mean_bsv_refined.csv", index=False)
    deltas.to_csv(args.output_dir / "delta_bsv_refined.csv", index=False)
    hits.to_csv(args.output_dir / "topk_retrieval_hits_refined.csv", index=False)
    filtered_mapping.to_csv(args.output_dir / "ontology_mapping_refined.csv", index=False)
    subclass_coverage.to_csv(args.output_dir / "subclass_coverage_audit.csv", index=False)
    comparison_df.to_csv(args.output_dir / "strict_universal_vs_refined_comparison.csv", index=False)

    run_meta = pd.DataFrame(
        {
            "query_alias": [query_alias],
            "reference_group": [reference_group],
            "normalization_mode": [args.normalization_mode],
            "top_k": [args.top_k],
            "available_universal_sources": ["; ".join(available_sources)],
            "unavailable_universal_sources": ["; ".join(unavailable_sources)],
            "tier1_axes": ["; ".join(TIER1_AXES)],
            "tier2_axes": ["; ".join(TIER2_AXES)],
        }
    )
    run_meta.to_csv(args.output_dir / "run_summary.csv", index=False)
    write_summary(args.output_dir / "metabolite_subclass_summary.md", subclass_coverage, deltas, comparison_df)

    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
