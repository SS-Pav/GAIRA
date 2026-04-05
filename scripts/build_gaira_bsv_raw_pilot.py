from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from gaira.demo.raw_bsv_pilot_utils import (
    ALL_AXES,
    RegistryInputs,
    build_bsv_profiles,
    choose_demo_alias,
    delta_bsv,
    derive_grounding_sources_for_alias,
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
DEFAULT_PHASE1_DIR = ROOT / "reports" / "gaira_phase1_registry_audit_v2"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "gaira_bsv_raw_pilot_v1"
DEFAULT_ONTOLOGY_RULES = ROOT / "config" / "phase2_bsv_ontology_rules_v1.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the minimal raw/direct GAIRA Phase 2 BSV pilot scaffold.")
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_PHASE1_DIR / "phase1_dataset_registry_v2.csv")
    parser.add_argument("--grounding-map-path", type=Path, default=DEFAULT_PHASE1_DIR / "phase1_target_grounding_map_v2.csv")
    parser.add_argument("--exclusions-path", type=Path, default=DEFAULT_PHASE1_DIR / "phase1_grounding_exclusions.csv")
    parser.add_argument("--ontology-rules-path", type=Path, default=DEFAULT_ONTOLOGY_RULES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--query-alias", type=str, default="")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--normalization-mode",
        choices=["raw_support", "per_spectrum_sum", "softmax_then_sum", "delta_zscore_placeholder"],
        default="per_spectrum_sum",
    )
    return parser.parse_args()


def load_query_dataframe(alias: str, inputs: RegistryInputs) -> pd.DataFrame:
    row = get_registry_row_by_alias(inputs.registry, alias)
    return load_biosample_subset(str(row["dataset_id"]), str(row["subset_id"]))


def load_grounding_dataframe(alias: str, inputs: RegistryInputs) -> tuple[pd.DataFrame, list[str], list[str]]:
    allowed_sources = derive_grounding_sources_for_alias(alias, inputs)
    enforce_grounding_exclusions(alias, allowed_sources, inputs.exclusions)
    frames = []
    available = []
    unavailable = []
    for source in allowed_sources:
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
        raise RuntimeError(f"No available grounding sources for alias {alias}.")
    return pd.concat(frames, ignore_index=True), available, unavailable


def save_barplot(delta_df: pd.DataFrame, output_path: Path) -> None:
    if delta_df.empty:
        return
    long_df = delta_df.melt(
        id_vars=["comparison"],
        value_vars=ALL_AXES,
        var_name="axis_name",
        value_name="delta_value",
    )
    plt.figure(figsize=(12, 5.6))
    sns.barplot(data=long_df, x="axis_name", y="delta_value", hue="comparison")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Delta BSV")
    plt.xlabel("")
    plt.title("Delta BSV Axes")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_group_heatmap(group_means: pd.DataFrame, output_path: Path) -> None:
    if group_means.empty:
        return
    heat = group_means.set_index("class_label")[ALL_AXES]
    plt.figure(figsize=(10.5, max(3.5, 0.55 * len(heat))))
    sns.heatmap(heat, cmap="mako", annot=False)
    plt.title("Group Mean BSV")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def write_note(
    output_path: Path,
    *,
    alias: str,
    normalization_mode: str,
    top_k: int,
    available_sources: list[str],
    unavailable_sources: list[str],
    reference_group: str,
) -> None:
    lines = [
        "# GAIRA Raw/Direct BSV Pilot Note",
        "",
        f"- query alias: `{alias}`",
        f"- similarity method: cosine similarity on aligned direct processed spectra using the poly3 vector-processed tables",
        f"- weighting method: `{normalization_mode}`",
        f"- top-K retrieval: `{top_k}` references per spectrum",
        "- ontology mapping format: deterministic CSV rules file (`config/phase2_bsv_ontology_rules_v1.csv`) applied to dataset/source/class/compound fields",
        "- exclusion enforcement: the pilot hard-fails if the Phase 1 v2 exclusion CSV is missing or if any selected grounding source is forbidden for the requested experiment family",
        f"- selected reference/background group for delta-BSV: `{reference_group}`",
        f"- available grounding sources: `{'; '.join(available_sources)}`",
        f"- unavailable but allowed grounding sources: `{'; '.join(unavailable_sources) if unavailable_sources else 'none'}`",
        "",
        "## Weak assumptions remaining",
        "- The ontology is intentionally minimal and broad. Unmapped references remain explicit rather than being guessed into categories.",
        "- Universal pure grounding is incomplete because `ramanbiolib` is listed in the registry but has no current processed grounding rows in the canonical DB.",
        "- The `delta_zscore_placeholder` mode is only a hook in this scaffold and does not yet implement a true cross-group normalization benchmark.",
        "- Cosine similarity on processed spectra is a simple direct baseline, not a final mechanistic similarity model.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_registry_inputs(args.registry_path, args.grounding_map_path, args.exclusions_path)
    query_alias = args.query_alias or choose_demo_alias(inputs.registry)

    query_df = load_query_dataframe(query_alias, inputs)
    grounding_df, available_sources, unavailable_sources = load_grounding_dataframe(query_alias, inputs)
    ontology_rules = load_ontology_rules(args.ontology_rules_path)
    mapping_df = map_references_to_axes(grounding_df, ontology_rules)

    per_spectrum_df, retrieval_df = build_bsv_profiles(
        query_df,
        grounding_df,
        mapping_df,
        top_k=args.top_k,
        normalization_mode=args.normalization_mode,
    )
    group_means_df = group_mean_bsv(per_spectrum_df)
    reference_group = resolve_reference_group(query_df)
    delta_df = delta_bsv(group_means_df, reference_group=reference_group)

    per_spectrum_df.to_csv(args.output_dir / "per_spectrum_bsv.csv", index=False)
    group_means_df.to_csv(args.output_dir / "group_mean_bsv.csv", index=False)
    delta_df.to_csv(args.output_dir / "delta_bsv.csv", index=False)
    retrieval_df.to_csv(args.output_dir / "topk_retrieval_hits.csv", index=False)
    mapping_df.to_csv(args.output_dir / "ontology_mapping_applied.csv", index=False)
    ontology_rules.to_csv(args.output_dir / "ontology_rules_used.csv", index=False)
    pd.DataFrame(
        {
            "query_alias": [query_alias],
            "normalization_mode": [args.normalization_mode],
            "top_k": [args.top_k],
            "reference_group": [reference_group],
            "available_grounding_sources": ["; ".join(available_sources)],
            "unavailable_grounding_sources": ["; ".join(unavailable_sources)],
        }
    ).to_csv(args.output_dir / "run_summary.csv", index=False)

    save_barplot(delta_df, args.output_dir / "delta_bsv_axes.png")
    save_group_heatmap(group_means_df, args.output_dir / "group_mean_bsv_heatmap.png")
    write_note(
        args.output_dir / "pilot_note.md",
        alias=query_alias,
        normalization_mode=args.normalization_mode,
        top_k=args.top_k,
        available_sources=available_sources,
        unavailable_sources=unavailable_sources,
        reference_group=reference_group,
    )

    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
