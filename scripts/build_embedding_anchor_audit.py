#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METADATA_PATH = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v5_full_true_gpu_run1/metadata.csv")
DEFAULT_OUTPUT_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_anchor_audit")
DEFAULT_ANCHOR_TABLE_PATH = DEFAULT_OUTPUT_DIR / "embedding_anchor_table_v1.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the GAIRAM cross-dataset anchor audit and harmonized anchor table.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anchor-table-path", type=Path, default=DEFAULT_ANCHOR_TABLE_PATH)
    return parser.parse_args()


def normalize_label(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def choose_anchor(
    sample_type: str,
    dataset_id: str,
    label_optional: str,
    current_semantic_group: str,
) -> tuple[str, str, str, str]:
    label = normalize_label(label_optional)
    if sample_type == "grounding":
        return grounding_anchor(dataset_id, label, current_semantic_group)
    if sample_type == "serum":
        return serum_anchor(dataset_id, label)
    if sample_type == "ev":
        return ev_anchor(dataset_id, label)
    return ("", "unmapped", "low", "No anchor mapping rule was defined for this record.")


def grounding_anchor(dataset_id: str, label: str, current_semantic_group: str) -> tuple[str, str, str, str]:
    theme = grounding_theme_for_label(label)
    if theme:
        return (
            theme,
            "grounding_linked_biochemical_theme",
            "high" if current_semantic_group else "medium",
            "Broad biochemical theme anchor derived from controlled grounding label identity.",
        )
    return (
        "grounding_other_controlled_reference",
        "dataset_native_class_anchor",
        "low",
        "Controlled grounding record without a broad biochemical theme bridge.",
    )


def serum_anchor(dataset_id: str, label: str) -> tuple[str, str, str, str]:
    if dataset_id == "cca_hcc_lm_serum_sers":
        if label == "healthy_control":
            return ("serum_control_like", "cross_dataset_harmonized_biological_anchor", "high", "Healthy serum control cohort.")
        if label in {"cca", "hcc", "lm"}:
            return (
                "serum_hepatobiliary_disease_enriched",
                "cross_dataset_harmonized_biological_anchor",
                "high",
                "Liver or hepatobiliary disease-enriched serum cohort.",
            )
    if dataset_id == "covid_serum_raman":
        if label in {"healthy_control", "tube_control"}:
            return ("serum_control_like", "cross_dataset_harmonized_biological_anchor", "high", "Control-like serum cohort.")
        if label in {"covid_confirmed", "suspected_case"}:
            return (
                "serum_inflammatory_or_infection_enriched",
                "cross_dataset_harmonized_biological_anchor",
                "high",
                "Inflammatory or infection-enriched serum cohort.",
            )
    if dataset_id == "serum_protocol_comparison":
        if label.startswith("p"):
            return (
                "serum_control_like",
                "cross_dataset_harmonized_biological_anchor",
                "low",
                "Single commercial serum observed through protocol variants; usable only as a weak baseline-like bridge.",
            )
    if dataset_id == "ergothioneine_serum":
        if label.startswith("erg_"):
            return (
                "oxidative_redox_associated",
                "grounding_linked_biochemical_theme",
                "medium",
                "Ergothioneine-spiked serum concentration series used as a cautious redox-associated bridge.",
            )
    if dataset_id == "cspp_serum":
        if label == "bkg":
            return ("serum_background_or_blank", "dataset_native_class_anchor", "medium", "Background paper or blank-like CSPP reference.")
        if label in {"erg", "hyp"}:
            return (
                "serum_metabolite_spiked_or_augmented",
                "cross_dataset_harmonized_biological_anchor",
                "medium",
                "Metabolite-spiked serum CSPP condition.",
            )
        if label in {"filtration", "unprocessed", "standard"} or label.endswith(("h", "w", "m")) or label.replace("_", "").replace(".", "").isdigit():
            return (
                "serum_process_or_protocol_variation",
                "process_or_protocol_anchor",
                "low",
                "Serum processing or substrate/protocol optimization condition.",
            )
    if dataset_id == "serum_ag_colloids":
        if label.startswith("serum"):
            if "spiked" in label or "enzyme" in label:
                return (
                    "serum_metabolite_spiked_or_augmented",
                    "cross_dataset_harmonized_biological_anchor",
                    "medium",
                    "Serum Ag-colloid archive spiked or enzyme-perturbed serum condition.",
                )
            return (
                "serum_control_like",
                "cross_dataset_harmonized_biological_anchor",
                "medium",
                "Serum Ag-colloid archive baseline serum matrix condition.",
            )
        theme = grounding_theme_for_label(label)
        if theme:
            return (
                theme,
                "grounding_linked_biochemical_theme",
                "medium",
                "Serum Ag-colloid controlled component mapped to a broad biochemical theme.",
            )
        return (
            "serum_background_component_reference",
            "dataset_native_class_anchor",
            "low",
            "Controlled serum archive component without a clean harmonized biochemical bridge.",
        )
    theme = grounding_theme_for_label(label)
    if theme:
        return (theme, "grounding_linked_biochemical_theme", "low", "Fallback biochemical-theme anchor for serum record.")
    return ("", "unmapped", "low", "No serum harmonized anchor rule matched.")


def ev_anchor(dataset_id: str, label: str) -> tuple[str, str, str, str]:
    if dataset_id == "diabetes_plasma_ev_sers":
        if label in {"impact", "strong_d"}:
            return (
                "ev_disease_or_stress",
                "cross_dataset_harmonized_biological_anchor",
                "high",
                "Perturbed plasma EV subgroup from the diabetes archive.",
            )
    if dataset_id == "shine_ev_sers":
        if label.startswith("d0_"):
            return (
                "ev_control_or_baseline",
                "cross_dataset_harmonized_biological_anchor",
                "medium",
                "Baseline-like SHINE EV state before stronger injury/stress conditions.",
            )
        if label.startswith("d1_") or label.startswith("d2_"):
            return (
                "ev_disease_or_stress",
                "cross_dataset_harmonized_biological_anchor",
                "medium",
                "Injury- or stress-enriched SHINE EV state.",
            )
    if dataset_id == "small2023_ev":
        if label.startswith("c"):
            return (
                "ev_control_or_baseline",
                "cross_dataset_harmonized_biological_anchor",
                "medium",
                "General EV composition-series record used as a broad baseline-like EV bridge.",
            )
        if label in {"hec", "hela", "ht", "mef", "thp"}:
            return (
                "ev_cell_line_model",
                "dataset_native_class_anchor",
                "low",
                "Cell-line-specific EV model; useful within dataset but not a strong cross-dataset anchor.",
            )
    return ("", "unmapped", "low", "No EV harmonized anchor rule matched.")


def grounding_theme_for_label(label: str) -> str:
    if not label:
        return ""
    purine = {
        "ade", "adenine", "adenine_100nano", "adenine_100pg", "adenine_10micro", "adenine_10nano", "adenine_10pg",
        "adenine_1micro", "adenine_1ng_after_two_weeks", "adenine_1ng_fresh", "adenine_1ng_ml",
        "adenine_1ng_replicate_series", "adenine_1ug_average", "adenine_1ug_colloid_average",
        "gua", "ua", "uafree", "uabound", "uahsa", "uahsafilterlower", "uahsafilterupper", "uaiso", "uaisohsa",
        "uaisohsafilterlower", "uaisohsafilterupper", "ura", "xanth", "hypox", "ribo", "thy",
    }
    nucleic = {"dna", "rna"}
    protein = {
        "alb", "ala", "arg", "asp", "gly", "his", "ile", "leu", "met", "methio", "phe", "pro", "ser", "trp", "tyr",
        "val", "valine", "glutamic", "glutamic_acid", "glut", "l_glu", "gluth", "hydroxypro", "cys", "homocysteine",
        "homocystine", "l_asparagine", "l_arginine", "l_cystathionine", "l_cystine", "l_cysteic_acid", "cys_gly",
    }
    lipid = {"chol", "oleic", "stearic", "triolein", "phosph", "phinositol", "phinositol", "coa"}
    carbohydrate = {"gluc", "glucose", "fruct", "galact", "mann", "glycerol", "glycogen", "citric", "lact"}
    redox = {"erg", "ergo", "ergothioneine", "asc", "biliverdin", "glutathione", "kynurenine", "caffeine", "dopamine"}
    if label in purine:
        return "purine_metabolite_associated"
    if label in nucleic:
        return "nucleic_acid_associated"
    if label in protein:
        return "protein_peptide_associated"
    if label in lipid:
        return "lipid_membrane_associated"
    if label in carbohydrate:
        return "carbohydrate_associated"
    if label in redox:
        return "oxidative_redox_associated"
    if "serum" in label or "bkg" == label:
        return "serum_matrix_associated"
    return ""


def current_semantic_group_tables(metadata_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    working = metadata_df.copy()
    working["current_semantic_group"] = working["semantic_group"].fillna("").astype(str)
    grouped = working[working["current_semantic_group"] != ""].groupby(["sample_type", "current_semantic_group"], dropna=False)

    group_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    total_records = len(working)

    overall_semantic_pairs = 0
    overall_cross_pairs = 0
    for (sample_type, group_name), group in grouped:
        dataset_counts = group.groupby("dataset_id").size()
        total_pairs = int(len(group) * (len(group) - 1) / 2)
        same_dataset_pairs = int(sum(int(count * (count - 1) / 2) for count in dataset_counts))
        cross_pairs = max(total_pairs - same_dataset_pairs, 0)
        overall_semantic_pairs += total_pairs
        overall_cross_pairs += cross_pairs
        group_rows.append(
            {
                "sample_type": sample_type,
                "current_semantic_group": group_name,
                "record_count": int(len(group)),
                "dataset_count": int(group["dataset_id"].nunique()),
                "datasets": ";".join(sorted(group["dataset_id"].astype(str).unique().tolist())),
                "cross_dataset_pair_count": cross_pairs,
                "same_dataset_pair_count": same_dataset_pairs,
                "cross_dataset_fraction": cross_pairs / total_pairs if total_pairs else np.nan,
            }
        )

    for sample_type, subset in working.groupby("sample_type", dropna=False):
        semantic_subset = subset[subset["current_semantic_group"] != ""].copy()
        if semantic_subset.empty:
            coverage_rows.append(
                {
                    "sample_type": sample_type,
                    "records": int(len(subset)),
                    "records_with_semantic_group": 0,
                    "semantic_group_count": 0,
                    "multi_dataset_group_count": 0,
                    "semantic_pair_count": 0,
                    "cross_dataset_pair_count": 0,
                    "cross_dataset_fraction": 0.0,
                }
            )
            continue
        groups = []
        semantic_pair_count = 0
        cross_pair_count = 0
        multi_dataset_group_count = 0
        for _, group in semantic_subset.groupby("current_semantic_group"):
            dataset_counts = group.groupby("dataset_id").size()
            total_pairs = int(len(group) * (len(group) - 1) / 2)
            same_dataset_pairs = int(sum(int(count * (count - 1) / 2) for count in dataset_counts))
            cross_pairs = max(total_pairs - same_dataset_pairs, 0)
            semantic_pair_count += total_pairs
            cross_pair_count += cross_pairs
            if int(group["dataset_id"].nunique()) >= 2:
                multi_dataset_group_count += 1
            groups.append(group["current_semantic_group"].iloc[0])
        coverage_rows.append(
            {
                "sample_type": sample_type,
                "records": int(len(subset)),
                "records_with_semantic_group": int(len(semantic_subset)),
                "semantic_group_count": int(len(groups)),
                "multi_dataset_group_count": int(multi_dataset_group_count),
                "semantic_pair_count": int(semantic_pair_count),
                "cross_dataset_pair_count": int(cross_pair_count),
                "cross_dataset_fraction": cross_pair_count / semantic_pair_count if semantic_pair_count else 0.0,
            }
        )

    report = textwrap.dedent(
        f"""
        Current semantic group report

        Source metadata:
        - records = {total_records}
        - records with semantic groups = {int((working['current_semantic_group'] != '').sum())}

        The current semantic-group layer is strong for grounding and sparse for serum/EV.
        Grounding uses molecule-based semantic groups and therefore already spans multiple datasets in a limited way.
        Serum and EV mostly use dataset-native class labels, so their current cross-dataset positive pool is effectively absent.

        Overall current semantic pair count = {overall_semantic_pairs}
        Overall current cross-dataset semantic pair count = {overall_cross_pairs}
        Overall current cross-dataset semantic pair fraction = {overall_cross_pairs / overall_semantic_pairs if overall_semantic_pairs else 0.0:.6f}
        """
    ).strip() + "\n"
    return pd.DataFrame(group_rows), pd.DataFrame(coverage_rows), report


def build_anchor_table(metadata_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in metadata_df.to_dict(orient="records"):
        harmonized_anchor, anchor_type, anchor_confidence, notes = choose_anchor(
            str(row["sample_type"]),
            str(row["dataset_id"]),
            str(row["label_optional"]),
            str(row.get("semantic_group", "")),
        )
        rows.append(
            {
                "sample_key": str(row["sample_key"]),
                "record_kind": str(row["record_kind"]),
                "sample_type": str(row["sample_type"]),
                "dataset_id": str(row["dataset_id"]),
                "original_label": str(row["label_optional"]),
                "current_semantic_group": str(row.get("semantic_group", "")),
                "proposed_harmonized_anchor": harmonized_anchor,
                "anchor_type": anchor_type,
                "anchor_confidence": anchor_confidence,
                "cross_dataset_usable": False,
                "notes": notes,
                "provenance": f"{row['dataset_id']}::{row['label_optional']}",
            }
        )
    anchor_df = pd.DataFrame(rows)
    coverage = (
        anchor_df[anchor_df["proposed_harmonized_anchor"] != ""]
        .groupby(["sample_type", "proposed_harmonized_anchor"])
        .agg(dataset_count=("dataset_id", "nunique"), record_count=("sample_key", "count"))
        .reset_index()
    )
    coverage_map = {
        (row["sample_type"], row["proposed_harmonized_anchor"]): int(row["dataset_count"])
        for row in coverage.to_dict(orient="records")
    }
    usable = []
    for row in anchor_df.to_dict(orient="records"):
        dataset_count = coverage_map.get((row["sample_type"], row["proposed_harmonized_anchor"]), 0)
        usable.append(
            bool(
                row["proposed_harmonized_anchor"]
                and dataset_count >= 2
                and row["anchor_confidence"] in {"high", "medium"}
                and row["anchor_type"] != "dataset_native_class_anchor"
            )
        )
    anchor_df["cross_dataset_usable"] = usable
    return anchor_df


def anchor_summary(anchor_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        anchor_df[anchor_df["proposed_harmonized_anchor"] != ""]
        .groupby(
            [
                "sample_type",
                "proposed_harmonized_anchor",
                "anchor_type",
                "anchor_confidence",
                "cross_dataset_usable",
            ]
        )
        .agg(record_count=("sample_key", "count"), dataset_count=("dataset_id", "nunique"))
        .reset_index()
        .sort_values(["sample_type", "cross_dataset_usable", "dataset_count", "record_count"], ascending=[True, False, False, False])
    )
    return summary


def cross_dataset_pairs_from_groups(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for sample_type, subset in df.groupby("sample_type"):
        grouped = subset[subset[group_col] != ""].groupby(group_col)
        total_pairs = 0
        cross_pairs = 0
        for _, group in grouped:
            dataset_counts = group.groupby("dataset_id").size()
            all_pairs = int(len(group) * (len(group) - 1) / 2)
            same_pairs = int(sum(int(count * (count - 1) / 2) for count in dataset_counts))
            total_pairs += all_pairs
            cross_pairs += max(all_pairs - same_pairs, 0)
        rows.append(
            {
                "sample_type": sample_type,
                "group_field": group_col,
                "pair_count_total": int(total_pairs),
                "pair_count_cross_dataset": int(cross_pairs),
                "cross_dataset_fraction": cross_pairs / total_pairs if total_pairs else 0.0,
            }
        )
    return pd.DataFrame(rows)


def training_readiness(anchor_df: pd.DataFrame, current_coverage_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    current_map = {
        row["sample_type"]: int(row["cross_dataset_pair_count"])
        for row in current_coverage_df.to_dict(orient="records")
    }
    current_fraction_map = {
        row["sample_type"]: float(row["cross_dataset_fraction"])
        for row in current_coverage_df.to_dict(orient="records")
    }
    usable_df = anchor_df[anchor_df["cross_dataset_usable"] & (anchor_df["proposed_harmonized_anchor"] != "")].copy()
    proposed_pairs_df = cross_dataset_pairs_from_groups(usable_df, "proposed_harmonized_anchor")
    coverage_rows = []
    for sample_type, subset in anchor_df.groupby("sample_type"):
        usable_subset = usable_df[usable_df["sample_type"] == sample_type]
        cross_pairs = 0
        cross_fraction = 0.0
        if not proposed_pairs_df[proposed_pairs_df["sample_type"] == sample_type].empty:
            row = proposed_pairs_df[proposed_pairs_df["sample_type"] == sample_type].iloc[0]
            cross_pairs = int(row["pair_count_cross_dataset"])
            cross_fraction = float(row["cross_dataset_fraction"])
        coverage_rows.append(
            {
                "sample_type": sample_type,
                "records": int(len(subset)),
                "records_with_anchor": int((subset["proposed_harmonized_anchor"] != "").sum()),
                "records_with_cross_dataset_anchor": int(len(usable_subset)),
                "anchor_coverage_fraction": len(usable_subset) / len(subset) if len(subset) else 0.0,
                "current_cross_dataset_pairs": int(current_map.get(sample_type, 0)),
                "proposed_cross_dataset_pairs": int(cross_pairs),
                "pair_count_increase": int(cross_pairs - current_map.get(sample_type, 0)),
                "current_cross_dataset_fraction": float(current_fraction_map.get(sample_type, 0.0)),
                "proposed_cross_dataset_fraction": float(cross_fraction),
                "connected_anchor_count": int(usable_subset["proposed_harmonized_anchor"].nunique()),
                "connected_dataset_count": int(usable_subset["dataset_id"].nunique()),
            }
        )
    readiness_df = pd.DataFrame(coverage_rows).sort_values("sample_type").reset_index(drop=True)
    report = textwrap.dedent(
        f"""
        Training readiness report

        This table compares the current semantic-positive pool against the proposed harmonized-anchor pool.
        The key question is whether future within-sample-type training can finally draw meaningful cross-dataset positives.

        {readiness_df.to_string(index=False)}

        Interpretation:
        - Serum should improve mainly through broad control-like, process-variation, and metabolite/redox-related anchors.
        - EV should improve primarily through disease_or_stress and control_or_baseline anchors spanning SHINE, diabetes, and part of small2023.
        - Grounding already had some cross-dataset bridge coverage; the new anchor layer mostly broadens it into theme families.
        """
    ).strip() + "\n"
    return readiness_df, report


def write_markdown(path: Path, title: str, paragraphs: list[str]) -> None:
    text = f"# {title}\n\n" + "\n\n".join(paragraphs).strip() + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.anchor_table_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_csv(args.metadata_path)
    group_df, current_coverage_df, current_report = current_semantic_group_tables(metadata_df)
    group_df.to_csv(args.output_dir / "current_semantic_group_coverage.csv", index=False)
    current_coverage_df.to_csv(args.output_dir / "current_cross_dataset_pair_coverage.csv", index=False)
    (args.output_dir / "current_semantic_group_report.md").write_text(current_report, encoding="utf-8")

    write_markdown(
        args.output_dir / "anchor_taxonomy.md",
        "Anchor Taxonomy",
        [
            "Anchor types used in this pass:",
            "- `dataset_native_class_anchor`: existing within-dataset classes retained for provenance and local structure, but not assumed to bridge datasets.",
            "- `cross_dataset_harmonized_biological_anchor`: cautious broad labels such as serum control-like, hepatobiliary-disease enriched, EV disease/stress, or EV control/baseline.",
            "- `grounding_linked_biochemical_theme`: broad biochemical bridges such as purine/metabolite, nucleic-acid, protein/peptide, lipid/membrane, carbohydrate, oxidative/redox, or serum-matrix associated.",
            "- `process_or_protocol_anchor`: processing or protocol variation labels retained to avoid pretending they are biological matches.",
            "Anchors are intentionally broad. This pass does not invent precise molecule labels for weakly grounded biosample classes.",
        ],
    )

    write_markdown(
        args.output_dir / "serum_anchor_harmonization.md",
        "Serum Anchor Harmonization",
        [
            "Serum can be harmonized only at broad biological or experimental levels. Direct disease-label equivalence across all serum datasets is not defensible.",
            "High-confidence serum bridges: `serum_control_like`, `serum_hepatobiliary_disease_enriched`, and `serum_inflammatory_or_infection_enriched` where cohort meaning is explicit.",
            "Medium-confidence serum bridges: `serum_metabolite_spiked_or_augmented` and broad biochemical-theme anchors for controlled serum component archives.",
            "Low-confidence serum anchors remain process- or protocol-oriented and are useful mainly to avoid false positives rather than to assert cross-dataset biology.",
        ],
    )

    write_markdown(
        args.output_dir / "ev_anchor_harmonization.md",
        "EV Anchor Harmonization",
        [
            "EV harmonization is narrower than serum but still feasible. The main defensible cross-dataset bridge is broad perturbation state rather than exact disease identity.",
            "Primary EV harmonized anchors: `ev_disease_or_stress` and `ev_control_or_baseline`.",
            "Small2023 contributes mostly broad baseline-like or composition-series anchors; SHINE and diabetes contribute the perturbation-heavy bridge.",
            "Cell-line-specific EV labels remain dataset-native and are not treated as strong cross-dataset positives.",
        ],
    )

    write_markdown(
        args.output_dir / "grounding_anchor_bridge_report.md",
        "Grounding Anchor Bridge Report",
        [
            "Grounding already provides the strongest cross-dataset bridge substrate because controlled references can be grouped into broad biochemical families.",
            "This pass uses theme families rather than hard molecule identity when building bridges: purine/metabolite, nucleic-acid, protein/peptide, lipid/membrane, carbohydrate, oxidative/redox, and serum-matrix associated.",
            "These bridges are intended for future training positives or prototypes, not for direct clinical interpretation.",
        ],
    )

    anchor_df = build_anchor_table(metadata_df)
    anchor_df.to_csv(args.anchor_table_path, index=False)
    summary_df = anchor_summary(anchor_df)
    summary_df.to_csv(args.output_dir / "embedding_anchor_summary.csv", index=False)
    write_markdown(
        args.output_dir / "embedding_anchor_report.md",
        "Embedding Anchor Report",
        [
            f"Anchor table records: {len(anchor_df)}",
            f"Records with proposed anchors: {int((anchor_df['proposed_harmonized_anchor'] != '').sum())}",
            f"Records with cross-dataset usable anchors: {int(anchor_df['cross_dataset_usable'].sum())}",
            "This table is intended as the first trainable harmonization layer for future v7 within-sample-type invariance runs.",
        ],
    )

    readiness_df, readiness_report = training_readiness(anchor_df, current_coverage_df)
    readiness_df.to_csv(args.output_dir / "training_readiness_metrics.csv", index=False)
    (args.output_dir / "training_readiness_report.md").write_text(readiness_report, encoding="utf-8")

    print(f"Saved anchor audit outputs to {args.output_dir}")
    print(f"Saved anchor table to {args.anchor_table_path}")


if __name__ == "__main__":
    main()
