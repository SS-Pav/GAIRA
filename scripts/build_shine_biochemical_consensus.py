import re
import sys
from pathlib import Path

import pandas as pd


REGION_BUCKETS = [
    (450, 700, "450-700"),
    (700, 900, "700-900"),
    (900, 1100, "900-1100"),
    (1100, 1300, "1100-1300"),
    (1300, 1500, "1300-1500"),
    (1500, 1700, "1500-1700"),
    (1700, 1800, "1700-1800"),
]


def sanitize_label(value: str | None) -> str:
    """Convert class labels into safe file-name fragments."""
    if value is None or str(value).strip() == "":
        return "unknown"
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()


def assign_region_bucket(peak_cm: float) -> str:
    """Map a peak position to a broad Raman region bucket."""
    for lower, upper, label in REGION_BUCKETS:
        if lower <= peak_cm < upper or (label == "1700-1800" and peak_cm <= upper):
            return label
    return "outside_window"


def build_support_score(df: pd.DataFrame) -> pd.Series:
    """Compute a simple cumulative support score from the stored peak match evidence."""
    return (
        df["query_peak_intensity"].astype(float)
        * df["rel_intensity"].astype(float)
        * df["idf_weight"].astype(float)
        / (1.0 + df["peak_delta_cm"].astype(float))
    )


def top_values_as_text(df: pd.DataFrame, label_column: str, score_column: str, top_n: int) -> str:
    """Format top labels as a short semicolon-separated string."""
    if df.empty:
        return ""
    top_df = df.sort_values(score_column, ascending=False).head(top_n)
    return "; ".join(str(value) for value in top_df[label_column].tolist())


def build_interpretation(
    top_classes_df: pd.DataFrame,
    top_components_df: pd.DataFrame,
    top_regions_df: pd.DataFrame,
    suspicious_flag: bool,
) -> str:
    """Create a conservative human-readable interpretation."""
    if top_classes_df.empty:
        return "No consistent biochemical consensus could be derived from the class-mean matches."

    top_class_names = [str(value) for value in top_classes_df["biochemical_class"].head(2).tolist()]
    top_region_names = [str(value) for value in top_regions_df["region_bucket"].head(2).tolist()]
    component_examples = [str(value) for value in top_components_df["component"].head(3).tolist()]

    text = (
        f"{' / '.join(top_class_names)} consensus with recurring support in the "
        f"{' and '.join(top_region_names)} cm^-1 regions."
    )
    if component_examples:
        text += (
            f" Example analog references include {', '.join(component_examples)}; "
            "these should not be treated as literal molecule IDs."
        )
    if suspicious_flag:
        text += (
            " Exact top molecule labels look unstable relative to the broader class-level evidence, "
            "so this class should be treated cautiously."
        )
    return text


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_storage_config, resolve_storage_path

    storage_config = get_storage_config()
    processed_root = resolve_storage_path(storage_config.get("processed_data"))

    if processed_root is None:
        print("The storage config is missing processed_data.")
        return

    matches_dir = processed_root / "shine_class_reference_matches"
    if not matches_dir.exists():
        print(f"Match output folder not found: {matches_dir}")
        return

    match_files = sorted(matches_dir.glob("class_*_matches.csv"))
    if not match_files:
        print(f"No class match files were found in: {matches_dir}")
        return

    summary_rows: list[dict] = []

    for match_file in match_files:
        peaks_file = match_file.with_name(match_file.name.replace("_matches.csv", "_peaks.csv"))
        if not peaks_file.exists():
            print(f"Skipping {match_file.name}: matching peak file not found.")
            continue

        matches_df = pd.read_csv(match_file)
        peaks_df = pd.read_csv(peaks_file)
        if matches_df.empty or peaks_df.empty:
            print(f"Skipping {match_file.name}: match or peak table is empty.")
            continue

        class_label = str(matches_df.iloc[0]["class_label"])
        subclass_label = str(matches_df.iloc[0]["subclass_label"])
        n_spectra = int(matches_df.iloc[0]["n_spectra"])

        peaks_df["support_score"] = build_support_score(peaks_df)
        peaks_df["region_bucket"] = peaks_df["query_peak_cm"].astype(float).apply(assign_region_bucket)

        top_classes_df = (
            peaks_df.groupby("biochemical_class", dropna=False)["support_score"]
            .sum()
            .reset_index()
            .sort_values("support_score", ascending=False)
            .reset_index(drop=True)
        )
        top_components_df = (
            matches_df.groupby("component", dropna=False)["final_score"]
            .sum()
            .reset_index()
            .sort_values("final_score", ascending=False)
            .reset_index(drop=True)
        )
        top_regions_df = (
            peaks_df.groupby("region_bucket", dropna=False)["support_score"]
            .sum()
            .reset_index()
            .sort_values("support_score", ascending=False)
            .reset_index(drop=True)
        )

        top_class_score = float(top_classes_df.iloc[0]["support_score"])
        second_class_score = float(top_classes_df.iloc[1]["support_score"]) if len(top_classes_df) > 1 else 0.0
        top_match_score = float(matches_df.iloc[0]["final_score"])
        second_match_score = float(matches_df.iloc[1]["final_score"]) if len(matches_df) > 1 else 0.0
        suspicious_flag = bool(
            (top_match_score >= second_match_score + 0.08)
            and (second_class_score > 0)
            and (top_class_score / second_class_score < 1.35)
        )

        interpretation_text = build_interpretation(
            top_classes_df=top_classes_df,
            top_components_df=top_components_df,
            top_regions_df=top_regions_df,
            suspicious_flag=suspicious_flag,
        )

        consensus_row = {
            "class_label": class_label,
            "subclass_label": subclass_label,
            "n_spectra": n_spectra,
            "top_biochemical_class_1": top_classes_df.iloc[0]["biochemical_class"] if len(top_classes_df) > 0 else None,
            "top_biochemical_class_2": top_classes_df.iloc[1]["biochemical_class"] if len(top_classes_df) > 1 else None,
            "top_component_examples": top_values_as_text(top_components_df, "component", "final_score", 3),
            "dominant_region_1": top_regions_df.iloc[0]["region_bucket"] if len(top_regions_df) > 0 else None,
            "dominant_region_2": top_regions_df.iloc[1]["region_bucket"] if len(top_regions_df) > 1 else None,
            "top_biochemical_classes_by_score": top_values_as_text(top_classes_df, "biochemical_class", "support_score", 5),
            "top_components_by_score": top_values_as_text(top_components_df, "component", "final_score", 5),
            "top_regions_by_score": top_values_as_text(top_regions_df, "region_bucket", "support_score", 5),
            "suspicious_flag": suspicious_flag,
            "interpretation_text": interpretation_text,
        }

        consensus_df = pd.DataFrame([consensus_row])
        output_path = matches_dir / (
            f"class_{sanitize_label(class_label)}_{sanitize_label(subclass_label)}_consensus.csv"
        )
        consensus_df.to_csv(output_path, index=False)
        summary_rows.append(consensus_row)

    if not summary_rows:
        print("No class consensus rows were created.")
        return

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["class_label", "subclass_label"]
    ).reset_index(drop=True)
    summary_path = matches_dir / "shine_class_consensus_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Created class consensus rows: {len(summary_df)}")
    print(f"Summary written to: {summary_path}")
    print("\nPer-class interpretations:")
    for row in summary_df.itertuples(index=False):
        print(f"- {row.class_label} ({row.subclass_label}): {row.interpretation_text}")


if __name__ == "__main__":
    main()
