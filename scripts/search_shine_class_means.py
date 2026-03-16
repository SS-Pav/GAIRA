import json
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd


DATASET_ID = "shine_ev_sers"
PROCESSING_VERSION = "v1_crop450_1800_interp1_minmax"


def sanitize_label(value: str | None) -> str:
    """Convert labels into safe file-name fragments."""
    if value is None or str(value).strip() == "":
        return "unknown"
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()


def load_class_summaries(db_path: Path) -> pd.DataFrame:
    """Load processed SHINE class-mean spectra from DuckDB."""
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return connection.execute(
            """
            SELECT
                summary_id,
                class_label,
                subclass_label,
                n_spectra,
                mean_wavenumbers_json,
                mean_intensity_json
            FROM biosample_class_summary
            WHERE dataset_id = ?
              AND processing_version = ?
            ORDER BY class_label, subclass_label
            """,
            [DATASET_ID, PROCESSING_VERSION],
        ).fetchdf()


def build_mean_spectrum_df(row: pd.Series) -> pd.DataFrame:
    """Reconstruct one class-mean spectrum from the stored JSON arrays."""
    wavenumbers = json.loads(row["mean_wavenumbers_json"])
    intensities = json.loads(row["mean_intensity_json"])

    if len(wavenumbers) != len(intensities):
        raise ValueError(
            f"Class summary {row['summary_id']} has mismatched mean spectrum array lengths."
        )

    spectrum_df = pd.DataFrame(
        {
            "wavenumber": pd.to_numeric(wavenumbers),
            "intensity": pd.to_numeric(intensities),
        }
    )
    return spectrum_df.sort_values("wavenumber").reset_index(drop=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import ensure_storage_dirs, resolve_storage_path
    from gaira.search import PeakSearchEngine

    storage_config = ensure_storage_dirs()
    processed_root = resolve_storage_path(storage_config.get("processed_data"))
    db_path = project_root / "data" / "gaira.duckdb"

    if processed_root is None:
        print("The storage config is missing processed_data.")
        return

    output_dir = processed_root / "shine_class_reference_matches"
    output_dir.mkdir(parents=True, exist_ok=True)

    class_summaries_df = load_class_summaries(db_path)
    if class_summaries_df.empty:
        print("No SHINE class summaries were found. Run process_biosample_dataset.py first.")
        return

    engine = PeakSearchEngine(db_path=db_path)
    processed_count = 0

    print(f"Writing class-level reference matches to: {output_dir}")

    for row in class_summaries_df.to_dict(orient="records"):
        row_series = pd.Series(row)
        class_label = row_series["class_label"]
        subclass_label = row_series["subclass_label"]
        n_spectra = int(row_series["n_spectra"])

        spectrum_df = build_mean_spectrum_df(row_series)
        normalized_df = engine.normalize_query_spectrum(spectrum_df)
        peaks_df = engine.detect_query_peaks(normalized_df)
        peak_matches_df = engine.search_reference_peaks(peaks_df)
        candidate_df_ref, _ = engine.score_candidates(peak_matches_df, peaks_df)
        candidate_df_similarity, candidate_df_reranked = engine.rerank_with_spectral_similarity(
            normalized_df,
            candidate_df_ref,
        )
        candidate_df_component = engine.aggregate_reranked_components(candidate_df_reranked)
        class_df = engine.summarize_classes(candidate_df_component)

        file_stub = f"class_{sanitize_label(class_label)}_{sanitize_label(subclass_label)}"
        matches_output_path = output_dir / f"{file_stub}_matches.csv"
        peaks_output_path = output_dir / f"{file_stub}_peaks.csv"

        peak_matches_for_export = peak_matches_df.copy()
        if not peak_matches_for_export.empty:
            peak_matches_for_export = peak_matches_for_export.sort_values(
                ["query_peak_cm", "peak_delta_cm", "idf_weight"],
                ascending=[True, True, False],
            ).reset_index(drop=True)

        top_matches_df = candidate_df_reranked.head(10).copy()
        if not top_matches_df.empty:
            top_matches_df.insert(0, "class_label", class_label)
            top_matches_df.insert(1, "subclass_label", subclass_label)
            top_matches_df.insert(2, "n_spectra", n_spectra)

        top_matches_df.to_csv(matches_output_path, index=False)
        peak_matches_for_export.to_csv(peaks_output_path, index=False)

        processed_count += 1

        print("-------------------------------------")
        print(f"Class: {class_label} ({subclass_label})")
        print(f"Spectra used: {n_spectra}")
        print("")
        print("Top molecules")
        if top_matches_df.empty:
            print("No molecule matches found.")
        else:
            for rank, match_row in enumerate(top_matches_df.itertuples(index=False), start=1):
                print(f"{rank} {match_row.component}")
        print("")
        print("Top biochemical classes")
        if class_df.empty:
            print("No biochemical class matches found.")
        else:
            for rank, class_row in enumerate(class_df.head(5).itertuples(index=False), start=1):
                print(f"{rank} {class_row.biochemical_class}")
        print("-------------------------------------")

    print(f"Processed class spectra: {processed_count}")


if __name__ == "__main__":
    main()
