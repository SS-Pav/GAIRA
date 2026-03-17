from pathlib import Path

import pandas as pd


def main() -> None:
    output_dir = Path("/Volumes/SSD_SPG/GAIRA_DATA/processed/gaira_grounding_search_v1")
    demo_path = output_dir / "grounding_search_demo_results.csv"
    band_path = output_dir / "grounding_band_query_results.csv"
    summary_path = output_dir / "grounding_tiered_evidence_examples.txt"

    if not demo_path.exists():
        print(f"Missing demo results: {demo_path}")
        return

    demo_df = pd.read_csv(demo_path)
    band_df = pd.read_csv(band_path) if band_path.exists() and band_path.stat().st_size > 0 else pd.DataFrame()

    print(f"Demo results rows: {len(demo_df)}")
    if not demo_df.empty:
        print()
        print("Results by evidence_tier and result_type:")
        print(
            demo_df.groupby(["evidence_tier", "result_type"])
            .size()
            .reset_index(name="n")
            .sort_values(["evidence_tier", "result_type"])
            .to_string(index=False)
        )

        print()
        print("Top results preview:")
        preview_cols = [
            "query_id",
            "evidence_tier",
            "result_type",
            "source_dataset_id",
            "source_label",
            "score",
        ]
        print(demo_df[preview_cols].head(20).to_string(index=False))

    print()
    print(f"Band query rows: {len(band_df)}")
    if not band_df.empty:
        print(
            band_df.groupby(["query_band_cm", "evidence_tier", "result_type"])
            .size()
            .reset_index(name="n")
            .sort_values(["query_band_cm", "evidence_tier", "result_type"])
            .to_string(index=False)
        )

    print()
    print(f"Summary text exists: {summary_path.exists()}")


if __name__ == "__main__":
    main()
