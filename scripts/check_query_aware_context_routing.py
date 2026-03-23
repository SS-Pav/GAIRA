from pathlib import Path
import sys

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_storage_paths, require_data_root_exists

    storage_paths = require_data_root_exists()
    base_dir = storage_paths["processed_data"] / "query_aware_context_routing"

    required = [
        base_dir / "query_family_design.md",
        base_dir / "tables" / "before_after_context_ranks.csv",
        base_dir / "tables" / "before_after_support_ranks.csv",
        base_dir / "tables" / "family_matched_hit_counts.csv",
        base_dir / "tables" / "cross_domain_contamination_summary.csv",
        base_dir / "tables" / "representative_case_table.csv",
        base_dir / "tables" / "track_improvement_summary.csv",
        base_dir / "tables" / "forced_routing_comparison.csv",
        base_dir / "tables" / "forced_routing_summary.csv",
        base_dir / "tables" / "routing_usefulness_by_family.csv",
        base_dir / "report" / "query_aware_context_routing_report.md",
        base_dir / "report" / "query_aware_context_routing_report.pdf",
        base_dir / "figures" / "figure1_query_family_routing_schematic.png",
        base_dir / "figures" / "figure7_final_routing_usefulness_summary.png",
        base_dir / "figures" / "figure8_counterfactual_routing_comparison.png",
    ]
    for path in required:
        print(f"{path.name}: {path.exists()}")

    track_path = base_dir / "tables" / "track_improvement_summary.csv"
    forced_path = base_dir / "tables" / "forced_routing_summary.csv"
    if not track_path.exists() or not forced_path.exists():
        raise SystemExit("Missing routing summary tables.")

    track_df = pd.read_csv(track_path)
    forced_df = pd.read_csv(forced_path)

    required_tracks = {
        "hepatobiliary_serum_routing",
        "general_serum_routing",
        "ev_routing",
        "analyte_routing",
    }
    missing_tracks = sorted(required_tracks - set(track_df["track_name"].dropna().astype(str)))
    if missing_tracks:
        raise SystemExit(f"Missing routing tracks: {', '.join(missing_tracks)}")

    print("\nTrack improvement summary:")
    print(track_df.to_string(index=False))

    print("\nForced routing summary head:")
    print(forced_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
