from pathlib import Path
import sys

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_storage_paths, require_data_root_exists

    storage_paths = require_data_root_exists()
    base_dir = storage_paths["processed_data"] / "ev_family_consolidation"

    required = [
        base_dir / "ev_family_audit.md",
        base_dir / "ev_family_design.md",
        base_dir / "tables" / "ev_family_before_after_metrics.csv",
        base_dir / "tables" / "ev_forced_routing_summary.csv",
        base_dir / "tables" / "ev_winner_margin_summary.csv",
        base_dir / "tables" / "no_regression_summary.csv",
        base_dir / "tables" / "final_family_best_routing_summary.csv",
        base_dir / "tables" / "representative_case_table.csv",
        base_dir / "report" / "ev_family_consolidation_report.md",
        base_dir / "report" / "ev_family_consolidation_report.pdf",
        base_dir / "figures" / "figure1_ev_family_consolidation_schematic.png",
        base_dir / "figures" / "figure2_query_family_forced_routing_heatmap.png",
        base_dir / "figures" / "figure3_ev_routing_before_after.png",
        base_dir / "figures" / "figure4_near_family_and_contamination_summary.png",
        base_dir / "figures" / "figure5_winner_runner_up_margin.png",
        base_dir / "figures" / "figure6_no_regression_summary.png",
    ]
    for path in required:
        print(f"{path.name}: {path.exists()}")

    best_path = base_dir / "tables" / "final_family_best_routing_summary.csv"
    if not best_path.exists():
        raise SystemExit("Missing final family best-routing summary.")
    best_df = pd.read_csv(best_path)

    expected = {
        "ev_general": "ev_general",
        "ev_disease_or_stress": "ev_disease_or_stress",
        "serum_general": "serum_general",
        "serum_liver_hepatobiliary": "serum_liver_hepatobiliary",
        "grounding_analyte": "grounding_analyte",
    }
    for intended_family, best_forced in expected.items():
        subset = best_df[best_df["intended_family"] == intended_family]
        if subset.empty:
            raise SystemExit(f"Missing intended family: {intended_family}")
        observed = str(subset.iloc[0]["best_forced_family"])
        print(f"{intended_family}: {observed}")
        if observed != best_forced:
            raise SystemExit(f"Unexpected best family for {intended_family}: {observed}")

    ev_before_after_path = base_dir / "tables" / "ev_family_before_after_metrics.csv"
    ev_before_after_df = pd.read_csv(ev_before_after_path)
    required_datasets = {"small2023_ev", "diabetes_plasma_ev_sers", "shine_ev_sers"}
    observed_datasets = set(ev_before_after_df["dataset_label"].dropna().astype(str))
    missing = sorted(required_datasets - observed_datasets)
    if missing:
        raise SystemExit(f"Missing EV datasets from before/after summary: {', '.join(missing)}")


if __name__ == "__main__":
    main()
