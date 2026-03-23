from pathlib import Path
import sys

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_storage_paths, require_data_root_exists

    storage_paths = require_data_root_exists()
    base_dir = storage_paths["processed_data"] / "hcc_holdout_calibration"
    required = [
        base_dir / "raw_outputs" / "hcc_holdout_calibrated_theme_outputs_long.csv",
        base_dir / "raw_outputs" / "hcc_holdout_calibrated_sample_summary.csv",
        base_dir / "tables" / "hcc_holdout_calibration_before_after_metrics.csv",
        base_dir / "tables" / "hcc_holdout_shared_background_summary.csv",
        base_dir / "tables" / "hcc_holdout_differential_evidence_summary.csv",
        base_dir / "report" / "hcc_holdout_calibration_report.md",
        base_dir / "report" / "hcc_holdout_calibration_report.pdf",
        base_dir / "figures" / "figure1_serum_differential_calibration_design.png",
        base_dir / "figures" / "figure8_calibration_usefulness_summary.png",
    ]
    for path in required:
        print(f"{path.name}: {path.exists()}")

    metric_path = base_dir / "tables" / "hcc_holdout_calibration_before_after_metrics.csv"
    if metric_path.exists():
        metric_df = pd.read_csv(metric_path)
        print("Before/after metrics:")
        print(metric_df.to_string(index=False))


if __name__ == "__main__":
    main()
