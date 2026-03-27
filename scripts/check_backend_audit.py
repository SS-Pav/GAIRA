from __future__ import annotations

from pathlib import Path


REQUIRED_FILES = [
    "ramanbiolib_registry_check.md",
    "ramanbiolib_db_counts.csv",
    "ramanbiolib_db_summary.md",
    "ramanbiolib_inference_visibility.csv",
    "ramanbiolib_inference_summary.md",
    "grounding_search_audit.md",
    "ramanbiolib_final_assessment.md",
    "cca_baseline_diagnostic.md",
    "cca_baseline_metrics.csv",
    "cca_baseline_before_after_comparison.csv",
    "cca_baseline_before_after_report.md",
    "cca_holdout_baseline_impact.csv",
    "cca_holdout_baseline_impact.md",
    "backend_audit_summary.md",
]


def main() -> None:
    base_dir = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/backend_audit")
    missing = [name for name in REQUIRED_FILES if not (base_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing backend audit outputs: {missing}")
    print("Backend audit outputs verified.")


if __name__ == "__main__":
    main()
