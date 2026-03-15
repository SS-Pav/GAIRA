from pathlib import Path

import pandas as pd


def count_in_top(series: pd.Series, cutoff: int) -> int:
    """Count how many non-null ranks fall within a cutoff."""
    numeric_series = pd.to_numeric(series, errors="coerce")
    return int((numeric_series <= cutoff).fillna(False).sum())


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_path = project_root / "data" / "processed" / "eval_results" / "eval_results.csv"

    if not results_path.exists():
        raise FileNotFoundError(
            f"Evaluation results not found: {results_path}. Run scripts/run_eval_suite.py first."
        )

    results_df = pd.read_csv(results_path)
    if results_df.empty:
        raise ValueError("The evaluation results file is empty.")

    component_rank_series = pd.to_numeric(
        results_df["component_rank_of_expected_component"],
        errors="coerce",
    )
    ref_rank_series = pd.to_numeric(
        results_df["ref_rank_of_expected_component_best_match"],
        errors="coerce",
    )

    print(f"Queries evaluated: {len(results_df)}")
    print(f"Median component rank: {component_rank_series.median()}")
    print(f"Median ref best-match rank: {ref_rank_series.median()}")
    print("")
    print(f"Component in top 1: {count_in_top(component_rank_series, 1)}")
    print(f"Component in top 3: {count_in_top(component_rank_series, 3)}")
    print(f"Component in top 5: {count_in_top(component_rank_series, 5)}")
    print(f"Ref best-match in top 1: {count_in_top(ref_rank_series, 1)}")
    print(f"Ref best-match in top 3: {count_in_top(ref_rank_series, 3)}")
    print(f"Ref best-match in top 5: {count_in_top(ref_rank_series, 5)}")


if __name__ == "__main__":
    main()
