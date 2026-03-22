import json
import sys
from pathlib import Path

import pandas as pd


def find_component_rank(component_df: pd.DataFrame, expected_component: str) -> int | None:
    """Return the 1-based rank of the expected component, if present."""
    if not expected_component:
        return None

    matches = component_df.index[
        component_df["component"].astype(str).str.casefold() == expected_component.casefold()
    ].tolist()
    return int(matches[0] + 1) if matches else None


def find_ref_best_rank(ref_df: pd.DataFrame, expected_component: str) -> int | None:
    """Return the first reranked ref row whose component matches the expected one."""
    if not expected_component:
        return None

    matches = ref_df.index[
        ref_df["component"].astype(str).str.casefold() == expected_component.casefold()
    ].tolist()
    return int(matches[0] + 1) if matches else None


def find_class_rank(class_df: pd.DataFrame, expected_class: str) -> int | None:
    """Return the 1-based rank of the expected biochemical class, if present."""
    if not expected_class:
        return None

    matches = class_df.index[
        class_df["biochemical_class"].astype(str).str.casefold() == expected_class.casefold()
    ].tolist()
    return int(matches[0] + 1) if matches else None


def main() -> None:
    # Make the src package importable when running from the project root.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_database_path
    from gaira.search import PeakSearchEngine

    manifest_path = project_root / "data" / "processed" / "eval_queries" / "eval_manifest.csv"
    results_csv_path = project_root / "data" / "processed" / "eval_results" / "eval_results.csv"
    results_json_path = project_root / "data" / "processed" / "eval_results" / "eval_results.json"
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Evaluation manifest not found: {manifest_path}. Run scripts/build_eval_manifest.py first."
        )

    manifest_df = pd.read_csv(manifest_path)
    if manifest_df.empty:
        raise ValueError("The evaluation manifest is empty.")

    engine = PeakSearchEngine(db_path=get_database_path())
    eval_rows: list[dict] = []

    for row in manifest_df.itertuples(index=False):
        query_path = project_root / str(row.query_path)
        print(f"Evaluating {row.query_id} from {row.query_path}")
        results = engine.run(query_path)

        ref_df = results["candidate_df_reranked"].copy()
        component_df = results["candidate_df_component_reranked"].copy()
        class_df = results["class_df_reranked"].copy()

        eval_rows.append(
            {
                "query_id": row.query_id,
                "query_path": row.query_path,
                "detected_peak_count": int(len(results["query_peaks_df"])),
                "expected_component": row.expected_component,
                "expected_class": row.expected_class,
                "ref_rank_of_expected_component_best_match": find_ref_best_rank(
                    ref_df,
                    str(row.expected_component),
                ),
                "component_rank_of_expected_component": find_component_rank(
                    component_df,
                    str(row.expected_component),
                ),
                "expected_class_rank": find_class_rank(class_df, str(row.expected_class)),
                "top_5_ref_candidates": json.dumps(
                    ref_df["component"].head(5).tolist(),
                    ensure_ascii=True,
                ),
                "top_5_component_candidates": json.dumps(
                    component_df["component"].head(5).tolist(),
                    ensure_ascii=True,
                ),
                "top_5_classes": json.dumps(
                    class_df["biochemical_class"].head(5).tolist(),
                    ensure_ascii=True,
                ),
            }
        )

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(results_csv_path, index=False)

    json_summary = {
        "query_count": int(len(eval_df)),
        "results": eval_rows,
    }
    results_json_path.write_text(json.dumps(json_summary, indent=2), encoding="utf-8")

    print(f"Saved evaluation CSV to {results_csv_path.relative_to(project_root)}")
    print(f"Saved evaluation JSON to {results_json_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()
