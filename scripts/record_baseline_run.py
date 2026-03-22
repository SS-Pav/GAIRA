import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    # Make the src package importable when running from the project root.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_database_path
    from gaira.search import PeakSearchEngine

    query_path = project_root / "data" / "processed" / "test_queries" / "collagen_example_query.csv"
    output_path = (
        project_root
        / "data"
        / "processed"
        / "baseline_runs"
        / "collagen_example_baseline.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    engine = PeakSearchEngine(db_path=get_database_path())
    results = engine.run(query_path)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_path": str(query_path.relative_to(project_root)),
        "detected_peak_count": int(len(results["query_peaks_df"])),
        "top_10_reranked_ref_candidates": results["candidate_df_reranked"]
        .head(10)
        .to_dict(orient="records"),
        "top_10_reranked_component_candidates": results["candidate_df_component_reranked"]
        .head(10)
        .to_dict(orient="records"),
        "top_5_reranked_biochemical_classes": results["class_df_reranked"]
        .head(5)
        .to_dict(orient="records"),
        "interpretation_text": results["interpretation_text"],
    }

    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Recorded baseline run to {output_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()
