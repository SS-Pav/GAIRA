import sys
from pathlib import Path


def main() -> None:
    # Make the src package importable when running from the project root.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_database_path
    from gaira.search import PeakSearchEngine

    db_path = get_database_path()
    query_path = project_root / "data" / "processed" / "test_queries" / "collagen_example_query.csv"

    engine = PeakSearchEngine(db_path=db_path)
    results = engine.run(query_path)

    print(f"Number of detected peaks: {len(results['query_peaks_df'])}")
    print("\nTop 10 ref_id-level candidates:")
    print(results["candidate_df_ref"].head(10).to_string(index=False))
    print("\nTop 10 spectral similarity matches:")
    print(results["candidate_df_similarity"].head(10).to_string(index=False))
    print("\nTop 10 final reranked candidates:")
    print(results["candidate_df_reranked"].head(10).to_string(index=False))
    print("\nTop 10 reranked component-level candidates:")
    print(results["candidate_df_component_reranked"].head(10).to_string(index=False))
    print("\nTop 5 reranked biochemical classes:")
    print(results["class_df_reranked"].head(5).to_string(index=False))
    print("\nInterpretation:")
    print(results["interpretation_text"])


if __name__ == "__main__":
    main()
