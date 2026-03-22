import argparse
import sys
from pathlib import Path


def main() -> None:
    # Make the src package importable when running from the project root.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_database_path
    from gaira.search import PeakSearchEngine

    parser = argparse.ArgumentParser(description="Run GAIRA peak-based Raman search.")
    parser.add_argument("query_csv", help="Path to the query CSV spectrum")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top candidates to print")
    parser.add_argument("--tolerance", type=float, default=5.0, help="Peak tolerance in cm-1")
    parser.add_argument("--prominence", type=float, default=0.03, help="Peak prominence threshold")
    parser.add_argument("--height", type=float, default=0.03, help="Peak height threshold")
    args = parser.parse_args()

    db_path = get_database_path()
    engine = PeakSearchEngine(
        db_path=db_path,
        peak_tolerance_cm=args.tolerance,
        min_peak_prominence=args.prominence,
        min_peak_height=args.height,
    )

    results = engine.run(Path(args.query_csv))

    print("Detected query peaks:")
    print(results["query_peaks_df"].to_string(index=False))

    print(f"\nTop ref_id-level candidates (top {args.top_n}):")
    top_ref_candidates = results["candidate_df_ref"].head(args.top_n)
    if top_ref_candidates.empty:
        print("No ref_id-level candidates found.")
    else:
        print(top_ref_candidates.to_string(index=False))

    print(f"\nTop spectral similarity matches (top {args.top_n}):")
    top_similarity_candidates = results["candidate_df_similarity"].head(args.top_n)
    if top_similarity_candidates.empty:
        print("No spectral similarity matches found.")
    else:
        print(top_similarity_candidates.to_string(index=False))

    print(f"\nFinal reranked candidates (top {args.top_n}):")
    top_reranked_candidates = results["candidate_df_reranked"].head(args.top_n)
    if top_reranked_candidates.empty:
        print("No reranked candidates found.")
    else:
        print(top_reranked_candidates.to_string(index=False))

    print(f"\nTop reranked component-level candidates (top {args.top_n}):")
    top_reranked_components = results["candidate_df_component_reranked"].head(args.top_n)
    if top_reranked_components.empty:
        print("No reranked component-level candidates found.")
    else:
        print(top_reranked_components.to_string(index=False))

    print("\nTop reranked biochemical classes:")
    top_classes = results["class_df_reranked"].head(args.top_n)
    if top_classes.empty:
        print("No biochemical class summary available.")
    else:
        print(top_classes.to_string(index=False))

    print("\nInterpretation:")
    print(results["interpretation_text"])


if __name__ == "__main__":
    main()
