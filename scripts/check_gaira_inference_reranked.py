import json
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_storage_paths, require_data_root_exists

    storage_paths = require_data_root_exists()
    output_dir = storage_paths["processed_data"] / "gaira_inference_v1_1"
    json_path = output_dir / "gaira_inference_reranked_results.json"
    comparison_path = output_dir / "gaira_inference_before_after_comparison.csv"

    if not json_path.exists():
        print(f"Missing reranked inference results: {json_path}")
        return

    results = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"Reranked inference result count: {len(results)}")
    for result in results:
        print()
        print(f"Query: {result['query_id']}")
        print(f"Domain pack: {result['domain_pack']}")
        before_tier1 = result.get("tier1_grounding_hits_before_reranking", [])
        after_tier1 = result.get("tier1_grounding_hits", [])
        before_tier2 = result.get("tier2_support_hits_before_reranking", [])
        after_tier2 = result.get("tier2_support_hits", [])
        if before_tier1:
            top = before_tier1[0]
            print(
                "Top tier-1 before: "
                f"{top['source_dataset_id']} / {top['source_label']} / {float(top['score']):.4f}"
            )
        if after_tier1:
            top = after_tier1[0]
            print(
                "Top tier-1 after: "
                f"{top['source_dataset_id']} / {top['source_label']} / "
                f"{float(top['reranked_score']):.4f} (w={float(top['domain_relevance_weight']):.2f})"
            )
        if before_tier2:
            top = before_tier2[0]
            print(
                "Top tier-2 before: "
                f"{top['source_dataset_id']} / {top['source_label']} / {float(top['score']):.4f}"
            )
        if after_tier2:
            top = after_tier2[0]
            print(
                "Top tier-2 after: "
                f"{top['source_dataset_id']} / {top['source_label']} / "
                f"{float(top['reranked_score']):.4f} (w={float(top['domain_relevance_weight']):.2f})"
            )

    print()
    print(f"Comparison CSV exists: {comparison_path.exists()}")


if __name__ == "__main__":
    main()
