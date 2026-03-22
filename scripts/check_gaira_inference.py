import json
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_storage_paths, require_data_root_exists

    storage_paths = require_data_root_exists()
    output_dir = get_storage_paths()["processed_data"] / "gaira_inference_v1"
    json_path = output_dir / "gaira_inference_demo_results.json"
    summary_path = output_dir / "gaira_inference_demo_summary.txt"

    if not json_path.exists():
        print(f"Missing inference results: {json_path}")
        return

    results = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"Inference result count: {len(results)}")
    for result in results:
        print()
        print(f"Query: {result['query_id']}")
        print(f"Domain pack: {result['domain_pack']}")
        print(f"Tier-1 hits: {len(result['tier1_grounding_hits'])}")
        print(f"Tier-2 hits: {len(result['tier2_support_hits'])}")
        print(f"Context hits: {len(result['domain_context_hits'])}")
        if result["tier1_grounding_hits"]:
            top = result["tier1_grounding_hits"][0]
            print(
                "Top tier-1: "
                f"{top['source_dataset_id']} / {top['source_label']} / "
                f"{float(top.get('reranked_score', top.get('score', 0.0))):.4f}"
            )
        if result["domain_context_hits"]:
            top_context = result["domain_context_hits"][0]
            print(
                "Top context: "
                f"{top_context['document_id']} / {top_context['section']} / {float(top_context['score']):.2f}"
            )

    print()
    print(f"Summary file exists: {summary_path.exists()}")


if __name__ == "__main__":
    main()
