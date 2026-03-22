import csv
import json
import sys
from pathlib import Path


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def build_before_after_rows(result: dict, tier_key_before: str, tier_key_after: str, tier_name: str) -> list[dict]:
    rows = []
    before_hits = result.get(tier_key_before, [])
    after_hits = result.get(tier_key_after, [])

    for rank, row in enumerate(before_hits[:5], start=1):
        rows.append(
            {
                "query_id": result["query_id"],
                "domain_pack": result["domain_pack"],
                "tier": tier_name,
                "stage": "before_reranking",
                "rank": rank,
                "source_dataset_id": row.get("source_dataset_id"),
                "source_label": row.get("source_label"),
                "result_type": row.get("result_type"),
                "base_score": row.get("score"),
                "domain_relevance_weight": "",
                "reranked_score": "",
                "rerank_reason": "",
            }
        )

    for rank, row in enumerate(after_hits[:5], start=1):
        rows.append(
            {
                "query_id": result["query_id"],
                "domain_pack": result["domain_pack"],
                "tier": tier_name,
                "stage": "after_reranking",
                "rank": rank,
                "source_dataset_id": row.get("source_dataset_id"),
                "source_label": row.get("source_label"),
                "result_type": row.get("result_type"),
                "base_score": row.get("base_score"),
                "domain_relevance_weight": row.get("domain_relevance_weight"),
                "reranked_score": row.get("reranked_score"),
                "rerank_reason": row.get("rerank_reason"),
            }
        )
    return rows


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_database_path, get_storage_paths, require_data_root_exists
    from gaira.inference import (
        GAIRAInferenceEngine,
        load_ev_class_mean_query,
        load_serum_class_mean_query,
    )

    storage_paths = require_data_root_exists()
    db_path = get_database_path()
    output_dir = storage_paths["processed_data"] / "gaira_inference_v1_1"
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = GAIRAInferenceEngine(db_path=db_path)
    requests = [
        load_serum_class_mean_query(
            db_path=db_path,
            dataset_id="serum_protocol_comparison",
            class_label="p1",
            subclass_label="protocol_comparison_archive",
        ),
        load_serum_class_mean_query(
            db_path=db_path,
            dataset_id="serum_protocol_comparison",
            class_label="p5",
            subclass_label="protocol_comparison_archive",
        ),
        load_ev_class_mean_query(
            db_path=db_path,
            dataset_id="small2023_ev",
            class_label="c00",
            subclass_label="normedprobe1",
        ),
    ]

    results = [engine.run_inference(request) for request in requests]
    (output_dir / "gaira_inference_reranked_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    comparison_rows = []
    for result in results:
        comparison_rows.extend(
            build_before_after_rows(
                result,
                "tier1_grounding_hits_before_reranking",
                "tier1_grounding_hits",
                "tier1",
            )
        )
        comparison_rows.extend(
            build_before_after_rows(
                result,
                "tier2_support_hits_before_reranking",
                "tier2_support_hits",
                "tier2",
            )
        )

    with (output_dir / "gaira_inference_before_after_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "domain_pack",
                "tier",
                "stage",
                "rank",
                "source_dataset_id",
                "source_label",
                "result_type",
                "base_score",
                "domain_relevance_weight",
                "reranked_score",
                "rerank_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary_lines = ["GAIRA inference v1.1 demo", ""]
    for result in results:
        summary_lines.append(result["final_summary"])
        summary_lines.append("")
        if result["domain_pack"] == "GAIRA_SERUM" and "p1" in result["query_id"]:
            write_text(output_dir / "gaira_inference_serum_example_reranked.txt", result["final_summary"])
        if result["domain_pack"] == "GAIRA_EV":
            write_text(output_dir / "gaira_inference_ev_example_reranked.txt", result["final_summary"])

    write_text(output_dir / "gaira_inference_reranked_summary.txt", "\n".join(summary_lines))
    print(f"Wrote GAIRA inference v1.1 demo outputs to: {output_dir}")


if __name__ == "__main__":
    main()
