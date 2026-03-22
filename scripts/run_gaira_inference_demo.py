import json
import sys
from pathlib import Path


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


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
    output_dir = storage_paths["processed_data"] / "gaira_inference_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = GAIRAInferenceEngine(db_path=db_path)
    requests = [
        load_serum_class_mean_query(
            db_path=db_path,
            dataset_id="serum_protocol_comparison",
            class_label="p1",
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
    json_path = output_dir / "gaira_inference_demo_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary_lines = ["GAIRA inference v1 demo", ""]
    for result in results:
        summary_lines.append(result["final_summary"])
        summary_lines.append("")
        if result["domain_pack"] == "GAIRA_SERUM":
            write_text(output_dir / "gaira_inference_serum_example.txt", result["final_summary"])
        if result["domain_pack"] == "GAIRA_EV":
            write_text(output_dir / "gaira_inference_ev_example.txt", result["final_summary"])

    write_text(output_dir / "gaira_inference_demo_summary.txt", "\n".join(summary_lines))
    print(f"Wrote GAIRA inference v1 demo outputs to: {output_dir}")


if __name__ == "__main__":
    main()
