import json
import sys
from pathlib import Path

import pandas as pd


OUTPUT_FOLDER_NAME = "gaira_grounding_search_v1"


def format_result_block(title: str, df: pd.DataFrame, limit: int = 5) -> list[str]:
    lines = [title]
    if df.empty:
        lines.append("  no results")
        return lines
    for row in df.head(limit).to_dict(orient="records"):
        lines.append(
            "  - "
            f"{row.get('evidence_tier', 'unknown')} | {row.get('result_type', 'unknown')} | "
            f"{row.get('source_dataset_id', 'unknown')} | {row.get('source_label', 'unknown')} | "
            f"score={float(row.get('score', 0.0)):.4f}"
        )
    return lines


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import ensure_storage_dirs, get_database_path, resolve_storage_path
    from gaira.grounding_search import GroundingSearchEngine

    storage_config = ensure_storage_dirs()
    processed_root = resolve_storage_path(storage_config.get("processed_data"))
    db_path = get_database_path()

    if processed_root is None:
        print("The storage config is missing processed_data.")
        return

    output_dir = processed_root / OUTPUT_FOLDER_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = GroundingSearchEngine(db_path=db_path)
    queries = engine.get_demo_queries()

    demo_rows: list[dict] = []
    summary_lines: list[str] = ["GAIRA grounding search v1", ""]

    for query in queries:
        direct_df = engine.search_direct_spectral_evidence(query, top_n_per_source=5)
        support_df = engine.search_supporting_literature_for_spectrum(query, top_n=8)

        if not direct_df.empty:
            demo_rows.extend(direct_df.to_dict(orient="records"))
        if not support_df.empty:
            demo_rows.extend(support_df.to_dict(orient="records"))

        summary_lines.append(
            f"Query {query.query_id} ({query.query_label}; {query.query_family}; {query.source_dataset_id})"
        )
        summary_lines.extend(format_result_block("Tier 1 direct evidence", direct_df))
        summary_lines.extend(format_result_block("Tier 2 literature support", support_df))
        summary_lines.append("")

    demo_df = pd.DataFrame(demo_rows)
    if not demo_df.empty:
        demo_df.to_csv(output_dir / "grounding_search_demo_results.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "grounding_search_demo_results.csv", index=False)

    band_queries = [1659.0, 725.0, 1003.0]
    band_rows: list[dict] = []
    for band_cm in band_queries:
        band_df = engine.search_band_evidence(band_cm=band_cm, tolerance_cm=10.0)
        if not band_df.empty:
            band_rows.extend(band_df.head(15).to_dict(orient="records"))

    band_result_df = pd.DataFrame(band_rows)
    if not band_result_df.empty:
        band_result_df.to_csv(output_dir / "grounding_band_query_results.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "grounding_band_query_results.csv", index=False)

    with (output_dir / "grounding_tiered_evidence_examples.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines))

    print(f"Wrote demo outputs to: {output_dir}")
    print(f"Spectrum queries processed: {len(queries)}")
    print(f"Band queries processed: {len(band_queries)}")


if __name__ == "__main__":
    main()
