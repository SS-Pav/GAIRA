import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_database_path
    from gaira.grounding_search import GroundingSearchEngine
    from gaira.serum_context import SerumContextRetriever

    db_path = get_database_path()
    output_dir = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_serum_context_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    grounding_engine = GroundingSearchEngine(db_path=db_path)
    context_retriever = SerumContextRetriever(db_path=db_path)

    demo_queries = grounding_engine.get_demo_queries()
    query = next(
        (
            item
            for item in demo_queries
            if item.source_dataset_id != "hcc_serum"
        ),
        None,
    )
    if query is None:
        raise ValueError("Could not find an active non-holdout serum-style demo query.")
    direct_df = grounding_engine.search_direct_spectral_evidence(query, top_n_per_source=3)
    top_labels = direct_df["source_label"].head(5).astype(str).tolist()
    top_bands = [725.0, 1003.0, 1450.0, 1659.0]

    context_from_labels_df = context_retriever.search_by_grounding_labels(top_labels, top_n=6)
    context_from_bands_df = context_retriever.search_by_bands(top_bands, top_n=6)
    context_from_text_df = context_retriever.search_by_text(
        "serum Ag colloids uric acid hypoxanthine batch caveat paper comparison covid spontaneous Raman cohort protocol",
        top_n=6,
    )

    direct_df.to_csv(output_dir / "demo_grounding_results.csv", index=False)
    context_from_labels_df.to_csv(output_dir / "demo_serum_context_from_labels.csv", index=False)
    context_from_bands_df.to_csv(output_dir / "demo_serum_context_from_bands.csv", index=False)
    context_from_text_df.to_csv(output_dir / "demo_serum_context_from_text.csv", index=False)

    summary_lines = [
        "GAIRA_SERUM_CONTEXT demo",
        "",
        f"Grounding query: {query.query_id}",
        "",
        "Top direct grounding labels:",
    ]
    for label in top_labels:
        summary_lines.append(f"- {label}")
    summary_lines.append("")
    summary_lines.append("Top serum-context chunks from grounding labels:")
    if context_from_labels_df.empty:
        summary_lines.append("- no label-context hits")
    else:
        for row in context_from_labels_df.to_dict(orient="records"):
            summary_lines.append(
                f"- {row['document_id']} | {row['section']} | score={float(row['score']):.2f} | {row['matched_tokens']}"
            )
    summary_lines.append("")
    summary_lines.append("Top serum-context chunks from bands 725, 1003, 1450, 1659:")
    if context_from_bands_df.empty:
        summary_lines.append("- no band-context hits")
    else:
        for row in context_from_bands_df.to_dict(orient="records"):
            summary_lines.append(
                f"- {row['document_id']} | {row['section']} | score={float(row['score']):.2f} | {row['matched_tokens']}"
            )

    (output_dir / "gaira_serum_context_demo.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print(f"Wrote serum-context demo outputs to: {output_dir}")


if __name__ == "__main__":
    main()
