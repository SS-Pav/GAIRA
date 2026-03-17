import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def load_small2023_query(db_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        row = connection.execute(
            """
            SELECT class_label, subclass_label, mean_wavenumbers_json, mean_intensity_json
            FROM biosample_class_summary
            WHERE dataset_id = 'small2023_ev'
              AND class_label = 'c00'
              AND subclass_label = 'normedprobe1'
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        raise ValueError("Could not load the small2023_ev c00 / normedprobe1 class summary.")
    class_label, subclass_label, x_json, y_json = row
    return np.asarray(json.loads(x_json), dtype=float), np.asarray(json.loads(y_json), dtype=float), f"{class_label}_{subclass_label}"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.grounding_search import GroundingSearchEngine, SpectrumQuery
    from gaira.ev_context import EVContextRetriever

    db_path = project_root / "data" / "gaira.duckdb"
    output_dir = Path("/Volumes/SSD_SPG/GAIRA_DATA/processed/gaira_ev_context_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    x_values, y_values, label_stub = load_small2023_query(db_path)
    query = SpectrumQuery(
        query_id="small2023_ev_c00_normedprobe1",
        query_label="c00",
        query_family="normedprobe1",
        source_dataset_id="small2023_ev",
        x=x_values,
        y=y_values,
        notes="small2023_ev processed class summary",
    )

    grounding_engine = GroundingSearchEngine(db_path=db_path)
    context_retriever = EVContextRetriever(db_path=db_path)

    direct_df = grounding_engine.search_direct_spectral_evidence(query, top_n_per_source=3)
    direct_df.to_csv(output_dir / "demo_grounding_results.csv", index=False)

    context_from_text_df = context_retriever.search_by_text(
        "small2023_ev probe1 probe2 v1 default transductive upper bound weak label extracellular vesicles substrate caveat",
        top_n=8,
    )
    context_from_labels_df = context_retriever.search_by_labels(
        ["small2023_ev", "normedprobe1", "normedprobe2", "Impact", "Strong-D", "shine_ev_sers"],
        top_n=8,
    )

    context_from_text_df.to_csv(output_dir / "demo_ev_context_from_text.csv", index=False)
    context_from_labels_df.to_csv(output_dir / "demo_ev_context_from_labels.csv", index=False)

    summary_lines = [
        "GAIRA_EV_CONTEXT demo",
        "",
        f"Grounding query: {query.query_id}",
        "",
        "Top direct grounding labels:",
    ]
    for label in direct_df["source_label"].head(6).astype(str).tolist():
        summary_lines.append(f"- {label}")
    summary_lines.append("")
    summary_lines.append("Top EV-context chunks from text query:")
    if context_from_text_df.empty:
        summary_lines.append("- no text-context hits")
    else:
        for row in context_from_text_df.to_dict(orient="records"):
            summary_lines.append(
                f"- {row['document_id']} | {row['section']} | score={float(row['score']):.2f} | {row['matched_tokens']}"
            )
    summary_lines.append("")
    summary_lines.append("Top EV-context chunks from label query:")
    if context_from_labels_df.empty:
        summary_lines.append("- no label-context hits")
    else:
        for row in context_from_labels_df.to_dict(orient="records"):
            summary_lines.append(
                f"- {row['document_id']} | {row['section']} | score={float(row['score']):.2f} | {row['matched_tokens']}"
            )

    (output_dir / "gaira_ev_context_demo.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote EV-context demo outputs to: {output_dir}")


if __name__ == "__main__":
    main()
