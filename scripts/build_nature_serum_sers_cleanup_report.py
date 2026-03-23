import json
from pathlib import Path

import duckdb
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path, get_storage_paths, require_data_root_exists
    from gaira.grounding_search import GroundingSearchEngine, SpectrumQuery
    from gaira.inference import GAIRAInferenceEngine, InferenceRequest

    require_data_root_exists()
    storage_paths = get_storage_paths()
    db_path = get_database_path()
    output_dir = storage_paths["processed_data"] / "nature_serum_sers_cleanup"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "nature_serum_sers_cleanup_report.md"

    with duckdb.connect(str(db_path), read_only=True) as connection:
        removed_tables = [
            "grounding_metadata",
            "grounding_spectra",
            "grounding_processed_spectra",
            "grounding_processed_points",
            "grounding_peaks",
            "grounding_class_summary",
            "grounding_support_documents",
            "grounding_support_chunks",
            "grounding_support_spectra",
            "grounding_support_spectrum_points",
            "dataset_domain_context",
            "subclass_domain_context",
        ]
        removal_counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table} WHERE dataset_id = ?", ["nature_serum_sers"]).fetchone()[0]
            for table in removed_tables
        }

        serum_row = connection.execute(
            """
            SELECT p.biosample_id, m.class_label, m.subclass_label, p.wavenumbers_json, p.intensity_json
            FROM biosample_processed_spectra p
            JOIN biosample_metadata m USING (biosample_id, dataset_id)
            WHERE p.dataset_id = 'cca_hcc_lm_serum_sers'
            ORDER BY p.biosample_id
            LIMIT 1
            """
        ).fetchone()

        metabolite_row = connection.execute(
            """
            SELECT p.grounding_id, m.class_label, p.wavenumbers_json, p.intensity_json
            FROM grounding_processed_spectra p
            JOIN grounding_metadata m USING (grounding_id, dataset_id)
            WHERE p.dataset_id = 'metabolite_sers63_support'
            ORDER BY p.grounding_id
            LIMIT 1
            """
        ).fetchone()

        ck18_row = connection.execute(
            """
            SELECT document_id, title, notes
            FROM domain_context_documents
            WHERE document_id = 'ck18_dili_biomarker_support'
            """
        ).fetchone()

        ck18_chunks = connection.execute(
            """
            SELECT section, chunk_text
            FROM domain_context_chunks
            WHERE document_id = 'ck18_dili_biomarker_support'
            ORDER BY chunk_order
            """
        ).fetchall()

    engine = GAIRAInferenceEngine(db_path=db_path, theme_layer_version="v3")
    serum_request = InferenceRequest(
        domain="serum",
        query_id=str(serum_row[0]),
        query_label=str(serum_row[1]),
        query_family=str(serum_row[2]),
        source_dataset_id="cca_hcc_lm_serum_sers",
        spectrum_query=SpectrumQuery(
            query_id=str(serum_row[0]),
            query_label=str(serum_row[1]),
            query_family=str(serum_row[2]),
            source_dataset_id="cca_hcc_lm_serum_sers",
            x=pd.Series(json.loads(serum_row[3]), dtype=float).to_numpy(),
            y=pd.Series(json.loads(serum_row[4]), dtype=float).to_numpy(),
            notes="Post-cleanup serum sanity query",
        ),
    )
    serum_result = engine.run_inference(serum_request)

    grounding_engine = GroundingSearchEngine(db_path=db_path)
    grounding_query = SpectrumQuery(
        query_id=str(metabolite_row[0]),
        query_label=str(metabolite_row[1]),
        query_family="metabolite_sers63_support",
        source_dataset_id="metabolite_sers63_support",
        x=pd.Series(json.loads(metabolite_row[2]), dtype=float).to_numpy(),
        y=pd.Series(json.loads(metabolite_row[3]), dtype=float).to_numpy(),
        notes="Post-cleanup metabolite grounding sanity query",
    )
    grounding_hits = grounding_engine.search_direct_spectral_evidence(grounding_query, top_n_per_source=5)
    same_dataset_hits = grounding_hits[grounding_hits["source_dataset_id"] == "metabolite_sers63_support"].head(3)

    lines = [
        "# nature_serum_sers Cleanup Report",
        "",
        "## Removed from active GAIRA",
        "- Registry row removed",
        "- GAIRA_GROUNDING pack membership removed",
        "- Parser registration/imports removed",
        "- download/ingest/process routing removed",
        "- dataset/subclass domain-context seed rows removed",
        "- live SSD_Rad DuckDB rows removed",
        "",
        "## Post-cleanup DB counts for `nature_serum_sers`",
    ]
    for table, count in removal_counts.items():
        lines.append(f"- `{table}`: {count}")

    lines.extend(
        [
            "",
            "## Preserved minimal CK18/K18 support",
            f"- document_id: `{ck18_row[0]}`",
            f"- title: {ck18_row[1]}",
            f"- notes: {ck18_row[2]}",
        ]
    )
    for section, chunk_text in ck18_chunks:
        lines.append(f"- chunk `{section}`: {chunk_text}")

    lines.extend(
        [
            "",
            "## Unaffected dataset checks",
            f"- `cca_hcc_lm_serum_sers` top tier-1: {serum_result['tier1_grounding_hits'][0]['source_dataset_id']} / {serum_result['tier1_grounding_hits'][0]['source_label']}",
            f"- `cca_hcc_lm_serum_sers` dominant themes: {', '.join(serum_result.get('dominant_themes', []))}",
            f"- `metabolite_sers63_support` top same-dataset hit: {same_dataset_hits.iloc[0]['source_label'] if not same_dataset_hits.empty else 'none'}",
            "",
            "## Summary",
            "GAIRA is cleaner conceptually after removal. The LFIA/K18 workbook no longer appears as an active integrated dataset, while a minimal CK18/K18 liver-injury biomarker note remains available only as support/context.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote cleanup report: {report_path}")


if __name__ == "__main__":
    main()
