import json
import re
from pathlib import Path

import duckdb
import pandas as pd


CONTEXT_LAYER = "GAIRA_EV_CONTEXT"


def split_paragraph_chunks(text: str) -> list[str]:
    return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_document(
    document_id: str,
    context_type: str,
    evidence_basis: str,
    source_dataset_id: str,
    source_file: str,
    title: str,
    notes: str,
    chunks: list[tuple[str, str, dict]],
) -> dict:
    return {
        "document_id": document_id,
        "context_layer": CONTEXT_LAYER,
        "intended_domain": "ev",
        "context_type": context_type,
        "evidence_basis": evidence_basis,
        "source_dataset_id": source_dataset_id,
        "source_file": source_file,
        "title": title,
        "use_for_rag": "yes",
        "notes": notes,
        "chunks": chunks,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "gaira.duckdb"

    default_embedding_path = project_root / "docs" / "default_embedding_status.md"
    v1_summary_path = Path(
        "/Volumes/SSD_SPG/GAIRA_DATA/processed/small2023_ev_invariant_embedding/embedding_summary.txt"
    )
    v2_summary_path = Path(
        "/Volumes/SSD_SPG/GAIRA_DATA/processed/small2023_ev_invariant_embedding_v2/embedding_summary_v2.txt"
    )
    v2_validation_path = Path(
        "/Volumes/SSD_SPG/GAIRA_DATA/processed/small2023_ev_invariant_embedding_v2_validation/v2_validation_summary.txt"
    )
    v3_summary_path = Path(
        "/Volumes/SSD_SPG/GAIRA_DATA/processed/small2023_ev_invariant_embedding_v3/embedding_summary_v3.txt"
    )

    documents: list[dict] = []

    default_text = read_text(default_embedding_path)
    default_chunks = []
    for idx, chunk_text in enumerate(split_paragraph_chunks(default_text), start=1):
        section = "embedding_registry_status" if idx <= 2 else "default_embedding_rationale"
        default_chunks.append(
            (
                section,
                chunk_text,
                {"source_kind": "default_embedding_doc", "chunk_index": idx},
            )
        )
    documents.append(
        build_document(
            document_id="gaira_ev_context_default_embedding_status",
            context_type="interpretive_note",
            evidence_basis="derived_from_embedding_benchmark",
            source_dataset_id="small2023_ev",
            source_file=str(default_embedding_path),
            title="Current GAIRA_EV default embedding status",
            notes=(
                "Curated EV-context note describing why small2023_ev_v1 is the current project-approved default."
            ),
            chunks=default_chunks,
        )
    )

    v1_text = read_text(v1_summary_path)
    v1_chunks = []
    for idx, chunk_text in enumerate(split_paragraph_chunks(v1_text), start=1):
        section = "v1_embedding_summary" if idx <= 3 else "v1_embedding_interpretation"
        v1_chunks.append(
            (
                section,
                chunk_text,
                {"source_kind": "embedding_summary_v1", "chunk_index": idx},
            )
        )
    documents.append(
        build_document(
            document_id="gaira_ev_context_small2023_v1",
            context_type="benchmark_summary",
            evidence_basis="derived_from_embedding_benchmark",
            source_dataset_id="small2023_ev",
            source_file=str(v1_summary_path),
            title="small2023_ev v1 embedding context",
            notes=(
                "Curated EV-context summary for the baseline/default small2023_ev invariant embedding benchmark."
            ),
            chunks=v1_chunks,
        )
    )

    v2_text = read_text(v2_summary_path)
    v2_chunks = []
    for idx, chunk_text in enumerate(split_paragraph_chunks(v2_text), start=1):
        section = "v2_embedding_summary" if idx <= 3 else "v2_embedding_geometry"
        v2_chunks.append(
            (
                section,
                chunk_text,
                {"source_kind": "embedding_summary_v2", "chunk_index": idx},
            )
        )
    documents.append(
        build_document(
            document_id="gaira_ev_context_small2023_v2_upper_bound",
            context_type="benchmark_summary",
            evidence_basis="derived_from_embedding_benchmark",
            source_dataset_id="small2023_ev",
            source_file=str(v2_summary_path),
            title="small2023_ev v2 upper-bound context",
            notes=(
                "Curated EV-context summary for the strong transductive upper-bound benchmark."
            ),
            chunks=v2_chunks,
        )
    )

    v2_validation_text = read_text(v2_validation_path)
    v2_validation_chunks = []
    for idx, chunk_text in enumerate(split_paragraph_chunks(v2_validation_text), start=1):
        section = "v2_validation_summary" if idx <= 2 else "v2_validation_caveat"
        v2_validation_chunks.append(
            (
                section,
                chunk_text,
                {"source_kind": "v2_validation_summary", "chunk_index": idx},
            )
        )
    documents.append(
        build_document(
            document_id="gaira_ev_context_small2023_v2_validation",
            context_type="caveat",
            evidence_basis="derived_from_embedding_benchmark",
            source_dataset_id="small2023_ev",
            source_file=str(v2_validation_path),
            title="small2023_ev v2 validation caveat",
            notes=(
                "Curated EV-context caveat that explains why v2 is strong but still transductive."
            ),
            chunks=v2_validation_chunks,
        )
    )

    v3_text = read_text(v3_summary_path)
    v3_chunks = []
    for idx, chunk_text in enumerate(split_paragraph_chunks(v3_text), start=1):
        section = "v3_strict_transfer_summary" if idx <= 3 else "v3_strict_transfer_failure"
        v3_chunks.append(
            (
                section,
                chunk_text,
                {"source_kind": "embedding_summary_v3", "chunk_index": idx},
            )
        )
    documents.append(
        build_document(
            document_id="gaira_ev_context_small2023_v3_negative_result",
            context_type="caveat",
            evidence_basis="derived_from_embedding_benchmark",
            source_dataset_id="small2023_ev",
            source_file=str(v3_summary_path),
            title="small2023_ev v3 strict-transfer caveat",
            notes=(
                "Curated EV-context note for the strict source-only negative-result benchmark."
            ),
            chunks=v3_chunks,
        )
    )

    with duckdb.connect(str(db_path), read_only=True) as connection:
        dataset_context_df = connection.execute(
            """
            SELECT dataset_id, biosample_type, measurement_mode, default_substrate_type,
                   default_substrate_material, substrate_vendor, instrument_context,
                   default_preprocessing_family, notes
            FROM dataset_domain_context
            WHERE dataset_id IN ('small2023_ev', 'shine_ev_sers', 'diabetes_plasma_ev_sers')
            ORDER BY dataset_id
            """
        ).fetchdf()
        subclass_context_df = connection.execute(
            """
            SELECT dataset_id, subclass_label, substrate_type, substrate_material, substrate_vendor,
                   substrate_batch_id, probe_family, spectral_axis_family,
                   cross_domain_intensity_comparable, preprocessing_family, notes
            FROM subclass_domain_context
            WHERE dataset_id IN ('small2023_ev', 'shine_ev_sers', 'diabetes_plasma_ev_sers')
            ORDER BY dataset_id, subclass_label
            """
        ).fetchdf()

    documents.append(
        build_document(
            document_id="gaira_ev_context_dataset_context",
            context_type="paper_summary",
            evidence_basis="derived_from_dataset_context",
            source_dataset_id="small2023_ev,shine_ev_sers,diabetes_plasma_ev_sers",
            source_file="dataset_domain_context + subclass_domain_context",
            title="Current GAIRA_EV dataset context",
            notes=(
                "Curated EV pack context compiled from existing GAIRA dataset and subclass domain context rows."
            ),
            chunks=[
                (
                    "ev_dataset_context",
                    "Dataset-level EV context:\n"
                    + dataset_context_df.to_string(index=False)
                    + "\n\nSubclass-level EV context:\n"
                    + subclass_context_df.to_string(index=False),
                    {"source_kind": "dataset_domain_context"},
                )
            ],
        )
    )

    documents.append(
        build_document(
            document_id="gaira_ev_context_cross_substrate_caveat",
            context_type="caveat",
            evidence_basis="derived_from_dataset_context",
            source_dataset_id="small2023_ev,shine_ev_sers",
            source_file="dataset_domain_context + subclass_domain_context + benchmark docs",
            title="EV cross-substrate comparability caveat",
            notes=(
                "Curated EV-specific caveat about probe/substrate comparability across EV SERS datasets."
            ),
            chunks=[
                (
                    "cross_substrate_caveat",
                    (
                        "The current EV pack spans multiple SERS substrate families and probe families. "
                        "small2023_ev explicitly separates Probe1 and Probe2 because raw intensities are not "
                        "directly comparable across those families, and shine_ev_sers is a separate gold "
                        "nanopillar EV-SERS study context. EV interpretation should therefore prioritize "
                        "study-matched context and processed-space comparisons over raw cross-substrate "
                        "intensity comparisons."
                    ),
                    {"source_kind": "cross_substrate_caveat"},
                )
            ],
        )
    )

    documents.append(
        build_document(
            document_id="gaira_ev_context_diabetes_weak_label_note",
            context_type="caveat",
            evidence_basis="derived_from_dataset_context",
            source_dataset_id="diabetes_plasma_ev_sers",
            source_file="dataset_domain_context + subclass_domain_context",
            title="Diabetes plasma EV weak-label framing",
            notes=(
                "Curated EV-context note that locks diabetes_plasma_ev_sers to its conservative weak-label framing."
            ),
            chunks=[
                (
                    "diabetes_weak_label_caveat",
                    (
                        "diabetes_plasma_ev_sers is usable as an onboarded plasma EV SERS dataset, but only "
                        "under the archive-supported Impact-vs-StrongD cohort-family framing. The four paper "
                        "subgroups were not reconstructed because patient/sample identifiers were not embedded "
                        "in the released MAT cells, so this dataset should not be treated as a four-subgroup "
                        "benchmark or a patient-level benchmark."
                    ),
                    {"source_kind": "weak_label_caveat"},
                )
            ],
        )
    )

    documents.append(
        build_document(
            document_id="gaira_ev_context_evidence_tiering_note",
            context_type="interpretive_note",
            evidence_basis="derived_from_embedding_benchmark",
            source_dataset_id="small2023_ev,ramanbiolib",
            source_file="default_embedding_status + embedding summaries",
            title="EV-specific evidence tiering note",
            notes=(
                "Curated EV-context note about how shared grounding should be interpreted through EV-domain context."
            ),
            chunks=[
                (
                    "ev_evidence_tiering",
                    (
                        "For EV SERS queries, shared grounding hits from RamanBioLib or other non-EV resources "
                        "should be treated as analog grounding rather than literal EV cargo identification. "
                        "The current EV pack relies on small2023_ev_v1 as the working default because it preserves "
                        "useful class structure while keeping probe-domain effects explicit. This means shared "
                        "grounding evidence should be interpreted through EV dataset context, probe family, and "
                        "weak-label status before making biological claims."
                    ),
                    {"source_kind": "evidence_tiering_policy"},
                )
            ],
        )
    )

    documents.append(
        build_document(
            document_id="gaira_ev_context_shine_note",
            context_type="interpretive_note",
            evidence_basis="derived_from_dataset_context",
            source_dataset_id="shine_ev_sers",
            source_file="dataset_domain_context",
            title="SHINE EV study context note",
            notes=(
                "Curated EV-context note for the current SHINE EV-SERS dataset framing."
            ),
            chunks=[
                (
                    "shine_ev_context_note",
                    (
                        "shine_ev_sers is the current onboarded EV-SERS hepatotoxicity study on a grounded gold "
                        "Silmeco nanopillar structured substrate. It should be interpreted as its own EV study "
                        "context rather than as directly interchangeable with small2023_ev probe families."
                    ),
                    {"source_kind": "shine_dataset_context"},
                )
            ],
        )
    )

    document_rows = []
    chunk_rows = []
    for document in documents:
        document_rows.append({key: value for key, value in document.items() if key != "chunks"})
        for chunk_order, (section, chunk_text, metadata) in enumerate(document["chunks"], start=1):
            chunk_rows.append(
                {
                    "chunk_id": f"{document['document_id']}_chunk_{chunk_order:02d}",
                    "document_id": document["document_id"],
                    "context_layer": CONTEXT_LAYER,
                    "intended_domain": "ev",
                    "chunk_order": chunk_order,
                    "section": section,
                    "chunk_text": chunk_text,
                    "metadata_json": json.dumps(metadata, sort_keys=True),
                }
            )

    documents_df = pd.DataFrame(document_rows)
    chunks_df = pd.DataFrame(chunk_rows)

    with duckdb.connect(str(db_path)) as connection:
        connection.execute(
            "DELETE FROM domain_context_chunks WHERE context_layer = ? AND intended_domain = 'ev'",
            [CONTEXT_LAYER],
        )
        connection.execute(
            "DELETE FROM domain_context_documents WHERE context_layer = ? AND intended_domain = 'ev'",
            [CONTEXT_LAYER],
        )

        connection.register("documents_df", documents_df)
        connection.execute("INSERT INTO domain_context_documents SELECT * FROM documents_df")
        connection.unregister("documents_df")

        connection.register("chunks_df", chunks_df)
        connection.execute("INSERT INTO domain_context_chunks SELECT * FROM chunks_df")
        connection.unregister("chunks_df")

    print("GAIRA_EV_CONTEXT ingest complete.")
    print(f"Inserted domain_context_documents rows: {len(documents_df)}")
    print(f"Inserted domain_context_chunks rows: {len(chunks_df)}")


if __name__ == "__main__":
    main()
