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


def maybe_read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return read_text(path)


def read_csv_preview(path: Path, n_rows: int = 5) -> str:
    return pd.read_csv(path).head(n_rows).to_string(index=False)


def maybe_read_csv_preview(path: Path, n_rows: int = 5) -> str | None:
    if not path.exists():
        return None
    return read_csv_preview(path, n_rows=n_rows)


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
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path, get_storage_paths, require_data_root_exists

    storage_paths = require_data_root_exists()
    db_path = get_database_path()
    processed_root = storage_paths["processed_data"]

    default_embedding_path = project_root / "docs" / "default_embedding_status.md"
    v1_summary_path = processed_root / "small2023_ev_invariant_embedding" / "embedding_summary.txt"
    v2_summary_path = processed_root / "small2023_ev_invariant_embedding_v2" / "embedding_summary_v2.txt"
    v2_validation_path = processed_root / "small2023_ev_invariant_embedding_v2_validation" / "v2_validation_summary.txt"
    v3_summary_path = processed_root / "small2023_ev_invariant_embedding_v3" / "embedding_summary_v3.txt"
    diabetes_mapping_summary_path = processed_root / "diabetes_plasma_ev_sers_mapping_summary.txt"
    small2023_match_example_path = (
        processed_root / "small2023_ev_class_reference_matches" / "class_c00_normedprobe1_matches.csv"
    )
    shine_consensus_example_path = (
        processed_root / "shine_class_reference_matches" / "class_d0_c0_set9_consensus.csv"
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

    v1_text = maybe_read_text(v1_summary_path)
    if v1_text:
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

    v2_text = maybe_read_text(v2_summary_path)
    if v2_text:
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
                notes=("Curated EV-context summary for the strong transductive upper-bound benchmark."),
                chunks=v2_chunks,
            )
        )

    v2_validation_text = maybe_read_text(v2_validation_path)
    if v2_validation_text:
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
                notes=("Curated EV-context caveat that explains why v2 is strong but still transductive."),
                chunks=v2_validation_chunks,
            )
        )

    v3_text = maybe_read_text(v3_summary_path)
    if v3_text:
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
                notes=("Curated EV-context note for the strict source-only negative-result benchmark."),
                chunks=v3_chunks,
            )
        )

    documents.append(
        build_document(
            document_id="gaira_ev_context_small2023_probe_family_note",
            context_type="interpretive_note",
            evidence_basis="derived_from_embedding_benchmark",
            source_dataset_id="small2023_ev",
            source_file=(
                f"{v1_summary_path}; {v2_summary_path}; {v2_validation_path}; {v3_summary_path}; "
                "dataset_domain_context + subclass_domain_context"
            ),
            title="small2023_ev probe-family and invariance note",
            notes=(
                "Curated EV-context note explaining Probe1/Probe2 separation and why substrate invariance matters."
            ),
            chunks=[
                (
                    "small2023_probe_family_note",
                    (
                        "small2023_ev spans multiple grounded probe/substrate families rather than one uniform EV "
                        "measurement domain. Probe1 and Probe2 carry the same mixture-style class labels but differ "
                        "in substrate batch, raw axis family, and preprocessing/cropping context, so raw intensities "
                        "should not be treated as directly comparable. This is why substrate invariance matters in "
                        "small2023_ev: the benchmark program exists to reduce probe-domain effects while preserving "
                        "biology-centered class structure in processed space."
                    ),
                    {"source_kind": "probe_family_invariance_note"},
                )
            ],
        )
    )

    documents.append(
        build_document(
            document_id="gaira_ev_context_small2023_benchmark_hierarchy_note",
            context_type="interpretive_note",
            evidence_basis="derived_from_embedding_benchmark",
            source_dataset_id="small2023_ev",
            source_file=f"{default_embedding_path}; {v2_validation_path}; {v3_summary_path}",
            title="small2023_ev benchmark hierarchy note",
            notes=(
                "Curated EV-context note explaining why v1 is default, v2 is upper-bound, and v3 remains a strict-transfer stress test."
            ),
            chunks=[
                (
                    "small2023_benchmark_hierarchy_note",
                    (
                        "The small2023_ev hierarchy should be read as three different roles rather than three "
                        "equivalent embeddings. v1 is the current default because it improved cross-probe transfer "
                        "without becoming a transductive upper-bound system. v2 is stronger numerically, but its own "
                        "validation notes show that most of the gain is driven by joint class-supervised projection "
                        "across both probes, so it is kept as a research upper-bound. v3 remains valuable precisely "
                        "because it failed under strict source-only transfer: it shows that substrate effects are not "
                        "solved just by declaring a stricter protocol."
                    ),
                    {"source_kind": "benchmark_hierarchy_note"},
                )
            ],
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
            document_id="gaira_ev_context_cargo_mixture_caveat",
            context_type="caveat",
            evidence_basis="derived_from_processed_summary",
            source_dataset_id="small2023_ev,shine_ev_sers,diabetes_plasma_ev_sers",
            source_file="embedding summaries + dataset_domain_context",
            title="EV cargo and mixed-signal caveat",
            notes=(
                "Curated EV-context caveat that frames EV SERS as a mixed membrane/cargo/adsorbate signal rather than a single-analyte readout."
            ),
            chunks=[
                (
                    "ev_cargo_mixture_caveat",
                    (
                        "EV SERS signals should be interpreted as mixed biochemical structure rather than single-molecule "
                        "fingerprints. Depending on substrate and protocol, the signal can reflect membrane lipids, "
                        "protein cargo, nucleic-acid contributions, and substrate-enhanced adsorbates at the same time. "
                        "This is why GAIRA treats shared grounding as biochemical analog support and then overlays EV-domain "
                        "context before making interpretation claims."
                    ),
                    {"source_kind": "ev_mixture_caveat"},
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
                            "benchmark or a patient-level benchmark. Within the linked paper framing, Impact tracks "
                            "the overweight / BMI > 25 cohort-family context, while Strong-D tracks the otherwise / "
                            "BMI < 25 / not-overweight diabetic cohort-family context."
                    ),
                    {"source_kind": "weak_label_caveat"},
                )
            ],
        )
    )

    diabetes_mapping_text = maybe_read_text(diabetes_mapping_summary_path)
    if diabetes_mapping_text:
        documents.append(
            build_document(
                document_id="gaira_ev_context_diabetes_mapping_audit_note",
                context_type="caveat",
                evidence_basis="derived_from_processed_summary",
                source_dataset_id="diabetes_plasma_ev_sers",
                source_file=str(diabetes_mapping_summary_path),
                title="Diabetes plasma EV mapping-audit caveat",
                notes=("Curated EV-context note grounded in the stored diabetes mapping audit summary."),
                chunks=[
                    (
                        "diabetes_mapping_audit_caveat",
                        diabetes_mapping_text,
                        {"source_kind": "mapping_audit_summary"},
                    )
                ],
            )
        )

    documents.append(
        build_document(
            document_id="gaira_ev_context_diabetes_use_note",
            context_type="interpretive_note",
            evidence_basis="derived_from_dataset_context",
            source_dataset_id="diabetes_plasma_ev_sers",
            source_file="dataset_domain_context + subclass_domain_context",
            title="Diabetes plasma EV usage note",
            notes=("Curated EV-context note describing the defensible use of diabetes_plasma_ev_sers inside GAIRA."),
            chunks=[
                (
                    "diabetes_use_note",
                    (
                        "Within GAIRA, diabetes_plasma_ev_sers should be used as a weak-label external EV stress test "
                        "and domain-framing resource rather than as a four-subgroup benchmark. The defensible read is "
                        "Impact-vs-StrongD cohort-family contrast under one processed archive family, with Impact "
                        "standing in for the overweight / BMI > 25 cohort-family context and Strong-D standing in "
                        "for the otherwise / BMI < 25 / not-overweight diabetic cohort-family context. Use it with "
                        "careful avoidance of patient-level or subgroup-specific overclaims."
                    ),
                    {"source_kind": "diabetes_usage_note"},
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

    small2023_match_preview = maybe_read_csv_preview(small2023_match_example_path, n_rows=6)
    shine_consensus_preview = maybe_read_csv_preview(shine_consensus_example_path, n_rows=3)
    if small2023_match_preview and shine_consensus_preview:
        documents.append(
            build_document(
                document_id="gaira_ev_context_multicomponent_analog_note",
                context_type="interpretive_note",
                evidence_basis="derived_from_processed_summary",
                source_dataset_id="small2023_ev,shine_ev_sers",
                source_file=f"{small2023_match_example_path}; {shine_consensus_example_path}",
                title="EV multicomponent analog-grounding note",
                notes=(
                    "Curated EV-context note grounded in existing small2023 and SHINE reference-match outputs."
                ),
                chunks=[
                    (
                        "ev_multicomponent_analog_note",
                        (
                            "Current EV-side reference matching already shows that EV SERS interpretation is usually "
                            "multicomponent rather than single-molecule. In local small2023 outputs, one EV class mean "
                            "pulls mixed saccharide, amino-acid, and protein analogs. In local SHINE consensus outputs, "
                            "the same class-level summary can carry protein, lipid/fatty-acid, and nucleic-acid analog "
                            "support at once. This means shared grounding hits should be read as mixed biochemical support "
                            "for EV-associated chemistry rather than literal cargo IDs.\n\n"
                            "small2023 example preview:\n"
                            + small2023_match_preview
                            + "\n\nSHINE consensus preview:\n"
                            + shine_consensus_preview
                        ),
                        {"source_kind": "ev_reference_match_summary"},
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

    if shine_consensus_preview:
        documents.append(
            build_document(
                document_id="gaira_ev_context_shine_consensus_note",
                context_type="interpretive_note",
                evidence_basis="derived_from_processed_summary",
                source_dataset_id="shine_ev_sers",
                source_file=str(shine_consensus_example_path),
                title="SHINE EV consensus-region note",
                notes=("Curated EV-context note grounded in the stored SHINE class-consensus export."),
                chunks=[
                    (
                        "shine_consensus_region_note",
                        (
                            "The current SHINE consensus exports show that EV-class interpretation is often region-based "
                            "and consensus-based rather than peak-to-single-molecule. For example, the local SHINE "
                            "consensus file for D0_C0 highlights proteins and lipid/fatty-acid support with dominant "
                            "regions around 1300-1500 and 450-700 cm^-1, while explicitly warning that the example analog "
                            "references should not be treated as literal molecule IDs.\n\n"
                            + shine_consensus_preview
                        ),
                        {"source_kind": "shine_consensus_summary"},
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
