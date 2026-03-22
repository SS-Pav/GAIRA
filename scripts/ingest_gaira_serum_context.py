import json
import re
from pathlib import Path

import duckdb
import pandas as pd


CONTEXT_LAYER = "GAIRA_SERUM_CONTEXT"


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def maybe_read_text_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return read_text_file(path)


def split_paragraph_chunks(text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    return chunks


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
        "intended_domain": "serum",
        "context_type": context_type,
        "evidence_basis": evidence_basis,
        "source_dataset_id": source_dataset_id,
        "source_file": source_file,
        "title": title,
        "use_for_rag": "yes",
        "notes": notes,
        "chunks": chunks,
    }


def load_band_examples(connection: duckdb.DuckDBPyConnection, pattern: str, limit: int = 4) -> list[str]:
    rows = connection.execute(
        """
        SELECT citation_label, chunk_text
        FROM grounding_support_chunks c
        JOIN grounding_support_documents d
          ON c.document_id = d.document_id
         AND c.dataset_id = d.dataset_id
        WHERE c.dataset_id = 'serum_ag_colloids_literature_grounding'
          AND c.section = 'reported_band_assignments'
          AND c.chunk_text LIKE ?
        ORDER BY d.citation_label
        LIMIT ?
        """,
        [f"%{pattern}%", limit],
    ).fetchall()
    return [f"{citation}: {text}" for citation, text in rows]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path, get_storage_paths, require_data_root_exists

    storage_paths = require_data_root_exists()
    db_path = get_database_path()
    processed_root = storage_paths["processed_data"]

    benchmark_summary_path = processed_root / "hcc_serum_benchmark_v1" / "hcc_serum_benchmark_summary.txt"
    paper_summary_path = processed_root / "hcc_serum_paper_comparison" / "hcc_serum_paper_comparison_summary.txt"

    documents: list[dict] = []

    benchmark_text = maybe_read_text_file(benchmark_summary_path)
    if benchmark_text:
        benchmark_chunks = []
        for idx, chunk_text in enumerate(split_paragraph_chunks(benchmark_text), start=1):
            section = "benchmark_summary" if idx <= 2 else "benchmark_interpretation"
            benchmark_chunks.append(
                (
                    section,
                    chunk_text,
                    {"source_kind": "benchmark_summary", "chunk_index": idx},
                )
            )
        documents.append(
            build_document(
                document_id="gaira_serum_context_hcc_benchmark_v1",
                context_type="benchmark_summary",
                evidence_basis="derived_from_benchmark_summary",
                source_dataset_id="hcc_serum",
                source_file=str(benchmark_summary_path),
                title="hcc_serum benchmark v1 context",
                notes=(
                    "Curated serum-context summary derived from the stored hcc_serum benchmark v1 output. "
                    "Used as a serum-specific interpretive overlay rather than as new primary evidence."
                ),
                chunks=benchmark_chunks,
            )
        )

    paper_text = maybe_read_text_file(paper_summary_path)
    if paper_text:
        paper_chunks = []
        for idx, chunk_text in enumerate(split_paragraph_chunks(paper_text), start=1):
            section = "paper_reproduction_summary" if idx <= 3 else "paper_reproduction_interpretation"
            paper_chunks.append(
                (
                    section,
                    chunk_text,
                    {"source_kind": "paper_comparison_summary", "chunk_index": idx},
                )
            )
        documents.append(
            build_document(
                document_id="gaira_serum_context_hcc_paper_comparison",
                context_type="paper_summary",
                evidence_basis="derived_from_paper_summary",
                source_dataset_id="hcc_serum",
                source_file=str(paper_summary_path),
                title="hcc_serum paper-comparison context",
                notes=(
                    "Curated serum-context summary derived from the stored hcc_serum paper-comparison output. "
                    "Captures the gap between paper-faithful PCA-LDA reproduction and the current GAIRA serum benchmark."
                ),
                chunks=paper_chunks,
            )
        )

    with duckdb.connect(str(db_path), read_only=True) as connection:
        dataset_context_df = connection.execute(
            """
            SELECT dataset_id, biosample_type, measurement_mode, default_substrate_type,
                   default_substrate_material, substrate_vendor, instrument_context,
                   default_preprocessing_family, notes
            FROM dataset_domain_context
            WHERE dataset_id IN (
                'hcc_serum',
                'serum_ag_colloids',
                'serum_protocol_comparison',
                'cspp_serum',
                'ergothioneine_serum'
            )
            ORDER BY dataset_id
            """
        ).fetchdf()

        subclass_context_df = connection.execute(
            """
            SELECT dataset_id, subclass_label, substrate_type, substrate_material, substrate_vendor,
                   substrate_batch_id, probe_family, spectral_axis_family,
                   cross_domain_intensity_comparable, preprocessing_family, notes
            FROM subclass_domain_context
            WHERE dataset_id IN (
                'hcc_serum',
                'serum_ag_colloids',
                'serum_protocol_comparison',
                'cspp_serum',
                'ergothioneine_serum'
            )
            ORDER BY dataset_id, subclass_label
            """
        ).fetchdf()

        serum_archive_chunk_text = (
            "Dataset-level serum context:\n"
            + dataset_context_df.to_string(index=False)
            + "\n\nSubclass-level serum context:\n"
            + subclass_context_df.to_string(index=False)
        )
        documents.append(
            build_document(
                document_id="gaira_serum_context_dataset_context",
                context_type="paper_summary",
                evidence_basis="derived_from_grounding",
                source_dataset_id=(
                    "hcc_serum,serum_ag_colloids,serum_protocol_comparison,cspp_serum,ergothioneine_serum"
                ),
                source_file="dataset_domain_context + subclass_domain_context",
                title="Current GAIRA serum dataset context",
                notes=(
                    "Curated serum pack context compiled from existing GAIRA dataset-domain and subclass-domain context rows."
                ),
                chunks=[
                    (
                        "serum_dataset_context",
                        serum_archive_chunk_text,
                        {"source_kind": "dataset_domain_context"},
                    )
                ],
            )
        )

        grounding_summary_df = connection.execute(
            """
            SELECT experiment_family, class_label, n_spectra
            FROM grounding_class_summary
            WHERE dataset_id = 'serum_ag_colloids_grounding'
              AND class_label IN ('UAfree','UAbound','Hypox','UA','UAiso','UA+HSA','UAiso+HSA')
            ORDER BY experiment_family, class_label
            """
        ).fetchdf()

        documents.append(
            build_document(
                document_id="gaira_serum_context_metabolite_dominance_caveat",
                context_type="caveat",
                evidence_basis="derived_from_grounding",
                source_dataset_id="serum_ag_colloids_grounding",
                source_file="grounding_class_summary",
                title="Serum Ag-colloid metabolite-dominance caveat",
                notes=(
                    "Curated serum-specific caveat derived from the direct serum_ag_colloids grounding families."
                ),
                chunks=[
                    (
                        "metabolite_dominance_caveat",
                        (
                            "Current study-matched serum Ag-colloid grounding already contains explicit uric-acid "
                            "and hypoxanthine-oriented direct references, including Hypox, UA, UAfree, UAbound, UAiso, "
                            "UA+HSA, and UAiso+HSA families. In this serum-specific context, strong small-molecule "
                            "matches around these families should be treated as expected archive-local grounding rather "
                            "than unexpected broad analog hits. This is specific to the current Ag-colloid serum release "
                            "and should not be generalized to all serum SERS substrates.\n\n"
                            + grounding_summary_df.to_string(index=False)
                        ),
                        {"source_kind": "direct_grounding_summary"},
                    )
                ],
            )
        )

        literature_context_df = connection.execute(
            """
            SELECT citation_label, chunk_text
            FROM grounding_support_chunks c
            JOIN grounding_support_documents d
              ON c.document_id = d.document_id
             AND c.dataset_id = d.dataset_id
            WHERE c.dataset_id = 'serum_ag_colloids_literature_grounding'
              AND c.section = 'study_metadata'
            ORDER BY c.document_id
            LIMIT 6
            """
        ).fetchdf()

        documents.append(
            build_document(
                document_id="gaira_serum_context_adsorption_protocol_caveat",
                context_type="caveat",
                evidence_basis="derived_from_grounding_and_literature",
                source_dataset_id="serum_ag_colloids_literature_grounding",
                source_file="grounding_support_chunks::study_metadata",
                title="Serum adsorption and protocol caveat",
                notes=(
                    "Curated serum-specific caveat derived from the serum_ag_colloids literature-support table."
                ),
                chunks=[
                    (
                        "adsorption_protocol_caveat",
                        (
                            "The current serum literature-support layer explicitly tracks metal, substrate, "
                            "deproteinization, colloid/serum ratio, incubation time, and protocol across studies. "
                            "Serum SERS interpretation on silver colloids is therefore protocol-sensitive and "
                            "adsorption-biased: a match can be locally meaningful while still being contingent on "
                            "substrate chemistry and sample-preparation choices. This is one reason the study-matched "
                            "serum Ag-colloid grounding layer should often be interpreted before broad cross-domain "
                            "Raman analogs.\n\nExample study metadata rows:\n"
                            + literature_context_df.to_string(index=False)
                        ),
                        {"source_kind": "literature_support_metadata"},
                    )
                ],
            )
        )

        documents.append(
            build_document(
                document_id="gaira_serum_context_tiering_note",
                context_type="interpretive_note",
                evidence_basis="derived_from_grounding",
                source_dataset_id="serum_ag_colloids_grounding,ramanbiolib",
                source_file="grounding_class_summary + reference_spectra",
                title="Serum-specific evidence tiering note",
                notes=(
                    "Curated note explaining how serum-local grounding should be weighted against broad shared references."
                ),
                chunks=[
                    (
                        "tiering_note",
                        (
                            "For serum queries on Ag-colloid-like SERS data, the current GAIRA_GROUNDING stack has "
                            "two different kinds of direct spectral evidence: broad RamanBioLib analog references and "
                            "study-matched serum Ag-colloid controlled references. The latter share substrate family, "
                            "measurement mode, and serum context with the serum_ag_colloids archive, so they often "
                            "deserve higher interpretive weight for serum-specific reasoning. RamanBioLib remains useful "
                            "for broad molecular analog grounding, but not as a serum-local replacement for study-matched "
                            "Ag-colloid evidence."
                        ),
                        {"source_kind": "evidence_tiering_policy"},
                    )
                ],
            )
        )

        band_patterns = {
            "721_730": ("721- 730", "serum_hypoxanthine_adenine_coa_note"),
            "1001_1013": ("1001-1013", "serum_phenylalanine_note"),
            "1440_1450_1580_1590": ("1440-1450", "serum_high_wavenumber_caveat"),
        }
        for key, (pattern, section_name) in band_patterns.items():
            examples = load_band_examples(connection, pattern=pattern, limit=4)
            if key == "721_730":
                title = "Serum note for the 721-730 cm^-1 region"
                chunk_text = (
                    "Within the current serum literature-support layer, the 721-730 cm^-1 region is repeatedly "
                    "associated with adenine, coenzyme A, or hypoxanthine depending on study context. The current "
                    "study-matched serum grounding layer also contains direct Hypox and multiple uric-acid-related "
                    "families, so this region should be treated as serum-relevant but not single-assignment.\n\n"
                    "Example current support rows:\n" + "\n".join(examples)
                )
            elif key == "1001_1013":
                title = "Serum note for the 1001-1013 cm^-1 region"
                chunk_text = (
                    "Within the current serum literature-support layer, the 1001-1013 cm^-1 region is repeatedly "
                    "reported as phenylalanine across multiple serum SERS studies. In GAIRA_SERUM_CONTEXT this is a "
                    "relatively stable serum-interpretation hint, but still remains literature-backed support rather "
                    "than a definitive identification.\n\nExample current support rows:\n" + "\n".join(examples)
                )
            else:
                title = "Serum caveat for the 1440-1450 and nearby high-wavenumber region"
                chunk_text = (
                    "Within the current serum literature-support layer, the 1440-1450 cm^-1 region and nearby "
                    "high-wavenumber assignments remain mixed, with reports including collagen, lipids, phospholipids, "
                    "proteins, acetoacetate, tryptophan, and hypoxanthine across different serum studies. This region "
                    "should therefore be treated as interpretation-rich but assignment-ambiguous in serum Ag-colloid SERS.\n\n"
                    "Example current support rows:\n" + "\n".join(examples)
                )

            documents.append(
                build_document(
                    document_id=f"gaira_serum_context_band_{key}",
                    context_type="band_note",
                    evidence_basis="derived_from_grounding_and_literature",
                    source_dataset_id="serum_ag_colloids_literature_grounding",
                    source_file="grounding_support_chunks::reported_band_assignments",
                    title=title,
                    notes="Curated serum-specific band note from current literature-support rows.",
                    chunks=[(section_name, chunk_text, {"source_kind": "band_note", "band_pattern": pattern})],
                )
            )

    document_rows = []
    chunk_rows = []
    for document in documents:
        document_rows.append(
            {
                key: value
                for key, value in document.items()
                if key != "chunks"
            }
        )
        for chunk_order, (section, chunk_text, metadata) in enumerate(document["chunks"], start=1):
            chunk_rows.append(
                {
                    "chunk_id": f"{document['document_id']}_chunk_{chunk_order:02d}",
                    "document_id": document["document_id"],
                    "context_layer": CONTEXT_LAYER,
                    "intended_domain": "serum",
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
            "DELETE FROM domain_context_chunks WHERE context_layer = ? AND intended_domain = 'serum'",
            [CONTEXT_LAYER],
        )
        connection.execute(
            "DELETE FROM domain_context_documents WHERE context_layer = ? AND intended_domain = 'serum'",
            [CONTEXT_LAYER],
        )

        connection.register("documents_df", documents_df)
        connection.execute("INSERT INTO domain_context_documents SELECT * FROM documents_df")
        connection.unregister("documents_df")

        connection.register("chunks_df", chunks_df)
        connection.execute("INSERT INTO domain_context_chunks SELECT * FROM chunks_df")
        connection.unregister("chunks_df")

    print("GAIRA_SERUM_CONTEXT ingest complete.")
    print(f"Inserted domain_context_documents rows: {len(documents_df)}")
    print(f"Inserted domain_context_chunks rows: {len(chunks_df)}")


if __name__ == "__main__":
    main()
