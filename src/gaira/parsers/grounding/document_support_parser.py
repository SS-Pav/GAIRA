import json
import re
from pathlib import Path

import duckdb
import pandas as pd


class DocumentSupportParser:
    """Support-only grounding parser for paper resources without clean numeric spectral packages."""

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        self.dataset_id = dataset_id
        self.dataset_root = Path(dataset_root)
        self.db_path = Path(db_path)
        self.documents = self._build_documents()

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _strip_html(self, html_text: str) -> str:
        no_script = re.sub(r"<script.*?>.*?</script>", " ", html_text, flags=re.DOTALL | re.IGNORECASE)
        no_style = re.sub(r"<style.*?>.*?</style>", " ", no_script, flags=re.DOTALL | re.IGNORECASE)
        without_tags = re.sub(r"<[^>]+>", " ", no_style)
        return self._clean_text(without_tags)

    def _extract_pubmed_abstract(self, html_text: str) -> str | None:
        match = re.search(
            r'<div class="abstract-content selected"[^>]*id="eng-abstract"[^>]*>(.*?)</div>',
            html_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match:
            return self._clean_text(self._strip_html(match.group(1)))

        meta_match = re.search(r'<meta name="description" content="([^"]+)"', html_text)
        if meta_match:
            return self._clean_text(meta_match.group(1))
        return None

    def _authors_from_crossref(self, payload: dict) -> str | None:
        author_rows = payload.get("message", {}).get("author", [])
        if not author_rows:
            return None
        formatted = []
        for author in author_rows[:8]:
            given = str(author.get("given", "")).strip()
            family = str(author.get("family", "")).strip()
            name = " ".join(part for part in [given, family] if part).strip()
            if name:
                formatted.append(name)
        return ", ".join(formatted) if formatted else None

    def _year_from_crossref(self, payload: dict) -> str | None:
        message = payload.get("message", {})
        for key in ("published-print", "published-online", "issued"):
            parts = message.get(key, {}).get("date-parts", [])
            if parts and parts[0]:
                return str(parts[0][0])
        return None

    def _build_workingpaper_document(self) -> dict:
        record_path = self.dataset_root / "record_14294417.json"
        pdf_path = self.dataset_root / "comparing1.pdf"
        if not record_path.exists():
            raise FileNotFoundError(f"Missing working-paper metadata file: {record_path}")
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing working-paper PDF: {pdf_path}")

        record = json.loads(record_path.read_text(encoding="utf-8"))
        metadata = record.get("metadata", {})
        description = self._clean_text(self._strip_html(metadata.get("description", "")))
        title = metadata.get("title", "SERS fingerprint working paper support")
        authors = ", ".join(
            creator.get("name", "").strip()
            for creator in metadata.get("creators", [])
            if creator.get("name")
        )
        year = str(metadata.get("publication_date", ""))[:4] or None
        doi = metadata.get("doi") or record.get("doi")

        return {
            "document_id": f"{self.dataset_id}_doc_001",
            "dataset_id": self.dataset_id,
            "source_dataset_id": self.dataset_id,
            "evidence_family": "literature_support",
            "evidence_tier": "tier2_literature_support",
            "support_type": "text",
            "citation_label": "Sparavigna_2024_workingpaper",
            "title": title,
            "authors": authors or None,
            "year": year,
            "journal": "Zenodo working paper",
            "doi": doi,
            "source_file": "record_14294417.json|comparing1.pdf",
            "is_digitized": "no",
            "use_for_primary_matching": "no",
            "use_for_supporting_comparison": "yes",
            "use_for_rag": "yes",
            "notes": (
                "Support-only grounding document from Zenodo record 14294417. The inspected record exposes "
                "one PDF working paper and no clean numeric spectral package, so GAIRA keeps it as tier-2 "
                "literature support rather than direct spectral evidence."
            ),
            "chunks": [
                (
                    "record_description",
                    description,
                    {"source_kind": "zenodo_record_description"},
                ),
                (
                    "availability_note",
                    (
                        "The inspected Zenodo record exposes a single PDF file comparing selected metabolite "
                        "SERS fingerprints to Raman references. No downloadable numeric spectra or table-backed "
                        "reference package were found in this pass, so this resource remains support-only."
                    ),
                    {"source_kind": "pass1_availability_assessment"},
                ),
            ],
        }

    def _build_sers24_document(self) -> dict:
        crossref_path = self.dataset_root / "crossref_10_1016_j_saa_2023_123587.json"
        pubmed_path = self.dataset_root / "pubmed_37918093.html"
        if not crossref_path.exists():
            raise FileNotFoundError(f"Missing Crossref metadata file: {crossref_path}")
        if not pubmed_path.exists():
            raise FileNotFoundError(f"Missing PubMed HTML file: {pubmed_path}")

        crossref_payload = json.loads(crossref_path.read_text(encoding="utf-8"))
        pubmed_html = pubmed_path.read_text(encoding="utf-8")
        message = crossref_payload.get("message", {})
        title = (message.get("title") or ["SERS database of 24 metabolites"])[0]
        abstract_text = self._extract_pubmed_abstract(pubmed_html) or (
            "No PubMed abstract text could be recovered from the downloaded page."
        )
        journal = ((message.get("container-title") or [None])[0]) or None
        doi = message.get("DOI")
        authors = self._authors_from_crossref(crossref_payload)
        year = self._year_from_crossref(crossref_payload)

        metadata_summary = self._clean_text(
            f"title: {title} | journal: {journal or 'unknown'} | doi: {doi or 'unknown'} | year: {year or 'unknown'}"
        )

        return {
            "document_id": f"{self.dataset_id}_doc_001",
            "dataset_id": self.dataset_id,
            "source_dataset_id": self.dataset_id,
            "evidence_family": "literature_support",
            "evidence_tier": "tier2_literature_support",
            "support_type": "text",
            "citation_label": "Zhang_2023_24metabolites",
            "title": title,
            "authors": authors,
            "year": year,
            "journal": journal,
            "doi": doi,
            "source_file": "pubmed_37918093.html|crossref_10_1016_j_saa_2023_123587.json",
            "is_digitized": "no",
            "use_for_primary_matching": "no",
            "use_for_supporting_comparison": "yes",
            "use_for_rag": "yes",
            "notes": (
                "Support-only grounding document for the 24-metabolite SERS database paper. The pass recovered "
                "article metadata and abstract text but did not find a clean downloadable numeric spectral package, "
                "so the resource remains tier-2 support rather than direct spectral evidence."
            ),
            "chunks": [
                (
                    "abstract",
                    abstract_text,
                    {"source_kind": "pubmed_abstract"},
                ),
                (
                    "metadata_summary",
                    metadata_summary,
                    {"source_kind": "crossref_metadata"},
                ),
                (
                    "availability_note",
                    (
                        "This pass did not find a downloadable spreadsheet, database file, or numeric supplementary "
                        "package for the 24-metabolite SERS database. GAIRA therefore integrates the paper as "
                        "support-only grounding context."
                    ),
                    {"source_kind": "pass1_availability_assessment"},
                ),
            ],
        }

    def _build_documents(self) -> list[dict]:
        if self.dataset_id == "sers_fingerprint_workingpaper_support":
            return [self._build_workingpaper_document()]
        if self.dataset_id == "sers24_metabolite_support":
            return [self._build_sers24_document()]
        raise ValueError(f"Unsupported support-only grounding dataset_id: {self.dataset_id}")

    def audit(self) -> None:
        print(f"{self.dataset_id} audit")
        print(f"Dataset root: {self.dataset_root}")
        print(f"Prepared support documents: {len(self.documents)}")
        for document in self.documents:
            print(f"  {document['citation_label']} | {document['title']}")

    def extract_documents(self) -> pd.DataFrame:
        rows = []
        for document in self.documents:
            rows.append(
                {
                    "document_id": document["document_id"],
                    "dataset_id": document["dataset_id"],
                    "source_dataset_id": document["source_dataset_id"],
                    "evidence_family": document["evidence_family"],
                    "evidence_tier": document["evidence_tier"],
                    "support_type": document["support_type"],
                    "citation_label": document["citation_label"],
                    "title": document["title"],
                    "authors": document["authors"],
                    "year": document["year"],
                    "journal": document["journal"],
                    "doi": document["doi"],
                    "source_file": document["source_file"],
                    "is_digitized": document["is_digitized"],
                    "use_for_primary_matching": document["use_for_primary_matching"],
                    "use_for_supporting_comparison": document["use_for_supporting_comparison"],
                    "use_for_rag": document["use_for_rag"],
                    "notes": document["notes"],
                }
            )
        return pd.DataFrame(rows)

    def extract_chunks(self) -> pd.DataFrame:
        rows = []
        for document in self.documents:
            for chunk_order, (section, chunk_text, metadata) in enumerate(document["chunks"], start=1):
                rows.append(
                    {
                        "chunk_id": f"{document['document_id']}_chunk_{chunk_order:02d}",
                        "document_id": document["document_id"],
                        "dataset_id": document["dataset_id"],
                        "chunk_order": chunk_order,
                        "section": section,
                        "chunk_text": chunk_text,
                        "metadata_json": json.dumps(metadata, sort_keys=True),
                    }
                )
        return pd.DataFrame(rows)

    def _insert_dataframe(
        self,
        connection: duckdb.DuckDBPyConnection,
        table_name: str,
        df: pd.DataFrame,
    ) -> int:
        if df.empty:
            return 0
        connection.register("temp_df", df)
        connection.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
        connection.unregister("temp_df")
        return int(len(df))

    def ingest(self) -> None:
        documents_df = self.extract_documents()
        chunks_df = self.extract_chunks()

        with duckdb.connect(str(self.db_path)) as connection:
            for table_name in (
                "grounding_support_documents",
                "grounding_support_chunks",
                "grounding_support_spectra",
                "grounding_support_spectrum_points",
            ):
                connection.execute(f"DELETE FROM {table_name} WHERE dataset_id = ?", [self.dataset_id])

            document_count = self._insert_dataframe(connection, "grounding_support_documents", documents_df)
            chunk_count = self._insert_dataframe(connection, "grounding_support_chunks", chunks_df)

        print(f"{self.dataset_id} ingestion complete.")
        print(f"Inserted grounding_support_documents rows: {document_count}")
        print(f"Inserted grounding_support_chunks rows: {chunk_count}")
        print("Inserted grounding_support_spectra rows: 0")
        print("Inserted grounding_support_spectrum_points rows: 0")
