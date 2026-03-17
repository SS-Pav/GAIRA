import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

import duckdb
import numpy as np
import pandas as pd


class SerumAgColloidsLiteratureGroundingParser:
    """Tier-2 literature-support parser for deferred serum_ag_colloids archive folders."""

    DATASET_ZIP = "dataset_spectral_data.zip"
    SOURCE_DATASET_ID = "serum_ag_colloids"
    ARTICLES_MEMBER = "literature/articles.csv"
    DIGITIZED_MEMBERS = [
        "digitized literature spectra/Gelder_2007.csv",
        "digitized literature spectra/Kim_1987.csv",
        "digitized literature spectra/Stewart_1999.csv",
    ]
    BAND_TEXT_COLUMNS = [
        "489-500",
        "530-535",
        "583-593",
        "631-641",
        "721- 730",
        "802-817",
        "884-893",
        "956-960",
        "1001-1013",
        "1126-1138",
        "1205-1210",
        "1220",
        "1440-1450",
        "1580-1590",
    ]

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        self.dataset_id = dataset_id
        self.dataset_root = Path(dataset_root)
        self.db_path = Path(db_path)
        self.dataset_zip = self.dataset_root / self.DATASET_ZIP
        self.article_rows, self.digitized_rows = self._build_rows()

    def _load_articles_df(self, archive: ZipFile) -> pd.DataFrame:
        raw = archive.read(self.ARTICLES_MEMBER)
        return pd.read_csv(io.BytesIO(raw))

    def _load_digitized_df(self, archive: ZipFile, member_name: str) -> pd.DataFrame:
        raw = archive.read(member_name)
        df = pd.read_csv(io.BytesIO(raw))
        df.columns = [str(column).strip().lower() for column in df.columns]
        if not {"x", "y"}.issubset(df.columns):
            raise ValueError(f"{member_name} does not expose x/y digitized columns.")
        return df

    def _clean_text(self, value: object) -> str | None:
        if value is None:
            return None
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return re.sub(r"\s+", " ", text).strip()

    def _citation_label_from_article_row(self, row: pd.Series, row_index: int) -> str:
        author = self._clean_text(row.get("first author")) or "unknown_author"
        year = self._clean_text(row.get("year")) or "unknown_year"
        author_slug = re.sub(r"[^A-Za-z0-9]+", "_", author.split(",")[0].split()[0]).strip("_")
        return f"{author_slug}_{year}_{row_index + 1}"

    def _build_article_chunks(self, row: pd.Series) -> list[tuple[str, str, dict]]:
        metadata_bits = []
        for key in [
            "title",
            "journal",
            "year",
            "doi",
            "metal",
            "substrate",
            "colloid concentration",
            "laser",
            "Deproteinization",
            "Volume ratio colloid/serum",
            "incubation time (min)",
            "Protocol",
        ]:
            value = self._clean_text(row.get(key))
            if value:
                metadata_bits.append(f"{key}: {value}")

        band_bits = []
        for column_name in self.BAND_TEXT_COLUMNS:
            value = self._clean_text(row.get(column_name))
            if value and value.lower() != "unassigned":
                band_bits.append(f"{column_name}: {value}")

        chunks: list[tuple[str, str, dict]] = []
        if metadata_bits:
            chunks.append(
                (
                    "study_metadata",
                    " | ".join(metadata_bits),
                    {"chunk_kind": "structured_article_metadata"},
                )
            )
        if band_bits:
            chunks.append(
                (
                    "reported_band_assignments",
                    " | ".join(band_bits),
                    {"chunk_kind": "reported_band_assignments"},
                )
            )
        return chunks

    def _build_rows(self) -> tuple[list[dict], list[dict]]:
        if not self.dataset_zip.exists():
            raise FileNotFoundError(f"Missing spectral archive: {self.dataset_zip}")

        article_rows: list[dict] = []
        digitized_rows: list[dict] = []
        with ZipFile(self.dataset_zip, "r") as archive:
            articles_df = self._load_articles_df(archive)
            for row_index, article_row in articles_df.iterrows():
                citation_label = self._citation_label_from_article_row(article_row, row_index)
                document_id = f"{self.dataset_id}_article_{row_index + 1:03d}"
                title = self._clean_text(article_row.get("title"))
                authors = self._clean_text(article_row.get("first author"))
                year = self._clean_text(article_row.get("year"))
                journal = self._clean_text(article_row.get("journal"))
                doi = self._clean_text(article_row.get("doi"))
                notes = (
                    "Released structured literature-support row from articles.csv. This is tier-2 "
                    "supporting evidence used for interpretive context and future RAG, not a primary "
                    "matching target."
                )
                article_rows.append(
                    {
                        "document_id": document_id,
                        "dataset_id": self.dataset_id,
                        "source_dataset_id": self.SOURCE_DATASET_ID,
                        "evidence_family": "literature",
                        "evidence_tier": "tier2_literature_support",
                        "support_type": "text",
                        "citation_label": citation_label,
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "journal": journal,
                        "doi": doi,
                        "source_file": self.ARTICLES_MEMBER,
                        "is_digitized": "no",
                        "use_for_primary_matching": "no",
                        "use_for_supporting_comparison": "yes",
                        "use_for_rag": "yes",
                        "notes": notes,
                        "chunks": self._build_article_chunks(article_row),
                    }
                )

            for member_name in self.DIGITIZED_MEMBERS:
                df = self._load_digitized_df(archive, member_name)
                citation_label = Path(member_name).stem
                document_id = f"{self.dataset_id}_digitized_{citation_label.lower()}"
                wavenumbers = df["x"].to_numpy(dtype=float)
                intensities = df["y"].to_numpy(dtype=float)
                digitized_rows.append(
                    {
                        "document_id": document_id,
                        "dataset_id": self.dataset_id,
                        "source_dataset_id": self.SOURCE_DATASET_ID,
                        "evidence_family": "digitized_literature_spectra",
                        "evidence_tier": "tier2_literature_support",
                        "support_type": "digitized_spectrum",
                        "citation_label": citation_label,
                        "title": f"Digitized literature spectrum: {citation_label}",
                        "authors": None,
                        "year": re.findall(r"(\d{4})", citation_label)[0] if re.findall(r"(\d{4})", citation_label) else None,
                        "journal": None,
                        "doi": None,
                        "source_file": member_name,
                        "is_digitized": "yes",
                        "use_for_primary_matching": "no",
                        "use_for_supporting_comparison": "yes",
                        "use_for_rag": "yes",
                        "notes": (
                            "Released digitized literature spectrum from the serum_ag_colloids archive. "
                            "This is secondary-confidence supporting evidence and must not be treated as "
                            "primary spectral ground truth."
                        ),
                        "chunk_text": (
                            f"Digitized literature support trace from {citation_label}. Source member: "
                            f"{member_name}. Stored as tier-2 supporting comparison evidence only."
                        ),
                        "wavenumbers": wavenumbers,
                        "intensity_values": intensities,
                    }
                )

        return article_rows, digitized_rows

    def audit(self) -> None:
        print("serum_ag_colloids_literature_grounding audit")
        print(f"Dataset root: {self.dataset_root}")
        print(f"Source archive: {self.dataset_zip}")
        print(f"Literature article rows: {len(self.article_rows)}")
        print(f"Digitized spectra: {len(self.digitized_rows)}")
        if self.digitized_rows:
            first_axis = self.digitized_rows[0]["wavenumbers"]
            print(
                "Digitized spectrum preview axis: "
                f"{float(np.min(first_axis)):.2f} to {float(np.max(first_axis)):.2f} cm^-1 "
                f"({len(first_axis)} points in first file)"
            )

    def extract_documents(self) -> pd.DataFrame:
        rows = []
        for row in self.article_rows + self.digitized_rows:
            rows.append(
                {
                    "document_id": row["document_id"],
                    "dataset_id": row["dataset_id"],
                    "source_dataset_id": row["source_dataset_id"],
                    "evidence_family": row["evidence_family"],
                    "evidence_tier": row["evidence_tier"],
                    "support_type": row["support_type"],
                    "citation_label": row["citation_label"],
                    "title": row["title"],
                    "authors": row["authors"],
                    "year": row["year"],
                    "journal": row["journal"],
                    "doi": row["doi"],
                    "source_file": row["source_file"],
                    "is_digitized": row["is_digitized"],
                    "use_for_primary_matching": row["use_for_primary_matching"],
                    "use_for_supporting_comparison": row["use_for_supporting_comparison"],
                    "use_for_rag": row["use_for_rag"],
                    "notes": row["notes"],
                }
            )
        return pd.DataFrame(rows)

    def extract_chunks(self) -> pd.DataFrame:
        rows = []
        for row in self.article_rows:
            for chunk_order, (section, chunk_text, metadata) in enumerate(row["chunks"], start=1):
                rows.append(
                    {
                        "chunk_id": f"{row['document_id']}_chunk_{chunk_order:02d}",
                        "document_id": row["document_id"],
                        "dataset_id": row["dataset_id"],
                        "chunk_order": chunk_order,
                        "section": section,
                        "chunk_text": chunk_text,
                        "metadata_json": json.dumps(metadata, sort_keys=True),
                    }
                )
        for row in self.digitized_rows:
            rows.append(
                {
                    "chunk_id": f"{row['document_id']}_chunk_01",
                    "document_id": row["document_id"],
                    "dataset_id": row["dataset_id"],
                    "chunk_order": 1,
                    "section": "digitized_spectrum_note",
                    "chunk_text": row["chunk_text"],
                    "metadata_json": json.dumps(
                        {
                            "chunk_kind": "digitized_support_note",
                            "is_digitized": True,
                            "use_for_primary_matching": False,
                        },
                        sort_keys=True,
                    ),
                }
            )
        return pd.DataFrame(rows)

    def extract_support_spectra(self) -> pd.DataFrame:
        rows = []
        for row in self.digitized_rows:
            wavenumbers = row["wavenumbers"]
            intensity_values = row["intensity_values"]
            rows.append(
                {
                    "support_spectrum_id": f"{row['document_id']}_spectrum",
                    "document_id": row["document_id"],
                    "dataset_id": row["dataset_id"],
                    "source_dataset_id": row["source_dataset_id"],
                    "evidence_family": row["evidence_family"],
                    "citation_label": row["citation_label"],
                    "x_min": float(np.min(wavenumbers)),
                    "x_max": float(np.max(wavenumbers)),
                    "n_points": int(len(wavenumbers)),
                    "wavenumbers_json": json.dumps([float(value) for value in wavenumbers]),
                    "intensity_json": json.dumps([float(value) for value in intensity_values]),
                    "is_digitized": row["is_digitized"],
                    "use_for_primary_matching": row["use_for_primary_matching"],
                    "use_for_supporting_comparison": row["use_for_supporting_comparison"],
                    "use_for_rag": row["use_for_rag"],
                    "notes": row["notes"],
                }
            )
        return pd.DataFrame(rows)

    def extract_support_points(self) -> pd.DataFrame:
        rows = []
        for row in self.digitized_rows:
            support_spectrum_id = f"{row['document_id']}_spectrum"
            for point_index, (wavenumber, intensity) in enumerate(
                zip(row["wavenumbers"], row["intensity_values"]),
                start=1,
            ):
                rows.append(
                    {
                        "support_spectrum_id": support_spectrum_id,
                        "document_id": row["document_id"],
                        "dataset_id": row["dataset_id"],
                        "point_index": point_index,
                        "wavenumber": float(wavenumber),
                        "intensity": float(intensity),
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
        support_spectra_df = self.extract_support_spectra()
        support_points_df = self.extract_support_points()

        with duckdb.connect(str(self.db_path)) as connection:
            for table_name in (
                "grounding_support_documents",
                "grounding_support_chunks",
                "grounding_support_spectra",
                "grounding_support_spectrum_points",
            ):
                connection.execute(f"DELETE FROM {table_name} WHERE dataset_id = ?", [self.dataset_id])

            document_count = self._insert_dataframe(
                connection, "grounding_support_documents", documents_df
            )
            chunk_count = self._insert_dataframe(connection, "grounding_support_chunks", chunks_df)
            spectrum_count = self._insert_dataframe(
                connection, "grounding_support_spectra", support_spectra_df
            )
            point_count = self._insert_dataframe(
                connection, "grounding_support_spectrum_points", support_points_df
            )

        print("serum_ag_colloids_literature_grounding ingestion complete.")
        print(f"Inserted grounding_support_documents rows: {document_count}")
        print(f"Inserted grounding_support_chunks rows: {chunk_count}")
        print(f"Inserted grounding_support_spectra rows: {spectrum_count}")
        print(f"Inserted grounding_support_spectrum_points rows: {point_count}")
