from pathlib import Path

import duckdb
import pandas as pd

from gaira.parsers.knowledge.base import KnowledgeParserBase


class RamanKnowledgeCoreParser(KnowledgeParserBase):
    """Parser for a local curated Raman knowledge package stored as CSV files."""

    REQUIRED_FILES = {
        "sources": "sources.csv",
        "peak_assignments": "peak_assignments.csv",
        "biomarker_claims": "biomarker_claims.csv",
        "confounder_notes": "confounder_notes.csv",
    }
    OPTIONAL_FILES = {
        "knowledge_chunks": "knowledge_chunks.csv",
        "semantic_regions": "semantic_regions.csv",
        "dataset_context": "dataset_context.csv",
    }

    REQUIRED_COLUMNS = {
        "sources": [
            "source_id",
            "dataset_id",
            "source_type",
            "title",
            "authors",
            "year",
            "journal",
            "doi",
            "url",
            "citation",
            "license",
            "notes",
        ],
        "peak_assignments": [
            "assignment_id",
            "source_id",
            "dataset_id",
            "peak_cm",
            "tolerance_cm",
            "assigned_molecule",
            "assigned_group",
            "matrix_context",
            "confidence_text",
            "evidence_text",
        ],
        "biomarker_claims": [
            "claim_id",
            "source_id",
            "dataset_id",
            "biomarker_name",
            "disease_context",
            "sample_type",
            "spectral_region",
            "claim_text",
            "evidence_strength",
            "notes",
        ],
        "confounder_notes": [
            "confounder_id",
            "source_id",
            "dataset_id",
            "confounder_name",
            "applies_to",
            "note_text",
            "mitigation_text",
        ],
        "knowledge_chunks": [
            "chunk_id",
            "source_id",
            "dataset_id",
            "section",
            "chunk_text",
            "chunk_order",
            "page_label",
            "metadata_json",
        ],
        "semantic_regions": [
            "region_id",
            "dataset_id",
            "region_label",
            "region_min_cm",
            "region_max_cm",
            "dominant_group",
            "secondary_groups",
            "typical_examples",
            "interpretation_note",
            "caution_note",
        ],
        "dataset_context": [
            "context_id",
            "dataset_id",
            "target_dataset_id",
            "modality",
            "sample_type",
            "measurement_state",
            "substrate_type",
            "enhancement_mode",
            "known_biases",
            "region_caution_450_700",
            "region_caution_700_900",
            "region_caution_900_1100",
            "region_caution_1100_1300",
            "region_caution_1300_1500",
            "region_caution_1500_1700",
            "interpretation_note",
            "do_not_overclaim_note",
        ],
    }

    def _dataset_dir(self) -> Path:
        """Return the local folder that should hold the curated knowledge CSV package."""
        return self.dataset_root

    def _path_for(self, key: str) -> Path:
        """Return the file path for one required or optional CSV."""
        file_name = self.REQUIRED_FILES.get(key) or self.OPTIONAL_FILES.get(key)
        return self._dataset_dir() / file_name

    def _missing_required_files(self) -> list[Path]:
        """List required CSV files that are not yet present."""
        return [self._path_for(key) for key in self.REQUIRED_FILES if not self._path_for(key).exists()]

    def _read_csv(self, key: str, required: bool = True) -> pd.DataFrame:
        """Read one CSV and validate the required schema."""
        csv_path = self._path_for(key)
        if not csv_path.exists():
            if required:
                raise FileNotFoundError(f"Required knowledge CSV not found: {csv_path}")
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS[key])

        df = pd.read_csv(csv_path)
        missing_columns = [column for column in self.REQUIRED_COLUMNS[key] if column not in df.columns]
        if missing_columns:
            raise ValueError(
                f"{csv_path.name} is missing required columns: {', '.join(missing_columns)}"
            )

        df = df[self.REQUIRED_COLUMNS[key]].copy()
        if "dataset_id" in df.columns:
            df["dataset_id"] = self.dataset_id
        return df

    def audit(self) -> None:
        """Inspect the local CSV package and report what is available."""
        dataset_dir = self._dataset_dir()
        print(f"Raman knowledge core audit for: {dataset_dir}")

        if not dataset_dir.exists():
            print("Knowledge package folder is missing.")
            print("Create this folder and add the curated CSV files:")
            for file_name in self.REQUIRED_FILES.values():
                print(f"  - {dataset_dir / file_name}")
            print("Optional:")
            for file_name in self.OPTIONAL_FILES.values():
                print(f"  - {dataset_dir / file_name}")
            return

        print("Files found:")
        for csv_path in sorted(dataset_dir.glob("*.csv")):
            print(f"  - {csv_path.name}")

        missing_files = self._missing_required_files()
        if missing_files:
            print("Missing required files:")
            for missing_path in missing_files:
                print(f"  - {missing_path.name}")
        else:
            print("All required knowledge CSV files are present.")

    def extract_metadata(self):
        raise NotImplementedError(
            "Knowledge datasets use extract_sources(), not extract_metadata()."
        )

    def extract_spectra(self):
        raise NotImplementedError(
            "Knowledge datasets do not use extract_spectra() in the biosample/reference sense."
        )

    def extract_sources(self) -> pd.DataFrame:
        """Load the source metadata table."""
        sources_df = self._read_csv("sources", required=True)
        print(f"Prepared knowledge_sources rows: {len(sources_df)}")
        return sources_df

    def extract_chunks(self) -> pd.DataFrame:
        """Load optional retrieval-ready text chunks."""
        chunks_df = self._read_csv("knowledge_chunks", required=False)
        print(f"Prepared knowledge_chunks rows: {len(chunks_df)}")
        return chunks_df

    def extract_peak_assignments(self) -> pd.DataFrame:
        """Load literature-derived peak assignment rows."""
        assignments_df = self._read_csv("peak_assignments", required=True)
        assignments_df["peak_cm"] = pd.to_numeric(assignments_df["peak_cm"], errors="coerce")
        assignments_df["tolerance_cm"] = pd.to_numeric(
            assignments_df["tolerance_cm"], errors="coerce"
        )
        assignments_df = assignments_df.dropna(subset=["peak_cm", "tolerance_cm"]).reset_index(drop=True)
        print(f"Prepared peak_assignments rows: {len(assignments_df)}")
        return assignments_df

    def extract_biomarker_claims(self) -> pd.DataFrame:
        """Load biomarker claim rows."""
        claims_df = self._read_csv("biomarker_claims", required=True)
        print(f"Prepared biomarker_claims rows: {len(claims_df)}")
        return claims_df

    def extract_confounder_notes(self) -> pd.DataFrame:
        """Load confounder note rows."""
        confounder_df = self._read_csv("confounder_notes", required=True)
        print(f"Prepared confounder_notes rows: {len(confounder_df)}")
        return confounder_df

    def extract_semantic_regions(self) -> pd.DataFrame:
        """Load optional curated semantic Raman regions."""
        semantic_df = self._read_csv("semantic_regions", required=False)
        if not semantic_df.empty:
            semantic_df["region_min_cm"] = pd.to_numeric(
                semantic_df["region_min_cm"], errors="coerce"
            )
            semantic_df["region_max_cm"] = pd.to_numeric(
                semantic_df["region_max_cm"], errors="coerce"
            )
            semantic_df = semantic_df.dropna(
                subset=["region_min_cm", "region_max_cm"]
            ).reset_index(drop=True)
        print(f"Prepared semantic_regions rows: {len(semantic_df)}")
        return semantic_df

    def extract_dataset_context(self) -> pd.DataFrame:
        """Load optional dataset-specific interpretation context rows."""
        context_df = self._read_csv("dataset_context", required=False)
        print(f"Prepared dataset_context rows: {len(context_df)}")
        return context_df

    def ingest(self) -> None:
        """Replace the knowledge-layer rows for this dataset with the curated CSV contents."""
        missing_files = self._missing_required_files()
        if missing_files:
            print("Cannot ingest Raman knowledge core because required files are missing:")
            for missing_path in missing_files:
                print(f"  - {missing_path}")
            print("See docs/raman_knowledge_core_template.md for the required schemas.")
            return

        sources_df = self.extract_sources()
        chunks_df = self.extract_chunks()
        assignments_df = self.extract_peak_assignments()
        claims_df = self.extract_biomarker_claims()
        confounder_df = self.extract_confounder_notes()
        semantic_df = self.extract_semantic_regions()
        context_df = self.extract_dataset_context()

        with duckdb.connect(str(self.db_path)) as connection:
            connection.execute("DELETE FROM dataset_context WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM semantic_regions WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM knowledge_chunks WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM peak_assignments WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM biomarker_claims WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM confounder_notes WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM knowledge_sources WHERE dataset_id = ?", [self.dataset_id])

            for table_name, df in [
                ("knowledge_sources", sources_df),
                ("peak_assignments", assignments_df),
                ("biomarker_claims", claims_df),
                ("confounder_notes", confounder_df),
                ("knowledge_chunks", chunks_df),
                ("semantic_regions", semantic_df),
                ("dataset_context", context_df),
            ]:
                if df.empty:
                    continue
                temp_name = f"tmp_{table_name}"
                connection.register(temp_name, df)
                connection.execute(f"INSERT INTO {table_name} SELECT * FROM {temp_name}")
                connection.unregister(temp_name)

        print("Raman knowledge core ingestion complete.")
        print(f"Inserted knowledge_sources rows: {len(sources_df)}")
        print(f"Inserted peak_assignments rows: {len(assignments_df)}")
        print(f"Inserted biomarker_claims rows: {len(claims_df)}")
        print(f"Inserted confounder_notes rows: {len(confounder_df)}")
        print(f"Inserted knowledge_chunks rows: {len(chunks_df)}")
        print(f"Inserted semantic_regions rows: {len(semantic_df)}")
        print(f"Inserted dataset_context rows: {len(context_df)}")
