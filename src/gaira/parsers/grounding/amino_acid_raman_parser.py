import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class AminoAcidRamanParser:
    """Parser for the uploaded amino-acid Raman workbook."""

    WORKBOOK_NAME = "aa.xlsx"
    SUPPORT_DOCUMENT_ID = "amino_acid_raman_grounding_doc_001"

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        self.dataset_id = dataset_id
        self.dataset_root = Path(dataset_root)
        self.db_path = Path(db_path)
        self.workbook_path = self.dataset_root / self.WORKBOOK_NAME
        self.rows = self._build_rows()

    def _slugify(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()

    def _infer_biochemical_class(self, label: str) -> str:
        normalized = label.strip().lower()
        amino_terms = {
            "valine",
            "glutamic acid",
            "l-glu",
            "leu",
            "phe",
            "pro",
            "ala",
            "arg",
            "asp",
            "gly",
            "his",
            "met",
            "ser",
            "gluth",
            "trp",
        }
        if normalized in amino_terms:
            return "amino_acid_like_reference"
        if normalized in {"glucose", "ure", "malic acid"}:
            return "small_molecule_comparator_reference"
        if normalized == "alb":
            return "protein_reference"
        return "mixed_comparator_reference"

    def _read_workbook(self) -> pd.DataFrame:
        if not self.workbook_path.exists():
            raise FileNotFoundError(f"Missing amino-acid workbook: {self.workbook_path}")
        df = pd.read_excel(self.workbook_path, sheet_name=0)
        if df.shape[1] < 2:
            raise ValueError("Amino-acid workbook must contain an axis column and at least one spectrum column.")
        df = df.rename(columns={df.columns[0]: "wavenumber"})
        df["wavenumber"] = pd.to_numeric(df["wavenumber"], errors="coerce")
        df = df.dropna(subset=["wavenumber"]).reset_index(drop=True)
        return df

    def _detect_peaks_for_spectrum(self, intensity_values: np.ndarray, wavenumbers: np.ndarray) -> list[dict]:
        shifted = np.asarray(intensity_values, dtype=float) - float(np.min(intensity_values))
        max_value = float(np.max(shifted))
        if max_value <= 0:
            return []
        normalized = shifted / max_value
        peak_indices, properties = find_peaks(normalized, prominence=0.03, height=0.03, distance=5)
        return [
            {
                "peak_cm": float(wavenumbers[peak_index]),
                "peak_intensity": float(properties["peak_heights"][peak_rank - 1]),
                "prominence": float(properties["prominences"][peak_rank - 1]),
            }
            for peak_rank, peak_index in enumerate(peak_indices, start=1)
        ]

    def _build_rows(self) -> list[dict]:
        df = self._read_workbook()
        wavenumbers = df["wavenumber"].to_numpy(dtype=float)
        if not np.all(np.diff(wavenumbers) > 0):
            raise ValueError("Amino-acid workbook axis is not strictly increasing.")

        rows: list[dict] = []
        for column in df.columns[1:]:
            label = str(column).strip()
            if not label:
                continue
            intensities = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
            valid_mask = np.isfinite(wavenumbers) & np.isfinite(intensities)
            x_values = wavenumbers[valid_mask]
            y_values = intensities[valid_mask]
            if len(x_values) < 50:
                continue
            source_row_id = self._slugify(label)
            rows.append(
                {
                    "grounding_id": f"{self.dataset_id}_{source_row_id}",
                    "dataset_id": self.dataset_id,
                    "source_dataset_id": self.dataset_id,
                    "source_row_id": source_row_id,
                    "source_file": self.WORKBOOK_NAME,
                    "experiment_family": "amino_acid_raman_reference_panel",
                    "grounding_role": "controlled_raman_grounding",
                    "compound_label": label,
                    "class_label": label,
                    "concentration_label": None,
                    "replicate_id": None,
                    "wavenumbers": x_values,
                    "intensity_values": y_values,
                    "biochemical_class": self._infer_biochemical_class(label),
                    "peak_rows": self._detect_peaks_for_spectrum(y_values, x_values),
                }
            )
        if not rows:
            raise ValueError("No usable amino-acid Raman spectra were parsed from aa.xlsx.")
        return rows

    def _preprocessing_summary(self) -> str:
        return (
            "Raw grounding ingest preserves the uploaded workbook's native Raman axis and per-column compound "
            "intensity traces exactly as distributed. GAIRA treats this as controlled Raman grounding rather than "
            "biosample evidence or study-matched SERS support."
        )

    def _base_notes(self, row: dict) -> str:
        return (
            f"Workbook-derived Raman reference trace for {row['compound_label']}. "
            f"Biochemical class={row['biochemical_class']}. "
            "Use as amino-acid-associated or comparator Raman grounding only. "
            "Because this panel is spontaneous Raman rather than SERS, modality mismatch caution applies when "
            "comparing directly to SERS biosample datasets."
        )

    def audit(self) -> None:
        axis_min = min(float(np.min(row["wavenumbers"])) for row in self.rows)
        axis_max = max(float(np.max(row["wavenumbers"])) for row in self.rows)
        point_counts = sorted({int(len(row["wavenumbers"])) for row in self.rows})
        family_counts = (
            pd.DataFrame(self.rows)[["compound_label", "biochemical_class"]]
            .groupby("biochemical_class")
            .size()
            .reset_index(name="n")
        )
        print("amino_acid_raman_grounding audit")
        print(f"Workbook: {self.workbook_path}")
        print(f"Parsed spectra: {len(self.rows)}")
        print(f"Native axis: {axis_min:.1f} to {axis_max:.1f} cm^-1; point families={point_counts}")
        print(family_counts.to_string(index=False))

    def extract_metadata(self) -> pd.DataFrame:
        preprocessing_summary = self._preprocessing_summary()
        rows = []
        for row in self.rows:
            wavenumbers = row["wavenumbers"]
            rows.append(
                {
                    "grounding_id": row["grounding_id"],
                    "dataset_id": row["dataset_id"],
                    "source_dataset_id": row["source_dataset_id"],
                    "source_row_id": row["source_row_id"],
                    "experiment_family": row["experiment_family"],
                    "grounding_role": row["grounding_role"],
                    "modality": "Raman",
                    "compound_label": row["compound_label"],
                    "class_label": row["class_label"],
                    "concentration_label": row["concentration_label"],
                    "replicate_id": row["replicate_id"],
                    "source_file": row["source_file"],
                    "biosample_context": "uploaded amino-acid Raman reference workbook",
                    "substrate_type": "spontaneous_raman_reference",
                    "substrate_material": "none",
                    "instrument": "uploaded amino-acid Raman workbook",
                    "laser_wavelength_nm": "unknown",
                    "spectral_range": f"{wavenumbers[0]:.2f}-{wavenumbers[-1]:.2f}",
                    "preprocessing_summary": preprocessing_summary,
                    "notes": self._base_notes(row),
                }
            )
        return pd.DataFrame(rows)

    def extract_spectra(self) -> pd.DataFrame:
        preprocessing_summary = self._preprocessing_summary()
        rows = []
        for row in self.rows:
            wavenumbers = row["wavenumbers"]
            rows.append(
                {
                    "grounding_id": row["grounding_id"],
                    "dataset_id": row["dataset_id"],
                    "source_dataset_id": row["source_dataset_id"],
                    "source_row_id": row["source_row_id"],
                    "x_min": float(wavenumbers[0]),
                    "x_max": float(wavenumbers[-1]),
                    "n_points": int(len(wavenumbers)),
                    "wavenumbers_json": json.dumps([float(value) for value in wavenumbers]),
                    "intensity_json": json.dumps([float(value) for value in row["intensity_values"]]),
                    "normalized_flag": "unknown",
                    "preprocessing_summary": preprocessing_summary,
                }
            )
        return pd.DataFrame(rows)

    def extract_spectrum_points(self) -> pd.DataFrame:
        rows = []
        for row in self.rows:
            for point_index, (wavenumber, intensity) in enumerate(
                zip(row["wavenumbers"], row["intensity_values"]),
                start=1,
            ):
                rows.append(
                    {
                        "grounding_id": row["grounding_id"],
                        "dataset_id": row["dataset_id"],
                        "source_dataset_id": row["source_dataset_id"],
                        "source_row_id": row["source_row_id"],
                        "point_index": point_index,
                        "wavenumber": float(wavenumber),
                        "intensity": float(intensity),
                    }
                )
        return pd.DataFrame(rows)

    def extract_peaks(self) -> pd.DataFrame:
        rows = []
        for row in self.rows:
            for peak_rank, peak_row in enumerate(row["peak_rows"], start=1):
                rows.append(
                    {
                        "grounding_id": row["grounding_id"],
                        "dataset_id": row["dataset_id"],
                        "source_dataset_id": row["source_dataset_id"],
                        "source_row_id": row["source_row_id"],
                        "peak_rank": peak_rank,
                        "peak_cm": float(peak_row["peak_cm"]),
                        "peak_intensity": float(peak_row["peak_intensity"]),
                        "prominence": float(peak_row["prominence"]),
                    }
                )
        return pd.DataFrame(rows)

    def extract_support_documents(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "document_id": self.SUPPORT_DOCUMENT_ID,
                    "dataset_id": self.dataset_id,
                    "source_dataset_id": self.dataset_id,
                    "evidence_family": "amino_acid_raman_reference_support",
                    "evidence_tier": "tier2_interpretive_support",
                    "support_type": "text",
                    "citation_label": "Amino_Acid_Raman_Workbook",
                    "title": "Amino-acid Raman reference workbook note",
                    "authors": None,
                    "year": "local",
                    "journal": "uploaded workbook",
                    "doi": None,
                    "source_file": self.WORKBOOK_NAME,
                    "is_digitized": "no",
                    "use_for_primary_matching": "no",
                    "use_for_supporting_comparison": "yes",
                    "use_for_rag": "yes",
                    "notes": (
                        "Support note for the uploaded amino-acid Raman workbook. "
                        "Used to explain that the asset is controlled Raman grounding and carries modality mismatch "
                        "caution relative to SERS biosample datasets."
                    ),
                }
            ]
        )

    def extract_support_chunks(self) -> pd.DataFrame:
        chunks = [
            (
                "dataset_role",
                (
                    "amino_acid_raman_grounding is a controlled Raman workbook with one native wavenumber axis and "
                    "one spectrum column per compound. It should be used for amino-acid-associated grounding and "
                    "band-level plausibility support rather than as biosample or disease evidence."
                ),
            ),
            (
                "modality_caution",
                (
                    "Because this workbook is spontaneous Raman rather than SERS, direct comparison to SERS biosample "
                    "datasets requires modality mismatch caution. Matches can strengthen biochemical plausibility but "
                    "should not be treated as study-matched surface-enhanced evidence."
                ),
            ),
        ]
        rows = []
        for chunk_order, (section, chunk_text) in enumerate(chunks, start=1):
            rows.append(
                {
                    "chunk_id": f"{self.SUPPORT_DOCUMENT_ID}_chunk_{chunk_order:02d}",
                    "document_id": self.SUPPORT_DOCUMENT_ID,
                    "dataset_id": self.dataset_id,
                    "chunk_order": chunk_order,
                    "section": section,
                    "chunk_text": chunk_text,
                    "metadata_json": json.dumps({"source_kind": "amino_acid_raman_context"}, sort_keys=True),
                }
            )
        return pd.DataFrame(rows)

    def _insert_dataframe(self, connection: duckdb.DuckDBPyConnection, table_name: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        connection.register("temp_df", df)
        connection.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
        connection.unregister("temp_df")
        return int(len(df))

    def ingest(self) -> None:
        metadata_df = self.extract_metadata()
        spectra_df = self.extract_spectra()
        points_df = self.extract_spectrum_points()
        peaks_df = self.extract_peaks()
        support_documents_df = self.extract_support_documents()
        support_chunks_df = self.extract_support_chunks()

        with duckdb.connect(str(self.db_path)) as connection:
            for table_name in (
                "grounding_peaks",
                "grounding_spectrum_points",
                "grounding_spectra",
                "grounding_metadata",
                "grounding_support_chunks",
                "grounding_support_documents",
            ):
                connection.execute(f"DELETE FROM {table_name} WHERE dataset_id = ?", [self.dataset_id])

            self._insert_dataframe(connection, "grounding_metadata", metadata_df)
            self._insert_dataframe(connection, "grounding_spectra", spectra_df)
            self._insert_dataframe(connection, "grounding_spectrum_points", points_df)
            self._insert_dataframe(connection, "grounding_peaks", peaks_df)
            self._insert_dataframe(connection, "grounding_support_documents", support_documents_df)
            self._insert_dataframe(connection, "grounding_support_chunks", support_chunks_df)

        print(f"Inserted grounding_metadata rows: {len(metadata_df)}")
        print(f"Inserted grounding_spectra rows: {len(spectra_df)}")
        print(f"Inserted grounding_spectrum_points rows: {len(points_df)}")
        print(f"Inserted grounding_peaks rows: {len(peaks_df)}")
        print(f"Inserted grounding_support_documents rows: {len(support_documents_df)}")
        print(f"Inserted grounding_support_chunks rows: {len(support_chunks_df)}")
