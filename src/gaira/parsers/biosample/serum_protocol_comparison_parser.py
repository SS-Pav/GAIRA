import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from gaira.parsers.biosample.base import BiosampleParserBase


class SerumProtocolComparisonParser(BiosampleParserBase):
    """Parser for Zenodo 11143059 human-serum protocol comparison SERS archive."""

    ZIP_NAME = "dataset_serum_spectra.zip"
    INSTRUCTIONS_NAME = "Instructions.docx"
    SCRIPT_NAME = "analysis.R"
    EXPECTED_SPECTRA = 75
    SUBCLASS_LABEL = "protocol_comparison_archive"

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        super().__init__(dataset_id=dataset_id, dataset_root=dataset_root, db_path=db_path)
        self.archive_path = self.dataset_root / self.ZIP_NAME
        self.instructions_path = self.dataset_root / self.INSTRUCTIONS_NAME
        self.script_path = self.dataset_root / self.SCRIPT_NAME
        self.rows = self._build_rows()

    def _read_bwtek_member(self, archive: ZipFile, member_name: str) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(
            io.BytesIO(archive.read(member_name)),
            sep=";",
            decimal=",",
            skiprows=90,
            header=None,
        )
        if df.shape[1] < 8:
            raise ValueError(f"{member_name} does not expose the expected BWtek TXT columns.")
        return df.iloc[:, 3].to_numpy(dtype=float), df.iloc[:, 7].to_numpy(dtype=float)

    def _parse_member_metadata(self, member_name: str) -> dict:
        stem = Path(member_name).stem
        parts = stem.split("_")
        if len(parts) < 4:
            raise ValueError(f"Unexpected filename layout for serum_protocol_comparison: {member_name}")

        spectrum_number = parts[0]
        replicate_id = parts[1]
        protocol_code = parts[2]
        acquisition_date = parts[3]
        return {
            "source_row_id": stem,
            "sample_id": spectrum_number,
            "replicate_id": replicate_id,
            "class_label": protocol_code,
            "acquisition_date": acquisition_date,
        }

    def _build_rows(self) -> list[dict]:
        if not self.archive_path.exists():
            raise FileNotFoundError(f"Missing TXT archive: {self.archive_path}")

        rows: list[dict] = []
        reference_axis: np.ndarray | None = None
        with ZipFile(self.archive_path, "r") as archive:
            members = sorted(member for member in archive.namelist() if member.lower().endswith(".txt"))
            if len(members) != self.EXPECTED_SPECTRA:
                raise ValueError(
                    f"Expected {self.EXPECTED_SPECTRA} TXT members in {self.archive_path}, found {len(members)}."
                )

            for member_name in members:
                wavenumbers, intensity_values = self._read_bwtek_member(archive, member_name)
                if reference_axis is None:
                    reference_axis = wavenumbers
                elif not np.allclose(reference_axis, wavenumbers, atol=0.05):
                    raise ValueError(f"{member_name} does not match the grounded native protocol-comparison axis.")

                parsed = self._parse_member_metadata(member_name)
                biosample_id = f"{self.dataset_id}_{parsed['source_row_id']}"
                rows.append(
                    {
                        "biosample_id": biosample_id,
                        "dataset_id": self.dataset_id,
                        "patient_id": None,
                        "subclass_label": self.SUBCLASS_LABEL,
                        "source_file": f"{self.ZIP_NAME}::{member_name}",
                        "wavenumbers": wavenumbers,
                        "intensity_values": intensity_values,
                        **parsed,
                    }
                )

        return rows

    def _preprocessing_summary(self) -> str:
        return (
            "Raw ingest preserves the native released BWtek TXT spectra using Raman Shift and Dark Subtracted #1 "
            "exactly as exposed in the instrument export. The released analysis.R script shows downstream crop "
            "(300-1900 then 400-1800 cm^-1), ALS baseline correction, and vector normalization, but those steps "
            "are not applied during GAIRA raw ingestion."
        )

    def _base_notes(self, row: dict) -> str:
        return (
            f"Released serum protocol-comparison spectrum from one commercial human serum sample. "
            f"Filename metadata preserve spectrum_number={row['sample_id']}, replicate={row['replicate_id']}, "
            f"protocol_code={row['class_label']}, and acquisition_date={row['acquisition_date']}. "
            "The instructions state that the archive contains 75 spectra spanning five protocols with "
            "15 measurements per protocol. Current GAIRA framing treats the class labels p1-p5 as protocol "
            "factors rather than biology classes."
        )

    def _detect_peaks_for_spectrum(self, intensity_values: np.ndarray, wavenumbers: np.ndarray) -> list[dict]:
        shifted = np.asarray(intensity_values, dtype=float) - float(np.min(intensity_values))
        max_value = float(np.max(shifted))
        if max_value <= 0:
            return []
        normalized = shifted / max_value
        peak_indices, properties = find_peaks(
            normalized,
            prominence=0.05,
            height=0.05,
            distance=5,
        )
        return [
            {
                "peak_rank": peak_rank,
                "peak_cm": float(wavenumbers[peak_index]),
                "peak_intensity": float(properties["peak_heights"][peak_rank - 1]),
                "prominence": float(properties["prominences"][peak_rank - 1]),
            }
            for peak_rank, peak_index in enumerate(peak_indices, start=1)
        ]

    def audit(self) -> None:
        axis = self.rows[0]["wavenumbers"]
        protocol_counts = Counter(row["class_label"] for row in self.rows)
        print("serum_protocol_comparison dataset audit")
        print(f"Dataset root: {self.dataset_root}")
        print(f"TXT archive: {self.archive_path}")
        print(f"Instructions: {self.instructions_path}")
        print(f"R script: {self.script_path}")
        print(f"Native axis: {axis[0]:.2f} to {axis[-1]:.2f} cm^-1 ({len(axis)} points)")
        print("Protocol counts:")
        for protocol_code, count in sorted(protocol_counts.items()):
            print(f"  {protocol_code}: {count}")
        print("This archive is a same-serum multi-protocol SERS release, not a disease benchmark.")

    def extract_metadata(self) -> pd.DataFrame:
        preprocessing_summary = self._preprocessing_summary()
        rows = []
        for row in self.rows:
            wavenumbers = row["wavenumbers"]
            rows.append(
                {
                    "biosample_id": row["biosample_id"],
                    "dataset_id": self.dataset_id,
                    "source_row_id": row["source_row_id"],
                    "sample_id": row["sample_id"],
                    "patient_id": row["patient_id"],
                    "replicate_id": row["replicate_id"],
                    "biosample_type": "serum",
                    "matrix": "serum",
                    "disease_context": None,
                    "class_label": row["class_label"],
                    "subclass_label": row["subclass_label"],
                    "collection_protocol": "commercial human serum sample analyzed under five released SERS protocols",
                    "preparation_protocol": None,
                    "instrument": "BWS465-785S / BTC665N-785S-SYS BWtek-format 785 nm system",
                    "laser_wavelength_nm": "785",
                    "spectral_range": f"{wavenumbers[0]:.2f}-{wavenumbers[-1]:.2f}",
                    "preprocessing_summary": preprocessing_summary,
                    "source_file": row["source_file"],
                    "notes": self._base_notes(row),
                }
            )
        metadata_df = pd.DataFrame(rows)
        print(f"Prepared {len(metadata_df)} biosample metadata rows.")
        return metadata_df

    def extract_spectra(self) -> pd.DataFrame:
        preprocessing_summary = self._preprocessing_summary()
        rows = []
        for row in self.rows:
            wavenumbers = row["wavenumbers"]
            rows.append(
                {
                    "biosample_id": row["biosample_id"],
                    "dataset_id": self.dataset_id,
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
        spectra_df = pd.DataFrame(rows)
        print(f"Prepared {len(spectra_df)} biosample spectra rows.")
        return spectra_df

    def extract_spectrum_points(self) -> pd.DataFrame:
        rows = []
        for row in self.rows:
            for point_index, (wavenumber, intensity) in enumerate(
                zip(row["wavenumbers"], row["intensity_values"]),
                start=1,
            ):
                rows.append(
                    {
                        "biosample_id": row["biosample_id"],
                        "dataset_id": self.dataset_id,
                        "source_row_id": row["source_row_id"],
                        "point_index": point_index,
                        "wavenumber": float(wavenumber),
                        "intensity": float(intensity),
                    }
                )
        points_df = pd.DataFrame(rows)
        print(f"Prepared {len(points_df)} biosample spectrum point rows.")
        return points_df

    def extract_peaks(self) -> pd.DataFrame:
        rows = []
        for row in self.rows:
            for peak_row in self._detect_peaks_for_spectrum(row["intensity_values"], row["wavenumbers"]):
                rows.append(
                    {
                        "biosample_id": row["biosample_id"],
                        "dataset_id": self.dataset_id,
                        "source_row_id": row["source_row_id"],
                        **peak_row,
                    }
                )
        peaks_df = pd.DataFrame(rows)
        print(f"Prepared {len(peaks_df)} biosample peak rows.")
        return peaks_df

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
        metadata_df = self.extract_metadata()
        spectra_df = self.extract_spectra()
        points_df = self.extract_spectrum_points()
        peaks_df = self.extract_peaks()

        with duckdb.connect(str(self.db_path)) as connection:
            for table_name in (
                "biosample_metadata",
                "biosample_spectra",
                "biosample_spectrum_points",
                "biosample_peaks",
            ):
                connection.execute(f"DELETE FROM {table_name} WHERE dataset_id = ?", [self.dataset_id])

            metadata_count = self._insert_dataframe(connection, "biosample_metadata", metadata_df)
            spectra_count = self._insert_dataframe(connection, "biosample_spectra", spectra_df)
            point_count = self._insert_dataframe(connection, "biosample_spectrum_points", points_df)
            peak_count = self._insert_dataframe(connection, "biosample_peaks", peaks_df)

        print("serum_protocol_comparison ingestion complete.")
        print(f"Inserted biosample_metadata rows: {metadata_count}")
        print(f"Inserted biosample_spectra rows: {spectra_count}")
        print(f"Inserted biosample_spectrum_points rows: {point_count}")
        print(f"Inserted biosample_peaks rows: {peak_count}")
