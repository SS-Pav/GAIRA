import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from gaira.parsers.biosample.base import BiosampleParserBase


class CCAHCCLMSerumSERSParser(BiosampleParserBase):
    """Parser for the multi-class cholangiocarcinoma/HCC/LM/healthy serum SERS archive."""

    ZIP_NAME = "Combination of label-free SERS-based nanosensor an.zip"
    SUBCLASS_LABEL = "released_zip_archive"
    CLASS_MAP = {
        "CCA (Cholangiocarcinoma)": "cca",
        "HCC (Hepatocellular carcinoma)": "hcc",
        "LM (liver metastasis)": "lm",
        "NC (Healthy control)": "healthy_control",
    }

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        super().__init__(dataset_id=dataset_id, dataset_root=dataset_root, db_path=db_path)
        self.archive_path = self.dataset_root / self.ZIP_NAME
        self.rows = self._build_rows()

    def _member_is_primary_spectrum(self, member_name: str) -> bool:
        parts = [part for part in member_name.split("/") if part]
        if "Try dilute condition" in parts:
            return False
        if not member_name.lower().endswith(".txt"):
            return False
        if len(parts) < 4:
            return False
        return Path(member_name).name.startswith("SER-")

    def _parse_single_spectrum(self, member_name: str, text: str) -> tuple[np.ndarray, list[tuple[str, np.ndarray, dict]]]:
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"{member_name} is empty.")

        first_tokens = [token for token in lines[0].split("\t") if token.strip()]
        if len(first_tokens) > 3:
            axis = np.asarray([float(token) for token in first_tokens], dtype=float)
            if float(np.nanmax(axis)) < 250.0:
                raise ValueError(f"{member_name} appears to be a spatial matrix rather than a Raman spectrum.")
            spectra_rows: list[tuple[str, np.ndarray, dict]] = []
            for row_index, line in enumerate(lines[1:], start=1):
                tokens = line.split("\t")
                nonempty = [token for token in tokens if token.strip()]
                if len(nonempty) < len(axis) + 2:
                    continue
                coord_x = float(nonempty[0])
                coord_y = float(nonempty[1])
                intensities = np.asarray([float(token) for token in nonempty[2 : 2 + len(axis)]], dtype=float)
                spectra_rows.append(
                    (
                        f"maprow_{row_index:02d}",
                        intensities,
                        {"map_x": coord_x, "map_y": coord_y},
                    )
                )
            if not spectra_rows:
                raise ValueError(f"{member_name} did not expose any map rows after the axis header.")
            return axis, spectra_rows

        values = []
        for line in lines:
            tokens = [token for token in line.split("\t") if token.strip()]
            if len(tokens) < 2:
                continue
            values.append((float(tokens[0]), float(tokens[1])))
        if len(values) < 10:
            raise ValueError(f"{member_name} did not expose a usable two-column spectrum.")
        axis = np.asarray([item[0] for item in values], dtype=float)
        if float(np.nanmax(axis)) < 250.0:
            raise ValueError(f"{member_name} appears to be a spatial matrix rather than a Raman spectrum.")
        intensity_values = np.asarray([item[1] for item in values], dtype=float)
        return axis, [("single", intensity_values, {})]

    def _build_rows(self) -> list[dict]:
        if not self.archive_path.exists():
            raise FileNotFoundError(f"Missing cholangio zip archive: {self.archive_path}")

        rows: list[dict] = []
        with ZipFile(self.archive_path, "r") as archive:
            for member_name in sorted(archive.namelist()):
                if not self._member_is_primary_spectrum(member_name):
                    continue

                parts = [part for part in member_name.split("/") if part]
                class_folder = parts[1]
                sample_id = parts[2]
                class_label = self.CLASS_MAP[class_folder]
                source_file = f"{self.ZIP_NAME}::{member_name}"
                text = archive.read(member_name).decode("utf-8", errors="ignore")
                try:
                    wavenumbers, spectra_rows = self._parse_single_spectrum(member_name, text)
                except ValueError as exc:
                    if "spatial matrix" in str(exc):
                        continue
                    raise
                replicate_stem = Path(member_name).stem

                for local_row_id, intensity_values, extras in spectra_rows:
                    source_row_id = f"{sample_id}__{replicate_stem}__{local_row_id}"
                    biosample_id = f"{self.dataset_id}_{source_row_id}"
                    rows.append(
                        {
                            "biosample_id": biosample_id,
                            "dataset_id": self.dataset_id,
                            "source_row_id": source_row_id,
                            "sample_id": sample_id,
                            "patient_id": sample_id,
                            "replicate_id": f"{replicate_stem}__{local_row_id}",
                            "class_label": class_label,
                            "subclass_label": self.SUBCLASS_LABEL,
                            "source_file": source_file,
                            "acquisition_variant": replicate_stem,
                            "wavenumbers": wavenumbers,
                            "intensity_values": intensity_values,
                            "extra_metadata": extras,
                        }
                    )

        if not rows:
            raise ValueError("No primary cholangio serum SERS TXT spectra were parsed from the zip archive.")
        return rows

    def _preprocessing_summary(self) -> str:
        return (
            "Raw ingest preserves the released serum SERS TXT spectra exactly as distributed inside the zip archive. "
            "Two file shapes occur in the release: plain two-column spectra and map-style TXT files where the first "
            "row stores the Raman axis and subsequent rows store multiple intensity traces plus coordinate metadata. "
            "GAIRA preserves those traces explicitly rather than collapsing them during raw ingest."
        )

    def _base_notes(self, row: dict) -> str:
        extra = ""
        if row["extra_metadata"]:
            extra = (
                f" Map coordinates were preserved as x={row['extra_metadata'].get('map_x')}, "
                f"y={row['extra_metadata'].get('map_y')}."
            )
        return (
            f"Released serum SERS spectrum from the multi-class cholangio archive. "
            f"Class label={row['class_label']}, sample folder={row['sample_id']}, source file={row['source_file'].split('::')[-1]}.{extra} "
            "The auxiliary 'Try dilute condition' branch and non-SER red-map helper TXT files are excluded from the "
            "primary biosample ingest so the main cohort remains clean and explicit."
        )

    def _detect_peaks_for_spectrum(self, intensity_values: np.ndarray, wavenumbers: np.ndarray) -> list[dict]:
        shifted = np.asarray(intensity_values, dtype=float) - float(np.min(intensity_values))
        max_value = float(np.max(shifted))
        if max_value <= 0:
            return []
        normalized = shifted / max_value
        peak_indices, properties = find_peaks(normalized, prominence=0.05, height=0.05, distance=5)
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
        metadata_df = pd.DataFrame(self.rows)
        print("cca_hcc_lm_serum_sers dataset audit")
        print(f"Dataset root: {self.dataset_root}")
        print(f"Archive: {self.archive_path}")
        print(f"Parsed biosample spectra: {len(metadata_df)}")
        print("Class counts:")
        print(metadata_df["class_label"].value_counts().sort_index().to_string())
        print("Sample counts:")
        print(metadata_df.groupby("class_label")["sample_id"].nunique().sort_index().to_string())
        point_counts = sorted({len(row["wavenumbers"]) for row in self.rows})
        print(f"Native axis point counts: {point_counts}")

    def extract_metadata(self) -> pd.DataFrame:
        preprocessing_summary = self._preprocessing_summary()
        rows = []
        for row in self.rows:
            wavenumbers = row["wavenumbers"]
            rows.append(
                {
                    "biosample_id": row["biosample_id"],
                    "dataset_id": row["dataset_id"],
                    "source_row_id": row["source_row_id"],
                    "sample_id": row["sample_id"],
                    "patient_id": row["patient_id"],
                    "replicate_id": row["replicate_id"],
                    "biosample_type": "serum",
                    "matrix": "serum",
                    "disease_context": "multi-class liver malignancy serum SERS cohort",
                    "class_label": row["class_label"],
                    "subclass_label": row["subclass_label"],
                    "collection_protocol": None,
                    "preparation_protocol": None,
                    "instrument": "785 nm serum SERS archive",
                    "laser_wavelength_nm": "785",
                    "spectral_range": f"{wavenumbers[0]:.2f}-{wavenumbers[-1]:.2f}",
                    "preprocessing_summary": preprocessing_summary,
                    "source_file": row["source_file"],
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
                    "biosample_id": row["biosample_id"],
                    "dataset_id": row["dataset_id"],
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
                        "biosample_id": row["biosample_id"],
                        "dataset_id": row["dataset_id"],
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
            for peak_row in self._detect_peaks_for_spectrum(row["intensity_values"], row["wavenumbers"]):
                rows.append(
                    {
                        "biosample_id": row["biosample_id"],
                        "dataset_id": row["dataset_id"],
                        "source_row_id": row["source_row_id"],
                        **peak_row,
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

        with duckdb.connect(str(self.db_path)) as connection:
            connection.execute("DELETE FROM biosample_peaks WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM biosample_spectrum_points WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM biosample_spectra WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM biosample_metadata WHERE dataset_id = ?", [self.dataset_id])

            self._insert_dataframe(connection, "biosample_metadata", metadata_df)
            self._insert_dataframe(connection, "biosample_spectra", spectra_df)
            self._insert_dataframe(connection, "biosample_spectrum_points", points_df)
            self._insert_dataframe(connection, "biosample_peaks", peaks_df)

        print(f"Inserted biosample_metadata rows: {len(metadata_df)}")
        print(f"Inserted biosample_spectra rows: {len(spectra_df)}")
        print(f"Inserted biosample_spectrum_points rows: {len(points_df)}")
        print(f"Inserted biosample_peaks rows: {len(peaks_df)}")
