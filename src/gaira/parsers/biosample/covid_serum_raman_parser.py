import json
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from gaira.parsers.biosample.base import BiosampleParserBase


class COVIDSerumRamanParser(BiosampleParserBase):
    """Parser for the Figshare COVID-19 serum spontaneous Raman cohort archive."""

    WAVE_FILE = "wave_number.txt"
    README_FILE = "readme.txt"
    CODE_FILE = "code.m"
    TABLE2_FILE = "table2_data.txt"
    SUBCLASS_LABEL = "covid19_serum_raman_archive"
    CLASS_FILES = {
        "covid_confirmed": "raw_COVID.txt",
        "healthy_control": "raw_Helthy.txt",
        "suspected_case": "raw_Suspected.txt",
        "tube_control": "raw_Tube.txt",
    }
    DISEASE_CONTEXT = {
        "covid_confirmed": "confirmed COVID-19 serum cohort",
        "healthy_control": "healthy serum cohort",
        "suspected_case": "suspected COVID-19 serum cohort",
        "tube_control": "cryopreservation tube saline control",
    }

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        super().__init__(dataset_id=dataset_id, dataset_root=dataset_root, db_path=db_path)
        self.wave_path = self.dataset_root / self.WAVE_FILE
        self.readme_path = self.dataset_root / self.README_FILE
        self.code_path = self.dataset_root / self.CODE_FILE
        self.table2_path = self.dataset_root / self.TABLE2_FILE
        self.wavenumbers = self._load_wavenumbers()
        self.rows = self._build_rows()

    def _load_wavenumbers(self) -> np.ndarray:
        if not self.wave_path.exists():
            raise FileNotFoundError(f"Missing wave-number file: {self.wave_path}")
        values = np.loadtxt(self.wave_path, delimiter="\t", dtype=float)
        wavenumbers = np.asarray(values, dtype=float).reshape(-1)
        if len(wavenumbers) < 10:
            raise ValueError("wave_number.txt does not expose the expected Raman axis.")
        return wavenumbers

    def _load_matrix(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Missing raw matrix file: {path}")
        matrix = np.loadtxt(path, delimiter="\t", dtype=float)
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"Expected a 2D raw matrix in {path.name}, found shape {matrix.shape}.")
        if matrix.shape[0] != len(self.wavenumbers):
            raise ValueError(
                f"{path.name} row count {matrix.shape[0]} does not match wave-number length {len(self.wavenumbers)}."
            )
        return matrix

    def _build_rows(self) -> list[dict]:
        rows: list[dict] = []
        for class_label, file_name in self.CLASS_FILES.items():
            matrix = self._load_matrix(self.dataset_root / file_name)
            for column_index in range(matrix.shape[1]):
                source_row_id = f"{Path(file_name).stem}_col_{column_index + 1:03d}"
                biosample_id = f"{self.dataset_id}_{source_row_id}"
                intensity_values = matrix[:, column_index].astype(float)
                rows.append(
                    {
                        "biosample_id": biosample_id,
                        "dataset_id": self.dataset_id,
                        "source_row_id": source_row_id,
                        "sample_id": source_row_id,
                        "patient_id": None,
                        "replicate_id": None,
                        "class_label": class_label,
                        "subclass_label": self.SUBCLASS_LABEL,
                        "source_file": f"{file_name}::column_{column_index + 1:03d}",
                        "wavenumbers": self.wavenumbers,
                        "intensity_values": intensity_values,
                        "disease_context": self.DISEASE_CONTEXT[class_label],
                    }
                )

        if not rows:
            raise ValueError("No COVID serum Raman spectra were parsed from the raw matrices.")
        return rows

    def _preprocessing_summary(self) -> str:
        return (
            "Raw ingest preserves the released spontaneous-Raman text matrices exactly as distributed. "
            "Each raw_*.txt file stores one shared wave-number axis against multiple cohort-specific spectra. "
            "The released MATLAB code provides downstream processing, but GAIRA raw ingestion keeps the original "
            "matrix values unchanged."
        )

    def _base_notes(self, row: dict) -> str:
        return (
            "Released spontaneous-Raman serum cohort spectrum from the COVID-19 Figshare archive. "
            f"Class label={row['class_label']} and source matrix={row['source_file'].split('::')[0]}. "
            "This dataset should be interpreted as serum spontaneous Raman cohort evidence, not as SERS and not as "
            "a direct replacement for the SERS-heavy serum stack."
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
        class_counts = Counter(row["class_label"] for row in self.rows)
        print("covid_serum_raman dataset audit")
        print(f"Dataset root: {self.dataset_root}")
        print(f"Wave-number file: {self.wave_path}")
        print(f"Readme: {self.readme_path}")
        print(f"MATLAB code: {self.code_path}")
        print(f"Native axis: {self.wavenumbers[0]:.2f} to {self.wavenumbers[-1]:.2f} cm^-1 ({len(self.wavenumbers)} points)")
        print("Class counts:")
        for class_label, count in sorted(class_counts.items()):
            print(f"  {class_label}: {count}")
        print("This archive is a spontaneous-Raman serum cohort dataset, not a SERS grounding resource.")

    def extract_metadata(self) -> pd.DataFrame:
        preprocessing_summary = self._preprocessing_summary()
        rows = []
        for row in self.rows:
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
                    "disease_context": row["disease_context"],
                    "class_label": row["class_label"],
                    "subclass_label": row["subclass_label"],
                    "collection_protocol": "released serum cohort archive with confirmed healthy suspected and tube-control matrices",
                    "preparation_protocol": None,
                    "instrument": "Volume Phase Holographic spectrograph / deep-cooled CCD / Raman probe setup",
                    "laser_wavelength_nm": "unknown",
                    "spectral_range": f"{self.wavenumbers[0]:.2f}-{self.wavenumbers[-1]:.2f}",
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
            rows.append(
                {
                    "biosample_id": row["biosample_id"],
                    "dataset_id": self.dataset_id,
                    "source_row_id": row["source_row_id"],
                    "x_min": float(self.wavenumbers[0]),
                    "x_max": float(self.wavenumbers[-1]),
                    "n_points": int(len(self.wavenumbers)),
                    "wavenumbers_json": json.dumps([float(value) for value in self.wavenumbers]),
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
                zip(self.wavenumbers, row["intensity_values"]),
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
        return pd.DataFrame(rows)

    def extract_peaks(self) -> pd.DataFrame:
        rows = []
        for row in self.rows:
            for peak_row in self._detect_peaks_for_spectrum(row["intensity_values"], self.wavenumbers):
                rows.append(
                    {
                        "biosample_id": row["biosample_id"],
                        "dataset_id": self.dataset_id,
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
