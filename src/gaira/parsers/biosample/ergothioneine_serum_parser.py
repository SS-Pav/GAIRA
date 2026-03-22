import json
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from gaira.parsers.biosample.base import BiosampleParserBase


class ErgothioneineSerumParser(BiosampleParserBase):
    """Parser for Zenodo 13791050 ergothioneine-in-serum SERS calibration data."""

    CSV_NAME = "ERG_calibration.csv"
    SUBCLASS_LABEL = "ergothioneine_calibration_archive"

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        super().__init__(dataset_id=dataset_id, dataset_root=dataset_root, db_path=db_path)
        self.csv_path = self.dataset_root / self.CSV_NAME
        self.rows = self._build_rows()

    def _load_dataframe(self) -> tuple[pd.DataFrame, list[str], np.ndarray]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing ergothioneine calibration CSV: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        spectral_columns = [column for column in df.columns if column not in {"laser", "power", "substrate", "c"}]
        if not spectral_columns:
            raise ValueError("ERG_calibration.csv does not expose numeric spectral columns.")

        wavenumbers = np.asarray([float(column) for column in spectral_columns], dtype=float)
        return df, spectral_columns, wavenumbers

    def _build_rows(self) -> list[dict]:
        df, spectral_columns, wavenumbers = self._load_dataframe()
        rows: list[dict] = []
        replicate_counters: Counter[float] = Counter()

        for row_index, row in df.iterrows():
            concentration_value = float(row["c"])
            replicate_counters[concentration_value] += 1
            replicate_id = f"rep{replicate_counters[concentration_value]}"
            source_row_id = f"row_{row_index + 1:03d}"
            biosample_id = f"{self.dataset_id}_{source_row_id}"
            class_label = f"erg_{concentration_value:.1f}_uM".replace(".", "p")
            intensity_values = row[spectral_columns].to_numpy(dtype=float)

            rows.append(
                {
                    "biosample_id": biosample_id,
                    "dataset_id": self.dataset_id,
                    "source_row_id": source_row_id,
                    "sample_id": f"erg_{concentration_value:.1f}_uM".replace(".", "p"),
                    "patient_id": None,
                    "replicate_id": replicate_id,
                    "class_label": class_label,
                    "subclass_label": self.SUBCLASS_LABEL,
                    "source_file": f"{self.CSV_NAME}::{source_row_id}",
                    "wavenumbers": wavenumbers,
                    "intensity_values": intensity_values,
                    "laser_wavelength_nm": str(row["laser"]),
                    "laser_power": str(row["power"]),
                    "substrate": str(row["substrate"]),
                    "concentration_uM": concentration_value,
                }
            )

        if not rows:
            raise ValueError("No ergothioneine serum rows were found in ERG_calibration.csv.")
        return rows

    def _preprocessing_summary(self) -> str:
        return (
            "Raw ingest preserves the released ERG_calibration.csv spectra exactly as distributed. "
            "The file stores serum SERS calibration rows on a shared native axis with metadata columns "
            "laser, power, substrate, and ergothioneine concentration c."
        )

    def _base_notes(self, row: dict) -> str:
        return (
            "Released ergothioneine-in-serum SERS calibration spectrum. "
            f"Preserved metadata: laser={row['laser_wavelength_nm']} nm, power={row['laser_power']}, "
            f"substrate={row['substrate']}, concentration_uM={row['concentration_uM']:.1f}, "
            f"replicate_id={row['replicate_id']}. Current GAIRA framing treats this as a serum "
            "metabolite-calibration archive rather than a disease benchmark."
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
        axis = self.rows[0]["wavenumbers"]
        class_counts = Counter(row["class_label"] for row in self.rows)
        print("ergothioneine_serum dataset audit")
        print(f"Dataset root: {self.dataset_root}")
        print(f"Calibration CSV: {self.csv_path}")
        print(f"Native axis: {axis[0]:.2f} to {axis[-1]:.2f} cm^-1 ({len(axis)} points)")
        print("Concentration counts:")
        for class_label, count in sorted(class_counts.items()):
            print(f"  {class_label}: {count}")
        print("This archive is a serum metabolite-calibration dataset, not a clinical benchmark.")

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
                    "collection_protocol": None,
                    "preparation_protocol": "ergothioneine spiking/calibration in human serum",
                    "instrument": f"{row['substrate']} substrate serum SERS acquisition",
                    "laser_wavelength_nm": row["laser_wavelength_nm"],
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
            for peak_row in self._detect_peaks_for_spectrum(row["intensity_values"], row["wavenumbers"]):
                rows.append(
                    {
                        "biosample_id": row["biosample_id"],
                        "dataset_id": self.dataset_id,
                        "source_row_id": row["source_row_id"],
                        **peak_row,
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
