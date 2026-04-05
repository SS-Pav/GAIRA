import csv
import json
import re
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from gaira.parsers.biosample.base import BiosampleParserBase


class MycoplasmaNASERSParser(BiosampleParserBase):
    """Parser for the consolidated Mycoplasma NA-SERS pathogen panel."""

    CSV_NAME = "NA-SERS specificity spectra.csv"
    EXCLUDED_LABELS = {"Bkg", "Media Ctl"}

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        super().__init__(dataset_id=dataset_id, dataset_root=dataset_root, db_path=db_path)
        self.csv_path = self.dataset_root / self.CSV_NAME
        self.excluded_counts: Counter[str] = Counter()
        self.rows = self._build_rows()

    def _slugify(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()

    def _load_matrix(self) -> tuple[np.ndarray, list[str], np.ndarray]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing Mycoplasma CSV: {self.csv_path}")

        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)

        if len(header) < 3:
            raise ValueError("Mycoplasma CSV does not expose the expected header width.")

        labels = [value.strip() for value in header[1:]]
        matrix = pd.read_csv(self.csv_path, skiprows=1, header=None).apply(pd.to_numeric, errors="coerce")
        matrix = matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if matrix.shape[1] != len(labels) + 1:
            raise ValueError(
                f"Header/data width mismatch for {self.csv_path.name}: "
                f"{len(labels) + 1} expected columns, found {matrix.shape[1]}."
            )

        wavenumbers = matrix.iloc[:, 0].to_numpy(dtype=float)
        intensity_matrix = matrix.iloc[:, 1:].to_numpy(dtype=float)
        if intensity_matrix.shape[1] != len(labels):
            raise ValueError("Failed to align intensity matrix with label columns.")
        return wavenumbers, labels, intensity_matrix

    def _class_to_subclass(self, class_label: str) -> str:
        normalized = class_label.strip()
        if normalized in {"M129", "FH"}:
            return "mycoplasma_pneumoniae_reference_strain"
        if normalized in self.EXCLUDED_LABELS:
            return "technical_control"
        if normalized.startswith("M.") or normalized.startswith("U.") or normalized.startswith("A."):
            return "non_target_species_panel"
        return "clinical_isolate_panel"

    def _disease_context(self, class_label: str) -> str | None:
        if class_label in self.EXCLUDED_LABELS:
            return None
        return "mycoplasma_specificity_and_strain_typing_panel"

    def _build_rows(self) -> list[dict]:
        wavenumbers, labels, intensity_matrix = self._load_matrix()
        rows: list[dict] = []
        label_counters: Counter[str] = Counter()

        for column_index, class_label in enumerate(labels):
            label_counters[class_label] += 1
            if class_label in self.EXCLUDED_LABELS:
                self.excluded_counts[class_label] += 1
                continue

            replicate_id = f"rep{label_counters[class_label]:02d}"
            source_row_id = f"{self._slugify(class_label)}__{replicate_id}"
            biosample_id = f"{self.dataset_id}_{source_row_id}"
            rows.append(
                {
                    "biosample_id": biosample_id,
                    "dataset_id": self.dataset_id,
                    "source_row_id": source_row_id,
                    "sample_id": self._slugify(class_label),
                    "patient_id": None,
                    "replicate_id": replicate_id,
                    "class_label": class_label,
                    "subclass_label": self._class_to_subclass(class_label),
                    "source_file": f"{self.CSV_NAME}::column_{column_index + 2:03d}",
                    "wavenumbers": wavenumbers,
                    "intensity_values": intensity_matrix[:, column_index],
                    "disease_context": self._disease_context(class_label),
                }
            )

        if not rows:
            raise ValueError("No usable Mycoplasma spectra were parsed after control exclusions.")
        return rows

    def _preprocessing_summary(self) -> str:
        return (
            "Raw ingest preserves each pathogen spectrum exactly as released in the consolidated "
            "NA-SERS specificity CSV. Background and media-control columns are excluded from the "
            "biosample ingest and recorded separately in the ingest artifacts."
        )

    def _base_notes(self, row: dict) -> str:
        return (
            "Released NA-SERS pathogen panel spectrum. "
            f"class_label={row['class_label']}, subclass_label={row['subclass_label']}, "
            f"source_file={row['source_file']}."
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
        counts = Counter(row["class_label"] for row in self.rows)
        axis = self.rows[0]["wavenumbers"]
        print("mycoplasma_na_sers dataset audit")
        print(f"Dataset root: {self.dataset_root}")
        print(f"CSV path: {self.csv_path}")
        print(f"Native axis: {axis.min():.2f} to {axis.max():.2f} cm^-1 ({len(axis)} points)")
        print(f"Usable spectra after exclusions: {len(self.rows)}")
        print("Excluded control counts:")
        for label, count in sorted(self.excluded_counts.items()):
            print(f"  {label}: {count}")
        print("Usable class counts:")
        for label, count in sorted(counts.items()):
            print(f"  {label}: {count}")

    def extract_metadata(self) -> pd.DataFrame:
        preprocessing_summary = self._preprocessing_summary()
        out = []
        for row in self.rows:
            wavenumbers = row["wavenumbers"]
            out.append(
                {
                    "biosample_id": row["biosample_id"],
                    "dataset_id": self.dataset_id,
                    "source_row_id": row["source_row_id"],
                    "sample_id": row["sample_id"],
                    "patient_id": row["patient_id"],
                    "replicate_id": row["replicate_id"],
                    "biosample_type": "pathogen",
                    "matrix": "mycoplasma_and_related_species_panel",
                    "disease_context": row["disease_context"],
                    "class_label": row["class_label"],
                    "subclass_label": row["subclass_label"],
                    "collection_protocol": None,
                    "preparation_protocol": "nanorod array SERS specificity and strain-typing panel",
                    "instrument": "NA-SERS pathogen panel acquisition",
                    "laser_wavelength_nm": None,
                    "spectral_range": f"{float(np.min(wavenumbers)):.2f}-{float(np.max(wavenumbers)):.2f}",
                    "preprocessing_summary": preprocessing_summary,
                    "source_file": row["source_file"],
                    "notes": self._base_notes(row),
                }
            )
        return pd.DataFrame(out)

    def extract_spectra(self) -> pd.DataFrame:
        preprocessing_summary = self._preprocessing_summary()
        out = []
        for row in self.rows:
            wavenumbers = row["wavenumbers"]
            out.append(
                {
                    "biosample_id": row["biosample_id"],
                    "dataset_id": self.dataset_id,
                    "source_row_id": row["source_row_id"],
                    "x_min": float(np.min(wavenumbers)),
                    "x_max": float(np.max(wavenumbers)),
                    "n_points": int(len(wavenumbers)),
                    "wavenumbers_json": json.dumps([float(v) for v in wavenumbers]),
                    "intensity_json": json.dumps([float(v) for v in row["intensity_values"]]),
                    "normalized_flag": "unknown",
                    "preprocessing_summary": preprocessing_summary,
                }
            )
        return pd.DataFrame(out)

    def extract_spectrum_points(self) -> pd.DataFrame:
        out = []
        for row in self.rows:
            for point_index, (wavenumber, intensity) in enumerate(
                zip(row["wavenumbers"], row["intensity_values"]),
                start=1,
            ):
                out.append(
                    {
                        "biosample_id": row["biosample_id"],
                        "dataset_id": self.dataset_id,
                        "source_row_id": row["source_row_id"],
                        "point_index": point_index,
                        "wavenumber": float(wavenumber),
                        "intensity": float(intensity),
                    }
                )
        return pd.DataFrame(out)

    def extract_peaks(self) -> pd.DataFrame:
        out = []
        for row in self.rows:
            for peak_row in self._detect_peaks_for_spectrum(row["intensity_values"], row["wavenumbers"]):
                out.append(
                    {
                        "biosample_id": row["biosample_id"],
                        "dataset_id": self.dataset_id,
                        "source_row_id": row["source_row_id"],
                        **peak_row,
                    }
                )
        return pd.DataFrame(out)

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
