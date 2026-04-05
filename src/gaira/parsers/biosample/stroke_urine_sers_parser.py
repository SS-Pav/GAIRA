import io
import json
import subprocess
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from gaira.parsers.biosample.base import BiosampleParserBase


class StrokeUrineSERSParser(BiosampleParserBase):
    """Parser for the curated stroke-urine cohort matrix inside data.rar."""

    ARCHIVE_NAME = "data.rar"
    MATRIX_MEMBER = "CBI/data.csv"
    METADATA_MEMBER = "PI/data.csv"

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        super().__init__(dataset_id=dataset_id, dataset_root=dataset_root, db_path=db_path)
        self.archive_path = self.dataset_root / self.ARCHIVE_NAME
        self.patient_metadata_df = self._load_patient_metadata()
        self.matrix_df = self._load_matrix()
        self.rows = self._build_rows()

    def _extract_member_bytes(self, member_name: str) -> bytes:
        if not self.archive_path.exists():
            raise FileNotFoundError(f"Missing stroke archive: {self.archive_path}")
        return subprocess.check_output(["bsdtar", "-xOf", str(self.archive_path), member_name])

    def _load_patient_metadata(self) -> pd.DataFrame:
        raw = self._extract_member_bytes(self.METADATA_MEMBER)
        return pd.read_csv(io.BytesIO(raw))

    def _load_matrix(self) -> pd.DataFrame:
        raw = self._extract_member_bytes(self.MATRIX_MEMBER)
        df = pd.read_csv(io.BytesIO(raw))
        if "labels" not in df.columns or "label2" not in df.columns:
            raise ValueError("Stroke matrix is missing required label columns.")

        spectral_columns = [column for column in df.columns if str(column).isdigit()]
        if len(spectral_columns) < 100:
            raise ValueError("Stroke matrix did not expose the expected spectral columns.")

        out = df.copy()
        out[spectral_columns] = out[spectral_columns].apply(pd.to_numeric, errors="coerce")
        out["labels"] = pd.to_numeric(out["labels"], errors="coerce")
        out["label2"] = pd.to_numeric(out["label2"], errors="coerce")
        out = out.dropna(subset=spectral_columns + ["labels", "label2"]).reset_index(drop=True)
        out.attrs["spectral_columns"] = spectral_columns
        return out

    def _class_label(self, label_value: float) -> str:
        return "stroke" if int(label_value) == 1 else "healthy_control"

    def _build_rows(self) -> list[dict]:
        spectral_columns = list(self.matrix_df.attrs["spectral_columns"])
        axis = np.asarray([float(column) for column in spectral_columns], dtype=float)
        rows: list[dict] = []
        replicate_counts: Counter[int] = Counter()

        for row_index, row in self.matrix_df.iterrows():
            group_id = int(row["label2"])
            replicate_counts[group_id] += 1
            replicate_id = f"rep{replicate_counts[group_id]:03d}"
            sample_id = f"group_{group_id:03d}"
            source_row_id = f"{sample_id}__{replicate_id}"
            biosample_id = f"{self.dataset_id}_{source_row_id}"
            intensity_values = row[spectral_columns].to_numpy(dtype=float)
            class_label = self._class_label(float(row["labels"]))

            rows.append(
                {
                    "biosample_id": biosample_id,
                    "dataset_id": self.dataset_id,
                    "source_row_id": source_row_id,
                    "sample_id": sample_id,
                    "patient_id": sample_id,
                    "replicate_id": replicate_id,
                    "class_label": class_label,
                    "subclass_label": sample_id,
                    "source_file": f"{self.ARCHIVE_NAME}::{self.MATRIX_MEMBER}::row_{row_index + 1:04d}",
                    "wavenumbers": axis,
                    "intensity_values": intensity_values,
                }
            )

        if not rows:
            raise ValueError("No usable stroke urine spectra were parsed from CBI/data.csv.")
        return rows

    def _preprocessing_summary(self) -> str:
        return (
            "The canonical biosample ingest preserves the released `CBI/data.csv` cohort matrix from "
            "`data.rar`. The matrix does not provide an explicit Raman-shift axis, so the released "
            "0-4095 spectral index columns are preserved verbatim as the native x-axis. `PI/data.csv` "
            "is retained as provenance metadata only because its participant cardinality does not align "
            "cleanly with the released cohort matrix groups."
        )

    def _base_notes(self, row: dict) -> str:
        return (
            "Stroke urine spectrum from the released CBI matrix. "
            f"class_label={row['class_label']}, sample_id={row['sample_id']}, "
            f"replicate_id={row['replicate_id']}, source_file={row['source_file']}."
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
        sample_counts = Counter(row["sample_id"] for row in self.rows)
        axis = self.rows[0]["wavenumbers"]
        print("stroke_urine_sers dataset audit")
        print(f"Dataset root: {self.dataset_root}")
        print(f"Archive path: {self.archive_path}")
        print(f"Released native axis: {axis.min():.0f} to {axis.max():.0f} ({len(axis)} points)")
        print(f"Usable spectra: {len(self.rows)}")
        print(f"Sample groups: {len(sample_counts)}")
        print(f"PI sidecar rows retained as provenance only: {len(self.patient_metadata_df)}")
        print("Class counts:")
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
                    "biosample_type": "urine",
                    "matrix": "human_urine",
                    "disease_context": "ischemic stroke urine cohort"
                    if row["class_label"] == "stroke"
                    else "healthy control urine cohort",
                    "class_label": row["class_label"],
                    "subclass_label": row["subclass_label"],
                    "collection_protocol": None,
                    "preparation_protocol": "released CBI cohort matrix from stroke urine SERS data.rar",
                    "instrument": "stroke urine SERS acquisition (released matrix axis only)",
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

        print(f"Ingested {len(metadata_df)} stroke urine spectra into DuckDB.")
