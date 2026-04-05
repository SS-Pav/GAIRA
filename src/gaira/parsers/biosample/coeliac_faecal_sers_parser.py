import csv
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from gaira.parsers.biosample.base import BiosampleParserBase


class CoeliacFaecalSERSParser(BiosampleParserBase):
    """Parser for the coeliac faecal SERS cohort zip."""

    ZIP_NAME = "coeliac_faecal_sers.zip"
    COHORT_PREFIX = "faecal samples dataset/"
    EXCLUDED_PREFIX = "pure metabolites dataset/"

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        super().__init__(dataset_id=dataset_id, dataset_root=dataset_root, db_path=db_path)
        self.zip_path = self._discover_zip_path()
        self.excluded_members: list[str] = []
        self.rows = self._build_rows()

    def _discover_zip_path(self) -> Path:
        candidates = sorted(self.dataset_root.glob("*.zip"))
        if not candidates:
            raise FileNotFoundError(f"No zip archive found under {self.dataset_root}")
        return candidates[0]

    def _parse_bwspec_txt(self, raw_text: str) -> tuple[np.ndarray, np.ndarray]:
        lines = raw_text.splitlines()
        header_index = next((idx for idx, line in enumerate(lines) if line.startswith("Pixel;")), None)
        if header_index is None:
            raise ValueError("BWSpec TXT did not expose the expected Pixel header row.")

        reader = csv.DictReader(lines[header_index:], delimiter=";")
        wavenumbers: list[float] = []
        intensities: list[float] = []
        for row in reader:
            if not row:
                continue
            shift = (row.get("Raman Shift") or "").replace(",", ".").strip()
            intensity = (row.get("Raw data #1") or "").replace(",", ".").strip()
            if not shift or not intensity:
                continue
            try:
                wavenumbers.append(float(shift))
                intensities.append(float(intensity))
            except ValueError:
                continue

        if len(wavenumbers) < 100:
            raise ValueError("BWSpec TXT did not yield enough numeric Raman points.")
        return np.asarray(wavenumbers, dtype=float), np.asarray(intensities, dtype=float)

    def _disease_context(self, class_label: str) -> str:
        return {
            "CTR": "healthy control faecal cohort",
            "CD": "coeliac disease faecal cohort",
            "GFD": "gluten-free diet faecal cohort",
        }[class_label]

    def _build_rows(self) -> list[dict]:
        rows: list[dict] = []
        with ZipFile(self.zip_path, "r") as archive:
            for member in sorted(name for name in archive.namelist() if name.endswith(".txt")):
                if member.startswith(self.EXCLUDED_PREFIX):
                    self.excluded_members.append(member)
                    continue
                if not member.startswith(self.COHORT_PREFIX):
                    continue

                stem = Path(member).stem
                parts = stem.split("_")
                if len(parts) != 4:
                    raise ValueError(f"Unexpected coeliac faecal filename pattern: {member}")
                sample_code, sex, age, class_label = parts
                wavenumbers, intensities = self._parse_bwspec_txt(archive.read(member).decode("utf-8", "ignore"))
                source_row_id = stem.lower()
                biosample_id = f"{self.dataset_id}_{source_row_id}"
                rows.append(
                    {
                        "biosample_id": biosample_id,
                        "dataset_id": self.dataset_id,
                        "source_row_id": source_row_id,
                        "sample_id": sample_code,
                        "patient_id": sample_code,
                        "replicate_id": None,
                        "class_label": class_label,
                        "subclass_label": "faecal_cohort",
                        "source_file": f"{self.zip_path.name}::{member}",
                        "wavenumbers": wavenumbers,
                        "intensity_values": intensities,
                        "disease_context": self._disease_context(class_label),
                        "sex": sex,
                        "age_years": age,
                    }
                )
        if not rows:
            raise ValueError("No coeliac faecal cohort spectra were parsed from the zip archive.")
        return rows

    def _preprocessing_summary(self) -> str:
        return (
            "Raw ingest preserves the released BWSpec semicolon-delimited text spectra exactly as "
            "distributed. Pure metabolite reference files are excluded from the biosample cohort lane."
        )

    def _base_notes(self, row: dict) -> str:
        return (
            "Released faecal cohort spectrum. "
            f"class_label={row['class_label']}, sex={row['sex']}, age_years={row['age_years']}, "
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
        print("coeliac_faecal_sers dataset audit")
        print(f"Dataset root: {self.dataset_root}")
        print(f"Zip path: {self.zip_path}")
        print(f"Native axis: {axis.min():.2f} to {axis.max():.2f} cm^-1 ({len(axis)} points)")
        print(f"Excluded pure-reference members: {len(self.excluded_members)}")
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
                    "biosample_type": "faeces",
                    "matrix": "faeces",
                    "disease_context": row["disease_context"],
                    "class_label": row["class_label"],
                    "subclass_label": row["subclass_label"],
                    "collection_protocol": None,
                    "preparation_protocol": "faecal SERS cohort release",
                    "instrument": "BWSpec 785 nm faecal SERS acquisition",
                    "laser_wavelength_nm": "785",
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
