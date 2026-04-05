import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from gaira.parsers.biosample.base import BiosampleParserBase


class OvarianPlasmaRamanSERSParser(BiosampleParserBase):
    """Parser for the ovarian plasma archive spanning Raman and SERS modalities."""

    ARCHIVES = {
        "raman": "Raman dataset.zip",
        "sers": "SERS dataset.zip",
    }

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        super().__init__(dataset_id=dataset_id, dataset_root=dataset_root, db_path=db_path)
        self.archive_paths = {mode: self.dataset_root / name for mode, name in self.ARCHIVES.items()}
        self.rows = self._build_rows()

    def _parse_txt_bytes(self, raw_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
        lines = raw_bytes.decode("utf-8", "ignore").splitlines()
        numeric_pairs: list[tuple[float, float]] = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\t+", line)
            if len(parts) < 2:
                continue
            try:
                numeric_pairs.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
        if len(numeric_pairs) < 100:
            raise ValueError("TXT member did not expose a valid two-column spectrum.")
        array = np.asarray(numeric_pairs, dtype=float)
        return array[:, 0], array[:, 1]

    def _parse_member_identity(self, mode: str, member: str) -> dict[str, str]:
        path = Path(member)
        class_folder = path.parts[-2]
        disease_map = {
            "Healthy Individuals": ("healthy_control", "healthy plasma cohort"),
            "Ovarian Cancer": ("ovarian_cancer", "ovarian cancer plasma cohort"),
        }
        class_label, disease_context = disease_map[class_folder]
        stem = path.stem
        replicate_match = re.search(r"\((\d+)\)$", stem)
        replicate_id = f"rep{int(replicate_match.group(1)):02d}" if replicate_match else None
        stem_no_rep = re.sub(r"\s*\(\d+\)$", "", stem)
        tokens = stem_no_rep.split("_")

        donor_id = tokens[2] if len(tokens) > 2 else stem_no_rep
        if mode == "sers" and donor_id.startswith("GU"):
            donor_id = donor_id

        source_row_id = f"{mode}__{class_label}__{donor_id}__{replicate_id or 'rep00'}"
        source_row_id = re.sub(r"[^A-Za-z0-9_]+", "_", source_row_id).strip("_").lower()
        return {
            "class_label": class_label,
            "disease_context": disease_context,
            "sample_id": donor_id,
            "patient_id": donor_id,
            "replicate_id": replicate_id,
            "subclass_label": mode,
            "source_row_id": source_row_id,
        }

    def _iter_archive_members(self):
        for mode, archive_path in self.archive_paths.items():
            if not archive_path.exists():
                raise FileNotFoundError(f"Missing ovarian archive: {archive_path}")
            with ZipFile(archive_path, "r") as archive:
                for member in sorted(name for name in archive.namelist() if name.endswith(".txt")):
                    yield mode, archive_path.name, member, archive.read(member)

    def _build_rows(self) -> list[dict]:
        rows: list[dict] = []
        for mode, archive_name, member, raw_bytes in self._iter_archive_members():
            wavenumbers, intensities = self._parse_txt_bytes(raw_bytes)
            identity = self._parse_member_identity(mode, member)
            biosample_id = f"{self.dataset_id}_{identity['source_row_id']}"
            rows.append(
                {
                    "biosample_id": biosample_id,
                    "dataset_id": self.dataset_id,
                    "source_row_id": identity["source_row_id"],
                    "sample_id": identity["sample_id"],
                    "patient_id": identity["patient_id"],
                    "replicate_id": identity["replicate_id"],
                    "class_label": identity["class_label"],
                    "subclass_label": identity["subclass_label"],
                    "source_file": f"{archive_name}::{member}",
                    "wavenumbers": wavenumbers,
                    "intensity_values": intensities,
                    "disease_context": identity["disease_context"],
                }
            )
        if not rows:
            raise ValueError("No ovarian plasma spectra were parsed from the released archives.")
        return rows

    def _preprocessing_summary(self) -> str:
        return (
            "Raw ingest preserves the released two-column plasma text spectra exactly as distributed. "
            "Raman and SERS members stay in the same dataset_id but are preserved via subclass_label."
        )

    def _base_notes(self, row: dict) -> str:
        return (
            "Released ovarian plasma spectrum. "
            f"modality={row['subclass_label']}, class_label={row['class_label']}, "
            f"patient_id={row['patient_id']}, source_file={row['source_file']}."
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
        class_counts = Counter((row["subclass_label"], row["class_label"]) for row in self.rows)
        patient_counts = Counter((row["subclass_label"], row["patient_id"]) for row in self.rows)
        print("ovarian_plasma_raman_sers dataset audit")
        print(f"Dataset root: {self.dataset_root}")
        for mode, archive_path in self.archive_paths.items():
            print(f"  {mode}: {archive_path}")
        print(f"Usable spectra: {len(self.rows)}")
        print("Counts by modality/class:")
        for (mode, class_label), count in sorted(class_counts.items()):
            print(f"  {mode} / {class_label}: {count}")
        print("Unique patient counts by modality:")
        by_mode: Counter[str] = Counter()
        for mode, patient_id in patient_counts:
            by_mode[mode] += 1
        for mode, count in sorted(by_mode.items()):
            print(f"  {mode}: {count}")

    def extract_metadata(self) -> pd.DataFrame:
        preprocessing_summary = self._preprocessing_summary()
        out = []
        for row in self.rows:
            wavenumbers = row["wavenumbers"]
            modality = row["subclass_label"]
            instrument = "plasma spontaneous Raman acquisition" if modality == "raman" else "plasma SERS acquisition"
            out.append(
                {
                    "biosample_id": row["biosample_id"],
                    "dataset_id": self.dataset_id,
                    "source_row_id": row["source_row_id"],
                    "sample_id": row["sample_id"],
                    "patient_id": row["patient_id"],
                    "replicate_id": row["replicate_id"],
                    "biosample_type": "plasma",
                    "matrix": "plasma",
                    "disease_context": row["disease_context"],
                    "class_label": row["class_label"],
                    "subclass_label": row["subclass_label"],
                    "collection_protocol": None,
                    "preparation_protocol": "blood plasma Raman and SERS cohort release",
                    "instrument": instrument,
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
