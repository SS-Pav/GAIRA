import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from gaira.parsers.biosample.base import BiosampleParserBase


class SingleVesicleEVRamanParser(BiosampleParserBase):
    """Parser for the raw single-vesicle EV Raman/SERS Figshare archive."""

    RAR_NAME = "fc35_raw_raman_data.rar"

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        super().__init__(dataset_id=dataset_id, dataset_root=dataset_root, db_path=db_path)
        self.rar_path = self._discover_rar_path()
        self.rows = self._build_rows()

    def _discover_rar_path(self) -> Path:
        candidates = sorted(self.dataset_root.glob("*.rar"))
        if not candidates:
            raise FileNotFoundError(f"No RAR archive found under {self.dataset_root}")
        return candidates[0]

    def _archive_members(self) -> list[str]:
        result = subprocess.run(
            ["bsdtar", "-tf", str(self.rar_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip().endswith(".txt")]

    def _read_member(self, member: str) -> tuple[np.ndarray, np.ndarray]:
        result = subprocess.run(
            ["bsdtar", "-xOf", str(self.rar_path), member],
            check=True,
            capture_output=True,
            text=True,
        )
        rows: list[tuple[float, float]] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\t+", line)
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
        if len(rows) < 100:
            raise ValueError(f"Archive member {member} did not expose a valid two-column spectrum.")
        arr = np.asarray(rows, dtype=float)
        return arr[:, 0], arr[:, 1]

    def _normalize_condition(self, tokens: list[str]) -> tuple[str, str]:
        clean = [token.lower() for token in tokens if token]
        joined = "_".join(clean)
        if joined.startswith("ctrl"):
            return "control", joined
        if joined.startswith("hras"):
            digits = "".join(ch for ch in joined if ch.isdigit())
            class_label = f"hras{digits}" if digits else "hras"
            return class_label, joined
        return joined, joined

    def _parse_member_identity(self, member: str) -> dict[str, str]:
        stem = Path(member).stem
        base, *coord_parts = stem.split("__")
        tokens = base.split("_")
        if len(tokens) < 5:
            raise ValueError(f"Unexpected FC35 filename pattern: {member}")

        laser_idx = next((idx for idx, token in enumerate(tokens) if token.lower().endswith("nm")), None)
        if laser_idx is None or laser_idx < 2:
            raise ValueError(f"Could not resolve laser token in {member}")

        mapping_id = tokens[0]
        condition_tokens = tokens[1:laser_idx]
        power_token = tokens[laser_idx + 1] if len(tokens) > laser_idx + 1 else None
        exposure_token = tokens[laser_idx + 2] if len(tokens) > laser_idx + 2 else None
        scan_token = tokens[laser_idx + 3] if len(tokens) > laser_idx + 3 else None

        class_label, sample_label = self._normalize_condition(condition_tokens)
        patient_id = sample_label
        replicate_id = f"scan_{scan_token}" if scan_token is not None else None
        source_row_id = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_").lower()

        note_fragments = [mapping_id]
        if coord_parts:
            note_fragments.extend(coord_parts)
        return {
            "class_label": class_label,
            "sample_id": sample_label,
            "patient_id": patient_id,
            "replicate_id": replicate_id,
            "subclass_label": mapping_id.lower(),
            "source_row_id": source_row_id,
            "laser_wavelength_nm": tokens[laser_idx].replace("nm", ""),
            "preparation_detail": f"power={power_token}; exposure={exposure_token}; detail={' | '.join(note_fragments)}",
        }

    def _build_rows(self) -> list[dict]:
        rows: list[dict] = []
        for member in sorted(self._archive_members()):
            wavenumbers, intensities = self._read_member(member)
            identity = self._parse_member_identity(member)
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
                    "source_file": f"{self.rar_path.name}::{member}",
                    "wavenumbers": wavenumbers,
                    "intensity_values": intensities,
                    "laser_wavelength_nm": identity["laser_wavelength_nm"],
                    "preparation_detail": identity["preparation_detail"],
                }
            )
        if not rows:
            raise ValueError("No single-vesicle EV spectra were parsed from the RAR archive.")
        return rows

    def _preprocessing_summary(self) -> str:
        return (
            "Raw ingest preserves each released single-vesicle two-column text spectrum from the Figshare "
            "RAR archive. Labels are reconstructed from filenames only and should be treated as archive-level "
            "condition metadata rather than perfect biological annotation."
        )

    def _base_notes(self, row: dict) -> str:
        return (
            "Released single-vesicle spectrum. "
            f"class_label={row['class_label']}, sample_id={row['sample_id']}, "
            f"preparation_detail={row['preparation_detail']}, source_file={row['source_file']}."
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
        axis_lengths = Counter(len(row["wavenumbers"]) for row in self.rows)
        print("single_vesicle_ev_raman dataset audit")
        print(f"Dataset root: {self.dataset_root}")
        print(f"RAR path: {self.rar_path}")
        print(f"Usable spectra: {len(self.rows)}")
        print("Class counts:")
        for label, count in sorted(class_counts.items()):
            print(f"  {label}: {count}")
        print("Axis-length counts:")
        for length, count in sorted(axis_lengths.items()):
            print(f"  {length}: {count}")

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
                    "biosample_type": "extracellular vesicles",
                    "matrix": "single_vesicle_ev",
                    "disease_context": None,
                    "class_label": row["class_label"],
                    "subclass_label": row["subclass_label"],
                    "collection_protocol": None,
                    "preparation_protocol": row["preparation_detail"],
                    "instrument": "single-vesicle Raman/SERS mapping acquisition",
                    "laser_wavelength_nm": row["laser_wavelength_nm"],
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
