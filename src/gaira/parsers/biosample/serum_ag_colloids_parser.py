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


class SerumAgColloidsParser(BiosampleParserBase):
    """Parser for the serum Ag colloids Zenodo archive."""

    DATASET_ZIP = "dataset_spectral_data.zip"
    SCRIPTS_ZIP = "scripts_spectral_data.zip"
    INSTRUCTIONS = "Instructions.docx"
    SERUM_FOLDERS = {
        "donors serum SERS": "donors_serum_sers",
        "SERS serum Merck": "commercial_serum_merck",
        "SERS spiked serum Merck": "spiked_commercial_serum_merck",
        "dataset uricase": "uricase_serum_experiment",
    }

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        super().__init__(dataset_id=dataset_id, dataset_root=dataset_root, db_path=db_path)
        self.dataset_zip = self.dataset_root / self.DATASET_ZIP
        self.scripts_zip = self.dataset_root / self.SCRIPTS_ZIP
        self.instructions_path = self.dataset_root / self.INSTRUCTIONS
        self.rows = self._build_rows()

    def _read_bwtek_member(self, archive: ZipFile, member_name: str) -> tuple[np.ndarray, np.ndarray]:
        raw = archive.read(member_name)
        df = pd.read_csv(
            io.BytesIO(raw),
            sep=";",
            decimal=",",
            skiprows=90,
            header=None,
        )
        if df.shape[1] < 8:
            raise ValueError(f"{member_name} does not expose the expected BWtek columns.")

        wavenumbers = df.iloc[:, 3].to_numpy(dtype=float)
        intensities = df.iloc[:, 7].to_numpy(dtype=float)
        return wavenumbers, intensities

    def _parse_member_metadata(self, folder_name: str, member_name: str) -> dict:
        stem = Path(member_name).stem
        parts = stem.split("_")

        if folder_name == "donors serum SERS":
            if len(parts) < 4:
                raise ValueError(f"Unexpected donor filename layout: {member_name}")
            sample_number, replica, protocol, sample_type = parts[:4]
            return {
                "class_label": sample_type,
                "sample_id": sample_number,
                "replicate_id": replica,
                "subclass_label": self.SERUM_FOLDERS[folder_name],
                "notes": (
                    "Released donor-serum serum SERS spectrum. The filename encodes sample number, "
                    "replica, and protocol. The instructions describe these as serum samples from "
                    "healthy blood donors collected at Ospedale Maggiore, Trieste."
                ),
                "protocol": protocol,
            }

        if folder_name == "SERS serum Merck":
            if len(parts) < 3:
                raise ValueError(f"Unexpected commercial-serum filename layout: {member_name}")
            spectrum_number, replica, sample_type = parts[:3]
            return {
                "class_label": sample_type,
                "sample_id": spectrum_number,
                "replicate_id": replica,
                "subclass_label": self.SERUM_FOLDERS[folder_name],
                "notes": (
                    "Released commercial-serum serum SERS spectrum from the Merck dataset family. "
                    "The instructions state that these are commercial serum spectra acquired with Protocol 1 "
                    "(3 samples, 5 replicas each)."
                ),
                "protocol": None,
            }

        if folder_name == "SERS spiked serum Merck":
            if len(parts) < 5:
                raise ValueError(f"Unexpected spiked-serum filename layout: {member_name}")
            _, _, sample_type, concentration, replica = parts[:5]
            return {
                "class_label": sample_type,
                "sample_id": f"{sample_type}_{concentration}",
                "replicate_id": replica,
                "subclass_label": self.SERUM_FOLDERS[folder_name],
                "notes": (
                    "Released commercial-serum spiking experiment spectrum. The filename encodes the "
                    "spiked metabolite type, the final spiking concentration, and the replica number. "
                    "The instructions describe this family as commercial serum purchased from Merck "
                    "spiked with metabolites at the corresponding physiological concentrations."
                ),
                "protocol": None,
                "spike_concentration": concentration,
            }

        if folder_name == "dataset uricase":
            if len(parts) < 4:
                raise ValueError(f"Unexpected uricase filename layout: {member_name}")
            spectrum_number, replica, sample_type, protocol = parts[:4]
            return {
                "class_label": sample_type,
                "sample_id": spectrum_number,
                "replicate_id": replica,
                "subclass_label": self.SERUM_FOLDERS[folder_name],
                "notes": (
                    "Released uricase-treatment serum SERS spectrum. The filename encodes the sample condition "
                    "(normal or hypoxanthine-spiked serum, before or after uricase treatment), the replica, and "
                    "the protocol number."
                ),
                "protocol": protocol,
            }

        raise ValueError(f"Unexpected serum folder: {folder_name}")

    def _build_rows(self) -> list[dict]:
        if not self.dataset_zip.exists():
            raise FileNotFoundError(f"Missing spectral archive: {self.dataset_zip}")

        rows: list[dict] = []
        reference_axis: np.ndarray | None = None

        with ZipFile(self.dataset_zip, "r") as archive:
            members = sorted(
                member
                for member in archive.namelist()
                if member.lower().endswith(".txt")
                and Path(member).parts
                and Path(member).parts[0] in self.SERUM_FOLDERS
            )

            for member_name in members:
                folder_name = Path(member_name).parts[0]
                wavenumbers, intensity_values = self._read_bwtek_member(archive, member_name)
                if reference_axis is None:
                    reference_axis = wavenumbers
                elif not np.allclose(reference_axis, wavenumbers, atol=0.05):
                    raise ValueError(f"{member_name} does not match the grounded native serum axis.")

                parsed = self._parse_member_metadata(folder_name=folder_name, member_name=member_name)
                source_row_id = Path(member_name).stem
                biosample_id = f"{self.dataset_id}_{source_row_id}"

                rows.append(
                    {
                        "biosample_id": biosample_id,
                        "dataset_id": self.dataset_id,
                        "source_row_id": source_row_id,
                        "sample_id": parsed["sample_id"],
                        "patient_id": None,
                        "replicate_id": parsed["replicate_id"],
                        "class_label": parsed["class_label"],
                        "subclass_label": parsed["subclass_label"],
                        "source_file": f"{self.DATASET_ZIP}::{member_name}",
                        "wavenumbers": wavenumbers,
                        "intensity_values": intensity_values,
                        "folder_name": folder_name,
                        "protocol": parsed.get("protocol"),
                        "row_note": parsed["notes"],
                        "spike_concentration": parsed.get("spike_concentration"),
                    }
                )

        if not rows:
            raise ValueError("No serum biosample TXT spectra were found in the serum Ag colloids archive.")
        return rows

    def _preprocessing_summary(self) -> str:
        return (
            "Raw ingest preserves the native released BWtek TXT spectra using Raman Shift and Dark Subtracted #1 "
            "columns exactly as exposed in the instrument export. Released R scripts show downstream crop "
            "(400-1800 cm^-1), modpolyfit baseline correction, and vector normalization steps, but those are not "
            "applied during raw GAIRA ingestion."
        )

    def _base_notes(self, row: dict) -> str:
        protocol_text = f" Protocol={row['protocol']}." if row.get("protocol") else ""
        spike_text = (
            f" Spiking concentration={row['spike_concentration']}."
            if row.get("spike_concentration")
            else ""
        )
        return (
            f"{row['row_note']} Current GAIRA framing treats serum_ag_colloids as a serum biochemical "
            f"interpretation archive on Ag colloids rather than a disease benchmark.{protocol_text}{spike_text} "
            "Non-serum folders in the same release (metabolite references, isotopic controls, literature files) "
            "were not ingested into the biosample tables."
        )

    def _collection_protocol(self, row: dict) -> str | None:
        if row["subclass_label"] == "donors_serum_sers":
            return "serum samples collected from healthy blood donors at Ospedale Maggiore, Trieste"
        return None

    def _preparation_protocol(self, row: dict) -> str | None:
        if row["subclass_label"] == "commercial_serum_merck":
            return "commercial serum purchased from Merck analyzed with Protocol 1"
        if row["subclass_label"] == "spiked_commercial_serum_merck":
            return "commercial serum purchased from Merck spiked with metabolites at released physiological concentrations"
        if row["subclass_label"] == "uricase_serum_experiment":
            return "commercial serum normal or hypoxanthine-spiked before and after uricase treatment"
        return None

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

        rows: list[dict] = []
        for peak_rank, peak_index in enumerate(peak_indices, start=1):
            rows.append(
                {
                    "peak_rank": peak_rank,
                    "peak_cm": float(wavenumbers[peak_index]),
                    "peak_intensity": float(properties["peak_heights"][peak_rank - 1]),
                    "prominence": float(properties["prominences"][peak_rank - 1]),
                }
            )
        return rows

    def audit(self) -> None:
        subclass_counts = Counter(row["subclass_label"] for row in self.rows)
        class_counts = Counter((row["subclass_label"], row["class_label"]) for row in self.rows)
        axis = self.rows[0]["wavenumbers"]

        print("serum_ag_colloids dataset audit")
        print(f"Dataset root: {self.dataset_root}")
        print(f"Spectral archive: {self.dataset_zip}")
        print(f"Scripts archive: {self.scripts_zip}")
        print(f"Instructions: {self.instructions_path}")
        print(f"Native axis: {axis[0]:.2f} to {axis[-1]:.2f} cm^-1 ({len(axis)} points)")
        print("Serum biosample subclass counts:")
        for subclass_label, count in sorted(subclass_counts.items()):
            print(f"  {subclass_label}: {count}")
        print("Serum biosample class counts preview:")
        for (subclass_label, class_label), count in sorted(class_counts.items()):
            print(f"  {subclass_label} / {class_label}: {count}")
        print("Non-serum folders in the release were intentionally excluded from biosample ingestion.")

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
                    "collection_protocol": self._collection_protocol(row),
                    "preparation_protocol": self._preparation_protocol(row),
                    "instrument": "B&Wtek i-Raman Plus portable system (BWS465-785S)",
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
            for peak_row in self._detect_peaks_for_spectrum(
                intensity_values=row["intensity_values"],
                wavenumbers=row["wavenumbers"],
            ):
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

        print("serum_ag_colloids ingestion complete.")
        print(f"Inserted biosample_metadata rows: {metadata_count}")
        print(f"Inserted biosample_spectra rows: {spectra_count}")
        print(f"Inserted biosample_spectrum_points rows: {point_count}")
        print(f"Inserted biosample_peaks rows: {peak_count}")
