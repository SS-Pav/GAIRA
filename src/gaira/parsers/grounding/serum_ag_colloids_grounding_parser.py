import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class SerumAgColloidsGroundingParser:
    """Grounding-layer parser for selected controlled reference folders from serum_ag_colloids."""

    DATASET_ZIP = "dataset_spectral_data.zip"
    SOURCE_DATASET_ID = "serum_ag_colloids"
    GROUNDING_FOLDERS = {
        "SERS metabolites": ("sers_metabolites", "direct_sers_metabolite_reference"),
        "SERS metabolites for fitting": ("sers_metabolites_for_fitting", "sers_fitting_reference"),
        "isotopic": ("isotopic", "isotopic_validation"),
    }

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        self.dataset_id = dataset_id
        self.dataset_root = Path(dataset_root)
        self.db_path = Path(db_path)
        self.dataset_zip = self.dataset_root / self.DATASET_ZIP
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
        experiment_family, grounding_role = self.GROUNDING_FOLDERS[folder_name]

        if folder_name == "SERS metabolites":
            if len(parts) < 5:
                raise ValueError(f"Unexpected SERS metabolite filename layout: {member_name}")
            _, _, compound_label, concentration_label, replicate_id = parts[:5]
            return {
                "experiment_family": experiment_family,
                "grounding_role": grounding_role,
                "compound_label": compound_label,
                "class_label": compound_label,
                "concentration_label": concentration_label,
                "replicate_id": replicate_id,
                "notes": (
                    "Released SERS metabolite reference spectrum at the physiological concentration given "
                    "in the filename. The release instructions explicitly describe this folder as SERS "
                    "spectra of metabolites analyzed with Protocol 1."
                ),
            }

        if folder_name == "SERS metabolites for fitting":
            if len(parts) < 3:
                raise ValueError(f"Unexpected fitting-reference filename layout: {member_name}")
            source_row_id, replicate_id, compound_label = parts[:3]
            return {
                "experiment_family": experiment_family,
                "grounding_role": grounding_role,
                "compound_label": compound_label,
                "class_label": compound_label,
                "concentration_label": None,
                "replicate_id": replicate_id,
                "sample_index": source_row_id,
                "notes": (
                    "Released SERS fitting-reference spectrum. The instructions explicitly describe this "
                    "folder as hypoxanthine and uric-acid references used for spectral fitting in Figure 9."
                ),
            }

        if folder_name == "isotopic":
            if len(parts) < 3:
                raise ValueError(f"Unexpected isotopic filename layout: {member_name}")
            source_row_id, replicate_id, class_label = parts[:3]
            return {
                "experiment_family": experiment_family,
                "grounding_role": grounding_role,
                "compound_label": class_label,
                "class_label": class_label,
                "concentration_label": "280uM_or_release_defined_condition",
                "replicate_id": replicate_id,
                "sample_index": source_row_id,
                "notes": (
                    "Released isotopic-control SERS spectrum. The instructions explicitly describe these as "
                    "normal and isotopically labeled uric acid, with and without HSA, before and after filtration."
                ),
            }

        raise ValueError(f"Unexpected grounding folder: {folder_name}")

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
                and Path(member).parts[0] in self.GROUNDING_FOLDERS
            )

            for member_name in members:
                folder_name = Path(member_name).parts[0]
                wavenumbers, intensity_values = self._read_bwtek_member(archive, member_name)
                if reference_axis is None:
                    reference_axis = wavenumbers
                elif not np.allclose(reference_axis, wavenumbers, atol=0.05):
                    raise ValueError(f"{member_name} does not match the grounded native grounding axis.")

                parsed = self._parse_member_metadata(folder_name=folder_name, member_name=member_name)
                source_row_id = Path(member_name).stem
                grounding_id = f"{self.dataset_id}_{source_row_id}"
                rows.append(
                    {
                        "grounding_id": grounding_id,
                        "dataset_id": self.dataset_id,
                        "source_dataset_id": self.SOURCE_DATASET_ID,
                        "source_row_id": source_row_id,
                        "source_file": f"{self.DATASET_ZIP}::{member_name}",
                        "wavenumbers": wavenumbers,
                        "intensity_values": intensity_values,
                        **parsed,
                    }
                )

        if not rows:
            raise ValueError("No selected grounding TXT spectra were found in the serum Ag colloids archive.")
        return rows

    def _preprocessing_summary(self) -> str:
        return (
            "Raw grounding ingest preserves the native released BWtek TXT spectra using Raman Shift and "
            "Dark Subtracted #1 columns exactly as exposed in the instrument export. Released R scripts "
            "apply downstream crop (400-1800 cm^-1), modpolyfit baseline correction, and vector normalization "
            "for analysis figures, but those steps are not applied during grounding-layer raw ingestion."
        )

    def _base_notes(self, row: dict) -> str:
        return (
            f"{row['notes']} Current GAIRA_GROUNDING framing keeps this as a controlled SERS grounding asset "
            f"from the serum_ag_colloids release rather than a biosample benchmark spectrum. Source family={row['experiment_family']}."
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
        family_counts = Counter(row["experiment_family"] for row in self.rows)
        class_counts = Counter((row["experiment_family"], row["class_label"]) for row in self.rows)
        axis = self.rows[0]["wavenumbers"]
        print("serum_ag_colloids_grounding audit")
        print(f"Dataset root: {self.dataset_root}")
        print(f"Source archive: {self.dataset_zip}")
        print(f"Grounded native axis: {axis[0]:.2f} to {axis[-1]:.2f} cm^-1 ({len(axis)} points)")
        print("Grounding family counts:")
        for family, count in sorted(family_counts.items()):
            print(f"  {family}: {count}")
        print("Grounding class counts preview:")
        for (family, class_label), count in sorted(class_counts.items()):
            print(f"  {family} / {class_label}: {count}")
        print("Deferred folders in the same archive: literature, digitized literature spectra, Raman metabolites.")

    def extract_metadata(self) -> pd.DataFrame:
        preprocessing_summary = self._preprocessing_summary()
        rows = []
        for row in self.rows:
            wavenumbers = row["wavenumbers"]
            rows.append(
                {
                    "grounding_id": row["grounding_id"],
                    "dataset_id": row["dataset_id"],
                    "source_dataset_id": row["source_dataset_id"],
                    "source_row_id": row["source_row_id"],
                    "experiment_family": row["experiment_family"],
                    "grounding_role": row["grounding_role"],
                    "modality": "SERS",
                    "compound_label": row["compound_label"],
                    "class_label": row["class_label"],
                    "concentration_label": row["concentration_label"],
                    "replicate_id": row["replicate_id"],
                    "source_file": row["source_file"],
                    "biosample_context": "serum Ag colloids study controlled reference family",
                    "substrate_type": "colloidal_sers_substrate",
                    "substrate_material": "silver",
                    "instrument": "B&Wtek i-Raman Plus portable system (BWS465-785S)",
                    "laser_wavelength_nm": "785",
                    "spectral_range": f"{wavenumbers[0]:.2f}-{wavenumbers[-1]:.2f}",
                    "preprocessing_summary": preprocessing_summary,
                    "notes": self._base_notes(row),
                }
            )
        metadata_df = pd.DataFrame(rows)
        print(f"Prepared {len(metadata_df)} grounding metadata rows.")
        return metadata_df

    def extract_spectra(self) -> pd.DataFrame:
        preprocessing_summary = self._preprocessing_summary()
        rows = []
        for row in self.rows:
            wavenumbers = row["wavenumbers"]
            rows.append(
                {
                    "grounding_id": row["grounding_id"],
                    "dataset_id": row["dataset_id"],
                    "source_dataset_id": row["source_dataset_id"],
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
        print(f"Prepared {len(spectra_df)} grounding spectra rows.")
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
                        "grounding_id": row["grounding_id"],
                        "dataset_id": row["dataset_id"],
                        "source_dataset_id": row["source_dataset_id"],
                        "source_row_id": row["source_row_id"],
                        "point_index": point_index,
                        "wavenumber": float(wavenumber),
                        "intensity": float(intensity),
                    }
                )
        points_df = pd.DataFrame(rows)
        print(f"Prepared {len(points_df)} grounding spectrum point rows.")
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
                        "grounding_id": row["grounding_id"],
                        "dataset_id": row["dataset_id"],
                        "source_dataset_id": row["source_dataset_id"],
                        "source_row_id": row["source_row_id"],
                        **peak_row,
                    }
                )
        peaks_df = pd.DataFrame(rows)
        print(f"Prepared {len(peaks_df)} grounding peak rows.")
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
                "grounding_metadata",
                "grounding_spectra",
                "grounding_spectrum_points",
                "grounding_peaks",
            ):
                connection.execute(f"DELETE FROM {table_name} WHERE dataset_id = ?", [self.dataset_id])

            metadata_count = self._insert_dataframe(connection, "grounding_metadata", metadata_df)
            spectra_count = self._insert_dataframe(connection, "grounding_spectra", spectra_df)
            point_count = self._insert_dataframe(connection, "grounding_spectrum_points", points_df)
            peak_count = self._insert_dataframe(connection, "grounding_peaks", peaks_df)

        print("serum_ag_colloids_grounding ingestion complete.")
        print(f"Inserted grounding_metadata rows: {metadata_count}")
        print(f"Inserted grounding_spectra rows: {spectra_count}")
        print(f"Inserted grounding_spectrum_points rows: {point_count}")
        print(f"Inserted grounding_peaks rows: {peak_count}")
