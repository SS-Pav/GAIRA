import io
import json
from pathlib import Path
from zipfile import ZipFile

import duckdb
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks

from gaira.parsers.biosample.base import BiosampleParserBase


class DiabetesPlasmaEVSERSParser(BiosampleParserBase):
    """Parser for the released Figure 3 SERS assets in the diabetes plasma EV Zenodo archive."""

    ZIP_NAME = "Diabetes_Raw_Data_Codes.zip"
    IMPACT_MEMBER = "Diabetes - Raw Data - Codes/Figure 3/data/RawDataImpact.mat"
    STRONG_MEMBER = "Diabetes - Raw Data - Codes/Figure 3/data/RawDataStrong.mat"
    PATIENT_MEMBER = "Diabetes - Raw Data - Codes/Figure 3/data/patient_data.csv"
    POINT_COUNT = 737
    PIX = np.array([263, 367, 492, 512, 590, 782, 872, 887], dtype=float)
    CAL = np.array([620.9, 795.8, 1001.4, 1031.8, 1155.3, 1450.5, 1583.1, 1602.3], dtype=float)
    RANGE = np.arange(162, 899, dtype=int) - 1

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        super().__init__(dataset_id=dataset_id, dataset_root=dataset_root, db_path=db_path)
        self.archive_path = self.dataset_root / self.ZIP_NAME
        self.wavenumbers = self._build_wavenumbers()
        self.patient_df = self._load_patient_table()
        self.asset_cache = self._load_all_assets()

    def _build_wavenumbers(self) -> np.ndarray:
        fit = np.polyfit(self.PIX, self.CAL, 3)
        calibrated = np.polyval(fit, np.arange(1, 1651, dtype=float))
        cropped = calibrated[self.RANGE]
        if len(cropped) != self.POINT_COUNT:
            raise ValueError(f"Expected {self.POINT_COUNT} cropped points, found {len(cropped)}.")
        return cropped.astype(float)

    def _load_patient_table(self) -> pd.DataFrame:
        if not self.archive_path.exists():
            raise FileNotFoundError(
                f"Missing archive: {self.archive_path}. Download diabetes_plasma_ev_sers first."
            )

        with ZipFile(self.archive_path, "r") as archive:
            patient_bytes = archive.read(self.PATIENT_MEMBER)

        patient_df = pd.read_csv(io.BytesIO(patient_bytes))
        patient_df.columns = [str(column).strip() for column in patient_df.columns]
        return patient_df

    def _load_mat_cells(self, member_name: str) -> list[np.ndarray]:
        with ZipFile(self.archive_path, "r") as archive:
            mat_bytes = archive.read(member_name)

        mat = loadmat(io.BytesIO(mat_bytes), squeeze_me=True, struct_as_record=False)
        if "smoothed_spectra" not in mat:
            raise ValueError(f"{member_name} does not contain smoothed_spectra.")

        cells = list(mat["smoothed_spectra"])
        normalized_cells: list[np.ndarray] = []
        for cell in cells:
            matrix = np.asarray(cell, dtype=float)
            if matrix.ndim != 2:
                raise ValueError(f"{member_name} contains a non-2D cell with shape {matrix.shape}.")
            if matrix.shape[0] != self.POINT_COUNT:
                raise ValueError(
                    f"{member_name} cell has {matrix.shape[0]} points; expected {self.POINT_COUNT}."
                )
            normalized_cells.append(matrix)

        return normalized_cells

    def _load_all_assets(self) -> dict[str, dict]:
        impact_cells = self._load_mat_cells(self.IMPACT_MEMBER)
        strong_cells = self._load_mat_cells(self.STRONG_MEMBER)
        return {
            "Impact": {
                "member": self.IMPACT_MEMBER,
                "sample_count": len(impact_cells),
                "scan_matrices": impact_cells,
            },
            "Strong-D": {
                "member": self.STRONG_MEMBER,
                "sample_count": len(strong_cells),
                "scan_matrices": strong_cells,
            },
        }

    def _base_notes(self, class_label: str) -> str:
        return (
            f"Released Figure 3 MAT asset for {class_label}. "
            "The archive stores preprocessed 737-point spectra in one cell per source folder. "
            "Patient IDs are not embedded in the released MAT cells, and patient_data.csv cannot be joined back "
            "to cell order defensibly from the release alone. Four-subgroup A-NWD/A-OWD/W-NWD/W-OWD mapping was "
            "therefore not reconstructed. Current GAIRA row labels are limited to the archive-supported cohort "
            "families Impact and Strong-D, so this dataset should be treated as a weak-label Impact-vs-StrongD "
            "comparison rather than a four-subgroup benchmark."
        )

    def _preprocessing_summary(self) -> str:
        return (
            "Released Figure 3 asset stores preprocessed spectra after cubic Raman calibration from 1650-pixel data "
            "and cropping to MATLAB range 162:898 (~447.9-1619.3 cm^-1). "
            "Helper code applies minimum-spectrum subtraction, ALS baseline correction, and Savitzky-Golay smoothing. "
            "An L2 normalization helper is included in the archive, but it is not explicitly called in the released raw-data reader scripts."
        )

    def _iter_all_rows(self):
        for class_label, asset in self.asset_cache.items():
            asset_key = Path(asset["member"]).stem.lower()
            for sample_index, matrix in enumerate(asset["scan_matrices"], start=1):
                sample_id = f"{asset_key}_sample_{sample_index:03d}"
                for replicate_index in range(matrix.shape[1]):
                    source_row_id = f"{asset_key}__sample_{sample_index:03d}__scan_{replicate_index + 1:04d}"
                    biosample_id = f"{self.dataset_id}_{source_row_id}"
                    yield {
                        "biosample_id": biosample_id,
                        "dataset_id": self.dataset_id,
                        "source_row_id": source_row_id,
                        "sample_id": sample_id,
                        "patient_id": None,
                        "replicate_id": f"scan_{replicate_index + 1:04d}",
                        "class_label": class_label,
                        "subclass_label": "figure3_processed_archive",
                        "source_file": f"{self.ZIP_NAME}::{asset['member']}::cell_{sample_index:03d}",
                        "wavenumbers": self.wavenumbers,
                        "intensity_values": np.asarray(matrix[:, replicate_index], dtype=float),
                    }

    def _detect_peaks_for_spectrum(self, intensity_values: np.ndarray) -> list[dict]:
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
                    "peak_cm": float(self.wavenumbers[peak_index]),
                    "peak_intensity": float(properties["peak_heights"][peak_rank - 1]),
                    "prominence": float(properties["prominences"][peak_rank - 1]),
                }
            )
        return rows

    def audit(self) -> None:
        print("diabetes_plasma_ev_sers dataset audit")
        print(f"Archive: {self.archive_path}")
        print(f"Patient table rows: {len(self.patient_df)}")
        print("Patient table counts by Group:")
        print(self.patient_df.groupby("Group").size().to_string())
        print(
            f"Stored Figure 3 spectral axis: {self.wavenumbers[0]:.3f} to {self.wavenumbers[-1]:.3f} cm^-1 "
            f"({len(self.wavenumbers)} points)"
        )
        for class_label, asset in self.asset_cache.items():
            scan_counts = sorted({matrix.shape[1] for matrix in asset["scan_matrices"]})
            total_spectra = int(sum(matrix.shape[1] for matrix in asset["scan_matrices"]))
            print(
                f"{class_label}: {asset['sample_count']} sample cells, "
                f"{total_spectra} total spectra, scan counts {scan_counts}"
            )
        print(
            "Archive listing contains Figure 3 SERS assets and Figure 2 characterization assets. "
            "No explicit RNA-seq files were found in the zip manifest."
        )
        print(
            "Initial onboarding is grounded to the released Figure 3 SERS MAT assets only. "
            "Paper-level A-NWD/A-OWD/W-NWD/W-OWD subgroup context is not assigned at row level because the released "
            "MAT cells do not embed patient IDs and the cell-to-patient mapping is not defensible from the release."
        )

    def extract_metadata(self) -> pd.DataFrame:
        rows = []
        preprocessing_summary = self._preprocessing_summary()
        for spectrum_row in self._iter_all_rows():
            rows.append(
                {
                    "biosample_id": spectrum_row["biosample_id"],
                    "dataset_id": self.dataset_id,
                    "source_row_id": spectrum_row["source_row_id"],
                    "sample_id": spectrum_row["sample_id"],
                    "patient_id": spectrum_row["patient_id"],
                    "replicate_id": spectrum_row["replicate_id"],
                    "biosample_type": "extracellular vesicles",
                    "matrix": "plasma",
                    "disease_context": "type 2 diabetes mellitus",
                    "class_label": spectrum_row["class_label"],
                    "subclass_label": spectrum_row["subclass_label"],
                    "collection_protocol": None,
                    "preparation_protocol": "intact EV-enriched plasma isolate",
                    "instrument": "custom-built Raman system",
                    "laser_wavelength_nm": "785",
                    "spectral_range": f"{self.wavenumbers[0]:.3f}-{self.wavenumbers[-1]:.3f}",
                    "preprocessing_summary": preprocessing_summary,
                    "source_file": spectrum_row["source_file"],
                    "notes": self._base_notes(spectrum_row["class_label"]),
                }
            )

        metadata_df = pd.DataFrame(rows)
        print(f"Prepared {len(metadata_df)} biosample metadata rows.")
        return metadata_df

    def extract_spectra(self) -> pd.DataFrame:
        rows = []
        preprocessing_summary = self._preprocessing_summary()
        for spectrum_row in self._iter_all_rows():
            rows.append(
                {
                    "biosample_id": spectrum_row["biosample_id"],
                    "dataset_id": self.dataset_id,
                    "source_row_id": spectrum_row["source_row_id"],
                    "x_min": float(self.wavenumbers[0]),
                    "x_max": float(self.wavenumbers[-1]),
                    "n_points": int(len(self.wavenumbers)),
                    "wavenumbers_json": json.dumps([float(value) for value in self.wavenumbers]),
                    "intensity_json": json.dumps([float(value) for value in spectrum_row["intensity_values"]]),
                    "normalized_flag": "unknown",
                    "preprocessing_summary": preprocessing_summary,
                }
            )

        spectra_df = pd.DataFrame(rows)
        print(f"Prepared {len(spectra_df)} biosample spectra rows.")
        return spectra_df

    def extract_spectrum_points(self) -> pd.DataFrame:
        rows = []
        for spectrum_row in self._iter_all_rows():
            for point_index, (wavenumber, intensity) in enumerate(
                zip(self.wavenumbers, spectrum_row["intensity_values"]),
                start=1,
            ):
                rows.append(
                    {
                        "biosample_id": spectrum_row["biosample_id"],
                        "dataset_id": self.dataset_id,
                        "source_row_id": spectrum_row["source_row_id"],
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
        for spectrum_row in self._iter_all_rows():
            for peak_row in self._detect_peaks_for_spectrum(spectrum_row["intensity_values"]):
                rows.append(
                    {
                        "biosample_id": spectrum_row["biosample_id"],
                        "dataset_id": self.dataset_id,
                        "source_row_id": spectrum_row["source_row_id"],
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

        print("diabetes_plasma_ev_sers ingestion complete.")
        print(f"Inserted biosample_metadata rows: {metadata_count}")
        print(f"Inserted biosample_spectra rows: {spectra_count}")
        print(f"Inserted biosample_spectrum_points rows: {point_count}")
        print(f"Inserted biosample_peaks rows: {peak_count}")
