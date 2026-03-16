import io
import json
from pathlib import Path
from zipfile import ZipFile

import duckdb
import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat
from scipy.signal import find_peaks

from gaira.parsers.biosample.base import BiosampleParserBase


class Small2023EVParser(BiosampleParserBase):
    """Concrete parser for the Small 2023 EV Raman dataset."""

    AXIS_ARCHIVE_MEMBER = "Main_Text/Fig3/Data/Raw/data.mat"
    FIG3_NORM_ARCHIVE_MEMBER = "Main_Text/Fig3/Data/Norm/data_BC_NORM.mat"
    EXPECTED_POINT_COUNT = 1131

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        super().__init__(dataset_id=dataset_id, dataset_root=dataset_root, db_path=db_path)
        self.readme_path = self.dataset_root / "Readme.docx"
        self.main_text_zip_path = self.dataset_root / "Main_Text.zip"
        self.probe_files = self._discover_probe_files()
        self.calibrated_wavenumbers = self._load_calibrated_wavenumbers()
        self.skipped_rows: list[tuple[str, str]] = []
        self.skipped_assets: list[tuple[str, str]] = []
        self.fig3_norm_available = self._check_fig3_norm_archive()

    def _discover_probe_files(self) -> list[Path]:
        """Find locally available normalized probe files."""
        probe_files = sorted(self.dataset_root.glob("NormedProbe*.mat"))
        if not probe_files:
            raise FileNotFoundError(
                f"No NormedProbe*.mat files were found under {self.dataset_root}. "
                "Download the normalized probe file(s) first."
            )
        return probe_files

    def _load_calibrated_wavenumbers(self) -> np.ndarray:
        """Load the real Raman shift axis from the released Main_Text archive."""
        if not self.main_text_zip_path.exists():
            raise FileNotFoundError(
                "Main_Text.zip is required to ground the native Raman shift axis for small2023_ev. "
                f"Expected file: {self.main_text_zip_path}"
            )

        with ZipFile(self.main_text_zip_path, "r") as archive:
            try:
                raw_bytes = archive.read(self.AXIS_ARCHIVE_MEMBER)
            except KeyError as exc:
                raise FileNotFoundError(
                    f"Could not find {self.AXIS_ARCHIVE_MEMBER} inside {self.main_text_zip_path}"
                ) from exc

        axis_mat = loadmat(io.BytesIO(raw_bytes), squeeze_me=True, struct_as_record=False)
        if "Calx" not in axis_mat:
            raise ValueError(
                f"The released axis file {self.AXIS_ARCHIVE_MEMBER} does not contain Calx."
            )

        calx = np.asarray(axis_mat["Calx"], dtype=float).ravel()
        if len(calx) != self.EXPECTED_POINT_COUNT:
            raise ValueError(
                f"Calx length mismatch: expected {self.EXPECTED_POINT_COUNT}, found {len(calx)}."
            )

        return calx

    def _check_fig3_norm_archive(self) -> bool:
        """Check whether the released normalized Fig3 archive asset is available."""
        if not self.main_text_zip_path.exists():
            return False

        with ZipFile(self.main_text_zip_path, "r") as archive:
            return self.FIG3_NORM_ARCHIVE_MEMBER in archive.namelist()

    def _load_probe_struct(self, probe_path: Path) -> tuple[str, object]:
        """Load one normalized probe MAT file and return its struct variable."""
        struct_info = whosmat(str(probe_path))
        struct_names = [name for name, _shape, kind in struct_info if kind == "struct"]
        if not struct_names:
            raise ValueError(f"No MATLAB struct was found in {probe_path}")

        struct_name = struct_names[0]
        mat = loadmat(str(probe_path), squeeze_me=True, struct_as_record=False)
        if struct_name not in mat:
            raise ValueError(f"Expected struct {struct_name} was not found in {probe_path}")

        return struct_name, mat[struct_name]

    def _load_probe_class_matrices(self, probe_path: Path) -> dict[str, np.ndarray]:
        """Read all numeric class matrices from one normalized probe file."""
        struct_name, struct_value = self._load_probe_struct(probe_path)
        class_matrices: dict[str, np.ndarray] = {}

        for field_name in sorted(name for name in dir(struct_value) if not name.startswith("_")):
            field_value = getattr(struct_value, field_name)
            if not isinstance(field_value, np.ndarray):
                continue
            if field_value.ndim != 2:
                continue
            if field_value.shape[1] != self.EXPECTED_POINT_COUNT:
                continue
            if not np.issubdtype(field_value.dtype, np.number):
                continue

            class_matrices[field_name] = np.asarray(field_value, dtype=float)

        if not class_matrices:
            raise ValueError(
                f"No 2D numeric class matrices with {self.EXPECTED_POINT_COUNT} columns were found "
                f"in {probe_path}::{struct_name}"
            )

        return class_matrices

    def _safe_load_probe_class_matrices(self, probe_path: Path) -> dict[str, np.ndarray] | None:
        """Load one probe MAT file, skipping incomplete downloads gracefully."""
        try:
            return self._load_probe_class_matrices(probe_path)
        except Exception as exc:
            self.skipped_assets.append((probe_path.name, str(exc)))
            return None

    def _load_fig3_norm_class_matrices(self) -> dict[str, np.ndarray]:
        """Load the released normalized Fig3 class matrices with explicit labels."""
        if not self.fig3_norm_available:
            return {}

        with ZipFile(self.main_text_zip_path, "r") as archive:
            raw_bytes = archive.read(self.FIG3_NORM_ARCHIVE_MEMBER)

        mat = loadmat(io.BytesIO(raw_bytes), squeeze_me=True, struct_as_record=False)
        class_matrices: dict[str, np.ndarray] = {}

        for field_name, field_value in sorted(mat.items()):
            if field_name.startswith("__") or field_name == "Calx":
                continue
            if not isinstance(field_value, np.ndarray):
                continue
            if field_value.ndim != 2:
                continue
            if field_value.shape[1] != self.EXPECTED_POINT_COUNT:
                continue
            if not np.issubdtype(field_value.dtype, np.number):
                continue
            class_matrices[field_name] = np.asarray(field_value, dtype=float)

        return class_matrices

    def _build_row_identity(
        self,
        probe_name: str,
        class_label: str,
        row_index: int,
    ) -> dict[str, str]:
        """Create stable row identifiers from the released file, class, and row index."""
        source_row_id = f"{probe_name}__{class_label}__{row_index:06d}"
        biosample_id = f"{self.dataset_id}_{source_row_id}"
        return {
            "biosample_id": biosample_id,
            "source_row_id": source_row_id,
            "source_file": f"{probe_name}.mat::{class_label}",
        }

    def _detect_peaks_for_spectrum(self, intensity_values: np.ndarray) -> list[dict]:
        """Detect conservative peaks from one normalized spectrum."""
        clipped_values = np.clip(np.asarray(intensity_values, dtype=float), a_min=0.0, a_max=None)
        shifted_values = clipped_values - float(clipped_values.min())
        value_range = float(shifted_values.max())
        if value_range <= 0:
            return []

        normalized_values = shifted_values / value_range
        peak_indices, properties = find_peaks(
            normalized_values,
            prominence=0.05,
            height=0.05,
            distance=5,
        )

        peak_rows: list[dict] = []
        for peak_rank, peak_index in enumerate(peak_indices, start=1):
            peak_rows.append(
                {
                    "peak_rank": peak_rank,
                    "peak_cm": float(self.calibrated_wavenumbers[peak_index]),
                    "peak_intensity": float(properties["peak_heights"][peak_rank - 1]),
                    "prominence": float(properties["prominences"][peak_rank - 1]),
                }
            )

        return peak_rows

    def _iter_all_rows(self):
        """Yield one released spectrum at a time across all grounded local assets."""
        for probe_path in self.probe_files:
            probe_name = probe_path.stem.lower()
            class_matrices = self._safe_load_probe_class_matrices(probe_path)
            if class_matrices is None:
                continue

            for class_label, matrix in class_matrices.items():
                for row_index, intensity_values in enumerate(matrix, start=1):
                    identity = self._build_row_identity(probe_name, class_label, row_index)
                    yield {
                        **identity,
                        "probe_name": probe_name,
                        "class_label": class_label,
                        "row_index": row_index,
                        "intensity_values": np.asarray(intensity_values, dtype=float),
                    }

        fig3_norm_class_matrices = self._load_fig3_norm_class_matrices()
        for class_label, matrix in fig3_norm_class_matrices.items():
            source_name = "fig3_norm_archive"
            for row_index, intensity_values in enumerate(matrix, start=1):
                source_row_id = f"{source_name}__{class_label}__{row_index:06d}"
                biosample_id = f"{self.dataset_id}_{source_row_id}"
                yield {
                    "biosample_id": biosample_id,
                    "source_row_id": source_row_id,
                    "source_file": f"{self.FIG3_NORM_ARCHIVE_MEMBER}::{class_label}",
                    "probe_name": source_name,
                    "class_label": class_label,
                    "row_index": row_index,
                    "intensity_values": np.asarray(intensity_values, dtype=float),
                }

    def audit(self) -> None:
        """Report the grounded structure found in the real local files."""
        print("small2023_ev dataset audit")
        print(f"Dataset root: {self.dataset_root}")
        print(f"Readme present: {self.readme_path.exists()}")
        print(f"Main_Text archive present: {self.main_text_zip_path.exists()}")
        print(f"Native Raman axis: {self.calibrated_wavenumbers[0]:.1f} to {self.calibrated_wavenumbers[-1]:.1f} cm^-1")
        print(f"Axis points: {len(self.calibrated_wavenumbers)}")
        print("Normalized probe files found:")

        total_rows = 0
        for probe_path in self.probe_files:
            print(f"  {probe_path.name}")
            class_matrices = self._safe_load_probe_class_matrices(probe_path)
            if class_matrices is None:
                print("    skipped: file is incomplete or unreadable")
                continue
            for class_label, matrix in class_matrices.items():
                print(f"    {class_label}: {matrix.shape[0]} spectra x {matrix.shape[1]} points")
                total_rows += int(matrix.shape[0])

        if self.fig3_norm_available:
            print("Archive normalized class matrices found:")
            for class_label, matrix in self._load_fig3_norm_class_matrices().items():
                print(f"  {class_label}: {matrix.shape[0]} spectra x {matrix.shape[1]} points")
                total_rows += int(matrix.shape[0])

        print(f"Total released spectra available from local normalized probe files: {total_rows}")
        print(
            "The released MAT file stores literal class codes such as c00 and c100. "
            "The main-text assets confirm this is a five-cell-line EV dataset with mixture experiments, "
            "but the local NormedProbe1 struct does not provide a direct row-level crosswalk from c00-c100 "
            "to a fuller biological label, so the parser preserves the released class codes as-is."
        )

    def extract_metadata(self) -> pd.DataFrame:
        """Build biosample metadata rows from the released normalized probe matrices."""
        metadata_rows: list[dict] = []

        for spectrum_row in self._iter_all_rows():
            metadata_rows.append(
                {
                    "biosample_id": spectrum_row["biosample_id"],
                    "dataset_id": self.dataset_id,
                    "source_row_id": spectrum_row["source_row_id"],
                    "sample_id": None,
                    "patient_id": None,
                    "replicate_id": None,
                    "biosample_type": "extracellular vesicles",
                    "matrix": None,
                    "disease_context": None,
                    "class_label": spectrum_row["class_label"],
                    "subclass_label": spectrum_row["probe_name"],
                    "collection_protocol": None,
                    "preparation_protocol": None,
                    "instrument": None,
                    "laser_wavelength_nm": None,
                    "spectral_range": f"{self.calibrated_wavenumbers[0]:.1f}-{self.calibrated_wavenumbers[-1]:.1f}",
                    "preprocessing_summary": (
                        "Normalized spectra were loaded from locally available NormedProbe*.mat files. "
                        "The Raman shift axis was grounded from Main_Text/Fig3/Data/Raw/data.mat Calx "
                        "inside the released Main_Text.zip archive."
                    ),
                    "source_file": spectrum_row["source_file"],
                    "notes": (
                        "class_label preserves the released MAT struct field name exactly. "
                        "The broader paper context confirms EV cell-line and mixture experiments, "
                        "but this local file does not provide a direct row-level label crosswalk beyond "
                        "the released class codes."
                    ),
                }
            )

        metadata_df = pd.DataFrame(metadata_rows)
        print(f"Prepared {len(metadata_df)} biosample metadata rows.")
        return metadata_df

    def extract_spectra(self) -> pd.DataFrame:
        """Build full-spectrum rows for the released normalized probe matrices."""
        spectra_rows: list[dict] = []

        for spectrum_row in self._iter_all_rows():
            intensity_values = spectrum_row["intensity_values"]
            spectra_rows.append(
                {
                    "biosample_id": spectrum_row["biosample_id"],
                    "dataset_id": self.dataset_id,
                    "source_row_id": spectrum_row["source_row_id"],
                    "x_min": float(self.calibrated_wavenumbers[0]),
                    "x_max": float(self.calibrated_wavenumbers[-1]),
                    "n_points": int(len(self.calibrated_wavenumbers)),
                    "wavenumbers_json": json.dumps(self.calibrated_wavenumbers.tolist()),
                    "intensity_json": json.dumps([float(value) for value in intensity_values]),
                    "normalized_flag": "yes",
                    "preprocessing_summary": (
                        "Released normalized spectra from NormedProbe*.mat. "
                        "The notebooks clip negative values to zero before downstream modeling, "
                        "but the stored raw biosample_spectra rows preserve the released values."
                    ),
                }
            )

        spectra_df = pd.DataFrame(spectra_rows)
        print(f"Prepared {len(spectra_df)} biosample spectra rows.")
        return spectra_df

    def extract_spectrum_points(self) -> pd.DataFrame:
        """Explode the released spectra into one row per spectral point."""
        point_rows: list[dict] = []

        for spectrum_row in self._iter_all_rows():
            for point_index, (wavenumber, intensity) in enumerate(
                zip(self.calibrated_wavenumbers, spectrum_row["intensity_values"]),
                start=1,
            ):
                point_rows.append(
                    {
                        "biosample_id": spectrum_row["biosample_id"],
                        "dataset_id": self.dataset_id,
                        "source_row_id": spectrum_row["source_row_id"],
                        "point_index": point_index,
                        "wavenumber": float(wavenumber),
                        "intensity": float(intensity),
                    }
                )

        points_df = pd.DataFrame(point_rows)
        print(f"Prepared {len(points_df)} biosample spectrum point rows.")
        return points_df

    def extract_peaks(self) -> pd.DataFrame:
        """Detect conservative peaks because the released normalized probe files do not include peak lists."""
        peak_rows: list[dict] = []

        for spectrum_row in self._iter_all_rows():
            detected_peaks = self._detect_peaks_for_spectrum(spectrum_row["intensity_values"])
            for peak_row in detected_peaks:
                peak_rows.append(
                    {
                        "biosample_id": spectrum_row["biosample_id"],
                        "dataset_id": self.dataset_id,
                        "source_row_id": spectrum_row["source_row_id"],
                        **peak_row,
                    }
                )

        peaks_df = pd.DataFrame(peak_rows)
        print(f"Prepared {len(peaks_df)} biosample peak rows.")
        return peaks_df

    def _insert_dataframe(self, connection: duckdb.DuckDBPyConnection, table_name: str, df: pd.DataFrame) -> int:
        """Append a dataframe to DuckDB when rows are available."""
        if df.empty:
            return 0

        connection.register("temp_df", df)
        connection.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
        connection.unregister("temp_df")
        return int(len(df))

    def _build_chunk_tables(
        self,
        probe_name: str,
        class_label: str,
        matrix_chunk: np.ndarray,
        row_start_index: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Build spectra, point, and peak tables for one chunk of released rows."""
        spectra_rows: list[dict] = []
        point_rows: list[dict] = []
        peak_rows: list[dict] = []

        for offset, intensity_values in enumerate(matrix_chunk, start=0):
            row_index = row_start_index + offset
            identity = self._build_row_identity(probe_name, class_label, row_index)
            intensity_values = np.asarray(intensity_values, dtype=float)

            spectra_rows.append(
                {
                    "biosample_id": identity["biosample_id"],
                    "dataset_id": self.dataset_id,
                    "source_row_id": identity["source_row_id"],
                    "x_min": float(self.calibrated_wavenumbers[0]),
                    "x_max": float(self.calibrated_wavenumbers[-1]),
                    "n_points": int(len(self.calibrated_wavenumbers)),
                    "wavenumbers_json": json.dumps(self.calibrated_wavenumbers.tolist()),
                    "intensity_json": json.dumps([float(value) for value in intensity_values]),
                    "normalized_flag": "yes",
                    "preprocessing_summary": (
                        "Released normalized spectra from NormedProbe*.mat. "
                        "The notebooks clip negative values to zero before downstream modeling, "
                        "but the stored raw biosample_spectra rows preserve the released values."
                    ),
                }
            )

            for point_index, (wavenumber, intensity) in enumerate(
                zip(self.calibrated_wavenumbers, intensity_values),
                start=1,
            ):
                point_rows.append(
                    {
                        "biosample_id": identity["biosample_id"],
                        "dataset_id": self.dataset_id,
                        "source_row_id": identity["source_row_id"],
                        "point_index": point_index,
                        "wavenumber": float(wavenumber),
                        "intensity": float(intensity),
                    }
                )

            detected_peaks = self._detect_peaks_for_spectrum(intensity_values)
            for peak_row in detected_peaks:
                peak_rows.append(
                    {
                        "biosample_id": identity["biosample_id"],
                        "dataset_id": self.dataset_id,
                        "source_row_id": identity["source_row_id"],
                        **peak_row,
                    }
                )

        return pd.DataFrame(spectra_rows), pd.DataFrame(point_rows), pd.DataFrame(peak_rows)

    def ingest(self) -> None:
        """Ingest the locally available normalized probe rows into the biosample tables."""
        metadata_df = self.extract_metadata()

        metadata_count = 0
        spectra_count = 0
        point_count = 0
        peak_count = 0
        chunk_size = 250

        with duckdb.connect(str(self.db_path)) as connection:
            for table_name in (
                "biosample_metadata",
                "biosample_spectra",
                "biosample_spectrum_points",
                "biosample_peaks",
            ):
                connection.execute(f"DELETE FROM {table_name} WHERE dataset_id = ?", [self.dataset_id])

            metadata_count = self._insert_dataframe(connection, "biosample_metadata", metadata_df)

            for probe_path in self.probe_files:
                probe_name = probe_path.stem.lower()
                class_matrices = self._safe_load_probe_class_matrices(probe_path)
                if class_matrices is None:
                    continue

                for class_label, matrix in class_matrices.items():
                    for start_index in range(0, matrix.shape[0], chunk_size):
                        matrix_chunk = matrix[start_index : start_index + chunk_size]
                        spectra_df, points_df, peaks_df = self._build_chunk_tables(
                            probe_name=probe_name,
                            class_label=class_label,
                            matrix_chunk=matrix_chunk,
                            row_start_index=start_index + 1,
                        )
                        spectra_count += self._insert_dataframe(connection, "biosample_spectra", spectra_df)
                        point_count += self._insert_dataframe(connection, "biosample_spectrum_points", points_df)
                        peak_count += self._insert_dataframe(connection, "biosample_peaks", peaks_df)
                        print(
                            f"Ingested {probe_name} {class_label} rows "
                            f"{start_index + 1}-{start_index + len(matrix_chunk)}: "
                            f"{len(spectra_df)} spectra, {len(points_df)} points, {len(peaks_df)} peaks."
                        )

            fig3_norm_class_matrices = self._load_fig3_norm_class_matrices()
            for class_label, matrix in fig3_norm_class_matrices.items():
                for start_index in range(0, matrix.shape[0], chunk_size):
                    matrix_chunk = matrix[start_index : start_index + chunk_size]
                    spectra_df, points_df, peaks_df = self._build_chunk_tables(
                        probe_name="fig3_norm_archive",
                        class_label=class_label,
                        matrix_chunk=matrix_chunk,
                        row_start_index=start_index + 1,
                    )
                    spectra_count += self._insert_dataframe(connection, "biosample_spectra", spectra_df)
                    point_count += self._insert_dataframe(connection, "biosample_spectrum_points", points_df)
                    peak_count += self._insert_dataframe(connection, "biosample_peaks", peaks_df)
                    print(
                        f"Ingested fig3_norm_archive {class_label} rows "
                        f"{start_index + 1}-{start_index + len(matrix_chunk)}: "
                        f"{len(spectra_df)} spectra, {len(points_df)} points, {len(peaks_df)} peaks."
                    )

        print("small2023_ev ingestion complete.")
        print(f"Inserted biosample_metadata rows: {metadata_count}")
        print(f"Inserted biosample_spectra rows: {spectra_count}")
        print(f"Inserted biosample_spectrum_points rows: {point_count}")
        print(f"Inserted biosample_peaks rows: {peak_count}")
        if self.skipped_assets:
            print(f"Skipped assets: {len(self.skipped_assets)}")
            for asset_name, reason in self.skipped_assets[:10]:
                print(f"  {asset_name}: {reason}")
