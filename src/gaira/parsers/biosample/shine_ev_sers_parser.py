from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from gaira.parsers.biosample.base import BiosampleParserBase


class ShineEVSERSParser(BiosampleParserBase):
    """Concrete parser for the SHINE EV hepatotoxicity SERS dataset."""

    # Calibration values are taken directly from Figure4/Fig4D/code/Fig4D.m.
    PIXELS = np.array([263, 367, 492, 512, 590, 782, 872, 887], dtype=float)
    CALIBRATED_CM = np.array([620.9, 795.8, 1001.4, 1031.8, 1155.3, 1450.5, 1583.1, 1602.3], dtype=float)
    EXPECTED_POINT_COUNT = 1650

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        super().__init__(dataset_id=dataset_id, dataset_root=dataset_root, db_path=db_path)
        self.project_root = self._detect_project_root()
        self.source_root = (
            self.dataset_root / "SERS-Hepatotoxicity_DATA_CODE_FIGURE" / "Figure4" / "data"
        )
        self.calibrated_wavenumbers = self._build_calibrated_wavenumbers()
        self.skipped_files: list[tuple[str, str]] = []

    def _detect_project_root(self) -> Path:
        """Find the GAIRA project root from the parser file location."""
        return Path(__file__).resolve().parents[4]

    def _build_calibrated_wavenumbers(self) -> np.ndarray:
        """Recreate the Raman shift axis from the supplied MATLAB calibration code."""
        coefficients = np.polyfit(self.PIXELS, self.CALIBRATED_CM, 3)
        return np.polyval(coefficients, np.arange(1, self.EXPECTED_POINT_COUNT + 1))

    def _discover_spectrum_files(self) -> list[Path]:
        """Find the real per-spectrum text files under Figure4/data."""
        if not self.source_root.exists():
            raise FileNotFoundError(
                f"Expected dataset data folder not found: {self.source_root}"
            )

        spectrum_files = sorted(
            path
            for path in self.source_root.rglob("s_*")
            if path.is_file() and path.name.startswith("s_")
        )
        if not spectrum_files:
            raise FileNotFoundError(
                f"No per-spectrum files named s_* were found under {self.source_root}"
            )

        return spectrum_files

    def _parse_path_metadata(self, spectrum_path: Path) -> dict:
        """Extract labels grounded in the folder names only."""
        relative_path = spectrum_path.relative_to(self.dataset_root)
        data_relative_parts = spectrum_path.relative_to(self.source_root).parts

        # The dataset uses two layouts:
        # 1. .../Set9/D0_C0/001_2/s_1
        # 2. .../Set9/D1_C0/s_463
        if len(data_relative_parts) == 3:
            set_name, class_label, replicate_id = data_relative_parts
            sample_id = class_label
        else:
            set_name, class_label, sample_id, replicate_id = data_relative_parts[-4:]
        source_row_id = relative_path.as_posix().replace("/", "__")
        biosample_id = f"{self.dataset_id}_{source_row_id}"

        return {
            "biosample_id": biosample_id,
            "source_row_id": source_row_id,
            "sample_id": sample_id,
            "replicate_id": replicate_id,
            "class_label": class_label,
            "subclass_label": set_name,
            "source_file": relative_path.as_posix(),
        }

    def _read_spectrum_file(self, spectrum_path: Path) -> pd.DataFrame | None:
        """Read one raw spectrum file and validate the expected structure."""
        try:
            spectrum_df = pd.read_csv(
                spectrum_path,
                header=None,
                names=["pixel_index", "intensity"],
            )
        except Exception as exc:
            self.skipped_files.append((str(spectrum_path), f"read_error: {exc}"))
            return None

        spectrum_df["pixel_index"] = pd.to_numeric(spectrum_df["pixel_index"], errors="coerce")
        spectrum_df["intensity"] = pd.to_numeric(spectrum_df["intensity"], errors="coerce")
        spectrum_df = spectrum_df.dropna(subset=["pixel_index", "intensity"]).reset_index(drop=True)

        if len(spectrum_df) != self.EXPECTED_POINT_COUNT:
            self.skipped_files.append(
                (str(spectrum_path), f"expected {self.EXPECTED_POINT_COUNT} rows but found {len(spectrum_df)}")
            )
            return None

        expected_pixels = np.arange(1, self.EXPECTED_POINT_COUNT + 1, dtype=float)
        if not np.array_equal(spectrum_df["pixel_index"].to_numpy(dtype=float), expected_pixels):
            self.skipped_files.append((str(spectrum_path), "pixel index sequence did not match 1..1650"))
            return None

        spectrum_df["wavenumber"] = self.calibrated_wavenumbers
        return spectrum_df

    def _build_metadata_df(self, file_paths: list[Path]) -> pd.DataFrame:
        """Build biosample metadata rows from file paths and grounded source notes."""
        rows: list[dict] = []
        for spectrum_path in file_paths:
            path_info = self._parse_path_metadata(spectrum_path)
            rows.append(
                {
                    "biosample_id": path_info["biosample_id"],
                    "dataset_id": self.dataset_id,
                    "source_row_id": path_info["source_row_id"],
                    "sample_id": path_info["sample_id"],
                    "patient_id": None,
                    "replicate_id": path_info["replicate_id"],
                    "biosample_type": "extracellular vesicles",
                    "matrix": None,
                    "disease_context": None,
                    "class_label": path_info["class_label"],
                    "subclass_label": path_info["subclass_label"],
                    "collection_protocol": None,
                    "preparation_protocol": None,
                    "instrument": None,
                    "laser_wavelength_nm": None,
                    "spectral_range": f"{self.calibrated_wavenumbers.min():.1f}-{self.calibrated_wavenumbers.max():.1f}",
                    "preprocessing_summary": (
                        "Per-spectrum text file ingested directly. Raman shift axis calibrated "
                        "using the polynomial defined in Figure4/Fig4D/code/Fig4D.m."
                    ),
                    "source_file": path_info["source_file"],
                    "notes": (
                        "class_label and subclass_label were derived from folder names only. "
                        "Dataset-level APAP hepatotoxicity context is known from the registry, "
                        "but no row-level disease label was explicitly present in the raw file."
                    ),
                }
            )

        return pd.DataFrame(rows)

    def _build_chunk_tables(self, file_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Read one chunk of spectra and build spectra, point, and peak tables."""
        spectra_rows: list[dict] = []
        point_rows: list[dict] = []
        peak_rows: list[dict] = []

        for spectrum_path in file_paths:
            path_info = self._parse_path_metadata(spectrum_path)
            spectrum_df = self._read_spectrum_file(spectrum_path)
            if spectrum_df is None:
                continue

            intensity_values = spectrum_df["intensity"].to_numpy(dtype=float)
            wavenumber_values = spectrum_df["wavenumber"].to_numpy(dtype=float)

            spectra_rows.append(
                {
                    "biosample_id": path_info["biosample_id"],
                    "dataset_id": self.dataset_id,
                    "source_row_id": path_info["source_row_id"],
                    "x_min": float(wavenumber_values.min()),
                    "x_max": float(wavenumber_values.max()),
                    "n_points": int(len(wavenumber_values)),
                    "wavenumbers_json": pd.Series(wavenumber_values).to_json(orient="values"),
                    "intensity_json": pd.Series(intensity_values).to_json(orient="values"),
                    "normalized_flag": "unknown",
                    "preprocessing_summary": (
                        "Per-spectrum text file ingested directly. Raman shift axis calibrated "
                        "using Figure4/Fig4D/code/Fig4D.m."
                    ),
                }
            )

            for point_index, (wavenumber, intensity) in enumerate(
                zip(wavenumber_values, intensity_values),
                start=1,
            ):
                point_rows.append(
                    {
                        "biosample_id": path_info["biosample_id"],
                        "dataset_id": self.dataset_id,
                        "source_row_id": path_info["source_row_id"],
                        "point_index": point_index,
                        "wavenumber": float(wavenumber),
                        "intensity": float(intensity),
                    }
                )

            normalized_intensity = intensity_values - intensity_values.min()
            intensity_range = normalized_intensity.max()
            if intensity_range > 0:
                normalized_intensity = normalized_intensity / intensity_range
                peak_indices, properties = find_peaks(
                    normalized_intensity,
                    prominence=0.05,
                    height=0.05,
                    distance=5,
                )
                for peak_rank, peak_index in enumerate(peak_indices, start=1):
                    peak_rows.append(
                        {
                            "biosample_id": path_info["biosample_id"],
                            "dataset_id": self.dataset_id,
                            "source_row_id": path_info["source_row_id"],
                            "peak_rank": peak_rank,
                            "peak_cm": float(wavenumber_values[peak_index]),
                            "peak_intensity": float(properties["peak_heights"][peak_rank - 1]),
                            "prominence": float(properties["prominences"][peak_rank - 1]),
                        }
                    )

        return (
            pd.DataFrame(spectra_rows),
            pd.DataFrame(point_rows),
            pd.DataFrame(peak_rows),
        )

    def audit(self) -> None:
        """Inspect the real dataset structure and report the grounded file layout."""
        spectrum_files = self._discover_spectrum_files()
        condition_counts: dict[str, int] = {}
        for spectrum_path in spectrum_files:
            condition_label = self._parse_path_metadata(spectrum_path)["class_label"]
            condition_counts[condition_label] = condition_counts.get(condition_label, 0) + 1

        print("SHINE EV SERS dataset audit")
        print(f"Dataset root: {self.dataset_root}")
        print(f"Usable spectrum files found: {len(spectrum_files)}")
        print("Main data folders:")
        print(f"  {self.source_root / 'Set9'}")
        print(f"  {self.source_root / 'Set10'}")
        print("Additional source files:")
        print(f"  {self.source_root / 'RawDataSet91.mat'}")
        print(f"  {self.source_root / 'RawDataset119.mat'}")
        print("Condition folder counts:")
        for condition_label, count in sorted(condition_counts.items()):
            print(f"  {condition_label}: {count}")
        print(
            "Per-spectrum files are two-column text files with 1,650 rows. "
            "The first column is a pixel index and the Raman shift axis is calibrated "
            "from Figure4/Fig4D/code/Fig4D.m."
        )

    def extract_metadata(self) -> pd.DataFrame:
        """Build biosample metadata rows from the real file paths."""
        spectrum_files = self._discover_spectrum_files()
        metadata_df = self._build_metadata_df(spectrum_files)
        print(f"Prepared {len(metadata_df)} biosample metadata rows.")
        return metadata_df

    def extract_spectra(self, file_paths: list[Path] | None = None) -> pd.DataFrame:
        """Build full-spectrum rows with JSON arrays for the given files."""
        selected_files = file_paths or self._discover_spectrum_files()
        spectra_df, _, _ = self._build_chunk_tables(selected_files)
        print(f"Prepared {len(spectra_df)} biosample spectra rows.")
        return spectra_df

    def extract_spectrum_points(self, file_paths: list[Path] | None = None) -> pd.DataFrame:
        """Explode the given spectra into one row per spectral point."""
        selected_files = file_paths or self._discover_spectrum_files()
        _, points_df, _ = self._build_chunk_tables(selected_files)
        print(f"Prepared {len(points_df)} biosample spectrum point rows.")
        return points_df

    def extract_peaks(self, file_paths: list[Path] | None = None) -> pd.DataFrame:
        """Detect conservative peaks from each spectrum because no peak list was supplied."""
        selected_files = file_paths or self._discover_spectrum_files()
        _, _, peaks_df = self._build_chunk_tables(selected_files)
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

    def ingest(self) -> None:
        """Run the real SHINE EV SERS ingestion into the biosample tables."""
        spectrum_files = self._discover_spectrum_files()
        metadata_df = self._build_metadata_df(spectrum_files)

        metadata_count = 0
        spectra_count = 0
        point_count = 0
        peak_count = 0
        chunk_size = 50

        with duckdb.connect(str(self.db_path)) as connection:
            for table_name in (
                "biosample_metadata",
                "biosample_spectra",
                "biosample_spectrum_points",
                "biosample_peaks",
            ):
                connection.execute(f"DELETE FROM {table_name} WHERE dataset_id = ?", [self.dataset_id])

            metadata_count = self._insert_dataframe(connection, "biosample_metadata", metadata_df)

            for start_index in range(0, len(spectrum_files), chunk_size):
                chunk_files = spectrum_files[start_index : start_index + chunk_size]
                spectra_df, points_df, peaks_df = self._build_chunk_tables(chunk_files)
                spectra_count += self._insert_dataframe(connection, "biosample_spectra", spectra_df)
                point_count += self._insert_dataframe(connection, "biosample_spectrum_points", points_df)
                peak_count += self._insert_dataframe(connection, "biosample_peaks", peaks_df)
                print(
                    f"Ingested chunk {start_index // chunk_size + 1}: "
                    f"{len(spectra_df)} spectra, {len(points_df)} points, {len(peaks_df)} peaks."
                )

        print("SHINE EV SERS ingestion complete.")
        print(f"Inserted biosample_metadata rows: {metadata_count}")
        print(f"Inserted biosample_spectra rows: {spectra_count}")
        print(f"Inserted biosample_spectrum_points rows: {point_count}")
        print(f"Inserted biosample_peaks rows: {peak_count}")
        print(f"Skipped files: {len(self.skipped_files)}")
        if self.skipped_files:
            print("First skipped examples:")
            for file_name, reason in self.skipped_files[:10]:
                print(f"  {file_name}: {reason}")
