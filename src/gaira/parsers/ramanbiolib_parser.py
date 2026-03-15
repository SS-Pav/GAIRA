import ast
import json
from pathlib import Path

import duckdb
import pandas as pd

from gaira.parsers.base import DatasetParser


class RamanBioLibParser(DatasetParser):
    """Parser for RamanBioLib reference metadata, peaks, and spectra."""

    METADATA_EXTENSIONS = {".csv", ".xlsx", ".json", ".yaml"}
    SPECTRA_EXTENSIONS = {".csv", ".txt", ".tsv", ".mat", ".npy"}
    CODE_DOC_EXTENSIONS = {".py", ".ipynb", ".md", ".pdf"}

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        super().__init__(dataset_id=dataset_id, dataset_root=dataset_root, db_path=db_path)

    def _detect_project_root(self) -> Path:
        """Choose the most likely extracted repository root for auditing."""
        if not self.dataset_root.exists():
            print(f"Dataset folder does not exist yet: {self.dataset_root}")
            return self.dataset_root

        child_directories = [path for path in self.dataset_root.iterdir() if path.is_dir()]
        if not child_directories:
            return self.dataset_root

        best_root = self.dataset_root
        best_score = -1

        for candidate in child_directories:
            score = len([path for path in candidate.rglob("*") if path.is_file()])
            if score > best_score:
                best_root = candidate
                best_score = score

        return best_root

    def _locate_db_files(self) -> dict[str, Path]:
        """Locate the three RamanBioLib database CSV files inside the repo."""
        project_root = self._detect_project_root()
        expected_files = {
            "metadata": "metadata_db.csv",
            "peaks": "raman_peaks_db.csv",
            "spectra": "raman_spectra_db.csv",
        }
        located_files: dict[str, Path] = {}

        for key, filename in expected_files.items():
            matches = [path for path in project_root.rglob(filename) if path.is_file()]
            clean_matches = [path for path in matches if not path.name.startswith("._")]

            if not clean_matches:
                raise FileNotFoundError(
                    f"Could not find {filename} under {project_root}. Check the downloaded RamanBioLib repo."
                )

            located_files[key] = clean_matches[0]

        print("Located RamanBioLib CSV files:")
        for key, path in located_files.items():
            print(f"  {key}: {path}")

        return located_files

    def _parse_list_string(self, value: str, row_label: str, field_name: str) -> list[float] | None:
        """Safely parse a Python-style list stored as text in the CSV."""
        if pd.isna(value):
            print(f"Skipping {row_label}: '{field_name}' is missing.")
            return None

        try:
            parsed = ast.literal_eval(str(value))
        except (SyntaxError, ValueError) as exc:
            print(f"Skipping {row_label}: could not parse '{field_name}' as a list ({exc}).")
            return None

        if not isinstance(parsed, list):
            print(f"Skipping {row_label}: '{field_name}' is not stored as a list.")
            return None

        return parsed

    def _validate_equal_lengths(
        self,
        left_values: list[float] | None,
        right_values: list[float] | None,
        row_label: str,
        left_name: str,
        right_name: str,
    ) -> bool:
        """Check that two parsed arrays exist and have matching lengths."""
        if left_values is None or right_values is None:
            return False

        if len(left_values) != len(right_values):
            print(
                f"Skipping {row_label}: {left_name} has {len(left_values)} values but {right_name} has {len(right_values)}."
            )
            return False

        return True

    def _read_csv(self, file_key: str) -> pd.DataFrame:
        """Read one of the RamanBioLib CSV database files."""
        csv_path = self._locate_db_files()[file_key]
        return pd.read_csv(csv_path)

    def _normalize_text_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Convert pandas missing values to Python None for cleaner DB inserts."""
        cleaned_df = dataframe.copy()
        cleaned_df = cleaned_df.where(pd.notna(cleaned_df), None)
        return cleaned_df

    def _build_ref_id(self, source_row_id: int) -> str:
        """Create a stable GAIRA reference identifier for one RamanBioLib row."""
        return f"{self.dataset_id}_{source_row_id}"

    def _list_dataset_files(self) -> tuple[Path, list[Path]]:
        """Collect all files below the likely project root."""
        project_root = self._detect_project_root()
        if not project_root.exists():
            return project_root, []

        all_files = sorted(path for path in project_root.rglob("*") if path.is_file())
        return project_root, all_files

    def _find_candidates(self, extensions: set[str]) -> tuple[Path, list[Path]]:
        """Find files whose extension matches the requested set."""
        project_root, all_files = self._list_dataset_files()
        return project_root, [path for path in all_files if path.suffix.lower() in extensions]

    def audit(self) -> None:
        """Inspect the dataset folder and report what is present."""
        print(f"Auditing dataset folder: {self.dataset_root}")
        project_root, all_files = self._list_dataset_files()
        print(f"Likely project root: {project_root}")

        if not all_files:
            print("No files were found. Download the dataset files first.")
            return

        extension_counts: dict[str, int] = {}
        for file_path in all_files:
            extension = file_path.suffix.lower() or "[no extension]"
            extension_counts[extension] = extension_counts.get(extension, 0) + 1

        _, metadata_files = self._find_candidates(self.METADATA_EXTENSIONS)
        _, spectra_files = self._find_candidates(self.SPECTRA_EXTENSIONS)
        _, code_doc_files = self._find_candidates(self.CODE_DOC_EXTENSIONS)

        print(f"Total files found: {len(all_files)}")
        print("File extensions present:")
        for extension, count in sorted(extension_counts.items()):
            print(f"  {extension}: {count}")

        self._print_candidate_group("Candidate metadata files", metadata_files)
        self._print_candidate_group("Candidate spectra files", spectra_files)
        self._print_candidate_group("Candidate notebooks/scripts/docs", code_doc_files)

        if code_doc_files and not metadata_files and not spectra_files:
            print(
                "This extracted repository currently looks documentation or code heavy, with no obvious spectral tables yet."
            )

    def extract_metadata(self) -> None:
        """Read and map RamanBioLib metadata into the reference_metadata schema."""
        print("Reading RamanBioLib metadata table.")
        metadata_df = self._read_csv("metadata")

        metadata_df.columns = [column.strip().lower() for column in metadata_df.columns]
        metadata_df = metadata_df.rename(
            columns={
                "id": "source_row_id",
                "type": "biochemical_class",
                "peak_identificaton": "peak_identification",
                "laser_wavelength": "laser_wavelength_nm",
            }
        )

        metadata_df["dataset_id"] = self.dataset_id
        metadata_df["source_row_id"] = metadata_df["source_row_id"].astype(int)
        metadata_df["ref_id"] = metadata_df["source_row_id"].apply(self._build_ref_id)

        ordered_columns = [
            "ref_id",
            "dataset_id",
            "source_row_id",
            "component",
            "biochemical_class",
            "submission_date",
            "contact",
            "source",
            "reference",
            "extraction_method",
            "peak_identification",
            "interpolation_method",
            "extra_preprocessing",
            "complete_sample_name",
            "sample_source",
            "sample_composition",
            "sample_preparation",
            "sample_substrate",
            "raman_technique",
            "raman_system",
            "delivery_optics",
            "laser_wavelength_nm",
            "laser_power",
            "acquisition_time",
            "orig_spectral_range",
            "orig_spectral_resolution",
            "orig_spatial_resolution",
            "detector",
            "calibration",
            "cropping",
            "spike_removal",
            "denoising",
            "background_removal",
            "baseline_removal",
            "normalization",
            "additional_info",
        ]
        metadata_df = self._normalize_text_columns(metadata_df[ordered_columns])

        print(f"Prepared {len(metadata_df)} metadata rows for insertion.")
        print(f"Unique components in metadata: {metadata_df['component'].nunique()}")
        return metadata_df

    def extract_peaks(self) -> pd.DataFrame:
        """Read RamanBioLib peak annotations and explode them into one row per peak."""
        print("Reading RamanBioLib peak table.")
        peaks_df = self._read_csv("peaks")
        peaks_df.columns = [column.strip().lower() for column in peaks_df.columns]
        peaks_df = peaks_df.rename(columns={"id": "source_row_id"})
        peaks_df["source_row_id"] = peaks_df["source_row_id"].astype(int)

        peak_rows: list[dict] = []
        valid_rows = 0
        skipped_rows = 0

        for row in peaks_df.itertuples(index=False):
            row_label = f"peak row {row.source_row_id} ({row.component})"
            peaks_list = self._parse_list_string(row.peaks, row_label, "peaks")
            intensity_list = self._parse_list_string(row.intensity, row_label, "intensity")

            if not self._validate_equal_lengths(
                peaks_list,
                intensity_list,
                row_label,
                "peaks",
                "intensity",
            ):
                skipped_rows += 1
                continue

            valid_rows += 1
            ref_id = self._build_ref_id(int(row.source_row_id))
            for peak_rank, (peak_cm, rel_intensity) in enumerate(
                zip(peaks_list, intensity_list),
                start=1,
            ):
                peak_rows.append(
                    {
                        "ref_id": ref_id,
                        "dataset_id": self.dataset_id,
                        "component": row.component,
                        "source_row_id": int(row.source_row_id),
                        "peak_rank": peak_rank,
                        "peak_cm": float(peak_cm),
                        "rel_intensity": float(rel_intensity),
                    }
                )

        peak_rows_df = pd.DataFrame(
            peak_rows,
            columns=[
                "ref_id",
                "dataset_id",
                "component",
                "source_row_id",
                "peak_rank",
                "peak_cm",
                "rel_intensity",
            ],
        )
        print(f"Peak validation summary: {valid_rows} valid rows, {skipped_rows} skipped rows.")
        print(f"Prepared {len(peak_rows_df)} exploded peak rows for insertion.")
        return peak_rows_df

    def extract_spectra(self) -> pd.DataFrame:
        """Read RamanBioLib spectra and store whole arrays as JSON text."""
        print("Reading RamanBioLib spectra table.")
        spectra_df = self._read_csv("spectra")
        spectra_df.columns = [column.strip().lower() for column in spectra_df.columns]
        spectra_df = spectra_df.rename(columns={"id": "source_row_id"})
        spectra_df["source_row_id"] = spectra_df["source_row_id"].astype(int)

        spectrum_rows: list[dict] = []
        valid_rows = 0
        skipped_rows = 0

        for row in spectra_df.itertuples(index=False):
            row_label = f"spectrum row {row.source_row_id} ({row.component})"
            wavenumbers = self._parse_list_string(row.wavenumbers, row_label, "wavenumbers")
            intensity = self._parse_list_string(row.intensity, row_label, "intensity")

            if not self._validate_equal_lengths(
                wavenumbers,
                intensity,
                row_label,
                "wavenumbers",
                "intensity",
            ):
                skipped_rows += 1
                continue

            wavenumbers = [float(value) for value in wavenumbers]
            intensity = [float(value) for value in intensity]
            valid_rows += 1

            spectrum_rows.append(
                {
                    "ref_id": self._build_ref_id(int(row.source_row_id)),
                    "dataset_id": self.dataset_id,
                    "component": row.component,
                    "source_row_id": int(row.source_row_id),
                    "x_min": min(wavenumbers),
                    "x_max": max(wavenumbers),
                    "n_points": len(wavenumbers),
                    "wavenumbers_json": json.dumps(wavenumbers),
                    "intensity_json": json.dumps(intensity),
                    "normalized_flag": "yes",
                    "preprocessing_summary": "Imported from RamanBioLib reference spectra database",
                }
            )

        spectra_rows_df = pd.DataFrame(
            spectrum_rows,
            columns=[
                "ref_id",
                "dataset_id",
                "component",
                "source_row_id",
                "x_min",
                "x_max",
                "n_points",
                "wavenumbers_json",
                "intensity_json",
                "normalized_flag",
                "preprocessing_summary",
            ],
        )
        print(f"Spectra validation summary: {valid_rows} valid rows, {skipped_rows} skipped rows.")
        print(f"Prepared {len(spectra_rows_df)} spectra rows for insertion.")
        return spectra_rows_df

    def extract_spectrum_points(self) -> pd.DataFrame:
        """Explode RamanBioLib spectra into one row per wavenumber-intensity pair."""
        print("Expanding RamanBioLib spectra into individual spectrum points.")
        spectra_df = self._read_csv("spectra")
        spectra_df.columns = [column.strip().lower() for column in spectra_df.columns]
        spectra_df = spectra_df.rename(columns={"id": "source_row_id"})
        spectra_df["source_row_id"] = spectra_df["source_row_id"].astype(int)

        point_rows: list[dict] = []
        valid_rows = 0
        skipped_rows = 0

        for row in spectra_df.itertuples(index=False):
            row_label = f"spectrum row {row.source_row_id} ({row.component})"
            wavenumbers = self._parse_list_string(row.wavenumbers, row_label, "wavenumbers")
            intensity = self._parse_list_string(row.intensity, row_label, "intensity")

            if not self._validate_equal_lengths(
                wavenumbers,
                intensity,
                row_label,
                "wavenumbers",
                "intensity",
            ):
                skipped_rows += 1
                continue

            valid_rows += 1
            ref_id = self._build_ref_id(int(row.source_row_id))
            for point_index, (wavenumber, point_intensity) in enumerate(
                zip(wavenumbers, intensity)
            ):
                point_rows.append(
                    {
                        "ref_id": ref_id,
                        "dataset_id": self.dataset_id,
                        "component": row.component,
                        "source_row_id": int(row.source_row_id),
                        "point_index": point_index,
                        "wavenumber": float(wavenumber),
                        "intensity": float(point_intensity),
                    }
                )

        point_rows_df = pd.DataFrame(
            point_rows,
            columns=[
                "ref_id",
                "dataset_id",
                "component",
                "source_row_id",
                "point_index",
                "wavenumber",
                "intensity",
            ],
        )
        print(
            f"Spectrum point validation summary: {valid_rows} valid rows, {skipped_rows} skipped rows."
        )
        print(f"Total exploded spectrum point rows created: {len(point_rows_df)}")
        return point_rows_df

    def ingest(self) -> None:
        """Insert the RamanBioLib reference tables into DuckDB."""
        print(f"Starting real ingestion for dataset: {self.dataset_id}")
        self.audit()
        metadata_df = self.extract_metadata()
        peaks_df = self.extract_peaks()
        spectra_df = self.extract_spectra()
        spectrum_points_df = self.extract_spectrum_points()

        try:
            with duckdb.connect(str(self.db_path)) as connection:
                for table_name in (
                    "reference_metadata",
                    "reference_peaks",
                    "reference_spectra",
                    "reference_spectrum_points",
                ):
                    connection.execute(
                        f"DELETE FROM {table_name} WHERE dataset_id = ?",
                        [self.dataset_id],
                    )

                connection.register("metadata_df", metadata_df)
                connection.register("peaks_df", peaks_df)
                connection.register("spectra_df", spectra_df)
                connection.register("spectrum_points_df", spectrum_points_df)

                connection.execute("INSERT INTO reference_metadata SELECT * FROM metadata_df")
                connection.execute("INSERT INTO reference_peaks SELECT * FROM peaks_df")
                connection.execute("INSERT INTO reference_spectra SELECT * FROM spectra_df")
                connection.execute(
                    "INSERT INTO reference_spectrum_points SELECT * FROM spectrum_points_df"
                )
        except duckdb.Error as exc:
            print("Could not write RamanBioLib data into DuckDB.")
            print("Please run `python scripts/init_database.py` after releasing any lock on data/gaira.duckdb.")
            print(f"DuckDB error: {exc}")
            return

        unique_components = metadata_df["component"].nunique() if not metadata_df.empty else 0
        print(f"Inserted {len(metadata_df)} metadata rows.")
        print(f"Inserted {len(peaks_df)} peak rows.")
        print(f"Inserted {len(spectra_df)} spectra rows.")
        print(f"Inserted {len(spectrum_points_df)} spectrum point rows.")
        print(f"Unique components ingested: {unique_components}")
        print(f"RamanBioLib ingestion complete. Database path: {self.db_path}")

    def _print_candidate_group(
        self,
        label: str,
        files: list[Path],
        use_full_paths: bool = False,
    ) -> None:
        """Print a short, readable list of candidate files."""
        print(f"{label}:")
        if not files:
            print("  None")
            return

        project_root = self._detect_project_root()
        for file_path in files:
            if use_full_paths:
                print(f"  {file_path}")
            else:
                print(f"  {file_path.relative_to(project_root)}")
