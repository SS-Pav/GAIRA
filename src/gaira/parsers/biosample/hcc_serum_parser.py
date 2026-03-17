import io
import json
from pathlib import Path
from zipfile import ZipFile

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from gaira.parsers.biosample.base import BiosampleParserBase


class HCCSerumParser(BiosampleParserBase):
    """Parser for the Zenodo HCC serum SERS dataset."""

    ZIP_NAME = "dataset.zip"
    CSV_NAME = "data.csv"
    R_CODE_NAME = "R_code.R"
    EXPECTED_SPECTRA = 144
    METADATA_COLUMNS = ["acquisition_date", "substrate_batch", "class", "sample_code"]
    SUBCLASS_LABEL = "released_txt_archive"

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        super().__init__(dataset_id=dataset_id, dataset_root=dataset_root, db_path=db_path)
        self.archive_path = self.dataset_root / self.ZIP_NAME
        self.csv_path = self.dataset_root / self.CSV_NAME
        self.r_code_path = self.dataset_root / self.R_CODE_NAME
        self.metadata_df = self._load_metadata_table()
        self.archive_members = self._discover_archive_members()
        self.wavenumbers = self._load_wavenumbers_from_metadata()
        self.rows = self._build_rows()

    def _load_metadata_table(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing CSV metadata file: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        missing = [column for column in self.METADATA_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"{self.CSV_NAME} is missing required metadata columns: {missing}")

        df["acquisition_date"] = df["acquisition_date"].astype(str).str.strip()
        df["substrate_batch"] = df["substrate_batch"].astype(str).str.strip()
        df["class"] = df["class"].astype(str).str.strip()
        df["sample_code"] = pd.to_numeric(df["sample_code"], errors="raise").astype(int)
        return df

    def _discover_archive_members(self) -> list[str]:
        if not self.archive_path.exists():
            raise FileNotFoundError(f"Missing TXT archive file: {self.archive_path}")

        with ZipFile(self.archive_path, "r") as archive:
            txt_members = sorted(
                member
                for member in archive.namelist()
                if member.lower().endswith(".txt")
            )
        if len(txt_members) != self.EXPECTED_SPECTRA:
            raise ValueError(
                f"Expected {self.EXPECTED_SPECTRA} TXT spectra in {self.archive_path}, found {len(txt_members)}."
            )
        return txt_members

    def _load_wavenumbers_from_metadata(self) -> np.ndarray:
        axis_columns = [column for column in self.metadata_df.columns if column not in self.METADATA_COLUMNS]
        if not axis_columns:
            raise ValueError("No spectral axis columns were found in data.csv.")
        return np.asarray([float(column) for column in axis_columns], dtype=float)

    def _format_member_name(self, acquisition_date: str, substrate_batch: str, class_label: str, sample_code: int) -> str:
        return f"{acquisition_date}_batch-{substrate_batch}_{class_label}_{sample_code:03d}.txt"

    def _read_member_spectrum(self, archive: ZipFile, member_name: str) -> np.ndarray:
        spectrum_df = pd.read_csv(
            io.BytesIO(archive.read(member_name)),
            sep=r"\s+",
            header=None,
            names=["wavenumber", "intensity"],
        )

        if len(spectrum_df) != len(self.wavenumbers):
            raise ValueError(
                f"{member_name} contains {len(spectrum_df)} rows but expected {len(self.wavenumbers)}."
            )

        file_wavenumbers = spectrum_df["wavenumber"].to_numpy(dtype=float)
        if not np.allclose(file_wavenumbers, self.wavenumbers, atol=0.05):
            raise ValueError(f"{member_name} wavenumbers do not match the grounded CSV axis.")

        return spectrum_df["intensity"].to_numpy(dtype=float)

    def _build_rows(self) -> list[dict]:
        rows: list[dict] = []
        archive_member_set = set(self.archive_members)
        with ZipFile(self.archive_path, "r") as archive:
            for row_index, row in self.metadata_df.iterrows():
                member_name = self._format_member_name(
                    acquisition_date=row["acquisition_date"],
                    substrate_batch=row["substrate_batch"],
                    class_label=row["class"],
                    sample_code=int(row["sample_code"]),
                )
                if member_name not in archive_member_set:
                    raise FileNotFoundError(
                        f"Expected TXT member {member_name} from data.csv, but it was not found in {self.archive_path}."
                    )

                intensity_values = self._read_member_spectrum(archive, member_name)
                source_row_id = Path(member_name).stem
                biosample_id = f"{self.dataset_id}_{source_row_id}"
                rows.append(
                    {
                        "biosample_id": biosample_id,
                        "dataset_id": self.dataset_id,
                        "source_row_id": source_row_id,
                        "sample_id": f"{int(row['sample_code']):03d}",
                        "patient_id": None,
                        "replicate_id": source_row_id,
                        "class_label": row["class"],
                        "subclass_label": self.SUBCLASS_LABEL,
                        "source_file": f"{self.ZIP_NAME}::{member_name}",
                        "acquisition_date": row["acquisition_date"],
                        "substrate_batch": row["substrate_batch"],
                        "wavenumbers": self.wavenumbers,
                        "intensity_values": intensity_values,
                    }
                )

        if len(rows) != self.EXPECTED_SPECTRA:
            raise ValueError(f"Expected {self.EXPECTED_SPECTRA} rows, built {len(rows)}.")
        return rows

    def _base_notes(self, row: dict) -> str:
        return (
            f"Released serum SERS TXT spectrum for class {row['class_label']}. "
            f"Metadata fields acquisition_date={row['acquisition_date']}, substrate_batch={row['substrate_batch']}, "
            f"sample_code={row['sample_id']} were preserved from the released data.csv and matched back to the TXT "
            f"member name {Path(row['source_file']).name if '::' not in row['source_file'] else row['source_file'].split('::')[-1]}. "
            "Current GAIRA framing treats this as a binary HCC-versus-control serum SERS dataset on Ag plasmonic paper "
            "substrates, without inventing patient-level mappings beyond the released sample-level metadata."
        )

    def _preprocessing_summary(self) -> str:
        return (
            "Raw ingest preserves the native released TXT spectra and native axis. "
            "The release also includes R_code.R showing downstream crop/interpolate, modpolyfit baseline correction, "
            "and normalization steps, but those preprocessing operations are not applied during raw GAIRA ingestion."
        )

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
        print("hcc_serum dataset audit")
        print(f"Dataset root: {self.dataset_root}")
        print(f"TXT archive: {self.archive_path}")
        print(f"CSV metadata file: {self.csv_path}")
        print(f"R code file: {self.r_code_path}")
        print(f"TXT members: {len(self.archive_members)}")
        print(f"CSV rows: {len(self.metadata_df)}")
        print(f"Native axis: {self.wavenumbers[0]:.2f} to {self.wavenumbers[-1]:.2f} cm^-1 ({len(self.wavenumbers)} points)")
        print("Class counts:")
        print(self.metadata_df["class"].value_counts().sort_index().to_string())
        print("Substrate batches:")
        print(self.metadata_df["substrate_batch"].value_counts().sort_index().to_string())
        print("The release exposes a flat TXT archive plus a metadata-rich CSV. GAIRA uses the TXT archive as the")
        print("primary raw ingest path and validates the member names against data.csv.")

    def extract_metadata(self) -> pd.DataFrame:
        rows = []
        preprocessing_summary = self._preprocessing_summary()
        for row in self.rows:
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
                    "disease_context": "hepatocellular carcinoma",
                    "class_label": row["class_label"],
                    "subclass_label": row["subclass_label"],
                    "collection_protocol": None,
                    "preparation_protocol": None,
                    "instrument": "785 nm SERS system",
                    "laser_wavelength_nm": "785",
                    "spectral_range": f"{self.wavenumbers[0]:.2f}-{self.wavenumbers[-1]:.2f}",
                    "preprocessing_summary": preprocessing_summary,
                    "source_file": row["source_file"],
                    "notes": self._base_notes(row),
                }
            )

        metadata_df = pd.DataFrame(rows)
        print(f"Prepared {len(metadata_df)} biosample metadata rows.")
        return metadata_df

    def extract_spectra(self) -> pd.DataFrame:
        rows = []
        preprocessing_summary = self._preprocessing_summary()
        for row in self.rows:
            rows.append(
                {
                    "biosample_id": row["biosample_id"],
                    "dataset_id": self.dataset_id,
                    "source_row_id": row["source_row_id"],
                    "x_min": float(self.wavenumbers[0]),
                    "x_max": float(self.wavenumbers[-1]),
                    "n_points": int(len(self.wavenumbers)),
                    "wavenumbers_json": json.dumps([float(value) for value in self.wavenumbers]),
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
                zip(self.wavenumbers, row["intensity_values"]),
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
            for peak_row in self._detect_peaks_for_spectrum(row["intensity_values"]):
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

        print("hcc_serum ingestion complete.")
        print(f"Inserted biosample_metadata rows: {metadata_count}")
        print(f"Inserted biosample_spectra rows: {spectra_count}")
        print(f"Inserted biosample_spectrum_points rows: {point_count}")
        print(f"Inserted biosample_peaks rows: {peak_count}")
