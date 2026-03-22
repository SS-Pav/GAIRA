import json
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from gaira.parsers.biosample.base import BiosampleParserBase


class CSPPSerumParser(BiosampleParserBase):
    """Parser for Zenodo 5644790 CSPP serum methodology archive."""

    FIGURE_CONFIGS = {
        "Figure-2_all-spectra-and-metadata.csv": {
            "subclass_label": "figure2_processing_comparison",
            "class_column": "processing",
            "sample_column": "ID",
            "replicate_column": "rep",
            "notes": (
                "Figure 2 serum processing-comparison family. The released metadata table compares unprocessed, "
                "filtration, and standard processing conditions."
            ),
        },
        "Figure-4_all-spectra-and-metadata.csv": {
            "subclass_label": "figure4_protocol_optimization",
            "class_column": "run",
            "sample_column": "run",
            "replicate_column": None,
            "notes": (
                "Figure 4 serum protocol-optimization family. The released metadata table stores experimental "
                "run identifiers plus method factors such as Tasc, Vcol, Tinc, Tcent, Rpm, Tatt, and response Y."
            ),
        },
        "Figure-5_all-spectra-and-metadata.csv": {
            "subclass_label": "figure5_strip_variability",
            "class_column": "strip",
            "sample_column": "strip",
            "replicate_column": "rep",
            "notes": (
                "Figure 5 serum strip-variability family. The released metadata table stores strip and replicate "
                "identifiers plus a released ratio field."
            ),
        },
        "Figure-6_all-spectra-and-metadata.csv": {
            "subclass_label": "figure6_shelf_life",
            "class_column": "shelf_life",
            "sample_column": "ID2",
            "replicate_column": "strip_rep",
            "notes": (
                "Figure 6 serum shelf-life family. The released metadata table stores shelf_life, strip_rep, "
                "date, hours, and derived ID2 identifiers."
            ),
        },
        "Figure-7_all-spectra-and-metadata.csv": {
            "subclass_label": "figure7_metabolite_spiking",
            "class_column": "metabolite",
            "sample_column": "num",
            "replicate_column": "rep",
            "notes": (
                "Figure 7 serum metabolite-spiking family. The released metadata table stores metabolite, "
                "concentration, replicate, and acquisition settings."
            ),
        },
    }

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        super().__init__(dataset_id=dataset_id, dataset_root=dataset_root, db_path=db_path)
        self.rows = self._build_rows()

    def _load_figure_dataframe(self, csv_name: str) -> tuple[pd.DataFrame, list[str], np.ndarray]:
        csv_path = self.dataset_root / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing figure CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        axis_columns = []
        for column in df.columns:
            try:
                float(column)
            except (TypeError, ValueError):
                continue
            axis_columns.append(column)

        if not axis_columns:
            raise ValueError(f"{csv_name} does not expose numeric spectral axis columns.")

        wavenumbers = np.asarray([float(column) for column in axis_columns], dtype=float)
        return df, axis_columns, wavenumbers

    def _build_source_row_id(self, csv_name: str, source_row_value: object) -> str:
        figure_slug = csv_name.replace("Figure-", "fig").replace("_all-spectra-and-metadata.csv", "")
        return f"{figure_slug}_row_{int(source_row_value):03d}"

    def _build_notes(self, base_note: str, metadata_row: pd.Series, metadata_columns: list[str]) -> str:
        preserved = ", ".join(f"{column}={metadata_row[column]}" for column in metadata_columns)
        return (
            f"{base_note} Current GAIRA framing treats cspp_serum as a serum methodology archive on centrifugal "
            f"silver plasmonic paper rather than a disease benchmark. Preserved released metadata: {preserved}."
        )

    def _build_rows(self) -> list[dict]:
        rows: list[dict] = []
        reference_axis: np.ndarray | None = None
        for csv_name, config in self.FIGURE_CONFIGS.items():
            figure_df, spectral_columns, wavenumbers = self._load_figure_dataframe(csv_name)
            if reference_axis is None:
                reference_axis = wavenumbers
            elif not np.allclose(reference_axis, wavenumbers, atol=0.1):
                raise ValueError(f"{csv_name} does not match the grounded shared CSPP serum axis.")

            metadata_columns = [column for column in figure_df.columns if column not in spectral_columns]
            for _, row in figure_df.iterrows():
                source_row_value = row["Unnamed: 0"] if "Unnamed: 0" in row else row.iloc[0]
                source_row_id = self._build_source_row_id(csv_name, source_row_value)
                biosample_id = f"{self.dataset_id}_{source_row_id}"
                intensity_values = row[spectral_columns].to_numpy(dtype=float)

                sample_value = row[config["sample_column"]]
                replicate_value = row[config["replicate_column"]] if config["replicate_column"] else None
                rows.append(
                    {
                        "biosample_id": biosample_id,
                        "dataset_id": self.dataset_id,
                        "source_row_id": source_row_id,
                        "sample_id": str(sample_value),
                        "patient_id": None,
                        "replicate_id": str(replicate_value) if replicate_value is not None else None,
                        "class_label": str(row[config["class_column"]]),
                        "subclass_label": config["subclass_label"],
                        "source_file": f"{csv_name}::{source_row_id}",
                        "wavenumbers": wavenumbers,
                        "intensity_values": intensity_values,
                        "row_note": self._build_notes(config["notes"], row, metadata_columns),
                    }
                )
        if not rows:
            raise ValueError("No rows were built for cspp_serum.")
        return rows

    def _preprocessing_summary(self) -> str:
        return (
            "Raw ingest preserves the released figure-level CSV spectra and native axis exactly as provided. "
            "The matching R scripts show downstream crop (400-1800 cm^-1), baseline correction, and normalization "
            "for specific figures, but those preprocessing operations are not applied during GAIRA raw ingestion."
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
        axis = self.rows[0]["wavenumbers"]
        subclass_counts = Counter(row["subclass_label"] for row in self.rows)
        class_counts = Counter((row["subclass_label"], row["class_label"]) for row in self.rows)
        print("cspp_serum dataset audit")
        print(f"Dataset root: {self.dataset_root}")
        print(f"Native axis: {axis[0]:.2f} to {axis[-1]:.2f} cm^-1 ({len(axis)} points)")
        print("Subclass counts:")
        for subclass_label, count in sorted(subclass_counts.items()):
            print(f"  {subclass_label}: {count}")
        print("Class-count preview:")
        for (subclass_label, class_label), count in sorted(class_counts.items()):
            print(f"  {subclass_label} / {class_label}: {count}")
        print("This archive is a serum methodology dataset with heterogeneous figure families, not a disease benchmark.")

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
                    "collection_protocol": None,
                    "preparation_protocol": "centrifugal silver plasmonic paper serum methodology archive",
                    "instrument": "BWS465-785S / BTC665N-785S-SYS BWtek-format 785 nm system",
                    "laser_wavelength_nm": "785",
                    "spectral_range": f"{wavenumbers[0]:.2f}-{wavenumbers[-1]:.2f}",
                    "preprocessing_summary": preprocessing_summary,
                    "source_file": row["source_file"],
                    "notes": row["row_note"],
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
            for peak_row in self._detect_peaks_for_spectrum(row["intensity_values"], row["wavenumbers"]):
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

    def _insert_dataframe(
        self,
        connection: duckdb.DuckDBPyConnection,
        table_name: str,
        df: pd.DataFrame,
    ) -> int:
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

        print("cspp_serum ingestion complete.")
        print(f"Inserted biosample_metadata rows: {metadata_count}")
        print(f"Inserted biosample_spectra rows: {spectra_count}")
        print(f"Inserted biosample_spectrum_points rows: {point_count}")
        print(f"Inserted biosample_peaks rows: {peak_count}")
