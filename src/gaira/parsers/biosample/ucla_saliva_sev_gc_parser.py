import json
import re
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from gaira.parsers.biosample.base import BiosampleParserBase


class UCLASalivaSEVGCParser(BiosampleParserBase):
    """Parser for the reconstructed UCLA saliva sEV gastric-cancer shard family."""

    MANIFEST_NAME = "shard_manifest.csv"

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        super().__init__(dataset_id=dataset_id, dataset_root=dataset_root, db_path=db_path)
        self.manifest_path = self.dataset_root / self.MANIFEST_NAME
        self.manifest_df = self._load_manifest()
        self.rows = self._build_rows()

    def _load_manifest(self) -> pd.DataFrame:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing UCLA shard manifest: {self.manifest_path}")
        manifest_df = pd.read_csv(self.manifest_path)
        if "recovered" not in manifest_df.columns:
            raise ValueError("UCLA shard manifest is missing the recovered column.")
        return manifest_df

    def _read_two_column_txt(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path, sep="\t", header=None, names=["wavenumber", "intensity"])
        df["wavenumber"] = pd.to_numeric(df["wavenumber"], errors="coerce")
        df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
        df = df.dropna(subset=["wavenumber", "intensity"]).reset_index(drop=True)
        if len(df) < 100:
            raise ValueError(f"{path} did not expose a valid two-column spectral trace.")
        return df["wavenumber"].to_numpy(dtype=float), df["intensity"].to_numpy(dtype=float)

    def _parse_identity(self, row: pd.Series) -> dict[str, str | None]:
        file_name = str(row["file_name"])
        stem = Path(file_name).stem
        article_id = str(row["article_id"])
        title = str(row.get("title", ""))
        source_record = str(row.get("source_record_url", ""))

        class_label = "healthy_control" if "cnt" in stem.lower() or "control" in title.lower() else "gastric_cancer"

        patient_match = re.search(r"cnt[_-]?(\d+)", stem, flags=re.IGNORECASE)
        if patient_match:
            patient_id = f"cnt_{int(patient_match.group(1)):02d}"
        else:
            patient_match = re.search(r"gc[_-]?(\d+)", stem, flags=re.IGNORECASE)
            patient_id = f"gc_{int(patient_match.group(1)):02d}" if patient_match else f"article_{article_id}"

        replicate_match = re.search(r"_([0-9]+)__", stem)
        replicate_id = f"rep{int(replicate_match.group(1)):04d}" if replicate_match else None

        mapping_match = re.match(r"(mapping[^_]+)", stem, flags=re.IGNORECASE)
        mapping_id = mapping_match.group(1).lower() if mapping_match else f"article_{article_id}"

        source_row_id = re.sub(r"[^A-Za-z0-9_]+", "_", f"{article_id}__{stem}").strip("_").lower()
        return {
            "class_label": class_label,
            "patient_id": patient_id,
            "sample_id": patient_id,
            "replicate_id": replicate_id,
            "subclass_label": mapping_id,
            "source_row_id": source_row_id,
            "source_record": source_record,
        }

    def _build_rows(self) -> list[dict]:
        rows: list[dict] = []
        recovered_df = self.manifest_df[self.manifest_df["recovered"] == True].copy()
        for _, manifest_row in recovered_df.iterrows():
            local_path = self.dataset_root / str(manifest_row["local_relpath"])
            if not local_path.exists():
                continue
            wavenumbers, intensities = self._read_two_column_txt(local_path)
            identity = self._parse_identity(manifest_row)
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
                    "source_file": str(manifest_row["local_relpath"]),
                    "wavenumbers": wavenumbers,
                    "intensity_values": intensities,
                    "source_record": identity["source_record"],
                }
            )

        if not rows:
            raise ValueError("No UCLA saliva sEV spectra were reconstructed from the shard manifest.")
        return rows

    def _preprocessing_summary(self) -> str:
        return (
            "Raw ingest preserves each recovered two-column Figshare shard txt file exactly as downloaded. "
            "Class labels and patient identifiers are reconstructed from shard titles and filenames, and "
            "the shard-level manifest is the canonical provenance layer for this dataset."
        )

    def _base_notes(self, row: dict) -> str:
        return (
            "Reconstructed UCLA saliva sEV shard spectrum. "
            f"class_label={row['class_label']}, patient_id={row['patient_id']}, "
            f"subclass_label={row['subclass_label']}, source_record={row['source_record']}."
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
        patient_counts = Counter(row["patient_id"] for row in self.rows)
        print("ucla_saliva_sev_gc dataset audit")
        print(f"Dataset root: {self.dataset_root}")
        print(f"Shard manifest: {self.manifest_path}")
        print(f"Recovered spectra: {len(self.rows)}")
        print(f"Unique patients: {len(patient_counts)}")
        print("Class counts:")
        for label, count in sorted(class_counts.items()):
            print(f"  {label}: {count}")

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
                    "matrix": "saliva_sEV",
                    "disease_context": "gastric cancer saliva sEV cohort" if row["class_label"] == "gastric_cancer" else "healthy control saliva sEV cohort",
                    "class_label": row["class_label"],
                    "subclass_label": row["subclass_label"],
                    "collection_protocol": None,
                    "preparation_protocol": "reconstructed saliva sEV shard family from Figshare records",
                    "instrument": "785 nm saliva sEV Raman/SERS mapping acquisition",
                    "laser_wavelength_nm": "785",
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
