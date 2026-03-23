import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class MetaboliteSERSParser:
    """Parser for the local metabolite SERS Fityk fit/peak archive."""

    SUPPORT_DOCUMENT_ID = "metabolite_sers63_support_doc_001"

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        self.dataset_id = dataset_id
        self.dataset_root = Path(dataset_root)
        self.db_path = Path(db_path)
        self.fit_dir = self.dataset_root / "fit"
        self.peaks_dir = self.dataset_root / "peaks"
        self.rows = self._build_rows()

    def _slugify(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()

    def _read_fit_spectrum(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        xs: list[float] = []
        ys: list[float] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = re.match(r"X\[(\d+)\]=([^,]+), Y\[\1\]=([^,]+)", line.strip())
            if match:
                xs.append(float(match.group(2)))
                ys.append(float(match.group(3)))

        if len(xs) < 50:
            raise ValueError(f"{path.name} did not expose enough numeric X/Y rows to reconstruct a spectrum.")

        wavenumbers = np.asarray(xs, dtype=float)
        intensity_values = np.asarray(ys, dtype=float)
        if not np.all(np.diff(wavenumbers) > 0):
            raise ValueError(f"{path.name} does not expose a strictly increasing Raman axis.")
        return wavenumbers, intensity_values

    def _parse_peaks_file(self, path: Path) -> list[dict]:
        rows: list[dict] = []
        if not path.exists():
            return rows

        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped.startswith("%_"):
                continue
            tokens = re.split(r"\s+", stripped)
            if len(tokens) < 6:
                continue
            try:
                peak_intensity = float(tokens[-4])
                peak_cm = float(tokens[-3])
                hwhm = float(tokens[-2])
            except ValueError:
                continue
            rows.append(
                {
                    "peak_cm": peak_cm,
                    "peak_intensity": peak_intensity,
                    "prominence": peak_intensity,
                    "hwhm": hwhm,
                }
            )
        return rows

    def _detect_peaks_for_spectrum(self, intensity_values: np.ndarray, wavenumbers: np.ndarray) -> list[dict]:
        shifted = np.asarray(intensity_values, dtype=float) - float(np.min(intensity_values))
        max_value = float(np.max(shifted))
        if max_value <= 0:
            return []
        normalized = shifted / max_value
        peak_indices, properties = find_peaks(normalized, prominence=0.05, height=0.05, distance=5)
        return [
            {
                "peak_cm": float(wavenumbers[peak_index]),
                "peak_intensity": float(properties["peak_heights"][peak_rank - 1]),
                "prominence": float(properties["prominences"][peak_rank - 1]),
                "hwhm": None,
            }
            for peak_rank, peak_index in enumerate(peak_indices, start=1)
        ]

    def _build_rows(self) -> list[dict]:
        if not self.fit_dir.exists():
            raise FileNotFoundError(f"Missing fit folder: {self.fit_dir}")
        if not self.peaks_dir.exists():
            raise FileNotFoundError(f"Missing peaks folder: {self.peaks_dir}")

        rows: list[dict] = []
        for fit_path in sorted(self.fit_dir.glob("*.fit")):
            if fit_path.name.startswith("._"):
                continue
            source_row_id = fit_path.stem
            class_label = self._slugify(source_row_id)
            wavenumbers, intensity_values = self._read_fit_spectrum(fit_path)
            peaks_path = self.peaks_dir / f"{source_row_id}-fingerprint.peaks"
            peak_rows = self._parse_peaks_file(peaks_path)
            if not peak_rows:
                peak_rows = self._detect_peaks_for_spectrum(intensity_values, wavenumbers)

            rows.append(
                {
                    "grounding_id": f"{self.dataset_id}_{class_label}",
                    "dataset_id": self.dataset_id,
                    "source_dataset_id": self.dataset_id,
                    "source_row_id": source_row_id,
                    "source_file": f"fit/{fit_path.name}",
                    "peaks_file": f"peaks/{peaks_path.name}" if peaks_path.exists() else None,
                    "experiment_family": "fityk_metabolite_fingerprint_archive",
                    "grounding_role": "direct_metabolite_grounding",
                    "compound_label": source_row_id,
                    "class_label": class_label,
                    "concentration_label": None,
                    "replicate_id": None,
                    "wavenumbers": wavenumbers,
                    "intensity_values": intensity_values,
                    "peak_rows": peak_rows,
                }
            )

        if not rows:
            raise ValueError("No metabolite Fityk spectra were reconstructed from the fit archive.")
        return rows

    def _preprocessing_summary(self) -> str:
        return (
            "Raw grounding ingest reconstructs the released metabolite SERS spectra directly from the Fityk .fit "
            "scripts by reading the stored X[i] and Y[i] arrays. Released -fingerprint.peaks files are used where "
            "present; otherwise GAIRA detects peaks from the reconstructed spectrum."
        )

    def _base_notes(self, row: dict) -> str:
        peaks_note = (
            f"Released peak list={row['peaks_file']}."
            if row["peaks_file"]
            else "No released -fingerprint.peaks file was present for this spectrum; GAIRA detected peaks directly."
        )
        return (
            f"Reconstructed metabolite SERS fingerprint from Fityk fit script {row['source_file']}. "
            f"Compound label={row['compound_label']}. {peaks_note} "
            "Current GAIRA framing treats this archive as direct small-molecule grounding and comparison support, "
            "not as biosample or disease evidence."
        )

    def audit(self) -> None:
        axis_min = min(float(np.min(row["wavenumbers"])) for row in self.rows)
        axis_max = max(float(np.max(row["wavenumbers"])) for row in self.rows)
        point_counts = sorted({int(len(row["wavenumbers"])) for row in self.rows})
        print("metabolite_sers63_support audit")
        print(f"Dataset root: {self.dataset_root}")
        print(f"Reconstructed spectra: {len(self.rows)}")
        print(f"Axis families: {point_counts} points spanning {axis_min:.2f} to {axis_max:.2f} cm^-1")
        with_peak_files = sum(1 for row in self.rows if row["peaks_file"])
        print(f"Rows with released .peaks files: {with_peak_files}")
        print(f"Rows without released .peaks files: {len(self.rows) - with_peak_files}")

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
                    "biosample_context": "released metabolite fingerprint archive",
                    "substrate_type": "unknown_sers_substrate",
                    "substrate_material": "unknown",
                    "instrument": "Fityk-released metabolite SERS fingerprint archive",
                    "laser_wavelength_nm": "unknown",
                    "spectral_range": f"{wavenumbers[0]:.2f}-{wavenumbers[-1]:.2f}",
                    "preprocessing_summary": preprocessing_summary,
                    "notes": self._base_notes(row),
                }
            )
        return pd.DataFrame(rows)

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
        return pd.DataFrame(rows)

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
        return pd.DataFrame(rows)

    def extract_peaks(self) -> pd.DataFrame:
        rows = []
        for row in self.rows:
            for peak_rank, peak_row in enumerate(row["peak_rows"], start=1):
                rows.append(
                    {
                        "grounding_id": row["grounding_id"],
                        "dataset_id": row["dataset_id"],
                        "source_dataset_id": row["source_dataset_id"],
                        "source_row_id": row["source_row_id"],
                        "peak_rank": peak_rank,
                        "peak_cm": float(peak_row["peak_cm"]),
                        "peak_intensity": float(peak_row["peak_intensity"]),
                        "prominence": float(peak_row["prominence"]),
                    }
                )
        return pd.DataFrame(rows)

    def extract_support_documents(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "document_id": self.SUPPORT_DOCUMENT_ID,
                    "dataset_id": self.dataset_id,
                    "source_dataset_id": self.dataset_id,
                    "evidence_family": "metabolite_fingerprint_support",
                    "evidence_tier": "tier2_interpretive_support",
                    "support_type": "text",
                    "citation_label": "Metabolite_Fityk_Fingerprint_Archive",
                    "title": "Metabolite SERS fingerprint Fityk archive note",
                    "authors": None,
                    "year": "2025",
                    "journal": "local supplementary archive",
                    "doi": None,
                    "source_file": "fit/*.fit|peaks/*.peaks",
                    "is_digitized": "no",
                    "use_for_primary_matching": "no",
                    "use_for_supporting_comparison": "yes",
                    "use_for_rag": "yes",
                    "notes": (
                        "Support note for the reconstructed metabolite SERS fingerprint archive. "
                        "Used to explain that the spectra were rebuilt from Fityk scripts and should remain "
                        "small-molecule grounding support rather than biosample evidence."
                    ),
                }
            ]
        )

    def extract_support_chunks(self) -> pd.DataFrame:
        chunks = [
            (
                "archive_structure",
                (
                    "metabolite_sers63_support reconstructs released metabolite SERS spectra from Fityk .fit scripts "
                    "and uses released -fingerprint.peaks files where available. The archive is useful for small-"
                    "molecule grounding and cautious band-level comparison."
                ),
            ),
            (
                "role_note",
                (
                    "Matches to this archive should be interpreted as metabolite-like grounding support only. "
                    "They can strengthen biochemical plausibility, but they do not by themselves establish that a "
                    "biosample spectrum contains that metabolite in isolation."
                ),
            ),
        ]
        rows = []
        for chunk_order, (section, chunk_text) in enumerate(chunks, start=1):
            rows.append(
                {
                    "chunk_id": f"{self.SUPPORT_DOCUMENT_ID}_chunk_{chunk_order:02d}",
                    "document_id": self.SUPPORT_DOCUMENT_ID,
                    "dataset_id": self.dataset_id,
                    "chunk_order": chunk_order,
                    "section": section,
                    "chunk_text": chunk_text,
                    "metadata_json": json.dumps({"source_kind": "metabolite_fityk_context"}, sort_keys=True),
                }
            )
        return pd.DataFrame(rows)

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
        support_documents_df = self.extract_support_documents()
        support_chunks_df = self.extract_support_chunks()

        with duckdb.connect(str(self.db_path)) as connection:
            connection.execute("DELETE FROM grounding_peaks WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM grounding_spectrum_points WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM grounding_spectra WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM grounding_metadata WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM grounding_support_chunks WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM grounding_support_documents WHERE dataset_id = ?", [self.dataset_id])

            self._insert_dataframe(connection, "grounding_metadata", metadata_df)
            self._insert_dataframe(connection, "grounding_spectra", spectra_df)
            self._insert_dataframe(connection, "grounding_spectrum_points", points_df)
            self._insert_dataframe(connection, "grounding_peaks", peaks_df)
            self._insert_dataframe(connection, "grounding_support_documents", support_documents_df)
            self._insert_dataframe(connection, "grounding_support_chunks", support_chunks_df)

        print(f"Inserted grounding_metadata rows: {len(metadata_df)}")
        print(f"Inserted grounding_spectra rows: {len(spectra_df)}")
        print(f"Inserted grounding_spectrum_points rows: {len(points_df)}")
        print(f"Inserted grounding_peaks rows: {len(peaks_df)}")
        print(f"Inserted grounding_support_documents rows: {len(support_documents_df)}")
        print(f"Inserted grounding_support_chunks rows: {len(support_chunks_df)}")
