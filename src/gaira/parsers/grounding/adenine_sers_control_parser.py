import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class AdenineSERSControlParser:
    """Parser for the adenine-focused controlled SERS archive (Zenodo 17035751)."""

    SUPPORT_DOCUMENT_ID = "adenine_sers_control_doc_001"
    SUPPORT_BACKGROUND_ID = "adenine_sers_control_bg_support"
    BACKGROUND_FILE = "bg.CSV"

    def __init__(self, dataset_id: str, dataset_root: Path, db_path: Path) -> None:
        self.dataset_id = dataset_id
        self.dataset_root = Path(dataset_root)
        self.db_path = Path(db_path)
        self.rows, self.background_row = self._build_rows()

    def _read_two_column_csv(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"Missing adenine CSV file: {path}")

        df = pd.read_csv(
            path,
            sep=";",
            header=None,
            names=["wavenumber", "intensity"],
            dtype=str,
            engine="python",
        )
        df = df.apply(lambda column: column.str.replace(",", ".", regex=False).str.strip())
        df["wavenumber"] = pd.to_numeric(df["wavenumber"], errors="coerce")
        df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
        df = df.dropna(subset=["wavenumber", "intensity"]).reset_index(drop=True)
        if df.empty:
            raise ValueError(f"{path.name} did not yield numeric two-column SERS data.")

        wavenumbers = df["wavenumber"].to_numpy(dtype=float)
        intensities = df["intensity"].to_numpy(dtype=float)
        if len(wavenumbers) < 50:
            raise ValueError(f"{path.name} exposed too few numeric rows to be a valid spectrum.")
        return wavenumbers, intensities

    def _slugify(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()

    def _parse_metadata(self, path: Path) -> dict:
        stem = path.stem
        lower_stem = stem.lower()

        if stem.startswith("Adenine_bAgNPs_"):
            release_concentration = stem.replace("Adenine_bAgNPs_", "")
            return {
                "experiment_family": "bag_nps_concentration_series",
                "grounding_role": "calibration_like_direct_grounding",
                "compound_label": "adenine",
                "class_label": f"adenine_{self._slugify(release_concentration)}",
                "concentration_label": release_concentration,
                "replicate_id": None,
                "source_label_note": (
                    "Released bAgNPs adenine concentration-series spectrum. "
                    "Useful as controlled adenine SERS grounding under one nanoparticle family."
                ),
                "substrate_type": "silver_nanoparticle_sers_substrate",
            }

        if stem == "Adenine_1ng_mL":
            return {
                "experiment_family": "lateral_flow_strip_reference",
                "grounding_role": "calibration_like_direct_grounding",
                "compound_label": "adenine",
                "class_label": "adenine_1ng_ml",
                "concentration_label": "1ng_mL",
                "replicate_id": None,
                "source_label_note": (
                    "Released adenine lateral-flow strip reference spectrum at 1 ng/mL."
                ),
                "substrate_type": "structured_lateral_flow_sers_surface",
            }

        if stem == "ad1ug_Average":
            return {
                "experiment_family": "average_reference",
                "grounding_role": "calibration_like_direct_grounding",
                "compound_label": "adenine",
                "class_label": "adenine_1ug_average",
                "concentration_label": "1ug",
                "replicate_id": "average",
                "source_label_note": (
                    "Released averaged adenine reference spectrum at the 1 ug release condition."
                ),
                "substrate_type": "structured_lateral_flow_sers_surface",
            }

        if stem == "bAg-koloid_ad1ug_0.5mW_Average":
            return {
                "experiment_family": "colloid_average_reference",
                "grounding_role": "calibration_like_direct_grounding",
                "compound_label": "adenine",
                "class_label": "adenine_1ug_colloid_average",
                "concentration_label": "1ug",
                "replicate_id": "average",
                "source_label_note": (
                    "Released averaged adenine colloid-reference spectrum at 1 ug and 0.5 mW."
                ),
                "substrate_type": "silver_nanoparticle_sers_substrate",
            }

        if re.fullmatch(r"bAgNPs_Adenine_1ng_\d+", stem):
            replicate_id = stem.rsplit("_", 1)[-1]
            return {
                "experiment_family": "bag_nps_replicate_series",
                "grounding_role": "calibration_like_direct_grounding",
                "compound_label": "adenine",
                "class_label": "adenine_1ng_replicate_series",
                "concentration_label": "1ng",
                "replicate_id": f"rep{replicate_id}",
                "source_label_note": (
                    "Released adenine 1 ng replicate spectrum on the bAgNPs condition."
                ),
                "substrate_type": "silver_nanoparticle_sers_substrate",
            }

        if stem == "ad1ng":
            return {
                "experiment_family": "stability_series",
                "grounding_role": "calibration_like_direct_grounding",
                "compound_label": "adenine",
                "class_label": "adenine_1ng_fresh",
                "concentration_label": "1ng",
                "replicate_id": "fresh",
                "source_label_note": (
                    "Released fresh adenine 1 ng control spectrum."
                ),
                "substrate_type": "structured_lateral_flow_sers_surface",
            }

        if stem == "ad1ng_after_two_weeks":
            return {
                "experiment_family": "stability_series",
                "grounding_role": "calibration_like_direct_grounding",
                "compound_label": "adenine",
                "class_label": "adenine_1ng_after_two_weeks",
                "concentration_label": "1ng_after_two_weeks",
                "replicate_id": "after_two_weeks",
                "source_label_note": (
                    "Released adenine 1 ng control spectrum after two weeks, useful as a stability reference."
                ),
                "substrate_type": "structured_lateral_flow_sers_surface",
            }

        raise ValueError(f"Unexpected adenine CSV naming pattern: {path.name}")

    def _build_rows(self) -> tuple[list[dict], dict | None]:
        direct_rows: list[dict] = []
        background_row: dict | None = None

        for path in sorted(self.dataset_root.glob("*.CSV")):
            wavenumbers, intensity_values = self._read_two_column_csv(path)

            if path.name == self.BACKGROUND_FILE:
                background_row = {
                    "support_spectrum_id": self.SUPPORT_BACKGROUND_ID,
                    "document_id": self.SUPPORT_DOCUMENT_ID,
                    "dataset_id": self.dataset_id,
                    "source_dataset_id": self.dataset_id,
                    "evidence_family": "controlled_background_support",
                    "citation_label": "Adenine_Control_Background",
                    "wavenumbers": wavenumbers,
                    "intensity_values": intensity_values,
                    "notes": (
                        "Released background/control spectrum from the adenine SERS archive. "
                        "Kept as support-only background context, not as direct tier-1 grounding."
                    ),
                }
                continue

            parsed = self._parse_metadata(path)
            source_row_id = path.stem
            direct_rows.append(
                {
                    "grounding_id": f"{self.dataset_id}_{source_row_id}",
                    "dataset_id": self.dataset_id,
                    "source_dataset_id": self.dataset_id,
                    "source_row_id": source_row_id,
                    "source_file": path.name,
                    "wavenumbers": wavenumbers,
                    "intensity_values": intensity_values,
                    **parsed,
                }
            )

        if not direct_rows:
            raise ValueError("No adenine-controlled CSV spectra were parsed into direct grounding rows.")
        return direct_rows, background_row

    def _preprocessing_summary(self) -> str:
        return (
            "Raw grounding ingest preserves the released two-column adenine SERS CSV spectra exactly as distributed. "
            "GAIRA keeps the native machine-readable axis and intensity values untouched at ingest time. "
            "Processed grounding later crops the spectra to 400-1800 cm^-1 and applies vector normalization for "
            "comparison-friendly controlled grounding."
        )

    def _base_notes(self, row: dict) -> str:
        return (
            f"{row['source_label_note']} Current GAIRA_GROUNDING framing treats this as controlled analyte "
            "and calibration-like SERS grounding for adenine/purine-like comparison rather than patient, serum, "
            f"or disease evidence. Source file={row['source_file']}."
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
        axis_min = min(float(np.min(row["wavenumbers"])) for row in self.rows)
        axis_max = max(float(np.max(row["wavenumbers"])) for row in self.rows)
        point_min = min(int(len(row["wavenumbers"])) for row in self.rows)
        point_max = max(int(len(row["wavenumbers"])) for row in self.rows)
        print("adenine_sers_control audit")
        print(f"Dataset root: {self.dataset_root}")
        print(
            f"Native axis families span {axis_min:.2f} to {axis_max:.2f} cm^-1 "
            f"with {point_min} to {point_max} points."
        )
        print("Direct grounding rows by family:")
        family_counts = (
            pd.DataFrame(self.rows)
            .groupby(["experiment_family", "class_label"])
            .size()
            .reset_index(name="n")
        )
        print(family_counts.to_string(index=False))
        if self.background_row is not None:
            print("Support-only background spectrum detected: bg.CSV")
        print("This archive is calibration-like controlled adenine grounding, not biosample evidence.")

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
                    "biosample_context": "controlled adenine SERS reference series",
                    "substrate_type": row["substrate_type"],
                    "substrate_material": "silver",
                    "instrument": "Controlled adenine SERS archive with lateral-flow and nanoparticle conditions",
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
            for peak_row in self._detect_peaks_for_spectrum(row["intensity_values"], row["wavenumbers"]):
                rows.append(
                    {
                        "grounding_id": row["grounding_id"],
                        "dataset_id": row["dataset_id"],
                        "source_dataset_id": row["source_dataset_id"],
                        "source_row_id": row["source_row_id"],
                        **peak_row,
                    }
                )
        return pd.DataFrame(rows)

    def extract_support_documents(self) -> pd.DataFrame:
        rows = [
            {
                "document_id": self.SUPPORT_DOCUMENT_ID,
                "dataset_id": self.dataset_id,
                "source_dataset_id": self.dataset_id,
                "evidence_family": "controlled_analyte_support",
                "evidence_tier": "tier2_interpretive_support",
                "support_type": "text",
                "citation_label": "Adenine_Control_Archive_2025",
                "title": "Adenine controlled SERS calibration note",
                "authors": None,
                "year": "2025",
                "journal": "Analytica Chimica Acta release archive",
                "doi": "10.1016/j.aca.2025.344595",
                "source_file": "1-s2.0-S0003267025009894-main.pdf|LOD_opakovatelnost.xlsx|adenine CSV series",
                "is_digitized": "no",
                "use_for_primary_matching": "no",
                "use_for_supporting_comparison": "yes",
                "use_for_rag": "yes",
                "notes": (
                    "Support-only interpretive note for the adenine-controlled SERS archive. "
                    "Used to explain the archive's calibration-like role and concentration-series framing."
                ),
            }
        ]
        return pd.DataFrame(rows)

    def extract_support_chunks(self) -> pd.DataFrame:
        chunks = [
            (
                "archive_role_note",
                (
                    "adenine_sers_control is a controlled analyte SERS archive rather than a biosample or disease dataset. "
                    "Its spectra should be interpreted as calibration-like grounding for adenine and purine-like comparison, "
                    "not as direct disease evidence."
                ),
            ),
            (
                "concentration_series_note",
                (
                    "The release includes adenine-focused concentration-series, replicate-series, average-reference, and "
                    "stability-control spectra across structured strip and silver nanoparticle conditions. This makes the "
                    "archive useful for concentration-response and nucleobase-like grounding, while still requiring caution "
                    "about substrate-condition dependence."
                ),
            ),
            (
                "do_not_overclaim_note",
                (
                    "A strong adenine-controlled match should be reported as controlled grounding support only. "
                    "It can strengthen purine-like or nucleobase-like interpretation, but it does not by itself establish "
                    "adenine as the definitive source of a biosample spectrum."
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
                    "metadata_json": json.dumps({"source_kind": "adenine_control_context"}, sort_keys=True),
                }
            )
        return pd.DataFrame(rows)

    def extract_support_spectra(self) -> pd.DataFrame:
        if self.background_row is None:
            return pd.DataFrame()
        wavenumbers = self.background_row["wavenumbers"]
        return pd.DataFrame(
            [
                {
                    "support_spectrum_id": self.background_row["support_spectrum_id"],
                    "document_id": self.background_row["document_id"],
                    "dataset_id": self.background_row["dataset_id"],
                    "source_dataset_id": self.background_row["source_dataset_id"],
                    "evidence_family": self.background_row["evidence_family"],
                    "citation_label": self.background_row["citation_label"],
                    "x_min": float(wavenumbers[0]),
                    "x_max": float(wavenumbers[-1]),
                    "n_points": int(len(wavenumbers)),
                    "wavenumbers_json": json.dumps([float(value) for value in wavenumbers]),
                    "intensity_json": json.dumps([float(value) for value in self.background_row["intensity_values"]]),
                    "is_digitized": "no",
                    "use_for_primary_matching": "no",
                    "use_for_supporting_comparison": "yes",
                    "use_for_rag": "no",
                    "notes": self.background_row["notes"],
                }
            ]
        )

    def extract_support_spectrum_points(self) -> pd.DataFrame:
        if self.background_row is None:
            return pd.DataFrame()
        rows = []
        for point_index, (wavenumber, intensity) in enumerate(
            zip(self.background_row["wavenumbers"], self.background_row["intensity_values"]),
            start=1,
        ):
            rows.append(
                {
                    "support_spectrum_id": self.background_row["support_spectrum_id"],
                    "document_id": self.background_row["document_id"],
                    "dataset_id": self.dataset_id,
                    "point_index": point_index,
                    "wavenumber": float(wavenumber),
                    "intensity": float(intensity),
                }
            )
        return pd.DataFrame(rows)

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
        support_documents_df = self.extract_support_documents()
        support_chunks_df = self.extract_support_chunks()
        support_spectra_df = self.extract_support_spectra()
        support_points_df = self.extract_support_spectrum_points()

        with duckdb.connect(str(self.db_path)) as connection:
            connection.execute("DELETE FROM grounding_peaks WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM grounding_spectrum_points WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM grounding_spectra WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM grounding_metadata WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM grounding_support_spectrum_points WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM grounding_support_spectra WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM grounding_support_chunks WHERE dataset_id = ?", [self.dataset_id])
            connection.execute("DELETE FROM grounding_support_documents WHERE dataset_id = ?", [self.dataset_id])

            self._insert_dataframe(connection, "grounding_metadata", metadata_df)
            self._insert_dataframe(connection, "grounding_spectra", spectra_df)
            self._insert_dataframe(connection, "grounding_spectrum_points", points_df)
            self._insert_dataframe(connection, "grounding_peaks", peaks_df)
            self._insert_dataframe(connection, "grounding_support_documents", support_documents_df)
            self._insert_dataframe(connection, "grounding_support_chunks", support_chunks_df)
            self._insert_dataframe(connection, "grounding_support_spectra", support_spectra_df)
            self._insert_dataframe(connection, "grounding_support_spectrum_points", support_points_df)

        print(f"Inserted grounding_metadata rows: {len(metadata_df)}")
        print(f"Inserted grounding_spectra rows: {len(spectra_df)}")
        print(f"Inserted grounding_spectrum_points rows: {len(points_df)}")
        print(f"Inserted grounding_peaks rows: {len(peaks_df)}")
        print(f"Inserted grounding_support_documents rows: {len(support_documents_df)}")
        print(f"Inserted grounding_support_chunks rows: {len(support_chunks_df)}")
        print(f"Inserted grounding_support_spectra rows: {len(support_spectra_df)}")
        print(f"Inserted grounding_support_spectrum_points rows: {len(support_points_df)}")
