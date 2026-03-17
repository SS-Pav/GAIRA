from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


HCC_SERUM_PROCESSING_VERSION = "v1_crop430_1730_interp1_minmax"
GROUNDING_PROCESSING_VERSION = "v1_crop400_1800_interp1_vector"


@dataclass
class SpectrumQuery:
    query_id: str
    query_label: str
    query_family: str
    source_dataset_id: str
    x: np.ndarray
    y: np.ndarray
    notes: str


def _parse_json_array(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm <= 0:
        return np.zeros_like(values, dtype=float)
    return values / norm


def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x_norm = _normalize_vector(x)
    y_norm = _normalize_vector(y)
    return float(np.dot(x_norm, y_norm))


def _sanitize_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()


def _align_candidate_to_query(
    query_x: np.ndarray,
    query_y: np.ndarray,
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
    min_overlap_points: int = 100,
) -> tuple[np.ndarray, np.ndarray] | None:
    overlap_min = max(float(np.min(query_x)), float(np.min(candidate_x)))
    overlap_max = min(float(np.max(query_x)), float(np.max(candidate_x)))
    mask = (query_x >= overlap_min) & (query_x <= overlap_max)
    overlap_query_x = query_x[mask]
    overlap_query_y = query_y[mask]
    if len(overlap_query_x) < min_overlap_points:
        return None
    aligned_candidate_y = np.interp(overlap_query_x, candidate_x, candidate_y)
    return overlap_query_y, aligned_candidate_y


def _detect_query_bands(x_values: np.ndarray, y_values: np.ndarray, top_n: int = 5) -> list[float]:
    normalized = _normalize_vector(y_values)
    peak_indices, properties = find_peaks(normalized, prominence=0.01, distance=5)
    if len(peak_indices) == 0:
        top_indices = np.argsort(normalized)[-top_n:]
        return sorted(float(x_values[index]) for index in top_indices)
    prominences = properties.get("prominences", np.ones(len(peak_indices)))
    ranked_indices = peak_indices[np.argsort(prominences)[::-1][:top_n]]
    return sorted(float(x_values[index]) for index in ranked_indices)


def _extract_band_ranges(chunk_text: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for start_text, end_text in re.findall(r"(\d{3,4})\s*-\s*(\d{2,4})\s*:", chunk_text):
        start_value = float(start_text)
        end_value = float(end_text)
        if end_value < 100:
            end_value = float(f"{start_text[:len(start_text)-len(end_text)]}{end_text}")
        ranges.append((min(start_value, end_value), max(start_value, end_value)))
    for single_text in re.findall(r"(?<!-)(?<!\d)(\d{3,4})\s*:", chunk_text):
        single_value = float(single_text)
        ranges.append((single_value, single_value))
    return ranges


class GroundingSearchEngine:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._load_cached_tables()

    def _load_cached_tables(self) -> None:
        with duckdb.connect(str(self.db_path), read_only=True) as connection:
            self.reference_df = connection.execute(
                """
                SELECT
                  s.ref_id,
                  s.dataset_id,
                  m.component,
                  m.biochemical_class,
                  m.reference,
                  s.wavenumbers_json,
                  s.intensity_json
                FROM reference_spectra s
                JOIN reference_metadata m
                  ON s.ref_id = m.ref_id
                 AND s.dataset_id = m.dataset_id
                WHERE s.dataset_id = 'ramanbiolib'
                """
            ).fetchdf()

            self.grounding_summary_df = connection.execute(
                """
                SELECT
                  g.summary_id,
                  g.dataset_id,
                  g.experiment_family,
                  g.class_label,
                  g.n_spectra,
                  g.mean_wavenumbers_json,
                  g.mean_intensity_json,
                  MIN(m.source_file) AS source_file
                FROM grounding_class_summary g
                JOIN grounding_metadata m
                  ON g.dataset_id = m.dataset_id
                 AND g.experiment_family = m.experiment_family
                 AND g.class_label = m.class_label
                WHERE g.dataset_id = 'serum_ag_colloids_grounding'
                  AND g.processing_version = ?
                GROUP BY
                  g.summary_id, g.dataset_id, g.experiment_family, g.class_label, g.n_spectra,
                  g.mean_wavenumbers_json, g.mean_intensity_json
                """,
                [GROUNDING_PROCESSING_VERSION],
            ).fetchdf()

            self.hcc_summary_df = connection.execute(
                """
                SELECT
                  summary_id,
                  dataset_id,
                  class_label,
                  subclass_label,
                  n_spectra,
                  mean_wavenumbers_json,
                  mean_intensity_json
                FROM biosample_class_summary
                WHERE dataset_id = 'hcc_serum'
                  AND processing_version = ?
                ORDER BY class_label
                """,
                [HCC_SERUM_PROCESSING_VERSION],
            ).fetchdf()

            self.support_chunk_df = connection.execute(
                """
                SELECT
                  c.chunk_id,
                  c.document_id,
                  c.dataset_id,
                  c.section,
                  c.chunk_text,
                  c.chunk_order,
                  d.evidence_family,
                  d.citation_label,
                  d.title,
                  d.source_file,
                  d.evidence_tier,
                  d.support_type
                FROM grounding_support_chunks c
                JOIN grounding_support_documents d
                  ON c.document_id = d.document_id
                 AND c.dataset_id = d.dataset_id
                WHERE c.dataset_id = 'serum_ag_colloids_literature_grounding'
                ORDER BY c.document_id, c.chunk_order
                """
            ).fetchdf()

            self.support_spectra_df = connection.execute(
                """
                SELECT
                  s.support_spectrum_id,
                  s.document_id,
                  s.dataset_id,
                  s.evidence_family,
                  s.citation_label,
                  s.x_min,
                  s.x_max,
                  s.n_points,
                  s.wavenumbers_json,
                  s.intensity_json,
                  s.is_digitized,
                  s.use_for_primary_matching,
                  s.use_for_supporting_comparison,
                  s.use_for_rag,
                  s.notes,
                  d.source_file
                FROM grounding_support_spectra s
                JOIN grounding_support_documents d
                  ON s.document_id = d.document_id
                 AND s.dataset_id = d.dataset_id
                WHERE s.dataset_id = 'serum_ag_colloids_literature_grounding'
                """
            ).fetchdf()

            self.reference_peaks_df = connection.execute(
                """
                SELECT
                  p.ref_id,
                  p.peak_cm,
                  p.rel_intensity,
                  m.component,
                  m.biochemical_class,
                  m.reference
                FROM reference_peaks p
                JOIN reference_metadata m
                  ON p.ref_id = m.ref_id
                 AND p.dataset_id = m.dataset_id
                WHERE p.dataset_id = 'ramanbiolib'
                """
            ).fetchdf()

    def get_demo_queries(self) -> list[SpectrumQuery]:
        queries: list[SpectrumQuery] = []
        for row in self.hcc_summary_df.to_dict(orient="records"):
            queries.append(
                SpectrumQuery(
                    query_id=f"hcc_serum_{_sanitize_label(row['class_label'])}",
                    query_label=str(row["class_label"]),
                    query_family="hcc_serum_class_mean",
                    source_dataset_id="hcc_serum",
                    x=_parse_json_array(row["mean_wavenumbers_json"]),
                    y=_parse_json_array(row["mean_intensity_json"]),
                    notes=f"hcc_serum class mean from subclass {row['subclass_label']}",
                )
            )

        selected_grounding_labels = ["UAfree", "Hypox", "UAiso+HSA"]
        for label in selected_grounding_labels:
            match_df = self.grounding_summary_df[self.grounding_summary_df["class_label"] == label]
            if match_df.empty:
                continue
            row = match_df.iloc[0]
            queries.append(
                SpectrumQuery(
                    query_id=f"grounding_{_sanitize_label(label)}",
                    query_label=str(label),
                    query_family=str(row["experiment_family"]),
                    source_dataset_id="serum_ag_colloids_grounding",
                    x=_parse_json_array(row["mean_wavenumbers_json"]),
                    y=_parse_json_array(row["mean_intensity_json"]),
                    notes="Processed grounding class summary",
                )
            )
        return queries

    def search_direct_spectral_evidence(
        self,
        query: SpectrumQuery,
        top_n_per_source: int = 5,
    ) -> pd.DataFrame:
        rows: list[dict] = []

        for row in self.reference_df.to_dict(orient="records"):
            candidate_x = _parse_json_array(row["wavenumbers_json"])
            candidate_y = _parse_json_array(row["intensity_json"])
            aligned = _align_candidate_to_query(query.x, query.y, candidate_x, candidate_y)
            if aligned is None:
                continue
            query_y, candidate_aligned_y = aligned
            score = _cosine_similarity(query_y, candidate_aligned_y)
            rows.append(
                {
                    "query_id": query.query_id,
                    "query_label": query.query_label,
                    "query_family": query.query_family,
                    "mode": "spectrum_to_grounding",
                    "evidence_tier": "tier1_direct_spectral_grounding",
                    "result_type": "ramanbiolib_reference",
                    "source_dataset_id": "ramanbiolib",
                    "source_family": row["biochemical_class"],
                    "source_label": row["component"],
                    "score": score,
                    "provenance": row["reference"],
                    "notes": "Cosine similarity on overlap-aligned spectra.",
                }
            )

        for row in self.grounding_summary_df.to_dict(orient="records"):
            candidate_x = _parse_json_array(row["mean_wavenumbers_json"])
            candidate_y = _parse_json_array(row["mean_intensity_json"])
            aligned = _align_candidate_to_query(query.x, query.y, candidate_x, candidate_y)
            if aligned is None:
                continue
            query_y, candidate_aligned_y = aligned
            score = _cosine_similarity(query_y, candidate_aligned_y)
            rows.append(
                {
                    "query_id": query.query_id,
                    "query_label": query.query_label,
                    "query_family": query.query_family,
                    "mode": "spectrum_to_grounding",
                    "evidence_tier": "tier1_direct_spectral_grounding",
                    "result_type": "study_matched_sers_grounding",
                    "source_dataset_id": row["dataset_id"],
                    "source_family": row["experiment_family"],
                    "source_label": row["class_label"],
                    "score": score,
                    "provenance": row["source_file"],
                    "notes": f"Processed class summary, n_spectra={int(row['n_spectra'])}.",
                }
            )

        result_df = pd.DataFrame(rows)
        if result_df.empty:
            return result_df
        result_df["rank_within_type"] = result_df.groupby("result_type")["score"].rank(
            method="first", ascending=False
        )
        return (
            result_df[result_df["rank_within_type"] <= top_n_per_source]
            .sort_values(["result_type", "score"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def search_supporting_literature_for_spectrum(
        self,
        query: SpectrumQuery,
        top_n: int = 8,
    ) -> pd.DataFrame:
        trigger_bands = _detect_query_bands(query.x, query.y, top_n=5)
        band_rows: list[dict] = []
        for band_cm in trigger_bands:
            band_rows.extend(
                self.search_band_evidence(band_cm=band_cm, tier_filter="tier2_literature_support")
                .head(3)
                .to_dict(orient="records")
            )

        if not band_rows:
            return pd.DataFrame()

        band_df = pd.DataFrame(band_rows)
        band_df["query_id"] = query.query_id
        band_df["query_label"] = query.query_label
        band_df["query_family"] = query.query_family
        band_df["mode"] = "spectrum_to_grounding"
        band_df["notes"] = band_df["notes"].fillna("") + " Triggered from top query bands."
        band_df = band_df.sort_values(["score"], ascending=False).drop_duplicates(
            subset=["result_type", "source_label", "provenance"], keep="first"
        )
        return band_df.head(top_n).reset_index(drop=True)

    def search_band_evidence(
        self,
        band_cm: float,
        tolerance_cm: float = 10.0,
        tier_filter: str | None = None,
    ) -> pd.DataFrame:
        rows: list[dict] = []

        tier1_ref_df = self.reference_peaks_df[
            (self.reference_peaks_df["peak_cm"] >= band_cm - tolerance_cm)
            & (self.reference_peaks_df["peak_cm"] <= band_cm + tolerance_cm)
        ]
        for row in tier1_ref_df.to_dict(orient="records"):
            score = float(row["rel_intensity"]) / (1.0 + abs(float(row["peak_cm"]) - band_cm))
            rows.append(
                {
                    "query_band_cm": band_cm,
                    "mode": "band_centered_search",
                    "evidence_tier": "tier1_direct_spectral_grounding",
                    "result_type": "ramanbiolib_peak_match",
                    "source_dataset_id": "ramanbiolib",
                    "source_family": row["biochemical_class"],
                    "source_label": row["component"],
                    "score": score,
                    "matched_band_cm": float(row["peak_cm"]),
                    "provenance": row["reference"],
                    "notes": "Reference peak within tolerance window.",
                }
            )

        for row in self.grounding_summary_df.to_dict(orient="records"):
            x_values = _parse_json_array(row["mean_wavenumbers_json"])
            y_values = _parse_json_array(row["mean_intensity_json"])
            if band_cm < float(np.min(x_values)) or band_cm > float(np.max(x_values)):
                continue
            local_intensity = float(np.interp(band_cm, x_values, y_values))
            rows.append(
                {
                    "query_band_cm": band_cm,
                    "mode": "band_centered_search",
                    "evidence_tier": "tier1_direct_spectral_grounding",
                    "result_type": "study_matched_grounding_band_support",
                    "source_dataset_id": row["dataset_id"],
                    "source_family": row["experiment_family"],
                    "source_label": row["class_label"],
                    "score": local_intensity,
                    "matched_band_cm": band_cm,
                    "provenance": row["source_file"],
                    "notes": "Interpolated processed grounding mean intensity at query band.",
                }
            )

        for row in self.support_chunk_df.to_dict(orient="records"):
            ranges = _extract_band_ranges(str(row["chunk_text"]))
            matching_ranges = [
                (range_min, range_max)
                for range_min, range_max in ranges
                if range_min <= band_cm <= range_max
            ]
            if matching_ranges:
                range_min, range_max = matching_ranges[0]
                midpoint = (range_min + range_max) / 2.0
                score = 1.0 / (1.0 + abs(band_cm - midpoint))
                rows.append(
                    {
                        "query_band_cm": band_cm,
                        "mode": "band_centered_search",
                        "evidence_tier": row["evidence_tier"],
                        "result_type": "literature_chunk_band_range",
                        "source_dataset_id": row["dataset_id"],
                        "source_family": row["section"],
                        "source_label": row["citation_label"],
                        "score": score,
                        "matched_band_cm": midpoint,
                        "provenance": row["source_file"],
                        "notes": str(row["chunk_text"])[:320],
                    }
                )

        for row in self.support_spectra_df.to_dict(orient="records"):
            x_values = _parse_json_array(row["wavenumbers_json"])
            y_values = _parse_json_array(row["intensity_json"])
            x_min = float(np.min(x_values))
            x_max = float(np.max(x_values))
            if not (x_min <= band_cm <= x_max):
                continue
            normalized_y = _normalize_vector(y_values)
            local_intensity = float(np.interp(band_cm, x_values, normalized_y))
            rows.append(
                {
                    "query_band_cm": band_cm,
                    "mode": "band_centered_search",
                    "evidence_tier": "tier2_literature_support",
                    "result_type": "digitized_support_spectrum",
                    "source_dataset_id": row["dataset_id"],
                    "source_family": row["evidence_family"],
                    "source_label": row["citation_label"],
                    "score": local_intensity,
                    "matched_band_cm": band_cm,
                    "provenance": row["source_file"],
                    "notes": "Digitized support-only literature trace. Not a primary matching target.",
                }
            )

        result_df = pd.DataFrame(rows)
        if result_df.empty:
            return result_df
        if tier_filter is not None:
            result_df = result_df[result_df["evidence_tier"] == tier_filter].copy()
        return result_df.sort_values("score", ascending=False).reset_index(drop=True)

