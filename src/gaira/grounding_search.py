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


def _tokenize_text(value: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_\+\-]+", value)
        if len(token) >= 3
    }


def _tokenize_labels(values: list[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        tokens |= _tokenize_text(text)
        tokens.add(text.lower())
    return tokens


def _expand_compound_identifier_tokens(value: str) -> set[str]:
    text = str(value).strip().lower()
    if not text:
        return set()
    parts = [part for part in re.split(r"[_\-/\s]+", text) if len(part) >= 2]
    return set(parts)


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
                  g.processing_version,
                  g.experiment_family,
                  g.class_label,
                  g.n_spectra,
                  g.mean_wavenumbers_json,
                  g.mean_intensity_json,
                  MIN(m.source_file) AS source_file,
                  MIN(m.grounding_role) AS grounding_role
                FROM grounding_class_summary g
                JOIN grounding_metadata m
                  ON g.dataset_id = m.dataset_id
                 AND g.experiment_family = m.experiment_family
                 AND g.class_label = m.class_label
                GROUP BY
                  g.summary_id, g.dataset_id, g.processing_version, g.experiment_family, g.class_label, g.n_spectra,
                  g.mean_wavenumbers_json, g.mean_intensity_json
                """
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
                  d.source_dataset_id,
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
                WHERE d.use_for_rag = 'yes'
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
                  s.source_dataset_id,
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
                WHERE s.use_for_rag = 'yes'
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

            self.knowledge_chunk_df = connection.execute(
                """
                SELECT
                  chunk_id,
                  source_id,
                  dataset_id,
                  section,
                  chunk_text,
                  chunk_order,
                  page_label,
                  metadata_json
                FROM knowledge_chunks
                WHERE dataset_id = 'raman_knowledge_core'
                ORDER BY source_id, chunk_order
                """
            ).fetchdf()

            self.peak_assignment_df = connection.execute(
                """
                SELECT
                  assignment_id,
                  source_id,
                  dataset_id,
                  peak_cm,
                  tolerance_cm,
                  assigned_molecule,
                  assigned_group,
                  matrix_context,
                  confidence_text,
                  evidence_text
                FROM peak_assignments
                WHERE dataset_id = 'raman_knowledge_core'
                ORDER BY peak_cm
                """
            ).fetchdf()

            self.biomarker_claim_df = connection.execute(
                """
                SELECT
                  claim_id,
                  source_id,
                  dataset_id,
                  biomarker_name,
                  disease_context,
                  sample_type,
                  spectral_region,
                  claim_text,
                  evidence_strength,
                  notes
                FROM biomarker_claims
                WHERE dataset_id = 'raman_knowledge_core'
                ORDER BY claim_id
                """
            ).fetchdf()

            self.confounder_note_df = connection.execute(
                """
                SELECT
                  confounder_id,
                  source_id,
                  dataset_id,
                  confounder_name,
                  applies_to,
                  note_text,
                  mitigation_text
                FROM confounder_notes
                WHERE dataset_id = 'raman_knowledge_core'
                ORDER BY confounder_id
                """
            ).fetchdf()

            self.semantic_region_df = connection.execute(
                """
                SELECT
                  region_id,
                  dataset_id,
                  region_label,
                  region_min_cm,
                  region_max_cm,
                  dominant_group,
                  secondary_groups,
                  typical_examples,
                  interpretation_note,
                  caution_note
                FROM semantic_regions
                WHERE dataset_id = 'raman_knowledge_core'
                ORDER BY region_min_cm
                """
            ).fetchdf()

            self.dataset_context_df = connection.execute(
                """
                SELECT
                  context_id,
                  dataset_id,
                  target_dataset_id,
                  modality,
                  sample_type,
                  measurement_state,
                  substrate_type,
                  enhancement_mode,
                  known_biases,
                  region_caution_450_700,
                  region_caution_700_900,
                  region_caution_900_1100,
                  region_caution_1100_1300,
                  region_caution_1300_1500,
                  region_caution_1500_1700,
                  interpretation_note,
                  do_not_overclaim_note
                FROM dataset_context
                WHERE dataset_id = 'raman_knowledge_core'
                """
            ).fetchdf()

    def _build_support_tokens(
        self,
        query: SpectrumQuery,
        seed_labels: list[str] | None = None,
        domain: str | None = None,
    ) -> set[str]:
        tokens = _tokenize_labels(
            [
                query.query_label,
                query.query_family,
                query.source_dataset_id,
                *(seed_labels or []),
            ]
        )
        tokens |= _expand_compound_identifier_tokens(query.query_label)
        tokens |= _expand_compound_identifier_tokens(query.query_family)
        tokens |= _expand_compound_identifier_tokens(query.source_dataset_id)
        for label in seed_labels or []:
            tokens |= _expand_compound_identifier_tokens(label)
        if domain == "ev":
            tokens |= {
                "ev",
                "extracellular",
                "vesicles",
                "probe1",
                "probe2",
                "substrate",
                "diabetes",
                "impact",
                "strongd",
                "normal-weight",
                "overweight",
                "subgroup",
                "heterogeneity",
                "insulin",
                "mitochondrial",
                "shine",
                "spectra",
                "apap",
                "hepatotoxicity",
                "injury",
                "dose-response",
                "day0",
                "day2",
            }
        elif domain == "serum":
            tokens |= {"serum", "ag", "colloid", "adsorption", "protocol", "batch"}
        return tokens

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
            if row["dataset_id"] == "serum_ag_colloids_grounding":
                result_type = "study_matched_sers_grounding"
            elif row["dataset_id"] == "adenine_sers_control":
                result_type = "controlled_analyte_grounding"
            else:
                result_type = "controlled_grounding_reference"
            rows.append(
                {
                    "query_id": query.query_id,
                    "query_label": query.query_label,
                    "query_family": query.query_family,
                    "mode": "spectrum_to_grounding",
                    "evidence_tier": "tier1_direct_spectral_grounding",
                    "result_type": result_type,
                    "source_dataset_id": row["dataset_id"],
                    "source_family": row["experiment_family"],
                    "source_label": row["class_label"],
                    "score": score,
                    "provenance": row["source_file"],
                    "notes": (
                        f"Processed class summary, n_spectra={int(row['n_spectra'])}, "
                        f"grounding_role={row['grounding_role']}."
                    ),
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
        seed_labels: list[str] | None = None,
        domain: str | None = None,
        top_n: int = 8,
    ) -> pd.DataFrame:
        trigger_bands = _detect_query_bands(query.x, query.y, top_n=5)
        support_tokens = self._build_support_tokens(query, seed_labels=seed_labels, domain=domain)
        rows: list[dict] = []

        for band_cm in trigger_bands:
            band_rows = self.search_band_evidence(band_cm=band_cm, tier_filter="tier2_literature_support")
            if not band_rows.empty:
                rows.extend(band_rows.head(4).to_dict(orient="records"))

        for row in self.support_chunk_df.to_dict(orient="records"):
            chunk_text = str(row["chunk_text"])
            title = str(row.get("title", ""))
            combined_text = " ".join([chunk_text, title, str(row.get("citation_label", ""))])
            chunk_tokens = _tokenize_text(combined_text)
            token_overlap = sorted(support_tokens & chunk_tokens)

            band_matches: list[str] = []
            band_score = 0.0
            for band_cm in trigger_bands:
                for range_min, range_max in _extract_band_ranges(chunk_text):
                    if range_min <= band_cm <= range_max:
                        midpoint = (range_min + range_max) / 2.0
                        band_score += 1.0 / (1.0 + abs(band_cm - midpoint))
                        if range_min == range_max:
                            band_matches.append(str(int(round(range_min))))
                        else:
                            band_matches.append(f"{int(round(range_min))}-{int(round(range_max))}")
                        break

            token_score = float(len(token_overlap))
            score = band_score + 0.35 * token_score
            if score <= 0:
                continue

            matched_tokens = sorted(set(band_matches + token_overlap[:8]))
            rows.append(
                {
                    "query_id": query.query_id,
                    "query_label": query.query_label,
                    "query_family": query.query_family,
                    "mode": "spectrum_to_grounding",
                    "evidence_tier": row["evidence_tier"],
                    "result_type": (
                        "support_document_match"
                        if row["dataset_id"] != "serum_ag_colloids_literature_grounding"
                        else "literature_chunk_support"
                    ),
                    "source_dataset_id": row["dataset_id"],
                    "target_dataset_id": row["source_dataset_id"],
                    "source_family": row["evidence_family"],
                    "source_label": row["citation_label"] or row["title"],
                    "score": score,
                    "matched_band_cm": None,
                    "provenance": row["source_file"],
                    "notes": (
                        f"{chunk_text[:320]} "
                        f"(matched: {', '.join(matched_tokens) if matched_tokens else 'bands/tokens'})"
                    ),
                }
            )

        for row in self.support_spectra_df.to_dict(orient="records"):
            x_values = _parse_json_array(row["wavenumbers_json"])
            y_values = _parse_json_array(row["intensity_json"])
            normalized_y = _normalize_vector(y_values)
            local_scores = []
            for band_cm in trigger_bands:
                if float(np.min(x_values)) <= band_cm <= float(np.max(x_values)):
                    local_scores.append(float(np.interp(band_cm, x_values, normalized_y)))
            if not local_scores:
                continue
            rows.append(
                {
                    "query_id": query.query_id,
                    "query_label": query.query_label,
                    "query_family": query.query_family,
                    "mode": "spectrum_to_grounding",
                    "evidence_tier": "tier2_literature_support",
                    "result_type": "digitized_support_spectrum",
                    "source_dataset_id": row["dataset_id"],
                    "target_dataset_id": row["source_dataset_id"],
                    "source_family": row["evidence_family"],
                    "source_label": row["citation_label"],
                    "score": float(max(local_scores)),
                    "matched_band_cm": None,
                    "provenance": row["source_file"],
                    "notes": "Digitized support-only literature trace. Not a primary matching target.",
                }
            )

        support_df = pd.DataFrame(rows)
        if support_df.empty:
            return support_df
        support_df["target_match_weight"] = support_df["target_dataset_id"].fillna("").astype(str).apply(
            lambda value: 1.0 if query.source_dataset_id and query.source_dataset_id in value else 0.0
        )
        support_df = support_df.sort_values(["target_match_weight", "score"], ascending=[False, False]).drop_duplicates(
            subset=["result_type", "source_dataset_id", "source_label", "provenance"],
            keep="first",
        )
        selected_rows: list[dict] = []
        selected_keys: set[tuple[str, str, str]] = set()
        reserve_slots = min(2, max(top_n - 1, 0))
        base_limit = max(top_n - reserve_slots, 0)
        for row in support_df.head(base_limit).to_dict(orient="records"):
            key = (str(row["result_type"]), str(row["source_dataset_id"]), str(row["source_label"]))
            selected_rows.append(row)
            selected_keys.add(key)

        support_only_df = support_df[
            (support_df["result_type"] == "support_document_match")
            & (~support_df["source_dataset_id"].isin(["serum_ag_colloids_literature_grounding"]))
            & (support_df["score"] >= 0.35)
        ]
        for row in support_only_df.to_dict(orient="records"):
            if len(selected_rows) >= top_n:
                break
            key = (str(row["result_type"]), str(row["source_dataset_id"]), str(row["source_label"]))
            if key in selected_keys:
                continue
            selected_rows.append(row)
            selected_keys.add(key)

        if len(selected_rows) < top_n:
            for row in support_df.to_dict(orient="records"):
                if len(selected_rows) >= top_n:
                    break
                key = (str(row["result_type"]), str(row["source_dataset_id"]), str(row["source_label"]))
                if key in selected_keys:
                    continue
                selected_rows.append(row)
                selected_keys.add(key)

        return pd.DataFrame(selected_rows).reset_index(drop=True)

    def search_knowledge_support(
        self,
        query: SpectrumQuery,
        seed_labels: list[str] | None = None,
        domain: str | None = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        trigger_bands = _detect_query_bands(query.x, query.y, top_n=5)
        support_tokens = self._build_support_tokens(query, seed_labels=seed_labels, domain=domain)
        rows: list[dict] = []
        matched_groups: set[str] = set()

        for row in self.peak_assignment_df.to_dict(orient="records"):
            peak_cm = float(row["peak_cm"])
            tolerance_cm = float(row["tolerance_cm"])
            best_distance = min(abs(band_cm - peak_cm) for band_cm in trigger_bands)
            if best_distance > tolerance_cm:
                continue
            matched_groups.add(str(row["assigned_group"]))
            rows.append(
                {
                    "query_id": query.query_id,
                    "query_label": query.query_label,
                    "query_family": query.query_family,
                    "mode": "spectrum_to_grounding",
                    "evidence_tier": "tier2_knowledge_support",
                    "result_type": "knowledge_peak_assignment",
                    "source_dataset_id": row["dataset_id"],
                    "source_family": row["assigned_group"],
                    "source_label": row["assigned_molecule"],
                    "score": 1.0 / (1.0 + best_distance),
                    "matched_band_cm": peak_cm,
                    "provenance": row["source_id"],
                    "notes": (
                        f"matrix={row['matrix_context']}; confidence={row['confidence_text']}; "
                        f"{row['evidence_text']}"
                    ),
                }
            )

        for row in self.semantic_region_df.to_dict(orient="records"):
            overlaps = [
                band_cm
                for band_cm in trigger_bands
                if float(row["region_min_cm"]) <= band_cm <= float(row["region_max_cm"])
            ]
            if not overlaps:
                continue
            matched_groups.add(str(row["dominant_group"]))
            rows.append(
                {
                    "query_id": query.query_id,
                    "query_label": query.query_label,
                    "query_family": query.query_family,
                    "mode": "spectrum_to_grounding",
                    "evidence_tier": "tier2_knowledge_support",
                    "result_type": "semantic_region_support",
                    "source_dataset_id": row["dataset_id"],
                    "source_family": row["dominant_group"],
                    "source_label": row["region_label"],
                    "score": float(len(overlaps)),
                    "matched_band_cm": float(np.mean(overlaps)),
                    "provenance": row["region_id"],
                    "notes": (
                        f"interpretation={row['interpretation_note']}; "
                        f"caution={row['caution_note']}"
                    ),
                }
            )

        extended_tokens = set(support_tokens)
        extended_tokens |= _tokenize_labels(sorted(matched_groups))
        for row in self.knowledge_chunk_df.to_dict(orient="records"):
            chunk_text = str(row["chunk_text"])
            overlap = sorted(extended_tokens & _tokenize_text(chunk_text))
            if not overlap:
                continue
            rows.append(
                {
                    "query_id": query.query_id,
                    "query_label": query.query_label,
                    "query_family": query.query_family,
                    "mode": "spectrum_to_grounding",
                    "evidence_tier": "tier2_knowledge_support",
                    "result_type": "knowledge_chunk_support",
                    "source_dataset_id": row["dataset_id"],
                    "source_family": row["section"],
                    "source_label": row["section"],
                    "score": float(len(overlap)),
                    "matched_band_cm": None,
                    "provenance": row["source_id"],
                    "notes": f"{chunk_text[:320]} (matched: {', '.join(overlap[:8])})",
                }
            )

        for row in self.confounder_note_df.to_dict(orient="records"):
            note_text = " ".join([str(row["confounder_name"]), str(row["applies_to"]), str(row["note_text"])])
            overlap = sorted(extended_tokens & _tokenize_text(note_text))
            domain_match = domain is not None and domain in str(row["applies_to"]).lower()
            if not overlap and not domain_match:
                continue
            rows.append(
                {
                    "query_id": query.query_id,
                    "query_label": query.query_label,
                    "query_family": query.query_family,
                    "mode": "spectrum_to_grounding",
                    "evidence_tier": "tier2_knowledge_support",
                    "result_type": "confounder_note_support",
                    "source_dataset_id": row["dataset_id"],
                    "source_family": row["applies_to"],
                    "source_label": row["confounder_name"],
                    "score": float(len(overlap)) + (1.0 if domain_match else 0.0),
                    "matched_band_cm": None,
                    "provenance": row["source_id"],
                    "notes": f"{row['note_text']} Mitigation: {row['mitigation_text']}",
                }
            )

        for row in self.biomarker_claim_df.to_dict(orient="records"):
            claim_text = " ".join(
                [str(row["biomarker_name"]), str(row["sample_type"]), str(row["claim_text"]), str(row["disease_context"])]
            )
            overlap = sorted(extended_tokens & _tokenize_text(claim_text))
            domain_match = domain is not None and domain in str(row["sample_type"]).lower()
            if not overlap and not domain_match:
                continue
            rows.append(
                {
                    "query_id": query.query_id,
                    "query_label": query.query_label,
                    "query_family": query.query_family,
                    "mode": "spectrum_to_grounding",
                    "evidence_tier": "tier2_knowledge_support",
                    "result_type": "biomarker_claim_support",
                    "source_dataset_id": row["dataset_id"],
                    "source_family": row["sample_type"],
                    "source_label": row["biomarker_name"],
                    "score": float(len(overlap)) + (0.5 if domain_match else 0.0),
                    "matched_band_cm": None,
                    "provenance": row["source_id"],
                    "notes": (
                        f"evidence_strength={row['evidence_strength']}; "
                        f"{row['claim_text']}"
                    ),
                }
            )

        for row in self.dataset_context_df.to_dict(orient="records"):
            if str(row["target_dataset_id"]) != str(query.source_dataset_id):
                continue
            notes = " ".join(
                [
                    str(row["known_biases"]),
                    str(row["interpretation_note"]),
                    str(row["do_not_overclaim_note"]),
                ]
            )
            rows.append(
                {
                    "query_id": query.query_id,
                    "query_label": query.query_label,
                    "query_family": query.query_family,
                    "mode": "spectrum_to_grounding",
                    "evidence_tier": "tier2_knowledge_support",
                    "result_type": "dataset_context_support",
                    "source_dataset_id": row["dataset_id"],
                    "source_family": row["sample_type"],
                    "source_label": str(row["target_dataset_id"]),
                    "score": 2.0,
                    "matched_band_cm": None,
                    "provenance": row["context_id"],
                    "notes": notes,
                }
            )

        knowledge_df = pd.DataFrame(rows)
        if knowledge_df.empty:
            return knowledge_df
        knowledge_df = knowledge_df.sort_values(["score"], ascending=False).drop_duplicates(
            subset=["result_type", "source_label", "provenance"],
            keep="first",
        )
        return knowledge_df.head(top_n).reset_index(drop=True)

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
            if row["dataset_id"] == "serum_ag_colloids_grounding":
                result_type = "study_matched_grounding_band_support"
            elif row["dataset_id"] == "adenine_sers_control":
                result_type = "controlled_analyte_band_support"
            else:
                result_type = "controlled_grounding_band_support"
            rows.append(
                {
                    "query_band_cm": band_cm,
                    "mode": "band_centered_search",
                    "evidence_tier": "tier1_direct_spectral_grounding",
                    "result_type": result_type,
                    "source_dataset_id": row["dataset_id"],
                    "source_family": row["experiment_family"],
                    "source_label": row["class_label"],
                    "score": local_intensity,
                    "matched_band_cm": band_cm,
                    "provenance": row["source_file"],
                    "notes": (
                        "Interpolated processed grounding mean intensity at query band. "
                        f"grounding_role={row['grounding_role']}."
                    ),
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
