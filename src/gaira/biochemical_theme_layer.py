from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from gaira.grounding_search import GroundingSearchEngine, SpectrumQuery
from gaira.theme_ontology import CAUTION_THEMES, EVIDENCE_WEIGHTS, POSITIVE_THEMES, ThemeDefinition, get_theme_definitions


@dataclass
class ThemeLayerInput:
    domain: str
    query_id: str
    query_label: str
    query_family: str
    source_dataset_id: str
    spectrum_query: SpectrumQuery
    tier1_hits: list[dict]
    tier2_hits: list[dict]
    knowledge_hits: list[dict]
    semantic_hits: list[dict]
    context_hits: list[dict]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compact_hit(hit: dict, score_key: str = "reranked_score") -> dict:
    return {
        "source_dataset_id": hit.get("source_dataset_id"),
        "source_label": hit.get("source_label"),
        "result_type": hit.get("result_type"),
        "score": round(_safe_float(hit.get(score_key, hit.get("score"))), 4),
    }


def _band_token_text(value: float) -> str:
    return str(int(round(value)))


class BiochemicalThemeLayer:
    def __init__(self, db_path: Path, version: str = "v1") -> None:
        if version not in {"v1", "v2"}:
            raise ValueError(f"Unsupported theme-layer version '{version}'.")
        self.db_path = Path(db_path)
        self.version = version
        self.grounding_engine = GroundingSearchEngine(db_path=db_path)
        self.theme_defs = get_theme_definitions()
        self.dataset_context_df = self._load_dataset_context()

    def _load_dataset_context(self) -> pd.DataFrame:
        with duckdb.connect(str(self.db_path), read_only=True) as connection:
            return connection.execute(
                """
                SELECT
                  dataset_id,
                  biosample_type,
                  measurement_mode,
                  default_substrate_type,
                  default_preprocessing_family,
                  notes
                FROM dataset_domain_context
                ORDER BY dataset_id
                """
            ).fetchdf()

    def _dataset_mode(self, dataset_id: str) -> str:
        match_df = self.dataset_context_df[self.dataset_context_df["dataset_id"] == dataset_id]
        if not match_df.empty:
            return str(match_df.iloc[0]["measurement_mode"]).strip()
        if dataset_id == "ramanbiolib":
            return "Raman"
        if dataset_id in {
            "serum_ag_colloids_grounding",
            "serum_ag_colloids_literature_grounding",
            "adenine_sers_control",
        }:
            return "SERS"
        if dataset_id in {"sers_fingerprint_workingpaper_support", "sers24_metabolite_support"}:
            return "support_only"
        if dataset_id.endswith("_grounding"):
            return "SERS"
        return "unknown"

    def _build_input_from_inference(self, request, inference_result: dict) -> ThemeLayerInput:
        return ThemeLayerInput(
            domain=request.domain,
            query_id=request.query_id,
            query_label=request.query_label,
            query_family=request.query_family,
            source_dataset_id=request.source_dataset_id,
            spectrum_query=request.spectrum_query,
            tier1_hits=inference_result.get("tier1_grounding_hits", []),
            tier2_hits=inference_result.get("tier2_support_hits", []),
            knowledge_hits=inference_result.get("knowledge_support_hits", []),
            semantic_hits=inference_result.get("semantic_region_support_hits", []),
            context_hits=inference_result.get("domain_context_hits", []),
        )

    def build_from_inference(self, request, inference_result: dict) -> dict:
        return self.build_from_input(self._build_input_from_inference(request, inference_result))

    def build_from_input(self, theme_input: ThemeLayerInput) -> dict:
        query_bands = self._detect_query_bands(theme_input.spectrum_query.x, theme_input.spectrum_query.y)
        band_df = self._collect_band_support(query_bands)

        raw_scores: dict[str, float] = {}
        evidence_cache: dict[str, dict] = {}
        for theme_name, definition in self.theme_defs.items():
            raw_score, evidence = self._score_theme(definition, theme_input, query_bands, band_df)
            raw_scores[theme_name] = raw_score
            evidence_cache[theme_name] = evidence

        caution_outputs = self._build_caution_outputs(raw_scores, evidence_cache, query_bands)
        caution_lookup = {row["theme_name"]: row["score"] for row in caution_outputs}
        global_caution_load = float(np.mean(list(caution_lookup.values()))) if caution_lookup else 0.0

        if self.version == "v2":
            positive_outputs = self._build_positive_outputs_v2(raw_scores, evidence_cache, query_bands, caution_lookup, theme_input)
        else:
            positive_outputs = self._build_positive_outputs_v1(raw_scores, evidence_cache, query_bands, caution_lookup)

        positive_outputs.sort(key=lambda row: row["score"], reverse=True)
        caution_outputs.sort(key=lambda row: row["score"], reverse=True)

        dominant_themes = [row["theme_name"] for row in positive_outputs[:3] if row["score"] >= 0.20]
        global_caveats = [row["theme_name"] for row in caution_outputs if row["score"] >= 0.25][:4]

        evidence_profile_summary = self._build_evidence_profile_summary(
            theme_input=theme_input,
            positive_outputs=positive_outputs,
            caution_outputs=caution_outputs,
            query_bands=query_bands,
            global_caution_load=global_caution_load,
        )
        what_not_to_claim = self._build_what_not_to_claim(theme_input, caution_outputs)
        biochemical_theme_summary = self._build_theme_summary(positive_outputs, caution_outputs)

        return {
            "biochemical_theme_layer_version": self.version,
            "biochemical_theme_outputs": positive_outputs + caution_outputs,
            "dominant_themes": dominant_themes,
            "biochemical_global_caveats": global_caveats,
            "evidence_profile_summary": evidence_profile_summary,
            "biochemical_what_not_to_claim": what_not_to_claim,
            "biochemical_theme_summary": biochemical_theme_summary,
            "query_bands_cm": query_bands,
        }

    def _detect_query_bands(self, x_values: np.ndarray, y_values: np.ndarray, top_n: int = 6) -> list[float]:
        shifted = np.asarray(y_values, dtype=float) - float(np.min(y_values))
        norm = float(np.linalg.norm(shifted))
        normalized = shifted / norm if norm > 0 else shifted
        peak_indices, properties = find_peaks(normalized, prominence=0.01, distance=5)
        if len(peak_indices) == 0:
            top_indices = np.argsort(normalized)[-top_n:]
            return sorted(float(x_values[index]) for index in top_indices)
        prominences = properties.get("prominences", np.ones(len(peak_indices)))
        ranked_indices = peak_indices[np.argsort(prominences)[::-1][:top_n]]
        return sorted(float(x_values[index]) for index in ranked_indices)

    def _collect_band_support(self, query_bands: list[float]) -> pd.DataFrame:
        frames = []
        for band_cm in query_bands:
            band_df = self.grounding_engine.search_band_evidence(band_cm=band_cm, tolerance_cm=12.0)
            if not band_df.empty:
                band_df = band_df.copy()
                band_df["query_band_cm"] = band_cm
                frames.append(band_df.head(12))
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        return combined.drop_duplicates(subset=["source_dataset_id", "source_label", "result_type", "query_band_cm"])

    def _normalize_hit_score(self, evidence_type: str, hit: dict) -> float:
        value = _safe_float(hit.get("reranked_score", hit.get("score")))
        scale = {
            "tier1": 1.0,
            "tier2": 2.0,
            "knowledge": 3.0,
            "semantic": 2.5,
            "context": 6.0,
            "band": 0.35,
        }[evidence_type]
        return min(1.0, max(0.0, value / scale))

    def _hit_text(self, hit: dict) -> str:
        return " ".join(
            str(hit.get(field, ""))
            for field in [
                "source_dataset_id",
                "source_family",
                "source_label",
                "result_type",
                "notes",
                "document_id",
                "section",
                "matched_tokens",
            ]
        ).lower()

    def _keyword_overlap(self, text: str, theme: ThemeDefinition) -> list[str]:
        return [keyword for keyword in theme.keywords if keyword in text]

    def _negative_overlap(self, text: str, theme: ThemeDefinition) -> list[str]:
        return [keyword for keyword in theme.negative_keywords if keyword in text]

    def _positive_theme_overlap_map(self, text: str) -> dict[str, list[str]]:
        overlap_map: dict[str, list[str]] = {}
        for theme_name in POSITIVE_THEMES:
            overlap = self._keyword_overlap(text, self.theme_defs[theme_name])
            if overlap:
                overlap_map[theme_name] = overlap
        return overlap_map

    def _theme_band_overlap(self, query_bands: list[float], theme: ThemeDefinition, tolerance: float = 18.0) -> list[float]:
        matched: list[float] = []
        for band_cm in query_bands:
            for anchor in theme.anchor_bands_cm:
                if abs(band_cm - anchor) <= tolerance:
                    matched.append(band_cm)
                    break
        return sorted(set(matched))

    def _source_multiplier(self, evidence_type: str, hit: dict, definition: ThemeDefinition, theme_input: ThemeLayerInput) -> float:
        dataset_id = str(hit.get("source_dataset_id", ""))
        multiplier = 1.0
        if self.version != "v2":
            return multiplier
        if theme_input.domain != "grounding" and dataset_id == "adenine_sers_control":
            if definition.theme_name == "nucleic_acid_purine_associated":
                multiplier *= 0.45
            else:
                multiplier *= 0.12
        if theme_input.domain in {"serum", "ev"} and dataset_id == "ramanbiolib":
            multiplier *= 0.82
        if theme_input.domain == "ev" and dataset_id == "serum_ag_colloids_grounding":
            multiplier *= 0.55
        if theme_input.domain == "ev" and dataset_id == "serum_ag_colloids_literature_grounding":
            multiplier *= 0.65
        if theme_input.domain == "serum" and evidence_type == "context" and dataset_id == "covid_serum_raman":
            multiplier *= 0.90
        return multiplier

    def _score_hit(self, evidence_type: str, hit: dict, theme: ThemeDefinition, theme_input: ThemeLayerInput) -> tuple[float, list[str], list[str]]:
        text = self._hit_text(hit)
        overlap = self._keyword_overlap(text, theme)
        negative_overlap = self._negative_overlap(text, theme)
        if not overlap:
            return 0.0, [], negative_overlap

        relevance = min(1.0, 0.42 + 0.12 * len(overlap))
        base_score = EVIDENCE_WEIGHTS[evidence_type] * relevance * self._normalize_hit_score(evidence_type, hit)

        if self.version == "v2" and theme.category == "positive":
            overlap_map = self._positive_theme_overlap_map(text)
            total_overlap_weight = sum(len(values) for values in overlap_map.values())
            if total_overlap_weight > 0:
                share = len(overlap) / total_overlap_weight
                base_score *= share

        base_score *= self._source_multiplier(evidence_type, hit, theme, theme_input)
        return base_score, overlap, negative_overlap

    def _score_theme(
        self,
        definition: ThemeDefinition,
        theme_input: ThemeLayerInput,
        query_bands: list[float],
        band_df: pd.DataFrame,
    ) -> tuple[float, dict]:
        evidence = {
            "tier1_contrib": 0.0,
            "tier2_contrib": 0.0,
            "knowledge_contrib": 0.0,
            "semantic_contrib": 0.0,
            "context_contrib": 0.0,
            "band_contrib": 0.0,
            "supporting_tier1_hits": [],
            "supporting_tier2_hits": [],
            "supporting_knowledge_hits": [],
            "supporting_semantic_regions": [],
            "supporting_context_hits": [],
            "supporting_bands": [],
            "limiting_hits": [],
            "negative_evidence": 0.0,
            "calibration_signal": 0.0,
            "broad_analog_signal": 0.0,
            "support_only_signal": 0.0,
            "matrix_signal": 0.0,
            "matched_keyword_count": 0,
            "raw_score_pre_normalization": 0.0,
            "normalized_score": 0.0,
            "competition_penalty": 0.0,
            "caution_penalty": 0.0,
            "calibration_penalty": 0.0,
            "specificity_index": 0.0,
        }

        for evidence_type, hits, support_key in [
            ("tier1", theme_input.tier1_hits, "supporting_tier1_hits"),
            ("tier2", theme_input.tier2_hits, "supporting_tier2_hits"),
            ("knowledge", theme_input.knowledge_hits, "supporting_knowledge_hits"),
            ("semantic", theme_input.semantic_hits, "supporting_semantic_regions"),
            ("context", theme_input.context_hits, "supporting_context_hits"),
        ]:
            for hit in hits:
                score, overlap, negative_overlap = self._score_hit(evidence_type, hit, definition, theme_input)
                if negative_overlap:
                    evidence["negative_evidence"] += min(0.25, 0.06 * len(negative_overlap))
                if score <= 0:
                    continue

                contrib_key = f"{evidence_type}_contrib"
                evidence[contrib_key] += score
                evidence["matched_keyword_count"] += len(overlap)
                source_dataset_id = str(hit.get("source_dataset_id", ""))
                if source_dataset_id == "adenine_sers_control":
                    evidence["calibration_signal"] += score
                if source_dataset_id == "ramanbiolib":
                    evidence["broad_analog_signal"] += score
                if str(hit.get("result_type", "")) == "support_document_match":
                    evidence["support_only_signal"] += score
                if any(token in self._hit_text(hit) for token in ["matrix", "metabolite dominance", "protein dominance", "serum-local"]):
                    evidence["matrix_signal"] += 0.15

                if evidence_type == "context":
                    evidence[support_key].append(
                        {
                            "document_id": hit.get("document_id"),
                            "section": hit.get("section"),
                            "score": round(_safe_float(hit.get("score")), 4),
                            "matched_keywords": overlap,
                        }
                    )
                else:
                    evidence[support_key].append({**_compact_hit(hit), "matched_keywords": overlap})

        if not band_df.empty and definition.anchor_bands_cm:
            for hit in band_df.to_dict(orient="records"):
                matched = [anchor for anchor in definition.anchor_bands_cm if abs(float(hit["query_band_cm"]) - anchor) <= 18.0]
                if not matched:
                    continue
                band_score = self._normalize_hit_score("band", hit)
                if self.version == "v2" and definition.category == "positive":
                    text = self._hit_text(hit)
                    overlap_map = self._positive_theme_overlap_map(text)
                    total_overlap_weight = sum(len(values) for values in overlap_map.values()) or 1
                    theme_share = len(overlap_map.get(definition.theme_name, [])) / total_overlap_weight if overlap_map else 0.5
                    band_score *= max(0.20, theme_share)
                score = EVIDENCE_WEIGHTS["band"] * band_score
                if self.version == "v2" and theme_input.domain != "grounding" and str(hit.get("source_dataset_id")) == "adenine_sers_control":
                    score *= 0.25 if definition.theme_name != "nucleic_acid_purine_associated" else 0.55
                evidence["band_contrib"] += score
                evidence["supporting_bands"].append(
                    {
                        "query_band_cm": round(float(hit["query_band_cm"]), 1),
                        "source_label": hit.get("source_label"),
                        "result_type": hit.get("result_type"),
                        "score": round(_safe_float(hit.get("score")), 4),
                    }
                )

        query_mode = self._dataset_mode(theme_input.source_dataset_id)
        top_evidence_modes = [
            self._dataset_mode(str(hit.get("source_dataset_id")))
            for hit in theme_input.tier1_hits[:6] + theme_input.tier2_hits[:6]
        ]
        mismatch_fraction = (
            sum(1 for mode in top_evidence_modes if mode not in {"unknown", "support_only"} and mode != query_mode) / len(top_evidence_modes)
            if top_evidence_modes
            else 0.0
        )

        if definition.theme_name == "modality_mismatch_caution" and query_mode != "unknown" and mismatch_fraction > 0:
            evidence["context_contrib"] += 0.6 * mismatch_fraction
            evidence["limiting_hits"].append(f"query_mode={query_mode}; evidence_mismatch_fraction={mismatch_fraction:.2f}")

        if definition.theme_name == "weak_label_or_cohort_caution":
            if theme_input.source_dataset_id in {"diabetes_plasma_ev_sers", "covid_serum_raman"}:
                evidence["context_contrib"] += 0.6
                evidence["limiting_hits"].append(f"dataset-level cohort caution for {theme_input.source_dataset_id}")
            if "suspected" in theme_input.query_label.lower():
                evidence["context_contrib"] += 0.25
                evidence["limiting_hits"].append("query label includes suspected cohort framing")

        if definition.theme_name == "probe_substrate_caution":
            if theme_input.domain == "ev" and any(token in theme_input.query_family.lower() for token in ["probe", "normedprobe"]):
                evidence["context_contrib"] += 0.35
            if theme_input.source_dataset_id in {"serum_protocol_comparison", "cspp_serum"}:
                evidence["context_contrib"] += 0.35

        if definition.theme_name == "matrix_dominance_caution":
            if theme_input.domain == "serum":
                evidence["context_contrib"] += 0.25
            if any("metabolite dominance" in self._hit_text(hit) or "protein dominance" in self._hit_text(hit) for hit in theme_input.context_hits):
                evidence["context_contrib"] += 0.35

        if definition.theme_name == "low_specificity_caution":
            if any(hit.get("source_dataset_id") == "adenine_sers_control" for hit in theme_input.tier1_hits[:4]):
                evidence["tier1_contrib"] += 0.25
                evidence["limiting_hits"].append("controlled analyte grounding is present and should stay calibration-like")
            if any(hit.get("source_dataset_id") == "ramanbiolib" for hit in theme_input.tier1_hits[:4]):
                evidence["tier1_contrib"] += 0.20
                evidence["limiting_hits"].append("broad RamanBioLib analog support remains prominent")
            if any(hit.get("result_type") == "support_document_match" for hit in theme_input.tier2_hits[:6]):
                evidence["tier2_contrib"] += 0.15

        total_score = float(
            sum(
                evidence[key]
                for key in [
                    "tier1_contrib",
                    "tier2_contrib",
                    "knowledge_contrib",
                    "semantic_contrib",
                    "context_contrib",
                    "band_contrib",
                ]
            )
        )
        evidence["raw_score_pre_normalization"] = total_score
        return total_score, evidence

    def _squash(self, raw_score: float) -> float:
        return round(float(1.0 - math.exp(-max(raw_score, 0.0))), 4)

    def _build_caution_outputs(self, raw_scores: dict[str, float], evidence_cache: dict[str, dict], query_bands: list[float]) -> list[dict]:
        caution_outputs: list[dict] = []
        placeholder_scores = {name: raw_scores[name] for name in CAUTION_THEMES}
        for theme_name in CAUTION_THEMES:
            definition = self.theme_defs[theme_name]
            evidence = evidence_cache[theme_name]
            score = self._squash(raw_scores[theme_name])
            confidence = self._compute_confidence_v1(definition, score, evidence, placeholder_scores)
            evidence["normalized_score"] = score
            caution_outputs.append(
                self._format_theme_output(
                    definition=definition,
                    score=score,
                    confidence=confidence,
                    evidence=evidence,
                    query_bands=query_bands,
                    caution_scores=placeholder_scores,
                )
            )
        return caution_outputs

    def _build_positive_outputs_v1(
        self,
        raw_scores: dict[str, float],
        evidence_cache: dict[str, dict],
        query_bands: list[float],
        caution_scores: dict[str, float],
    ) -> list[dict]:
        outputs: list[dict] = []
        for theme_name in POSITIVE_THEMES:
            definition = self.theme_defs[theme_name]
            evidence = evidence_cache[theme_name]
            score = self._squash(raw_scores[theme_name])
            confidence = self._compute_confidence_v1(definition, score, evidence, caution_scores)
            evidence["normalized_score"] = score
            outputs.append(
                self._format_theme_output(
                    definition=definition,
                    score=score,
                    confidence=confidence,
                    evidence=evidence,
                    query_bands=query_bands,
                    caution_scores=caution_scores,
                )
            )
        return outputs

    def _build_positive_outputs_v2(
        self,
        raw_scores: dict[str, float],
        evidence_cache: dict[str, dict],
        query_bands: list[float],
        caution_scores: dict[str, float],
        theme_input: ThemeLayerInput,
    ) -> list[dict]:
        positive_raw = {name: raw_scores[name] for name in POSITIVE_THEMES}
        total_positive_raw = sum(positive_raw.values()) + 1e-9
        outputs: list[dict] = []

        normalized_scores: dict[str, float] = {}
        specificity_lookup: dict[str, float] = {}
        caution_penalty_lookup: dict[str, float] = {}
        calibration_penalty_lookup: dict[str, float] = {}
        competition_penalty_lookup: dict[str, float] = {}

        for theme_name in POSITIVE_THEMES:
            evidence = evidence_cache[theme_name]
            raw = positive_raw[theme_name]
            other_raw = total_positive_raw - raw
            relative_share = raw / total_positive_raw if total_positive_raw > 0 else 0.0
            relative_strength = raw / (raw + 0.85 * other_raw + 0.15) if raw > 0 else 0.0
            absolute_strength = 1.0 - math.exp(-0.90 * max(raw, 0.0))

            negative_penalty = min(0.75, evidence["negative_evidence"])
            broadness_penalty = min(0.45, evidence["broad_analog_signal"] / (raw + 1e-9) * 0.28 if raw > 0 else 0.0)
            competition_penalty = min(0.80, 1.0 - relative_strength)
            calibration_penalty = 0.0
            if theme_input.domain != "grounding" and raw > 0:
                calibration_penalty = min(0.70, evidence["calibration_signal"] / (raw + 1e-9) * 0.55)
            caution_penalty = min(
                0.75,
                0.24 * caution_scores.get("matrix_dominance_caution", 0.0)
                + 0.22 * caution_scores.get("probe_substrate_caution", 0.0)
                + 0.24 * caution_scores.get("modality_mismatch_caution", 0.0)
                + 0.18 * caution_scores.get("low_specificity_caution", 0.0)
                + 0.12 * caution_scores.get("weak_label_or_cohort_caution", 0.0),
            )
            specificity_index = max(
                0.0,
                min(
                    1.0,
                    (0.30 + 0.70 * relative_strength)
                    * (1.0 - 0.70 * negative_penalty)
                    * (1.0 - broadness_penalty)
                    * (1.0 - 0.80 * calibration_penalty),
                ),
            )

            normalized = absolute_strength * specificity_index * (1.0 - 0.35 * caution_penalty)
            normalized = min(1.0, max(0.0, normalized))

            evidence["competition_penalty"] = round(float(competition_penalty), 4)
            evidence["caution_penalty"] = round(float(caution_penalty), 4)
            evidence["calibration_penalty"] = round(float(calibration_penalty), 4)
            evidence["specificity_index"] = round(float(specificity_index), 4)
            evidence["normalized_score"] = round(float(normalized), 4)

            normalized_scores[theme_name] = normalized
            specificity_lookup[theme_name] = specificity_index
            caution_penalty_lookup[theme_name] = caution_penalty
            calibration_penalty_lookup[theme_name] = calibration_penalty
            competition_penalty_lookup[theme_name] = competition_penalty

        sorted_positive = sorted(normalized_scores.items(), key=lambda item: item[1], reverse=True)
        top_score = sorted_positive[0][1] if sorted_positive else 0.0
        second_score = sorted_positive[1][1] if len(sorted_positive) > 1 else 0.0

        for theme_name in POSITIVE_THEMES:
            definition = self.theme_defs[theme_name]
            evidence = evidence_cache[theme_name]
            score = round(float(normalized_scores[theme_name]), 4)
            margin = max(0.0, score - second_score) if theme_name == sorted_positive[0][0] else max(0.0, score - top_score)
            evidence_type_count = sum(
                1
                for key in ["tier1_contrib", "tier2_contrib", "knowledge_contrib", "semantic_contrib", "context_contrib", "band_contrib"]
                if evidence[key] > 0.02
            )
            agreement = min(1.0, evidence_type_count / 4.0)
            confidence = score * (
                0.35
                + 0.25 * agreement
                + 0.25 * specificity_lookup[theme_name]
                + 0.15 * min(1.0, margin * 2.5)
            )
            confidence *= 1.0 - 0.70 * caution_penalty_lookup[theme_name]
            confidence *= 1.0 - 0.60 * calibration_penalty_lookup[theme_name]
            confidence = round(float(min(1.0, max(0.0, confidence))), 4)

            outputs.append(
                self._format_theme_output(
                    definition=definition,
                    score=score,
                    confidence=confidence,
                    evidence=evidence,
                    query_bands=query_bands,
                    caution_scores=caution_scores,
                )
            )

        return outputs

    def _compute_confidence_v1(
        self,
        definition: ThemeDefinition,
        score: float,
        evidence: dict,
        caution_scores: dict[str, float],
    ) -> float:
        evidence_type_count = sum(
            1
            for key in ["tier1_contrib", "tier2_contrib", "knowledge_contrib", "semantic_contrib", "context_contrib", "band_contrib"]
            if evidence[key] > 0
        )
        diversity = min(1.0, evidence_type_count / 4.0)
        caution_penalty = 0.0
        if definition.category == "positive":
            caution_penalty = min(
                0.55,
                0.20 * caution_scores.get("low_specificity_caution", 0.0)
                + 0.20 * caution_scores.get("modality_mismatch_caution", 0.0)
                + 0.15 * caution_scores.get("probe_substrate_caution", 0.0),
            )
        confidence = score * (0.45 + 0.55 * diversity) * (1.0 - caution_penalty)
        if definition.category == "caution":
            confidence = score * (0.55 + 0.45 * diversity)
        return round(float(min(1.0, max(0.0, confidence))), 4)

    def _format_theme_output(
        self,
        definition: ThemeDefinition,
        score: float,
        confidence: float,
        evidence: dict,
        query_bands: list[float],
        caution_scores: dict[str, float],
    ) -> dict:
        evidence_balance_summary = (
            f"tier1={evidence['tier1_contrib']:.2f}, tier2={evidence['tier2_contrib']:.2f}, "
            f"knowledge={evidence['knowledge_contrib']:.2f}, semantic={evidence['semantic_contrib']:.2f}, "
            f"context={evidence['context_contrib']:.2f}, band={evidence['band_contrib']:.2f}"
        )
        limiting = list(evidence["limiting_hits"])
        if definition.category == "positive":
            for caution_name in ["low_specificity_caution", "modality_mismatch_caution", "probe_substrate_caution", "matrix_dominance_caution"]:
                if caution_scores.get(caution_name, 0.0) >= 0.25:
                    limiting.append(caution_name)
            if evidence.get("negative_evidence", 0.0) > 0.15:
                limiting.append("theme-competitive or contradictory evidence is present")
            if evidence.get("calibration_penalty", 0.0) >= 0.15:
                limiting.append("calibration-like grounding was downweighted in biosample interpretation")
        return {
            "theme_name": definition.theme_name,
            "category": definition.category,
            "score": round(float(score), 4),
            "confidence": round(float(confidence), 4),
            "raw_score_pre_normalization": round(float(evidence.get("raw_score_pre_normalization", 0.0)), 4),
            "normalized_score": round(float(evidence.get("normalized_score", score)), 4),
            "competition_penalty": round(float(evidence.get("competition_penalty", 0.0)), 4),
            "caution_penalty": round(float(evidence.get("caution_penalty", 0.0)), 4),
            "calibration_penalty": round(float(evidence.get("calibration_penalty", 0.0)), 4),
            "specificity_index": round(float(evidence.get("specificity_index", 0.0)), 4),
            "evidence_contributions": {
                "tier1": round(evidence["tier1_contrib"], 4),
                "tier2": round(evidence["tier2_contrib"], 4),
                "knowledge": round(evidence["knowledge_contrib"], 4),
                "semantic": round(evidence["semantic_contrib"], 4),
                "context": round(evidence["context_contrib"], 4),
                "band": round(evidence["band_contrib"], 4),
            },
            "evidence_balance_summary": evidence_balance_summary,
            "supporting_tier1_hits": evidence["supporting_tier1_hits"][:5],
            "supporting_tier2_hits": evidence["supporting_tier2_hits"][:5],
            "supporting_knowledge_hits": evidence["supporting_knowledge_hits"][:5],
            "supporting_semantic_regions": evidence["supporting_semantic_regions"][:4],
            "supporting_bands": evidence["supporting_bands"][:6] if evidence["supporting_bands"] else [_band_token_text(b) for b in self._theme_band_overlap(query_bands, definition)],
            "opposing_or_limiting_evidence": limiting[:6],
            "notes": definition.description,
        }

    def _build_evidence_profile_summary(
        self,
        theme_input: ThemeLayerInput,
        positive_outputs: list[dict],
        caution_outputs: list[dict],
        query_bands: list[float],
        global_caution_load: float,
    ) -> str:
        top_positive = ", ".join(f"{row['theme_name']}={row['score']:.2f}" for row in positive_outputs[:3]) or "no dominant positive themes"
        top_caution = ", ".join(f"{row['theme_name']}={row['score']:.2f}" for row in caution_outputs[:3]) or "no dominant cautions"
        return (
            f"Theme layer {self.version}. "
            f"Query bands {', '.join(_band_token_text(value) for value in query_bands[:6])}. "
            f"Dominant biochemical themes: {top_positive}. "
            f"Caution load: {top_caution}. "
            f"Mean caution burden={global_caution_load:.2f}."
        )

    def _build_theme_summary(self, positive_outputs: list[dict], caution_outputs: list[dict]) -> str:
        top_theme_text = ", ".join(
            f"{row['theme_name']} ({row['score']:.2f})" for row in positive_outputs[:3] if row["score"] >= 0.15
        )
        caution_text = ", ".join(
            f"{row['theme_name']} ({row['score']:.2f})" for row in caution_outputs[:3] if row["score"] >= 0.20
        )
        return (
            f"Theme layer {self.version} readout: {top_theme_text or 'no strong positive themes'}; "
            f"main cautions: {caution_text or 'no strong cautions'}."
        )

    def _build_what_not_to_claim(self, theme_input: ThemeLayerInput, caution_outputs: list[dict]) -> list[str]:
        warnings = [
            "Do not treat the theme layer as definitive molecular identification.",
            "Do not convert broad analog support into direct clinical diagnosis.",
        ]
        caution_scores = {row["theme_name"]: row["score"] for row in caution_outputs}
        if caution_scores.get("modality_mismatch_caution", 0.0) >= 0.25:
            warnings.append("Do not ignore modality mismatch between the query and the supporting evidence base.")
        if caution_scores.get("weak_label_or_cohort_caution", 0.0) >= 0.25:
            warnings.append("Do not over-interpret cohort-level or weak-label signal as subject-level certainty.")
        if theme_input.source_dataset_id == "covid_serum_raman":
            warnings.append("Do not treat spontaneous-Raman cohort structure as interchangeable with serum SERS behavior.")
        if caution_scores.get("low_specificity_caution", 0.0) >= 0.25:
            warnings.append("Do not claim single-molecule specificity when broad analog evidence dominates.")
        return warnings
