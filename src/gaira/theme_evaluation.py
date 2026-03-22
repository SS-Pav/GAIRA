from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from gaira.biochemical_theme_layer import BiochemicalThemeLayer, ThemeLayerInput
from gaira.inference import GAIRAInferenceEngine, InferenceRequest, load_ev_class_mean_query, load_serum_class_mean_query
from gaira.grounding_search import GroundingSearchEngine, SpectrumQuery
from gaira.theme_ontology import CAUTION_THEMES, POSITIVE_THEMES


@dataclass
class EvaluationBundle:
    query_df: pd.DataFrame
    theme_df: pd.DataFrame
    summary_df: pd.DataFrame


class ThemeEvaluationRunner:
    def __init__(self, db_path: Path, theme_layer_version: str = "v1") -> None:
        self.db_path = Path(db_path)
        self.theme_layer_version = theme_layer_version
        self.inference_engine = GAIRAInferenceEngine(db_path=db_path, theme_layer_version=theme_layer_version)
        self.grounding_engine = GroundingSearchEngine(db_path=db_path)
        self.theme_layer = BiochemicalThemeLayer(db_path=db_path, version=theme_layer_version)

    def load_serum_class_mean_requests(
        self,
        dataset_id: str,
        subclass_label: str,
        processing_version: str | None = None,
        class_order: list[str] | None = None,
    ) -> list[InferenceRequest]:
        with duckdb.connect(str(self.db_path), read_only=True) as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT class_label
                FROM biosample_class_summary
                WHERE dataset_id = ?
                  AND subclass_label = ?
                  AND (? IS NULL OR processing_version = ?)
                ORDER BY class_label
                """,
                [dataset_id, subclass_label, processing_version, processing_version],
            ).fetchall()
        labels = [str(row[0]) for row in rows]
        if class_order is not None:
            order_lookup = {label: index for index, label in enumerate(class_order)}
            labels = sorted(labels, key=lambda value: order_lookup.get(value, 999))
        return [
            load_serum_class_mean_query(
                db_path=self.db_path,
                dataset_id=dataset_id,
                class_label=class_label,
                subclass_label=subclass_label,
                processing_version=processing_version,
            )
            for class_label in labels
        ]

    def load_ev_class_mean_requests(
        self,
        dataset_id: str,
        subclass_labels: list[str],
        class_order: list[str],
        processing_version: str,
    ) -> list[InferenceRequest]:
        requests: list[InferenceRequest] = []
        for subclass_label in subclass_labels:
            for class_label in class_order:
                requests.append(
                    load_ev_class_mean_query(
                        db_path=self.db_path,
                        dataset_id=dataset_id,
                        class_label=class_label,
                        subclass_label=subclass_label,
                        processing_version=processing_version,
                    )
                )
        return requests

    def load_biosample_processed_requests(
        self,
        dataset_id: str,
        domain: str,
        processing_version: str,
        limit: int | None = None,
    ) -> list[InferenceRequest]:
        with duckdb.connect(str(self.db_path), read_only=True) as connection:
            query = """
                SELECT
                  p.biosample_id,
                  m.class_label,
                  m.subclass_label,
                  p.wavenumbers_json,
                  p.intensity_json
                FROM biosample_processed_spectra p
                JOIN biosample_metadata m
                  ON p.dataset_id = m.dataset_id
                 AND p.biosample_id = m.biosample_id
                WHERE p.dataset_id = ?
                  AND p.processing_version = ?
                ORDER BY m.class_label, p.biosample_id
            """
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            rows = connection.execute(query, [dataset_id, processing_version]).fetchall()

        requests = []
        for biosample_id, class_label, subclass_label, x_json, y_json in rows:
            spectrum_query = SpectrumQuery(
                query_id=str(biosample_id),
                query_label=str(class_label),
                query_family=str(subclass_label),
                source_dataset_id=dataset_id,
                x=np.asarray(json.loads(x_json), dtype=float),
                y=np.asarray(json.loads(y_json), dtype=float),
                notes=f"Processed biosample spectrum ({processing_version})",
            )
            requests.append(
                InferenceRequest(
                    domain=domain,
                    query_id=str(biosample_id),
                    query_label=str(class_label),
                    query_family=str(subclass_label),
                    source_dataset_id=dataset_id,
                    spectrum_query=spectrum_query,
                )
            )
        return requests

    def load_grounding_class_summary_queries(
        self,
        dataset_id: str,
        processing_version: str,
    ) -> list[ThemeLayerInput]:
        with duckdb.connect(str(self.db_path), read_only=True) as connection:
            rows = connection.execute(
                """
                SELECT class_label, experiment_family, mean_wavenumbers_json, mean_intensity_json
                FROM grounding_class_summary
                WHERE dataset_id = ?
                  AND processing_version = ?
                ORDER BY experiment_family, class_label
                """,
                [dataset_id, processing_version],
            ).fetchall()

        theme_inputs: list[ThemeLayerInput] = []
        for class_label, experiment_family, x_json, y_json in rows:
            spectrum_query = SpectrumQuery(
                query_id=f"{dataset_id}_{class_label}",
                query_label=str(class_label),
                query_family=str(experiment_family),
                source_dataset_id=dataset_id,
                x=np.asarray(json.loads(x_json), dtype=float),
                y=np.asarray(json.loads(y_json), dtype=float),
                notes=f"Processed grounding class summary ({processing_version})",
            )
            direct_df = self.grounding_engine.search_direct_spectral_evidence(spectrum_query, top_n_per_source=5)
            tier1_df = direct_df[direct_df["evidence_tier"] == "tier1_direct_spectral_grounding"].copy()
            seed_labels = tier1_df["source_label"].head(6).astype(str).tolist() if not tier1_df.empty else []
            tier2_df = self.grounding_engine.search_supporting_literature_for_spectrum(
                spectrum_query,
                seed_labels=seed_labels,
                domain=None,
                top_n=8,
            )
            knowledge_df = self.grounding_engine.search_knowledge_support(
                spectrum_query,
                seed_labels=seed_labels,
                domain=None,
                top_n=10,
            )
            semantic_df = (
                knowledge_df[knowledge_df["result_type"] == "semantic_region_support"].head(5).reset_index(drop=True)
                if not knowledge_df.empty
                else pd.DataFrame()
            )
            knowledge_only_df = (
                knowledge_df[knowledge_df["result_type"] != "semantic_region_support"].head(8).reset_index(drop=True)
                if not knowledge_df.empty
                else pd.DataFrame()
            )
            theme_inputs.append(
                ThemeLayerInput(
                    domain="grounding",
                    query_id=spectrum_query.query_id,
                    query_label=spectrum_query.query_label,
                    query_family=spectrum_query.query_family,
                    source_dataset_id=dataset_id,
                    spectrum_query=spectrum_query,
                    tier1_hits=tier1_df.head(10).to_dict(orient="records"),
                    tier2_hits=tier2_df.head(10).to_dict(orient="records") if not tier2_df.empty else [],
                    knowledge_hits=knowledge_only_df.to_dict(orient="records") if not knowledge_only_df.empty else [],
                    semantic_hits=semantic_df.to_dict(orient="records") if not semantic_df.empty else [],
                    context_hits=[],
                )
            )
        return theme_inputs

    def load_grounding_processed_queries(
        self,
        dataset_id: str,
        processing_version: str,
        experiment_family: str | None = None,
    ) -> list[ThemeLayerInput]:
        with duckdb.connect(str(self.db_path), read_only=True) as connection:
            rows = connection.execute(
                """
                SELECT
                  p.grounding_id,
                  m.class_label,
                  m.experiment_family,
                  p.wavenumbers_json,
                  p.intensity_json
                FROM grounding_processed_spectra p
                JOIN grounding_metadata m
                  ON p.dataset_id = m.dataset_id
                 AND p.grounding_id = m.grounding_id
                WHERE p.dataset_id = ?
                  AND p.processing_version = ?
                  AND (? IS NULL OR m.experiment_family = ?)
                ORDER BY m.class_label, p.grounding_id
                """,
                [dataset_id, processing_version, experiment_family, experiment_family],
            ).fetchall()

        theme_inputs: list[ThemeLayerInput] = []
        for grounding_id, class_label, query_family, x_json, y_json in rows:
            spectrum_query = SpectrumQuery(
                query_id=str(grounding_id),
                query_label=str(class_label),
                query_family=str(query_family),
                source_dataset_id=dataset_id,
                x=np.asarray(json.loads(x_json), dtype=float),
                y=np.asarray(json.loads(y_json), dtype=float),
                notes=f"Processed grounding spectrum ({processing_version})",
            )
            direct_df = self.grounding_engine.search_direct_spectral_evidence(spectrum_query, top_n_per_source=5)
            tier1_df = direct_df[direct_df["evidence_tier"] == "tier1_direct_spectral_grounding"].copy()
            seed_labels = tier1_df["source_label"].head(6).astype(str).tolist() if not tier1_df.empty else []
            tier2_df = self.grounding_engine.search_supporting_literature_for_spectrum(
                spectrum_query,
                seed_labels=seed_labels,
                domain=None,
                top_n=8,
            )
            knowledge_df = self.grounding_engine.search_knowledge_support(
                spectrum_query,
                seed_labels=seed_labels,
                domain=None,
                top_n=10,
            )
            semantic_df = (
                knowledge_df[knowledge_df["result_type"] == "semantic_region_support"].head(5).reset_index(drop=True)
                if not knowledge_df.empty
                else pd.DataFrame()
            )
            knowledge_only_df = (
                knowledge_df[knowledge_df["result_type"] != "semantic_region_support"].head(8).reset_index(drop=True)
                if not knowledge_df.empty
                else pd.DataFrame()
            )
            theme_inputs.append(
                ThemeLayerInput(
                    domain="grounding",
                    query_id=spectrum_query.query_id,
                    query_label=spectrum_query.query_label,
                    query_family=spectrum_query.query_family,
                    source_dataset_id=dataset_id,
                    spectrum_query=spectrum_query,
                    tier1_hits=tier1_df.head(10).to_dict(orient="records"),
                    tier2_hits=tier2_df.head(10).to_dict(orient="records") if not tier2_df.empty else [],
                    knowledge_hits=knowledge_only_df.to_dict(orient="records") if not knowledge_only_df.empty else [],
                    semantic_hits=semantic_df.to_dict(orient="records") if not semantic_df.empty else [],
                    context_hits=[],
                )
            )
        return theme_inputs

    def evaluate_inference_requests(self, track_name: str, requests: list[InferenceRequest]) -> EvaluationBundle:
        query_rows = []
        theme_rows = []
        for request in requests:
            result = self.inference_engine.run_inference(request)
            query_rows.append(
                {
                    "track_name": track_name,
                    "query_id": request.query_id,
                    "source_dataset_id": request.source_dataset_id,
                    "query_label": request.query_label,
                    "query_family": request.query_family,
                    "dominant_themes": "|".join(result.get("dominant_themes", [])),
                    "global_caveats": "|".join(result.get("biochemical_global_caveats", [])),
                    "theme_summary": result.get("biochemical_theme_summary", ""),
                    "evidence_profile_summary": result.get("evidence_profile_summary", ""),
                }
            )
            theme_rows.extend(self._flatten_theme_outputs(track_name, request.query_id, request.source_dataset_id, request.query_label, request.query_family, result["biochemical_theme_outputs"]))

        query_df = pd.DataFrame(query_rows)
        theme_df = pd.DataFrame(theme_rows)
        summary_df = self._summarize_themes(theme_df)
        return EvaluationBundle(query_df=query_df, theme_df=theme_df, summary_df=summary_df)

    def evaluate_theme_inputs(self, track_name: str, theme_inputs: list[ThemeLayerInput]) -> EvaluationBundle:
        query_rows = []
        theme_rows = []
        for theme_input in theme_inputs:
            result = self.theme_layer.build_from_input(theme_input)
            query_rows.append(
                {
                    "track_name": track_name,
                    "query_id": theme_input.query_id,
                    "source_dataset_id": theme_input.source_dataset_id,
                    "query_label": theme_input.query_label,
                    "query_family": theme_input.query_family,
                    "dominant_themes": "|".join(result.get("dominant_themes", [])),
                    "global_caveats": "|".join(result.get("biochemical_global_caveats", [])),
                    "theme_summary": result.get("biochemical_theme_summary", ""),
                    "evidence_profile_summary": result.get("evidence_profile_summary", ""),
                }
            )
            theme_rows.extend(self._flatten_theme_outputs(track_name, theme_input.query_id, theme_input.source_dataset_id, theme_input.query_label, theme_input.query_family, result["biochemical_theme_outputs"]))

        query_df = pd.DataFrame(query_rows)
        theme_df = pd.DataFrame(theme_rows)
        summary_df = self._summarize_themes(theme_df)
        return EvaluationBundle(query_df=query_df, theme_df=theme_df, summary_df=summary_df)

    def _flatten_theme_outputs(
        self,
        track_name: str,
        query_id: str,
        source_dataset_id: str,
        query_label: str,
        query_family: str,
        theme_outputs: list[dict],
    ) -> list[dict]:
        rows = []
        for row in theme_outputs:
            rows.append(
                {
                    "track_name": track_name,
                    "query_id": query_id,
                    "source_dataset_id": source_dataset_id,
                    "query_label": query_label,
                    "query_family": query_family,
                    "theme_name": row["theme_name"],
                    "category": row["category"],
                    "theme_layer_version": self.theme_layer_version,
                    "score": row["score"],
                    "confidence": row["confidence"],
                    "raw_score_pre_normalization": row.get("raw_score_pre_normalization", row["score"]),
                    "normalized_score": row.get("normalized_score", row["score"]),
                    "competition_penalty": row.get("competition_penalty", 0.0),
                    "caution_penalty": row.get("caution_penalty", 0.0),
                    "calibration_penalty": row.get("calibration_penalty", 0.0),
                    "specificity_index": row.get("specificity_index", 0.0),
                    "evidence_balance_summary": row["evidence_balance_summary"],
                    "tier1_contrib": row["evidence_contributions"]["tier1"],
                    "tier2_contrib": row["evidence_contributions"]["tier2"],
                    "knowledge_contrib": row["evidence_contributions"]["knowledge"],
                    "semantic_contrib": row["evidence_contributions"]["semantic"],
                    "context_contrib": row["evidence_contributions"]["context"],
                    "band_contrib": row["evidence_contributions"]["band"],
                    "n_tier1_hits": len(row.get("supporting_tier1_hits", [])),
                    "n_tier2_hits": len(row.get("supporting_tier2_hits", [])),
                    "n_knowledge_hits": len(row.get("supporting_knowledge_hits", [])),
                    "n_semantic_hits": len(row.get("supporting_semantic_regions", [])),
                    "n_bands": len(row.get("supporting_bands", [])),
                    "limiting_evidence": "|".join(str(item) for item in row.get("opposing_or_limiting_evidence", [])),
                }
            )
        return rows

    def _summarize_themes(self, theme_df: pd.DataFrame) -> pd.DataFrame:
        if theme_df.empty:
            return pd.DataFrame()
        return (
            theme_df.groupby(["track_name", "theme_name", "category"], as_index=False)
            .agg(
                mean_score=("score", "mean"),
                std_score=("score", "std"),
                mean_confidence=("confidence", "mean"),
                mean_raw_score=("raw_score_pre_normalization", "mean"),
                mean_specificity_index=("specificity_index", "mean"),
                mean_competition_penalty=("competition_penalty", "mean"),
                mean_caution_penalty=("caution_penalty", "mean"),
                mean_calibration_penalty=("calibration_penalty", "mean"),
                mean_tier1_hits=("n_tier1_hits", "mean"),
                mean_tier2_hits=("n_tier2_hits", "mean"),
                mean_knowledge_hits=("n_knowledge_hits", "mean"),
            )
            .fillna(0.0)
        )


def build_track_metrics(track_name: str, theme_df: pd.DataFrame) -> pd.DataFrame:
    if theme_df.empty:
        return pd.DataFrame()

    pivot = theme_df.pivot_table(
        index="query_id",
        columns="theme_name",
        values="score",
        aggfunc="mean",
    ).fillna(0.0)
    confidence_pivot = theme_df.pivot_table(index="query_id", columns="theme_name", values="confidence", aggfunc="mean").fillna(0.0)

    rows = []
    positive_cols = [col for col in POSITIVE_THEMES if col in pivot.columns]
    caution_cols = [col for col in CAUTION_THEMES if col in pivot.columns]
    if positive_cols:
        positive_matrix = pivot[positive_cols].to_numpy()
        positive_sums = positive_matrix.sum(axis=1, keepdims=True)
        normalized = positive_matrix / np.clip(positive_sums, 1e-9, None)
        entropy = -np.sum(np.where(normalized > 0, normalized * np.log(np.clip(normalized, 1e-9, None)), 0.0), axis=1)
        max_entropy = math.log(len(positive_cols)) if len(positive_cols) > 1 else 1.0
        entropy_inverse = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0.0)
        sorted_vals = np.sort(positive_matrix, axis=1)
        top_vals = sorted_vals[:, -1]
        second_vals = sorted_vals[:, -2] if sorted_vals.shape[1] > 1 else np.zeros_like(top_vals)
        dominance_margin = top_vals - second_vals
        rows.append({"track_name": track_name, "metric_name": "mean_dominance_margin", "metric_value": float(np.mean(dominance_margin))})
        rows.append({"track_name": track_name, "metric_name": "positive_entropy_inverse", "metric_value": float(np.mean(entropy_inverse))})
        if not confidence_pivot.empty:
            conf_vals = confidence_pivot[[col for col in positive_cols if col in confidence_pivot.columns]].mean(axis=1)
            caution_mean = pivot[caution_cols].mean(axis=1) if caution_cols else pd.Series(0.0, index=pivot.index)
            rows.append(
                {
                    "track_name": track_name,
                    "metric_name": "caution_adjusted_confidence",
                    "metric_value": float((conf_vals * (1.0 - 0.6 * caution_mean)).mean()),
                }
            )

    if track_name == "adenine_controlled_specificity":
        purine = pivot.get("nucleic_acid_purine_associated", pd.Series(dtype=float))
        other_cols = [col for col in POSITIVE_THEMES if col != "nucleic_acid_purine_associated" and col in pivot.columns]
        other_max = pivot[other_cols].max(axis=1) if other_cols else pd.Series(0.0, index=pivot.index)
        other_mean = pivot[other_cols].mean(axis=1) if other_cols else pd.Series(0.0, index=pivot.index)
        rows.append(
            {
                "track_name": track_name,
                "metric_name": "purine_dominance_margin",
                "metric_value": float((purine - other_max).mean()),
            }
        )
        rows.append(
            {
                "track_name": track_name,
                "metric_name": "purine_top_fraction",
                "metric_value": float((purine >= other_max).mean()),
            }
        )
        rows.append(
            {
                "track_name": track_name,
                "metric_name": "off_theme_suppression",
                "metric_value": float((1.0 - other_mean).mean()),
            }
        )
    elif track_name == "serum_protocol_robustness":
        positive_std = pivot[positive_cols].std().mean()
        caution_mean = pivot[caution_cols].mean().mean() if caution_cols else 0.0
        rows.append({"track_name": track_name, "metric_name": "positive_theme_stability", "metric_value": float(1.0 / (1.0 + positive_std))})
        rows.append({"track_name": track_name, "metric_name": "caution_presence", "metric_value": float(caution_mean)})
    elif track_name == "ev_mixture_coherence":
        positive_std = pivot[positive_cols].std().mean()
        caution_std = pivot[caution_cols].std().mean() if caution_cols else 0.0
        rows.append({"track_name": track_name, "metric_name": "theme_smoothness_proxy", "metric_value": float(1.0 / (1.0 + positive_std))})
        rows.append({"track_name": track_name, "metric_name": "caution_variability", "metric_value": float(caution_std)})
    elif track_name == "covid_serum_usefulness":
        modality = pivot.get("modality_mismatch_caution", pd.Series(dtype=float))
        low_spec = pivot.get("low_specificity_caution", pd.Series(dtype=float))
        positive_mean = pivot[positive_cols].mean().mean()
        rows.append({"track_name": track_name, "metric_name": "modality_caution_mean", "metric_value": float(modality.mean() if not modality.empty else 0.0)})
        rows.append({"track_name": track_name, "metric_name": "low_specificity_mean", "metric_value": float(low_spec.mean() if not low_spec.empty else 0.0)})
        rows.append({"track_name": track_name, "metric_name": "positive_signal_mean", "metric_value": float(positive_mean)})
    elif track_name == "replicate_consistency":
        positive_std = pivot[positive_cols].std().mean()
        rows.append({"track_name": track_name, "metric_name": "within_condition_consistency", "metric_value": float(1.0 / (1.0 + positive_std))})

    return pd.DataFrame(rows)
