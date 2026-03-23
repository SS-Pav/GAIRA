from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from gaira.biochemical_theme_layer import BiochemicalThemeLayer
from gaira.domain_pack_registry import get_domain_pack
from gaira.ev_context import EVContextRetriever
from gaira.grounding_search import GroundingSearchEngine, SpectrumQuery
from gaira.inference_reranking import rerank_grounding_hits
from gaira.query_routing import (
    classify_context_family,
    infer_query_family,
    routing_weight,
    summarize_routing_weights,
)
from gaira.serum_context import SerumContextRetriever


HCC_SERUM_PROCESSING_VERSION = "v1_crop430_1730_interp1_minmax"
SMALL2023_PROCESSING_VERSION = "v1_crop670_1800_interp1_minmax"


@dataclass
class InferenceRequest:
    domain: str
    query_id: str
    query_label: str
    query_family: str
    source_dataset_id: str
    spectrum_query: SpectrumQuery
    sample_type: str | None = None
    modality: str | None = None
    substrate_context: str | None = None
    use_case_domain: str | None = None
    forced_query_family: str | None = None
    disable_query_routing: bool = False


def _default_serum_use_case_domain(dataset_id: str) -> str:
    if dataset_id == "cca_hcc_lm_serum_sers":
        return "liver/hepatobiliary"
    return "general"


def _default_ev_use_case_domain(dataset_id: str) -> str:
    if dataset_id == "diabetes_plasma_ev_sers":
        return "metabolic/diabetes"
    if dataset_id == "shine_ev_sers":
        return "injury/perturbation"
    return "general"


def _default_serum_modality(dataset_id: str) -> str:
    if dataset_id == "covid_serum_raman":
        return "raman"
    return "sers"


def load_grounding_class_mean_query(
    db_path: Path,
    dataset_id: str,
    class_label: str,
    experiment_family: str | None = None,
    processing_version: str | None = None,
) -> InferenceRequest:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        row = connection.execute(
            """
            SELECT mean_wavenumbers_json, mean_intensity_json, processing_version, experiment_family
            FROM grounding_class_summary
            WHERE dataset_id = ?
              AND class_label = ?
              AND (? IS NULL OR experiment_family = ?)
              AND (? IS NULL OR processing_version = ?)
            ORDER BY experiment_family, summary_id
            LIMIT 1
            """,
            [dataset_id, class_label, experiment_family, experiment_family, processing_version, processing_version],
        ).fetchone()
    if row is None:
        raise ValueError(f"Missing grounding class summary for {dataset_id} / {class_label}.")
    x_values = _parse_json_array(row[0])
    y_values = _parse_json_array(row[1])
    resolved_processing_version = str(row[2])
    resolved_family = str(row[3])
    query = SpectrumQuery(
        query_id=f"{dataset_id}_{class_label.lower()}_{resolved_family.lower()}",
        query_label=class_label,
        query_family=resolved_family,
        source_dataset_id=dataset_id,
        x=x_values,
        y=y_values,
        notes=f"Grounding class summary query ({resolved_processing_version})",
    )
    return InferenceRequest(
        domain="grounding",
        query_id=query.query_id,
        query_label=class_label,
        query_family=resolved_family,
        source_dataset_id=dataset_id,
        spectrum_query=query,
        sample_type="grounding",
        modality="sers",
        use_case_domain="analyte",
    )


def _parse_json_array(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def load_serum_class_mean_query(
    db_path: Path,
    dataset_id: str,
    class_label: str,
    subclass_label: str,
    processing_version: str | None = None,
) -> InferenceRequest:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        if processing_version is None:
            row = connection.execute(
                """
                SELECT mean_wavenumbers_json, mean_intensity_json, processing_version
                FROM biosample_class_summary
                WHERE dataset_id = ?
                  AND class_label = ?
                  AND subclass_label = ?
                ORDER BY summary_id
                LIMIT 1
                """,
                [dataset_id, class_label, subclass_label],
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT mean_wavenumbers_json, mean_intensity_json, processing_version
                FROM biosample_class_summary
                WHERE dataset_id = ?
                  AND class_label = ?
                  AND subclass_label = ?
                  AND processing_version = ?
                LIMIT 1
                """,
                [dataset_id, class_label, subclass_label, processing_version],
            ).fetchone()
    if row is None:
        raise ValueError(f"Missing serum class summary for {dataset_id} / {class_label} / {subclass_label}.")
    x_values = _parse_json_array(row[0])
    y_values = _parse_json_array(row[1])
    resolved_processing_version = str(row[2])
    query = SpectrumQuery(
        query_id=f"{dataset_id}_{class_label.lower()}_{subclass_label.lower()}",
        query_label=class_label,
        query_family=subclass_label,
        source_dataset_id=dataset_id,
        x=x_values,
        y=y_values,
        notes=f"Serum biosample class summary query ({resolved_processing_version})",
    )
    return InferenceRequest(
        domain="serum",
        query_id=query.query_id,
        query_label=class_label,
        query_family=subclass_label,
        source_dataset_id=dataset_id,
        spectrum_query=query,
        sample_type="serum",
        modality=_default_serum_modality(dataset_id),
        use_case_domain=_default_serum_use_case_domain(dataset_id),
    )


def load_ev_class_mean_query(
    db_path: Path,
    dataset_id: str,
    class_label: str,
    subclass_label: str,
    processing_version: str | None = SMALL2023_PROCESSING_VERSION,
) -> InferenceRequest:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        if processing_version is None:
            row = connection.execute(
                """
                SELECT mean_wavenumbers_json, mean_intensity_json, processing_version
                FROM biosample_class_summary
                WHERE dataset_id = ?
                  AND class_label = ?
                  AND subclass_label = ?
                ORDER BY summary_id
                LIMIT 1
                """,
                [dataset_id, class_label, subclass_label],
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT mean_wavenumbers_json, mean_intensity_json, processing_version
                FROM biosample_class_summary
                WHERE dataset_id = ?
                  AND class_label = ?
                  AND subclass_label = ?
                  AND processing_version = ?
                LIMIT 1
                """,
                [dataset_id, class_label, subclass_label, processing_version],
            ).fetchone()
    if row is None:
        raise ValueError(f"Missing EV class summary for {dataset_id} / {class_label} / {subclass_label}.")
    x_values = _parse_json_array(row[0])
    y_values = _parse_json_array(row[1])
    resolved_processing_version = str(row[2])
    query = SpectrumQuery(
        query_id=f"{dataset_id}_{class_label.lower()}_{subclass_label.lower()}",
        query_label=class_label,
        query_family=subclass_label,
        source_dataset_id=dataset_id,
        x=x_values,
        y=y_values,
        notes=f"EV biosample class summary query ({resolved_processing_version})",
    )
    return InferenceRequest(
        domain="ev",
        query_id=query.query_id,
        query_label=class_label,
        query_family=subclass_label,
        source_dataset_id=dataset_id,
        spectrum_query=query,
        sample_type="ev",
        modality="sers",
        use_case_domain=_default_ev_use_case_domain(dataset_id),
    )


class GAIRAInferenceEngine:
    def __init__(self, db_path: Path, theme_layer_version: str = "v1") -> None:
        self.db_path = Path(db_path)
        self.theme_layer_version = theme_layer_version
        self.grounding_engine = GroundingSearchEngine(db_path=db_path)
        self.serum_context = SerumContextRetriever(db_path=db_path)
        self.ev_context = EVContextRetriever(db_path=db_path)
        self.theme_layer = BiochemicalThemeLayer(db_path=db_path, version=theme_layer_version)

    def _select_pack(self, domain: str) -> dict:
        if domain == "serum":
            return get_domain_pack("GAIRA_SERUM")
        if domain == "ev":
            return get_domain_pack("GAIRA_EV")
        if domain == "grounding":
            return get_domain_pack("GAIRA_GROUNDING")
        raise ValueError(f"Unsupported domain '{domain}'. Use 'serum', 'ev', or 'grounding'.")

    def _select_context_hits(self, request: InferenceRequest, direct_df: pd.DataFrame, query_routing_family: str | None) -> pd.DataFrame:
        top_labels = direct_df["source_label"].head(6).astype(str).tolist() if not direct_df.empty else []
        if request.domain == "serum":
            serum_text_fragments = [
                "serum uric acid hypoxanthine adsorption bias batch caveat paper comparison",
                request.source_dataset_id,
                request.query_family,
                request.query_label,
                "protocol optimization strip variability shelf life metabolite spiking ergothioneine",
            ]
            if query_routing_family == "serum_liver_hepatobiliary":
                serum_text_fragments.extend(
                    [
                        "liver hepatobiliary hcc cca lm cholangiocarcinoma cirrhosis hepatitis hbv hcv",
                        "dili liver injury nafld nash masld bile duct hepatocellular carcinoma metastatic liver",
                        "purine oxidative stress lipid remodeling albumin bilirubin serum differential caution",
                    ]
                )
            elif query_routing_family == "serum_metabolic":
                serum_text_fragments.extend(
                    [
                        "metabolic diabetes obesity insulin lipoprotein mitochondrial serum heterogeneity",
                        "oxidative stress uric acid purine metabolic syndrome caution",
                    ]
                )
            else:
                serum_text_fragments.extend(
                    [
                        "generic serum matrix dominance modality caution protocol comparison support",
                    ]
                )
            label_df = self.serum_context.search_by_grounding_labels(top_labels, top_n=6)
            band_df = self.serum_context.search_by_bands([725.0, 1003.0, 1450.0, 1659.0], top_n=6)
            text_df = self.serum_context.search_by_text(
                " ".join(str(fragment) for fragment in serum_text_fragments if str(fragment).strip()),
                top_n=6,
            )
        else:
            if request.domain == "ev":
                label_df = self.ev_context.search_by_labels(
                    top_labels + [request.source_dataset_id, request.query_family, request.query_label],
                    top_n=6,
                )
                band_df = pd.DataFrame()
                text_df = self.ev_context.search_by_text(
                    " ".join(
                        [
                            "extracellular vesicles probe1 probe2 substrate caveat weak label default embedding transductive",
                            request.source_dataset_id,
                            request.query_family,
                            request.query_label,
                            "shine consensus analog gold nanopillar diabetes mapping impact strongd cargo mixture",
                            "diabetes heterogeneity subgroup overlap normal-weight overweight insulin mitochondrial lipoprotein",
                            "apap hepatotoxicity injury dose-response day0 day2 albumin cck8 monoculture preclinical",
                        ]
                    ),
                    top_n=6,
                )
            else:
                return pd.DataFrame()

        frames = [df for df in [label_df, band_df, text_df] if not df.empty]
        if not frames:
            return pd.DataFrame()
        context_df = pd.concat(frames, ignore_index=True)
        context_df["context_family"] = context_df.apply(
            lambda row: classify_context_family(row.to_dict(), domain=request.domain),
            axis=1,
        )
        context_df["routing_relevance_weight"] = context_df["context_family"].apply(
            lambda family: routing_weight(query_routing_family, family, channel="context")
        )
        context_df["routing_score"] = context_df["score"].astype(float) * context_df["routing_relevance_weight"].astype(float)
        if request.domain == "ev" and "source_dataset_id" in context_df.columns:
            source_values = context_df["source_dataset_id"].fillna("").astype(str)
            exact_match = source_values.str.contains(request.source_dataset_id, regex=False)
            generic_match = source_values.eq("") | source_values.str.contains("small2023_ev,shine_ev_sers,diabetes_plasma_ev_sers", regex=False)
            context_df["dataset_match_weight"] = 0.0
            context_df.loc[exact_match, "dataset_match_weight"] = 2.0
            context_df.loc[~exact_match & generic_match, "dataset_match_weight"] = 1.0
        else:
            context_df["dataset_match_weight"] = 0.0
        context_df = context_df.sort_values(
            ["routing_score", "dataset_match_weight", "score", "document_id"],
            ascending=[False, False, False, True],
        )
        return context_df.drop_duplicates(subset=["document_id", "section", "chunk_text"]).head(8).reset_index(drop=True)

    def _build_summary(
        self,
        request: InferenceRequest,
        query_routing_family: str | None,
        pack_entry: dict,
        tier1_before_df: pd.DataFrame,
        tier1_df: pd.DataFrame,
        tier2_before_df: pd.DataFrame,
        tier2_df: pd.DataFrame,
        context_df: pd.DataFrame,
        knowledge_df: pd.DataFrame,
        semantic_df: pd.DataFrame,
        theme_summary: str,
        global_caveats: list[str],
    ) -> str:
        lines = [
            f"Domain pack: {pack_entry['pack_id']}",
            f"Query: {request.query_id} ({request.source_dataset_id} / {request.query_label} / {request.query_family})",
            f"Query-aware routing family: {query_routing_family or 'legacy'}",
            "",
            "1. Direct grounding evidence after domain-aware reranking",
        ]
        if tier1_df.empty:
            lines.append("No tier-1 grounding hits.")
        else:
            for row in tier1_df.head(6).to_dict(orient="records"):
                lines.append(
                    f"- {row['result_type']} | {row['source_dataset_id']} | {row['source_label']} | "
                    f"base={float(row['base_score']):.4f} | weight={float(row['domain_relevance_weight']):.2f} | "
                    f"reranked={float(row['reranked_score']):.4f}"
                )

        lines.append("")
        lines.append("2. Literature/support evidence after domain-aware reranking")
        if tier2_df.empty:
            lines.append("No tier-2 support hits.")
        else:
            for row in tier2_df.head(6).to_dict(orient="records"):
                lines.append(
                    f"- {row['result_type']} | {row['source_dataset_id']} | {row['source_label']} | "
                    f"base={float(row['base_score']):.4f} | weight={float(row['domain_relevance_weight']):.2f} | "
                    f"reranked={float(row['reranked_score']):.4f}"
                )

        lines.append("")
        lines.append("3. Domain-specific interpretation notes")
        if context_df.empty:
            lines.append("No domain-context hits.")
        else:
            for row in context_df.head(6).to_dict(orient="records"):
                lines.append(
                    f"- {row['document_id']} | {row['section']} | score={float(row['score']):.2f} | {row['matched_tokens']}"
                )

        lines.append("")
        lines.append("4. Knowledge and semantic interpretation support")
        if knowledge_df.empty and semantic_df.empty:
            lines.append("No knowledge-core support hits.")
        else:
            for row in semantic_df.head(4).to_dict(orient="records"):
                lines.append(
                    f"- semantic | {row['source_label']} | reranked={float(row['reranked_score']):.4f} | {row['notes']}"
                )
            for row in knowledge_df.head(6).to_dict(orient="records"):
                lines.append(
                    f"- {row['result_type']} | {row['source_label']} | reranked={float(row['reranked_score']):.4f} | "
                    f"{row['notes'][:220]}"
                )

        top_tier1 = tier1_df.head(3)["source_label"].astype(str).tolist() if not tier1_df.empty else []
        top_tier2 = tier2_df.head(3)["source_label"].astype(str).tolist() if not tier2_df.empty else []
        top_context = context_df.head(3)["document_id"].astype(str).tolist() if not context_df.empty else []
        top_knowledge = knowledge_df.head(3)["source_label"].astype(str).tolist() if not knowledge_df.empty else []
        top_semantic = semantic_df.head(3)["source_label"].astype(str).tolist() if not semantic_df.empty else []

        lines.append("")
        lines.append("5. Biochemical theme layer")
        lines.append(f"Theme layer version: {self.theme_layer_version}")
        lines.append(theme_summary)
        if global_caveats:
            lines.append(f"Global caveats: {', '.join(global_caveats)}")

        lines.append("")
        lines.append("6. Final integrated interpretation")
        if request.domain == "serum":
            lines.append(
                "The shared grounding layer returned both broad RamanBioLib analogs and serum-local Ag-colloid "
                "grounding support. Domain-aware reranking boosts study-matched serum grounding and modestly boosts "
                "serum literature support while keeping RamanBioLib visible as broad molecular evidence. Serum-context "
                "notes still add caveats about metabolite dominance, adsorption/protocol sensitivity, and batch-aware interpretation. "
                "Knowledge-core and semantic-region support remain interpretive aids rather than definitive molecular identification."
            )
        else:
            lines.append(
                "The shared grounding layer returned broad analog evidence, but EV interpretation must be filtered "
                "through EV-domain context. Domain-aware reranking keeps broad shared grounding neutral while "
                "downweighting serum-specific grounding and serum literature support unless they are unusually strong. "
                "EV-context notes then add probe/substrate-family caution, the small2023_ev benchmark hierarchy, and "
                "weak-label caveats for diabetes_plasma_ev_sers. Knowledge-core and semantic-region support should be "
                "read as conservative biochemical interpretation support, not as literal EV cargo identification."
            )
        if top_tier1:
            lines.append(f"Top tier-1 labels: {', '.join(top_tier1)}")
        if top_tier2:
            lines.append(f"Top tier-2 support labels: {', '.join(top_tier2)}")
        if top_context:
            lines.append(f"Top context documents: {', '.join(top_context)}")
        if top_semantic:
            lines.append(f"Top semantic regions: {', '.join(top_semantic)}")
        if top_knowledge:
            lines.append(f"Top knowledge support labels: {', '.join(top_knowledge)}")
        if not tier1_before_df.empty and not tier1_df.empty:
            lines.append(
                f"Top tier-1 before reranking: {', '.join(tier1_before_df.head(3)['source_label'].astype(str).tolist())}"
            )
        if not tier2_before_df.empty and not tier2_df.empty:
            lines.append(
                f"Top tier-2 before reranking: {', '.join(tier2_before_df.head(3)['source_label'].astype(str).tolist())}"
            )
        return "\n".join(lines)

    def run_inference(self, request: InferenceRequest) -> dict:
        pack_entry = self._select_pack(request.domain)
        query_routing_family = infer_query_family(
            domain=request.domain,
            source_dataset_id=request.source_dataset_id,
            sample_type=request.sample_type,
            modality=request.modality,
            use_case_domain=request.use_case_domain,
            query_label=request.query_label,
            query_family=request.query_family,
            forced_query_family=request.forced_query_family,
            disable_query_routing=request.disable_query_routing,
        )
        direct_df = self.grounding_engine.search_direct_spectral_evidence(
            request.spectrum_query,
            top_n_per_source=5,
        )
        tier1_before_df = direct_df[direct_df["evidence_tier"] == "tier1_direct_spectral_grounding"].copy()
        seed_labels = tier1_before_df["source_label"].head(6).astype(str).tolist() if not tier1_before_df.empty else []
        tier2_before_df = self.grounding_engine.search_supporting_literature_for_spectrum(
            request.spectrum_query,
            seed_labels=seed_labels,
            domain=request.domain,
            top_n=8,
        )
        knowledge_before_df = self.grounding_engine.search_knowledge_support(
            request.spectrum_query,
            seed_labels=seed_labels,
            domain=request.domain,
            top_n=10,
        )
        tier1_df = rerank_grounding_hits(
            tier1_before_df,
            domain=request.domain,
            tier="tier1",
            query_source_dataset_id=request.source_dataset_id,
            query_routing_family=query_routing_family,
        )
        tier2_df = rerank_grounding_hits(
            tier2_before_df,
            domain=request.domain,
            tier="tier2",
            query_source_dataset_id=request.source_dataset_id,
            query_routing_family=query_routing_family,
        )
        knowledge_all_df = rerank_grounding_hits(
            knowledge_before_df,
            domain=request.domain,
            tier="tier2",
            query_source_dataset_id=request.source_dataset_id,
            query_routing_family=query_routing_family,
        )
        semantic_df = (
            knowledge_all_df[knowledge_all_df["result_type"] == "semantic_region_support"]
            .head(5)
            .reset_index(drop=True)
            if not knowledge_all_df.empty
            else pd.DataFrame()
        )
        knowledge_df = (
            knowledge_all_df[knowledge_all_df["result_type"] != "semantic_region_support"]
            .head(8)
            .reset_index(drop=True)
            if not knowledge_all_df.empty
            else pd.DataFrame()
        )
        context_df = self._select_context_hits(request, tier1_df, query_routing_family=query_routing_family)
        theme_result = self.theme_layer.build_from_inference(
            request,
            {
                "tier1_grounding_hits": tier1_df.to_dict(orient="records"),
                "tier2_support_hits": tier2_df.to_dict(orient="records") if not tier2_df.empty else [],
                "knowledge_support_hits": knowledge_df.to_dict(orient="records") if not knowledge_df.empty else [],
                "semantic_region_support_hits": semantic_df.to_dict(orient="records") if not semantic_df.empty else [],
                "domain_context_hits": context_df.to_dict(orient="records") if not context_df.empty else [],
            },
        )

        result = {
            "domain_pack": pack_entry["pack_id"],
            "query_id": request.query_id,
            "query_label": request.query_label,
            "query_family": request.query_family,
            "query_routing_family": query_routing_family,
            "source_dataset_id": request.source_dataset_id,
            "biochemical_theme_layer_version": theme_result["biochemical_theme_layer_version"],
            "tier1_grounding_hits_before_reranking": tier1_before_df.head(10).to_dict(orient="records"),
            "tier1_grounding_hits": tier1_df.head(10).to_dict(orient="records"),
            "tier2_support_hits_before_reranking": tier2_before_df.head(10).to_dict(orient="records") if not tier2_before_df.empty else [],
            "tier2_support_hits": tier2_df.head(10).to_dict(orient="records") if not tier2_df.empty else [],
            "knowledge_support_hits_before_reranking": knowledge_before_df.head(10).to_dict(orient="records") if not knowledge_before_df.empty else [],
            "knowledge_support_hits": knowledge_df.to_dict(orient="records") if not knowledge_df.empty else [],
            "semantic_region_support_hits": semantic_df.to_dict(orient="records") if not semantic_df.empty else [],
            "domain_context_hits": context_df.head(10).to_dict(orient="records") if not context_df.empty else [],
            "family_matched_support_hits": int(
                sum(1 for row in tier2_df.head(10).to_dict(orient="records") if str(row.get("support_family", "")) == str(query_routing_family))
            ),
            "family_matched_context_hits": int(
                sum(1 for row in context_df.head(10).to_dict(orient="records") if str(row.get("context_family", "")) == str(query_routing_family))
            ),
            "routing_weight_summary": {
                "query_routing_family": query_routing_family or "legacy",
                "tier2": summarize_routing_weights(
                    query_routing_family,
                    [str(row.get("support_family", "")) for row in tier2_df.head(10).to_dict(orient="records")],
                    channel="support",
                ),
                "context": summarize_routing_weights(
                    query_routing_family,
                    [str(row.get("context_family", "")) for row in context_df.head(10).to_dict(orient="records")],
                    channel="context",
                ),
            },
            "biochemical_theme_outputs": theme_result["biochemical_theme_outputs"],
            "biochemical_theme_summary": theme_result["biochemical_theme_summary"],
            "biochemical_global_caveats": theme_result["biochemical_global_caveats"],
            "biochemical_what_not_to_claim": theme_result["biochemical_what_not_to_claim"],
            "dominant_themes": theme_result["dominant_themes"],
            "evidence_profile_summary": theme_result["evidence_profile_summary"],
            "query_bands_cm": theme_result["query_bands_cm"],
            "final_summary": self._build_summary(
                request,
                query_routing_family,
                pack_entry,
                tier1_before_df,
                tier1_df,
                tier2_before_df,
                tier2_df,
                context_df,
                knowledge_df,
                semantic_df,
                theme_result["biochemical_theme_summary"],
                theme_result["biochemical_global_caveats"],
            ),
        }
        return result
