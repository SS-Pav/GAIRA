from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from gaira.domain_pack_registry import get_domain_pack
from gaira.ev_context import EVContextRetriever
from gaira.grounding_search import GroundingSearchEngine, SpectrumQuery
from gaira.inference_reranking import rerank_grounding_hits
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
    )


class GAIRAInferenceEngine:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.grounding_engine = GroundingSearchEngine(db_path=db_path)
        self.serum_context = SerumContextRetriever(db_path=db_path)
        self.ev_context = EVContextRetriever(db_path=db_path)

    def _select_pack(self, domain: str) -> dict:
        if domain == "serum":
            return get_domain_pack("GAIRA_SERUM")
        if domain == "ev":
            return get_domain_pack("GAIRA_EV")
        raise ValueError(f"Unsupported domain '{domain}'. Use 'serum' or 'ev'.")

    def _select_context_hits(self, request: InferenceRequest, direct_df: pd.DataFrame) -> pd.DataFrame:
        top_labels = direct_df["source_label"].head(6).astype(str).tolist() if not direct_df.empty else []
        if request.domain == "serum":
            label_df = self.serum_context.search_by_grounding_labels(top_labels, top_n=6)
            band_df = self.serum_context.search_by_bands([725.0, 1003.0, 1450.0, 1659.0], top_n=6)
            text_df = self.serum_context.search_by_text(
                " ".join(
                    [
                        "serum uric acid hypoxanthine adsorption bias batch caveat paper comparison",
                        request.source_dataset_id,
                        request.query_family,
                        request.query_label,
                        "protocol optimization strip variability shelf life metabolite spiking ergothioneine",
                    ]
                ),
                top_n=6,
            )
        else:
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
                    ]
                ),
                top_n=6,
            )

        frames = [df for df in [label_df, band_df, text_df] if not df.empty]
        if not frames:
            return pd.DataFrame()
        context_df = pd.concat(frames, ignore_index=True)
        context_df = context_df.sort_values(["score", "document_id"], ascending=[False, True])
        return context_df.drop_duplicates(subset=["document_id", "section", "chunk_text"]).head(8).reset_index(drop=True)

    def _build_summary(
        self,
        request: InferenceRequest,
        pack_entry: dict,
        tier1_before_df: pd.DataFrame,
        tier1_df: pd.DataFrame,
        tier2_before_df: pd.DataFrame,
        tier2_df: pd.DataFrame,
        context_df: pd.DataFrame,
        knowledge_df: pd.DataFrame,
        semantic_df: pd.DataFrame,
    ) -> str:
        lines = [
            f"Domain pack: {pack_entry['pack_id']}",
            f"Query: {request.query_id} ({request.source_dataset_id} / {request.query_label} / {request.query_family})",
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
        lines.append("5. Final integrated interpretation")
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
        tier1_df = rerank_grounding_hits(tier1_before_df, domain=request.domain, tier="tier1")
        tier2_df = rerank_grounding_hits(tier2_before_df, domain=request.domain, tier="tier2")
        knowledge_all_df = rerank_grounding_hits(knowledge_before_df, domain=request.domain, tier="tier2")
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
        context_df = self._select_context_hits(request, tier1_df)

        result = {
            "domain_pack": pack_entry["pack_id"],
            "query_id": request.query_id,
            "query_label": request.query_label,
            "query_family": request.query_family,
            "source_dataset_id": request.source_dataset_id,
            "tier1_grounding_hits_before_reranking": tier1_before_df.head(10).to_dict(orient="records"),
            "tier1_grounding_hits": tier1_df.head(10).to_dict(orient="records"),
            "tier2_support_hits_before_reranking": tier2_before_df.head(10).to_dict(orient="records") if not tier2_before_df.empty else [],
            "tier2_support_hits": tier2_df.head(10).to_dict(orient="records") if not tier2_df.empty else [],
            "knowledge_support_hits_before_reranking": knowledge_before_df.head(10).to_dict(orient="records") if not knowledge_before_df.empty else [],
            "knowledge_support_hits": knowledge_df.to_dict(orient="records") if not knowledge_df.empty else [],
            "semantic_region_support_hits": semantic_df.to_dict(orient="records") if not semantic_df.empty else [],
            "domain_context_hits": context_df.head(10).to_dict(orient="records") if not context_df.empty else [],
            "final_summary": self._build_summary(
                request,
                pack_entry,
                tier1_before_df,
                tier1_df,
                tier2_before_df,
                tier2_df,
                context_df,
                knowledge_df,
                semantic_df,
            ),
        }
        return result
