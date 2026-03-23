from __future__ import annotations

from dataclasses import dataclass


QUERY_FAMILIES = [
    "serum_liver_hepatobiliary",
    "serum_general",
    "serum_metabolic",
    "ev_general",
    "ev_metabolic_or_diabetes",
    "ev_injury_response",
    "grounding_analyte",
]


@dataclass(frozen=True)
class QueryFamilyDefinition:
    family: str
    sample_type: str
    emphasis: str
    boost: tuple[str, ...]
    downweight: tuple[str, ...]
    keep_visible: tuple[str, ...]


QUERY_FAMILY_DEFINITIONS: dict[str, QueryFamilyDefinition] = {
    "serum_liver_hepatobiliary": QueryFamilyDefinition(
        family="serum_liver_hepatobiliary",
        sample_type="serum",
        emphasis="liver/hepatobiliary serum interpretation",
        boost=("liver_serum_literature_support", "liver-serum serum-context notes"),
        downweight=("EV-specific notes", "analyte/control-only support"),
        keep_visible=("generic serum context", "shared knowledge support"),
    ),
    "serum_general": QueryFamilyDefinition(
        family="serum_general",
        sample_type="serum",
        emphasis="generic serum interpretation",
        boost=("generic serum context", "serum Ag-colloid literature support"),
        downweight=("liver-disease-specific serum notes", "EV-specific notes"),
        keep_visible=("shared knowledge support", "limited liver-serum support when overlap is real"),
    ),
    "serum_metabolic": QueryFamilyDefinition(
        family="serum_metabolic",
        sample_type="serum",
        emphasis="metabolic serum interpretation",
        boost=("generic serum context", "metabolic serum notes"),
        downweight=("hepatobiliary-specific notes", "EV-specific notes"),
        keep_visible=("shared biochemical support",),
    ),
    "ev_general": QueryFamilyDefinition(
        family="ev_general",
        sample_type="ev",
        emphasis="general EV interpretation",
        boost=("generic EV context", "shared EV support"),
        downweight=("serum/liver notes", "analyte-only support"),
        keep_visible=("shared grounding", "shared knowledge support"),
    ),
    "ev_metabolic_or_diabetes": QueryFamilyDefinition(
        family="ev_metabolic_or_diabetes",
        sample_type="ev",
        emphasis="metabolic/diabetes EV interpretation",
        boost=("diabetes EV support/context", "generic EV notes"),
        downweight=("serum/liver notes", "injury-response EV notes unless overlap is real"),
        keep_visible=("shared grounding", "shared knowledge support"),
    ),
    "ev_injury_response": QueryFamilyDefinition(
        family="ev_injury_response",
        sample_type="ev",
        emphasis="injury-response / perturbation EV interpretation",
        boost=("SHINE/SPECTRA EV support/context", "generic EV notes"),
        downweight=("serum/liver notes", "metabolic EV notes unless overlap is real"),
        keep_visible=("shared grounding", "shared knowledge support"),
    ),
    "grounding_analyte": QueryFamilyDefinition(
        family="grounding_analyte",
        sample_type="grounding",
        emphasis="analyte/metabolite grounding interpretation",
        boost=("metabolite/analyte grounding", "RamanBioLib/shared grounding"),
        downweight=("serum cohort literature", "EV context"),
        keep_visible=("shared knowledge support",),
    ),
}


def _normalized_text(*values: str | None) -> str:
    return " ".join(str(value or "").strip().lower() for value in values if str(value or "").strip())


def infer_query_family(
    *,
    domain: str,
    source_dataset_id: str,
    sample_type: str | None = None,
    modality: str | None = None,
    use_case_domain: str | None = None,
    query_label: str | None = None,
    query_family: str | None = None,
    forced_query_family: str | None = None,
    disable_query_routing: bool = False,
) -> str | None:
    if disable_query_routing:
        return None
    if forced_query_family:
        return forced_query_family

    sample = (sample_type or domain or "").strip().lower()
    text = _normalized_text(source_dataset_id, use_case_domain, query_label, query_family, modality)

    if domain == "grounding" or sample in {"grounding", "analyte", "metabolite"}:
        return "grounding_analyte"

    if sample == "ev" or domain == "ev":
        if any(token in text for token in ["diabetes", "metabolic", "obesity", "insulin", "mitochond"]):
            return "ev_metabolic_or_diabetes"
        if any(token in text for token in ["injury", "apap", "hepatotox", "perturb", "dose-response", "dose response", "shine", "spectra"]):
            return "ev_injury_response"
        return "ev_general"

    if sample == "serum" or domain == "serum":
        if any(token in text for token in ["liver", "hepat", "hcc", "cca", "cirrho", "hepatitis", "nafld", "nash", "masld", "dili", "cholangio", "biliary"]):
            return "serum_liver_hepatobiliary"
        if any(token in text for token in ["metabolic", "diabetes", "obesity"]):
            return "serum_metabolic"
        return "serum_general"

    return None


def classify_support_family(row: dict) -> str:
    dataset_id = str(row.get("source_dataset_id", "")).strip().lower()
    result_type = str(row.get("result_type", "")).strip().lower()
    source_label = str(row.get("source_label", "")).strip().lower()
    notes = str(row.get("notes", "")).strip().lower()
    target_dataset_id = str(row.get("target_dataset_id", "")).strip().lower()
    text = _normalized_text(dataset_id, result_type, source_label, notes, target_dataset_id)

    if dataset_id in {"metabolite_sers63_support", "adenine_sers_control", "ramanbiolib"}:
        return "grounding_analyte"
    if dataset_id == "liver_serum_literature_support":
        return "serum_liver_hepatobiliary"
    if dataset_id in {"serum_ag_colloids_literature_grounding", "serum_ag_colloids_grounding"}:
        return "serum_general"
    if dataset_id == "diabetes_ev_context_support" or "diabetes" in text:
        return "ev_metabolic_or_diabetes"
    if dataset_id == "shine_spectra_context_support" or any(token in text for token in ["shine", "spectra", "apap", "injury", "hepatotox"]):
        return "ev_injury_response"
    if any(token in text for token in ["small2023_ev", "extracellular vesicle", "probe1", "probe2"]):
        return "ev_general"
    return "shared_generic"


def classify_context_family(row: dict, domain: str) -> str:
    document_id = str(row.get("document_id", "")).strip().lower()
    source_dataset_id = str(row.get("source_dataset_id", "")).strip().lower()
    context_type = str(row.get("context_type", "")).strip().lower()
    chunk_text = str(row.get("chunk_text", "")).strip().lower()
    text = _normalized_text(document_id, source_dataset_id, context_type, chunk_text)

    if domain == "serum":
        if any(token in text for token in ["gaira_serum_context_liver_", "gaira_serum_context_hcc_", "gaira_serum_context_metabolic_liver", "ck18_dili_biomarker_support", "hepatobiliary", "cirrhosis", "hepatitis", "dili", "nafld", "nash", "masld", "cholangio"]):
            return "serum_liver_hepatobiliary"
        if any(token in text for token in ["metabolic", "diabetes", "obesity"]):
            return "serum_metabolic"
        return "serum_general"

    if domain == "ev":
        if any(
            token in text
            for token in [
                "gaira_ev_context_dataset_context",
                "gaira_ev_context_cross_substrate_caveat",
                "gaira_ev_context_cargo_mixture_caveat",
                "gaira_ev_context_evidence_tiering_note",
                "gaira_ev_context_default_embedding_status",
                "gaira_ev_context_small2023_benchmark_hierarchy_note",
                "gaira_ev_context_small2023_probe_family_note",
            ]
        ):
            return "ev_general"
        if source_dataset_id == "diabetes_plasma_ev_sers" or any(token in text for token in ["diabetes", "metabolic", "insulin", "mitochondrial", "impact", "strong-d", "strongd"]):
            return "ev_metabolic_or_diabetes"
        if source_dataset_id == "shine_ev_sers" or any(token in text for token in ["shine", "spectra", "apap", "injury", "day0", "day2", "dose-response", "dose response"]):
            return "ev_injury_response"
        return "ev_general"

    return "shared_generic"


def classify_knowledge_family(row: dict) -> str:
    text = _normalized_text(
        row.get("source_label"),
        row.get("source_family"),
        row.get("notes"),
        row.get("result_type"),
    )
    if any(token in text for token in ["liver", "hepat", "hcc", "cca", "cirrho", "hepatitis", "dili", "nafld", "nash", "masld", "bile"]):
        return "serum_liver_hepatobiliary"
    if any(token in text for token in ["diabetes", "metabolic", "insulin", "mitochond", "lipoprotein"]):
        return "ev_metabolic_or_diabetes"
    if any(token in text for token in ["injury", "hepatotox", "apap", "perturb", "stress-response", "stress response"]):
        return "ev_injury_response"
    if any(token in text for token in ["adenine", "methyladenine", "caffeine", "metabolite", "purine", "xanthine", "nicotinamide"]):
        return "grounding_analyte"
    return "shared_generic"


def routing_weight(query_family: str | None, candidate_family: str, channel: str) -> float:
    if not query_family:
        return 1.0
    if candidate_family == "shared_generic":
        return 1.0
    if candidate_family == query_family:
        return {"context": 1.45, "support": 1.35, "knowledge": 1.20}.get(channel, 1.0)

    serum_families = {"serum_liver_hepatobiliary", "serum_general", "serum_metabolic"}
    ev_families = {"ev_general", "ev_metabolic_or_diabetes", "ev_injury_response"}

    if query_family in serum_families and candidate_family in serum_families:
        if query_family == "serum_liver_hepatobiliary" and candidate_family == "serum_general":
            return {"context": 0.82, "support": 0.72, "knowledge": 0.92}.get(channel, 0.72)
        if query_family == "serum_general" and candidate_family == "serum_liver_hepatobiliary":
            return 0.72
        if query_family == "serum_metabolic" and candidate_family == "serum_general":
            return 1.08
        return 0.86

    if query_family in ev_families and candidate_family in ev_families:
        if query_family in {"ev_metabolic_or_diabetes", "ev_injury_response"} and candidate_family == "ev_general":
            return 1.10
        if query_family == "ev_general" and candidate_family in {"ev_metabolic_or_diabetes", "ev_injury_response"}:
            return 0.78
        return 0.78

    if query_family == "grounding_analyte":
        return {"context": 0.25, "support": 0.25, "knowledge": 0.45}.get(channel, 0.25)

    if candidate_family == "grounding_analyte":
        return {"context": 0.25, "support": 0.55, "knowledge": 0.85}.get(channel, 0.55)

    return {"context": 0.30, "support": 0.40, "knowledge": 0.60}.get(channel, 0.40)


def summarize_routing_weights(query_family: str | None, families: list[str], channel: str) -> str:
    if not query_family:
        return "legacy_routing"
    seen: list[str] = []
    for family in families:
        if family and family not in seen:
            seen.append(family)
    return ", ".join(f"{family}:{routing_weight(query_family, family, channel):.2f}" for family in seen[:6])
