from __future__ import annotations

from dataclasses import replace
from typing import Any

from gaira.query_routing import QUERY_FAMILY_DEFINITIONS, infer_query_family

from app.services.inference import run_query_aware_inference


ROUTING_PRESETS = {
    "Serum liver disease SERS": {
        "sample_type": "serum",
        "modality": "sers",
        "substrate": "Ag colloid / serum SERS",
        "use_case_domain": "liver/hepatobiliary",
    },
    "General serum Raman": {
        "sample_type": "serum",
        "modality": "raman",
        "substrate": "glass / Raman",
        "use_case_domain": "general",
    },
    "EV disease/stress": {
        "sample_type": "ev",
        "modality": "sers",
        "substrate": "EV nanoprobe / SERS",
        "use_case_domain": "disease/stress EV",
    },
    "Pure analyte": {
        "sample_type": "analyte",
        "modality": "sers",
        "substrate": "purified analyte / SERS",
        "use_case_domain": "analyte",
    },
}


def predict_query_family(
    *,
    domain: str,
    source_dataset_id: str,
    sample_type: str | None,
    modality: str | None,
    use_case_domain: str | None,
    query_label: str | None,
    query_family: str | None,
) -> str | None:
    return infer_query_family(
        domain=domain,
        source_dataset_id=source_dataset_id,
        sample_type=sample_type,
        modality=modality,
        use_case_domain=use_case_domain,
        query_label=query_label,
        query_family=query_family,
    )


def get_family_definitions() -> dict[str, Any]:
    return QUERY_FAMILY_DEFINITIONS


def run_routing_preview(
    request,
    *,
    sample_type: str | None,
    modality: str | None,
    substrate_context: str | None,
    use_case_domain: str | None,
    forced_family: str | None = None,
) -> dict[str, Any]:
    return run_query_aware_inference(
        request,
        sample_type=sample_type,
        modality=modality,
        substrate_context=substrate_context,
        use_case_domain=use_case_domain,
        forced_query_family=forced_family,
    )
