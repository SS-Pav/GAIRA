from __future__ import annotations

import streamlit as st

from app.components.inference_panels import render_result_panel
from app.components.routing_flow import render_routing_flow
from app.services.inference import build_database_request
from app.services.routing import ROUTING_PRESETS, get_family_definitions, predict_query_family, run_routing_preview


st.set_page_config(page_title="GAIRAM v1 • Routing Engine", page_icon="🧭", layout="wide")
st.title("GAIRAM v1 • Routing Engine")
st.caption("How explicit sample metadata routes the evidence stack before biochemical interpretation.")

preset_col, _, _ = st.columns([1.3, 0.1, 1.6])
with preset_col:
    preset_name = st.selectbox("Preset", list(ROUTING_PRESETS.keys()), index=0)
    preset = ROUTING_PRESETS[preset_name]

sample_col, flow_col, result_col = st.columns([1.0, 1.0, 1.15], gap="large")

with sample_col:
    st.markdown("### Input metadata")
    sample_type = st.selectbox("Sample type", ["serum", "ev", "analyte"], index=["serum", "ev", "analyte"].index(preset["sample_type"]))
    modality = st.selectbox("Modality", ["raman", "sers"], index=["raman", "sers"].index(preset["modality"]))
    substrate = st.text_input("Substrate", value=preset["substrate"])
    use_case = st.selectbox(
        "Use-case domain",
        ["general", "liver/hepatobiliary", "disease/stress EV", "analyte", "metabolic/diabetes"],
        index=0 if preset["use_case_domain"] not in ["general", "liver/hepatobiliary", "disease/stress EV", "analyte", "metabolic/diabetes"] else ["general", "liver/hepatobiliary", "disease/stress EV", "analyte", "metabolic/diabetes"].index(preset["use_case_domain"]),
    )
    anchor_name = st.selectbox(
        "Reference spectrum anchor",
        [
            "CCA / HCC / LM serum: HCC mean",
            "Serum protocol comparison: p1",
            "COVID serum Raman: healthy control",
            "small2023 EV: c50",
            "SHINE EV: D2_C40",
            "Diabetes EV: Impact",
            "Adenine control: 1 ng/mL",
        ],
    )


def _anchor_request(anchor_label: str):
    if anchor_label == "CCA / HCC / LM serum: HCC mean":
        return build_database_request("serum", "cca_hcc_lm_serum_sers", "hcc", "released_zip_archive")
    if anchor_label == "Serum protocol comparison: p1":
        return build_database_request("serum", "serum_protocol_comparison", "p1", "protocol_comparison_archive")
    if anchor_label == "COVID serum Raman: healthy control":
        return build_database_request("serum", "covid_serum_raman", "healthy_control", "covid19_serum_raman_archive")
    if anchor_label == "small2023 EV: c50":
        return build_database_request("ev", "small2023_ev", "c50", "normedprobe1")
    if anchor_label == "SHINE EV: D2_C40":
        return build_database_request("ev", "shine_ev_sers", "D2_C40", "Set9")
    if anchor_label == "Diabetes EV: Impact":
        return build_database_request("ev", "diabetes_plasma_ev_sers", "Impact", "figure3_processed_archive")
    return build_database_request("grounding", "adenine_sers_control", "adenine_1ng_ml", "lateral_flow_strip_reference")


request = _anchor_request(anchor_name)
predicted_family = predict_query_family(
    domain=request.domain,
    source_dataset_id=request.source_dataset_id,
    sample_type=sample_type,
    modality=modality,
    use_case_domain=use_case,
    query_label=request.query_label,
    query_family=request.query_family,
)

with flow_col:
    st.markdown("### Routing flow")
    render_routing_flow(predicted_family)

forced_options = [None, "serum_general", "serum_liver_hepatobiliary", "ev_general", "ev_disease_or_stress", "grounding_analyte"]
forced_label = st.selectbox("Forced comparison family", ["None"] + [item for item in forced_options[1:]], index=0)
forced_family = None if forced_label == "None" else forced_label

routed_result = run_routing_preview(
    request,
    sample_type=sample_type,
    modality=modality,
    substrate_context=substrate,
    use_case_domain=use_case,
    forced_family=None,
)
forced_result = None
if forced_family:
    forced_result = run_routing_preview(
        request,
        sample_type=sample_type,
        modality=modality,
        substrate_context=substrate,
        use_case_domain=use_case,
        forced_family=forced_family,
    )

with result_col:
    st.markdown("### Route comparison")
    st.metric("Predicted routing family", routed_result.get("query_routing_family") or "legacy")
    st.metric("Matched context hits", int(routed_result.get("family_matched_context_hits", 0)))
    st.metric("Matched support hits", int(routed_result.get("family_matched_support_hits", 0)))
    rationale = get_family_definitions().get(routed_result.get("query_routing_family"))
    if rationale:
        st.write(f"**Emphasis:** {rationale.emphasis}")
        st.write(f"**Boosts:** {', '.join(rationale.boost)}")
    if forced_result:
        st.markdown("#### Forced family comparison")
        st.write(f"Forced family: `{forced_family}`")
        st.write(
            f"Matched context hits: {int(forced_result.get('family_matched_context_hits', 0))} • matched support hits: {int(forced_result.get('family_matched_support_hits', 0))}"
        )
        st.caption("Wrong-family routing should usually reduce family-matched context/support while leaving generic support visible.")

st.divider()
render_result_panel(routed_result, title="Routed retrieval preview")
