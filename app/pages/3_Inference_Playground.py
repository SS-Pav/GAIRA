from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.inference_panels import render_result_panel, render_what_changed
from app.components.spectrum_plot import render_single_spectrum
from app.services.catalog import get_class_mean_spectrum, get_dataset_selector_options
from app.services.inference import (
    build_database_request,
    build_uploaded_request,
    compare_results,
    run_general_inference,
    run_query_aware_inference,
)


st.set_page_config(page_title="GAIRAM v1 • Inference Playground", page_icon="🔬", layout="wide")
st.title("GAIRAM v1 • Inference Playground")
st.caption("Pick a representative spectrum, run general inference, then compare it against query-aware inference with explicit sample context.")

selector_options = get_dataset_selector_options()

tab_db, tab_upload = st.tabs(["Choose from GAIRA database", "Upload processed CSV"])

selected_request = None
spectrum_payload = None

with tab_db:
    source_family = st.selectbox("Source family", ["Grounding", "EV", "Serum"])
    domain_key = source_family.lower()
    buckets = selector_options[domain_key]
    bucket_name = st.selectbox("Bucket", list(buckets.keys()))
    datasets = buckets[bucket_name]
    dataset_lookup = {item["display_name"]: item for item in datasets}
    dataset_name = st.selectbox("Dataset", list(dataset_lookup.keys()))
    dataset = dataset_lookup[dataset_name]
    item_rows = dataset["items"]
    item_label_key = "class_label"
    labels = [row["class_label"] for row in item_rows]
    selected_label = st.selectbox("Class / subset", labels)
    selected_item = next(row for row in item_rows if row["class_label"] == selected_label)
    family_label = (
        selected_item.get("subclass_label")
        or selected_item.get("experiment_family")
        or selected_item.get("family_label")
        or selected_item.get("class_label")
    )
    selected_request = build_database_request(domain_key if domain_key != "grounding" else "grounding", dataset["dataset_id"], selected_label, family_label)
    spectrum_payload = {
        "x": selected_request.spectrum_query.x.tolist(),
        "y": selected_request.spectrum_query.y.tolist(),
        "title": f"{dataset['display_name']} • {selected_label}",
        "meta": {
            "source family": source_family,
            "dataset": dataset["display_name"],
            "class": selected_label,
            "modality": selected_request.modality or "sers",
            "mode": "Representative class mean",
        },
    }

with tab_upload:
    upload = st.file_uploader("Upload processed CSV", type=["csv"])
    upload_domain = st.selectbox("Upload domain", ["grounding", "ev", "serum"])
    upload_label = st.text_input("Upload label", value="uploaded_spectrum")
    if upload is not None:
        df = pd.read_csv(upload)
        required_cols = {"wavenumber", "intensity"}
        if required_cols.issubset(df.columns):
            st.success("Parsed processed spectrum successfully.")
            selected_request = build_uploaded_request(
                domain=upload_domain,
                query_id=f"uploaded_{upload_label}",
                query_label=upload_label,
                family_label="uploaded_processed_csv",
                source_dataset_id="uploaded_processed_csv",
                x_values=df["wavenumber"].astype(float).tolist(),
                y_values=df["intensity"].astype(float).tolist(),
                sample_type=upload_domain,
                modality="sers",
                substrate_context="uploaded",
                use_case_domain="general",
            )
            spectrum_payload = {
                "x": df["wavenumber"].astype(float).tolist(),
                "y": df["intensity"].astype(float).tolist(),
                "title": f"Uploaded processed spectrum • {upload_label}",
                "meta": {
                    "source family": upload_domain.title(),
                    "dataset": "Uploaded CSV",
                    "class": upload_label,
                    "modality": "sers",
                    "mode": "Upload",
                },
            }
        else:
            st.error("Expected columns: `wavenumber`, `intensity`.")

st.markdown("## Spectrum display + general inference")
if spectrum_payload:
    visual_baseline_default = bool(selected_request and selected_request.domain == "serum")
    top_col, meta_col = st.columns([1.5, 0.8], gap="large")
    with top_col:
        apply_visual_baseline = st.toggle(
            "Apply baseline correction (visual only)",
            value=visual_baseline_default,
            help="Display-only preprocessing. Inference still uses the current backend processed representation.",
        )
        render_single_spectrum(
            spectrum_payload["x"],
            spectrum_payload["y"],
            title=spectrum_payload["title"],
            apply_baseline_correction=apply_visual_baseline,
        )
    with meta_col:
        st.markdown("### Metadata")
        for key, value in spectrum_payload["meta"].items():
            st.write(f"**{key.title()}**: {value}")

    if st.button("Run general inference", width="stretch"):
        st.session_state["general_result"] = run_general_inference(selected_request)

if "general_result" in st.session_state:
    render_result_panel(st.session_state["general_result"], title="General inference")

st.markdown("## Query-aware inference")
qa_col1, qa_col2, qa_col3 = st.columns([1.0, 1.0, 1.0], gap="large")
with qa_col1:
    qa_sample_type = st.selectbox("Sample type", ["grounding", "ev", "serum"], index=0 if not selected_request else ["grounding", "ev", "serum"].index(selected_request.domain))
    qa_modality = st.selectbox("Modality", ["raman", "sers"], index=1)
with qa_col2:
    qa_substrate = st.text_input("Substrate context", value=spectrum_payload["meta"]["mode"] if spectrum_payload else "representative mean")
    qa_use_case = st.selectbox("Use-case domain", ["general", "liver/hepatobiliary", "disease/stress EV", "metabolic/diabetes", "analyte"], index=0)
with qa_col3:
    qa_notes = st.text_input("Optional free text", value="")

if selected_request and st.button("Run query-aware inference", width="stretch"):
    routed_result = run_query_aware_inference(
        selected_request,
        sample_type=qa_sample_type,
        modality=qa_modality,
        substrate_context=qa_substrate,
        use_case_domain=qa_use_case if qa_notes.strip() == "" else f"{qa_use_case}; {qa_notes}",
    )
    st.session_state["routed_result"] = routed_result
    st.session_state["comparison_result"] = compare_results(st.session_state.get("general_result", run_general_inference(selected_request)), routed_result)

if "routed_result" in st.session_state:
    left, right = st.columns(2, gap="large")
    with left:
        render_result_panel(st.session_state["general_result"], title="General inference")
    with right:
        render_result_panel(st.session_state["routed_result"], title="Query-aware inference")
    render_what_changed(st.session_state["comparison_result"])
