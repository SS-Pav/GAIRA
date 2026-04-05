from __future__ import annotations

import streamlit as st

from app.components.layer_explorer import render_layer_explorer
from app.components.metric_cards import render_metric_cards
from app.components.rag_panel import render_rag_panel
from app.services.catalog import get_global_kpis, get_layer_tree, get_rag_inventory


st.set_page_config(page_title="GAIRAM v1 • Knowledge Map", page_icon="🗺️", layout="wide")
st.title("GAIRAM v1 • Knowledge Map")
st.caption("The live knowledge base behind GAIRAM v1: spectra, dataset structure, literature support, context notes, and cautions.")

kpis = get_global_kpis()
render_metric_cards(
    [
        ("Live datasets", f"{kpis['total_live_datasets']}", "Current integrated datasets"),
        ("Live spectra", f"{kpis['total_live_spectra']:,}", "Grounding + biosample spectra"),
        ("Support/context docs", f"{kpis['total_support_context_docs']}", "Retrievable support notes and context"),
        ("Holdout", "hcc_serum", "Holdout-only, not in live serum routing"),
        ("Layers", f"{kpis['live_layers']}", "GAIRA_GROUNDING, GAIRA_EV, GAIRA_SERUM"),
    ]
)

layer_tree = get_layer_tree()
rag_inventory = get_rag_inventory()

left_col, right_col = st.columns([1.1, 1.0], gap="large")

with left_col:
    tabs = st.tabs(["GAIRA_GROUNDING", "GAIRA_EV", "GAIRA_SERUM"])
    for tab, layer_name in zip(tabs, ["GAIRA_GROUNDING", "GAIRA_EV", "GAIRA_SERUM"]):
        with tab:
            render_layer_explorer(layer_name, layer_tree[layer_name], key_prefix=layer_name.lower())
            if layer_name == "GAIRA_SERUM":
                st.info("`hcc_serum` is holdout-only and is intentionally excluded from live GAIRAM v1 serum routing.")

with right_col:
    tabs = st.tabs(["Grounding", "EV", "Serum"])
    for tab, layer_name in zip(tabs, ["Grounding", "EV", "Serum"]):
        with tab:
            render_rag_panel(layer_name, rag_inventory[layer_name], key_prefix=layer_name.lower())
