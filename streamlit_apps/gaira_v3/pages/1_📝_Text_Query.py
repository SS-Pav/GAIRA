"""GAIRA v3 — Text Query Page.

Literature-grounded biochemical Q&A with optional Gemini LLM synthesis.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

import plotly.graph_objects as go
import streamlit as st

from gaira.llm.response_schema import GAIRAResponse
from gaira.llm.local_synthesizer import synthesize_local_response
from gaira.retrieval.text_query_retriever import TextQueryRetriever
from gaira.retrieval.evidence_packet_builder import build_packet
from gaira.retrieval.section_linker import link_sections_to_evidence
from gaira.retrieval.confidence_composer import compose_confidence
from gaira.retrieval.literature_bsv_builder import (
    BSV_COMPONENTS, build_literature_bsv_profile,
)
from gaira.retrieval.motif_theme_mapper import (
    THEME_DISPLAY as THEME_DISPLAY_MAP, map_evidence_to_motifs_themes_bsv,
)
from gaira.retrieval.trust_graph_builder import TIER_COLORS, TIER_DISPLAY, build_trust_graph
from gaira.retrieval.trust_graph_render import render_trust_graph


EXAMPLE_QUERIES = [
    "What biochemical changes are associated with HCC in serum SERS spectra?",
    "How does HCC differ from healthy in serum SERS?",
    "Compare HCC, CCA, and liver metastases biochemical composition",
    "What are the 8 BSV components and what biochemistry do they capture?",
    "What biochemical changes are associated with NAFLD?",
]

PARSED_SECTIONS = [
    ("answer_summary", "Summary"),
    ("biochemical_themes", "Biochemical Themes"),
    ("strongest_evidence", "Strongest Evidence"),
    ("supporting_evidence", "Supporting Evidence"),
    ("caveats", "Caveats"),
    ("confidence_notes", "Confidence Notes"),
]

BSV_DISPLAY = {
    "membrane_lipid": "Lipid", "protein_backbone": "Protein",
    "aromatic_amino_acid": "Aromatic AA", "purine_nucleotide": "Purine",
    "pyrimidine_nucleotide": "Pyrimidine", "glycan_carbohydrate": "Glycan",
    "redox_metabolite": "Redox", "nucleic_acid_backbone": "Nuc. Backbone",
}

CONDITION_COLORS = {
    "HCC": "#E74C3C", "NAFLD_NASH": "#F39C12", "cholangiocarcinoma": "#9B59B6",
    "healthy_control": "#2ECC71", "hepatitis": "#3498DB",
    "liver_cancer_unspecified": "#E67E22", "fibrosis": "#1ABC9C",
}

BG = "#1a1a2e"


def _hex_to_rgba(h, a=0.12):
    h = h.lstrip("#")
    if len(h) != 6:
        return f"rgba(170,170,170,{a})"
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


@st.cache_resource
def get_retriever():
    r = TextQueryRetriever()
    r.load_sources()
    return r


# ── Page ───────────────────────────────────────────────────────────────

st.header("📝 Text Query")
st.caption("Literature-grounded biochemical Q&A.")

retriever = get_retriever()

# Sidebar controls
with st.sidebar:
    st.header("Text Query Settings")
    n_sec = sum(s["n_sections"] for s in retriever.source_summary())
    st.caption(f"{n_sec} sections · {len(retriever.documents)} documents")
    top_k = st.slider("Evidence items", min_value=3, max_value=12, value=8)

    st.divider()
    response_mode = st.radio(
        "Response mode",
        ["Local synthesis", "Gemini LLM"],
        index=0,
        help="Local = no API call. Gemini = external LLM (requires API key + quota).",
    )

# Query input
example = st.selectbox("Example queries:", ["(custom)"] + EXAMPLE_QUERIES)
default_q = example if example != "(custom)" else st.session_state.get("tq_last_query", "")
query = st.text_area("Your question:", value=default_q, height=80,
                      placeholder="e.g. What biochemical changes are associated with HCC?")
run = st.button("Run Text Query", type="primary", disabled=not query.strip())


def _run_text_query(q, mode):
    with st.spinner("Retrieving evidence..."):
        retrieved = retriever.retrieve(q, top_k=top_k)

    if len(retrieved) < 2:
        st.warning("Too few evidence items. Try broadening your query.")
        st.stop()

    packet = build_packet(q, retrieved)
    bsv_data = build_literature_bsv_profile(q, retrieved)
    mtb_map = map_evidence_to_motifs_themes_bsv(retrieved)

    if mode == "Gemini LLM":
        from gaira.llm.gemini_client import generate_text
        from gaira.llm.prompt_builder import build_prompt

        prompt = build_prompt(
            user_query=q, evidence=packet["evidence"],
            provenance=packet["provenance"], caveats=packet["caveats"],
            domain_context=packet["domain_context"],
        )
        with st.spinner("Querying Gemini..."):
            try:
                result = generate_text(prompt)
                raw_response = result.text
                parsed = GAIRAResponse.from_raw(raw_response)
                model_used = result.model_used
            except RuntimeError as e:
                st.warning(f"Gemini unavailable: {e}. Falling back to local synthesis.")
                mode = "Local synthesis"

    if mode == "Local synthesis":
        synth = synthesize_local_response(q, retrieved, mtb_map, bsv_data, packet)
        parsed = GAIRAResponse(
            raw_text="[Local synthesis]", parse_success=True, **synth,
        )
        raw_response = parsed.raw_text
        model_used = "local-synthesis"

    section_links = link_sections_to_evidence(parsed, retrieved)
    confidence = compose_confidence(retrieved, packet, section_links)
    graph_data = build_trust_graph(q, retrieved, mtb_map, bsv_data)

    st.session_state.update({
        "tq_last_query": q, "tq_retrieved": retrieved, "tq_parsed": parsed,
        "tq_raw": raw_response, "tq_links": section_links,
        "tq_conf": confidence, "tq_graph": graph_data,
        "tq_bsv": bsv_data, "tq_mtb": mtb_map,
        "tq_model": model_used, "tq_mode": mode,
    })


if run and query.strip():
    _run_text_query(query.strip(), response_mode)

if "tq_parsed" not in st.session_state:
    st.stop()

# ── Render results ─────────────────────────────────────────────────────

parsed = st.session_state["tq_parsed"]
bsv_data = st.session_state["tq_bsv"]
confidence = st.session_state["tq_conf"]
graph_data = st.session_state["tq_graph"]
section_links = st.session_state["tq_links"]

st.divider()
st.subheader("GAIRA Response")

mode = st.session_state.get("tq_mode", "Local synthesis")
model = st.session_state.get("tq_model", "?")
st.caption(f"Mode: **{mode}**" + (f" · Model: `{model}`" if mode == "Gemini LLM" else ""))

if parsed.parse_success:
    for attr, label in PARSED_SECTIONS:
        content = getattr(parsed, attr, "")
        if content:
            st.markdown(f"#### {label}")
            st.markdown(content)

with st.expander("Raw response"):
    st.text(st.session_state.get("tq_raw", ""))

# BSV Radar
if bsv_data.get("available"):
    st.divider()
    st.subheader("Literature-Grounded BSV Profile")
    cats = [BSV_DISPLAY.get(c, c) for c in BSV_COMPONENTS]
    fig = go.Figure()
    for cond, prof in bsv_data["profiles"].items():
        vals = [prof.get(c, 0) for c in BSV_COMPONENTS] + [prof.get(BSV_COMPONENTS[0], 0)]
        color = CONDITION_COLORS.get(cond, "#AAAAAA")
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            fill="toself", fillcolor=_hex_to_rgba(color, 0.12),
            line=dict(color=color, width=2), name=cond.replace("_", " "),
        ))
    fig.update_layout(
        polar=dict(bgcolor=BG,
                   radialaxis=dict(visible=True, range=[0, 1.1],
                                   gridcolor="rgba(255,255,255,0.08)"),
                   angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                                    tickfont=dict(size=10, color="rgba(255,255,255,0.7)"))),
        paper_bgcolor=BG, font=dict(color="rgba(255,255,255,0.8)"),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=60, t=30, b=30), height=350,
        showlegend=len(bsv_data["profiles"]) > 1,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# Reasoning traversal
st.divider()
st.subheader("Reasoning Traversal")
badge = {"strongly grounded": "green", "well grounded": "blue",
         "benchmark-supported": "violet", "partially grounded": "orange",
         "weakly grounded": "red"}.get(confidence["label"], "gray")
st.markdown(f"**Confidence:** :{badge}[{confidence['label']}]")
st.caption(confidence["explanation"])

fig_g = render_trust_graph(graph_data)
st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})
