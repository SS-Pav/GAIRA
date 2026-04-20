"""GAIRA v4 — Text Query Page.

Improvements over v3:
- Per-condition trust graphs for comparison queries
- Comparison summary above stacked graphs
- Cleaner evidence labels
- Condition-by-axis expected BSV heatmap
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from gaira.llm.response_schema import GAIRAResponse
from gaira.llm.local_synthesizer import synthesize_local_response
from gaira.retrieval.text_query_retriever import TextQueryRetriever
from gaira.retrieval.evidence_packet_builder import build_packet
from gaira.retrieval.section_linker import link_sections_to_evidence
from gaira.retrieval.confidence_composer import compose_confidence
from gaira.retrieval.literature_bsv_builder import (
    BSV_COMPONENTS, build_literature_bsv_profile, detect_conditions,
)
from gaira.retrieval.motif_theme_mapper import (
    THEME_DISPLAY as THEME_DISPLAY_MAP, map_evidence_to_motifs_themes_bsv,
)
from gaira.retrieval.trust_graph_builder import (
    TIER_COLORS, TIER_DISPLAY,
    build_trust_graph, build_per_condition_traversals,
)
from gaira.retrieval.trust_graph_render import render_trust_graph


EXAMPLE_QUERIES = [
    "What biochemical changes are associated with HCC in serum SERS spectra?",
    "How does HCC differ from healthy in serum SERS?",
    "Compare HCC, CCA, and liver metastases biochemical composition",
    "What are the 8 BSV components and what biochemistry do they capture?",
    "Compare healthy, HCC, and NAFLD",
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
st.caption("Literature-grounded biochemical Q&A. Comparison queries render per-condition trust graphs.")

retriever = get_retriever()

with st.sidebar:
    st.header("Text Query Settings")
    n_sec = sum(s["n_sections"] for s in retriever.source_summary())
    st.caption(f"{n_sec} sections · {len(retriever.documents)} sources")
    top_k = st.slider("Evidence items", 3, 12, 8)

    st.divider()
    response_mode = st.radio(
        "Response mode",
        ["Local synthesis", "Gemini LLM"],
        index=0,
        help="Local = no API. Gemini = external LLM.",
    )

example = st.selectbox("Example queries:", ["(custom)"] + EXAMPLE_QUERIES)
default_q = example if example != "(custom)" else st.session_state.get("tq_last_query", "")
query = st.text_area("Your question:", value=default_q, height=80)
run = st.button("Run Text Query", type="primary", disabled=not query.strip())


def _run_text_query(q: str, mode: str):
    with st.spinner("Retrieving evidence..."):
        retrieved = retriever.retrieve(q, top_k=top_k)

    if len(retrieved) < 2:
        st.warning("Too few evidence items.")
        st.stop()

    packet = build_packet(q, retrieved)
    bsv_data = build_literature_bsv_profile(q, retrieved)
    mtb_map = map_evidence_to_motifs_themes_bsv(retrieved)

    if mode == "Gemini LLM":
        from gaira.llm.gemini_client import generate_text
        from gaira.llm.prompt_builder import build_prompt
        try:
            with st.spinner("Querying Gemini..."):
                prompt = build_prompt(
                    user_query=q, evidence=packet["evidence"],
                    provenance=packet["provenance"], caveats=packet["caveats"],
                    domain_context=packet["domain_context"],
                )
                result = generate_text(prompt)
                raw_response = result.text
                parsed = GAIRAResponse.from_raw(raw_response)
                model_used = result.model_used
        except Exception as e:
            st.warning(f"Gemini unavailable: {e}. Falling back to local synthesis.")
            mode = "Local synthesis"

    if mode == "Local synthesis":
        synth = synthesize_local_response(q, retrieved, mtb_map, bsv_data, packet)
        parsed = GAIRAResponse(raw_text="[Local synthesis]", parse_success=True, **synth)
        raw_response = parsed.raw_text
        model_used = "local-synthesis"

    section_links = link_sections_to_evidence(parsed, retrieved)
    confidence = compose_confidence(retrieved, packet, section_links)

    # Detect conditions for comparison rendering
    conditions = bsv_data.get("conditions", [])

    # Build per-condition traversals if >1 condition
    per_cond_traversals = []
    if len(conditions) >= 2:
        per_cond_traversals = build_per_condition_traversals(
            q, retrieved, mtb_map, bsv_data, conditions, retriever=retriever,
        )

    # Single-condition fallback graph
    single_graph = build_trust_graph(q, retrieved, mtb_map, bsv_data)

    st.session_state.update({
        "tq_last_query": q, "tq_retrieved": retrieved, "tq_parsed": parsed,
        "tq_raw": raw_response, "tq_links": section_links, "tq_conf": confidence,
        "tq_bsv": bsv_data, "tq_mtb": mtb_map, "tq_packet": packet,
        "tq_model": model_used, "tq_mode": mode,
        "tq_conditions": conditions, "tq_per_cond": per_cond_traversals,
        "tq_single_graph": single_graph,
    })


if run and query.strip():
    _run_text_query(query.strip(), response_mode)

if "tq_parsed" not in st.session_state:
    st.stop()

# ── Render results ─────────────────────────────────────────────────────

parsed = st.session_state["tq_parsed"]
bsv_data = st.session_state["tq_bsv"]
confidence = st.session_state["tq_conf"]
per_cond = st.session_state["tq_per_cond"]
single_graph = st.session_state["tq_single_graph"]
conditions = st.session_state["tq_conditions"]
retrieved = st.session_state["tq_retrieved"]

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

# ── Literature-grounded expected BSV ───────────────────────────────────

if bsv_data.get("available"):
    st.divider()
    st.subheader("Literature-Grounded Expected Biochemical Composition")
    st.caption("Profiles from GAIRA's curated literature corpus. Post-hoc reference, not measured data.")

    cats = [BSV_DISPLAY.get(c, c) for c in BSV_COMPONENTS]
    profiles = bsv_data["profiles"]

    # Radar
    fig = go.Figure()
    for cond, prof in profiles.items():
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
        showlegend=len(profiles) > 1,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Expected BSV heatmap per condition
    if len(profiles) >= 1:
        st.markdown("##### Expected BSV by Condition")
        hmat = np.array([[profiles[c].get(comp, 0) for comp in BSV_COMPONENTS]
                          for c in profiles.keys()])
        fig_h = go.Figure(go.Heatmap(
            z=hmat, x=cats,
            y=[c.replace("_", " ") for c in profiles.keys()],
            colorscale="YlOrRd", texttemplate="%{z:.2f}", textfont=dict(size=10),
        ))
        fig_h.update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
                             height=max(180, 55*len(profiles)),
                             margin=dict(l=10, r=10, t=20, b=10),
                             font=dict(color="rgba(255,255,255,0.8)"))
        st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})

    # Delta-vs-reference if healthy is present
    if "healthy_control" in profiles and len(profiles) > 1:
        st.markdown("##### Expected Delta vs healthy_control")
        ref_prof = profiles["healthy_control"]
        delta_rows = []
        delta_labels = []
        for c, prof in profiles.items():
            if c == "healthy_control":
                continue
            delta_rows.append([prof.get(comp, 0) - ref_prof.get(comp, 0) for comp in BSV_COMPONENTS])
            delta_labels.append(c.replace("_", " "))
        if delta_rows:
            dmat = np.array(delta_rows)
            vm = max(abs(dmat.min()), abs(dmat.max())) * 1.1 or 0.01
            fig_d = go.Figure(go.Heatmap(
                z=dmat, x=cats, y=delta_labels,
                colorscale="RdBu_r", zmid=0, zmin=-vm, zmax=vm,
                texttemplate="%{z:+.2f}", textfont=dict(size=10),
            ))
            fig_d.update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
                                 height=max(180, 55*len(delta_rows)),
                                 margin=dict(l=10, r=10, t=20, b=10),
                                 font=dict(color="rgba(255,255,255,0.8)"))
            st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar": False})

# ── Trust graphs ──────────────────────────────────────────────────────

st.divider()
st.subheader("Reasoning Traversal")

# Overall confidence
badge = {"strongly grounded": "green", "well grounded": "blue",
         "benchmark-supported": "violet", "partially grounded": "orange",
         "weakly grounded": "red"}.get(confidence["label"], "gray")
st.markdown(f"**Overall confidence:** :{badge}[{confidence['label']}]")
st.caption(confidence["explanation"])

if per_cond:
    # Comparison query — per-condition graphs stacked
    st.markdown(f"##### Comparison across {len(per_cond)} conditions")

    # Summary table
    rows = []
    for trav in per_cond:
        cond = trav["condition"]
        graph = trav["graph"]
        summary = graph.get("summary", {})
        rows.append({
            "Condition": cond.replace("_", " "),
            "Evidence items": trav["n_evidence"],
            "Motifs detected": summary.get("n_motifs", 0),
            "Active BSV axes": summary.get("n_bsv_active", 0),
        })
    st.table(rows)

    st.caption(
        "Each condition has its own trust graph below, showing the evidence "
        "and motifs supporting its expected biochemical profile."
    )

    for trav in per_cond:
        cond = trav["condition"]
        st.markdown(f"---")
        st.markdown(f"##### Expected literature traversal — **{cond.replace('_', ' ')}**")
        fig_tg = render_trust_graph(trav["graph"])
        st.plotly_chart(fig_tg, use_container_width=True,
                         config={"displayModeBar": False})
else:
    # Single-condition / generic query — one graph
    fig_tg = render_trust_graph(single_graph)
    st.plotly_chart(fig_tg, use_container_width=True, config={"displayModeBar": False})
