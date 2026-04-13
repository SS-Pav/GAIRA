"""GAIRA_LFM_v1 — Literature-Grounded Text Query.

GAIRA-native text query with:
- evidence-tiered retrieval
- motif → theme → BSV mapping
- literature-grounded BSV radar plots
- 6-column trust graph
- local synthesis mode (no external LLM required)
- optional Gemini mode with model fallback

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v1_text_query.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import plotly.graph_objects as go
import streamlit as st

from gaira.llm.response_schema import GAIRAResponse
from gaira.llm.local_synthesizer import synthesize_local_response
from gaira.retrieval.text_query_retriever import TextQueryRetriever
from gaira.retrieval.evidence_packet_builder import build_packet
from gaira.retrieval.section_linker import link_sections_to_evidence
from gaira.retrieval.confidence_composer import compose_confidence
from gaira.retrieval.literature_bsv_builder import (
    BSV_COMPONENTS,
    build_literature_bsv_profile,
)
from gaira.retrieval.motif_theme_mapper import (
    THEME_DISPLAY as THEME_DISPLAY_MAP,
    map_evidence_to_motifs_themes_bsv,
)
from gaira.retrieval.trust_graph_builder import (
    TIER_COLORS, TIER_DISPLAY, build_trust_graph, build_per_condition_traversals,
)
from gaira.retrieval.source_registry import TIER_DISPLAY_NAMES
from gaira.retrieval.trust_graph_render import render_trust_graph


EXAMPLE_QUERIES = [
    "What biochemical changes are associated with HCC in serum SERS spectra?",
    "How does HCC differ from healthy in serum SERS?",
    "Compare HCC, CCA, and liver metastases biochemical composition",
    "What are the most reliable Raman biomarkers for liver disease?",
    "Why do nucleic acid bands behave differently on Au vs AgNP substrates?",
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

MIN_EVIDENCE_ITEMS = 2


@st.cache_resource
def get_retriever():
    r = TextQueryRetriever()
    r.load_sources()
    return r


# ── Radar plot ─────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(170,170,170,{alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _render_bsv_radar(bsv_data: dict):
    profiles = bsv_data.get("profiles", {})
    if not profiles:
        return

    categories = [BSV_DISPLAY.get(c, c) for c in BSV_COMPONENTS]
    fig = go.Figure()
    for cond, prof in profiles.items():
        values = [prof.get(c, 0) for c in BSV_COMPONENTS]
        values.append(values[0])
        color = CONDITION_COLORS.get(cond, "#AAAAAA")
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories + [categories[0]],
            fill="toself", fillcolor=_hex_to_rgba(color, 0.12),
            line=dict(color=color, width=2),
            name=cond.replace("_", " "),
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#1a1a2e",
            radialaxis=dict(visible=True, range=[0, 1.1],
                            gridcolor="rgba(255,255,255,0.08)",
                            tickfont=dict(size=8, color="rgba(255,255,255,0.4)")),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                             tickfont=dict(size=10, color="rgba(255,255,255,0.7)")),
        ),
        paper_bgcolor="#1a1a2e",
        font=dict(color="rgba(255,255,255,0.8)"),
        legend=dict(font=dict(size=10, color="rgba(255,255,255,0.7)"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=60, t=30, b=30), height=350,
        showlegend=len(profiles) > 1,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Rendering helpers ──────────────────────────────────────────────────

def _render_sidebar_evidence(retrieved):
    with st.sidebar:
        st.divider()
        st.subheader("Retrieved Evidence")
        for i, item in enumerate(retrieved, 1):
            tier = TIER_DISPLAY.get(item.source_tier, item.source_tier)
            color = TIER_COLORS.get(item.source_tier, "#999")
            header = f"{i}. {item.title or '(untitled)'}"
            with st.expander(header, expanded=False):
                st.markdown(
                    f"<span style='color:{color}'>●</span> **{tier}** · "
                    f"score {item.retrieval_score} · `{item.source.split('/')[-1]}`",
                    unsafe_allow_html=True,
                )
                st.markdown(item.text)


def _render_section_supports(field: str, supports: list[dict]):
    if not supports:
        return
    parts = []
    for s in supports[:3]:
        tier = TIER_DISPLAY.get(s["tier"], s["tier"])
        color = TIER_COLORS.get(s["tier"], "#999")
        parts.append(
            f"<span style='color:{color}'>●</span> {s['title'][:40]} "
            f"<small>({tier}, {s['support_score']:.2f})</small>"
        )
    st.markdown(
        "<small>Supported by: " + " &nbsp;·&nbsp; ".join(parts) + "</small>",
        unsafe_allow_html=True,
    )


def _render_trust_summary(confidence: dict, summary: dict):
    label = confidence["label"]
    badge_map = {
        "strongly grounded": "green", "well grounded": "blue",
        "benchmark-supported": "violet", "partially grounded": "orange",
        "weakly grounded": "red", "no evidence": "red",
    }
    color = badge_map.get(label, "gray")
    cols = st.columns([2.5, 1, 1, 1, 1, 1])
    cols[0].markdown(f"**Confidence:** :{color}[{label}]")
    cols[1].metric("Grounded", confidence["grounded_count"])
    cols[2].metric("Context", confidence["context_count"])
    cols[3].metric("Motifs", summary.get("n_motifs", 0))
    cols[4].metric("Themes", summary.get("n_themes", 0))
    cols[5].metric("BSV Axes", summary.get("n_bsv_active", 0))
    st.caption(confidence["explanation"])


# ── App ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="GAIRA_LFM_v1", layout="wide")

st.title("GAIRA_LFM_v1")
st.caption(
    "Literature-grounded biochemical reasoning engine. "
    "Query → Evidence → Motifs → Themes → BSV composition → Answer."
)

st.info(
    "**GAIRA** interprets Raman/SERS biochemistry through its structured evidence corpus — "
    "curated component definitions, context sources, and analysis summaries. "
    "Not a classifier. Not spectral query. Not direct paper retrieval.",
    icon="🔬",
)

retriever = get_retriever()

# ── Sidebar ────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Source Registry")
    n_sec = sum(s["n_sections"] for s in retriever.source_summary())
    st.caption(f"{n_sec} sections · {len(retriever.documents)} documents")
    with st.expander("Sources by tier", expanded=False):
        current_tier = ""
        for s in sorted(retriever.source_summary(), key=lambda x: x["tier"]):
            if s["tier"] != current_tier:
                current_tier = s["tier"]
                st.markdown(f"**{TIER_DISPLAY_NAMES.get(current_tier, current_tier)}**")
            st.markdown(f"- `{s['path']}` ({s['n_sections']} sec)")

    top_k = st.slider("Evidence items", min_value=3, max_value=12, value=8)

    st.divider()
    response_mode = st.radio(
        "Response mode",
        ["Local synthesis", "Gemini"],
        index=0,
        help="Local synthesis generates a structured response without an external LLM call. "
             "Gemini mode requires API access and may hit rate limits.",
    )

# ── Query ──────────────────────────────────────────────────────────────

st.subheader("Query")
example = st.selectbox("Start from an example, or type your own:", ["(custom)"] + EXAMPLE_QUERIES)
default_query = example if example != "(custom)" else st.session_state.get("last_query", "")

query = st.text_area(
    "Your question:", value=default_query, height=80,
    placeholder="e.g. What biochemical changes are associated with HCC in serum SERS?",
)
run_clicked = st.button("Run GAIRA query", type="primary", disabled=not query.strip())


# ── Pipeline ───────────────────────────────────────────────────────────

def _run_pipeline(q: str, mode: str):
    with st.spinner("Retrieving evidence..."):
        retrieved = retriever.retrieve(q, top_k=top_k)

    if len(retrieved) < MIN_EVIDENCE_ITEMS:
        st.warning(f"Only {len(retrieved)} evidence items found. Try broadening your query.")
        st.stop()

    packet = build_packet(q, retrieved)
    bsv_data = build_literature_bsv_profile(q, retrieved)
    mtb_map = map_evidence_to_motifs_themes_bsv(retrieved)

    # ── Generate response ──────────────────────────────────────
    if mode == "Gemini":
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
            except RuntimeError as e:
                st.warning(f"Gemini unavailable: {e}\n\nFalling back to local synthesis.")
                mode = "Local synthesis"  # fall through below

        if mode == "Gemini":
            raw_response = result.text
            parsed = GAIRAResponse.from_raw(raw_response)
            model_used = result.model_used
            fallback_used = result.fallback_used
        # else: falls through to local synthesis

    if mode == "Local synthesis":
        synth = synthesize_local_response(
            query=q,
            retrieved_items=retrieved,
            motifs_themes_bsv=mtb_map,
            literature_bsv_profile=bsv_data,
            evidence_packet=packet,
        )
        # Build a GAIRAResponse from the synth dict
        parsed = GAIRAResponse(
            raw_text="[Local synthesis — no external LLM call]",
            answer_summary=synth["answer_summary"],
            biochemical_themes=synth["biochemical_themes"],
            strongest_evidence=synth["strongest_evidence"],
            supporting_evidence=synth["supporting_evidence"],
            caveats=synth["caveats"],
            confidence_notes=synth["confidence_notes"],
            parse_success=True,
        )
        raw_response = parsed.raw_text
        model_used = "local-synthesis"
        fallback_used = False

    section_links = link_sections_to_evidence(parsed, retrieved)
    confidence = compose_confidence(retrieved, packet, section_links)
    graph_data = build_trust_graph(q, retrieved, mtb_map, bsv_data)

    # Per-condition traversals for comparison queries
    conditions = bsv_data.get("conditions", [])
    if len(conditions) >= 2:
        per_cond_traversals = build_per_condition_traversals(
            q, retrieved, mtb_map, bsv_data, conditions,
            retriever=retriever,
        )
    else:
        per_cond_traversals = []

    st.session_state.update({
        "last_query": q, "last_retrieved": retrieved, "last_packet": packet,
        "last_bsv": bsv_data, "last_mtb": mtb_map,
        "last_raw": raw_response, "last_parsed": parsed,
        "last_section_links": section_links, "last_confidence": confidence,
        "last_graph": graph_data,
        "last_per_cond": per_cond_traversals,
        "last_model_used": model_used,
        "last_fallback_used": fallback_used,
        "last_mode": mode,
    })


def _render_results():
    parsed = st.session_state["last_parsed"]
    raw_response = st.session_state["last_raw"]
    retrieved = st.session_state["last_retrieved"]
    section_links = st.session_state["last_section_links"]
    confidence = st.session_state["last_confidence"]
    graph_data = st.session_state["last_graph"]
    bsv_data = st.session_state["last_bsv"]
    mtb_map = st.session_state["last_mtb"]

    _render_sidebar_evidence(retrieved)

    # ── Response ───────────────────────────────────────────────
    st.divider()
    st.subheader("GAIRA Response")

    mode = st.session_state.get("last_mode", "Local synthesis")
    model_used = st.session_state.get("last_model_used", "unknown")
    fallback_used = st.session_state.get("last_fallback_used", False)

    if mode == "Gemini":
        if fallback_used:
            st.caption(f"Model: `{model_used}` (fallback — primary quota exceeded)")
        else:
            st.caption(f"Model: `{model_used}`")
    else:
        st.caption("Mode: **Local synthesis** — no external LLM call")

    if parsed.parse_success:
        for attr, label in PARSED_SECTIONS:
            content = getattr(parsed, attr, "")
            if content:
                st.markdown(f"#### {label}")
                st.markdown(content)
                _render_section_supports(attr, section_links.get(attr, []))
    else:
        st.warning("Structured parsing incomplete. Showing raw response.")
        st.markdown(raw_response)

    # ── Debug / inputs view ────────────────────────────────────
    with st.expander("Synthesis inputs / debug", expanded=False):
        if mode == "Gemini":
            st.markdown("**Raw Gemini response:**")
            st.text(raw_response)
        else:
            st.markdown("**Synthesis basis:**")
            # Retrieved sources
            st.markdown(f"- **Retrieved items:** {len(retrieved)}")
            tier_counts: dict[str, int] = {}
            for it in retrieved:
                t = it.source_tier or "unknown"
                tier_counts[t] = tier_counts.get(t, 0) + 1
            tier_str = ", ".join(f"{k}={v}" for k, v in sorted(tier_counts.items()))
            st.markdown(f"- **Tier mix:** {tier_str}")

            # Detected conditions
            conditions = bsv_data.get("conditions", [])
            st.markdown(f"- **Detected conditions:** {', '.join(conditions) if conditions else '(none)'}")

            # Motifs and themes
            motifs = mtb_map.get("motifs", [])
            themes = mtb_map.get("themes", [])
            st.markdown(f"- **Motifs detected:** {len(motifs)}")
            st.markdown(f"- **Themes detected:** {len(themes)}")

            # BSV profile
            if bsv_data.get("available"):
                st.markdown("- **BSV profiles:**")
                for cond, prof in bsv_data.get("profiles", {}).items():
                    top = sorted(prof.items(), key=lambda x: -x[1])[:4]
                    top_str = ", ".join(f"{k}={v:.2f}" for k, v in top if v > 0)
                    st.markdown(f"  - {cond}: {top_str}")

    # ── BSV Radar ──────────────────────────────────────────────
    if bsv_data.get("available"):
        st.divider()
        st.subheader("Reference BSV Profile")
        st.caption(
            "Literature-grounded Biochemical State Vector from GAIRA's curated registry. "
            "Reflects relative evidence support across published studies — "
            "not measured spectral composition."
        )
        _render_bsv_radar(bsv_data)

        conditions = bsv_data.get("conditions", [])
        from gaira.retrieval.literature_bsv_builder import CONDITION_ALIASES
        alias_notes = []
        for cond in conditions:
            aliases = [k for k, v in CONDITION_ALIASES.items() if v == cond]
            if aliases and cond != aliases[0]:
                alias_notes.append(f"'{aliases[0]}' → `{cond}`")
        if alias_notes:
            st.caption("Condition mappings: " + " · ".join(alias_notes))
        if bsv_data.get("notes"):
            st.caption(bsv_data["notes"])

    # ── Evidence Chain ─────────────────────────────────────────
    st.divider()
    per_cond = st.session_state.get("last_per_cond", [])

    if per_cond:
        # Comparison mode: per-condition traversals
        st.subheader("Comparison Traversals")
        st.caption(
            f"Separate reasoning chains for {len(per_cond)} conditions. "
            "Each shows its own evidence → motifs → themes → BSV path."
        )

        # Overall summary first
        _render_trust_summary(confidence, graph_data["summary"])

        for trav in per_cond:
            cond = trav["condition"]
            cond_graph = trav["graph"]
            n_ev = trav["n_evidence"]
            cond_summary = cond_graph["summary"]

            st.markdown(f"---")
            st.markdown(f"##### {cond.replace('_', ' ')}  "
                        f"<small>({n_ev} evidence items · "
                        f"{cond_summary.get('n_motifs', 0)} motifs · "
                        f"{cond_summary.get('n_themes', 0)} themes · "
                        f"{cond_summary.get('n_bsv_active', 0)} BSV axes)</small>",
                        unsafe_allow_html=True)

            fig = render_trust_graph(cond_graph, height=340)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        # Single condition: one traversal
        st.subheader("Reasoning Traversal")
        _render_trust_summary(confidence, graph_data["summary"])

        st.caption("Query → Evidence → Motifs → Themes → BSV → Output. Hover nodes for details.")
        fig = render_trust_graph(graph_data)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Inspector ──────────────────────────────────────────────
    with st.expander("Detected motifs & themes", expanded=False):
        themes = mtb_map.get("themes", [])
        motifs = mtb_map.get("motifs", [])
        if themes:
            st.markdown("**Themes** (evidence-supported biochemical categories)")
            for t in themes:
                st.markdown(
                    f"- **{t['display']}** — {t['evidence_count']} evidence items, "
                    f"motifs: {', '.join(t['motifs'][:4])}"
                )
        if motifs:
            st.markdown("**Motifs** (specific molecular features detected)")
            for m in motifs[:10]:
                peaks = m.get("peaks", "")
                peak_str = f" · {peaks}" if peaks else ""
                theme_display = THEME_DISPLAY_MAP.get(m["theme"], m["theme"])
                st.markdown(
                    f"- **{m['name']}**{peak_str} → {theme_display} ({m['hit_count']} hits)"
                )

    with st.expander("Section support map", expanded=False):
        for field, display in PARSED_SECTIONS:
            supports = section_links.get(field, [])
            if supports:
                items_str = ", ".join(
                    f"{s['title'][:30]} ({s['support_score']:.2f})" for s in supports[:3]
                )
                st.markdown(f"**{display}** → {items_str}")
            else:
                st.markdown(f"**{display}** → *(no direct link)*")


if run_clicked and query.strip():
    _run_pipeline(query.strip(), response_mode)
    _render_results()
elif "last_parsed" in st.session_state:
    _render_results()
