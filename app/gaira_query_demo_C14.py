"""GAIRA Query Demo v1.4 — coverage-aware comparative reasoning.

Run: streamlit run app/gaira_query_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

st.set_page_config(page_title="GAIRA Query Demo", page_icon="🔬", layout="wide")
st.title("GAIRA — Graph-Grounded Spectral Interpretation")
st.caption("Deterministic graph queries | 1,887 evidence rows | 137 sources | 181 motifs | 750 neighborhoods")

with st.sidebar:
    st.header("Neo4j Connection")
    neo4j_uri = st.text_input("URI", value="bolt://localhost:7687")
    neo4j_user = st.text_input("User", value="neo4j")
    neo4j_pass = st.text_input("Password", value="gaira2026", type="password")
    st.divider()
    st.markdown("""
    **Single:** `What about HCC?`
    **Compare:** `Compare HCC vs healthy control`
    **Enrichment:** `What is enriched in HCC vs rest?`
    **Peak:** `Peak at 1005?`
    **Theme:** `Lipid signal?`
    **Chemistry:** `Amide I to biology?`
    """)
    st.caption("No LLM. Deterministic scoring.")

query = st.text_input("Ask GAIRA:", placeholder="e.g., Compare HCC vs healthy control")
if not query:
    st.info("Enter a query above."); st.stop()

from graph.phaseC1_query_router import route_query
from graph.phaseC1_query_engine import execute_query
from graph.phaseC1_scoring import score_result
from graph.phaseC1_templates import format_explanation, render_text_explanation
from app.graph_preview import build_preview_graph, LEGEND_HTML

parsed = route_query(query)
if parsed.query_type == "unknown":
    st.warning(f"Could not parse query. {parsed.notes}"); st.stop()

entity_display = (parsed.entities[0] if parsed.entities else str(parsed.peak_cm)).replace("_", " ")
mode_labels = {"single": "ASSOCIATIVE", "pairwise": "COMPARATIVE", "one_vs_rest": "ENRICHMENT"}
header = f"**{parsed.query_type.upper()}** | **{entity_display}**"
if parsed.comparator: header += f" vs **{parsed.comparator.replace('_',' ')}**"
header += f" | {mode_labels.get(parsed.query_mode,'')}"
st.info(header)

try:
    graph_result = execute_query(parsed, uri=neo4j_uri, user=neo4j_user, password=neo4j_pass)
except Exception as e:
    st.error(f"Neo4j connection failed: {e}"); st.stop()
if graph_result.total_evidence_rows == 0:
    st.warning("No evidence found."); st.stop()

scored = score_result(graph_result)
explanation = format_explanation(scored, graph_result)
is_comp = scored.query_mode in ("pairwise", "one_vs_rest")

# ── KPIs ────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Evidence", scored.evidence_count)
c2.metric("Sources", scored.source_count)
c3.metric("Motifs", len(scored.motif_summary))
c4.metric("Themes", len(scored.themes))

# ── COMPARATOR SUMMARY ─────────────────────────────────────
if scored.comparator_summary:
    cs = scored.comparator_summary
    with st.container():
        st.subheader("Comparator Summary")
        cc1, cc2, cc3 = st.columns(3)
        cc1.markdown(f"**Query**: {cs.query_condition.replace('_',' ')}  \n{cs.query_evidence} rows, {cs.query_sources} sources")
        cc2.markdown(f"**Comparator**: {cs.comparator_condition.replace('_',' ')}  \n{cs.comparator_evidence} rows, {cs.comparator_sources} sources")
        adequacy_color = {"adequate": "normal", "sparse": "off", "very_sparse": "off"}
        cc3.metric("Adequacy", cs.comparator_adequacy.replace("_", " ").upper())
        if cs.inferred_sample_type != "unknown":
            st.caption(f"Dominant sample type: **{cs.inferred_sample_type}**")

# ── THEME TABLE BUILDER ─────────────────────────────────────
def _theme_df(themes, show_comp=False):
    rows = []
    for t in themes:
        row = {
            "Theme": t["theme"] + (" (broad)" if t.get("is_broad") else ""),
            "Score": t["final_score"],
            "Conf.": t["confidence"].upper(),
        }
        if show_comp:
            row["Q direct"] = t.get("query_direct", 0)
            row["Q src"] = t.get("query_sources", 0)
            row["C direct"] = t.get("comp_direct", 0)
            row["C src"] = t.get("comp_sources", 0)
            row["Enrich"] = f"{t['enrichment_ratio']}x" if t.get("enrichment_ratio") else ""
            if t.get("coverage_flag"):
                row["Flag"] = t["coverage_flag"].replace("_", " ")
        else:
            row["Direct"] = t.get("direct_evidence", 0)
            row["Sources"] = t.get("source_diversity", 0)
            row["Motifs"] = t.get("motif_links", 0)
        rows.append(row)
    return pd.DataFrame(rows)

# ── THEMES ──────────────────────────────────────────────────
if is_comp:
    for key, title in [("D_enriched_themes", f"Enriched in {entity_display}"),
                       ("D2_associated_themes", "Associated (not clearly enriched)"),
                       ("D3_shared_themes", "Shared"),
                       ("D4_depleted_themes", "Depleted")]:
        items = explanation.get(key, [])
        if not items: continue
        if key in ("D3_shared_themes", "D4_depleted_themes"):
            with st.expander(f"{title} ({len(items)})"):
                st.dataframe(_theme_df(items, True), use_container_width=True, hide_index=True)
        else:
            st.subheader(title)
            st.dataframe(_theme_df(items, True), use_container_width=True, hide_index=True)
else:
    top = explanation.get("D_top_themes", [])
    sec = explanation.get("D2_secondary_themes", [])
    if top:
        st.subheader("Top Themes")
        st.dataframe(_theme_df(top), use_container_width=True, hide_index=True)
    if sec:
        with st.expander(f"Secondary Themes ({len(sec)})"):
            st.dataframe(_theme_df(sec), use_container_width=True, hide_index=True)

# ── MOTIFS + BIOMOLECULES ───────────────────────────────────
col_l, col_r = st.columns(2)
with col_l:
    st.subheader("Top Motifs")
    if scored.motif_summary:
        mrows = [{"Subfamily": m.subfamily.replace("_"," "), "Members": m.member_count,
                  "Status": m.motif_interpretation.replace("-"," ")} for m in scored.motif_summary[:6]]
        if is_comp:
            for i, m in enumerate(scored.motif_summary[:6]):
                mrows[i]["Comp."] = m.comparator_members
        st.dataframe(pd.DataFrame(mrows), use_container_width=True, hide_index=True)

with col_r:
    st.subheader("Top Biomolecules")
    if scored.top_biomolecules:
        st.dataframe(pd.DataFrame([{"Biomolecule": b["biomolecule"].replace("_"," "), "Evidence": b["count"]}
                                    for b in scored.top_biomolecules[:6]]), use_container_width=True, hide_index=True)

if scored.top_functional_groups:
    with st.expander("Functional Groups"):
        st.dataframe(pd.DataFrame([{"Group": f["functional_group"], "Count": f["count"],
                                     "Generic": "yes" if f.get("generic_flag") else ""}
                                    for f in scored.top_functional_groups[:6]]), use_container_width=True, hide_index=True)

st.subheader("Sample Evidence")
if scored.evidence_sample:
    st.dataframe(pd.DataFrame([{"Peak": ev.get("peak_cm",""), "Meaning": str(ev.get("meaning",""))[:70].replace("_"," "),
                                 "Level": str(ev.get("level","")).replace("_"," "), "Source": str(ev.get("source",""))[:25]}
                                for ev in scored.evidence_sample]), use_container_width=True, hide_index=True)

if scored.caveats:
    st.subheader("Caveats")
    for cav in scored.caveats: st.warning(cav)

# ── GRAPH PREVIEW ───────────────────────────────────────────
st.subheader("Graph Preview")
st.markdown(LEGEND_HTML, unsafe_allow_html=True)
graph_html = build_preview_graph(graph_result)
if graph_html: components.html(graph_html, height=480, scrolling=False)

# ── CYPHER + DETAILS ────────────────────────────────────────
cypher_label = "Comparative subgraph" if is_comp else "Associative subgraph"
with st.expander(f"Neo4j Browser ({cypher_label})"):
    st.code(graph_result.viz_cypher, language="cypher")
    st.markdown("1. Open Neo4j Browser (localhost:7474)\n2. Paste and run\n3. Graph view\n4. Double-click to expand")

with st.expander("Traversal Details"):
    exp_c = explanation["C_graph_expansion"]
    st.markdown(f"**Path**: {exp_c['traversal_summary']}")
    st.markdown(f"Direct: {exp_c['direct_support_edges']} | Inferred: {exp_c['inferred_support_edges']}")

with st.expander("Full Text Explanation"):
    st.markdown(render_text_explanation(explanation))
