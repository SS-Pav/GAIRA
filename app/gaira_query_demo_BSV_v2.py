# GAIRA QUERY DEMO — BSV v2
# Refined radar overlay, delta bars, side-by-side table, component confidence

VERSION = "BSV v2"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

st.set_page_config(page_title=f"GAIRA {VERSION}", page_icon="🔬", layout="wide")
st.title(f"GAIRA — Biochemical State Vector {VERSION}")
st.caption(f"Version {VERSION} | Radar overlay + delta bars + component confidence | Graph-backed explanations")

with st.sidebar:
    st.header("Neo4j Connection")
    neo4j_uri = st.text_input("URI", value="bolt://localhost:7687")
    neo4j_user = st.text_input("User", value="neo4j")
    neo4j_pass = st.text_input("Password", value="gaira2026", type="password")
    st.divider()
    st.markdown("""
    **BSV queries:**
    `Compare HCC vs healthy serum within liver sources`
    `Compare NAFLD vs healthy serum within liver sources`
    `Compare HCC vs NAFLD in serum`
    `What about HCC?`
    """)
    st.caption(f"{VERSION}: improved radar overlay, delta readability, component confidence.")

query = st.text_input("Ask GAIRA:", placeholder="e.g., Compare HCC vs healthy serum within liver sources")
if not query: st.info("Enter a query above."); st.stop()

from graph.phaseC1_query_router import route_query
from graph.phaseC1_query_engine import execute_query
from graph.phaseC1_scoring import score_result
from graph.phaseC1_templates import format_explanation, render_text_explanation
from graph.bsv_v1_scoring import compute_bsv_comparison
from graph.bsv_v1_explainer import generate_bsv_explanation
from app.bsv_v1_visuals import render_radar_plot, render_delta_plot
from app.graph_preview import build_preview_graph, LEGEND_HTML

parsed = route_query(query)
if parsed.query_type == "unknown": st.warning(parsed.notes); st.stop()

entity = (parsed.entities[0] if parsed.entities else str(parsed.peak_cm)).replace("_"," ")
mode_labels = {"single":"ASSOCIATIVE","pairwise":"COMPARATIVE","one_vs_rest":"ENRICHMENT"}
header = f"**{parsed.query_type.upper()}** | **{entity}**"
if parsed.comparator: header += f" vs **{parsed.comparator.replace('_',' ')}**"
header += f" | {mode_labels.get(parsed.query_mode,'')}"
if parsed.sample_scope: header += f" | {parsed.sample_scope}"
st.info(header)

try: gr = execute_query(parsed, uri=neo4j_uri, user=neo4j_user, password=neo4j_pass)
except Exception as e: st.error(f"Neo4j: {e}"); st.stop()
if gr.total_evidence_rows == 0: st.warning("No evidence."); st.stop()

scored = score_result(gr)
explanation = format_explanation(scored, gr)
is_comp = scored.query_mode in ("pairwise","one_vs_rest")
bsv = compute_bsv_comparison(scored)

# ── KPIs ────────────────────────────────────────────────────
k1,k2,k3,k4 = st.columns(4)
k1.metric("Evidence", scored.evidence_count)
k2.metric("Sources", scored.source_count)
k3.metric("Motifs", len(scored.motif_summary))
k4.metric("BSV Components", sum(1 for c in bsv.query_bsv.components if c.coverage_note != "absent"))

# ── COMPARATOR SUMMARY ─────────────────────────────────────
if scored.comparator_summary:
    cs = scored.comparator_summary
    cc1,cc2,cc3 = st.columns(3)
    cc1.markdown(f"**Query**: {cs.query_condition.replace('_',' ')} ({cs.query_evidence} rows)")
    cc2.markdown(f"**Comparator**: {cs.comparator_condition.replace('_',' ')} ({cs.comparator_evidence} rows)")
    cc3.metric("Adequacy", cs.comparator_adequacy.replace("_"," ").upper())

# ── BSV RADAR + DELTA ───────────────────────────────────────
st.subheader("Biochemical State Vector")

if is_comp:
    col_radar, col_delta = st.columns([1.1, 0.9])
    with col_radar:
        fig_r = render_radar_plot(bsv)
        st.pyplot(fig_r)
    with col_delta:
        fig_d = render_delta_plot(bsv)
        if fig_d: st.pyplot(fig_d)
else:
    fig_r = render_radar_plot(bsv)
    st.pyplot(fig_r)

import matplotlib.pyplot as plt
plt.close("all")

# ── SIDE-BY-SIDE BSV TABLE ──────────────────────────────────
st.subheader("BSV Component Detail")

_CONF_COLORS = {"strong": "🟢", "moderate": "🟡", "weak": "🟠", "exploratory": "⚪"}

bsv_table_rows = []
for i, qc in enumerate(bsv.query_bsv.components):
    name = qc.name.replace("_", " ").title()
    row = {
        "Component": name,
        "Query": round(qc.normalized_score, 3),
        "Raw Q": round(qc.raw_score, 2),
    }
    if bsv.comparator_bsv:
        cc = bsv.comparator_bsv.components[i]
        row["Comparator"] = round(cc.normalized_score, 3)
        row["Raw C"] = round(cc.raw_score, 2)
        delta = next((d for d in bsv.delta_components if d["component"] == qc.name), None)
        if delta:
            row["Delta"] = f"{delta['delta']:+.3f}"
            row["Shift"] = delta["direction"]
    row["Stability"] = qc.dominant_stability
    row["Conf."] = f"{_CONF_COLORS.get(qc.confidence,'')} {qc.confidence}"
    row["Motifs"] = qc.motif_count
    bsv_table_rows.append(row)

st.dataframe(pd.DataFrame(bsv_table_rows), use_container_width=True, hide_index=True)

# ── BSV COMPONENT EXPLANATIONS ──────────────────────────────
bsv_exps = generate_bsv_explanation(bsv)
st.subheader("Component Explanations")
for exp in bsv_exps:
    if exp.get("coverage") == "absent":
        continue
    conf_icon = _CONF_COLORS.get(exp.get("confidence",""), "")
    with st.expander(f"{conf_icon} {exp['component'].replace('_',' ').title()} — score {exp.get('normalized_score',0):.2f}"):
        st.markdown(exp.get("summary",""))
        if exp.get("contributing_motifs"):
            st.caption(f"**Motifs**: {', '.join(exp['contributing_motifs'][:5])}")
        if exp.get("delta") is not None:
            st.caption(f"**Delta**: {exp['delta']:+.3f} ({exp.get('delta_direction','')})")
        if exp.get("neo4j_query"):
            st.code(exp["neo4j_query"], language="cypher")

# ── MOTIF DIFFERENTIAL ──────────────────────────────────────
if scored.motif_differentials:
    with st.expander(f"Motif Differential + Stability ({len(scored.motif_differentials)} motifs)"):
        md_rows = [{"Motif": d.subfamily.replace("_"," ") or d.motif_id,
                    "Q": d.q_direct, "C": d.c_direct,
                    "Norm": f"{d.norm_ratio}x", "Stability": d.stability_label,
                    "Interp.": d.interpretation.replace("_"," ")}
                   for d in scored.motif_differentials[:10]]
        st.dataframe(pd.DataFrame(md_rows), use_container_width=True, hide_index=True)

# ── THEMES + EVIDENCE + CAVEATS ─────────────────────────────
with st.expander("Theme Scores"):
    top = explanation.get("D_top_themes") or explanation.get("D_enriched_themes",[])
    if top:
        st.dataframe(pd.DataFrame([{"Theme": t["theme"], "Score": t["final_score"],
                                     "Conf.": t["confidence"].upper()} for t in top[:6]]),
                     use_container_width=True, hide_index=True)

with st.expander("Sample Evidence"):
    if scored.evidence_sample:
        st.dataframe(pd.DataFrame([{"Peak": ev.get("peak_cm",""),
                                     "Meaning": str(ev.get("meaning",""))[:60].replace("_"," "),
                                     "Source": str(ev.get("source",""))[:25]}
                                    for ev in scored.evidence_sample[:6]]),
                     use_container_width=True, hide_index=True)

if scored.caveats:
    with st.expander(f"Caveats ({len(scored.caveats)})"):
        for c in scored.caveats: st.warning(c)

# ── GRAPH + CYPHER ──────────────────────────────────────────
with st.expander("Graph Preview"):
    st.markdown(LEGEND_HTML, unsafe_allow_html=True)
    gh = build_preview_graph(gr)
    if gh: components.html(gh, height=380, scrolling=False)

with st.expander("Neo4j Cypher"):
    st.code(gr.viz_cypher, language="cypher")

with st.expander("Full Text"):
    st.markdown(render_text_explanation(explanation))
