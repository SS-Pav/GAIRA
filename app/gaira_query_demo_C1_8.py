# GAIRA QUERY DEMO — C1.8
# Biochemical State Vector (BSV) v1 + Radar/Delta Visualization

VERSION = "C1.8"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

st.set_page_config(page_title=f"GAIRA {VERSION}", page_icon="🔬", layout="wide")
st.title(f"GAIRA — Biochemical State Vector {VERSION}")
st.caption(f"Version {VERSION} | BSV radar + delta | Motif differential + stability | 1,887 evidence rows")

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

    **Standard:**
    `What about HCC?`  `Peak 1005?`
    """)
    st.caption(f"{VERSION}: BSV biochemical abstraction over motif differential/stability.")

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

# ── KPIs ────────────────────────────────────────────────────
k1,k2,k3,k4 = st.columns(4)
k1.metric("Evidence", scored.evidence_count)
k2.metric("Sources", scored.source_count)
k3.metric("Motifs", len(scored.motif_summary))
k4.metric("Themes", len(scored.themes))

# ── COMPARATOR SUMMARY ─────────────────────────────────────
if scored.comparator_summary:
    cs = scored.comparator_summary
    with st.container():
        cc1,cc2,cc3 = st.columns(3)
        cc1.markdown(f"**Query**: {cs.query_condition.replace('_',' ')} ({cs.query_evidence} rows)")
        cc2.markdown(f"**Comparator**: {cs.comparator_condition.replace('_',' ')} ({cs.comparator_evidence} rows)")
        cc3.metric("Adequacy", cs.comparator_adequacy.replace("_"," ").upper())

# ── BSV SECTION ─────────────────────────────────────────────
bsv_comparison = compute_bsv_comparison(scored)

st.subheader("Biochemical State Vector")

# Radar + Delta side by side
if is_comp:
    radar_col, delta_col = st.columns(2)
    with radar_col:
        fig_radar = render_radar_plot(bsv_comparison)
        st.pyplot(fig_radar)
    with delta_col:
        fig_delta = render_delta_plot(bsv_comparison)
        if fig_delta:
            st.pyplot(fig_delta)
else:
    fig_radar = render_radar_plot(bsv_comparison)
    st.pyplot(fig_radar)

# BSV component table
st.subheader("BSV Component Scores")
bsv_rows = []
for c in bsv_comparison.query_bsv.components:
    row = {
        "Component": c.name.replace("_"," ").title(),
        "Score": c.normalized_score,
        "Raw": round(c.raw_score, 2),
        "Motifs": c.motif_count,
        "Stability": c.dominant_stability,
        "Coverage": c.coverage_note,
    }
    if bsv_comparison.delta_components:
        d = next((d for d in bsv_comparison.delta_components if d["component"] == c.name), None)
        if d:
            row["Delta"] = f"{d['delta']:+.3f}"
            row["Shift"] = d["direction"]
    bsv_rows.append(row)
st.dataframe(pd.DataFrame(bsv_rows), use_container_width=True, hide_index=True)

# Component explanations
bsv_exps = generate_bsv_explanation(bsv_comparison)
with st.expander("BSV Component Explanations"):
    for exp in bsv_exps:
        if exp.get("coverage") != "absent":
            st.markdown(f"**{exp['component'].replace('_',' ').title()}**: {exp.get('summary','')}")
            if exp.get("contributing_motifs"):
                st.caption(f"Motifs: {', '.join(exp['contributing_motifs'][:4])}")

# ── MOTIF DIFFERENTIAL + STABILITY ─────────────────────────
if scored.motif_differentials:
    st.subheader("Motif Differential + Stability")
    md_rows = [{"Motif": d.subfamily.replace("_"," ") or d.motif_id,
                "Q": d.q_direct, "C": d.c_direct,
                "Norm": f"{d.norm_ratio}x", "Bal.": f"{d.balance:.2f}",
                "Interp.": d.interpretation.replace("_"," "),
                "Studies": d.studies_total,
                "E": d.studies_enriched, "S": d.studies_shared, "D": d.studies_depleted,
                "Stability": d.stability_label,
                } for d in scored.motif_differentials[:10]]
    st.dataframe(pd.DataFrame(md_rows), use_container_width=True, hide_index=True)

    # Stability summary
    ss = explanation.get("E3_stability_summary")
    if ss:
        scol1,scol2,scol3,scol4 = st.columns(4)
        if ss["stable"]: scol1.success(f"STABLE: {', '.join(ss['stable'][:4])}")
        else: scol1.info("No stable")
        if ss["mixed"]: scol2.warning(f"MIXED: {', '.join(ss['mixed'][:4])}")
        if ss["unstable"]: scol3.error(f"UNSTABLE: {', '.join(ss['unstable'][:3])}")
        if ss["insufficient"]: scol4.info(f"INSUFF: {', '.join(ss['insufficient'][:3])}")

# ── THEMES (compact) ───────────────────────────────────────
with st.expander("Theme-Level Scores"):
    top = explanation.get("D_top_themes") or explanation.get("D_enriched_themes", [])
    if top:
        st.dataframe(pd.DataFrame([{
            "Theme": t["theme"], "Score": t["final_score"], "Conf.": t["confidence"].upper(),
        } for t in top[:6]]), use_container_width=True, hide_index=True)

# ── EVIDENCE + CAVEATS ─────────────────────────────────────
with st.expander("Sample Evidence"):
    if scored.evidence_sample:
        st.dataframe(pd.DataFrame([{"Peak": ev.get("peak_cm",""), "Meaning": str(ev.get("meaning",""))[:60].replace("_"," "),
                                     "Source": str(ev.get("source",""))[:25]}
                                    for ev in scored.evidence_sample[:6]]), use_container_width=True, hide_index=True)

if scored.caveats:
    with st.expander(f"Caveats ({len(scored.caveats)})"):
        for c in scored.caveats: st.warning(c)

# ── GRAPH PREVIEW ───────────────────────────────────────────
with st.expander("Graph Preview"):
    st.markdown(LEGEND_HTML, unsafe_allow_html=True)
    gh = build_preview_graph(gr)
    if gh: components.html(gh, height=400, scrolling=False)

with st.expander("Neo4j Cypher"):
    st.code(gr.viz_cypher, language="cypher")

with st.expander("Full Text"):
    st.markdown(render_text_explanation(explanation))
