"""GAIRA Query Demo v1.6 — scope-aware comparator with sample/domain filtering.

Run: streamlit run app/gaira_query_demo_C1_6.py
"""

VERSION = "C1.6"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

st.set_page_config(page_title=f"GAIRA {VERSION}", page_icon="🔬", layout="wide")
st.title(f"GAIRA — Graph-Grounded Spectral Interpretation {VERSION}")
st.caption(f"Version {VERSION} | 1,887 evidence rows | 137 sources | Scope-aware comparative scoring")

with st.sidebar:
    st.header("Neo4j Connection")
    neo4j_uri = st.text_input("URI", value="bolt://localhost:7687")
    neo4j_user = st.text_input("User", value="neo4j")
    neo4j_pass = st.text_input("Password", value="gaira2026", type="password")
    st.divider()
    st.markdown("""
    **Scoped queries (new):**
    `Compare HCC vs healthy serum`
    `Compare HCC vs healthy serum within liver sources`
    `Compare NAFLD vs healthy serum within liver sources`
    `Compare HCC vs NAFLD in serum`

    **Standard:**
    `What about HCC?`
    `Compare HCC vs healthy control`
    `Peak 1005?` `Lipid signal?`
    """)
    st.caption(f"{VERSION}: sample-type & domain scoping, fallback handling.")

query = st.text_input("Ask GAIRA:", placeholder="e.g., Compare HCC vs healthy serum within liver sources")
if not query: st.info("Enter a query above."); st.stop()

from graph.phaseC1_query_router import route_query
from graph.phaseC1_query_engine import execute_query
from graph.phaseC1_scoring import score_result
from graph.phaseC1_templates import format_explanation, render_text_explanation
from app.graph_preview import build_preview_graph, LEGEND_HTML

parsed = route_query(query)
if parsed.query_type == "unknown": st.warning(parsed.notes); st.stop()

entity = (parsed.entities[0] if parsed.entities else str(parsed.peak_cm)).replace("_"," ")
mode_labels = {"single":"ASSOCIATIVE","pairwise":"COMPARATIVE","one_vs_rest":"ENRICHMENT"}
header = f"**{parsed.query_type.upper()}** | **{entity}**"
if parsed.comparator: header += f" vs **{parsed.comparator.replace('_',' ')}**"
header += f" | {mode_labels.get(parsed.query_mode,'')}"
if parsed.sample_scope: header += f" | matrix: **{parsed.sample_scope}**"
if parsed.domain_scope: header += f" | domain: **{parsed.domain_scope}**"
st.info(header)

try: gr = execute_query(parsed, uri=neo4j_uri, user=neo4j_user, password=neo4j_pass)
except Exception as e: st.error(f"Neo4j: {e}"); st.stop()
if gr.total_evidence_rows == 0: st.warning("No evidence found."); st.stop()

scored = score_result(gr)
explanation = format_explanation(scored, gr)
is_comp = scored.query_mode in ("pairwise","one_vs_rest")

# ── KPIs ────────────────────────────────────────────────────
k1,k2,k3,k4 = st.columns(4)
k1.metric("Evidence", scored.evidence_count)
k2.metric("Sources", scored.source_count)
k3.metric("Motifs", len(scored.motif_summary))
k4.metric("Themes", len(scored.themes))

# ── SCOPE + COMPARATOR SUMMARY ─────────────────────────────
if scored.comparator_summary:
    cs = scored.comparator_summary
    st.subheader("Comparator Summary")
    cc1,cc2,cc3 = st.columns(3)
    cc1.markdown(f"**Query**: {cs.query_condition.replace('_',' ')}  \n{cs.query_evidence} rows, {cs.query_sources} sources")
    cc2.markdown(f"**Comparator**: {cs.comparator_condition.replace('_',' ')}  \n{cs.comparator_evidence} rows, {cs.comparator_sources} sources")
    cc3.metric("Adequacy", cs.comparator_adequacy.replace("_"," ").upper())

    scope_parts = [f"**Scope**: {cs.scope_mode.replace('_',' ')}"]
    if cs.sample_scope: scope_parts.append(f"matrix: **{cs.sample_scope}**")
    if cs.domain_scope: scope_parts.append(f"domain: **{cs.domain_scope}**")
    scope_parts.append(f"sample type: **{cs.inferred_sample_type}**")
    scope_parts.append(f"balance: **{cs.evidence_balance:.2f}**")
    st.caption(" | ".join(scope_parts))

    if cs.scope_fallback:
        st.warning(f"Scope fallback: {cs.scope_fallback}")

# ── THEME TABLES ────────────────────────────────────────────
def _tdf(themes, show_comp=False):
    rows = []
    for t in themes:
        r = {"Theme": t["theme"]+(" (broad)" if t.get("is_broad") else ""),
             "Score": t["final_score"], "Conf.": t["confidence"].upper()}
        if show_comp:
            r["Q"] = f"{t.get('query_direct',0)}d/{t.get('query_sources',0)}s"
            r["C"] = f"{t.get('comp_direct',0)}d/{t.get('comp_sources',0)}s"
            r["Raw"] = f"{t['enrichment_ratio']}x" if t.get("enrichment_ratio") else ""
            r["Norm"] = f"{t['norm_enrichment']}x" if t.get("norm_enrichment") else ""
            r["Bal."] = f"{t.get('evidence_balance',0):.2f}"
            if t.get("coverage_flag"): r["Flag"] = t["coverage_flag"].replace("_"," ")
        else:
            r["Direct"] = t.get("direct_evidence",0)
            r["Sources"] = t.get("source_diversity",0)
            r["Motifs"] = t.get("motif_links",0)
        rows.append(r)
    return pd.DataFrame(rows)

if is_comp:
    for key, title in [("D_enriched_themes",f"Enriched in {entity}"),
                       ("D2_associated_themes","Associated"),
                       ("D3_shared_themes","Shared"),("D4_depleted_themes","Depleted")]:
        items = explanation.get(key,[])
        if not items: continue
        if key in ("D3_shared_themes","D4_depleted_themes"):
            with st.expander(f"{title} ({len(items)})"): st.dataframe(_tdf(items,True), use_container_width=True, hide_index=True)
        else:
            st.subheader(title); st.dataframe(_tdf(items,True), use_container_width=True, hide_index=True)
else:
    top = explanation.get("D_top_themes",[]); sec = explanation.get("D2_secondary_themes",[])
    if top: st.subheader("Top Themes"); st.dataframe(_tdf(top), use_container_width=True, hide_index=True)
    if sec:
        with st.expander(f"Secondary ({len(sec)})"): st.dataframe(_tdf(sec), use_container_width=True, hide_index=True)

# ── MOTIFS + BIO ────────────────────────────────────────────
cl,cr = st.columns(2)
with cl:
    st.subheader("Top Motifs")
    if scored.motif_summary:
        mr = [{"Subfamily": m.subfamily.replace("_"," "), "Members": m.member_count,
               "Status": m.motif_interpretation.replace("-"," ")} for m in scored.motif_summary[:6]]
        if is_comp:
            for i,m in enumerate(scored.motif_summary[:6]):
                mr[i]["Comp."] = m.comparator_members
                if m.coverage_flag: mr[i]["Flag"] = m.coverage_flag.replace("_"," ")
        st.dataframe(pd.DataFrame(mr), use_container_width=True, hide_index=True)
with cr:
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
    for c in scored.caveats: st.warning(c)

st.subheader("Graph Preview")
st.markdown(LEGEND_HTML, unsafe_allow_html=True)
gh = build_preview_graph(gr)
if gh: components.html(gh, height=480, scrolling=False)

lbl = "Comparative subgraph" if is_comp else "Associative subgraph"
with st.expander(f"Neo4j Browser ({lbl})"): st.code(gr.viz_cypher, language="cypher")
with st.expander("Traversal"): st.markdown(explanation["C_graph_expansion"]["traversal_summary"])
with st.expander("Full Text"): st.markdown(render_text_explanation(explanation))
