"""GAIRA v4 — Unified Biochemical Reasoning Engine (UI Cleanup).

Incremental improvements over v3:
  - Text Query: per-condition trust graphs for comparison queries
  - Spectral Query: expected comparator trust graphs, demoted overlay radars
  - Both: cleaner evidence labels, clearer terminology

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src streamlit run streamlit_apps/gaira_v4/gaira_v4.py
"""
import streamlit as st

st.set_page_config(
    page_title="GAIRA v4",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧬 GAIRA v4")
st.caption("Unified Biochemical Reasoning Engine — UI cleanup over v3")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📝 Text Query")
    st.markdown(
        "Ask questions about Raman/SERS biochemistry. "
        "GAIRA retrieves evidence from its curated knowledge corpus "
        "and synthesizes a structured, uncertainty-aware answer."
    )
    st.markdown(
        "- Literature-grounded retrieval\n"
        "- Motif → theme → BSV mapping\n"
        "- **NEW**: per-condition trust graphs for comparison queries\n"
        "- **NEW**: human-readable evidence labels with hover explanations\n"
        "- Local synthesis or Gemini LLM\n"
    )
    st.page_link("pages/1_📝_Text_Query.py", label="Open Text Query →", icon="📝")

with col2:
    st.markdown("### 🔬 Spectral Query")
    st.markdown(
        "Select a measured spectral dataset and inspect its "
        "cohort-level biochemical composition. Compare observed "
        "spectral BSV against literature-expected profiles."
    )
    st.markdown(
        "- Direct spectral → BSV projection\n"
        "- Band-level drivers with motif annotations\n"
        "- **NEW**: expected-comparator trust graphs per cohort\n"
        "- **NEW**: demoted overlay radars (no longer the hero figure)\n"
        "- Delta-shift agreement as primary validation metric\n"
    )
    st.page_link("pages/2_🔬_Spectral_Query.py", label="Open Spectral Query →", icon="🔬")

st.markdown("---")

st.markdown("### What Changed from v3")
st.markdown(
    "- **Text Query**: comparison queries (e.g. 'HCC vs healthy vs CCA') now "
    "render a separate trust graph per condition, stacked vertically, with a "
    "comparison summary above.\n"
    "- **Text Query**: evidence nodes show human-readable labels with hover "
    "explanations of what each source represents.\n"
    "- **Spectral Query**: each cohort's expected comparator now has its own "
    "literature trust graph showing the evidence basis for that comparator.\n"
    "- **Spectral Query**: observed-vs-expected overlay radars are moved to the bottom "
    "and clearly labeled as 'visual comparison only'. The similarity matrix, "
    "alignment table, and delta-shift comparison are now the primary figures.\n"
    "- **Both**: consistent terminology — 'expected comparator', 'observed BSV', "
    "'literature-grounded', 'disease-vs-reference shift'."
)

st.markdown("---")

st.markdown(
    "**Not a classifier. Not a predictor.** "
    "GAIRA interprets Raman/SERS biochemistry through structured evidence — "
    "spectral projections, literature grounding, and transparent reasoning."
)

st.caption("GAIRA v4 · Text Query + Spectral Query · Unified demo")
