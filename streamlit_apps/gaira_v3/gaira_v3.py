"""GAIRA v3 — Unified Biochemical Reasoning Engine.

Integrates:
  📝 Text Query — literature-grounded biochemical Q&A
  🔬 Spectral Query — dataset-grounded BSV composition analysis

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src streamlit run streamlit_apps/gaira_v3/gaira_v3.py
"""
import streamlit as st

st.set_page_config(
    page_title="GAIRA v3",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧬 GAIRA v3")
st.caption("Unified Biochemical Reasoning Engine for Raman/SERS Spectroscopy")

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
        "- Radar plots for detected conditions\n"
        "- Local synthesis or Gemini LLM\n"
        "- Trust graph with evidence provenance"
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
        "- Disease-vs-reference shift analysis\n"
        "- Expected literature comparator validation\n"
        "- Optional Gemini interpretation"
    )
    st.page_link("pages/2_🔬_Spectral_Query.py", label="Open Spectral Query →", icon="🔬")

st.markdown("---")

st.markdown(
    "**Not a classifier. Not a predictor.** "
    "GAIRA interprets Raman/SERS biochemistry through structured evidence — "
    "spectral projections, literature grounding, and transparent reasoning."
)

st.caption("GAIRA v3 · Text Query + Spectral Query · Integrated Demo")
