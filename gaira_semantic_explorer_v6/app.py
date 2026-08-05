"""GAIRA Semantic Explorer V6 — the rebuilt interpretation hierarchy.

Additive. The Foundation Explorers V1–V6 (including the V6 detection-gate explorer at
gaira_foundation_explorer_v6/) are untouched. This app covers the V6 SEMANTIC rebuild:
components → MSS motifs → chemical themes.

Reads only committed artifacts under results/v6_rebuild/ plus assets/foundation/.

    streamlit run gaira_semantic_explorer_v6/app.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v6_ui import ui, data as D              # noqa: E402
from v6_ui.pages import PAGES                # noqa: E402

st.set_page_config(page_title="GAIRA V6 — Semantic Hierarchy", page_icon="🧬",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(ui.CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### GAIRA **V6**")
    st.caption("Hierarchical semantic rebuild — in-domain Raman only")
    choice = st.radio("", [n for n, _ in PAGES], label_visibility="collapsed")
    st.markdown("---")
    h = D.headline()
    st.caption(f"**atlas** `{h['fingerprint'][:16]}…`  \n"
               f"24 components → {h['n_motifs']} motifs → {h['n_themes']} themes  \n"
               f"theme top-1 **{h['theme_top1']:.3f}** (null {h['null']:.3f})")
    st.caption("The frozen foundation is unchanged. V6 rebuilds only the layers above it.")

dict(PAGES)[choice]()
