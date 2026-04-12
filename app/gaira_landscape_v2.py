# GAIRA LANDSCAPE v2 — Visual Analytics + Clustering Inspection

VERSION = "Landscape v2"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(page_title=f"GAIRA {VERSION}", page_icon="🗺", layout="wide")
st.title(f"GAIRA — {VERSION}")
st.caption("Coverage, similarity, BSV landscape, motif structure, source bias | 1,887 evidence rows | 137 sources")

rdir = '/Volumes/SSD_Rad/GAIRA_DATA/structured_evidence_v2/reports/'
ldir = Path(__file__).resolve().parent.parent / 'landscape'

# ── COVERAGE ────────────────────────────────────────────────
st.header("1. Coverage Overview")

try:
    cov = pd.read_csv(rdir + 'gaira_landscape_v2_condition_coverage.csv')
    cov_with_ev = cov[cov['evidence_count'].astype(int) > 0].sort_values('evidence_count', ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Conditions", len(cov))
    c2.metric("With Evidence", len(cov_with_ev))
    c3.metric("Strong (>=40 rows, 3+ src)", len(cov[cov['coverage_tier']=='strong']))
    c4.metric("Single-Source Risk", len(cov[(cov['source_count'].astype(int)==1) & (cov['evidence_count'].astype(int)>0)]))

    st.dataframe(cov_with_ev[['condition','evidence_count','source_count','motif_count','peak_count','coverage_tier','sample_types']].head(15),
                 use_container_width=True, hide_index=True)

    with st.expander("Sample Type Coverage"):
        st.dataframe(pd.read_csv(rdir + 'gaira_landscape_v2_sampletype_coverage.csv'), use_container_width=True, hide_index=True)
except Exception as e:
    st.warning(f"Coverage data not found: {e}")

# ── CIRRHOSIS NOTE ──────────────────────────────────────────
st.info("**Cirrhosis gap**: Cirrhosis is absent as a condition. Only 'fibrosis' exists with 2 evidence rows. This is a significant gap for a liver-focused system.")

# ── CONDITION SIMILARITY ────────────────────────────────────
st.header("2. Condition Similarity")

try:
    img_heat = Image.open(rdir + 'gaira_landscape_v2_condition_similarity_heatmap.png')
    img_dend = Image.open(rdir + 'gaira_landscape_v2_condition_dendrogram.png')
    col_h, col_d = st.columns(2)
    with col_h: st.image(img_heat, caption="Cosine Similarity Heatmap", use_container_width=True)
    with col_d: st.image(img_dend, caption="Hierarchical Clustering", use_container_width=True)
except: st.warning("Similarity images not found.")

with st.expander("Nearest Neighbors"):
    try: st.dataframe(pd.read_csv(str(ldir / 'condition_neighbors.csv')).head(15), use_container_width=True, hide_index=True)
    except: st.info("Not available")

# ── BSV LANDSCAPE ───────────────────────────────────────────
st.header("3. BSV Landscape")

try:
    img_bsv = Image.open(rdir + 'gaira_landscape_v2_bsv_heatmap.png')
    img_bsv_d = Image.open(rdir + 'gaira_landscape_v2_bsv_dendrogram.png')
    col_b1, col_b2 = st.columns(2)
    with col_b1: st.image(img_bsv, caption="BSV Heatmap", use_container_width=True)
    with col_b2: st.image(img_bsv_d, caption="BSV Clustering (Ward)", use_container_width=True)
except: st.warning("BSV images not found.")

with st.expander("BSV Matrix"):
    try: st.dataframe(pd.read_csv(rdir + 'gaira_landscape_v2_bsv_matrix.csv').head(15), use_container_width=True, hide_index=True)
    except: st.info("Not available")

# ── MOTIF NETWORK ───────────────────────────────────────────
st.header("4. Motif Co-occurrence")

try:
    img_net = Image.open(rdir + 'gaira_landscape_v2_motif_network.png')
    st.image(img_net, caption="Motif Co-occurrence Network (Jaccard >= 0.15)", use_container_width=True)
except: st.warning("Network image not found.")

with st.expander("Motif Hubs"):
    try: st.dataframe(pd.read_csv(rdir + 'gaira_landscape_v2_motif_hubs.csv'), use_container_width=True, hide_index=True)
    except: st.info("Not available")

# ── EXPLORER TABLES ─────────────────────────────────────────
st.header("5. Condition Explorer")

col_m, col_b = st.columns(2)
with col_m:
    st.subheader("Top Motifs per Condition")
    try: st.dataframe(pd.read_csv(rdir + 'gaira_landscape_v2_top_motifs_per_condition.csv').head(20), use_container_width=True, hide_index=True)
    except: st.info("Not available")
with col_b:
    st.subheader("Top BSV per Condition")
    try: st.dataframe(pd.read_csv(rdir + 'gaira_landscape_v2_top_bsv_per_condition.csv').head(20), use_container_width=True, hide_index=True)
    except: st.info("Not available")

with st.expander("Condition Pair Overlaps"):
    try: st.dataframe(pd.read_csv(rdir + 'gaira_landscape_v2_condition_pair_overlap.csv'), use_container_width=True, hide_index=True)
    except: st.info("Not available")

# ── SOURCE BIAS ─────────────────────────────────────────────
st.header("6. Source Bias Assessment")

try:
    bias = pd.read_csv(rdir + 'gaira_landscape_v2_source_bias_audit.csv')
    high_bias = bias[bias['bias_risk']=='high']
    st.metric("Conditions with High Bias Risk (single-source)", len(high_bias))
    st.dataframe(bias.head(15), use_container_width=True, hide_index=True)
except: st.warning("Bias data not found.")

# ── NEIGHBORHOODS ───────────────────────────────────────────
st.header("7. Emergent Biochemical Neighborhoods")
try:
    nbh = (ldir / 'biochemical_neighborhoods.md').read_text()
    st.markdown(nbh)
except: st.info("See landscape/biochemical_neighborhoods.md")

# ── SUMMARY ─────────────────────────────────────────────────
st.header("8. Key Findings")
st.markdown("""
- **HCC and NAFLD cluster together** (cosine 0.43) — shared liver disease biochemistry
- **Bacterial identification is distinct** — unique Raman fingerprint separates from serum diseases
- **14 conditions are single-source** — high bias risk, cross-validation needed
- **Dominant signal is serum baseline** (Phe + protein + lipid) — condition-specific signals are sparser
- **BSV adds interpretable structure** but most conditions are too sparse for confident BSV-level claims
- **Cirrhosis is absent** — a major gap for liver-focused analysis
""")
