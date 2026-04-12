# GAIRA LANDSCAPE v3 — BSV fix + ontology cleanup + trust filters

VERSION = "Landscape v3"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(page_title=f"GAIRA {VERSION}", page_icon="🗺", layout="wide")
st.title(f"GAIRA — {VERSION}")
st.caption(f"BSV fixed | Biological/non-biological split | Trust filters | Coverage audit")

rdir = '/Volumes/SSD_Rad/GAIRA_DATA/structured_evidence_v2/reports/'
ldir = Path(__file__).resolve().parent.parent / 'landscape'

# Load data
@st.cache_data
def load_data():
    trust = pd.read_csv(str(ldir / 'condition_trust_summary.csv'))
    classify = pd.read_csv(str(ldir / 'condition_classification.csv'))
    try: bsv = pd.read_csv(rdir + 'gaira_landscape_v3_bsv_matrix.csv')
    except: bsv = pd.DataFrame()
    try: top_bsv = pd.read_csv(rdir + 'gaira_landscape_v3_top_bsv_per_condition.csv')
    except: top_bsv = pd.DataFrame()
    try: top_motifs = pd.read_csv(rdir + 'gaira_landscape_v2_top_motifs_per_condition.csv')
    except: top_motifs = pd.DataFrame()
    try: pairs = pd.read_csv(rdir + 'gaira_landscape_v2_condition_pair_overlap.csv')
    except: pairs = pd.DataFrame()
    try: bias = pd.read_csv(rdir + 'gaira_landscape_v2_source_bias_audit.csv')
    except: bias = pd.DataFrame()
    try: debug = pd.read_csv(str(ldir / 'bsv_mapping_debug.csv'))
    except: debug = pd.DataFrame()
    return trust, classify, bsv, top_bsv, top_motifs, pairs, bias, debug

trust_df, class_df, bsv_df, top_bsv_df, top_motifs_df, pairs_df, bias_df, debug_df = load_data()

# ── FILTERS (sidebar) ──────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    scope = st.selectbox("Condition Scope", ["biological-only", "all", "controls only", "non-biological only"], index=0)
    min_evidence = st.slider("Min evidence rows", 0, 50, 5)
    min_sources = st.slider("Min sources", 0, 10, 2)
    exclude_single = st.checkbox("Exclude single-source", value=True)
    bsv_ready = st.checkbox("BSV-mapped only", value=False)

# Apply filters
filtered = trust_df.copy()
if scope == "biological-only":
    filtered = filtered[filtered['ontology_class'].isin(['biological_condition','healthy_or_control'])]
elif scope == "controls only":
    filtered = filtered[filtered['ontology_class'] == 'healthy_or_control']
elif scope == "non-biological only":
    filtered = filtered[~filtered['ontology_class'].isin(['biological_condition','healthy_or_control'])]

filtered = filtered[filtered['evidence_count'] >= min_evidence]
filtered = filtered[filtered['source_count'] >= min_sources]
if exclude_single:
    filtered = filtered[filtered['single_source_flag'] != 'yes']
if bsv_ready:
    filtered = filtered[filtered['nonzero_bsv_components'] > 0]

active_conds = set(filtered['condition'])

st.sidebar.metric("Active conditions", len(active_conds))
st.sidebar.caption(f"Total conditions: {len(trust_df)}")

# ── COVERAGE ────────────────────────────────────────────────
st.header("1. Coverage + Trust")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Active", len(active_conds))
c2.metric("Strong trust", len(filtered[filtered['trust_label']=='strong']))
c3.metric("Moderate", len(filtered[filtered['trust_label']=='moderate']))
c4.metric("Weak/Low", len(filtered[filtered['trust_label'].isin(['weak','low','insufficient'])]))

st.dataframe(filtered[['condition','evidence_count','source_count','ontology_class','support_tier',
                         'trust_label','mapped_bsv_fraction','nonzero_bsv_components','caution_note']]
             .sort_values('evidence_count', ascending=False).head(20),
             use_container_width=True, hide_index=True)

# ── CIRRHOSIS NOTE ──────────────────────────────────────────
if 'cirrhosis' not in active_conds and 'fibrosis' not in active_conds:
    st.warning("Cirrhosis/fibrosis absent from active conditions. Major liver gap.")

# ── SIMILARITY ──────────────────────────────────────────────
st.header("2. Condition Similarity")

try:
    if scope == "biological-only":
        img = Image.open(rdir + 'gaira_landscape_v3_condition_similarity_bio.png')
        dend = Image.open(rdir + 'gaira_landscape_v3_condition_dendrogram_bio.png')
    else:
        img = Image.open(rdir + 'gaira_landscape_v2_condition_similarity_heatmap.png')
        dend = Image.open(rdir + 'gaira_landscape_v2_condition_dendrogram.png')
    ch, cd = st.columns(2)
    with ch: st.image(img, caption="Similarity Heatmap", use_container_width=True)
    with cd: st.image(dend, caption="Clustering", use_container_width=True)
except: st.info("Similarity images not available for this filter set.")

with st.expander("Nearest Neighbors"):
    try: st.dataframe(pd.read_csv(str(ldir / 'condition_neighbors.csv')).head(15), use_container_width=True, hide_index=True)
    except: st.info("Not available")

# ── BSV LANDSCAPE ───────────────────────────────────────────
st.header("3. BSV Landscape")

try:
    bsv_h = Image.open(rdir + 'gaira_landscape_v3_bsv_heatmap.png')
    bsv_d = Image.open(rdir + 'gaira_landscape_v3_bsv_dendrogram.png')
    b1, b2 = st.columns(2)
    with b1: st.image(bsv_h, caption="BSV Heatmap (biological, v3 fixed)", use_container_width=True)
    with b2: st.image(bsv_d, caption="BSV Clustering (Ward)", use_container_width=True)
except: st.warning("BSV images not found.")

st.subheader("Top BSV Components per Condition")
if not top_bsv_df.empty:
    filtered_bsv = top_bsv_df[top_bsv_df['condition'].isin(active_conds)]
    st.dataframe(filtered_bsv.head(25), use_container_width=True, hide_index=True)

with st.expander("BSV Matrix (raw)"):
    if not bsv_df.empty:
        filtered_bsv_m = bsv_df[bsv_df['condition'].isin(active_conds)]
        st.dataframe(filtered_bsv_m, use_container_width=True, hide_index=True)

# ── MOTIFS ──────────────────────────────────────────────────
st.header("4. Motif Structure")

col_m, col_p = st.columns(2)
with col_m:
    st.subheader("Top Motifs per Condition")
    if not top_motifs_df.empty:
        fm = top_motifs_df[top_motifs_df['condition'].isin(active_conds)]
        st.dataframe(fm.head(20), use_container_width=True, hide_index=True)
with col_p:
    st.subheader("Condition Pair Overlap")
    if not pairs_df.empty:
        st.dataframe(pairs_df, use_container_width=True, hide_index=True)

try:
    mn = Image.open(rdir + 'gaira_landscape_v2_motif_network.png')
    with st.expander("Motif Network"): st.image(mn, use_container_width=True)
except: pass

# ── TRUST + BIAS ────────────────────────────────────────────
st.header("5. Trustworthiness + Bias")

st.subheader("Trust Distribution")
trust_counts = filtered['trust_label'].value_counts()
st.bar_chart(trust_counts)

with st.expander("Single-Source Risk"):
    ss = trust_df[(trust_df['single_source_flag']=='yes') & (trust_df['evidence_count']>0)]
    st.dataframe(ss[['condition','evidence_count','ontology_class','trust_label']].sort_values('evidence_count', ascending=False),
                 use_container_width=True, hide_index=True)

with st.expander("BSV Mapping Failures"):
    zero_bsv = trust_df[(trust_df['nonzero_bsv_components']==0) & (trust_df['evidence_count']>0)]
    st.dataframe(zero_bsv[['condition','evidence_count','mapped_bsv_fraction','ontology_class']],
                 use_container_width=True, hide_index=True)

with st.expander("Excluded Non-Biological Labels"):
    non_bio = class_df[~class_df['ontology_class'].isin(['biological_condition','healthy_or_control'])]
    st.dataframe(non_bio, use_container_width=True, hide_index=True)

# ── DEBUG ───────────────────────────────────────────────────
st.header("6. Debug / Audit")

with st.expander("BSV Mapping Debug (all conditions)"):
    if not debug_df.empty:
        st.dataframe(debug_df.sort_values('mapped_fraction', ascending=False), use_container_width=True, hide_index=True)

with st.expander("HCC / NAFLD / Healthy Trace"):
    try:
        trace = pd.read_csv(str(ldir / 'bsv_trace_HCC_NAFLD_healthy.csv'))
        st.dataframe(trace, use_container_width=True, hide_index=True)
    except: st.info("Trace not available")

with st.expander("Unmapped Motifs"):
    try:
        um = pd.read_csv(str(ldir / 'bsv_unmapped_motifs_summary.csv'))
        st.dataframe(um, use_container_width=True, hide_index=True)
    except: st.info("Not available")
