# GAIRA LANDSCAPE v5.1 — Inference Hardening
VERSION = "Landscape v5.1"

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st; import pandas as pd; from PIL import Image

st.set_page_config(page_title=f"GAIRA {VERSION}", page_icon="🗺", layout="wide")
st.title(f"GAIRA — {VERSION}")
st.caption("Bounded confidence | Support-weighted deltas | Inference readiness gates | Correction audit")

out = '/Users/suraj/projects/GAIRA/outputs/landscape_v5_1/'

st.header("1. Inference Readiness")
try:
    rd = pd.read_csv(out + 'condition_inference_readiness.csv')
    ig = rd[rd['readiness_label']=='inference_grade']; pv = rd[rd['readiness_label']=='provisional']
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Inference Grade", len(ig)); c2.metric("Provisional", len(pv))
    c3.metric("Exploratory", len(rd[rd['readiness_label']=='exploratory']))
    c4.metric("Insufficient", len(rd[rd['readiness_label']=='insufficient']))
    st.dataframe(rd[rd['readiness_label'].isin(['inference_grade','provisional','exploratory'])]
                 .sort_values('aggregate_confidence',ascending=False).head(15),
                 use_container_width=True, hide_index=True)
except Exception as e: st.warning(str(e))

st.header("2. Confidence Summary")
try:
    cf = pd.read_csv(out + 'confidence_condition_summary.csv')
    cf_active = cf[cf['aggregate_confidence']>0].sort_values('aggregate_confidence',ascending=False)
    st.dataframe(cf_active.head(15), use_container_width=True, hide_index=True)
except: st.info("Not available")

with st.expander("Confidence Factor Breakdown"):
    try: st.dataframe(pd.read_csv(out+'confidence_factor_breakdown.csv').head(15), use_container_width=True, hide_index=True)
    except: pass

st.header("3. Delta Variants")
tab1,tab2,tab3,tab4 = st.tabs(["Raw","Support-Weighted","Corrected","Corrected+SW"])
for tab, name in [(tab1,'raw'),(tab2,'support_weighted'),(tab3,'corrected'),(tab4,'corrected_support_weighted')]:
    with tab:
        try: st.dataframe(pd.read_csv(out+f'bsv_delta_{name}.csv').head(10), use_container_width=True, hide_index=True)
        except: st.info(f"{name} not available")

st.header("4. Correction Audit")
try:
    rc = pd.read_csv(out+'raw_vs_corrected_delta.csv')
    risks = rc[rc['flag'].str.contains('overcorrection|large',na=False)]
    st.metric("Overcorrection Risks", len(risks))
    if not risks.empty: st.dataframe(risks, use_container_width=True, hide_index=True)
    else: st.success("No overcorrection risks detected.")
except: st.info("Not available")

st.header("5. PCA (Inference-Grade)")
try:
    img = Image.open(out+'pca_delta_biplot.png')
    st.image(img, caption="Support-Weighted Delta Biplot (inference-grade conditions)", use_container_width=True)
except: st.info("Biplot not available")

with st.expander("PCA Scores"):
    try: st.dataframe(pd.read_csv(out+'pca_delta_scores.csv'), use_container_width=True, hide_index=True)
    except: pass
with st.expander("PCA Loadings"):
    try: st.dataframe(pd.read_csv(out+'pca_delta_loadings.csv'), use_container_width=True, hide_index=True)
    except: pass

st.header("6. Filter Membership")
try: st.dataframe(pd.read_csv(out+'filter_membership_audit.csv').head(20), use_container_width=True, hide_index=True)
except: st.info("Not available")
