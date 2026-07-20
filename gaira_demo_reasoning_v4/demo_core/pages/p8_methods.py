"""Page 8 — Methods & Provenance. Nothing hidden. (Live.)"""
from __future__ import annotations
import json
import streamlit as st
from .. import components as C


def render(bridge):
    s = bridge.platform_stats()
    C.page_header(
        "Provenance · nothing hidden",
        "Methods & provenance",
        "Every layer is versioned and inspectable. The atlas, ontology, component registry, theme "
        "weights, BSV equations, confidence engine and MSS layer are frozen; this demo measures "
        "them and never modifies them.")

    v = s["versions"]
    st.markdown("### Versions & fingerprints")
    st.table({
        "layer": ["Raman Reference Atlas", "Atlas fingerprint", "Biochemical Ontology",
                  "Component Registry", "MSS layer", "BSV"],
        "version": [v.get("atlas", "v0.1"), s["fingerprint"][:16] + "…",
                    v.get("biochemical_ontology", "v2"), v.get("component_registry", "v1"),
                    bridge.mss.version, "v2"],
    })

    st.markdown("### The frozen equations (BSV v2)")
    st.code(
        "coord_j       = a_j / sum_k a_k                  # L1 evidence share per component\n"
        "z_j           = (coord_j - center_j) / spread_j  # robust z vs reference frame\n"
        "composition_t = sum_j W[j,t] * coord_j           # theme evidence share\n"
        "elevation_t   = sum_j W[j,t] * z_j               # elevation vs pure references\n"
        "confidence_t  = stability_t * evidence_t * (1 - OOD)\n"
        "MSS_elev_m    = sum_j M[j,m] * z_j               # motif elevation (M derived, frozen)",
        language="text")

    st.markdown("### Studies behind the frozen system")
    st.markdown(
        "- Raman-only foundation (Phases C1–C7): representation benchmark → frozen NMF k=24 atlas.\n"
        "- Reference Atlas Component Audit: 24-component inventory, coherence, plausibility.\n"
        "- Serum Spike Projection Validation + Perturbation Response Audit: calibration behaviour.\n"
        "- BSV Validation: the BSV as a coordinate system (monotonicity, geometry, confidence).\n"
        "- MSS layer v1: derived motifs bridging components → themes (this demo's centerpiece).")

    with st.expander("Raw version manifest (JSON)"):
        st.code(json.dumps(v, indent=2), language="json")

    C.provenance_footer(s)
