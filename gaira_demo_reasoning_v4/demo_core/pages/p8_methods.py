"""Page 8 — Methods & Provenance. The definitive transparency page: can every
displayed claim be traced and reproduced? All versions/fingerprints are live; the
atlas fingerprint is verified on app load.
"""
from __future__ import annotations
import json
import streamlit as st

from .. import components as C, biological as B, serum as S

# validation library — each committed study's key conclusion + limitation
VALIDATION = [
    ("Reference Atlas Component Audit", "results/v5_rebuild/reference_atlas_audit/",
     "The atlas learns molecular CLASS, not species; components are stable but low-purity.",
     "Sterol/heme chemistries under-represented; single labels are misleading (hence MSS)."),
    ("Serum Spike Projection Validation", "results/v5_rebuild/spike_validation/",
     "Concentration registers; chemical identity registers only for strong Ag adsorbers.",
     "Most serum analytes are not identity-recoverable (matrix competition)."),
    ("Perturbation Response Audit", "results/v5_rebuild/perturbation_response/",
     "The loop closes at component IDENTITY, not always the coarse theme label.",
     "Component→theme labels can lag the evidence (motivates the MSS layer)."),
    ("BSV Validation", "GAIRA_BSV_VALIDATION.md",
     "Dose responses are monotonic + saturating (all permutation p=0.002); geometry recovers "
     "chemistry (purines cluster).", "Effective dimensionality ≈4 of 11; confidence tracks "
     "domain OOD, NOT analyte recoverability."),
    ("Raman-only Foundation (C1–C7)", "results/v5_rebuild/foundation/",
     "NMF k=24 selected by benchmark (not PCA by default); explained variance 0.712; excitation "
     "transfer 0.918.", "Reconstruction error is moderate (mixture-first, not fingerprint)."),
    ("Preprocessing AutoResearch (B0)", "results/v5_rebuild/preprocessing_autoresearch/",
     "No preprocessing pipeline rescues cross-modal comparability without overprocessing.",
     "Apparent gains were artefacts of overprocessing — none frozen."),
]


def _versions(bridge, s):
    st.markdown("### A · Version manifest & fingerprints")
    v = s["versions"]
    st.table({
        "layer": ["Raman Reference Atlas", "Atlas fingerprint", "Preprocessing", "NMF basis",
                  "Component Registry", "MSS Registry", "Biochemical Ontology",
                  "Component→theme weights", "BSV", "Confidence engine", "Evidence engine",
                  "Domain context", "Demo build"],
        "version / value": [
            v.get("atlas", "v0.1"), s["fingerprint"][:20] + "…", "asls+savgol+L2 (frozen)",
            f"NMF k={s['n_components']} (held fixed)", v.get("component_registry", "v1"),
            bridge.mss.version, v.get("biochemical_ontology", "v2"),
            v.get("component_theme_weights", "v1"), "v2", "v2 (stability×evidence×(1−OOD))",
            v.get("evidence_engine", "v1"), "serum/ev/buffer/tissue/dart", "gaira_demo_reasoning_v4"],
    })
    ok = bridge.eng.atlas.meta["fingerprint"] == s["fingerprint"]
    st.markdown(f'<div class="{"gaira-take" if ok else "gaira-caveat"}">Atlas fingerprint verified '
                f'on load: <b>{"MATCH" if ok else "MISMATCH"}</b> '
                f'(<code>{s["fingerprint"][:16]}…</code>). The frozen coordinate system is '
                f'byte-identical to the pinned version.</div>', unsafe_allow_html=True)


def _data_provenance(bridge, s):
    st.markdown("### B · Data provenance")
    rows = {"dataset / stage": [], "source (sanitized)": [], "units": [], "status": [],
            "pipeline / notes": []}

    def add(name, src, units, status, note):
        rows["dataset / stage"].append(name); rows["source (sanitized)"].append(src)
        rows["units"].append(units); rows["status"].append(status); rows["pipeline / notes"].append(note)

    add("Frozen Raman atlas", "RamanBioLib + gobbato + amino-acid grounding",
        f"{s['n_reference_spectra']} spectra / {s['n_reference_analytes']} analytes", "REAL",
        "asls+savgol+L2 → NMF k=24 (frozen)")
    add("Serum spike validation", "results/v5_rebuild/spike_validation/tables (committed)",
        f"{S.recoverability_summary()['n_analytes']} analytes", "REAL",
        "frozen-atlas projections; phase7 recoverability")
    for key, meta in B.available().items():
        add(meta["display_name"], f"biological_artifacts/{key}.json (from raw volume; sanitized)",
            f"{meta['n_units']} {meta['aggregation']}s", meta["status"],
            "GAIRAEngine.infer; NO demographics; anonymised IDs")
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.markdown('<div class="gaira-caption">Private volume paths and patient identifiers are never '
                'displayed. Diabetes cohort labels come from the .mat file, not the demographic '
                'CSV; identifiers are anonymised indices.</div>', unsafe_allow_html=True)


def _equations():
    st.markdown("### C · Equations")
    st.markdown("**Implemented (frozen)** — every symbol defined below.")
    st.code(
        "a        = NNLS(query, frozen NMF dictionary)      # 24 non-negative activations\n"
        "coord_j  = a_j / Σ_k a_k                            # L1 evidence share per component\n"
        "z_j      = (coord_j − center_j) / spread_j          # robust z vs reference frame\n"
        "W[j,t]   = component→theme weight (rows sum to 1)   # ontology mapping\n"
        "M[j,m]   = component→MSS weight (derived, frozen)   # MSS layer\n"
        "comp_t   = Σ_j W[j,t]·coord_j                       # BSV theme composition (radar)\n"
        "elev_t   = Σ_j W[j,t]·z_j                           # theme elevation vs references\n"
        "MSS_m    = Σ_j M[j,m]·z_j                           # motif elevation\n"
        "OOD      = 1 − mean cosine to k nearest reference support vectors\n"
        "conf_t   = stability_t · evidence_t · (1 − OOD)     # per-theme confidence\n"
        "ΔBSV     = comp_t(group A) − comp_t(group B)        # comparator contrast",
        language="text")
    st.markdown('<div class="gaira-caption"><b>Symbols.</b> a = frozen NMF activations; coord = L1 '
                'share; z = robust z-score vs the median/MAD reference frame; W, M = frozen weight '
                'matrices; stability = bootstrap component stability; evidence = 1 − normalised '
                'entropy of the theme\'s evidence mass.</div>', unsafe_allow_html=True)
    st.markdown('<div class="gaira-caveat"><b>Conceptual (future, NOT implemented).</b> DART '
                'trajectory operators (potential×time), Au-SERS observation model, and '
                'modality-aware recoverability priors are described on the Serum and DART pages as '
                'future work — no such equations run today.</div>', unsafe_allow_html=True)


def _evidence_tiers():
    st.markdown("### D · Evidence tiers")
    st.markdown(
        "- **Tier 1 — direct spectral evidence**: frozen-atlas component loadings + characteristic "
        "bands (reference_loading + spectral_band lines in the ontology weights).\n"
        "- **Tier 2 — perturbation + literature**: dose-response / serum-spike / depletion "
        "responsiveness and curated literature anchors.\n"
        "- **Domain context**: matrix-specific caveats (serum/EV/buffer/tissue) that shape "
        "interpretation but never mutate the BSV.\n\n"
        "Every interpretation on every page is built from Tier-1 evidence, optionally reinforced by "
        "Tier-2, and framed by domain context — surfaced through the Evidence Engine.")


def _validation_library():
    st.markdown("### E · Validation library")
    for name, path, concl, limit in VALIDATION:
        st.markdown(f"**{name}**  ·  `{path}`  \n"
                    f'<span class="gaira-caption"><b>Conclusion.</b> {concl} '
                    f'<b>Limitation.</b> {limit}</span>', unsafe_allow_html=True)


def _reproducibility():
    st.markdown("### F · Reproducibility")
    st.code(
        "# environment (from repo root)\n"
        "python -m venv .venv && source .venv/bin/activate\n"
        "pip install -r requirements.txt        # numpy scipy scikit-learn pandas plotly streamlit\n\n"
        "# launch the demo\n"
        "cd gaira_demo_reasoning_v4 && streamlit run app.py      # or ./run_demo.sh\n\n"
        "# headless self-check (renders every figure, verifies atlas fingerprint)\n"
        "python selfcheck.py\n\n"
        "# regenerate biological V6 artifacts (needs the data volume mounted; else DEGRADED)\n"
        "python tools/build_biological_v6.py\n\n"
        "# regenerate the MSS registry (pure function of frozen artifacts)\n"
        "python ../results/v5_rebuild/engine_v1/tools/build_mss_registry.py\n\n"
        "# tests (from repo root)\n"
        "python -m pytest tests/test_v6_engine.py tests/test_v6_mss_layer.py \\\n"
        "                 tests/test_v6_demo_v4.py tests/test_v6_bsv_validation.py -q",
        language="bash")
    st.markdown('<div class="gaira-caption">Data-availability check: the demo runs on a fresh '
                'checkout using committed artifacts (frozen atlas, spike-validation tables, '
                'biological_artifacts). The private data volume is only needed to REGENERATE '
                'biological artifacts, never to view them.</div>', unsafe_allow_html=True)


def render(bridge):
    s = bridge.platform_stats()
    C.page_header(
        "Provenance · nothing hidden",
        "Methods & provenance",
        "Every layer is versioned and inspectable. The atlas, preprocessing, NMF, ontology, "
        "component registry, theme weights, BSV equations, confidence engine and MSS layer are "
        "frozen; this demo measures them and never modifies them.")
    C.question("Can every displayed claim be traced to a version, a fingerprint, a committed "
               "artifact, or a validated study — and reproduced?")

    _versions(bridge, s)
    st.markdown("<hr/>", unsafe_allow_html=True)
    _data_provenance(bridge, s)
    st.markdown("<hr/>", unsafe_allow_html=True)
    _equations()
    st.markdown("<hr/>", unsafe_allow_html=True)
    _evidence_tiers()
    st.markdown("<hr/>", unsafe_allow_html=True)
    _validation_library()
    st.markdown("<hr/>", unsafe_allow_html=True)
    _reproducibility()

    with st.expander("Raw version manifest (JSON)"):
        st.code(json.dumps(s["versions"], indent=2), language="json")
    C.related(["2 · Reference Atlas", "4 · Calibration", "6 · Biological Studies"])
    C.provenance_footer(s)
