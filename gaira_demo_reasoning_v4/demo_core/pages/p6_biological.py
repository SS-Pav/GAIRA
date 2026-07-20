"""Page 6 — Biological Studies. Real cohorts through the V6 engine, one standardized
template. Every displayed value is a genuine GAIRAEngine output (committed artifacts
built by tools/build_biological_v6.py); nothing is a relabelled legacy radar.
"""
from __future__ import annotations
import numpy as np
import streamlit as st

from .. import components as C, figures as F, biological as B

# datasets known to the demo but NOT yet projected through V6 — shown honestly, no output
DEFERRED = {
    "small2023_ev": ("EV single-vesicle SERS (small2023)", "probe titration, not a disease "
                     "contrast; embedded wavenumber axis not yet recovered"),
    "shine_ev_sers": ("SHINE EV-SERS (hepatotoxicity)", "dose×day EV series; ingestion not yet "
                      "wired for V6"),
    "cca_hcc_lm_serum_sers": ("Liver serum SERS (CCA/HCC/LM/NC)", "4-class per-sample txt; "
                              "ingestion not yet wired for V6"),
}
DOMAIN_NOTE = {
    "serum": "Serum: abundant protein background; metabolite/adsorption selection on Ag; purine "
             "enrichment where it adsorbs; weak adsorbers can be invisible.",
    "ev": "EV: multicomponent (membrane/lipid, protein, nucleic-acid, glycan, redox). Isolation, "
          "probe chemistry and weak labels all affect interpretation.",
}


def _status_badge(status):
    color = {"REAL": F.T.GOOD, "DEGRADED": F.T.WARN, "UNAVAILABLE": F.T.BAD}.get(status, F.T.FAINT)
    return f'<span style="color:{color};font-weight:700">● {status}</span>'


# ── Section A ──
def _registry():
    st.markdown("### A · Biological study registry")
    avail = B.available()
    rows = {"dataset": [], "status": [], "domain": [], "aggregation": [], "units": [], "groups": []}
    for key, meta in avail.items():
        rows["dataset"].append(meta["display_name"]); rows["status"].append(meta["status"])
        rows["domain"].append(meta["domain"]); rows["aggregation"].append(meta["aggregation"])
        rows["units"].append(meta["n_units"])
        rows["groups"].append(", ".join(f"{g}={n}" for g, n in meta["n_by_group"].items()))
    for key, (name, why) in DEFERRED.items():
        rows["dataset"].append(name); rows["status"].append("UNAVAILABLE")
        rows["domain"].append("—"); rows["aggregation"].append("—"); rows["units"].append(0)
        rows["groups"].append(why)
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.markdown('<div class="gaira-caption"><b>REAL</b> = raw data projected live through the V6 '
                'engine (committed sanitized artifacts). <b>UNAVAILABLE</b> = present on disk but '
                'not yet wired for V6 — no output is fabricated. The prior demo\'s per-cohort '
                'radars used a <b>pre-V6</b> 8-axis engine and are NOT reused here.</div>',
                unsafe_allow_html=True)


# ── Section B: standardized per-study template ──
def _study(art, bridge):
    gc = B.group_contrast(art)
    a, b = gc["a"], gc["b"]
    patient_level = art["aggregation"] == "patient"

    # 1 · framing
    st.markdown(f"#### {art['display_name']}")
    st.markdown(
        f"- **Domain / modality**: {art['domain']} · {art['modality']} · "
        f"{art['substrate']} · {art['excitation_nm']} nm\n"
        f"- **Contrast**: {a} (n={gc['na']}) vs {b} (n={gc['nb']})  ·  "
        f"aggregation: **{art['aggregation']}-level**\n"
        f"- **Source**: `{art['source']}`")
    for cav in art["caveats"]:
        st.markdown(f'<div class="gaira-caveat">{cav}</div>', unsafe_allow_html=True)
    st.write("")

    # 2 · data quality
    C.figure(F.group_quality(art),
             cap="OOD, confidence and matrix-share distributions per group — the data-quality "
                 "context for every downstream claim.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        # 3 · absolute atlas-position radar
        series = [{"name": g, "axes": B.group_radar_axes(art, g)} for g in art["groups"]]
        C.figure(F.multi_radar(series, title=f"{art['display_name']} — absolute BSV"),
                 cap="Each group's absolute position in the frozen atlas frame (NOT cohort-mean "
                     "normalised).")
    with c2:
        # 4 · comparator ΔBSV (forest with effect sizes)
        C.figure(F.forest_plot(gc["rows"], a, b, title=f"ΔBSV ({a} − {b})"),
                 cap="Signed theme difference with 95% bootstrap CI; green = FDR q<0.05; δ = "
                     "Cliff's delta (effect size, emphasised over p).")

    # 5 · MSS differences
    _, _, motif_ids, mdelta = B.motif_contrast(art)
    mnames = {m.id: m.name for m in bridge.mss.motifs}
    bio_motif_idx = [i for i, mid in enumerate(motif_ids)
                     if mid not in ("colloid_matrix_background",)]
    means = B.group_means_by(art, "motifs_mat")
    c3, c4 = st.columns(2, gap="large")
    with c3:
        C.figure(F.difference_bars([mnames.get(motif_ids[i], motif_ids[i]) for i in bio_motif_idx],
                                   [means[b][i] for i in bio_motif_idx],
                                   [means[a][i] for i in bio_motif_idx],
                                   title=f"ΔMSS motifs ({a} − {b})"),
                 cap="Which spectral motifs drive the broad BSV change.")
    with c4:
        # 6 · component provenance + 7 · sample space
        proj, var = B.pca_2d(art["themes_mat"])
        C.figure(F.bio_pca(proj, art["group"], var, title="BSV sample space (PCA)"),
                 cap="Per-sample BSV, PCA to 2-D (X = group centroid). PCA axes are NOT "
                     "biochemical themes.")

    # 6 · component provenance (numeric)
    top = B.top_components_for(art, mdelta)
    st.markdown("**Component provenance** (largest per-group coordinate differences): "
                + ", ".join(f"c{j} ({d:+.3f})" for j, d in top))

    # 8 · statistical summary + 9 · evidence interpretation
    top_row = gc["rows"][0]
    lvl = ("patient-level (1 patient = 1 n)" if patient_level
           else "spectrum-level, EXPLORATORY (subject mapping undocumented; not patient-level "
                "inference)")
    st.markdown(f'<div class="gaira-card"><b>Statistical summary.</b> Mann-Whitney U per theme, '
                f'Benjamini-Hochberg FDR, Cliff\'s delta effect size, 2000× bootstrap CIs. '
                f'Inference level: <b>{lvl}</b>.<br><b>Leading difference.</b> '
                f'<i>{F.THEME_SHORT.get(top_row["theme"], top_row["theme"])}</i> '
                f'Δ={top_row["delta"]:+.3f} (δ={top_row["cliffs_delta"]:+.2f}, q={top_row["q"]:.3f}). '
                f'This is <b>consistent with</b> a shift in the {a} group relative to {b}; it does '
                f'not prove any molecule, pathway or diagnosis.</div>', unsafe_allow_html=True)

    # 10 · interpretation summary (cautious language)
    n_sig = sum(r["sig"] for r in gc["rows"])
    if patient_level and abs(top_row["cliffs_delta"]) > 0.5:
        verdict = (f"A robust patient-level biochemical difference: {n_sig} themes reach FDR "
                   f"significance with large effect sizes. Consistent with a real biochemical "
                   f"contrast between {a} and {b} in this cohort.")
    elif max(abs(r["cliffs_delta"]) for r in gc["rows"]) < 0.33:
        verdict = (f"Minimal biochemical separation: even where p is small (large n), effect sizes "
                   f"are tiny (|δ|<0.33). The groups are barely distinguishable in BSV space — an "
                   f"honest near-null result. Low OOD here reflects that {art['modality']} serum is "
                   f"relatively in-domain, not that a strong signal was found.")
    else:
        verdict = (f"A moderate, exploratory difference ({n_sig} themes FDR-significant). Read as "
                   f"associated-with, pending patient-level replication.")
    C.takeaways([verdict])


# ── Section E ──
def _generalization():
    st.markdown("### E · Cross-study generalization")
    sc = B.study_centroids()
    if sc is None:
        st.info("No V6 biological artifacts available."); return
    C.figure(F.study_centroid_map(sc),
             cap="Group centroids of all V6 cohorts in one shared BSV PCA space; colour = mean OOD.",
             interp="COVID (serum Raman) is most in-domain and its groups cluster tightly; HCC "
                     "(Ag-SERS) is most out-of-domain; diabetes groups separate most.",
             limits="PC1 largely reflects DOMAIN / matrix / modality, not biology. This is an "
                     "overview, NOT evidence of cross-domain biological equivalence.")


def render(bridge):
    s = bridge.platform_stats()
    C.page_header(
        "Application · biological cohorts",
        "Biological studies through the V6 engine",
        "Real biological cohorts, each projected live through the frozen V6 engine (component "
        "coordinates → BSV → MSS → domain context) and shown with one standardized template. "
        "Every value is a genuine engine output; nothing is a relabelled legacy radar; cautious, "
        "association-level language throughout.")
    C.question("What biochemical-state differences does the V6 architecture identify in real "
               "biological datasets — and how much should each be trusted?")

    _registry()
    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("### B · Standardized study template")
    avail = {k: v for k, v in B.available().items() if v["status"] == "REAL"}
    if not avail:
        st.warning("No REAL V6 biological artifacts found. Run "
                   "`python tools/build_biological_v6.py` with the data volume mounted.")
    else:
        key = st.selectbox("Study", list(avail.keys()),
                           format_func=lambda k: avail[k]["display_name"])
        art = B.load(key)
        _study(art, bridge)
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("### C · Domain-specific interpretation")
        st.markdown(f'<div class="gaira-card">{DOMAIN_NOTE.get(art["domain"], "")}</div>',
                    unsafe_allow_html=True)
        st.markdown("### D · V6 vs prior demo")
        st.markdown('<div class="gaira-caveat">The prior demo scored these cohorts with a '
                    '<b>pre-V6</b> 8-axis autoresearch engine. These V6 results are computed '
                    'afresh from raw spectra through the frozen atlas + ontology v2 + MSS; where '
                    'they differ, the V6 numbers stand (no attempt is made to reproduce the old '
                    'radar). Qualitative agreement and divergence are catalogued in '
                    'DATA_PROVENANCE_AUDIT.md.</div>', unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    _generalization()

    C.caveats([
        "Biological SERS/Raman in complex matrices is out-of-domain for the Raman atlas; read "
        "group differences with OOD and confidence, never as absolute quantitation.",
        "Cohort differences reflect biochemical <i>systems</i>, not single-molecule concentrations "
        "or diagnoses.",
        "Impact / Strong-D / H0T are the sources' own labels, shown verbatim.",
    ])
    C.related(["3 · How GAIRA Reasons", "5 · Serum Spike Stress Test", "8 · Methods & Provenance"])
    C.provenance_footer(s)
