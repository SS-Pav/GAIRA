"""Page 6 — Biological Studies. Real cohorts through the V6 engine, one standardized
template. Every displayed value is a genuine GAIRAEngine output (committed artifacts
built by tools/build_biological_v6.py); nothing is a relabelled legacy radar.
"""
from __future__ import annotations
import numpy as np
import streamlit as st

from .. import components as C, figures as F, biological as B

# datasets present on disk but NOT yet projected through V6 — shown honestly, no output
DEFERRED = {
    "cca_hcc_lm_serum_sers": ("Liver serum SERS (CCA/HCC/LM/NC)", "4-class per-sample txt; "
                              "ingestion not yet wired for V6"),
    "ovarian_plasma_raman_sers": ("Ovarian plasma Raman+SERS", "paired Raman/SERS zips; not yet "
                                  "wired for V6"),
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
    single = [k for k in avail if k not in B.CONTRAST]
    if single:
        names = ", ".join(avail[k]["display_name"] for k in single)
        st.markdown(f'<div class="gaira-caveat">Single-group characterization cohorts ({names}) '
                    'have no disease contrast, so they are not shown in the two-group template '
                    'below. The 81 real donor sera are analysed on '
                    '<b>Page 5 · Serum Spike Stress Test (Section B++)</b>.</div>',
                    unsafe_allow_html=True)


# ── Section B: standardized per-study template ──
def _study(art, bridge):
    gc = B.group_contrast(art)
    a, b = gc["a"], gc["b"]
    patient_level = art["aggregation"] == "patient"
    characterization = art["dataset_id"] in B.CHARACTERIZATION_ONLY
    mnames = {m.id: m.name for m in bridge.mss.motifs}

    # 1 · framing + sample structure
    st.markdown(f"#### {art['display_name']}")
    counts = f"{art['n_units']} units"
    st.markdown(
        f"- **Domain / modality**: {art['domain']} · {art['modality']} · {art['substrate']} · "
        f"{art['excitation_nm']} nm\n"
        f"- **Sample structure**: {counts} · aggregation **{art['aggregation']}-level**"
        + (f" · groups: {', '.join(f'{g}={n}' for g, n in art['n_by_group'].items())}") + "\n"
        f"- **Contrast**: {a} (n={gc['na']}) vs {b} (n={gc['nb']})"
        + ("  ·  ⚠ characterization only (probe-loading, not biology)" if characterization else "")
        + f"\n- **Source**: `{art['source']}`")
    for cav in art["caveats"]:
        st.markdown(f'<div class="gaira-caveat">{cav}</div>', unsafe_allow_html=True)
    st.write("")

    C.debug_panel(bridge, np.array(art["records"][0]["coord"]), art["dataset_id"],
                  extra={"n_units": art["n_units"], "aggregation": art["aggregation"]})

    # 2 · data-quality / OOD
    C.figure(F.group_quality(art),
             cap="OOD, confidence and matrix-share per group — the data-quality context for every "
                 "downstream claim.")

    # 3 · THEME effect sizes (PRIMARY — the group DIFFERENCE, two ways; never overlapping absolutes)
    st.markdown("##### Theme effect sizes  ·  the primary group comparison")
    _, _, dax, dmax = B.group_delta_axes(art)
    fc1, fc2 = st.columns([1.15, 1.0], gap="large")
    with fc1:
        C.figure(F.forest_plot(gc["rows"], a, b, title=f"ΔBSV ({a} − {b})"),
                 cap="Signed theme difference with 95% bootstrap CI; green = FDR q<0.05; δ = "
                     "Cliff's delta (effect size, emphasised over p).",
                 interp="Effect size (δ), not the p-value, is the headline.")
    with fc2:
        C.figure(F.delta_radar(dax, dmax, f"{art['display_name'].split()[0]} ΔBSV",
                               f"{a} − {b} (shared scale ±{dmax:.3f})"),
                 cap="The SAME group difference as a signed radar (lobes out = higher in "
                     f"{a}, in = higher in {b}). Absolute-composition group radars overlap and "
                     "hide this — so the delta is shown instead.")

    # 4 · MSS drivers + 5 · component provenance
    _, _, motif_ids, mdelta = B.motif_contrast(art)
    bmi = [i for i, mid in enumerate(motif_ids) if mid != "colloid_matrix_background"]
    means = B.group_means_by(art, "motifs_mat")
    c3, c4 = st.columns(2, gap="large")
    with c3:
        C.figure(F.difference_bars([mnames.get(motif_ids[i], motif_ids[i]) for i in bmi],
                                   [means[b][i] for i in bmi], [means[a][i] for i in bmi],
                                   title=f"ΔMSS motif drivers ({a} − {b})"),
                 cap="Which spectral motifs drive the theme change.")
    with c4:
        Zt, gt, lt, sat = B.heatmap_matrix(art, "themes_mat")
        C.figure(F.sample_heatmap(Zt, gt, lt, strata=sat),
                 cap="Sample-level BSV (z-scored for display only). Reveals sample heterogeneity "
                     "the group means hide.")
    top = B.top_components_for(art, mdelta)
    st.markdown("**Component provenance** (largest per-group coordinate differences): "
                + ", ".join(f"c{j} ({d:+.3f})" for j, d in top))

    # 6 · sample heterogeneity (distance) — is the difference big vs biological spread?
    dist = B.distance_summary(art)
    d1, d2 = st.columns([1.0, 1.0], gap="large")
    with d1:
        C.figure(F.distance_bars(dist),
                 cap="Between-group distance vs within-group variability.",
                 interp=f"ratio = {dist['ratio']:.2f} — "
                        + ("the group difference exceeds biological heterogeneity."
                           if dist["ratio"] > 1 else
                           "the group difference is SMALL relative to within-group spread."))
    with d2:
        # 7 · dataset-specific: paired (SHINE) else the within-cohort BALANCED view (all cohorts).
        # The balanced view standardizes each theme's deviation so a dominant baseline (the same
        # problem COVID's absolute radar has) no longer hides the real group difference.
        if art["dataset_id"] in B.PAIRED:
            th = gc["rows"][0]["theme"]
            _, _, paired = B.paired_timepoint_theme(art, th)
            C.figure(F.paired_slope(a, b, paired, th,
                                    title=f"Paired {b}→{a} per dose — {F.THEME_SHORT.get(th, th)}"),
                     cap="Longitudinal change within each dose cohort (cohort-level pairing).",
                     interp="Paired-by-dose structure that pooled analysis hides (e.g. a dose×time "
                             "interaction).")
        else:
            themes, za, zb = B.balanced_view(art)
            C.figure(F.balanced_bars(themes, za, zb, a, b),
                     cap="Within-cohort BALANCED view: each theme's deviation standardized so no "
                         "dominant baseline hides the rest (the fix for overlapping absolute "
                         "radars). Display only — the canonical BSV is unchanged.",
                     interp="This is how to see a real-but-small group difference for near-null "
                             "cohorts, instead of two overlapping absolute polygons.")

    # 8 · summary + interpretation (demoted radar/PCA to expanders)
    with st.expander("Absolute BSV radar + sample-space PCA (summary views)"):
        rc1, rc2 = st.columns(2, gap="large")
        with rc1:
            series = [{"name": g, "axes": B.group_radar_axes(art, g)} for g in art["groups"]]
            C.figure(F.multi_radar(series, title="Absolute BSV (composition)"),
                     cap="Absolute atlas position; intentionally coarse (see effect sizes above).")
        with rc2:
            proj, var = B.pca_2d(art["themes_mat"])
            C.figure(F.bio_pca(proj, art["group"], var),
                     cap="Sample BSV PCA (visualization only; axes are NOT themes). Overlap is "
                         "shown honestly.")

    top_row = gc["rows"][0]
    lvl = ("patient-level (1 patient = 1 n)" if patient_level
           else "spectrum-level, EXPLORATORY (no subject mapping; not patient-level inference)")
    n_sig = sum(r["sig"] for r in gc["rows"])
    st.markdown(f'<div class="gaira-card"><b>Statistical summary.</b> Mann-Whitney U · BH-FDR · '
                f'Cliff\'s delta · 2000× bootstrap CIs. Inference level: <b>{lvl}</b>. '
                f'Leading theme: <i>{F.THEME_SHORT.get(top_row["theme"], top_row["theme"])}</i> '
                f'Δ={top_row["delta"]:+.3f} (δ={top_row["cliffs_delta"]:+.2f}, q={top_row["q"]:.3f}); '
                f'{n_sig} themes FDR-significant; heterogeneity ratio {dist["ratio"]:.2f}.</div>',
                unsafe_allow_html=True)

    # per-cohort biochemical interpretation (top MSS motif drivers, cautious language) — for ALL
    _, _, mids2, mdelta2 = B.motif_contrast(art)
    mnames2 = {m.id: m.name for m in bridge.mss.motifs}
    mdrivers = sorted([(mnames2[mids2[i]], mdelta2[i]) for i in range(len(mids2))
                       if mids2[i] != "colloid_matrix_background"], key=lambda x: -abs(x[1]))[:3]
    driver_txt = ", ".join(f"{n} ({d:+.3f})" for n, d in mdrivers)
    st.markdown(f'<div class="gaira-card"><b>Biochemical interpretation ({a} vs {b}).</b> '
                f'The leading MSS motif drivers are {driver_txt}. Consistent with a relative shift '
                f'in {F.THEME_SHORT.get(top_row["theme"], top_row["theme"])}-associated chemistry '
                f'within this cohort; this is an association at the motif level, not a molecule, '
                f'pathway or diagnosis. {DOMAIN_NOTE.get(art["domain"], "")}</div>',
                unsafe_allow_html=True)

    if characterization:
        verdict = (f"{art['display_name']} is a characterization dataset: the contrast is a "
                   f"probe-loading effect, not biology. Use it for EV biochemical-state-space "
                   f"structure, not for a disease claim.")
    elif patient_level and abs(top_row["cliffs_delta"]) > 0.5:
        verdict = (f"A robust <b>patient-level</b> biochemical difference: {n_sig} themes reach FDR "
                   f"significance with large effect sizes — consistent with a real biochemical "
                   f"contrast between {a} and {b} within this cohort.")
    elif max(abs(r["cliffs_delta"]) for r in gc["rows"]) < 0.33:
        verdict = (f"Minimal biochemical separation: effect sizes are tiny (|δ|<0.33) even where p "
                   f"is small at large n. The groups barely separate in BSV space — an honest "
                   f"near-null result, reported as such.")
    else:
        verdict = (f"A moderate, exploratory difference ({n_sig} themes FDR-significant). Consistent "
                   f"with a relative shift; read as associated-with, pending replication.")
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
    C.radar_guide()

    _registry()
    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("### B · Standardized study template")
    avail = {k: v for k, v in B.available().items()
             if v["status"] == "REAL" and k in B.CONTRAST}     # two-group template only
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
