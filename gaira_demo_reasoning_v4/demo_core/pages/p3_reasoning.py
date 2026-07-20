"""Page 3 — How GAIRA Reasons. The MSS layer is the centerpiece.

    Radar  -> which biochemical systems changed?
    MSS    -> which biochemical spectral motifs explain those changes?   (centerpiece)
    Comps  -> what latent evidence supports those motifs?
    Ref    -> which reference chemistries contributed?
"""
from __future__ import annotations
import numpy as np
import streamlit as st
from .. import components as C, figures as F, data as D


def _example_inputs(bridge):
    """A menu of real, cached example spectra to run through the engine."""
    ex = {}
    for cal in D.CALIBRATIONS:
        try:
            Z, meta = D.load_projection(cal.projection)
        except FileNotFoundError:
            continue
        if cal.level_col and cal.level_col in meta.columns:
            i = int(np.asarray(meta[cal.level_col], float).argmax())    # highest dose
            ex[f"{cal.title} (max dose)"] = ("serum" if "serum" in cal.projection else "buffer", Z[i])
        else:
            ex[f"{cal.title}"] = ("serum", Z[-1])
    # a balanced reference point for contrast
    ex["Balanced reference (flat)"] = ("buffer", np.full(D.K, 1.0 / D.K))
    return ex


def render(bridge):
    s = bridge.platform_stats()
    C.page_header(
        "Inference · the reasoning hierarchy",
        "How GAIRA reasons about a spectrum",
        "One spectrum, four levels of explanation. The radar answers <i>which biochemical "
        "systems</i>; the Molecular Spectral Signatures answer <i>which spectral motifs</i>; the "
        "latent components are the <i>evidence</i>; the reference analytes are the <i>grounding "
        "chemistry</i>. This is the layer that distinguishes GAIRA from a Raman classifier.")
    C.question("Given a projected spectrum, what does the engine conclude, and can every "
               "conclusion be traced down to the reference chemistry that supports it?")

    examples = _example_inputs(bridge)
    choice = st.selectbox("Example spectrum (real cached projections through the frozen atlas)",
                          list(examples.keys()))
    domain, coord = examples[choice]
    out, acts = bridge.bsv_and_mss(coord, domain=domain)
    bio_acts = [a for a in acts if not a.non_biochemical]

    # headline state
    C.stat_row([
        (f"{out.bsv.overall_confidence:.2f}", "overall confidence"),
        (f"{out.bsv.ood_score:.2f}", "OOD score"),
        (f"{out.bsv.non_biochemical.get('background_matrix', 0):.2f}", "matrix share"),
        (bio_acts[0].name.split()[0] if bio_acts else "—", "top motif"),
        (domain, "sample domain"),
    ])
    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Level 1: radar ──
    st.markdown("### Level 1 — Radar · which biochemical systems?")
    rc1, rc2 = st.columns([1.0, 1.0], gap="large")
    with rc1:
        C.figure(F.radar(out.radar["axes"]),
                 cap="The 11 evidence-backed biochemical themes (engine radar backend, composition "
                     "share). One visualisation of the BSV — the coarse, top-level summary.",
                 limits="Axes are coupled evidence shares, NOT independent concentrations "
                         "(BSV validation: effective dimensionality ≈ 4 of 11).")
    with rc2:
        st.markdown("**Top themes by evidence share**")
        rows = sorted(out.bsv.biochemical_themes().items(), key=lambda kv: -kv[1])[:5]
        for t, v in rows:
            st.markdown(f"- **{F.THEME_SHORT.get(t, t)}** — share {v:.2f}, "
                        f"confidence {out.bsv.confidence[t]:.2f}")
        st.markdown('<div class="gaira-caption">The radar is intentionally coarse. To see '
                    '<i>why</i> a system is elevated, drop to the MSS layer below.</div>',
                    unsafe_allow_html=True)

    # ── Level 2: MSS — the centerpiece ──
    st.markdown("### Level 2 — Molecular Spectral Signatures · which motifs? "
                "<span style='color:#b2182b'>(centerpiece)</span>", unsafe_allow_html=True)
    C.figure(F.mss_hierarchy(acts),
             cap="Every biochemical motif's elevation vs the pure-Raman reference (signed). "
                 "Warm = elevated, cool = depleted; c = derived motif confidence.",
             interp="This is where spectroscopy, chemistry and biology meet. Motifs are validated "
                     "spectral patterns shared across families of chemistries — not molecules.")

    # MSS explorer — drill into one motif's provenance
    st.markdown("#### Inspect a motif → its components → its reference chemistry")
    mid = st.selectbox("Molecular Spectral Signature",
                       [a.id for a in bio_acts],
                       format_func=lambda i: bridge.motif_by_id(i).name)
    motif = bridge.motif_by_id(mid)
    act = next(a for a in acts if a.id == mid)
    mc1, mc2 = st.columns([1.05, 1.0], gap="large")
    with mc1:
        st.markdown(f"**{motif.name}** — parent theme *{F.THEME_SHORT.get(motif.parent_theme, motif.parent_theme)}*")
        st.markdown(f'<div class="gaira-caption">{motif.description.strip()}</div>',
                    unsafe_allow_html=True)
        st.markdown(f"- Characteristic bands: {', '.join(str(int(b)) for b in motif.bands_cm)} cm⁻¹")
        st.markdown(f"- Derived confidence **{motif.confidence:.2f}** "
                    f"(stability {motif.stability:.2f} × evidence breadth {motif.evidence_breadth:.2f})")
        st.markdown(f"- This query: elevation **{act.elevation:+.2f}**, display {act.display:.2f}")
        pert = motif.perturbation
        st.markdown(
            f"- Perturbation evidence: **{len(pert['dose_responsive_components'])}** dose-responsive "
            f"components, **{len(pert['serum_spike_matches'])}** serum-spike matches"
            + (f", **{len(pert['depletion_matches'])}** depletion matches"
               if pert["depletion_matches"] else ""))
    with mc2:
        st.markdown("**Level 3 — contributing latent components (evidence)**")
        for c in motif.contributors:
            j = c["component"]
            share = float(np.asarray(out.bsv.component_coord)[j])
            st.markdown(f"- **c{j}** · motif weight {c['weight']:.2f} · this-query share {share:.3f}  \n"
                        f"  <span class='gaira-caption'>band {c['band']:.2f} · exemplar "
                        f"{c['exemplar']:.2f} · theme {c['theme']:.2f}</span>",
                        unsafe_allow_html=True)
        if motif.reference_analytes:
            st.markdown("**Level 4 — grounding reference chemistries**")
            st.markdown('<div class="gaira-caption">'
                        + ", ".join(motif.reference_analytes[:10]) + "</div>", unsafe_allow_html=True)

    # top contributing component's basis spectrum
    top_j = motif.contributors[0]["component"]
    grid, spec = bridge.basis_spectrum(top_j)
    C.figure(F.basis_spectrum(grid, spec, bands=motif.bands_cm,
                              title=f"Frozen basis spectrum of c{top_j} (top contributor) "
                                    f"with {motif.name} bands"),
             cap=f"The actual frozen NMF loading of component c{top_j}. Dashed lines mark the "
                 f"motif's curated bands — region-based, never exact-peak.")

    # ── Level 3 full fingerprint ──
    with st.expander("Full 24-component fingerprint (the complete latent evidence)"):
        C.figure(F.component_fingerprint(out.bsv.component_coord,
                                         highlight=[c["component"] for c in motif.contributors]),
                 cap=f"All 24 latent Raman motif coordinates for this query. Highlighted bars are "
                     f"the components contributing to {motif.name}.")

    # ── collisions (educational) ──
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### Why one region ≠ one molecule — spectral collisions")
    st.markdown(
        "Real spectra are mixtures. Multiple motifs place characteristic bands in the same "
        "spectral region, so a single peak cannot be assigned to a single molecule. GAIRA "
        "resolves this by decomposing the spectrum into the frozen component basis first, then "
        "attributing components to motifs — the collision is handled in coordinate space, not by "
        "peak-picking.")
    C.figure(F.band_collision_map(bridge.mss.motifs),
             cap="Characteristic bands of every biochemical motif on a shared axis. Shaded columns "
                 "are regions claimed by ≥3 motifs (e.g. ~640 and ~1330 cm⁻¹).",
             interp="Overlaps are the norm, not the exception. This is exactly why GAIRA reasons "
                     "in the frozen component space rather than matching peaks to molecules.")

    C.takeaways([
        "Every radar conclusion traces down through MSS motifs → components → reference chemistry.",
        "The MSS layer converts 24 mathematical components into ~12 chemically-named motifs.",
        "Motif confidence and perturbation evidence are DERIVED from frozen artifacts, not asserted.",
        "Spectral collisions are resolved in coordinate space — the reason GAIRA is not a peak matcher.",
    ])
    C.caveats([
        "Elevations for SERS inputs are large because SERS is out-of-domain for a Raman atlas — "
        "read the ordering and the confidence, not the absolute z.",
        "A motif contributing to several themes (e.g. c3 → purine, sterol, sulfur) is a real "
        "collision, surfaced here rather than hidden.",
        "MSS is a parallel interpretive view; it does not alter the BSV, which maps components to "
        "themes directly.",
    ])
    C.provenance_footer(s)
