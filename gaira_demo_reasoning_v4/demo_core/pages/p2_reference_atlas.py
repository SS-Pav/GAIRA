"""Page 2 — Reference Atlas. What biochemical evidence GAIRA knows, and how the
frozen coordinate system is constructed. Fully grounded in committed frozen artifacts.
"""
from __future__ import annotations
import numpy as np
import streamlit as st

from .. import components as C, figures as F, interactive as IV
from ..engine_bridge import get_bridge


@st.cache_data(show_spinner=False)
def _ref_map():
    return get_bridge().reference_map()


@st.cache_data(show_spinner=False)
def _sankey():
    return get_bridge().sankey_links()


def _conf_tier(v):
    if isinstance(v, str):
        s = v.lower()
        return "high" if "high" in s else "low" if "low" in s else "moderate"
    try:
        v = float(v)
        return "high" if v >= 0.66 else "low" if v < 0.4 else "moderate"
    except Exception:
        return "moderate"


# ── Section A ──
def _corpus_summary(b, s):
    st.markdown("### A · Corpus summary")
    st.markdown('<div class="gaira-caption">Every count below is read live from the frozen atlas '
                'manifest and registries — never hard-coded. Corpus stages are labelled precisely '
                'and are <b>not</b> merged.</div>', unsafe_allow_html=True)
    C.stat_row([
        (f"{s['n_reference_analytes']}", "reference analytes"),
        (f"{s['n_reference_spectra']}", "atlas-input spectra"),
        (f"{len(s['sources'])}", "Raman sources"),
        (f"{s['n_components']}", "latent components"),
        (f"{s['n_biochemical_mss']}", "MSS motifs"),
        (f"{s['n_biochemical_themes']}", "biochemical themes"),
    ])
    st.write("")
    c1, c2 = st.columns([1.1, 1.0], gap="large")
    with c1:
        C.figure(F.corpus_breakdown(s["sources"], s["excitations"]),
                 cap="Spectra per Raman source and per excitation line, from the atlas corpus card.")
    with c2:
        st.markdown("**Corpus stages (labelled, not merged)**")
        ex785 = s["excitations"].get("785.0", s["excitations"].get("785"))
        st.table({
            "stage": ["Frozen atlas input", "785 nm subset", "Raman sources",
                      "Latent representation", "Interpretation layers",
                      "Calibration (Ag-SERS, excluded)", "Excluded domains"],
            "count / detail": [
                f"{s['n_reference_spectra']} spectra · {s['n_reference_analytes']} analytes",
                f"{ex785} spectra @ 785 nm",
                " · ".join(f"{k} ({v})" for k, v in s["sources"].items()),
                f"NMF k={s['n_components']} · {s['n_biochemical_mss']} MSS · "
                f"{s['n_biochemical_themes']} themes",
                "Component Registry v1 · MSS v1 · Ontology v2",
                "adenine · ergothioneine · uricase (Pages 4–5)",
                ", ".join(b.eng.atlas.meta["corpus_card"].get("excluded_domains", [])[:4]) + " …",
            ]})
    st.markdown('<div class="gaira-caveat"><b>Not shown as atlas counts.</b> Matched cross-modality '
                'Ag-SERS subsets, biological cohorts and the full acquisition pool are separate '
                'corpus stages presented on the Serum Spike (Page 5) and Biological Studies '
                '(Page 6) pages — they are NOT part of the frozen Raman atlas and are not merged '
                'into the counts above.</div>', unsafe_allow_html=True)


# ── Section B — NMF-native (primary): the learned representation itself ──
def _component_map(b):
    st.markdown("### B · The learned representation — NMF component map")
    st.markdown('<div class="gaira-caption">GAIRA inference IS the frozen NMF decomposition, so '
                'the primary atlas view is the 24 components themselves — not a PCA of spectra. '
                'This map visualises component SIMILARITY (classical MDS on cosine distance between '
                'basis spectra); it is not the inference space.</div>', unsafe_allow_html=True)
    D = b.component_distance()
    tbc = {j: b.component_dominant_theme(j) for j in range(24)}
    m1, m2 = st.columns([1.05, 1.0], gap="large")
    with m1:
        C.figure(F.component_similarity_map(D, tbc),
                 cap="24 NMF components (MDS on basis-spectrum cosine distance), coloured by "
                     "dominant biochemical theme. Deterministic and distance-preserving.",
                 interp="Components with similar basis spectra sit together (e.g. the glycan and "
                         "purine families). This is the representation GAIRA reasons in.",
                 limits="A 2-D projection of a 24×24 distance; read clusters, not exact positions.")
    with m2:
        C.figure(F.component_dendrogram(D, tbc),
                 cap="Hierarchical clustering (average linkage) of the same basis-spectrum "
                     "distances, annotated by dominant theme.")


# ── Section B2 — secondary/exploratory reference-spectrum PCA ──
def _reference_map(b):
    st.markdown("### B2 · Reference family map  ·  <i>exploratory, secondary</i>",
                unsafe_allow_html=True)
    st.markdown('<div class="gaira-caption"><b>Exploratory visualization of reference-spectrum '
                'variation; not used for inference.</b> PCA of the 167 reference analytes in the '
                'frozen 24-component space, coloured by family. Weak family separation here is a '
                'property of PCA on overlapping mixtures — NOT a failure of the NMF engine.</div>',
                unsafe_allow_html=True)
    rm = _ref_map()
    fam_all = sorted(set(rm["families"]))
    pick = st.multiselect("Show families", fam_all, default=fam_all, key="p2_fam")
    fig, var = IV.reference_pca(rm, show_families=set(pick))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f'<div class="gaira-caption">PC1 {var[0]:.0%}, PC2 {var[1]:.0%} of variance. '
                f'Families overlap because Raman spectra are <i>not</i> unique molecular '
                f'barcodes — a mixture-first, not fingerprint, view.</div>', unsafe_allow_html=True)


# ── Section C ──
def _component_explorer(b, s):
    st.markdown("### C · Component explorer — all 24 latent Raman motifs")
    tbl = b.reg.summary_table()
    tbl["tier"] = tbl["confidence"].map(_conf_tier)
    with st.expander("Component overview grid (all 24, with confidence tier)"):
        st.dataframe(tbl[["component", "interpretation", "stability", "purity",
                          "n_dose_responsive", "tier"]], use_container_width=True, hide_index=True)

    j = st.select_slider("Component", options=list(range(s["n_components"])), value=3,
                         key="p2_comp")
    row = b.component_row(j)
    tier = _conf_tier(b.reg.value(j, "interpretation_confidence"))
    tier_css = {"high": "gaira-take", "moderate": "gaira-card", "low": "gaira-caveat"}[tier]

    cc1, cc2 = st.columns([1.25, 1.0], gap="large")
    with cc1:
        grid, spec = b.basis_spectrum(j)
        C.figure(F.basis_spectrum(grid, spec, bands=row["bands"],
                                  title=f"c{j} — frozen basis spectrum"),
                 cap=f"Dominant bands: {', '.join(str(int(x)) for x in row['bands'])} cm⁻¹.")
    with cc2:
        st.markdown(f'<div class="{tier_css}"><b>c{j}</b> · <b>{tier}</b>-confidence interpretation'
                    f'<br>stability {row["stability"]:.2f} · purity {row["purity"]:.2f} · '
                    f'{row["n_dose_responsive"]} dose-responsive experiments</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="gaira-caption">{row["interpretation"]}</div>',
                    unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3, gap="large")
    with d1:
        st.markdown("**Top reference analytes**")
        for l in row["loadings"][:6]:
            st.markdown(f"- {l['analyte']} ({l['contribution_pct']:.1f}%) · *{l['family']}*")
    with d2:
        st.markdown("**Linked MSS motifs**")
        lm = b.component_linked_motifs(j)
        if lm:
            for m in lm[:6]:
                st.markdown(f"- {m['name']} ({m['weight']:.2f})")
        else:
            st.markdown("*none above threshold*")
    with d3:
        st.markdown("**Component → theme weights**")
        for w in b.component_theme_weights(j):
            st.markdown(f"- {F.THEME_SHORT.get(w['theme'], w['theme'])} ({w['weight']:.2f})")

    if j == 3:
        st.markdown('<div class="gaira-take"><b>c3 — an educational case.</b> The earlier coarse '
                    'audit label called c3 "sterol", which is misleading. Its strongest reference '
                    'loading is <b>adenine</b> and it is the component that responds most to the '
                    'adenine perturbation series — so reference-loading + perturbation evidence '
                    'support a strong <b>purine-associated</b> interpretation. GAIRA shows the '
                    'many-to-many evidence rather than forcing one label.</div>',
                    unsafe_allow_html=True)


# ── Section D ──
def _mss_atlas(b):
    st.markdown("### D · MSS atlas — the interpretable motifs")
    mid = st.selectbox("Molecular Spectral Signature", [m.id for m in b.mss.motifs],
                       format_func=lambda i: b.motif_by_id(i).name, key="p2_mss")
    m = b.motif_by_id(mid)
    scope = ("broad biochemical" if m.evidence_breadth >= 0.95 and len(m.contributors) >= 5
             else "molecular-like" if len(m.reference_analytes) <= 3 else "subclass-like")
    e1, e2 = st.columns([1.0, 1.0], gap="large")
    with e1:
        st.markdown(f"**{m.name}** · scope: *{scope}* · parent theme "
                    f"*{F.THEME_SHORT.get(m.parent_theme, m.parent_theme)}*"
                    + ("  ·  ⚠ non-biochemical" if m.non_biochemical else ""))
        st.markdown(f'<div class="gaira-caption">{m.description.strip()}</div>', unsafe_allow_html=True)
        st.markdown(f"- Characteristic bands: {', '.join(str(int(x)) for x in m.bands_cm)} cm⁻¹")
        st.markdown(f"- Confidence **{m.confidence:.2f}** (stability {m.stability:.2f} × breadth "
                    f"{m.evidence_breadth:.2f})")
        st.markdown(f"- Exemplar reference analytes: {', '.join(m.reference_analytes[:8]) or '—'}")
    with e2:
        st.markdown("**Contributing components (evidence-derived weights)**")
        for c in m.contributors:
            st.markdown(f"- c{c['component']} · weight {c['weight']:.2f} "
                        f"<span class='gaira-caption'>(band {c['band']:.2f} · exemplar "
                        f"{c['exemplar']:.2f} · theme {c['theme']:.2f})</span>",
                        unsafe_allow_html=True)
        pert = m.perturbation
        st.markdown(f"**Perturbation evidence** · {len(pert['dose_responsive_components'])} "
                    f"dose-responsive, {len(pert['serum_spike_matches'])} serum-spike"
                    + (f", {len(pert['depletion_matches'])} depletion" if pert["depletion_matches"]
                       else ""))
    st.markdown('<div class="gaira-caption"><b>Ambiguity / collisions.</b> A component can feed '
                'several motifs and a motif several themes — the flow below is deliberately '
                'many-to-many. No strict one-to-one mapping is implied.</div>', unsafe_allow_html=True)
    st.markdown("#### Components → MSS → biochemical themes")
    st.plotly_chart(IV.component_theme_sankey(_sankey()), use_container_width=True)


def render(bridge):
    s = bridge.platform_stats()
    C.page_header(
        "Reference · the frozen coordinate system",
        "What the atlas learned",
        "The atlas is a frozen NMF k=24 decomposition of the pure-Raman reference corpus — chosen "
        "by benchmark, not PCA by default, and held fixed forever. It supplies stable numerical "
        "coordinates; the Component Registry and MSS layer supply evidence-based interpretation; "
        "the ontology groups motifs into broad biochemical themes.")
    C.question("What biochemical evidence does GAIRA know, and how is the frozen coordinate system "
               "constructed from it?")

    _corpus_summary(bridge, s)
    st.markdown("<hr/>", unsafe_allow_html=True)
    _component_map(bridge)                    # NMF-native primary view
    st.markdown("<hr/>", unsafe_allow_html=True)
    _component_explorer(bridge, s)
    st.markdown("<hr/>", unsafe_allow_html=True)
    _reference_map(bridge)                    # PCA demoted to exploratory secondary
    st.markdown("<hr/>", unsafe_allow_html=True)
    _mss_atlas(bridge)
    st.markdown("<hr/>", unsafe_allow_html=True)

    C.takeaways([
        "The frozen atlas supplies stable numerical coordinates; nothing downstream changes it.",
        "The Component Registry and MSS layer supply evidence-based, many-to-many interpretation.",
        "The ontology groups motifs into broad biochemical themes (the radar).",
        "The atlas is fixed; interpretation remains versioned and improvable.",
    ])
    C.caveats([
        "Components are molecular <i>classes</i>, not species; several are low-purity mixtures "
        "(shown honestly with stability, purity and confidence tier).",
        "Sterol and heme chemistries are under-represented in the corpus — a coverage limit.",
        "The PCA and Sankey are explanatory views; the inference model is the fixed NMF basis.",
    ])
    C.related(["3 · How GAIRA Reasons", "4 · Calibration", "8 · Methods & Provenance"])
    C.provenance_footer(s)
