"""Page 2 — Building the Reference Space. How GAIRA's biochemical coordinate system is
constructed from pure Raman spectra, taught step by step. Fully grounded in committed
frozen artifacts; NMF-native throughout (PCA is a labelled appendix).
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


# ── A · corpus summary ──
def _corpus_summary(b, s):
    st.markdown("### A · What GAIRA knows — the reference corpus")
    st.markdown('<div class="gaira-caption">Every count is read live from the frozen atlas '
                'manifest + registries — never hard-coded. Corpus stages are labelled precisely '
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
                 cap="Spectra per Raman source and per excitation line (atlas corpus card).")
    with c2:
        ex785 = s["excitations"].get("785.0", s["excitations"].get("785"))
        st.markdown("**Corpus stages (labelled, not merged)**")
        st.table({
            "stage": ["Frozen atlas input", "785 nm subset", "Latent representation",
                      "Interpretation layers", "Calibration (Ag-SERS, excluded)", "Excluded domains"],
            "count / detail": [
                f"{s['n_reference_spectra']} spectra · {s['n_reference_analytes']} analytes",
                f"{ex785} spectra @ 785 nm",
                f"NMF k={s['n_components']} · {s['n_biochemical_mss']} MSS · "
                f"{s['n_biochemical_themes']} themes",
                "Component Registry v1 · MSS v1 · Ontology v2",
                "adenine · ergothioneine · uricase (Pages 4–5)",
                ", ".join(b.eng.atlas.meta["corpus_card"].get("excluded_domains", [])[:4]) + " …"]})


# ── B · how NMF builds the space (educational, Part 3) ──
def _nmf_education(b):
    st.markdown("### B · How the reference space is built — NMF")
    st.markdown('<div class="gaira-caption">GAIRA does not memorise spectra. Non-negative matrix '
                'factorisation (NMF) discovers <b>recurring spectral motifs</b> shared across the '
                'reference corpus — from spectroscopy alone.</div>', unsafe_allow_html=True)

    st.markdown("##### Step 1 · the factorisation")
    C.figure(F.nmf_schematic(),
             cap="The 375×676 reference matrix ≈ a 375×24 coefficient matrix (W) times a 24×676 "
                 "basis matrix (H). The 24 basis rows are the latent Raman motifs; each spectrum "
                 "is a non-negative sum of them.",
             interp="NMF is unsupervised: it sees only spectra — <b>no disease, no labels, no "
                     "biology</b>. The biochemistry is attached afterwards by the ontology + MSS.")

    with st.expander("Why NMF — and not PCA / UMAP / an autoencoder?"):
        st.markdown(
            "GAIRA reasons in the NMF basis because it is the only decomposition that matches the "
            "physics of Raman *and* can be frozen as a shared coordinate system:\n\n"
            "| method | non-negative? | additive (parts)? | a real spectrum you can read? | "
            "deterministic / freezable? |\n"
            "|---|---|---|---|---|\n"
            "| **NMF** (used) | ✅ | ✅ mixture = Σ amount×motif | ✅ each basis IS a spectrum | ✅ |\n"
            "| PCA | ❌ (± lobes) | ❌ (subtraction) | ❌ anti-peaks | ✅ |\n"
            "| UMAP / t-SNE | n/a | ❌ | ❌ axes meaningless | ❌ random init |\n"
            "| autoencoder / CNN | usually ❌ | ❌ | ❌ entangled black box | ❌ |\n\n"
            "- A **Raman mixture is literally a non-negative sum** of its pure components' spectra — "
            "NMF's math *is* the physics; PCA subtracts, UMAP has no generative model, an "
            "autoencoder is non-physical and entangled.\n"
            "- A **component is itself a spectrum** you can plot and match to chemistry (below). A "
            "PC or a UMAP axis is not.\n"
            "- **Freezable + deterministic**: UMAP and autoencoders give a different answer every "
            "run, so they cannot be a canonical, fingerprinted coordinate system; NMF (fixed seed) "
            "is byte-identical.\n"
            "- NMF was chosen **by benchmark**, not by default — it beat PCA, ICA, sparse-dict and "
            "an autoencoder (which scored *worst* on component stability) on the Foundation study.\n\n"
            "This is also the difference from the old V2 / per-dataset approach: instead of running "
            "PCA/clustering *on each dataset* and naming its axes (which can't be compared across "
            "datasets), NMF builds **one fixed axis set from pure reference chemicals**, and every "
            "dataset is projected into it — so cohorts become comparable in the same coordinates.")

    st.markdown("##### Step 2 · a spectrum is a sum of motifs — try it")
    rm = _ref_map()
    default = rm["analytes"].index("adenine") if "adenine" in rm["analytes"] else 0
    analyte = st.selectbox("Reference analyte", rm["analytes"], index=default, key="p2_nmf_analyte")
    coeff = b.analyte_coeffs(analyte)
    share = coeff / (coeff.sum() + 1e-12)
    order = [int(j) for j in np.argsort(-share)]
    k = st.slider("Reconstruct using the top-k components", 1, 8, 3, key="p2_recon_k")
    c1, c2 = st.columns([1.0, 1.15], gap="large")
    with c1:
        C.figure(F.decomposition_bars(coeff, analyte),
                 cap=f"{analyte.capitalize()}'s strongest latent motifs, as a share of its "
                     f"evidence.")
    with c2:
        grid, full = b.reconstruct_from(coeff)
        _, partial = b.reconstruct_from(coeff, keep=order[:k])
        cum = share[order[:k]].sum()
        C.figure(F.reconstruction_overlay(grid, full, partial, k, analyte),
                 cap=f"Top {k} components already capture {cum*100:.0f}% of "
                     f"{analyte}'s evidence.",
                 interp="Adding components adds spectral detail — components are ADDITIVE motifs, "
                         "not clusters.")
    st.markdown('<div class="gaira-take">This is the whole idea: every query spectrum GAIRA sees '
                'is expressed as a non-negative sum of these 24 fixed motifs. Those 24 coefficients '
                'ARE the coordinate system the rest of the engine reasons in.</div>',
                unsafe_allow_html=True)


# ── C · component atlas (primary NMF-native explorer) ──
def _component_explorer(b, s):
    st.markdown("### C · Component Atlas — the 24 latent Raman motifs")
    tbl = b.reg.summary_table()
    tbl["tier"] = tbl["confidence"].map(_conf_tier)
    with st.expander("Overview grid (all 24 components + confidence tier)"):
        st.dataframe(tbl[["component", "interpretation", "stability", "purity",
                          "n_dose_responsive", "tier"]], use_container_width=True, hide_index=True)

    j = st.select_slider("Component", options=list(range(s["n_components"])), value=3, key="p2_comp")
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
        for m in (lm[:6] or [{"name": "*none above threshold*", "weight": 0}]):
            st.markdown(f"- {m['name']}" + (f" ({m['weight']:.2f})" if m['weight'] else ""))
    with d3:
        st.markdown("**Component → theme weights**")
        for w in b.component_theme_weights(j):
            st.markdown(f"- {F.THEME_SHORT.get(w['theme'], w['theme'])} ({w['weight']:.2f})")

    lm = b.component_linked_motifs(j)
    if lm:
        C.figure(F.component_ego_network(j, lm),
                 cap=f"c{j} → its MSS motifs → their biochemical themes (edge width ∝ weight). "
                     "Many-to-many by design.")
    if j == 3:
        st.markdown('<div class="gaira-take"><b>c3 — an educational case.</b> The earlier coarse '
                    'audit label called c3 "sterol", which is misleading. Its strongest reference '
                    'loading is <b>adenine</b> and it responds most to the adenine perturbation '
                    'series — so the evidence supports a <b>purine-associated</b> reading '
                    '(nucleic_purine weight 0.47), shown above.</div>', unsafe_allow_html=True)


# ── D · component relationships (hierarchy primary; MDS optional) ──
def _component_relationships(b):
    st.markdown("### D · How the 24 components relate")
    st.markdown('<div class="gaira-caption">Grouping components by the cosine distance between '
                'their basis spectra shows which motifs are spectrally similar. This is the '
                'representation itself — not the inference space.</div>', unsafe_allow_html=True)
    D = b.component_distance()
    tbc = {j: b.component_dominant_theme(j) for j in range(24)}
    C.figure(F.component_dendrogram(D, tbc),
             cap="Average-linkage hierarchical clustering of the 24 basis spectra, annotated by "
                 "dominant theme. Deterministic.",
             interp="Spectrally similar motifs cluster (e.g. the saccharide and purine components) "
                     "— structure that is chemically interpretable, unlike a raw scatter.")
    with st.expander("Optional · 2-D similarity map (MDS) — what am I looking at?"):
        st.markdown('<div class="gaira-caption">Classical MDS places the 24 components so that '
                    'distances approximate basis-spectrum dissimilarity. <b>The axes have no '
                    'physical meaning</b>; only relative distances matter — nearby components have '
                    'similar basis spectra, distant ones represent different motifs. Provided for '
                    'exploration, not as a primary result.</div>', unsafe_allow_html=True)
        C.figure(F.component_similarity_map(D, tbc),
                 cap="Classical MDS of the 24 components (exploratory).")


# ── E · MSS atlas (focused default, full-network toggle) ──
def _mss_atlas(b):
    st.markdown("### E · MSS atlas — motifs bridge components to themes")
    mid = st.selectbox("Molecular Spectral Signature", [m.id for m in b.mss.motifs],
                       format_func=lambda i: b.motif_by_id(i).name, key="p2_mss")
    m = b.motif_by_id(mid)
    scope = ("broad biochemical" if m.evidence_breadth >= 0.95 and len(m.contributors) >= 5
             else "molecular-like" if len(m.reference_analytes) <= 3 else "subclass-like")
    e1, e2 = st.columns([1.0, 1.0], gap="large")
    with e1:
        st.markdown(f"**{m.name}** · scope *{scope}* · parent theme "
                    f"*{F.THEME_SHORT.get(m.parent_theme, m.parent_theme)}*"
                    + ("  ·  ⚠ non-biochemical" if m.non_biochemical else ""))
        st.markdown(f'<div class="gaira-caption">{m.description.strip()}</div>', unsafe_allow_html=True)
        st.markdown(f"- Characteristic bands: {', '.join(str(int(x)) for x in m.bands_cm)} cm⁻¹")
        st.markdown(f"- Confidence **{m.confidence:.2f}** (stability {m.stability:.2f} × breadth "
                    f"{m.evidence_breadth:.2f})")
        st.markdown(f"- Exemplars: {', '.join(m.reference_analytes[:8]) or '—'}")
    with e2:
        st.markdown("**Contributing components (evidence-derived weights)**")
        for c in m.contributors:
            st.markdown(f"- c{c['component']} · weight {c['weight']:.2f} "
                        f"<span class='gaira-caption'>(band {c['band']:.2f} · exemplar "
                        f"{c['exemplar']:.2f} · theme {c['theme']:.2f})</span>",
                        unsafe_allow_html=True)
    if st.toggle("Show the full component → MSS → theme network", value=False, key="p2_fullnet"):
        st.markdown('<div class="gaira-caption">The complete many-to-many map (deliberately dense; '
                    'no one-to-one implied). Use the component explorer above for the focused '
                    'view.</div>', unsafe_allow_html=True)
        st.plotly_chart(IV.component_theme_sankey(_sankey()), use_container_width=True)


# ── F · appendix: PCA of NMF coefficient vectors ──
def _pca_appendix(b):
    st.markdown("### F · Appendix · PCA of reference-analyte coefficient vectors")
    st.markdown('<div class="gaira-caption"><b>What is plotted:</b> PCA of the 167 reference '
                'analytes\' <b>24-component NMF coefficient vectors</b> — i.e. the representation '
                'GAIRA actually reasons with (option B), <i>not</i> raw spectra and <i>not</i> the '
                'basis vectors. It is an <b>exploratory</b> 2-D view; inference uses the full 24-D '
                'coordinates, never this projection.</div>', unsafe_allow_html=True)
    rm = _ref_map()
    fam_all = sorted(set(rm["families"]))
    pick = st.multiselect("Show families", fam_all, default=fam_all, key="p2_fam")
    fig, var = IV.reference_pca(rm, show_families=set(pick))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f'<div class="gaira-caption">PC1 {var[0]:.0%}, PC2 {var[1]:.0%} of coefficient '
                f'variance. Families overlap because Raman spectra are mixtures, not unique '
                f'molecular barcodes — expected, and not a failure of the NMF engine.</div>',
                unsafe_allow_html=True)


def render(bridge):
    s = bridge.platform_stats()
    C.page_header(
        "Reference · building the coordinate system",
        "Building the reference space",
        "GAIRA's biochemical coordinate system is a frozen NMF decomposition of pure Raman "
        "reference spectra — chosen by benchmark, held fixed forever. This page shows how it is "
        "built (NMF), what each of the 24 latent motifs means, how they relate, and how motifs "
        "bridge to biochemical themes. Everything downstream reasons in these coordinates.")
    C.question("What biochemical evidence does GAIRA know, and how is the frozen coordinate system "
               "constructed from it?")

    _corpus_summary(bridge, s)
    st.markdown("<hr/>", unsafe_allow_html=True)
    _nmf_education(bridge)
    st.markdown("<hr/>", unsafe_allow_html=True)
    _component_explorer(bridge, s)
    st.markdown("<hr/>", unsafe_allow_html=True)
    _component_relationships(bridge)
    st.markdown("<hr/>", unsafe_allow_html=True)
    _mss_atlas(bridge)
    st.markdown("<hr/>", unsafe_allow_html=True)
    _pca_appendix(bridge)
    st.markdown("<hr/>", unsafe_allow_html=True)

    C.takeaways([
        "NMF discovers 24 recurring Raman motifs from spectroscopy alone — no labels, no biology.",
        "Every spectrum is a non-negative SUM of those motifs; the 24 coefficients are the "
        "coordinate system.",
        "The Component Registry + MSS layer + ontology attach evidence-based, many-to-many "
        "biochemistry afterwards.",
        "The atlas is fixed; interpretation stays versioned and improvable.",
    ])
    C.caveats([
        "Components are molecular <i>classes</i>, not species; several are low-purity mixtures "
        "(shown with stability, purity, confidence).",
        "Sterol and heme chemistries are under-represented in the corpus — a coverage limit.",
        "The MDS map and the coefficient PCA are exploratory views; inference uses the fixed NMF "
        "basis and the full 24-D coordinates.",
    ])
    C.related(["3 · How GAIRA Reasons", "4 · Calibration", "8 · Methods & Provenance"])
    C.provenance_footer(s)
