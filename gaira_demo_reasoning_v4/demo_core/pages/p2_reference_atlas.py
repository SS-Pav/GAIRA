"""Page 2 — Reference Atlas. What the frozen atlas learned. (Scaffold + live stats.)"""
from __future__ import annotations
import streamlit as st
from .. import components as C, figures as F


def render(bridge):
    s = bridge.platform_stats()
    C.page_header(
        "Reference · the frozen coordinate system",
        "What the atlas learned",
        "The atlas is a frozen NMF k=24 decomposition of the pure-Raman reference corpus. It "
        "learns molecular <i>classes</i>, not species — the components are latent Raman motifs, "
        "chosen by benchmark (not PCA by default) and held fixed forever.")
    C.question("What chemistry does each of the 24 latent components represent, and how much "
               "reference evidence stands behind it?")

    C.stat_row([
        (f"{s['n_reference_spectra']}", "spectra"),
        (f"{s['n_reference_analytes']}", "analytes"),
        (f"{s['explained_variance']:.0%}" if s.get("explained_variance") else "—", "variance explained"),
        (f"{len(s['excitations'])}", "excitation lines"),
        (f"{s['n_components']}", "components"),
    ])
    st.write("")
    # a working live view already: browse any component's frozen basis spectrum
    st.markdown("### Latent component explorer (live)")
    j = st.slider("Component", 0, s["n_components"] - 1, 3)
    row = bridge.component_row(j)
    cc1, cc2 = st.columns([1.2, 1.0], gap="large")
    with cc1:
        grid, spec = bridge.basis_spectrum(j)
        C.figure(F.basis_spectrum(grid, spec, bands=row["bands"],
                                  title=f"c{j} — frozen basis spectrum"),
                 cap=f"Dominant bands: {', '.join(str(int(b)) for b in row['bands'])} cm⁻¹.")
    with cc2:
        st.markdown(f"**c{j}** · stability {row['stability']:.2f} · purity {row['purity']:.2f}")
        st.markdown(f'<div class="gaira-caption">{row["interpretation"]}</div>',
                    unsafe_allow_html=True)
        st.markdown("**Top reference analytes**")
        for l in row["loadings"][:6]:
            st.markdown(f"- {l['analyte']} ({l['contribution_pct']:.1f}%) · *{l['family']}*")

    C.scaffold_note([
        "PCA of reference spectra coloured by biochemical family (hover: analyte / family / source).",
        "Per-component perturbation-evidence and theme-weight panels, expandable to full provenance.",
        "Source & excitation breakdown with the excitation-invariance validation.",
    ])
    C.caveats([
        "Components are molecular <i>classes</i>, not species; several are low-purity mixtures "
        "(reported honestly in the registry).",
        "Sterol/heme chemistries are under-represented in the corpus — a coverage limit, not a bug.",
    ])
    C.provenance_footer(s)
