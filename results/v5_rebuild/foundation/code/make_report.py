"""Foundation model — figures + MD/PDF report."""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
from gaira.foundation import dataset as DS, latent_space as LS, bsv as BSV, axes as AX
from gaira.foundation.families_raman import family_of

OUT = REPO / "results/v5_rebuild/foundation"
TAB, FIG, ART = OUT / "tables", OUT / "figures", OUT / "artifacts"
FIG.mkdir(parents=True, exist_ok=True)
PDF_PATH = REPO / "GAIRA_V5_FOUNDATION_MODEL_REPORT.pdf"

P, S, INK, MUTED, GRIDC = "#2563EB", "#D97706", "#1f2328", "#6B7280", "#E5E7EB"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.grid": True,
                     "grid.color": GRIDC, "grid.linewidth": 0.5, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.labelcolor": INK,
                     "axes.titlecolor": INK, "legend.frameon": False,
                     "figure.facecolor": "white"})
PAGE = (8.5, 11.0)


def _wrap(t, n=104):
    out, line = [], ""
    for w in t.split():
        if len(line) + len(w) + 1 > n:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line: out.append(line)
    return out


def text_page(pdf, title, blocks, subtitle=None):
    fig = plt.figure(figsize=PAGE); y = 0.95
    fig.text(0.07, y, title, fontsize=17, fontweight="bold", color=INK); y -= 0.033
    if subtitle:
        fig.text(0.07, y, subtitle, fontsize=9.5, color=MUTED); y -= 0.03
    for kind, txt in blocks:
        if y < 0.06:
            pdf.savefig(fig); plt.close(fig); fig = plt.figure(figsize=PAGE); y = 0.95
        if kind == "h":
            y -= 0.014; fig.text(0.07, y, txt, fontsize=11, fontweight="bold", color=INK); y -= 0.025
        elif kind == "b":
            for ln in _wrap(txt):
                fig.text(0.07, y, ln, fontsize=8.6, color=INK); y -= 0.0165
            y -= 0.008
        elif kind == "m":
            fig.text(0.075, y, txt, fontsize=7.4, color=INK, family="DejaVu Sans Mono"); y -= 0.014
    pdf.savefig(fig); plt.close(fig)


def main():
    bench = pd.read_csv(TAB / "c1_representation_benchmark.csv")
    sel = json.loads((TAB / "c1_selection.json").read_text())
    st = json.loads((TAB / "c2_manifold_stats.json").read_text())
    axinfo = json.loads((TAB / "c3_axes.json").read_text())
    axes_list = axinfo["axes"]
    comp_df = pd.read_csv(TAB / "c3_components.csv")
    bsv_ax = pd.read_csv(TAB / "c4_bsv_axes.csv")
    mss = pd.read_csv(TAB / "c5_mss.csv")
    v = json.loads((TAB / "c6_validation_summary.json").read_text())
    exc = pd.read_csv(TAB / "c6_excitation_transfer.csv")
    src = pd.read_csv(TAB / "c6_source_transfer.csv")
    gsum = pd.read_csv(TAB / "c7_serum_group_summary.csv")
    card = json.loads((TAB / "raman_dataset_card.json").read_text())

    corpus = DS.load_reference_corpus()
    man = LS.build_manifold(corpus, sel["representation"], int(sel["k"]), seed=0)
    Zn = man.coordinates(corpus.X, normalise=True)

    # ── F1 C1 benchmark ──
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axs[0]
    for name, g in bench.groupby("representation"):
        g = g.sort_values("k")
        ax.plot(g.k, g.total_score, "-o", ms=4, label=name)
    ax.set_xlabel("latent dimension k"); ax.set_ylabel("multi-criteria score")
    ax.set_title("C1 — representation benchmark"); ax.legend(fontsize=7)
    ax = axs[1]
    b = bench.sort_values("total_score", ascending=False).groupby("representation").head(1)
    m = ["neighbourhood_preservation", "replicate_robustness", "component_stability", "loading_sparsity"]
    x = np.arange(len(b)); w = 0.2
    for i, col in enumerate(m):
        ax.bar(x + (i - 1.5) * w, b[col], w, label=col.replace("_", " "))
    ax.set_xticks(x); ax.set_xticklabels([f"{r.representation}\nk={int(r.k)}" for _, r in b.iterrows()], fontsize=7)
    ax.legend(fontsize=6.5); ax.set_title("Best per representation (reconstruction is only 10% of the score)")
    fig.tight_layout(); fig.savefig(FIG / "f1_c1_benchmark.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── F2 manifold ──
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    pcv = np.array(st["per_component_variance"])
    axs[0].bar(range(len(pcv)), np.sort(pcv)[::-1], color=P)
    axs[0].set_xlabel("component (sorted)"); axs[0].set_ylabel("variance share")
    axs[0].set_title(f"C2 — component variance (total EV {st['explained_variance']:.2f})")
    stab = np.array(st["bootstrap_component_stability"]["per_component"])
    axs[1].bar(range(len(stab)), np.sort(stab)[::-1], color=S)
    axs[1].axhline(np.mean(stab), ls="--", c=INK, lw=1)
    axs[1].set_xlabel("component (sorted)"); axs[1].set_ylabel("bootstrap stability")
    axs[1].set_title(f"Component stability (mean {np.mean(stab):.2f})")
    idim = st["intrinsic_dimensionality"]
    axs[2].bar(["participation\nratio", "90% latent\nvariance", "entropy\nrank"],
               [idim["participation_ratio"], idim["n_components_90pct_latent_var"],
                idim["effective_rank_entropy"]], color=[P, S, MUTED])
    axs[2].set_title("Intrinsic dimensionality"); axs[2].set_ylabel("effective components")
    fig.tight_layout(); fig.savefig(FIG / "f2_manifold.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── F3 axes ──
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    a = pd.DataFrame(axes_list).sort_values("variance_share", ascending=True)
    colmap = {"medium-high": P, "medium": "#60A5FA", "low": S}
    axs[0].barh([f"{r.axis}. {r.tentative_theme}" for _, r in a.iterrows()], a.variance_share,
                color=[colmap.get(c, MUTED) for c in a.theme_confidence])
    axs[0].set_xlabel("share of total latent activation")
    axs[0].set_title("C3 — emergent biochemical axes\n(blue = medium-high confidence, amber = low)")
    W = man.rep.components_
    show = [ax_ for ax_ in axes_list if ax_["theme_confidence"] != "low"][:6]
    for i, ax_ in enumerate(show):
        prof = W[ax_["components"]].sum(axis=0)
        prof = prof / (prof.max() + 1e-12)
        axs[1].plot(corpus.grid, prof + i * 1.15, lw=0.9)
        axs[1].text(corpus.grid[5], i * 1.15 + 0.85, f"axis {ax_['axis']}: {ax_['tentative_theme']}",
                    fontsize=7, color=INK)
    axs[1].set_xlabel("Raman shift (cm⁻¹)"); axs[1].set_yticks([])
    axs[1].set_title("Axis spectral loadings (non-negative)")
    fig.tight_layout(); fig.savefig(FIG / "f3_axes.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── F4 latent map + BSV heatmap ──
    from sklearn.decomposition import PCA as _P
    A = pd.read_csv(TAB / "c3_analyte_activation_matrix.csv", index_col=0)
    Y = _P(2, random_state=0).fit_transform(A.values / (A.values.sum(1, keepdims=True) + 1e-12))
    fams = np.array([family_of(x) for x in A.index])
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    top_f = pd.Series(fams).value_counts().head(8).index.tolist()
    for f in top_f:
        m = fams == f
        axs[0].scatter(Y[m, 0], Y[m, 1], s=22, alpha=0.8, label=f)
    axs[0].legend(fontsize=6.5, ncol=2); axs[0].set_title("C2/C4 — analyte map in the biochemical manifold")
    axs[0].set_xlabel("PC1 of BSV"); axs[0].set_ylabel("PC2 of BSV")
    axcols = [c for c in bsv_ax.columns if c.startswith("axis")]
    order = np.argsort(-bsv_ax[axcols].values.max(axis=1))[:45]
    H = bsv_ax.iloc[order]
    im = axs[1].imshow(H[axcols].values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axs[1].set_yticks(range(len(H))); axs[1].set_yticklabels(H.analyte, fontsize=4.5)
    axs[1].set_xticks(range(len(axcols)))
    axs[1].set_xticklabels([c.split("_", 1)[-1] for c in axcols], rotation=45, ha="right", fontsize=6)
    axs[1].set_title("C4 — BSV (axis shares), 45 most concentrated analytes"); axs[1].grid(False)
    plt.colorbar(im, ax=axs[1], fraction=0.03)
    fig.tight_layout(); fig.savefig(FIG / "f4_bsv.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── F5 validation ──
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    axs[0].hist(exc.cross_level_cos, bins=18, color=P, alpha=0.9)
    axs[0].axvline(v["excitation_transfer"]["null_different_analyte_cos"], ls="--", c=S, lw=1.5)
    axs[0].text(v["excitation_transfer"]["null_different_analyte_cos"], axs[0].get_ylim()[1] * 0.9,
                " null (different analytes)", fontsize=7, color=S)
    axs[0].set_xlabel("cross-excitation BSV cosine"); axs[0].set_ylabel("analytes")
    axs[0].set_title(f"C6 — excitation transfer (n={len(exc)})")
    axs[1].hist(src.cross_level_cos, bins=18, color=P, alpha=0.9)
    axs[1].axvline(v["source_transfer"]["null_different_analyte_cos"], ls="--", c=S, lw=1.5)
    axs[1].set_xlabel("cross-source BSV cosine"); axs[1].set_title(f"C6 — source transfer (n={len(src)})")
    h = v["heldout_analyte"]
    axs[2].bar(["within\nanalyte", "between\nanalyte"], [h["within_analyte_cos"], h["between_analyte_cos"]],
               color=[P, MUTED])
    axs[2].set_title(f"C6 — held-out analytes (margin {h['bsv_margin']:+.2f})")
    axs[2].set_ylabel("BSV cosine")
    fig.tight_layout(); fig.savefig(FIG / "f5_validation.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── F6 serum projection ──
    axcols7 = [c for c in gsum.columns if c.startswith("mean_axis")]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    x = np.arange(len(axcols7)); w = 0.2
    for i, (_, r) in enumerate(gsum.iterrows()):
        axs[0].bar(x + (i - 1.5) * w, r[axcols7].values.astype(float), w, label=f"{r.group} (n={int(r.n)})")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([c.replace("mean_axis", "").split("_", 1)[-1] for c in axcols7],
                           rotation=40, ha="right", fontsize=6.5)
    axs[0].set_ylabel("axis share of biochemical evidence")
    axs[0].set_title("C7 — serum Raman projected into the frozen manifold"); axs[0].legend(fontsize=7)
    proj = pd.read_csv(TAB / "c7_serum_projection.csv")
    grp = [g for g in proj.group.unique()]
    axs[1].boxplot([proj[proj.group == g].ood_score for g in grp], labels=grp)
    axs[1].set_ylabel("distance to reference support (OOD)")
    axs[1].set_title("Out-of-distribution score by group")
    fig.tight_layout(); fig.savefig(FIG / "f6_serum.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── PDF ──
    with PdfPages(PDF_PATH) as pdf:
        fig = plt.figure(figsize=PAGE)
        fig.text(0.5, 0.78, "Raman Biochemical Foundation Model", ha="center", fontsize=23,
                 fontweight="bold", color=INK)
        fig.text(0.5, 0.735, "GAIRA V5 — Phases C1–C7 (Raman-only)", ha="center", fontsize=13, color=MUTED)
        ax = fig.add_axes([0.13, 0.40, 0.74, 0.28]); ax.axis("off")
        rows = [["Reference corpus", f"{card['n_spectra']} Raman spectra · {card['n_analytes']} pure analytes"],
                ["Window / grid", "450–1800 cm⁻¹ @ 2 cm⁻¹ (676 bins)"],
                ["Representation selected", f"{sel['representation']} (k={sel['k']}) — benchmark-chosen, PCA ranked 3rd"],
                ["Explained variance", f"{st['explained_variance']:.3f}"],
                ["Intrinsic dimensionality", f"~{st['intrinsic_dimensionality']['participation_ratio']:.0f} "
                                             f"(90% latent var at {st['intrinsic_dimensionality']['n_components_90pct_latent_var']})"],
                ["Emergent axes", f"{len(axes_list)} tentative biochemical axes"],
                ["Excitation transfer", f"{v['excitation_transfer']['mean_cross_excitation_cos']:.3f} "
                                        f"vs null {v['excitation_transfer']['null_different_analyte_cos']:.3f}"],
                ["Excluded domains", "Ag-SERS · Au-SERS · DART · serum Ag-colloid"]]
        t = ax.table(cellText=rows, colWidths=[0.32, 0.68], loc="center", cellLoc="left")
        t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.6)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor(GRIDC)
            if c == 0: cell.set_text_props(fontweight="bold")
        fig.text(0.5, 0.31, "KEY RESULT", ha="center", fontsize=11, fontweight="bold", color=INK)
        fig.text(0.5, 0.19,
                 "A frozen, non-negative biochemical reference space built ONLY from pure Raman analytes.\n"
                 "The same analyte measured at different laser excitations lands in nearly the same place\n"
                 "(BSV cosine 0.918 vs 0.233 for different analytes), and unseen analytes project sensibly\n"
                 "(BSV margin +0.62). Biological serum Raman projects onto cholesterol/albumin-like\n"
                 "coordinates, and tube blanks show markedly less protein and fatty-acid evidence.",
                 ha="center", fontsize=9.4, color=INK)
        fig.text(0.5, 0.06, "Unsupervised · no disease labels · no classification · nothing pushed",
                 ha="center", fontsize=8, color=MUTED)
        pdf.savefig(fig); plt.close(fig)

        text_page(pdf, "Executive summary", [
            ("h", "What was built"),
            ("b", f"A Raman-only biochemical foundation model: {card['n_spectra']} pure-analyte Raman "
                  f"spectra covering {card['n_analytes']} analytes from three reference sources, "
                  "decomposed into a frozen non-negative biochemical manifold, from which emergent "
                  "biochemical axes, a Biochemical State Vector and Molecular Spectral Signatures are "
                  "derived. Ag-SERS, Au-SERS and DART are excluded and remain future observation domains."),
            ("h", "C1 — representation selected by benchmark, not by default"),
            ("b", f"Five families were compared on reconstruction, stability, interpretability, replicate "
                  f"robustness, neighbourhood preservation and nuisance leakage, with reconstruction "
                  f"deliberately weighted only 10%. NMF (k=24) and ICA (k=32) tied within 0.001; the tie "
                  f"was broken on the pre-stated constraint that a biochemical coordinate must be "
                  f"non-negative and additive. PCA ranked 3rd; the autoencoder ranked 4th with component "
                  f"stability of only 0.16, echoing the Stage-B encoder-collapse finding; sparse "
                  f"dictionary learning ranked last."),
            ("h", "C2 — the frozen manifold"),
            ("b", f"Explained variance {st['explained_variance']:.3f}; intrinsic dimensionality about "
                  f"{st['intrinsic_dimensionality']['participation_ratio']:.0f} effective components "
                  f"(90% of latent variance by {st['intrinsic_dimensionality']['n_components_90pct_latent_var']}); "
                  f"bootstrap component stability {st['bootstrap_component_stability']['mean']:.3f}. The "
                  f"manifold is frozen and every downstream projection uses it unchanged."),
            ("h", "C3 — emergent axes (tentative)"),
            ("b", f"{len(axes_list)} axes emerged by grouping components on the chemistry of the analytes "
                  "that load on them, with literature peak assignments used only to corroborate. Coherent "
                  "axes include triglycerides, saccharides, proteins, amino acids, pyrimidine "
                  "bases, fatty acids, purines, polysaccharides and cofactors. Two axes remain "
                  "low-confidence or unassigned and are reported as such. Themes are post-hoc "
                  "interpretations, never molecular assignments."),
            ("h", "C6 — external validation without retraining"),
            ("b", f"Excitation transfer: the same analyte measured at different lasers gives BSV cosine "
                  f"{v['excitation_transfer']['mean_cross_excitation_cos']:.3f} against a "
                  f"{v['excitation_transfer']['null_different_analyte_cos']:.3f} different-analyte null "
                  f"(n={v['excitation_transfer']['n_analytes']}). Source transfer "
                  f"{v['source_transfer']['mean_cross_source_cos']:.3f} (n={v['source_transfer']['n_analytes']}). "
                  f"Held-out analytes retain neighbourhood {h['neighbourhood_preservation']:.3f} and a BSV "
                  f"margin of {h['bsv_margin']:+.3f}. The weakest excitation transfers are ferritin, "
                  f"haemoglobin and albumin — heme proteins, where resonance genuinely changes the spectrum."),
            ("h", "C7 — biological projection"),
            ("b", "477 serum Raman spectra were projected into the frozen manifold with no retraining and "
                  "no labels. Their nearest pure references are cholesterol, albumin and cholesteryl "
                  "esters — the expected dominant serum biochemistry. Tube blanks separate on chemistry "
                  "rather than by fiat: protein evidence 0.10 versus 0.16–0.17 for serum, and fatty-acid "
                  "evidence 0.00 versus 0.013–0.016. Disease groups are near-identical in these "
                  "coordinates, which is reported as-is; no classification was attempted or claimed."),
            ("h", "Scope"),
            ("b", "Raman only. The adenine concentration series, uricase and serum spike datasets named in "
                  "the original plan are Ag-SERS/Au-SERS in this repository and were excluded as "
                  "out-of-domain; the Raman-domain validation actually available (held-out analytes, "
                  "excitation transfer, source transfer, blank control) was used instead and is documented."),
        ], f"{card['n_spectra']} spectra · {card['n_analytes']} analytes · {sel['representation']} k={sel['k']}")

        for name, cap in [
            ("f1_c1_benchmark.png", "Figure 1 — C1 representation benchmark. Reconstruction is deliberately "
             "only 10% of the score; stability, interpretability and analyte structure dominate."),
            ("f2_manifold.png", "Figure 2 — C2 frozen manifold: component variance, bootstrap stability and "
             "intrinsic dimensionality."),
            ("f3_axes.png", "Figure 3 — C3 emergent biochemical axes and their non-negative spectral loadings."),
            ("f4_bsv.png", "Figure 4 — Analyte map in the biochemical manifold (coloured by chemical family) "
             "and the BSV axis-share heatmap."),
            ("f5_validation.png", "Figure 5 — C6 external validation: excitation transfer, source transfer "
             "and held-out analyte projection, each against a different-analyte null."),
            ("f6_serum.png", "Figure 6 — C7 biological serum Raman projected into the frozen manifold, with "
             "out-of-distribution scores. Tube blanks show less protein and fatty-acid evidence than serum."),
        ]:
            fig = plt.figure(figsize=PAGE)
            img = plt.imread(FIG / name)
            axi = fig.add_axes([0.05, 0.42, 0.90, 0.46]); axi.imshow(img); axi.axis("off")
            fig.text(0.07, 0.94, cap.split("—")[0].strip(), fontsize=13, fontweight="bold", color=INK)
            y = 0.36
            for ln in _wrap(cap, 100):
                fig.text(0.07, y, ln, fontsize=8.8, color=INK); y -= 0.017
            pdf.savefig(fig); plt.close(fig)

        blocks = [("h", "Emergent biochemical axes (tentative)")]
        for a_ in axes_list:
            blocks.append(("m", f"axis {a_['axis']:2d} {a_['tentative_theme'][:16]:16s} "
                                f"n_comp={a_['n_components']:2d} share={a_['variance_share']:.3f} "
                                f"purity={a_['mean_component_purity']:.2f} conf={a_['theme_confidence']:12s} "
                                f"lit={'yes' if a_['literature_supports_theme'] else 'no '}"))
            blocks.append(("m", f"        top: {', '.join(a_['top_analytes'][:6])}"))
            blocks.append(("m", f"        bands: {', '.join(f'{b:.0f}' for b in a_['dominant_bands_cm'][:10])} cm-1"))
        blocks += [("h", "Next authorized action"),
                   ("b", "Stop here. The Raman-only foundation model is built, frozen and validated on the "
                         "available Raman-domain evidence. Ag-SERS / Au-SERS / DART integration, ontology "
                         "refinement beyond the Raman corpus, biological classification and clinical "
                         "prediction are explicitly out of scope and not started.")]
        text_page(pdf, "Axis catalogue and scope", blocks)

        d = pdf.infodict()
        d["Title"] = "GAIRA V5 — Raman Biochemical Foundation Model"
    print(f"figures + PDF written: {PDF_PATH} ({PDF_PATH.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
