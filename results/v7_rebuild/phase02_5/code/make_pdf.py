#!/usr/bin/env python3
"""GAIRA V7 — Phase 02.5: assemble the figure PDF.

Every phase from 02.5 onward ships one PDF carrying all of its figures with captions, because
that is the artefact that actually gets circulated and read away from the repository. Figures
are embedded from the committed PNGs, so the PDF cannot drift from what the run produced.

    python results/v7_rebuild/phase02_5/code/make_pdf.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
PH = HERE.parent
REPO = PH.parents[2]
F, A, R = PH / "figures", PH / "artifacts", PH / "reports"
INK, MUTED, RULE = "#1a1a1a", "#6b7280", "#d1d5db"
PAGE = (11.0, 8.5)            # US Letter landscape — figures here are all wide

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

CAPTIONS = {
    "fig01": "Phase 02.5 pipeline. Chemistry and source labels are excluded from every "
             "representation and every distance, and revealed only at the neighbourhood step.",
    "fig02": "Metric benchmark. Probes are scale-free — each divided by that metric's own "
             "median observed distance — because a background separation of 0.106 under "
             "Euclidean and 0.006 under cosine cannot be compared until both are read against "
             "the scale each metric works on. Wasserstein selected.",
    "fig03": "Observed pairwise distances are unimodal with no density valley (depth 0.003), "
             "and local intrinsic dimension averages 3.86 against an ambient 676. Both point "
             "to a continuum rather than discrete islands.",
    "fig04": "PCA scores. Only PC1 has a reproducible loading (stability 0.82); PC2 falls to "
             "0.52 and PC3 to 0.32. Chemistry colour was attached after fitting.",
    "fig05": "PCA loadings. PC1 is CH₂/CH₃ scissoring at 1442 cm⁻¹ — an aliphatic-content "
             "axis, and the only component worth interpreting.",
    "fig06": "UMAP stability sweep over n_neighbors × min_dist × seed. Neighbourhoods survive "
             "the parameters (Jaccard 0.60–0.65 against the high-dimensional geometry); the "
             "global arrangement does not (Procrustes disparity 0.45).",
    "fig07": "Diffusion map and its eigenvalue spectrum. The absence of a sharp gap "
             "(1.000, 0.186, 0.150, 0.119) is the first quantitative evidence against discrete "
             "clusters.",
    "fig08": "Hierarchical clustering. Leaf colour is chemistry class, attached after the tree "
             "was built.",
    "fig09": "Spectral cosine similarity, ordered by the primary geometry's dendrogram. Cosine "
             "is the worst metric on background separation (0.008) — it is dominated by the "
             "shared broad envelope.",
    "fig10": "The primary multi-view geometry (weighted similarity fusion).",
    "fig11": "k-nearest-neighbour graph (k = 5). Modularity 0.436 against a degree-preserving "
             "null of 0.070 ± 0.003. Labelled: bridges in red, isolates in grey.",
    "fig12": "Force-directed layout of the same graph (Kamada–Kawai).",
    "fig13": "Minimum spanning tree — the backbone of motif space.",
    "fig14": "Cluster quality across K and linkage. Silhouette (0.551) and Calinski–Harabasz "
             "agree on K = 2 under four of five linkages, with bootstrap ARI 0.879; "
             "Davies–Bouldin runs monotonically to the largest K, its behaviour when no "
             "further structure exists. One defensible cut, at the top.",
    "fig15": "Nearest-neighbour cards. 188 of 250 links (75%) are indistinguishable from the "
             "null — the same lesson Phase 02 learned at the edge level, one level up.",
    "fig16": "Geometry coloured by chemistry class. PERMANOVA R² = 0.617, p = 0.001.",
    "fig17": "Geometry coloured by source. R² = 0.130, p = 0.022 — significant but 4.7× weaker "
             "than chemistry. A caution, not a veto. 8 of 50 motifs are single-source and "
             "therefore untestable.",
    "fig18": "Geometry coloured by excitation. R² = 0.178, p = 0.044.",
    "fig19": "Lipid neighbourhood (Phase 02 proposal00). Separation ratio 2.05, local dimension "
             "2.82 — the most low-dimensional group in the corpus. Ordered along diffusion "
             "coordinate 1 it runs amino acid → sterol → free fatty acids → acylglycerols: an "
             "aliphatic chain-order gradient, with the sterol adjacent but outside.",
    "fig20": "Polar skeletal neighbourhood (proposal03). Separation ratio 2.31, the strongest "
             "of the three. Shared bands at 1074/1120 cm⁻¹ confirm that protein and "
             "polysaccharide are close through skeletal C–O/C–C overlap, not glycoprotein "
             "biology.",
    "fig21": "Heterocyclic ring-system neighbourhood (proposal16). Internal stability 1.000. "
             "The cofactor motifs sit near purine because coenzyme A and acetyl-CoA contain "
             "adenine — which is why merging the group destroyed reconstruction.",
    "fig22": "Bridge motifs: high betweenness, low local clustering. Seven motifs sit on the "
             "paths between neighbourhoods without belonging to one. A hard theme assignment "
             "would misplace each of them.",
    "fig23": "Isolated motifs. Five have no close neighbour; chromophore_pigment.m00 is the "
             "clearest case — a conjugated carotenoid C=C system with no analogue in this "
             "corpus. Candidates for singleton themes or for exclusion.",
    "fig24": "The ten provisional Phase 03 priors on the primary geometry. These constrain "
             "Phase 03; they do not decide it.",
    "fig25": "GAIRA V7 architecture after Phase 02.5.",
}


def _page(pdf, draw):
    fig = plt.figure(figsize=PAGE)
    draw(fig)
    pdf.savefig(fig)
    plt.close(fig)


def cover(fig, state, priors):
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    # limits pinned: a plot() call would otherwise autoscale the axes and move every text
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.08, 0.86, "GAIRA V7 — Phase 02.5", fontsize=27, color=INK, weight="bold")
    ax.text(0.08, 0.805, "Latent Geometry of Spectral Motif Space", fontsize=17, color=INK)
    ax.plot([0.08, 0.92], [0.775, 0.775], color=RULE, lw=1.2)
    ax.text(0.08, 0.735, "Figure report — all 25 figures with captions", fontsize=11,
            color=MUTED)
    rows = [
        ("Status", "COMPLETE — analysis only; nothing refitted, no themes created"),
        ("Frozen atlas", state["atlas_fingerprint"]),
        ("Frozen LSM registry", state["lsm_registry_fingerprint"]),
        ("Frozen CSM dictionary", state["csm_dictionary_fingerprint"]),
        ("Primary spectral metric", state["primary_spectral_metric"]),
        ("Primary fused geometry", state["primary_geometry"]),
        ("Neighbourhoods computed on", state["neighbourhoods_computed_on"]),
        ("Metric vs fused kNN agreement", f"{state['metric_vs_fused_knn_agreement']:.3f}"),
        ("Geometry verdict", state["geometry_verdict"] + " — overlapping continua"),
        ("kNN chemical coherence", f"{state['knn_coherence']:.3f} "
                                   f"(p = {state['knn_coherence_p']:.4f})"),
        ("Graph modularity z", f"{state['modularity_z']:.1f} vs degree-preserving null"),
        ("Provisional Phase 03 priors", str(state["n_priors"])),
        ("Source-untestable motifs", f"{state['n_single_source_motifs']} of 50"),
        ("Completed", state["completed_utc"][:19].replace("T", " ") + " UTC"),
    ]
    y = 0.66
    for k, v in rows:
        ax.text(0.08, y, k, fontsize=9.5, color=MUTED)
        ax.text(0.40, y, str(v), fontsize=9.5, color=INK, family="DejaVu Sans")
        y -= 0.035
    ax.text(0.08, 0.10, "Committed source of record: reports/PHASE_02_5_LATENT_GEOMETRY_REPORT.md\n"
                        "Figures are embedded from the committed PNGs, so this PDF cannot "
                        "drift from the run that produced them.",
            fontsize=8.5, color=MUTED)


def contents(fig):
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.06, 0.93, "Figures", fontsize=19, color=INK, weight="bold")
    ax.plot([0.06, 0.94], [0.905, 0.905], color=RULE, lw=1.0)
    names = sorted(F.glob("fig*.png"))
    left, right = names[:13], names[13:]
    for col, group in ((0.06, left), (0.52, right)):
        y = 0.86
        for p in group:
            num = p.stem.split("_")[0]
            title = " ".join(p.stem.split("_")[1:]).replace("-", " ")
            ax.text(col, y, num.replace("fig", ""), fontsize=9, color=MUTED)
            ax.text(col + 0.035, y, title, fontsize=9, color=INK)
            y -= 0.033


def main() -> int:
    state = json.loads((PH / "PHASE_STATE.json").read_text())
    priors = json.loads((A / "phase03_geometry_priors.json").read_text())
    figs = sorted(F.glob("fig*.png"))
    if not figs:
        print("no figures found — run make_figures.py first")
        return 1
    R.mkdir(parents=True, exist_ok=True)
    out = R / "PHASE_02_5_FIGURES.pdf"

    with PdfPages(out) as pdf:
        _page(pdf, lambda f: cover(f, state, priors))
        _page(pdf, contents)
        for p in figs:
            img = mpimg.imread(p)
            h, w = img.shape[:2]

            def draw(fig, img=img, p=p, w=w, h=h):
                fig.patch.set_facecolor("white")
                cap = CAPTIONS.get(p.stem.split("_")[0], "")
                # Fit the image into the box above the caption, preserving aspect. Work in
                # INCHES, not in a guessed dpi: the PNGs are 200 dpi, and dividing pixels by
                # 100 shrank every figure to a thumbnail.
                box_w_in, box_h_in = 0.88 * PAGE[0], 0.76 * PAGE[1]
                aspect = h / w
                if box_w_in * aspect <= box_h_in:
                    draw_w_in, draw_h_in = box_w_in, box_w_in * aspect
                else:
                    draw_h_in, draw_w_in = box_h_in, box_h_in / aspect
                fw, fh = draw_w_in / PAGE[0], draw_h_in / PAGE[1]
                # centred in the band between the header and the caption rule, rather than
                # pinned to the top — wide figures otherwise leave half the page blank
                top, bottom = 0.92, 0.16
                ax = fig.add_axes([(1 - fw) / 2, bottom + (top - bottom - fh) / 2, fw, fh])
                ax.imshow(img); ax.axis("off")
                ft = fig.add_axes([0, 0, 1, 1]); ft.axis("off")
                ft.set_xlim(0, 1); ft.set_ylim(0, 1)
                ft.text(0.06, 0.965, p.stem.split("_")[0].replace("fig", "Figure "),
                        fontsize=11, color=INK, weight="bold")
                ft.plot([0.06, 0.94], [0.115, 0.115], color=RULE, lw=0.8)
                ft.text(0.06, 0.085, cap, fontsize=8.8, color=INK, wrap=True, va="top")
            _page(pdf, draw)

        d = pdf.infodict()
        d["Title"] = "GAIRA V7 Phase 02.5 — Latent Geometry of Spectral Motif Space (figures)"
        d["Subject"] = "25 figures with captions; analysis-only phase, no themes created"
        d["Keywords"] = "GAIRA V7 Raman motif geometry"

    print(f"  {out.relative_to(REPO)}  ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(figs) + 2} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
