#!/usr/bin/env python3
"""GAIRA V7 — Phase 06.5: assemble the audit report PDF.

Figures are embedded from the committed PNGs so the PDF cannot drift from the figure script.

    python results/v7_rebuild/phase06_5/code/make_pdf.py
"""
from __future__ import annotations

import json
import sys
import textwrap
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
PAGE = (11.69, 8.27)

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

CAPTIONS = {
    "F01": "Cluster stability across 14 values of K and four fixed-K algorithms. NO index has an "
           "interior optimum: silhouette rises monotonically to K=30 (Spearman +1.00), "
           "neighbour preservation falls monotonically (-1.00), membership entropy rises "
           "monotonically (+1.00). Bootstrap stability is U-shaped because coarse partitions and "
           "near-singleton partitions are both trivially reproducible. This is what a continuum "
           "looks like.",
    "F02": "Is K=16 genuinely stable or merely convenient? Merely convenient: 2 of 4 algorithms "
           "call it stable and its mean bootstrap ARI (0.673) is unremarkable against its "
           "neighbours. The free-K algorithms choose 3-24, never 16. K=16 is adopted as a "
           "reporting convention for comparability with the ontology, not as a discovery.",
    "F03": "Composition of every emergent cluster, and the rule-based classification whose "
           "thresholds were declared before any cluster was inspected. Five clusters are 100% "
           "chemically pure without a label having been used; four are acquisition-confounded, "
           "their source or excitation purity exceeding their chemistry purity.",
    "F04": "Mean spectrum +/- 1 sd of the six largest emergent clusters, with each cluster's "
           "dominant CSM bands marked. The clusters are spectroscopically coherent objects, not "
           "arbitrary partitions of a cloud.",
    "F05": "The CSM activation map and the molecule x molecule distance matrix, both ordered by "
           "the emergent partition. Block structure is visible and soft-edged.",
    "F06": "PCA, UMAP and MDS embeddings coloured by curated chemistry (top) and by emergent "
           "cluster (bottom). Visualisation only: PCA components 1-2 explain 15.5% of variance "
           "and UMAP distances must not be read quantitatively.",
    "F07": "Force-directed 5-NN graph of the 154 molecules. Modularity 0.718 against a "
           "degree-preserving null of 0.347 (z = 40) - the graph is genuinely modular - while "
           "35% of molecules have neighbours in more than one community.",
    "F08": "Is the geometry chemistry or the instrument? Chemistry explains 3.8x more of the "
           "distance structure than excitation and 11x more than source library. The caveat: "
           "globally chemistry wins, but four individual clusters are acquisition-confounded.",
    "F09": "Dendrogram, distance distribution, split gains and the shape verdict. The space is "
           "modular (z = 40) and tree-like (cophenetic 0.870) while having no preferred cut "
           "height - a hierarchy with continuous branch lengths. The two intrinsic-dimension "
           "estimators disagree by 3.3x, so neither is quoted.",
    "F10": "Continuous Spectral Coordinates: kernel x temperature sweep selected on label-free "
           "neighbour preservation, and the comparison against hard cluster ids. The "
           "coordinates carry nearly double the neighbourhood information of a cluster id.",
    "F11": "Coordinate robustness under six Raman perturbations at five levels each. Mean "
           "coordinate cosine 0.958, argmax stability 0.949.",
    "F12": "The retrieval benchmark, molecule-grouped with the clustering refitted inside every "
           "training fold. Adding the coordinate prior changes molecule top-1 by +0.016 with a "
           "confidence interval crossing zero (p = 0.180) and chemistry top-1 by +0.003 "
           "(p = 1.000). NEITHER IS SIGNIFICANT. This is the finding the recommendation rests on.",
    "F13": "Emergent geometry versus the curated ontology. Completeness (0.812) exceeds "
           "homogeneity (0.732): curated classes stay together while emergent clusters merge "
           "them. Neither partition is wrong - they answer different questions.",
    "F14": "Decision gate and architecture recommendation. Five of seven Section 9 criteria "
           "pass; the two that fail are the two that would justify an architecture change. "
           "Option A is retained and the coordinates remain a scientific instrument.",
}


def _page(pdf, draw, size=PAGE):
    fig = plt.figure(figsize=size)
    draw(fig)
    pdf.savefig(fig)
    plt.close(fig)


def cover(fig, state, s):
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.06, 0.89, "GAIRA V7 — Phase 06.5", fontsize=26, color=INK, weight="bold")
    ax.text(0.06, 0.836, "Latent Spectral Geometry Audit", fontsize=17, color=INK)
    ax.plot([0.06, 0.94], [0.806, 0.806], color=RULE, lw=1.2)
    ax.text(0.06, 0.772, "What biochemical organisation emerges from the frozen CSM "
            "representation, independent of the curated ontology?", fontsize=10, color=MUTED)
    ks, hy, co, ag = (s["k_selection"], s["hierarchy"], s["coordinates"], s["agreement"])
    r2 = {r["factor"]: r["marginal_R2"] for r in s["confounding"]["variance_partition"]}
    sg = s["retrieval_significance"]
    rows = [
        ("Status", f"{state['status']} — AUDIT ONLY, no architecture changed"),
        ("Recommendation", f"{s['recommendation']['option']} — current architecture retained"),
        ("Unit of analysis", "canonical molecule (154), not spectrum"),
        ("Labels in construction", "NONE — chemistry is an external validation target only"),
        ("", ""),
        ("Preferred cluster count", f"NONE — {ks['n_indices_with_interior_optimum']} of 7 "
                                    f"indices have an interior optimum"),
        ("Chemistry vs acquisition", f"PERMANOVA R² {r2['fine_chemistry']:.3f} vs excitation "
                                     f"{r2['excitation']:.3f}, source {r2['source']:.3f}"),
        ("Graph modularity", f"{hy['modularity']['modularity']:.3f} vs null "
                             f"{hy['modularity']['null_mean']:.3f} (z = "
                             f"{hy['modularity']['z_score']:.0f})"),
        ("Shape of the space", f"{hy['shape']} — hierarchy with continuous branch lengths"),
        ("Coordinates vs hard ids", f"k-NN preservation "
                                    f"{co['neighbour_preservation_k10']:.3f} vs "
                                    f"{co['neighbour_preservation_hard_ids']:.3f}"),
        ("Coordinate reproducibility", f"{co['reproducibility']:.3f}"),
        ("Coordinate robustness", f"mean cosine "
                                  f"{s['coordinate_robustness_mean_cosine']:.3f}"),
        ("Retrieval gain from coordinates", f"molecule Δ{sg['molecule']['delta']:+.3f} "
                                            f"(p = {sg['molecule']['p_value']:.3f}) · chemistry "
                                            f"Δ{sg['chemistry']['delta']:+.3f} "
                                            f"(p = {sg['chemistry']['p_value']:.3f}) — "
                                            f"NEITHER SIGNIFICANT"),
        ("Agreement with the ontology", f"AMI {ag['AMI']:.3f} at K = 16 — but AMI is monotone "
                                        f"in K and peaks at K = 24"),
        ("", ""),
        ("Frozen inputs", "LSM 208482d6… · CSM 0b4aa550… · engine 20d8bd99… — verified"),
        ("Phase 07", "NOT BEGUN"),
        ("Completed", state["finished"][:19].replace("T", " ") + " UTC"),
    ]
    y = 0.722
    for k, v in rows:
        if k:
            ax.text(0.06, y, k, fontsize=9.0, color=MUTED)
            ax.text(0.36, y, v, fontsize=9.0, color=INK)
        y -= 0.0315
    ax.text(0.06, 0.070,
            "Sources of record: reports/PHASE_06_5_LATENT_GEOMETRY_AUDIT.md and "
            "reports/PHASE_06_5_SCIENTIFIC_AUDIT.md\n"
            "The Continuous Spectral Coordinates are retained as a SCIENTIFIC INSTRUMENT. They "
            "are not a chemistry probability, not a concentration,\nand they do not enter the "
            "GAIRA inference path.", fontsize=8.2, color=MUTED)


def contents(fig):
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.06, 0.93, "Figures", fontsize=19, color=INK, weight="bold")
    ax.plot([0.06, 0.94], [0.905, 0.905], color=RULE, lw=1.0)
    names = sorted(F.glob("F*.png"))
    for col, group in ((0.06, names[:7]), (0.52, names[7:])):
        y = 0.86
        for p in group:
            num = p.stem.split("_")[0]
            title = " ".join(p.stem.split("_")[1:]).replace("_", " ")
            ax.text(col, y, num.replace("F", ""), fontsize=9, color=MUTED)
            ax.text(col + 0.035, y, title, fontsize=9, color=INK)
            y -= 0.037


def main() -> int:
    state = json.loads((PH / "PHASE_STATE.json").read_text())
    s = json.loads((A / "phase06_5_summary_v1.json").read_text())
    figs = sorted(F.glob("F*.png"))
    if not figs:
        print("no figures found — run make_figures.py first")
        return 1
    R.mkdir(parents=True, exist_ok=True)
    out = R / "PHASE_06_5_LATENT_GEOMETRY_AUDIT.pdf"
    with PdfPages(out) as pdf:
        _page(pdf, lambda f: cover(f, state, s))
        _page(pdf, contents)
        for p in figs:
            img = mpimg.imread(p)
            h, w = img.shape[:2]

            def draw(fig, img=img, p=p):
                fig.patch.set_facecolor("white")
                cap = CAPTIONS.get(p.stem.split("_")[0], "")
                box_w, box_h = 0.90 * PAGE[0], 0.74 * PAGE[1]
                aspect = h / w
                if box_w * aspect <= box_h:
                    dw, dh = box_w, box_w * aspect
                else:
                    dh, dw = box_h, box_h / aspect
                fw, fh = dw / PAGE[0], dh / PAGE[1]
                top, bottom = 0.93, 0.16
                ax = fig.add_axes([(1 - fw) / 2, bottom + (top - bottom - fh) / 2, fw, fh])
                ax.imshow(img); ax.axis("off")
                ft = fig.add_axes([0, 0, 1, 1]); ft.axis("off")
                ft.set_xlim(0, 1); ft.set_ylim(0, 1)
                ft.text(0.05, 0.965, p.stem.split("_")[0].replace("F", "Figure "),
                        fontsize=11, color=INK, weight="bold")
                ft.plot([0.05, 0.95], [0.125, 0.125], color=RULE, lw=0.8)
                yy = 0.098
                for line in textwrap.wrap(cap, 148):
                    ft.text(0.05, yy, line, fontsize=8.4, color=INK)
                    yy -= 0.021
            _page(pdf, draw)
        d = pdf.infodict()
        d["Title"] = "GAIRA V7 Phase 06.5 — Latent Spectral Geometry Audit"
        d["Subject"] = "14 figures; what geometry emerges from the frozen CSM manifold without labels"
        d["Keywords"] = "GAIRA V7 Raman latent geometry clustering continuous coordinates"
    print(f"  {out.relative_to(REPO)}  ({out.stat().st_size / 1e6:.1f} MB, {len(figs) + 2} pages)")
    print(f"  tracked by git: {'yes' if not _ignored(out) else 'NO (gitignored)'}")
    return 0


def _ignored(p: Path) -> bool:
    import subprocess
    return subprocess.run(["git", "check-ignore", "-q", str(p)], cwd=REPO).returncode == 0


if __name__ == "__main__":
    raise SystemExit(main())
