#!/usr/bin/env python3
"""GAIRA V7 — Phase 04.5: assemble the figure report PDF.

Every phase from 02.5 onward ships one PDF carrying all of its figures with captions, because
that is the artefact that actually gets circulated and read away from the repository. Figures
are embedded from the committed PNGs, so the PDF cannot drift from what the run produced.

    python results/v7_rebuild/phase04_5/code/make_pdf.py
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
    "fig01": "Where Meta Components sit in the architecture, and the verdict. Everything left "
             "of the red box is frozen and was neither refitted nor recomputed.",
    "fig02": "The hierarchical NMF workflow. A is spectra x frozen CSM activations — not "
             "spectra, not a similarity matrix, not a graph. The geometry prior is a one-sided "
             "smoothness reward that cannot push distant CSMs apart.",
    "fig03": "Model selection over eight K and two variants. Stability carries 0.40 of the "
             "Pareto composite and reconstruction 0.14, as the brief requires — which is what "
             "pulls K down to 3, and K = 3 is where the information has gone.",
    "fig04": "Meta Component loadings over the 49 frozen CSMs. Amber ticks mark Phase 02.5 "
             "bridge CSMs; MC-01 loads almost entirely on them.",
    "fig05": "Component composition. MC-02 is the only component with a clean non-bridge "
             "reading — acylglycerol, fatty acid and phospholipid CSMs co-activating, the "
             "lipid programme every previous phase has also found.",
    "fig06": "Programme usage per spectrum, and chemistry class by dominant programme. MC-03 "
             "dominates 233 of 375 spectra: that is a background, not a programme.",
    "fig07": "Four representations on identical frozen splits. Meta Components retain 0.185 of "
             "the CSM layer's information and 0.458 of its class retrieval, well below the "
             "0.50 informativeness floor.",
    "fig08": "Molecule-retrieval degradation under twelve physically-motivated perturbations, "
             "each normalised by that representation's own clean performance. Meta Components "
             "collapse fastest on molecule identity under almost every corruption.",
    "fig09": "Mean area under the robustness curve. Meta Components win on activation "
             "stability and class retrieval and lose catastrophically on molecule identity — "
             "the profile of a low-information representation, not of a better one.",
    "fig10": "Would a different K have saved it? No. The best achievable class retrieval over "
             "all sixteen (variant, K) combinations is 0.677 against the CSM layer's 0.856. "
             "Reported as a diagnostic — selecting K on this metric would be circular.",
    "fig11": "Bootstrap component recovery is high at every K and therefore says very little; "
             "component redundancy rises with K.",
    "fig12": "Bridge CSM occupancy per programme against the base rate. MC-01 sits far above "
             "it — the one component with a clean single-class signature is built from the "
             "CSMs the geometry already flagged as ambiguous.",
    "fig13": "Meta Component occupancy over the frozen Phase 02.5 CSM geometry.",
    "fig14": "The whole result in one plot. Meta Components buy robustness that a "
             "low-information representation gets for free, and pay for it with more than half "
             "the clean accuracy.",
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
    ax.text(0.08, 0.86, "GAIRA V7 — Phase 04.5", fontsize=27, color=INK, weight="bold")
    ax.text(0.08, 0.805, "Hierarchical NMF over Frozen CSM Activations", fontsize=17,
            color=INK)
    ax.plot([0.08, 0.92], [0.775, 0.775], color=RULE, lw=1.2)
    ax.text(0.08, 0.735, "Figure report — all 14 figures with captions", fontsize=11,
            color=MUTED)
    cm = state["comparison"]
    rows = [
        ("Status", state["status"]),
        ("VERDICT", state["recommended_action"].upper()),
        ("Selected", f"{state['selected_variant']}, K = {state['K']}"),
        ("Meta fingerprint", state["meta_fingerprint"]),
        ("Frozen inputs", "atlas / LSM / CSM / theme — all verified, none refitted"),
        ("CSM class top-1", f"{cm['CSM']['B_top1']:.3f}"),
        ("Meta class top-1", f"{cm['META']['B_top1']:.3f}"),
        ("CSM macro F1", f"{cm['CSM']['B_macro_f1']:.3f}"),
        ("Meta macro F1", f"{cm['META']['B_macro_f1']:.3f}"),
        ("Information retained vs CSM", f"{state['information_retained_ratio']:.3f} "
                                        f"(floor 0.50)"),
        ("Class retrieval ratio", f"{state['class_retrieval_ratio']:.3f} (floor 0.50)"),
        ("Informativeness floor", "PASS" if state["informativeness_floor_passed"] else "FAIL"),
        ("Axes improved", f"{state['n_axes_improved']} of 8 — all stability, none informative"),
        ("Robustness delta vs CSM", f"{state['robustness_delta_vs_csm']:+.4f} AURC"),
        ("Clean accuracy cost", f"{-state['clean_accuracy_cost']:+.4f} top-1"),
        ("Completed", state["completed_utc"][:19].replace("T", " ") + " UTC"),
    ]
    y = 0.66
    for k, v in rows:
        ax.text(0.08, y, k, fontsize=9.5, color=MUTED)
        ax.text(0.40, y, str(v), fontsize=9.5, color=INK, family="DejaVu Sans")
        y -= 0.035
    ax.text(0.08, 0.10,
            "Committed sources of record: reports/PHASE_04_5_REPORT.md and "
            "reports/PHASE_04_5_SCIENTIFIC_AUDIT.md\n"
            "This is a NEGATIVE result. The CSM layer remains the canonical inference "
            "representation.\n"
            "Figures are embedded from the committed PNGs, so this PDF cannot drift from the "
            "run that produced them.",
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
    priors = json.loads((A / "verdict_v1.json").read_text())
    figs = sorted(F.glob("fig*.png"))
    if not figs:
        print("no figures found — run make_figures.py first")
        return 1
    R.mkdir(parents=True, exist_ok=True)
    out = R / "PHASE_04_5_RESULTS.pdf"

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
        d["Title"] = "GAIRA V7 Phase 04.5 — Hierarchical NMF over CSM Activations (figures)"
        d["Subject"] = "14 figures; negative result — Meta Components discarded"
        d["Keywords"] = "GAIRA V7 Raman hierarchical NMF meta components"

    print(f"  {out.relative_to(REPO)}  ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(figs) + 2} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
