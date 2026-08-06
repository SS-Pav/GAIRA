#!/usr/bin/env python3
"""GAIRA V7 — Phase 04: assemble the figure report PDF.

Every phase from 02.5 onward ships one PDF carrying all of its figures with captions, because
that is the artefact that actually gets circulated and read away from the repository. Figures
are embedded from the committed PNGs, so the PDF cannot drift from what the run produced.

    python results/v7_rebuild/phase04/code/make_pdf.py
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
    "fig01": "The canonical inference pathway. Projection only — no fitting, no random number, "
             "batch-independent, bit-identical on re-run. Every output traces back to named "
             "canonical molecules.",
    "fig02": "Projection estimators. Elastic net selected on replicate consistency x noise "
             "stability among estimators with zero negative mass, not on reconstruction alone. "
             "Ridge and ARD reconstruct at 0.24 and 0.35 — an unconstrained fit on a coherent "
             "dictionary is not usable.",
    "fig03": "Aggregation, theme mode and BSV definition. The softmax theme mode scored BEST "
             "on replicate consistency (0.993) and was rejected: it activates themes for which "
             "the spectrum has no CSM evidence at all. Adding uncertainty channels INTO the "
             "BSV lowers molecule separation, so they stay beside it.",
    "fig04": "Out-of-sample manifold extension. Nystrom is the principled choice for a "
             "diffusion map and the worst here — 49 reference coordinates are too few and too "
             "spread for a kernel average to localise.",
    "fig05": "THE CENTRAL RESULT. The LSM/CSM layers raise chemistry-class retrieval on UNSEEN "
             "molecules from 0.608 to 0.855 — transfer, not recall. The theme layer trades "
             "both retrieval axes for the highest replicate consistency in the stack. "
             "Effective rank falls far below nominal dimension at every level.",
    "fig06": "Retrieval by level under both splits. Split A holds out one spectrum with its "
             "molecule still represented; split B holds out the whole molecule, where "
             "molecule top-k is undefined by construction and is not shown.",
    "fig07": "Dictionary-level leakage, measured rather than assumed. The frozen dictionary was "
             "fitted on every molecule, so grouping the folds cannot remove it; refitting per "
             "fold quantifies it at +0.055 top-1.",
    "fig08": "THE FAILING GATE. On real Ag-SERS the OOD score is at chance, and SERS spectra "
             "are BETTER explained than the references. A non-negative dictionary of Raman "
             "motifs reconstructs SERS of the same metabolites comfortably, so the atlas "
             "cannot tell modality. The synthetic probe reaching 0.946 is precisely why the "
             "real probe was necessary.",
    "fig09": "The Biochemical State Vector: absolute, non-negative, four axes, effective rank "
             "2.40. Replicates cohere at cosine 0.979 with a between/within separation ratio "
             "of 7.26, and the vector survives noise at sigma = 0.05.",
    "fig10": "Per chemistry class, unseen molecules, at the CSM level.",
    "fig11": "All 375 reference spectra projected into the frozen manifold, coloured by "
             "chemistry, by engine confidence and by OOD score. Neighbourhood purity is 4.06x "
             "chance.",
    "fig12": "Confidence is monotone in accuracy but badly overconfident (ECE 0.486) — a "
             "reported failure. Activation recovery against each molecule's own reference "
             "profile rises with abstraction.",
    "fig13": "A worked example end to end, with the explanation chain resolved from the frozen "
             "registries at call time so it can never disagree with the atlas.",
    "fig14": "GAIRA V7 architecture after Phase 04, with the open failures stated on the "
             "diagram rather than in a footnote.",
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
    ax.text(0.08, 0.86, "GAIRA V7 — Phase 04", fontsize=27, color=INK, weight="bold")
    ax.text(0.08, 0.805, "Frozen Projection Engine and Hierarchical Inference", fontsize=17,
            color=INK)
    ax.plot([0.08, 0.92], [0.775, 0.775], color=RULE, lw=1.2)
    ax.text(0.08, 0.735, "Figure report — all 14 figures with captions", fontsize=11,
            color=MUTED)
    r = state["retrieval"]["L3_csm"]
    rows = [
        ("Status", state["status"] + " — 10 of 11 gates; OOD on real SERS fails"),
        ("Engine version", state["engine_version"]),
        ("Frozen atlas / LSM / CSM / theme", "all four verified before any computation"),
        ("Projection", state["engine_config"]["projection_method"]),
        ("LSM to CSM", state["engine_config"]["aggregation_method"]),
        ("Theme mode", state["engine_config"]["theme_mode"]),
        ("BSV", f"{state['engine_config']['bsv_variant']}, dimension "
                f"{state['bsv_dimension']}"),
        ("BSV effective rank",
         f"{state['bsv_effective_rank']['participation_ratio']:.2f} of "
         f"{state['bsv_effective_rank']['nominal_K']}"),
        ("Geometry extension", state["engine_config"]["geometry_extension"]),
        ("Held-out molecule top-1 (CSM)", f"{r['A_molecule_top1']:.3f}"),
        ("Held-out class top-1, unseen molecule", f"{r['B_class_top1']:.3f} "
                                                  f"(raw spectrum 0.608)"),
        ("Dictionary leakage", f"+{state['leakage_inflation_top1']:.3f} top-1"),
        ("OOD AUROC (real Ag-SERS)", f"{state['ood_auroc']:.3f} — FAILS"),
        ("Confidence calibration", f"ECE {state['calibration_ece']:.3f} — poor"),
        ("Completed", state["completed_utc"][:19].replace("T", " ") + " UTC"),
    ]
    y = 0.66
    for k, v in rows:
        ax.text(0.08, y, k, fontsize=9.5, color=MUTED)
        ax.text(0.40, y, str(v), fontsize=9.5, color=INK, family="DejaVu Sans")
        y -= 0.035
    ax.text(0.08, 0.10,
            "Committed sources of record: reports/PHASE_04_REPORT.md and "
            "reports/PHASE_04_SCIENTIFIC_AUDIT.md\n"
            "Projection only: nothing in Phases 00-03 was refitted, recomputed or replaced.\n"
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
    priors = json.loads((A / "engine_config_v1.json").read_text())
    figs = sorted(F.glob("fig*.png"))
    if not figs:
        print("no figures found — run make_figures.py first")
        return 1
    R.mkdir(parents=True, exist_ok=True)
    out = R / "PHASE_04_FIGURES.pdf"

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
        d["Title"] = "GAIRA V7 Phase 04 — Frozen Projection Engine (figures)"
        d["Subject"] = "14 figures with captions; projection only, nothing refitted"
        d["Keywords"] = "GAIRA V7 Raman inference engine BSV"

    print(f"  {out.relative_to(REPO)}  ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(figs) + 2} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
