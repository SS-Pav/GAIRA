#!/usr/bin/env python3
"""GAIRA V7 — Phase 05: assemble the figure report PDF.

Figures are embedded from the committed PNGs, so the PDF cannot drift from what the run
produced.

    python results/v7_rebuild/phase05/code/make_pdf.py
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
F, A, R, T = PH / "figures", PH / "artifacts", PH / "reports", PH / "tables"
INK, MUTED, RULE = "#1a1a1a", "#6b7280", "#d1d5db"
PAGE = (11.0, 8.5)

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

CAPTIONS = {
    "F01": "The canonical pipeline. A spectrum becomes a 49-d frozen CSM activation vector and "
           "five inference branches read that one vector. Nothing is fitted at inference; "
           "geometry appears nowhere in the path.",
    "F02": "Direct NNLS projection onto the frozen dictionary. Mean explained variance 0.821 "
           "with 9.6 of 49 motifs active — the shape a mixture model should have. The minimum "
           "of 0.206 is real: those spectra are what the rejection channels exist for.",
    "F03": "Chemistry-class confusion on molecules the atlas has never seen (Split B). The "
           "failures sit where a spectroscopist would put them — small classes dominated by "
           "chains and rings they share with larger ones.",
    "F04": "Calibration benchmark. Platt scaling wins ECE (0.080) and loses Brier (0.242) "
           "because it reports the base rate for every spectrum. Selection is on Brier, which "
           "is proper and cannot be won by flattening.",
    "F05": "Reliability of the selected Dirichlet calibrator. ECE 0.130 — above the "
           "pre-declared 0.10 gate, which is reported as a failure rather than relaxed.",
    "F06": "Retrieval under both splits. Molecule top-k is undefined under Split B, not zero: "
           "the correct answer is not among the candidates.",
    "F07": "Confidence separates correct from wrong answers (AUROC 0.891) and thresholding "
           "trades coverage for accuracy monotonically — what a usable confidence must do.",
    "F08": "Open-set rejection on synthetic negatives. Two channels score below chance because "
           "in a sparse non-negative code the population centre is where degraded spectra "
           "land, not where real ones do. Signs are reported unflipped.",
    "F09": "Biochemical Evidence Profiles. Eleven declared axes, no factorisation. Spoke "
           "thickness and marker size encode confidence, so an axis resting on one motif in a "
           "poorly reconstructed spectrum looks thin.",
    "F10": "Provenance waterfall. Contributions are the actual additive terms, so the listed "
           "CSMs sum exactly to the axis value. 3,133 chains verified, none broken.",
    "F11": "Noise robustness on molecule-grouped banks. CSM projection beats the raw spectrum "
           "on both halves of the hypothesis: more discriminative on unseen molecules (0.845 "
           "vs 0.592) and slower to degrade (retention 0.935 vs 0.895).",
    "F12": "Three complete inference reports: spectrum, retrieval with calibrated confidence, "
           "chemistry class, diagnostics and evidence radar. The top case has margin 0.000 — "
           "an honest near-tie between two structurally similar lipids.",
    "F13": "Which frozen motifs fire, and for what chemistry. The block structure is the "
           "reason class inference generalises to molecules the atlas has not seen.",
    "F14": "The abstraction chain and the frozen CSM to axis map. Mean 3.8 axes per CSM; a CSM "
           "loads on an axis only when one of its diagnostic bands falls in an axis window.",
    "F15": "Phase summary. Accuracy against robustness, axis grounding, headline numbers and "
           "all sixteen gates — fifteen pass, and the one that fails is argued in section 6.",
}


def _page(pdf, draw):
    fig = plt.figure(figsize=PAGE)
    draw(fig)
    pdf.savefig(fig)
    plt.close(fig)


def cover(fig, state, s):
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.08, 0.87, "GAIRA V7 — Phase 05", fontsize=27, color=INK, weight="bold")
    ax.text(0.08, 0.815, "The Canonical CSM Inference Engine", fontsize=17, color=INK)
    ax.plot([0.08, 0.92], [0.785, 0.785], color=RULE, lw=1.2)
    ax.text(0.08, 0.748, "Figure report — all 15 figures with captions · Raman only",
            fontsize=11, color=MUTED)
    rows = [
        ("Status", f"{state['status']} — {s['gates']['n'] - s['gates']['failed']} of "
                    f"{s['gates']['n']} gates pass"),
        ("Replaces", "the Phase 04 Theme/BSV inference path"),
        ("Engine fingerprint", state["engine_fingerprint"]),
        ("Frozen inputs", "LSM + CSM verified, nothing refitted"),
        ("Representation", "49-d frozen CSM activation vector"),
        ("Similarity metric", f"{s['metric']['selected']} (nested grouped CV)"),
        ("Calibration", f"{s['split_a']['calibration']} (selected on Brier)"),
        ("", ""),
        ("Split A molecule top-1 / top-5", f"{s['split_a']['molecule_top1']:.3f} / "
                                           f"{s['split_a']['molecule_top5']:.3f}"),
        ("Split B class top-1 / top-3", f"{s['split_b']['class_top1']:.3f} / "
                                        f"{s['split_b']['class_top3']:.3f}"),
        ("Split B macro F1", f"{s['split_b']['macro_f1']:.3f}"),
        ("Calibration ECE / discrimination", f"{s['split_a']['ece']:.3f} / "
                                             f"{s['split_a']['discrimination']:.3f}"),
        ("Open-set joint AUROC", f"{s['openset']['joint_auroc']:.3f} (synthetic negatives)"),
        ("Evidence axes grounded", f"{s['evidence']['n_grounded']} of 11 "
                                   f"(±1 under the window sweep)"),
        ("Provenance chains broken", f"{s['provenance']['broken']} of "
                                     f"{s['provenance']['n_chains']}"),
        ("Robustness retention, CSM vs raw", "0.935 vs 0.895 (class, molecule-grouped)"),
        ("Completed", state["finished"][:19].replace("T", " ") + " UTC"),
    ]
    y = 0.695
    for k, v in rows:
        if k:
            ax.text(0.08, y, k, fontsize=9.5, color=MUTED)
            ax.text(0.46, y, str(v), fontsize=9.5, color=INK)
        y -= 0.033
    ax.text(0.08, 0.085,
            "Committed sources of record: reports/PHASE_05_CANONICAL_INFERENCE_ENGINE.md and "
            "reports/PHASE_05_SCIENTIFIC_AUDIT.md\n"
            "G6 (calibration ECE <= 0.10) FAILS at 0.130. It was not relaxed: on this corpus "
            "that threshold is reachable only by a\ncalibrator that reports the same confidence "
            "for every spectrum. Figures are embedded from the committed PNGs.",
            fontsize=8.5, color=MUTED)


def contents(fig):
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.06, 0.93, "Figures", fontsize=19, color=INK, weight="bold")
    ax.plot([0.06, 0.94], [0.905, 0.905], color=RULE, lw=1.0)
    names = sorted(F.glob("F*.png"))
    for col, group in ((0.06, names[:8]), (0.52, names[8:])):
        y = 0.86
        for p in group:
            num = p.stem.split("_")[0]
            title = " ".join(p.stem.split("_")[1:]).replace("_", " ")
            ax.text(col, y, num.replace("F", ""), fontsize=9, color=MUTED)
            ax.text(col + 0.035, y, title, fontsize=9, color=INK)
            y -= 0.037


def main() -> int:
    state = json.loads((PH / "PHASE_STATE.json").read_text())
    s = json.loads((A / "phase05_summary_v1.json").read_text())
    figs = sorted(F.glob("F*.png"))
    if not figs:
        print("no figures found — run make_figures.py first")
        return 1
    R.mkdir(parents=True, exist_ok=True)
    out = R / "PHASE_05_RESULTS.pdf"

    with PdfPages(out) as pdf:
        _page(pdf, lambda f: cover(f, state, s))
        _page(pdf, contents)
        for p in figs:
            img = mpimg.imread(p)
            h, w = img.shape[:2]

            def draw(fig, img=img, p=p, w=w, h=h):
                fig.patch.set_facecolor("white")
                cap = CAPTIONS.get(p.stem.split("_")[0], "")
                box_w_in, box_h_in = 0.88 * PAGE[0], 0.74 * PAGE[1]
                aspect = h / w
                if box_w_in * aspect <= box_h_in:
                    draw_w_in, draw_h_in = box_w_in, box_w_in * aspect
                else:
                    draw_h_in, draw_w_in = box_h_in, box_h_in / aspect
                fw, fh = draw_w_in / PAGE[0], draw_h_in / PAGE[1]
                top, bottom = 0.92, 0.18
                ax = fig.add_axes([(1 - fw) / 2, bottom + (top - bottom - fh) / 2, fw, fh])
                ax.imshow(img); ax.axis("off")
                ft = fig.add_axes([0, 0, 1, 1]); ft.axis("off")
                ft.set_xlim(0, 1); ft.set_ylim(0, 1)
                ft.text(0.06, 0.965, p.stem.split("_")[0].replace("F", "Figure "),
                        fontsize=11, color=INK, weight="bold")
                ft.plot([0.06, 0.94], [0.135, 0.135], color=RULE, lw=0.8)
                ft.text(0.06, 0.105, cap, fontsize=8.8, color=INK, wrap=True, va="top")
            _page(pdf, draw)

        d = pdf.infodict()
        d["Title"] = "GAIRA V7 Phase 05 — The Canonical CSM Inference Engine (figures)"
        d["Subject"] = "15 figures; Raman-only frozen projection inference engine"
        d["Keywords"] = "GAIRA V7 Raman CSM inference calibration open-set provenance"

    print(f"  {out.relative_to(REPO)}  ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(figs) + 2} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
