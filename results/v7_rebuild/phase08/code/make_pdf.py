#!/usr/bin/env python3
"""GAIRA V7 — Phase 08: assemble PHASE_08_FIGURES.pdf from the committed PNGs."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
PH = HERE.parent
REPO = PH.parents[2]
F, A, R = PH / "figures", PH / "artifacts", PH / "reports"
INK, MUTED, RULE = "#1a1a1a", "#6b7280", "#d1d5db"
PAGE = (11.69, 8.27)
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

CAPTIONS = {
    "F01": "Phase 08 architecture and canonical inference path. BSV2 is not on it and is not "
           "imported anywhere in the retrieval package. The result box carries the paired test "
           "that decides the phase.",
    "F02": "Split A retrieval for all five models, with paired tests against Model B. Model C "
           "selected beta = gamma = delta = 0 in every fold: the inner cross-validation chose "
           "to use no chemistry at all, so C is identical to B and the difference is exactly "
           "zero.",
    "F03": "Rank distribution, and the held-out-molecule split where molecule top-1 is "
           "undefined rather than zero. Chemistry reranking changes neither.",
    "F04": "Calibration from the score margin, risk-coverage and abstention. An earlier version "
           "derived confidence from 1/rank -- a function of the quantity being predicted -- and "
           "scored a discrimination of exactly 1.000. That was circular, not good.",
    "F05": "Noise robustness across seven perturbations. Raw spectrum appears most robust here "
           "because the bank is in-sample: a perturbed spectrum still matches its own "
           "unperturbed reference. It is not a held-out number.",
    "F06": "Failure analysis and chemistry-axis permutation importance. Both are flat by "
           "construction: with zero chemistry weight there is nothing to help, hurt, or permute.",
    "F07": "Evidence decomposition. Every Model C score is four weighted terms that sum to the "
           "displayed total, checked against the model's own output. Zero of 120 decompositions "
           "failed to reconcile.",
    "F08": "Candidate evolution from CSM ranking to final ranking. The two coincide exactly.",
    "F09": "Summary and decision gate. Outcome A: keep direct CSM retrieval.",
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
    ax.text(0.06, 0.89, "GAIRA V7 — Phase 08", fontsize=26, color=INK, weight="bold")
    ax.text(0.06, 0.836, "Hierarchical molecular retrieval", fontsize=17, color=INK)
    ax.plot([0.06, 0.94], [0.806, 0.806], color=RULE, lw=1.2)
    ax.text(0.06, 0.772, "Can chemistry-aware retrieval beat direct CSM retrieval, while "
            "remaining fully explainable?", fontsize=10, color=MUTED)
    a = {r["model"]: r for r in [dict(zip(("model",), (k,)), **v)
                                 for k, v in s["split_a"].items()]}
    d = s["decision"]
    rows = [
        ("Status", f"{state['status']} — outcome {d['outcome']}"),
        ("Decision", d["action"]),
        ("Baselines reproduced", "EXACT" if s["baselines_reproduced"] else "NO"),
        ("BSV2 on the inference path", "NO — not imported anywhere in the package"),
        ("", ""),
        ("A raw spectrum   top-1 / MRR", f"{s['split_a']['A_raw_spectrum']['top1']:.4f} / "
                                         f"{s['split_a']['A_raw_spectrum']['mrr']:.4f}"),
        ("B CSM            top-1 / MRR", f"{s['split_a']['B_csm']['top1']:.4f} / "
                                         f"{s['split_a']['B_csm']['mrr']:.4f}"),
        ("C chemistry      top-1 / MRR", f"{s['split_a']['C_chemistry_rerank']['top1']:.4f} / "
                                         f"{s['split_a']['C_chemistry_rerank']['mrr']:.4f}"),
        ("D probabilistic  top-1 / MRR", f"{s['split_a']['D_probabilistic']['top1']:.4f} / "
                                         f"{s['split_a']['D_probabilistic']['mrr']:.4f}"),
        ("E Bayesian       top-1 / MRR", f"{s['split_a']['E_bayesian_fusion']['top1']:.4f} / "
                                         f"{s['split_a']['E_bayesian_fusion']['mrr']:.4f}"),
        ("", ""),
        ("C vs B  delta top-1", f"{d['delta_top1']:+.4f}  95% CI "
                                f"[{d['delta_top1_ci'][0]:+.4f}, {d['delta_top1_ci'][1]:+.4f}]"),
        ("C vs B  McNemar p", f"{d['delta_top1_p']:.4f}"),
        ("Model C weights chosen in-fold", "alpha=0.4, beta=gamma=delta=0 in ALL five folds"),
        ("Decompositions non-reconciling", f"{s['explainability']['non_reconciling']} of "
                                           f"{s['explainability']['decompositions_checked']}"),
        ("", ""),
        ("Frozen inputs", "LSM 208482d6… · CSM 0b4aa550… · engine 20d8bd99… — verified"),
        ("Completed", state["finished"][:19].replace("T", " ") + " UTC"),
    ]
    y = 0.726
    for k, v in rows:
        if k:
            ax.text(0.06, y, k, fontsize=8.8, color=MUTED)
            ax.text(0.44, y, v, fontsize=8.8, color=INK)
        y -= 0.0298
    ax.text(0.06, 0.060,
            "Sources of record: reports/PHASE_08_REPORT.md · PHASE_08_SCIENTIFIC_AUDIT.md · "
            "PHASE_08_DECISION_GATE.md\n"
            "The nested weight search selected zero chemistry weight in every fold. Model C is "
            "therefore identical to Model B, and the\ndifference between them is exactly zero "
            "by construction rather than by measurement error.", fontsize=8.2, color=MUTED)


def contents(fig):
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.06, 0.93, "Figures", fontsize=19, color=INK, weight="bold")
    ax.plot([0.06, 0.94], [0.905, 0.905], color=RULE, lw=1.0)
    names = sorted(F.glob("F*.png"))
    for col, group in ((0.06, names[:5]), (0.52, names[5:])):
        y = 0.86
        for p in group:
            num = p.stem.split("_")[0]
            title = " ".join(p.stem.split("_")[1:]).replace("_", " ")
            ax.text(col, y, num.replace("F", ""), fontsize=9, color=MUTED)
            ax.text(col + 0.035, y, title, fontsize=9, color=INK)
            y -= 0.040


def main() -> int:
    state = json.loads((PH / "PHASE_STATE.json").read_text())
    s = json.loads((A / "phase08_summary_v1.json").read_text())
    figs = sorted(F.glob("F*.png"))
    if not figs:
        print("no figures — run make_figures.py first")
        return 1
    R.mkdir(parents=True, exist_ok=True)
    out = R / "PHASE_08_FIGURES.pdf"
    with PdfPages(out) as pdf:
        _page(pdf, lambda f: cover(f, state, s))
        _page(pdf, contents)
        for p in figs:
            img = mpimg.imread(p)
            h, w = img.shape[:2]

            def draw(fig, img=img, p=p):
                fig.patch.set_facecolor("white")
                cap = CAPTIONS.get(p.stem.split("_")[0], "")
                bw, bh = 0.90 * PAGE[0], 0.74 * PAGE[1]
                asp = h / w
                dw, dh = (bw, bw * asp) if bw * asp <= bh else (bh / asp, bh)
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
        d["Title"] = "GAIRA V7 Phase 08 — Hierarchical molecular retrieval"
        d["Subject"] = "9 figures; does chemistry-aware reranking beat direct CSM retrieval?"
        d["Keywords"] = "GAIRA V7 Raman molecular retrieval chemistry evidence"
    print(f"  {out.relative_to(REPO)}  ({out.stat().st_size / 1e6:.1f} MB, {len(figs) + 2} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
