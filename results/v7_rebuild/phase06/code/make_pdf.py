#!/usr/bin/env python3
"""GAIRA V7 — Phase 06: assemble the figure report PDF.

Figures are embedded from the committed PNGs so the PDF cannot drift from the figure script.

    python results/v7_rebuild/phase06/code/make_pdf.py
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
    "F01": "The GAIRA V7 pipeline through Phase 06. Everything left of the purple rule is "
           "frozen; Phase 06 adds the 16-dimensional Chemistry Evidence layer and its "
           "calibration. BSV2 and hierarchical retrieval are explicitly out of scope.",
    "F02": "The exact inference mathematics, traced to source rather than to report prose. "
           "Phase 05 returned a label from one nearest molecule; Phase 06 returns a "
           "16-dimensional vector with a class-size correction and soft broad-superclass "
           "routing, then calibrates it.",
    "F03": "All 16 frozen chemistry classes printed explicitly, with molecules, spectra, source "
           "distribution and imbalance. The largest class holds 80 spectra and the smallest 3 — "
           "a 26.7x ratio that §8 of the report shows is the dominant driver of per-class F1.",
    "F04": "All 37 candidate models across four families under flat grouped CV. This ranking is "
           "shown for the record only: selection was made by nested CV, and Figure 5 shows the "
           "0.045 macro-F1 difference between the two.",
    "F05": "Nested molecule-grouped cross-validation. Three different models won across five "
           "folds; the flat-vs-nested gap of 0.045 is exactly the selection bias that nested CV "
           "exists to remove.",
    "F06": "16-class confusion on molecule-grouped outer folds. Amber cells mark pre-declared "
           "chemically adjacent pairs — 45% of errors land on them against a 11% chance rate.",
    "F07": "Per-class precision, recall and F1. The four weakest classes are the four smallest, "
           "which is a corpus property rather than a modelling defect.",
    "F08": "Calibration benchmark. Isotonic regression wins ECE and loses log loss by a factor "
           "of four — per-class calibration then renormalised fixes the marginals and destroys "
           "the joint likelihood. Selecting on ECE would have chosen it.",
    "F09": "Reliability of the selected temperature calibrator. Classwise ECE of 0.026 is the "
           "reassuring number: small classes are not systematically miscalibrated, which "
           "top-label ECE alone would hide.",
    "F10": "Selective accuracy versus coverage, and the confidence distribution split by "
           "correctness. Discrimination of 0.668 is modest and Figure 16 shows the consequence.",
    "F11": "The full 375 x 16 evidence matrix, spectra grouped by true class. The bright "
           "diagonal is the signal; the off-diagonal texture is the ambiguity a hard label "
           "discards.",
    "F12": "Entropy 0.603, true-class rank, and effective rank 12.12 of a nominal 16 — the "
           "three numbers that together establish this is a genuinely soft, multi-dimensional "
           "representation rather than a disguised classifier.",
    "F13": "Replicate consistency 0.947 and a within/between-class separation of 0.497 on the "
           "evidence vector itself.",
    "F14": "Eleven Raman perturbations at five levels each. The 16-d layer tracks the CSM layer "
           "it is computed from (retention 0.937 vs 0.943) and clearly beats the raw spectrum "
           "on both accuracy and retention.",
    "F15": "Held-out chemistry novelty — an entire class withheld from the atlas. Five of six "
           "classes are detected; acylglycerol fails at chance because fatty acids remain and "
           "share the acyl-chain motifs that dominate its spectrum.",
    "F16": "Sixteen-axis radars for seven cases chosen by rule rather than by eye. Note urea, "
           "misclassified as a free amino acid at confidence 0.99 — calibration reduces average "
           "over-confidence, it does not catch a confidently wrong answer.",
    "F17": "The ordered-bar alternative. Sixteen spokes are cluttered and a filled polygon "
           "invites a composition reading; the same numbers as bars carry neither problem.",
    "F18": "Provenance waterfall. The decomposition is mathematically exact for the selected "
           "family: 1,125 chains verified against the frozen registries, none broken.",
    "F19": "The low-EV tail by name, accuracy by source, and the relationship between "
           "reconstruction quality and confidence. Low EV degrades accuracy to 0.625 from 0.844 "
           "but does not destroy it.",
    "F20": "Every semantic layer on identical outer folds, and the semantic comparators. The "
           "unsupervised bar is amber because it is not a fair comparison — predicting a "
           "proximity-defined grouping using proximity is near self-prediction.",
    "F21": "Curated ontology versus unsupervised grouping. ARI 0.595 and AMI 0.725 say the "
           "curated partition is substantially but not fully recoverable from spectra alone; "
           "the disagreement is where a curated layer earns its place.",
    "F22": "Phase 06 summary — accuracy against dimension, F1 against class size, headline "
           "numbers, and all 18 gates.",
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
    ax.text(0.06, 0.88, "GAIRA V7 — Phase 06", fontsize=27, color=INK, weight="bold")
    ax.text(0.06, 0.825, "The Validated Chemistry Evidence Layer", fontsize=17, color=INK)
    ax.plot([0.06, 0.94], [0.795, 0.795], color=RULE, lw=1.2)
    ax.text(0.06, 0.760, "Figure report — all 22 figures with captions · Raman only",
            fontsize=10.5, color=MUTED)
    p, cal, nov = s["performance"], s["calibration"], s["novelty"]
    rows = [
        ("Status", f"{state['status']} — 18 of 18 gates pass"),
        ("Implements", "A-19  49-d CSM activation → 16-d Chemistry Evidence"),
        ("Does not implement", "BSV2 (Phase 07) · hierarchical retrieval (Phase 08)"),
        ("Selected model", f"{s['selected_model']['candidate']}  (nested CV, modal of 5 folds)"),
        ("Calibration", f"{cal['method']}  (log loss, subject to non-degeneracy floors)"),
        ("", ""),
        ("fine-class top-1", f"{p['top1']['value']:.3f}   95% CI "
                             f"[{p['top1']['ci95'][0]:.3f}, {p['top1']['ci95'][1]:.3f}]"),
        ("fine-class top-3", f"{p['top3']['value']:.3f}   95% CI "
                             f"[{p['top3']['ci95'][0]:.3f}, {p['top3']['ci95'][1]:.3f}]"),
        ("macro-F1", f"{p['macro_f1']['value']:.3f}   (classes with ≥5 molecules: "
                     f"{s['macro_f1_classes_ge5_molecules']:.3f})"),
        ("balanced accuracy", f"{p['balanced_accuracy']['value']:.3f}"),
        ("ECE / classwise ECE", f"{cal['ece']:.3f} / {cal['classwise_ece']:.3f}"),
        ("replicate consistency", f"{s['soft_evidence']['replicate_consistency']:.3f}"),
        ("held-out chemistry novelty", f"mean AUROC {nov['mean_auroc']:.3f} over "
                                       f"{nov['n_classes_tested']} classes — one fails at chance"),
        ("broken provenance chains", f"{s['provenance']['broken']} of "
                                     f"{s['provenance']['n_chains']}"),
        ("", ""),
        ("Frozen inputs", "LSM 208482d6… · CSM 0b4aa550… · engine 20d8bd99… — all verified"),
        ("Completed", state["finished"][:19].replace("T", " ") + " UTC"),
    ]
    y = 0.700
    for k, v in rows:
        if k:
            ax.text(0.06, y, k, fontsize=9.2, color=MUTED)
            ax.text(0.34, y, v, fontsize=9.2, color=INK)
        y -= 0.0335
    ax.text(0.06, 0.075,
            "Committed sources of record: reports/PHASE_06_CHEMISTRY_EVIDENCE_LAYER.md and "
            "reports/PHASE_06_SCIENTIFIC_AUDIT.md\n"
            "The evidence vector is support within the chemistry represented by the Raman "
            "reference atlas. It is NOT a concentration, NOT a composition,\nand NOT a mixture "
            "statement. Every spectrum in this phase is a pure Raman reference.",
            fontsize=8.2, color=MUTED)


def contents(fig):
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.06, 0.93, "Figures", fontsize=19, color=INK, weight="bold")
    ax.plot([0.06, 0.94], [0.905, 0.905], color=RULE, lw=1.0)
    names = sorted(F.glob("F*.png"))
    for col, group in ((0.06, names[:11]), (0.52, names[11:])):
        y = 0.86
        for p in group:
            num = p.stem.split("_")[0]
            title = " ".join(p.stem.split("_")[1:]).replace("_", " ")
            ax.text(col, y, num.replace("F", ""), fontsize=9, color=MUTED)
            ax.text(col + 0.035, y, title, fontsize=9, color=INK)
            y -= 0.037


def main() -> int:
    state = json.loads((PH / "PHASE_STATE.json").read_text())
    s = json.loads((A / "phase06_summary_v1.json").read_text())
    figs = sorted(F.glob("F*.png"))
    if not figs:
        print("no figures found — run make_figures.py first")
        return 1
    R.mkdir(parents=True, exist_ok=True)
    out = R / "PHASE_06_RESULTS.pdf"
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
        d["Title"] = "GAIRA V7 Phase 06 — The Validated Chemistry Evidence Layer (figures)"
        d["Subject"] = "22 figures; 16-dimensional chemistry evidence over the frozen CSM basis"
        d["Keywords"] = "GAIRA V7 Raman chemistry evidence calibration novelty provenance"
    print(f"  {out.relative_to(REPO)}  ({out.stat().st_size / 1e6:.1f} MB, {len(figs) + 2} pages)")
    print(f"  tracked by git: {'yes' if not _ignored(out) else 'NO (gitignored)'}")
    return 0


def _ignored(p: Path) -> bool:
    import subprocess
    return subprocess.run(["git", "check-ignore", "-q", str(p)], cwd=REPO).returncode == 0


if __name__ == "__main__":
    raise SystemExit(main())
