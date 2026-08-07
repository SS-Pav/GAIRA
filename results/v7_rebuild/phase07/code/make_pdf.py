#!/usr/bin/env python3
"""GAIRA V7 — Phase 07: assemble PHASE_07_FIGURES.pdf from the committed PNGs."""
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
    "F01": "Phase 07 architecture. BSV2 is learned from the validated 16-dimensional Chemistry "
           "Evidence matrix and nothing else: no spectra, no CSM activations, no geometry, no "
           "cluster ids, no coordinates, no theme layer, no legacy BSV. Frozen CSM and LSM "
           "artefacts are read for explanation only, after fitting.",
    "F02": "Programme-number sweep: six model families over K = 2 to 16, with the pre-registered "
           "floors drawn. The floors on K (<= the input's effective rank) and on single-axis "
           "share were added after a first run selected K = 16, where NMF learns a permutation "
           "of the identity and every programme equals one chemistry class.",
    "F03": "Programme loadings over the 16 chemistry axes, with the percentages that generate "
           "each programme's automatic description. Three programmes are genuine multi-chemistry "
           "compressions - lipid, energy metabolism, carbohydrate - and six are near-single-class.",
    "F04": "Programme overlap and usage. No two programmes duplicate each other and none "
           "dominates: the top programme wins for 22% of spectra against a 60% ceiling.",
    "F05": "Programme activation map: 375 spectra by 9 programmes, spectra grouped by curated "
           "chemistry class. The class labels are revealed for display only and were never an "
           "input to the factorisation.",
    "F06": "The molecules that activate each programme most. Evidence first: these lists are "
           "what the automatic descriptions are composed from.",
    "F07": "Mean Raman spectrum of each programme's top molecules, with the programme's dominant "
           "CSM bands marked. Spectra are shown for interpretation only.",
    "F08": "Stability under three separate perturbations - resampling spectra, refitting from a "
           "new seed, and withholding a whole molecule-grouped fold - and generalisation to "
           "held-out molecules.",
    "F09": "Noise robustness propagated through the entire frozen chain: spectrum to CSM to "
           "Chemistry Evidence to BSV2. This is the only place in the phase where spectra are "
           "touched, and only to perturb them.",
    "F10": "Reconstruction of the Chemistry Evidence vector, globally and per chemistry axis. "
           "The axes reconstructed worst are the ones with the least evidence to begin with.",
    "F11": "Programme similarity graph. Edge width is the cosine between programme loadings, "
           "node size is usage share.",
    "F12": "Chemistry Evidence versus BSV2 versus a PCA control at the same K, and the "
           "decision gate.",
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
    ax.text(0.06, 0.89, "GAIRA V7 — Phase 07", fontsize=26, color=INK, weight="bold")
    ax.text(0.06, 0.836, "BSV2 — the biochemical programme layer", fontsize=17, color=INK)
    ax.plot([0.06, 0.94], [0.806, 0.806], color=RULE, lw=1.2)
    ax.text(0.06, 0.772, "What biochemical programmes best explain the validated chemistry "
            "evidence?", fontsize=10, color=MUTED)
    r, st, comp = s["reconstruction"], s["stability"], s["compositeness"]
    gen = np.mean([g["explained_variance"] for g in s["generalisation"]])
    p02 = s.get("p02_compliant_alternative")
    rows = [
        ("Status", f"{state['status']} — BSV2 adopted: {s['adopted']}"),
        ("Input", "Chemistry Evidence 375 x 16, and nothing else"),
        ("Selected", f"{s['model']['family']} at K = {s['model']['K']} "
                     f"(pre-registered rule, unadjusted)"),
        ("Compression", f"{16 / s['model']['K']:.1f}x  (16 → {s['model']['K']})"),
        ("", ""),
        ("Reconstruction", f"EV {r['explained_variance']:.3f} · cosine {r['mean_cosine']:.3f} · "
                           f"RMSE {r['rmse']:.3f}"),
        ("Held-out reconstruction", f"EV {gen:.3f} (gap {r['explained_variance'] - gen:+.3f})"),
        ("Stability", f"bootstrap {st['bootstrap']:.3f} · seed {st['seed']:.3f} · fold "
                      f"{st['fold']:.3f}"),
        ("Programmes below 0.70 recovery", f"{st['n_programmes_below_0.7']} of "
                                           f"{s['model']['K']}"),
        ("Information retained", f"{s['information_retained']:.3f} (floor 0.50)"),
        ("Noise robustness", f"programme cosine "
                             f"{s['noise_robustness']['mean_programme_cosine']:.3f}"),
        ("", ""),
        ("Genuine multi-chemistry programmes", f"{comp['n_genuinely_composite']} of "
                                               f"{s['model']['K']} — the rest are "
                                               f"near-single-class"),
        ("Programme redundancy", f"max pairwise overlap "
                                 f"{max(max(r_) for r_ in np.array(s['overlap_matrix']) - np.eye(s['model']['K'])):.3f}"),
        ("P-02-compliant alternative", (f"{p02['family']} K={p02['K']}, objective cost "
                                        f"{p02['objective_cost_vs_rule_winner']:+.3f}")
                                       if p02 else "none"),
        ("", ""),
        ("Frozen inputs", "LSM 208482d6… · CSM 0b4aa550… · engine 20d8bd99… — verified"),
        ("Phase 08", "NOT BEGUN"),
        ("Completed", state["finished"][:19].replace("T", " ") + " UTC"),
    ]
    y = 0.726
    for k, v in rows:
        if k:
            ax.text(0.06, y, k, fontsize=9.0, color=MUTED)
            ax.text(0.42, y, v, fontsize=9.0, color=INK)
        y -= 0.0305
    ax.text(0.06, 0.062,
            "Sources of record: reports/PHASE_07_REPORT.md · PHASE_07_SCIENTIFIC_AUDIT.md · "
            "PHASE_07_DECISION_GATE.md\n"
            "BSV2 is not chemistry, not molecules, not motifs. It is a latent programme space "
            "over the chemistry evidence layer.", fontsize=8.2, color=MUTED)


def contents(fig):
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.06, 0.93, "Figures", fontsize=19, color=INK, weight="bold")
    ax.plot([0.06, 0.94], [0.905, 0.905], color=RULE, lw=1.0)
    names = sorted(F.glob("F*.png"))
    for col, group in ((0.06, names[:6]), (0.52, names[6:])):
        y = 0.86
        for p in group:
            num = p.stem.split("_")[0]
            title = " ".join(p.stem.split("_")[1:]).replace("_", " ")
            ax.text(col, y, num.replace("F", ""), fontsize=9, color=MUTED)
            ax.text(col + 0.035, y, title, fontsize=9, color=INK)
            y -= 0.040


def main() -> int:
    state = json.loads((PH / "PHASE_STATE.json").read_text())
    s = json.loads((A / "phase07_summary_v1.json").read_text())
    figs = sorted(F.glob("F*.png"))
    if not figs:
        print("no figures — run make_figures.py first")
        return 1
    R.mkdir(parents=True, exist_ok=True)
    out = R / "PHASE_07_FIGURES.pdf"
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
        d["Title"] = "GAIRA V7 Phase 07 — BSV2 biochemical programme layer"
        d["Subject"] = "12 figures; programmes learned from the chemistry evidence layer"
        d["Keywords"] = "GAIRA V7 Raman BSV2 biochemical programmes NMF"
    print(f"  {out.relative_to(REPO)}  ({out.stat().st_size / 1e6:.1f} MB, {len(figs) + 2} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
