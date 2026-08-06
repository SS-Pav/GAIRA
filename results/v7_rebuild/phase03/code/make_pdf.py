#!/usr/bin/env python3
"""GAIRA V7 — Phase 03: assemble the figure report PDF.

Every phase from 02.5 onward ships one PDF carrying all of its figures with captions, because
that is the artefact that actually gets circulated and read away from the repository. Figures
are embedded from the committed PNGs, so the PDF cannot drift from what the run produced.

    python results/v7_rebuild/phase03/code/make_pdf.py
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
    "fig01": "Model and K selection on label-free criteria. Stability is high wherever the "
             "membership is degenerate, so it must be read alongside information retained and "
             "the effective number of themes actually used; degenerate points are marked. "
             "Selected: archetypal analysis at K = 5.",
    "fig02": "The five theme spectra with their dominant bands. Names are derived from the "
             "bands and assigned only after validation. Theme-02 (1064, 1130, 1298, 1442, "
             "1658 cm-1) reproduces the acyl-chain-plus-cis-unsaturation signature that was "
             "the single accepted Phase 02 equivalence.",
    "fig03": "Soft membership S (49 CSMs x 5 themes), rows summing to one. Fifteen CSMs carry "
             "genuinely split membership and were left that way; nine are claimed by a theme "
             "that does not reconstruct them.",
    "fig04": "Theme overlap network — edge weight is shared membership mass. Themes are "
             "allowed to overlap, and the CSMs sitting on those edges are the bridges a hard "
             "assignment would misplace.",
    "fig05": "Inferred hierarchy. The number of levels was inferred rather than assumed. The "
             "top-level split separates aliphatic/unsaturated from polar/ring/carboxyl — the "
             "same hydrophobic/polar bipartition Phase 02.5 found independently in the CSM "
             "geometry, and the axis PCA found as its one reproducible component.",
    "fig06": "Per-theme stability against the 0.60 rejection floor, source and excitation "
             "holdouts, and the distribution of membership entropy. Theme-03 was rejected at "
             "0.59 despite having the most supporting CSMs — membership breadth is not "
             "evidence.",
    "fig07": "Bridge CSMs, and why bridges and poorly-explained CSMs are opposite findings: a "
             "bridge is well explained by several themes, a poorly-explained CSM by none.",
    "fig08": "Poorly-explained CSMs. A theme claims each of them, but does not reconstruct "
             "them. Recorded rather than absorbed — inventing a theme for an isolate is a "
             "motif borrowing foreign mass (limitation L-03).",
    "fig09": "Theme membership against diffusion coordinates. Five of fifteen pairs are "
             "significant gradients against a permutation null: membership varies smoothly "
             "along the manifold rather than switching, which is what respecting the Phase "
             "02.5 continuum means in practice.",
    "fig10": "The CSM map coloured by dominant theme and, separately, by curated chemistry — "
             "the latter revealed only after the themes were fixed. Adjusted mutual "
             "information 0.157, p < 0.0001: better than chance, far from a re-derivation of "
             "the ontology.",
    "fig11": "Reconstruction from the five-theme basis, per CSM, and the three worst cases. "
             "The theme basis explains 0.549 against the CSM basis's 1.000 at 9.8x "
             "compression — the price of abstraction, stated rather than smoothed over.",
    "fig12": "Evidence summary across the five themes. Every accepted theme also carries "
             "recorded counter-evidence and alternative explanations in the catalogue; a "
             "registry invariant enforces that a theme with neither has not been examined.",
    "fig13": "GAIRA V7 architecture after Phase 03, and exactly what Phase 04 consumes.",
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
    ax.text(0.08, 0.86, "GAIRA V7 — Phase 03", fontsize=27, color=INK, weight="bold")
    ax.text(0.08, 0.805, "Emergent Biochemical Theme Discovery", fontsize=17, color=INK)
    ax.plot([0.08, 0.92], [0.775, 0.775], color=RULE, lw=1.2)
    ax.text(0.08, 0.735, "Figure report — all 13 figures with captions", fontsize=11,
            color=MUTED)
    th = state["themes"]
    rows = [
        ("Status", state["status"] + " — themes derived FROM the frozen CSMs"),
        ("Frozen atlas", state["atlas_fingerprint"]),
        ("Frozen LSM registry", state["lsm_registry_fingerprint"]),
        ("Frozen CSM dictionary", state["csm_dictionary_fingerprint"]),
        ("Membership model", state["selected_model"]),
        ("K (= BSV dimension)", str(state["K"])),
        ("Themes accepted / rejected", f"{th['n_accepted']} / {th['n_rejected']}"),
        ("Bridge CSMs", f"{th['n_bridge_csms']} of 49 — left as bridges"),
        ("Poorly-explained CSMs", f"{th['n_unassigned_csms']} of 49 — left unplaced"),
        ("Bootstrap stability", f"{state['bootstrap_mean']:.3f}"),
        ("Leave-one-out stability", f"{state['loo_mean']:.3f}"),
        ("Ontology agreement (post hoc)", f"AMI {state['ontology_ami']:.3f}, "
                                          f"p = {state['ontology_p']:.4f}"),
        ("Theme layer adds value over CSMs", str(state["theme_layer_adds_value"])),
        ("Theme fingerprint", state["theme_fingerprint"]),
        ("Completed", state["completed_utc"][:19].replace("T", " ") + " UTC"),
    ]
    y = 0.66
    for k, v in rows:
        ax.text(0.08, y, k, fontsize=9.5, color=MUTED)
        ax.text(0.40, y, str(v), fontsize=9.5, color=INK, family="DejaVu Sans")
        y -= 0.035
    ax.text(0.08, 0.10,
            "Committed sources of record: reports/PHASE_03_REPORT.md and "
            "reports/PHASE_03_SCIENTIFIC_AUDIT.md\n"
            "No chemistry label was visible during discovery; labels were revealed once, "
            "after K was selected and validated.\n"
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
    priors = json.loads((A / "theme_registry_v1.json").read_text())
    figs = sorted(F.glob("fig*.png"))
    if not figs:
        print("no figures found — run make_figures.py first")
        return 1
    R.mkdir(parents=True, exist_ok=True)
    out = R / "PHASE_03_FIGURES.pdf"

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
        d["Title"] = "GAIRA V7 Phase 03 — Emergent Biochemical Theme Discovery (figures)"
        d["Subject"] = "13 figures with captions; K = 5, four themes accepted, one rejected"
        d["Keywords"] = "GAIRA V7 Raman biochemical themes"

    print(f"  {out.relative_to(REPO)}  ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(figs) + 2} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
