#!/usr/bin/env python3
"""Generate the ten GAIRA V7 planning diagrams (SVG vector + PNG preview).

Documentation figures only — no scientific computation, no model fitting, no data loading.
Every arrow in every diagram corresponds to a defined computational operation described in
GAIRA_v7_rebuild/architecture/.  Deterministic: no RNG, no timestamps in the output.

Repo policy gitignores *.pdf, so vector output is SVG (the Markdown/SVG source is what the
repository tracks).

    python GAIRA_v7_rebuild/results/figures/planning/make_planning_figures.py
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = pathlib.Path(__file__).resolve().parent

# ── palette ────────────────────────────────────────────────────────────────────
INK = "#1a1a1a"
MUTED = "#6b7280"
LINE = "#9ca3af"
FROZEN = "#6b7280"      # frozen / legacy
LEARN = "#2563eb"       # offline learning
INFER = "#059669"       # live inference
WARN = "#b45309"        # problem / limitation
GOOD = "#15803d"        # what works
BAD = "#b91c1c"         # what fails
NEUTRAL = "#374151"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.18,
    "svg.fonttype": "none",     # keep text as text in the SVG
})


# ── primitives ─────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, text, *, fc="white", ec=NEUTRAL, tc=INK, fs=8.5,
        lw=1.1, ls="-", rounding=0.02, weight="normal", va="center"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={rounding}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va=va, fontsize=fs,
            color=tc, zorder=3, linespacing=1.45, weight=weight)


def arrow(ax, p0, p1, *, color=LINE, lw=1.2, ls="-", label=None, lfs=7.0,
          loff=(0.0, 0.0), lcolor=MUTED, style="-|>", ms=7):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=ms, color=color,
        linewidth=lw, linestyle=ls, shrinkA=1.5, shrinkB=1.5, zorder=1))
    if label:
        ax.text((p0[0] + p1[0]) / 2 + loff[0], (p0[1] + p1[1]) / 2 + loff[1],
                label, ha="center", va="center", fontsize=lfs, color=lcolor,
                zorder=3, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none"))


def canvas(w, h, title, subtitle=None):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(0, 99, title, fontsize=12.5, weight="bold", color=INK, va="top")
    if subtitle:
        ax.text(0, 94.2, subtitle, fontsize=8.2, color=MUTED, va="top")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", format="svg")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  {name}.svg + {name}.png")


def footer(ax, text, y=1.5):
    ax.text(0, y, text, fontsize=7.0, color=MUTED, va="bottom", linespacing=1.5)


# ── 01 flat vs hierarchical ────────────────────────────────────────────────────
def fig01():
    fig, ax = canvas(11.5, 7.6,
                     "1 — Flat global architecture (V5/V6) vs hierarchical architecture (V7)",
                     "Left: one basis serves every chemistry. Right: capacity is allocated per class, then reintegrated.")
    ax.plot([50, 50], [4, 86], color="#e5e7eb", lw=1.0, zorder=0)
    ax.text(24, 88.0, "V5 / V6 / V6.2  —  FLAT", ha="center", fontsize=9.5,
            weight="bold", color=FROZEN)
    ax.text(75, 88.0, "V7  —  HIERARCHICAL", ha="center", fontsize=9.5,
            weight="bold", color=LEARN)

    L = [("375 Raman spectra\n167 analytes", "white", NEUTRAL),
         ("canonical preprocessing\nasls → savgol → L2", "white", NEUTRAL),
         ("SINGLE GLOBAL NMF\none basis, all chemistry", "#fef3c7", WARN),
         ("24 components\n3 of 24 purity ≥ 0.5", "#fef3c7", WARN),
         ("MSS overlay\n13 → 18 curated motifs", "#fef3c7", WARN),
         ("13 themes (v2) / 6 (V6.2)", "white", NEUTRAL),
         ("BSV", "white", NEUTRAL)]
    y = 78
    for i, (t, fc, ec) in enumerate(L):
        box(ax, 5, y, 38, 6.4, t, fc=fc, ec=ec, fs=8.0)
        if i < len(L) - 1:
            arrow(ax, (24, y), (24, y - 4.0))
        y -= 10.4

    R = [("375 Raman spectra\n167 canonical molecules", "white", NEUTRAL),
         ("canonical preprocessing  (UNCHANGED)", "white", NEUTRAL),
         ("balanced references\none molecule = one unit", "#dbeafe", LEARN),
         ("per-class local NMF   ·   adaptive k_c\nprotein | sterol | flavin | …", "#dbeafe", LEARN),
         ("stable Local Spectral Motifs (LSMs)", "#dbeafe", LEARN),
         ("Consensus Spectral Motifs (CSMs)\ncross-class consensus + provenance", "#dbeafe", LEARN),
         ("K soft biochemical themes", "white", NEUTRAL),
         ("absolute BSV", "white", NEUTRAL)]
    y = 78
    for i, (t, fc, ec) in enumerate(R):
        h = 6.4
        box(ax, 56, y, 38, h, t, fc=fc, ec=ec, fs=8.0)
        if i < len(R) - 1:
            arrow(ax, (75, y), (75, y - 2.6), color=LEARN)
        y -= 9.0

    ax.text(24, 6.5,
            "one basis must represent broad shared\nstructure AND rare specific structure —\n"
            "capacity allocated by spectrum count",
            ha="center", fontsize=7.6, color=BAD, linespacing=1.5)
    ax.text(75, 6.5,
            "each chemistry gets its own capacity,\nthen local dictionaries are reintegrated\n"
            "into one comparable coordinate system",
            ha="center", fontsize=7.6, color=GOOD, linespacing=1.5)
    save(fig, "fig01_flat_vs_hierarchical")


# ── 02 learning pipeline ───────────────────────────────────────────────────────
def fig02():
    fig, ax = canvas(9.6, 10.4, "2 — V7 learning-mode pipeline (offline)",
                     "Every arrow is a defined computational operation. All fitting happens here and nowhere else.")
    steps = [
        ("Raw Raman grounding corpus\n375 spectra · 167 analytes · pure Raman only", "white", NEUTRAL, ""),
        ("canonical preprocessing\ncrop → resample → asls → savgol → L2", "#f3f4f6", FROZEN, "unchanged from V5"),
        ("canonical molecule IDs · replicate groups · quality q\nfrozen analyte-grouped CV splits", "#dbeafe", LEARN, "Phase 00"),
        ("balanced reference construction\nselect from {A control, B weighted, C prototype}", "#dbeafe", LEARN, "Phase 01"),
        ("chemical-family partition   X → {X_c}", "#dbeafe", LEARN, "organisational prior"),
        ("class-specific repeated non-negative decomposition\nX_c ≈ W_c H_c   ·   adaptive k_c   ·   R repeated fits", "#dbeafe", LEARN, "Phase 02"),
        ("stable Local Spectral Motifs\nHungarian alignment → recurrence → stability threshold", "#dbeafe", LEARN, ""),
        ("LSM similarity graph   ·   6 edge features", "#dbeafe", LEARN, "Phase 03"),
        ("Consensus Spectral Motifs\nintegration method SELECTED ON EVIDENCE", "#dbeafe", LEARN, ""),
        ("CSM → theme soft mapping   t = Sᵀc\nS sparse, non-negative, row-normalised", "#dbeafe", LEARN, "Phase 04"),
        ("continuous BSV reference space\nreference stats · OOD support · uncertainty", "#dbeafe", LEARN, "Phase 05"),
        ("FROZEN GAIRA V7 ATLAS\nmulti-layer fingerprint over 7 layers", "#dcfce7", GOOD, "Phase 06"),
    ]
    y, h, gap = 86.0, 5.6, 6.95
    for i, (t, fc, ec, note) in enumerate(steps):
        box(ax, 8, y, 76, h, t, fc=fc, ec=ec, fs=8.0,
            weight="bold" if i == len(steps) - 1 else "normal")
        if note:
            ax.text(86.5, y + h / 2, note, fontsize=6.8, color=MUTED, va="center")
        if i < len(steps) - 1:
            arrow(ax, (46, y), (46, y - (gap - h)), color=LEARN)
        y -= gap
    footer(ax, "Fitting operations: NMF (per class) · graph construction · consensus clustering /\n"
               "community detection · membership derivation · reference statistics · PCA (frozen view).\n"
               "None of these ever runs at inference time.", y=1.0)
    save(fig, "fig02_learning_pipeline")


# ── 03 inference pipeline ──────────────────────────────────────────────────────
def fig03():
    fig, ax = canvas(9.6, 9.0, "3 — V7 inference-mode pipeline (live)",
                     "Projection and arithmetic only. No fitting, no batch statistics, no RNG.")
    steps = [
        ("NEW RAMAN SPECTRUM\nwavenumbers · intensities · metadata", "white", NEUTRAL, ""),
        ("canonical preprocessing\ncrop → resample → asls → savgol → L2", "#d1fae5", INFER, "deterministic, per-spectrum"),
        ("fixed-dictionary non-negative projection\nc(x) = argmin_{c≥0} ‖x − cᵀ·CSM‖²", "#d1fae5", INFER, "NNLS"),
        ("CSM activation evidence   c(x) ∈ ℝ₊^M\nbands · supporting analytes · singleton/anchor flags", "#d1fae5", INFER, "lookup"),
        ("LSM activation evidence   [optional]", "#d1fae5", INFER, "explanation only"),
        ("soft biochemical themes   t(x) = Sᵀ c(x)", "#d1fae5", INFER, "matrix multiply"),
        ("ABSOLUTE BSV   BSV(x) = t(x) ∈ ℝ₊^K", "#dcfce7", GOOD, "not a delta, not a label"),
        ("reference comparison · QC · uncertainty\nelevation z · OOD · residual · band fidelity", "#d1fae5", INFER, "frozen affine + distances"),
        ("domain-context interpretation\nserum / EV / plasma / tissue / pathogen", "white", NEUTRAL, "downstream only"),
    ]
    y, h, gap = 82.0, 6.0, 8.6
    for i, (t, fc, ec, note) in enumerate(steps):
        box(ax, 6, y, 74, h, t, fc=fc, ec=ec, fs=8.0,
            weight="bold" if i == 6 else "normal")
        if note:
            ax.text(82, y + h / 2, note, fontsize=6.8, color=MUTED, va="center")
        if i < len(steps) - 1:
            arrow(ax, (43, y), (43, y - (gap - h)), color=INFER)
        y -= gap
    ax.add_patch(Rectangle((6, y + gap - 1.2), 74, 0.0, fill=False, ec="none"))
    footer(ax,
           "PROHIBITED at inference: NMF fit · PCA fit · UMAP · clustering · community detection · ontology "
           "optimisation · threshold tuning on incoming data · any batch statistic.\n"
           "General principle: a spectrum's output must be identical alone and in a batch of ten thousand.")
    save(fig, "fig03_inference_pipeline")


# ── 04 coverage imbalance ──────────────────────────────────────────────────────
def fig04():
    fig, ax = canvas(11.0, 7.2, "4 — Coverage imbalance and the V7 correction",
                     "Corpus: 167 analytes across 18 chemical families. Source: results/v6_rebuild/tables/p2_family_census.csv")
    fam = [("protein", 32, 17), ("saccharide", 27, 17), ("amino_acid", 17, 11),
           ("triglyceride", 15, 14), ("organic_acid", 15, 8), ("fatty_acid", 12, 10),
           ("sterol", 9, 7), ("cofactor", 6, 2), ("unknown", 6, 4), ("purine", 5, 0),
           ("polysaccharide", 5, 4), ("lipid", 5, 4), ("nucleic_acid", 3, 3),
           ("pyrimidine", 3, 0), ("phospholipid", 2, 2), ("small_nitrog.", 2, 1),
           ("carotenoid", 2, 2), ("polyol", 1, 1)]
    x0, ytop, bh, sc = 15.0, 82.0, 3.35, 1.02
    ax.text(x0 + 17, 87.5, "analytes per chemical family  (bar = analytes, dark = covered by a v1 motif)",
            fontsize=8.0, color=INK, ha="center")
    for i, (n, tot, unc) in enumerate(fam):
        y = ytop - i * (bh + 0.85)
        cov = tot - unc
        ax.add_patch(Rectangle((x0, y), tot * sc, bh, fc="#fecaca", ec=BAD, lw=0.5, zorder=2))
        if cov:
            ax.add_patch(Rectangle((x0, y), cov * sc, bh, fc="#4b5563", ec=NEUTRAL, lw=0.5, zorder=3))
        ax.text(x0 - 1.2, y + bh / 2, n, ha="right", va="center", fontsize=7.0, color=INK)
        ax.text(x0 + tot * sc + 1.0, y + bh / 2, str(tot), ha="left", va="center",
                fontsize=7.0, color=MUTED)
    ax.text(x0 + 18, ytop - 18 * (bh + 0.85) - 2.0,
            "107 of 167 analytes (64.1%) uncovered by any v1 motif",
            fontsize=7.4, color=BAD, ha="center")

    box(ax, 54, 58, 44, 26,
        "THE PROBLEM\n\n"
        "min ‖X − WH‖²  sums over ROWS of X.\nA row is one spectrum.\n\n"
        "Top-5 families = 106/167 analytes (63%)\nBottom-4 families = 8 analytes (4.8%)\n\n"
        "→ dense chemistry sets the objective\n→ rare chemistry gets no dedicated capacity",
        fc="#fef3c7", ec=WARN, fs=7.5)

    box(ax, 54, 28, 44, 27,
        "THE V7 CORRECTION\n\n"
        "A  all spectra, equal weight        [control]\n"
        "B  analyte-balanced weighted fitting\n"
        "C  one robust prototype per analyte\n"
        "D  class-specific decomposition     [structural]\n"
        "E  adaptive k_c per class\n"
        "F  anchored atoms for rare chemistry\n\n"
        "Unit change:  one spectrum = one vote\n         →  one molecule = one reference unit",
        fc="#dbeafe", ec=LEARN, fs=7.5)

    box(ax, 54, 7, 44, 18,
        "WHAT BALANCING CANNOT DO\n\n"
        "Phospholipid has 2 analytes before balancing\nand 2 after.  Sphingolipids are absent entirely.\n"
        "Rare classes are NEVER bootstrapped by\nduplicating spectra — that adds no information.",
        fc="#fee2e2", ec=BAD, fs=7.6)
    save(fig, "fig04_coverage_imbalance")


# ── 05 hierarchy ───────────────────────────────────────────────────────────────
def fig05():
    fig, ax = canvas(10.6, 7.8, "5 — The V7 representation hierarchy: LSM → CSM → theme → BSV",
                     "Each level is non-negative, provenance-carrying, and derived from the level below.")
    lv = [
        (78, "canonical Raman references", "ℝ₊^676 per canonical molecule · balanced", "white", NEUTRAL),
        (63, "Local Spectral Motifs (LSMs)", "H_c ∈ ℝ₊^{k_c×676} · learned per class · stability-selected", "#dbeafe", LEARN),
        (48, "Consensus Spectral Motifs (CSMs)", "CSM ∈ ℝ₊^{M×676} · cross-class consensus · THE EVIDENCE UNIT", "#bfdbfe", LEARN),
        (33, "biochemical themes", "t = Sᵀc · S ∈ ℝ₊^{M×K} sparse, non-negative, row-normalised", "#d1fae5", INFER),
        (18, "Biochemical State Vector", "BSV ∈ ℝ₊^K · ABSOLUTE · fixed global coordinate system", "#dcfce7", GOOD),
    ]
    for i, (y, t, sub, fc, ec) in enumerate(lv):
        box(ax, 8, y, 60, 10.0, "", fc=fc, ec=ec)
        ax.text(11, y + 6.6, t, fontsize=9.2, weight="bold", color=INK, va="center")
        ax.text(11, y + 3.1, sub, fontsize=7.3, color=MUTED, va="center")
        if i < len(lv) - 1:
            arrow(ax, (38, y), (38, y - 5.0), color=LEARN if i < 2 else INFER)

    ops = ["class-local NMF\n+ stability selection",
           "similarity graph\n+ consensus integration",
           "soft membership\nmatrix S",
           "identity\nBSV = t"]
    for (y, _, _, _, _), op in zip(lv[:-1], ops):
        ax.text(70, y - 2.5, op, fontsize=6.9, color=MUTED, va="center", linespacing=1.4)

    box(ax, 8, 4, 84, 10.5,
        "DERIVED FROM THE BSV — none of these is a BSV\n"
        "ΔBSV = BSV₂ − BSV₁  (signed)      ·      elevation z = (t−μ)/σ  (signed)      ·"
        "      cohort-standardised view  (visualisation)\n"
        "DART trajectory BSV(E,t)  (sequence of absolute BSVs)      ·      PCA view y = Pᵀ(BSV−μ)  "
        "(visualisation; P applied, never fitted)",
        fc="#f9fafb", ec=LINE, fs=7.4)
    save(fig, "fig05_representation_hierarchy")


# ── 06 offline vs live ─────────────────────────────────────────────────────────
def fig06():
    fig, ax = canvas(10.6, 6.8, "6 — Offline learning versus live inference",
                     "The most important architectural line in the system. It exists to make spectra comparable across labs and years.")
    box(ax, 3, 24, 42, 62, "", fc="#eff6ff", ec=LEARN, lw=1.4)
    box(ax, 55, 24, 42, 62, "", fc="#ecfdf5", ec=INFER, lw=1.4)
    ax.text(24, 82, "OFFLINE  —  LEARNING MODE", ha="center", fontsize=9.5, weight="bold", color=LEARN)
    ax.text(76, 82, "LIVE  —  INFERENCE MODE", ha="center", fontsize=9.5, weight="bold", color=INFER)

    left = ["NMF fitting (per class)", "repeated fits + bootstrap", "Hungarian alignment",
            "graph construction", "consensus clustering /\n   community detection",
            "membership derivation", "reference statistics", "PCA fitting (for the frozen view)",
            "ontology / theme optimisation"]
    y = 76
    for t in left:
        ax.text(7, y, "▸  " + t, fontsize=7.6, color=INK, va="top", linespacing=1.4)
        y -= 5.2 if "\n" not in t else 8.4

    right_ok = ["canonical preprocessing", "NNLS against frozen dictionary",
                "matrix multiply by frozen S", "frozen affine (μ, σ)",
                "APPLY frozen PCA (picture only)", "distances to frozen references",
                "uncertainty propagation", "trajectory append"]
    y = 76
    for t in right_ok:
        ax.text(58, y, "✓  " + t, fontsize=7.6, color=GOOD, va="top")
        y -= 5.2
    ax.text(58, y - 1.5, "✗  NMF · PCA · UMAP · clustering ·\n     community detection · batch stats",
            fontsize=7.6, color=BAD, va="top", linespacing=1.45)

    arrow(ax, (45.8, 55), (54.2, 55), color=NEUTRAL, lw=1.6, ms=10)
    ax.text(50, 59.5, "FROZEN\nATLAS", ha="center", fontsize=6.9, color=NEUTRAL,
            weight="bold", linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    box(ax, 3, 6, 94, 13,
        "WHY THE LINE IS ABSOLUTE\n"
        "Two spectra measured in different labs years apart are comparable only if projected onto the SAME FIXED AXES.\n"
        "Any fitting at inference — even a PCA re-fitted 'just for the plot' — makes the coordinate system depend on the batch,\n"
        "and the comparison silently becomes meaningless.  Testable form: output must be identical alone and in a batch of N.",
        fc="#f9fafb", ec=LINE, fs=7.5)
    save(fig, "fig06_offline_vs_live")


# ── 07 phase roadmap ───────────────────────────────────────────────────────────
def fig07():
    fig, ax = canvas(10.6, 8.8, "7 — Phase dependency roadmap",
                     "Sequential critical path. Each gate is binding: a failed gate stops the phase.")
    phases = [
        ("00", "Benchmark lock", "no leakage · baseline reproduced\nsplits deterministic · criteria frozen"),
        ("01", "Balanced references", "rule pre-registered · no supervision\nbalance improved, fidelity preserved"),
        ("02", "Local Spectral Motifs", "every LSM stable · k_c justified\nrare classes routed explicitly"),
        ("03", "Consensus Spectral Motifs", "provenance complete · M justified\nmethod chosen on evidence"),
        ("04", "Biochemical themes", "chemistry only · soft membership\nK justified · value over CSM shown"),
        ("05", "Biochemical State Vector", "deterministic · absolute ≠ delta\nuncertainty · effective rank"),
        ("06", "Engine integration", "no inference fitting · batch independent\nclean clone · fingerprint verified"),
        ("07", "Raman validation", "PRE-REGISTERED SUCCESS CRITERIA MET\nor V5 atlas is retained"),
    ]
    y, h, gap = 82.0, 6.9, 8.15
    for i, (n, name, gate) in enumerate(phases):
        dec = (i == 7)
        box(ax, 6, y, 13, h, n, fc="#dcfce7" if dec else "#dbeafe",
            ec=GOOD if dec else LEARN, fs=11, weight="bold")
        box(ax, 20, y, 30, h, name, fc="white", ec=GOOD if dec else NEUTRAL,
            fs=8.8, weight="bold" if dec else "normal")
        ax.text(53, y + h / 2, gate, fontsize=6.9, color=MUTED, va="center", linespacing=1.4)
        if i < len(phases) - 1:
            arrow(ax, (12.5, y), (12.5, y - (gap - h)), color=LEARN)
        y -= gap

    ytop = y + gap            # y of the Phase-07 box
    ybar = ytop - 4.5         # horizontal branch line
    arrow(ax, (12.5, ytop), (12.5, ybar), color=MUTED, ls=(0, (3, 2)), style="-")
    arrow(ax, (12.5, ybar), (66, ybar), color=MUTED, ls=(0, (3, 2)), style="-")
    ybox = ybar - 9.0
    arrow(ax, (24, ybar), (24, ybox + 6.4), color=MUTED, ls=(0, (3, 2)))
    arrow(ax, (66, ybar), (66, ybox + 6.4), color=MUTED, ls=(0, (3, 2)))
    box(ax, 6, ybox, 36, 6.4, "08  Chemistry-aware learning\nDEFERRED",
        fc="#f9fafb", ec=MUTED, ls=(0, (3, 2)), fs=7.8)
    box(ax, 48, ybox, 36, 6.4, "09  Targeted corpus expansion\nDEFERRED",
        fc="#f9fafb", ec=MUTED, ls=(0, (3, 2)), fs=7.8)
    footer(ax, "Phase 07 has a double dependency: Phase 06's engine AND Phase 00's frozen splits,\n"
               "metrics, and criteria — the yardstick is fixed before anything is built with it.", y=1.0)
    save(fig, "fig07_phase_roadmap")


# ── 08 failure taxonomy ────────────────────────────────────────────────────────
def fig08():
    fig, ax = canvas(10.6, 7.0, "8 — Failure taxonomy: why the fine-resolution ceiling is a representation problem",
                     "V6.3 revalidation, n = 167 analytes. Source: results/v6_rebuild/v63_ontology_revalidation/tables/v63_waterfall.csv")
    ax.text(50, 87, "MSS layer — 54 failures under the old ontology, decomposed",
            ha="center", fontsize=8.6, weight="bold", color=INK)
    cats = [("resolved by\nbetter fine labels", 7, "#dbeafe", LEARN),
            ("near-miss\n(right broad class)", 16, "#e5e7eb", MUTED),
            ("TRUE REPRESENTATION\nERRORS", 31, "#fecaca", BAD),
            ("new failures\nintroduced", 8, "#fef3c7", WARN)]
    x, total_w = 8.0, 84.0
    denom = sum(c[1] for c in cats)
    for name, n, fc, ec in cats:
        w = total_w * n / denom
        box(ax, x, 68, w, 12, "", fc=fc, ec=ec, lw=1.2)
        ax.text(x + w / 2, 75.5, str(n), ha="center", fontsize=14, weight="bold", color=ec)
        ax.text(x + w / 2, 71.0, f"{100*n/54:.0f}%", ha="center", fontsize=8, color=ec)
        ax.text(x + w / 2, 65.0, name, ha="center", va="top", fontsize=7.3,
                color=INK, linespacing=1.35,
                weight="bold" if "TRUE" in name else "normal")
        x += w

    ax.text(50, 53,
            "57.4% of MSS failures survived ontology cleanup as true representation errors.\n"
            "These spectra are not mislabelled and not near-misses — the coordinate system does not separate them.",
            ha="center", fontsize=8.2, color=BAD, linespacing=1.5)

    rows = [("coord (24 components)", 57, 9, 22, 26, 11),
            ("MSS (17 motifs)", 54, 7, 16, 31, 8),
            ("themes (6)", 66, 10, 21, 35, 7),
            ("systems (4)", 82, 15, 18, 49, 5)]
    hdr = ["level", "old\nfailures", "resolved\nby labels", "near-miss\nbroad", "TRUE\nerrors", "newly\nbroken"]
    cx = [8, 34, 45, 57, 69, 82]
    for h, x in zip(hdr, cx):
        ax.text(x, 44, h, fontsize=7.0, color=MUTED, va="top",
                ha="left" if x == 8 else "center", linespacing=1.3)
    y = 36
    for r in rows:
        bold = r[0].startswith("MSS")
        ax.text(cx[0], y, r[0], fontsize=7.4, color=INK, va="center",
                weight="bold" if bold else "normal")
        for v, x in zip(r[1:], cx[1:]):
            ax.text(x, y, str(v), fontsize=7.4, ha="center", va="center",
                    color=BAD if (bold and v == 31) else INK,
                    weight="bold" if (bold and v == 31) else "normal")
        y -= 5.0

    box(ax, 6, 3, 88, 13.5,
        "WHY ONTOLOGY CLEANUP WAS NOT THE FIX   (McNemar, n = 167)\n"
        "coord old vs fine:  Δ = −0.012, p = 0.82  (n.s.)          MSS old vs fine:  Δ = −0.006, p = 1.00  (n.s.)\n"
        "theme old vs fine:  Δ = +0.018, p = 0.63  (n.s.)          system old vs fine:  Δ = +0.060, p = 0.041\n"
        "← the system level is the ONLY significant fine-level gain",
        fc="#f9fafb", ec=LINE, fs=7.2)
    save(fig, "fig08_failure_taxonomy")


# ── 09 atlas structure ─────────────────────────────────────────────────────────
def fig09():
    fig, ax = canvas(10.6, 8.2, "9 — Atlas asset structure: V5 (frozen) vs V7 (target)",
                     "V7's fingerprint must cover every behaviour-determining layer, not only the projection basis.")
    ax.text(24, 89, "V5 FROZEN ATLAS", ha="center", fontsize=9.5, weight="bold", color=FROZEN)
    ax.text(24, 85.6, "fingerprint 09ed804a40836f4a05a91ba10900cded", ha="center",
            fontsize=6.8, color=MUTED)
    v5 = ["manifold_components.npz\n(NMF basis H, 24×676)  ← HASHED",
          "manifold.json  (metadata, corpus card)",
          "component_registry_v1.json",
          "component_theme_weights_v1.json  (24×13)",
          "biochemical_ontology_v2.yaml  (13 themes)",
          "mss_motifs_v1.yaml  (13 curated motifs)",
          "reference_normalization_v1.json",
          "reference_support.npz  (OOD)",
          "MANIFEST.json"]
    y = 76
    for i, t in enumerate(v5):
        box(ax, 4, y, 40, 6.0 if i == 0 else 4.6,
            t, fc="#f3f4f6" if i == 0 else "white", ec=FROZEN, fs=7.2)
        y -= (7.0 if i == 0 else 5.6)

    ax.text(75, 89, "V7 TARGET ATLAS", ha="center", fontsize=9.5, weight="bold", color=LEARN)
    ax.text(75, 85.6, "fingerprint over 7 layers, in fixed order", ha="center",
            fontsize=6.8, color=MUTED)
    v7 = [("preprocessing_spec_v1.json", True),
          ("csm_dictionary_v1.npz  (M×676)  [PROJECTION]", True),
          ("lsm_dictionary_v1.npz  (per class)  [EVIDENCE]", True),
          ("theme_membership_v1.npz  (S, M×K)", True),
          ("theme_registry_v1.yaml  (K themes)", True),
          ("bsv_reference_v1.json", True),
          ("bsv_ood_support_v1.npz", True),
          ("csm_registry_v1.json  ·  lsm_registry_v1.json", False),
          ("canonical_analytes_v1.csv  ·  PROVENANCE.json", False),
          ("bsv_pca_v1.npz  (visualisation only)", False),
          ("MANIFEST.json  (+ per-layer hashes)", False)]
    y = 76
    for t, hashed in v7:
        box(ax, 56, y, 40, 4.6, t, fc="#dbeafe" if hashed else "white",
            ec=LEARN if hashed else MUTED, fs=7.0)
        ax.text(97.5, y + 2.3, "#" if hashed else "", fontsize=8, color=LEARN, va="center")
        y -= 5.6

    ax.text(75, y + 2.6, "#  = in the atlas fingerprint", fontsize=6.8, color=LEARN, ha="center")

    box(ax, 4, 2, 92, 14,
        "WHY THE V5 SCHEME DOES NOT GENERALISE\n"
        "V5 hashes ONLY the NMF basis. That was adequate because the basis effectively WAS the atlas —\n"
        "the overlay layers were thin, curated, and separately versioned.\n"
        "It is not adequate for V7: an atlas with an identical CSM basis but a different S produces different BSVs,\n"
        "so a basis-only fingerprint would make two behaviourally different atlases indistinguishable.",
        fc="#fffbeb", ec=WARN, fs=7.2)
    save(fig, "fig09_atlas_structure")


# ── 10 DART trajectory ─────────────────────────────────────────────────────────
def fig10():
    fig, ax = canvas(10.6, 6.6, "10 — Future DART concept: BSV trajectories in a fixed coordinate system",
                     "Conceptual schematic — no data. Illustrates why the BSV must be absolute, continuous, and non-negative.")
    ox, oy, w, h = 8.0, 22.0, 44.0, 56.0
    ax.add_patch(Rectangle((ox, oy), w, h, fc="#f9fafb", ec=LINE, lw=1.0, zorder=1))
    ax.text(ox + w / 2, oy + h + 3.5, "fixed BSV coordinate system  (frozen atlas)",
            ha="center", fontsize=7.8, color=MUTED)
    ax.text(ox - 2.0, oy + h / 2, "theme axis  t_j", rotation=90, ha="center",
            va="center", fontsize=7.4, color=MUTED)
    ax.text(ox + w / 2, oy - 3.5, "theme axis  t_i", ha="center", fontsize=7.4, color=MUTED)

    # a schematic trajectory: monotone-ish drift with curvature, hand-specified (no RNG)
    pts = [(16, 32), (21, 37), (27, 43), (33, 50), (38, 58), (41, 66), (42, 71)]
    for a, b in zip(pts[:-1], pts[1:]):
        arrow(ax, a, b, color=INFER, lw=1.6, ms=8)
    for i, (px, py) in enumerate(pts):
        ax.plot(px, py, "o", ms=5.5, color=INFER, zorder=4)
        ax.text(px + 1.6, py - 1.8, f"t{i}", fontsize=6.5, color=MUTED)
    ax.plot(pts[0][0], pts[0][1], "o", ms=8, color=NEUTRAL, zorder=5)
    ax.text(pts[0][0] - 1.5, pts[0][1] - 3.2, "baseline\nBSV(E, t₀)", fontsize=6.8,
            color=NEUTRAL, ha="center", va="top", linespacing=1.3)
    arrow(ax, pts[0], pts[-1], color=WARN, lw=1.2, ls=(0, (4, 2)), style="-")
    ax.text(30, 62, "ΔBSV\n(derived, signed)", fontsize=6.9, color=WARN,
            ha="center", linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

    box(ax, 57, 56, 39, 22,
        "EACH POINT IS AN ABSOLUTE BSV\n\n"
        "BSV(E, t) = Sᵀ c(x(E,t)) ∈ ℝ₊^K\n\n"
        "A trajectory is only meaningful because every\n"
        "point sits in the SAME fixed frame. The frame is\n"
        "frozen before the experiment starts.",
        fc="#dcfce7", ec=GOOD, fs=7.5)

    box(ax, 57, 33, 39, 21,
        "DERIVED QUANTITIES  (none is a BSV)\n\n"
        "displacement   BSV(tᵢ) − BSV(t₀)      signed\n"
        "velocity           d BSV / dt                  signed\n"
        "path length     Σ ‖BSV(tᵢ₊₁) − BSV(tᵢ)‖\n"
        "direction         normalised displacement",
        fc="white", ec=NEUTRAL, fs=7.3)

    box(ax, 57, 12, 39, 19,
        "WHAT DART MUST NOT DO\n\n"
        "✗ trajectory-specific re-fitting\n"
        "✗ per-run normalisation  (breaks cross-run\n     comparability — the point of DART)\n"
        "✗ hard classification per time point\n"
        "✗ cohort standardisation inside a trajectory",
        fc="#fee2e2", ec=BAD, fs=7.3)

    footer(ax, "Requirements this places on V7: absolute BSV · fixed frozen frame · continuous non-negative "
               "activations · per-point uncertainty · support-aware uncertainty inflation.", y=4.0)
    save(fig, "fig10_dart_trajectory")


if __name__ == "__main__":
    print(f"writing planning figures to {OUT}")
    for f in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10):
        f()
    print("done — 10 figures (SVG vector + PNG preview)")
