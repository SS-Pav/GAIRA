#!/usr/bin/env python3
"""GAIRA V7 — architecture diagrams, regenerated after Phase 05.

Five publication-quality figures, written as **SVG** (vector, as requested for this
documentation pass) and **PNG** at 200 dpi (raster, for embedding in the PDF report and for
consistency with the per-phase figure convention).

    python GAIRA_v7_rebuild/results/figures/planning/make_architecture_figures.py

Documentation only. Reads nothing, computes nothing, fits nothing. Every number that appears in
a diagram is a literal quoted from a committed phase table and is named in the caption of the
master report.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
INK, MUTED, RULE = "#1a1a1a", "#6b7280", "#d1d5db"
BLUE, GREEN, AMBER, RED = "#2563eb", "#15803d", "#b45309", "#b91c1c"
PURPLE, TEAL, SLATE = "#7c3aed", "#0f766e", "#475569"
# Fill / edge pairs by status.
ACTIVE = ("#ecfdf5", GREEN)
FROZEN = ("#eff6ff", BLUE)
PLANNED = ("#f5f3ff", PURPLE)
ARCHIVED = ("#f1f5f9", "#94a3b8")
FAILED = ("#fef2f2", RED)
NOTE = ("#fffbeb", AMBER)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "svg.fonttype": "path", "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.2})


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", format="svg")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  {name}.svg + .png")


def canvas(w, h, title, subtitle=None):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.0, 1.035, title, fontsize=13, weight="bold", color=INK,
            transform=ax.transAxes)
    if subtitle:
        ax.text(0.0, 1.005, subtitle, fontsize=8.8, color=MUTED, transform=ax.transAxes)
    return fig, ax


def box(ax, x, y, w, h, text, style=ACTIVE, fs=8.0, weight="normal", lw=1.2, alpha=1.0):
    fc, ec = style
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008", fc=fc, ec=ec,
                                lw=lw, alpha=alpha, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            weight=weight, linespacing=1.4, zorder=3)


def arrow(ax, p0, p1, col=SLATE, lw=1.3, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=12, color=col,
                                 lw=lw, linestyle=ls, shrinkA=1, shrinkB=1, zorder=1))


def side(ax, x, y, text, col=MUTED, fs=7.2, ha="left"):
    ax.text(x, y, text, fontsize=fs, color=col, ha=ha, va="center", linespacing=1.35)


def legend(ax, y=-0.06, items=None):
    items = items or [("frozen / complete", GREEN), ("planned", PURPLE),
                      ("archived", "#94a3b8"), ("visualisation only", AMBER)]
    h = [Line2D([], [], marker="s", ms=8, ls="", mfc=c, mec=c, label=l) for l, c in items]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, y), ncol=len(items),
              frameon=False, fontsize=7.6)


# ── FIG 1 · Learning mode ────────────────────────────────────────────────────
def fig_learning():
    fig, ax = canvas(12.6, 9.0, "Figure A1 · GAIRA V7 learning mode",
                     "Where every object is learned, and what is frozen. Offline only — this "
                     "path never runs at inference.")
    X, W, H = 0.155, 0.46, 0.066
    rows = [
        (0.905, "Raw Raman grounding corpus\n375 spectra · 154 canonical molecules · pure Raman",
         FROZEN, "input", "corpus"),
        (0.822, "canonical preprocessing\nasLS → SG → L2 · 450–1800 cm⁻¹ · 2.0 step · 676 bins",
         FROZEN, "PHASE 00 ✔", "spec — unchanged\nfrom V5"),
        (0.739, "canonical IDs · quality · 5 grouped CV folds · 16-class partition\n"
                "V5 control baseline · FROZEN success criteria",
         ACTIVE, "PHASE 00 ✔", "learned: nothing\nfrozen: the yardstick"),
        (0.656, "balanced reference construction\none canonical molecule = one reference unit",
         ACTIVE, "PHASE 01 ✔", "derived"),
        (0.573, "16 class-local non-negative decompositions\nadaptive k_c · repeated resampling and seeds",
         ACTIVE, "PHASE 01 ✔", "LEARNED  W_c, H_c"),
        (0.490, "50 stable Local Spectral Motifs (LSMs)\nHungarian alignment → recurrence → threshold",
         ACTIVE, "PHASE 01 ✔", "FROZEN\n208482d6…"),
        (0.407, "seven-feature Consensus Spectral Graph\nnull calibration → sweep → consensus → falsification",
         ACTIVE, "PHASE 02 ✔", "derived"),
        (0.324, "49 Consensus Spectral Motifs (CSMs)\n1 merge of 4 proposed · 48 singletons · provenance",
         ACTIVE, "PHASE 02 ✔", "FROZEN  0b4aa550…\n★ CANONICAL"),
        (0.241, "frozen projection engine\nreference bank · metric · calibrator · rejection",
         ACTIVE, "PHASE 05 ✔", "FROZEN\n20d8bd99…"),
        (0.158, "Chemistry Evidence map  E ∈ ℝ₊⁴⁹ˣ¹⁶  + calibrator\n"
                "training folds only · unassigned mass reported",
         PLANNED, "PHASE 06 ▶", "TO BE LEARNED"),
        (0.075, "BSV2 programme dictionary  P ∈ ℝ₊^{K×16}\n"
                "hierarchical NMF over Chemistry Evidence ONLY",
         PLANNED, "PHASE 07 ▶", "TO BE LEARNED"),
    ]
    for i, (y, txt, st, phase, note) in enumerate(rows):
        bold = "bold" if "CANONICAL" in note else "normal"
        box(ax, X, y, W, H, txt, st, 7.2, bold)
        side(ax, X - 0.012, y + H / 2, phase, INK, 7.4, "right")
        side(ax, X + W + 0.012, y + H / 2, note, MUTED, 6.8)
        if i:
            arrow(ax, (X + W / 2, rows[i - 1][0]), (X + W / 2, y + H))
    box(ax, 0.775, 0.395, 0.215, 0.088,
        "PHASE 02.5 ✔\nlatent geometry — continuum,\nnot islands · K = 2 hydrophobic/polar\n"
        "VISUALISATION AND PRIOR ONLY", NOTE, 6.8)
    arrow(ax, (X + W, 0.357), (0.775, 0.435), AMBER, 1.0, ls=(0, (4, 2)))
    box(ax, 0.775, 0.215, 0.215, 0.145,
        "ARCHIVED ON EVIDENCE\n\n03 themes (K = 5) → A-13\n04 BSV (K = 4) → A-14\n"
        "04.5 Meta components → A-15\n05 11 evidence axes → A-16", ARCHIVED, 6.8)
    arrow(ax, (X + W, 0.330), (0.775, 0.300), "#94a3b8", 1.0, ls=(0, (4, 2)))
    box(ax, 0.02, 0.010, 0.96, 0.046,
        "FROZEN V7 ATLAS BUNDLE — one fingerprint over every layer:   preprocessing spec · LSM "
        "dictionaries · CSM basis · reference bank · calibrator ·\nrejection thresholds · "
        "chemistry-evidence map · BSV2 programmes · provenance · manifest", FROZEN, 7.2)
    legend(ax, -0.030)
    save(fig, "A1_learning_mode")


# ── FIG 2 · Inference mode ───────────────────────────────────────────────────
def fig_inference():
    fig, ax = canvas(11.0, 6.6, "Figure A2 · GAIRA V7 inference mode",
                     "Projection and arithmetic only. Nothing is fitted; no output depends on "
                     "which other spectra are in the batch.")
    box(ax, 0.02, 0.79, 0.17, 0.10, "new Raman spectrum\nwavenumbers + intensities\n+ metadata",
        FROZEN, 7.6, "bold")
    box(ax, 0.22, 0.79, 0.19, 0.10,
        "canonical preprocessing\ncrop → resample → asLS\n→ SG → L2", FROZEN, 7.6)
    box(ax, 0.44, 0.79, 0.20, 0.10,
        "NNLS onto the frozen\nCSM basis\nc = argmin‖x − cᵀ·CSM‖², c ≥ 0", ACTIVE, 7.6)
    box(ax, 0.67, 0.79, 0.16, 0.10, "CSM activation\nc(x) ∈ ℝ₊⁴⁹\n★ CANONICAL",
        ACTIVE, 7.8, "bold")
    for a, b in ((0.19, 0.22), (0.41, 0.44), (0.64, 0.67)):
        arrow(ax, (a, 0.84), (b, 0.84))
    box(ax, 0.86, 0.79, 0.12, 0.10, "open-set\nrejection\nAUROC 0.921", NOTE, 7.4)
    arrow(ax, (0.83, 0.84), (0.86, 0.84), AMBER)

    branches = [
        (0.02, "analyte retrieval", "154 reference vectors\ncosine · calibrated\ntop-1 0.605 · top-5 0.795", ACTIVE),
        (0.21, "chemistry-class", "16 fine classes\n0.845 top-1 on\nUNSEEN molecules", ACTIVE),
        (0.40, "evidence profile", "11 declared axes\n7 of 11 grounded\nARCHIVED A-16", ARCHIVED),
        (0.59, "provenance", "axis → CSM → LSM →\nmolecule → spectra\n3,133 chains · 0 broken", ACTIVE),
        (0.78, "uncertainty", "residual · margin ·\nentropy · rejection\nECE 0.130", ACTIVE),
    ]
    for x, title, sub, st in branches:
        box(ax, x, 0.545, 0.185, 0.145, f"{title}\n\n{sub}", st, 7.3)
        arrow(ax, (0.75, 0.79), (x + 0.0925, 0.69), RULE, 0.9)
    ax.plot([0.0, 1.0], [0.495, 0.495], color=RULE, lw=1.0)
    ax.text(0.0, 0.515, "CURRENT  (Phase 05)", fontsize=8.2, weight="bold", color=GREEN)
    ax.text(0.0, 0.470, "PLANNED  (gated)", fontsize=8.2, weight="bold", color=PURPLE)

    # Planned row, column-aligned under the branch it consumes so no arrow crosses a heading.
    box(ax, 0.21, 0.315, 0.20, 0.100,
        "Chemistry Evidence\ne(x) = Γ(Eᵀc(x)) ∈ ℝ₊¹⁶\nΣ e ≤ 1, unassigned reported",
        PLANNED, 7.2, "bold")
    box(ax, 0.45, 0.315, 0.16, 0.100,
        "BSV2\nfrozen NNLS of e(x)\nonto P → b(x) ∈ ℝ₊^K", PLANNED, 7.2, "bold")
    box(ax, 0.655, 0.315, 0.30, 0.100,
        "hierarchical molecular retrieval\nsoft chemistry prior → class-conditioned →\n"
        "prototype + residual → ranked top-k", PLANNED, 7.2, "bold")
    arrow(ax, (0.3025, 0.545), (0.3025, 0.415), PURPLE, 1.2)      # chemistry-class → evidence
    arrow(ax, (0.41, 0.372), (0.45, 0.372), PURPLE)               # evidence → BSV2
    arrow(ax, (0.41, 0.332), (0.655, 0.332), PURPLE, 1.0, ls=(0, (4, 2)))   # evidence → retrieval
    arrow(ax, (0.61, 0.372), (0.655, 0.372), PURPLE, 1.0, ls=(0, (4, 2)))   # BSV2 → retrieval
    side(ax, 0.805, 0.437, "+ CSM activation c(x) — read directly, not through the chemistry layer",
         PURPLE, 6.9, "center")
    side(ax, 0.805, 0.283, "chemistry is a SOFT prior, never a filter: a class error must stay recoverable",
         PURPLE, 6.9, "center")
    side(ax, 0.31, 0.283, "evidence supporting no class is reported\nas unassigned mass, never redistributed",
         PURPLE, 6.9, "center")

    box(ax, 0.02, 0.115, 0.46, 0.085,
        "PROHIBITED at inference\nNMF · PCA · UMAP / t-SNE · clustering · community detection ·\n"
        "ontology optimisation · threshold tuning on the batch ·\n"
        "ANY operation whose result depends on the batch", FAILED, 7.0)
    box(ax, 0.52, 0.115, 0.46, 0.085,
        "PERMITTED at inference\ncanonical preprocessing · NNLS on a frozen dictionary ·\n"
        "multiplication by frozen matrices · frozen calibrator ·\n"
        "distances to frozen references · frozen visualisation transform", ACTIVE, 7.0)
    box(ax, 0.02, 0.015, 0.96, 0.075,
        "domain-context interpretation — serum / EV / plasma / tissue / pathogen\n"
        "STRICTLY DOWNSTREAM. No domain object is reachable from any module above this line, "
        "and nothing here feeds back upstream.", NOTE, 7.4)
    arrow(ax, (0.50, 0.115), (0.50, 0.090), AMBER)
    save(fig, "A2_inference_mode")


# ── FIG 3 · Validation pipeline ──────────────────────────────────────────────
def fig_validation():
    fig, ax = canvas(13.0, 9.2, "Figure A3 · GAIRA V7 validation pipeline",
                     "Every remaining phase ends with the same four-part gate. A gate can send "
                     "the phase back or change the architecture.")
    box(ax, 0.02, 0.885, 0.46, 0.085,
        "FROZEN IN PHASE 00 — never adjusted afterwards (P-13)\n"
        "154 canonical molecules · 5 folds grouped by canonical_id · v7_harness_v1\n"
        "V5 control baseline · Tier-1 success criteria S-01 … S-07", FROZEN, 7.2)
    box(ax, 0.52, 0.885, 0.46, 0.085,
        "TWO EVALUATION SPLITS — the distinction Phase 04 lacked\n"
        "Split A: molecule present in the bank → molecule top-k is defined\n"
        "Split B: molecule absent → molecule top-k is UNDEFINED, not zero", NOTE, 7.2)

    gates = [
        (0.740, "DG-06", "Chemistry\nEvidence Layer",
         "clearly exceeds the archived 11-axis profile (0.664 → ≥ 0.744, significant)\n"
         "informativeness floor ≥ 0.50 of the CSM layer · calibration informative\n"
         "R-01 class-agnostic control reported, whatever it shows"),
        (0.595, "DG-07", "BSV2\nDiscovery",
         "informativeness floor PRE-REGISTERED before the K sweep\n"
         "K on a published Pareto frontier over eight axes · never reconstruction alone\n"
         "stability gains count ONLY after the floor is cleared"),
        (0.450, "DG-08", "Hierarchical\nMolecular Retrieval",
         "molecule top-1 > 0.605, significant · no class silently harmed\n"
         "rejection not degraded (AUROC ≥ 0.921)\n"
         "hard-filter negative control reported"),
        (0.305, "DG-09", "V5 head-to-head\n[DECISION]",
         "the frozen Tier-1 criteria, UNADJUSTED, under v7_harness_v1\n"
         "outcomes: replace · partial adoption (justified separately) · retain V5"),
    ]
    for i, (y, gid, name, req) in enumerate(gates):
        box(ax, 0.02, y, 0.155, 0.105, f"{gid}\n{name}", PLANNED, 7.4, "bold")
        box(ax, 0.195, y, 0.44, 0.105, req, FROZEN, 7.0)
        for j, (lab, st, col) in enumerate((("Proceed", ACTIVE, GREEN),
                                            ("Repeat", NOTE, AMBER),
                                            ("Redesign", FAILED, RED))):
            box(ax, 0.665 + j * 0.112, y + 0.020, 0.104, 0.065, lab, st, 7.4, "bold")
            arrow(ax, (0.635, y + 0.0525), (0.665 + j * 0.112, y + 0.0525), col, 0.8)
        if i:
            arrow(ax, (0.0975, gates[i - 1][0]), (0.0975, y + 0.105), GREEN)

    box(ax, 0.665, 0.140, 0.315, 0.145,
        "WHAT THE THREE OUTCOMES MEAN\n"
        "Proceed — all three validations pass\n"
        "Repeat — a DEFECT was found; re-run after the fix.\nNot licence to try a new threshold\n"
        "Redesign — the layer is sound, the evidence is not.\nArchive it, rewrite the plan", FROZEN, 6.6)
    box(ax, 0.02, 0.140, 0.62, 0.145,
        "THE FOUR PARTS OF EVERY GATE\n"
        "1 · SCIENTIFIC — did the layer do what it claimed, on held-out data,\n"
        "     against a PRE-REGISTERED threshold?\n"
        "2 · ENGINEERING — deterministic · batch-independent · no inference-time fitting ·\n"
        "     fingerprints verified · tests pass\n"
        "3 · ARCHITECTURE — non-negativity · provenance intact · layer isolation ·\n"
        "     no upstream artefact modified · P-18 respected\n"
        "4 · DECISION — Proceed / Repeat / Redesign, recorded in PHASE_STATE.json",
        ACTIVE, 6.6)
    box(ax, 0.02, 0.015, 0.96, 0.110,
        "P-18 — STABILITY WITHOUT INFORMATIVENESS IS NOT EVIDENCE\n"
        "Four times in V7 a consistency metric was maximised by an output that said nearly the "
        "same thing about every spectrum:\n"
        "Phase 03 softmax themes  ·  Phase 04 the same mode promoted into the engine\n"
        "Phase 04.5 Meta Components (0.185 of CSM information, won every stability axis)  ·  "
        "Phase 05 ECE-optimal constant calibrator (0.605 for every spectrum)\n"
        "A fifth instance is expected. Every remaining gate declares its floor BEFORE the "
        "measurement.", FAILED, 6.9)
    save(fig, "A3_validation_pipeline")


# ── FIG 4 · Legacy vs current vs future ──────────────────────────────────────
def fig_legacy_vs_current():
    fig, ax = canvas(11.0, 6.4, "Figure A4 · Legacy, current and future architecture",
                     "The legacy path is preserved, not deleted. Every retirement is an "
                     "evidenced decision, not a change of taste.")
    cols = [(0.02, "LEGACY — archived", ARCHIVED, "#64748b"),
            (0.355, "CURRENT — active", ACTIVE, GREEN),
            (0.69, "FUTURE — planned", PLANNED, PURPLE)]
    for x, title, st, col in cols:
        ax.text(x + 0.145, 0.955, title, fontsize=9.6, weight="bold", color=col, ha="center")
    W = 0.29

    legacy = [(0.845, "spectrum", ARCHIVED), (0.755, "CSM (49)", ARCHIVED),
              (0.665, "soft themes (4)\nA-13 · class 0.405", ARCHIVED),
              (0.575, "BSV = Sᵀc (4)\nA-14 · effective rank 2.40", ARCHIVED),
              (0.485, "Meta Components (3)\nA-15 · 0.185 info retained", ARCHIVED),
              (0.395, "11 declared axes\nA-16 · class 0.664", ARCHIVED)]
    current = [(0.845, "spectrum", FROZEN), (0.755, "CSM (49)\n★ CANONICAL", ACTIVE),
               (0.665, "analyte retrieval\ntop-1 0.605 · top-5 0.795", ACTIVE),
               (0.575, "chemistry class\n0.845 on unseen molecules", ACTIVE),
               (0.485, "provenance · uncertainty\n0 broken chains · AUROC 0.921", ACTIVE)]
    future = [(0.845, "spectrum", FROZEN), (0.755, "CSM (49)\n★ CANONICAL", ACTIVE),
              (0.665, "Chemistry Evidence (16)\nDG-06", PLANNED),
              (0.575, "BSV2 — programmes (K)\nDG-07", PLANNED),
              (0.485, "hierarchical retrieval\nDG-08", PLANNED),
              (0.395, "V5 head-to-head\nDG-09 · frozen criteria", PLANNED)]
    for (x, _, _, col), rows in zip(cols, (legacy, current, future)):
        for i, (y, txt, st) in enumerate(rows):
            bold = "bold" if "★" in txt else "normal"
            box(ax, x, y, W, 0.072, txt, st, 7.3, bold)
            if i:
                arrow(ax, (x + W / 2, rows[i - 1][0]), (x + W / 2, y + 0.072),
                      col if i > 1 else SLATE, 1.1)

    ax.text(0.165, 0.355, "every layer above the CSM lost information", fontsize=7.6,
            color=RED, ha="center", style="italic")
    ax.text(0.50, 0.435, "no interpretive layer between\nrepresentation and retrieval",
            fontsize=7.6, color=GREEN, ha="center", style="italic")
    ax.text(0.835, 0.355, "one is put back — different label space,\nand a gate that can reject it",
            fontsize=7.6, color=PURPLE, ha="center", style="italic")

    ax.plot([0.02, 0.98], [0.30, 0.30], color=RULE, lw=1.0)
    ax.text(0.02, 0.265, "THE EVIDENCE THAT RETIRED THE LEGACY PATH — chemistry-class top-1 on "
            "molecules the atlas has never seen, identical frozen folds",
            fontsize=8.4, weight="bold", color=INK)
    bars = [("raw spectrum", 676, 0.608, "#94a3b8"), ("LSM", 50, 0.850, BLUE),
            ("CSM", 49, 0.855, GREEN), ("11 axes", 11, 0.664, "#94a3b8"),
            ("theme / BSV", 4, 0.405, "#94a3b8"), ("Meta comps", 3, 0.392, "#94a3b8")]
    x0, bw = 0.06, 0.125
    for i, (lab, dim, v, c) in enumerate(bars):
        x = x0 + i * 0.155
        ax.add_patch(FancyBboxPatch((x, 0.055), bw, v * 0.185, boxstyle="round,pad=0.002",
                                    fc=c, ec=c, alpha=0.85))
        ax.text(x + bw / 2, 0.055 + v * 0.185 + 0.017, f"{v:.3f}", ha="center", fontsize=7.6,
                weight="bold", color=c)
        ax.text(x + bw / 2, 0.028, f"{lab} · {dim}-d", ha="center", fontsize=7.0, color=INK)
    ax.plot([x0 - 0.01, 0.98], [0.055, 0.055], color=INK, lw=0.9)
    save(fig, "A4_legacy_current_future")


# ── FIG 5 · Object provenance map ────────────────────────────────────────────
def fig_objects():
    fig, ax = canvas(10.6, 5.6, "Figure A5 · Where every object comes from",
                     "Learned offline · derived deterministically · curated by a human · "
                     "visualised only. Nothing crosses from right to left.")
    heads = [("LEARNED — fitted offline", 0.04, GREEN), ("DERIVED — deterministic", 0.28, BLUE),
             ("CURATED — human, justified", 0.52, AMBER), ("VISUALISED — never an input", 0.76, PURPLE)]
    for t, x, c in heads:
        ax.text(x + 0.10, 0.93, t, fontsize=8.4, weight="bold", color=c, ha="center")
        ax.plot([x, x + 0.20], [0.915, 0.915], color=c, lw=1.2)
    cols = {
        0.04: [("LSM dictionaries H_c", "Phase 01", ACTIVE), ("CSM basis (49×676)", "Phase 02", ACTIVE),
               ("calibrator params", "Phase 05", ACTIVE), ("chemistry map E", "Phase 06 ▶", PLANNED),
               ("BSV2 dictionary P", "Phase 07 ▶", PLANNED), ("prior weight λ", "Phase 08 ▶", PLANNED),
               ("membership S", "Phase 03 · A-13", ARCHIVED), ("Meta H (3×49)", "Phase 04.5 · A-15", ARCHIVED)],
        0.28: [("CSM activation c(x)", "inference", ACTIVE), ("reference bank (154×49)", "Phase 05", ACTIVE),
               ("rejection channels", "Phase 05", ACTIVE), ("Chemistry Evidence e(x)", "Phase 06 ▶", PLANNED),
               ("BSV2 b(x)", "Phase 07 ▶", PLANNED), ("retrieval ranking", "Phase 08 ▶", PLANNED),
               ("BSV = Sᵀc", "Phase 04 · A-14", ARCHIVED), ("ΔBSV2, elevation", "analysis", PLANNED)],
        0.52: [("canonical molecule IDs", "Phase 00", ACTIVE), ("16-class partition", "Phase 00", ACTIVE),
               ("preprocessing spec", "Phase 00", ACTIVE), ("quality score q", "Phase 00", ACTIVE),
               ("v7_fine_16 ontology", "Phase 00", ACTIVE), ("success criteria", "Phase 00 · FROZEN", FROZEN),
               ("11 evidence axes", "Phase 05 · A-16", ARCHIVED), ("domain rules", "downstream", NOTE)],
        0.76: [("motif geometry", "Phase 02.5", NOTE), ("UMAP / PCA plots", "never inference", NOTE),
               ("radar charts", "Phase 05", NOTE), ("provenance waterfalls", "Phase 05", NOTE),
               ("cohort-standardised views", "not portable", NOTE)],
    }
    for x, items in cols.items():
        y = 0.845
        for name, where, st in items:
            box(ax, x, y, 0.20, 0.062, f"{name}\n{where}", st, 6.9)
            y -= 0.075
    box(ax, 0.04, 0.075, 0.92, 0.105,
        "THE ONE-WAY RULE\n"
        "Curated objects may organise a fit but never supervise it (P-06, amended).  Learned "
        "objects are frozen before inference (P-09).\n"
        "Visualised objects are never an input to interpretation, retrieval or scoring.  Domain "
        "context is applied last and never feeds back upstream.", FROZEN, 7.2)
    save(fig, "A5_object_provenance")


def main():
    print("[architecture figures]")
    fig_learning(); fig_inference(); fig_validation(); fig_legacy_vs_current(); fig_objects()
    print(f"[architecture figures] written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
