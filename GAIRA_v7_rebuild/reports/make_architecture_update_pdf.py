#!/usr/bin/env python3
"""GAIRA V7 — assemble V7_ARCHITECTURE_UPDATE_AFTER_PHASE05.pdf.

A supplementary-architecture document: what the architecture is now, what changed after Phase 05,
why, and the evidence for every change. Text pages are typeset here; the five architecture
diagrams are embedded from the committed PNGs so the PDF cannot drift from the figure script.

    python GAIRA_v7_rebuild/reports/make_architecture_update_pdf.py

Documentation only. Reads nothing scientific, computes nothing, fits nothing. Every number is a
literal quoted from a committed phase table, named in §Provenance.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGS = ROOT / "results" / "figures" / "planning"
OUT = HERE / "V7_ARCHITECTURE_UPDATE_AFTER_PHASE05.pdf"
PAGE = (8.27, 11.69)                      # A4 portrait for text; diagrams get landscape pages
LAND = (11.69, 8.27)

INK, MUTED, RULE = "#1a1a1a", "#6b7280", "#d1d5db"
GREEN, RED, PURPLE, BLUE, AMBER = "#15803d", "#b91c1c", "#7c3aed", "#2563eb", "#b45309"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

CAPTIONS = {
    "A1": "Figure A1 · Learning mode. Every fitted object and the phase that produced it. "
          "Green is complete and frozen; purple is planned; grey is archived on evidence; amber "
          "is visualisation only. Fingerprints are the committed values checked on every load.",
    "A2": "Figure A2 · Inference mode. Above the rule is what runs today; below it is what is "
          "planned, and every planned stage is gated. Note that molecular retrieval reads the "
          "CSM activation directly — the chemistry layer sits beside the representation, not "
          "instead of it. That is the structural lesson of the archived path.",
    "A3": "Figure A3 · Validation pipeline. Four remaining gates, each with the same four parts "
          "and the same three possible decisions. The P-18 banner records the four occasions on "
          "which a consistency metric was maximised by an uninformative output.",
    "A4": "Figure A4 · Legacy, current and future architecture side by side, with the evidence "
          "that retired the legacy path. Chemistry-class top-1 on molecules the atlas has never "
          "seen, identical frozen folds: the curve rises to the CSM layer and falls after it.",
    "A5": "Figure A5 · Object provenance. Learned, derived, curated and visualised objects, with "
          "the one-way rule that governs how they may interact.",
}


def _page(pdf, draw, size=PAGE):
    fig = plt.figure(figsize=size)
    draw(fig)
    pdf.savefig(fig)
    plt.close(fig)


def frame(fig):
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return ax


def head(ax, title, kicker=None):
    y = 0.945
    if kicker:
        ax.text(0.08, 0.962, kicker, fontsize=7.6, color=MUTED, weight="bold")
    ax.text(0.08, y, title, fontsize=15, color=INK, weight="bold")
    ax.plot([0.08, 0.92], [y - 0.018, y - 0.018], color=RULE, lw=1.0)
    return y - 0.045


def para(ax, y, text, fs=8.6, width=104, color=INK, lead=0.0175, indent=0.08, weight="normal"):
    for line in text.split("\n"):
        if not line.strip():
            y -= lead * 0.6
            continue
        for w in textwrap.wrap(line, width) or [""]:
            ax.text(indent, y, w, fontsize=fs, color=color, weight=weight)
            y -= lead
    return y


def bullet(ax, y, items, fs=8.4, width=98):
    for it in items:
        wrapped = textwrap.wrap(it, width)
        ax.text(0.085, y, "•", fontsize=fs, color=MUTED)
        for k, w in enumerate(wrapped):
            ax.text(0.105, y, w, fontsize=fs, color=INK)
            y -= 0.0165
        y -= 0.004
    return y


def table(ax, y, rows, widths, fs=7.6, header=True, lead=0.0165, x0=0.08, colors=None):
    for i, row in enumerate(rows):
        x = x0
        wt = "bold" if (header and i == 0) else "normal"
        col = INK if not colors or i == 0 else colors[i - 1]
        for cell, w in zip(row, widths):
            ax.text(x, y, str(cell), fontsize=fs, color=col, weight=wt)
            x += w
        if header and i == 0:
            ax.plot([x0, x0 + sum(widths) - 0.01], [y - 0.006, y - 0.006], color=RULE, lw=0.7)
            y -= 0.006
        y -= lead
    return y


def callout(ax, y, title, body, col=RED, h=None, fs=8.2):
    lines = sum(len(textwrap.wrap(l, 96)) or 1 for l in body.split("\n"))
    h = h or (0.020 + lines * 0.0165 + 0.016)
    ax.add_patch(FancyBboxPatch((0.075, y - h), 0.85, h, boxstyle="round,pad=0.006",
                                fc="#ffffff", ec=col, lw=1.2))
    ax.text(0.095, y - 0.020, title, fontsize=8.6, color=col, weight="bold")
    yy = y - 0.039
    for l in body.split("\n"):
        for w in textwrap.wrap(l, 96) or [""]:
            ax.text(0.095, yy, w, fontsize=fs, color=INK)
            yy -= 0.0165
    return y - h - 0.014


# ── pages ────────────────────────────────────────────────────────────────────
def p_cover(fig):
    ax = frame(fig)
    ax.text(0.08, 0.86, "GAIRA V7", fontsize=30, color=INK, weight="bold")
    ax.text(0.08, 0.805, "Architecture Update After Phase 05", fontsize=17, color=INK)
    ax.plot([0.08, 0.92], [0.775, 0.775], color=RULE, lw=1.3)
    ax.text(0.08, 0.740, "Supplementary architecture document · documentation only · "
            "no code, no algorithm, no artefact was changed", fontsize=9, color=MUTED)
    rows = [
        ("Date", "2026-08-06"),
        ("Branch", "gaira-v7-rebuild"),
        ("Phases complete", "00 · 01 · 02 · 02.5 · 03 · 04 · 04.5 · 05"),
        ("Phases archived on evidence", "03 themes · 04 BSV · 04.5 Meta Components · 05 evidence axes"),
        ("Canonical representation", "49-dimensional CSM activation vector"),
        ("Next phase", "06 — Chemistry Evidence Layer (16-d)"),
        ("", ""),
        ("V5 atlas", "09ed804a40836f4a05a91ba10900cded — unmodified, in production"),
        ("LSM registry", "208482d6f7178b5b8f16cace91be55b0"),
        ("CSM dictionary", "0b4aa550ccefed3edabdbde5bae11c8d"),
        ("Phase 05 engine", "20d8bd99ce71f45a125c6a2b1d719e51"),
        ("", ""),
        ("Corpus", "375 spectra · 154 canonical molecules · 16 fine chemistry classes"),
        ("Evaluation", "v7_harness_v1 · 5 folds grouped by canonical_id · criteria FROZEN in Phase 00"),
    ]
    y = 0.665
    for k, v in rows:
        if k:
            ax.text(0.08, y, k, fontsize=9, color=MUTED)
            ax.text(0.36, y, v, fontsize=9, color=INK)
        y -= 0.030
    callout(ax, 0.235, "THE ONE-SENTENCE SUMMARY",
            "Six phases of evidence show that GAIRA's abstraction stack pays up to the "
            "Consensus Spectral Motif layer and not after it; three independent attempts to "
            "build a layer above it all lost information, so the CSM activation vector becomes "
            "the canonical representation and the interpretable layer is rebuilt with a label "
            "space that matches the project's own frozen evaluation ontology.", GREEN)
    ax.text(0.08, 0.075, "Sources of record", fontsize=8.6, color=INK, weight="bold")
    ax.text(0.08, 0.055, "context/GAIRA_V7_ARCHITECTURE_STATUS_AFTER_PHASE05.md  ·  "
            "architecture/GAIRA_V7_TARGET_ARCHITECTURE.md", fontsize=7.6, color=MUTED)
    ax.text(0.08, 0.038, "plan/GAIRA_V7_REBUILD_PLAN.md  ·  plan/SUCCESS_CRITERIA.md  ·  "
            "plan/VALIDATION_AND_DECISION_RULES.md  ·  context/TERMINOLOGY_AND_DEFINITIONS.md",
            fontsize=7.6, color=MUTED)


def p_contents(fig):
    ax = frame(fig)
    y = head(ax, "Contents")
    items = [
        ("1", "The current architecture", "what runs today, and what is canonical"),
        ("2", "What changed", "four architectural retirements and three additions"),
        ("3", "Why it changed", "the abstraction curve, and the pattern behind three failures"),
        ("4", "Evidence for every change", "the full ledger, with the table each number came from"),
        ("5", "Legacy vs current vs future", "the three architectures, side by side"),
        ("6", "The remaining roadmap", "Phases 06 – 11 and their gates"),
        ("7", "Scientific risks", "what could still go wrong, stated before it does"),
        ("8", "Expected outcomes", "what each gate is likely to return, and what that would mean"),
        ("9", "Inconsistencies found and resolved", "including one that has stood since Phase 00"),
        ("A1–A5", "Architecture diagrams", "learning mode · inference mode · validation · legacy/current/future · provenance"),
    ]
    for num, title, sub in items:
        ax.text(0.09, y, num, fontsize=9.4, color=MUTED, weight="bold")
        ax.text(0.19, y, title, fontsize=9.8, color=INK, weight="bold")
        ax.text(0.19, y - 0.017, sub, fontsize=8.2, color=MUTED)
        y -= 0.052
    y -= 0.02
    callout(ax, y, "HOW TO READ THIS DOCUMENT",
            "Nothing here was recomputed. Every number is quoted from a committed phase table "
            "and the table is named. If a number in this document disagrees with a phase table, "
            "the phase table is correct and this document is wrong.\n"
            "Archived work is preserved, not deleted: every retired layer keeps its report, "
            "audit, figures, tables, artefacts and tests exactly as committed, and remains "
            "reproducible.", BLUE)


def p_current(fig):
    ax = frame(fig)
    y = head(ax, "1 · The current architecture", "WHAT RUNS TODAY")
    y = para(ax, y, "GAIRA V7 today is a frozen projection engine over a 49-dimensional "
             "consensus motif basis. A spectrum is preprocessed on the canonical grid, projected "
             "by non-negative least squares onto the frozen CSM dictionary, and everything "
             "downstream reads that one vector. Nothing is fitted at inference; no output "
             "depends on which other spectra are in the batch.")
    y -= 0.012
    y = table(ax, y, [
        ("layer", "dim", "what it does", "status"),
        ("canonical preprocessing", "676", "asLS → SG → L2, 450–1800 cm⁻¹", "frozen, unchanged from V5"),
        ("LSM dictionary", "50", "class-local motifs, 16 independent fits", "frozen"),
        ("CSM dictionary", "49", "consensus motifs, 1 merge of 4 proposed", "frozen · CANONICAL"),
        ("reference bank", "154×49", "one activation vector per molecule", "frozen"),
        ("calibrator", "—", "Dirichlet, selected on Brier", "frozen"),
        ("rejection channels", "8", "residual, sparsity, entropy, distance …", "frozen"),
        ("evidence axes", "11", "declared from band assignments", "ARCHIVED, pending DG-06"),
    ], [0.24, 0.08, 0.34, 0.24])
    y -= 0.015
    y = para(ax, y, "What the engine delivers, measured on the frozen folds:", weight="bold")
    y -= 0.006
    y = table(ax, y, [
        ("capability", "measure", "value", "source"),
        ("chemistry class, unseen molecule", "top-1 / top-3", "0.845 / 0.971", "Phase 05"),
        ("chemistry class, unseen molecule", "macro F1", "0.807", "Phase 05"),
        ("molecule identity, molecule present", "top-1 / top-5", "0.605 / 0.795", "Phase 05"),
        ("rejection of degraded spectra", "joint AUROC", "0.921", "Phase 05"),
        ("robustness, 7 × 5 perturbations", "class retention", "0.935 (raw 0.895)", "Phase 05"),
        ("provenance", "chains broken", "0 of 3,133", "Phase 05"),
        ("calibration", "ECE / discrimination", "0.130 / 0.891", "Phase 05"),
        ("reconstruction", "mean explained variance", "0.821", "Phase 05"),
    ], [0.30, 0.20, 0.20, 0.16])
    y -= 0.018
    y = callout(ax, y, "THE ONE NUMBER THAT DEFINES THE ARCHITECTURE",
                "Chemistry-class top-1 on molecules the atlas has never seen rises 0.608 (raw "
                "spectrum) → 0.850 (LSM) → 0.855 (CSM), then falls to 0.664 (11 declared axes), "
                "0.405 (themes / BSV) and 0.392 (Meta Components). The abstraction stack pays up "
                "to the CSM layer and not after it. Everything in §2 follows from that shape.",
                GREEN)
    para(ax, y, "One honest caveat carried from Phase 05: molecule identification is a "
         "shortlist, not an answer. Top-1 of 0.605 on a 154-way problem is far above chance and "
         "is not identification. The engine should be read as a chemistry-class instrument that "
         "also returns a ranked molecular shortlist.", color=MUTED)


def p_what_changed(fig):
    ax = frame(fig)
    y = head(ax, "2 · What changed", "FOUR RETIREMENTS, THREE ADDITIONS, ONE AMENDED PRINCIPLE")
    y = para(ax, y, "The change is not cosmetic and it is not a change of taste. Four objects "
             "were built, measured against pre-registered criteria, and retired. Each remains on "
             "disk, fingerprinted and reproducible.")
    y -= 0.010
    y = table(ax, y, [
        ("id", "retired", "measured", "verdict"),
        ("A-13", "soft biochemical themes (4)", "class top-1 0.855 → 0.405", "archived"),
        ("A-14", "BSV = Sᵀc (4)", "inherits A-13; eff. rank 2.40 of 4", "archived"),
        ("A-15", "Meta Components (3)", "0.185 of CSM information retained", "discarded"),
        ("A-16", "11 declared evidence axes", "class top-1 0.664 vs CSM 0.845", "archived*"),
        ("A-17", "SERS out-of-domain gate", "AUROC 0.548 on real Ag-SERS", "out of scope"),
    ], [0.08, 0.32, 0.34, 0.20])
    y = para(ax, y, "* A-16 is archived pending DG-06. If the Chemistry Evidence layer does not "
             "clearly exceed it, the 11-axis profile is reinstated.", fs=7.8, color=MUTED)
    y -= 0.014
    y = para(ax, y, "Added:", weight="bold")
    y -= 0.004
    y = table(ax, y, [
        ("id", "added", "input", "status"),
        ("A-19", "Chemistry Evidence (16-d)", "CSM activation", "Phase 06 · DG-06"),
        ("A-20", "BSV2 — biochemical programmes", "Chemistry Evidence ONLY", "Phase 07 · DG-07"),
        ("A-21", "hierarchical molecular retrieval", "CSM + chemistry prior", "Phase 08 · DG-08"),
    ], [0.08, 0.34, 0.32, 0.20])
    y -= 0.016
    y = para(ax, y, "Promoted:", weight="bold")
    y = para(ax, y, "A-08 — the 49-dimensional CSM activation vector becomes the canonical "
             "representation. It was always the projection basis; it is now also the object "
             "every downstream branch reads.")
    y -= 0.008
    y = para(ax, y, "Amended principles:", weight="bold")
    y = bullet(ax, y, [
        "P-06 — chemical class remains an organisational prior and must never supervise a local "
        "fit. It is now admissible as an intermediate probabilistic evidence coordinate carrying "
        "uncertainty. It remains inadmissible as a terminal hard label.",
        "P-07 — unchanged in substance; 'theme' becomes legacy vocabulary and the principle now "
        "governs Chemistry Evidence axes and BSV2 programmes.",
        "P-18 (NEW) — stability without informativeness is not evidence. No layer, mode, "
        "calibrator or model may be selected on a reproducibility, stability or calibration "
        "metric without first clearing a pre-registered informativeness floor.",
    ])
    y -= 0.006
    callout(ax, y, "WHAT DID NOT CHANGE",
            "Non-negativity at every layer. Learning offline, inference by projection. "
            "Determinism and fingerprint verification. Provenance as a first-class field. "
            "One canonical molecule = one reference unit. Class partitions the decomposition and "
            "never supervises it. The frozen V5 atlas remains the control arm, and the frozen "
            "Tier-1 success criteria are untouched.", BLUE)


def p_why(fig):
    ax = frame(fig)
    y = head(ax, "3 · Why it changed", "THE ABSTRACTION CURVE, AND A PATTERN ACROSS THREE FAILURES")
    y = para(ax, y, "Phase 04 measured six abstraction levels on identical frozen splits. The "
             "question it answered was not 'is the representation good' but 'where does the "
             "hierarchy stop paying'.")
    y -= 0.010
    y = table(ax, y, [
        ("level", "dim", "molecule top-1 (Split A)", "class top-1, molecule unseen"),
        ("L1 raw spectrum", "676", "0.790", "0.608"),
        ("L2 LSM", "50", "0.806", "0.850"),
        ("L3 CSM", "49", "0.799", "0.855"),
        ("L4 theme", "4", "0.553", "0.405"),
        ("L5 BSV", "4", "0.553", "0.405"),
        ("L6 geometry", "5", "0.495", "0.541"),
    ], [0.28, 0.10, 0.30, 0.28])
    y -= 0.014
    y = para(ax, y, "Three separate constructions then occupied the slot above L3, by three "
             "different mechanisms:")
    y = bullet(ax, y, [
        "discovered by archetypal analysis over CSM co-activation — Phase 03 themes, 0.405;",
        "discovered by non-negative factorisation of the CSM activation matrix — Phase 04.5 Meta "
        "Components, 0.392, retaining 0.185 of the CSM layer's information;",
        "declared from Raman band assignments with no fitting at all — Phase 05 evidence axes, "
        "0.664.",
    ])
    y = para(ax, y, "Declaring the axes rather than discovering them closed roughly half the "
             "gap. It did not close it. That is the strongest statement V7 has made about "
             "abstraction over this corpus: the loss is not an artefact of the discovery step.")
    y -= 0.012
    y = callout(ax, y, "THE PATTERN THAT MADE THIS HARD TO SEE — AND WHY P-18 EXISTS",
                "Two of the three losing layers scored BETTER than the CSM layer on stability. "
                "The Phase 03 softmax theme mode was the most reproducible option available. "
                "Meta Components won every stability axis outright — replicate consistency 0.980 "
                "vs 0.893, activation-stability AURC 0.975 vs 0.928, class-retrieval AURC 0.944 "
                "vs 0.936 — while retaining 0.185 of the information.\n"
                "A representation that says nearly the same thing about every spectrum is "
                "perfectly reproducible and useless. A fourth instance appeared in Phase 05 "
                "calibration, where the ECE-optimal calibrator reported 0.605 — the base rate — "
                "for every spectrum, with sharpness exactly 0.000.\n"
                "Each was caught only after an explicit informativeness constraint was added. "
                "P-18 makes the constraint a precondition rather than a rescue.", RED)
    y -= 0.004
    para(ax, y, "The SERS retirement has a different cause and is worth separating. Phase 04's "
         "out-of-domain gate failed at AUROC 0.548 because a non-negative Raman-motif dictionary "
         "reconstructs Ag-SERS of the same metabolites comfortably. That is a true finding about "
         "the representation, not a defect — and it was answering a question the project does "
         "not ask. Raman-only scope removes it.", color=MUTED)


def p_evidence(fig):
    ax = frame(fig)
    y = head(ax, "4 · Evidence for every change", "EVERY ROW CITES THE COMMITTED TABLE IT COMES FROM")
    y = para(ax, y, "Demonstrated — measured, and in most cases replicated across phases.")
    y -= 0.006
    y = table(ax, y, [
        ("#", "finding", "value", "source"),
        ("D-01", "corpus, folds, metrics, V5 control reproducible", "byte-identical atlas", "Phase 00"),
        ("D-04", "local motifs rarely describe the same phenomenon", "1 merge of 1,225 pairs", "Phase 02"),
        ("D-05", "motif space is a continuum with one bipartition", "modularity 0.620 vs null 0.070", "Phase 02.5"),
        ("D-06", "the abstraction stack pays up to the CSM layer", "0.608 → 0.850 → 0.855", "Phase 04"),
        ("D-08", "class inference generalises to unseen molecules", "0.845 top-1, macro F1 0.807", "Phase 05"),
        ("D-09", "CSM is more accurate AND more robust than raw", "0.845 / 0.935 vs 0.592 / 0.895", "Phase 05"),
        ("D-11", "provenance complete end to end", "0 broken of 3,133 chains", "Phase 05"),
        ("D-13", "degraded and structureless spectra are rejected", "joint AUROC 0.921", "Phase 05"),
    ], [0.07, 0.42, 0.30, 0.15])
    y -= 0.014
    y = para(ax, y, "Falsified — tested against pre-registered criteria and did not hold.")
    y -= 0.006
    y = table(ax, y, [
        ("#", "claim", "measurement", "verdict"),
        ("F-01", "themes are the semantic axes of the BSV", "0.855 → 0.405", "falsified"),
        ("F-02", "the BSV is the canonical output coordinate", "eff. rank 2.40 of 4", "falsified"),
        ("F-03", "second-order NMF recovers a programme layer", "0.185 information retained", "falsified"),
        ("F-04", "declaring the axes avoids the information loss", "0.664 vs 0.845", "partly falsified"),
        ("F-05", "the atlas detects real Ag-SERS as out-of-domain", "AUROC 0.548", "falsified"),
        ("F-06", "ECE ≤ 0.10 is achievable for the confidence", "0.130 informative / 0.080 constant", "falsified"),
    ], [0.07, 0.40, 0.30, 0.17])
    y -= 0.014
    y = para(ax, y, "Unknown — not measured by any phase so far.")
    y -= 0.006
    y = table(ax, y, [
        ("#", "unknown", "addressed by"),
        ("U-01", "does a 16-d chemistry layer preserve what 4-, 11- and 3-d layers did not?", "Phase 06"),
        ("U-02", "how much of the class signal is an imprint of the class partition? (R-01)", "Phase 06 control"),
        ("U-03", "does the engine reject genuinely novel chemistry?", "Phase 06 holdout"),
        ("U-04", "does BSV2 escape the pattern that discarded three predecessors?", "Phase 07"),
        ("U-05", "does a chemistry prior beat direct cosine on molecule identity?", "Phase 08"),
        ("U-06", "does V7 clear the frozen Tier-1 bar? NEVER MEASURED", "Phase 06 / 09"),
        ("U-08", "does the CSM layer earn its place over the LSM layer? 0.845 vs 0.848", "Phase 06"),
    ], [0.07, 0.60, 0.24])
    y -= 0.012
    callout(ax, y, "THE MOST UNCOMFORTABLE ROW IN THIS TABLE",
            "U-02. The Chemistry Evidence layer predicts the same sixteen classes that "
            "partitioned the Phase 01 local decompositions. Risk R-01 has been flagged since "
            "Phase 00 and has never been controlled. Until the class-agnostic decomposition "
            "control is run, 0.845 cannot be described as a property of the representation "
            "rather than of the partition. DG-06 makes the control mandatory.", AMBER)


def p_roadmap(fig):
    ax = frame(fig)
    y = head(ax, "6 · The remaining roadmap", "PHASES 06 – 11")
    y = table(ax, y, [
        ("phase", "objective", "gate", "the bar"),
        ("06 Chemistry Evidence", "frozen calibrated map ℝ₊⁴⁹ → ℝ₊¹⁶", "DG-06", "> 0.744 vs 0.664"),
        ("07 BSV2 Discovery", "hierarchical NMF over evidence only", "DG-07", "≥ 0.50 info floor"),
        ("08 Hierarchical Retrieval", "soft chemistry prior on molecule ID", "DG-08", "> 0.605 top-1"),
        ("09 V5 head-to-head", "the replacement decision", "DG-09", "frozen Tier-1"),
        ("10 Chemistry-aware learning", "deferred", "—", "—"),
        ("11 Corpus expansion", "deferred", "—", "—"),
    ], [0.26, 0.34, 0.12, 0.22])
    y -= 0.016
    for title, body, col in [
        ("PHASE 06 — Chemistry Evidence Layer",
         "Learn a frozen map E ∈ ℝ₊⁴⁹ˣ¹⁶ and a calibrator, on training folds only. Sixteen is "
         "not a hyperparameter: it is the size of the evaluation ontology frozen in Phase 00, "
         "which is also the label space of the frozen success criteria.\n"
         "Deliverables: top-1/3, macro F1, balanced accuracy, per-class precision and recall, "
         "confusion matrix, calibration (ECE, Brier, sharpness, discrimination), replicate "
         "consistency, cross-fold stability, noise robustness, radar examples, provenance "
         "chains, the R-01 control, and the first measurement against the frozen Tier-1 bar.\n"
         "DG-06 proceeds only if the layer CLEARLY exceeds the archived 11-axis profile.", PURPLE),
        ("PHASE 07 — BSV2 Discovery",
         "Hierarchical NMF over Chemistry Evidence ONLY — never CSM activations. That "
         "restriction is what makes BSV2 a different object from the discarded Meta Components: "
         "it factorises chemistry co-occurrence rather than motif usage.\n"
         "K ∈ {2,3,4,5,6,8,10,12,14}, selected on a Pareto frontier over eight axes: "
         "reconstruction, held-out chemistry prediction, programme stability, interpretability, "
         "mutual information, noise robustness, calibration, compression. Never reconstruction "
         "alone; never accuracy alone; the full frontier is published.\n"
         "Output: biochemical programmes — not themes, not manual mappings.", PURPLE),
        ("PHASE 08 — Hierarchical Molecular Retrieval",
         "Inputs: CSM activation AND Chemistry Evidence as a soft prior. Implement the soft "
         "prior, class-conditioned retrieval, prototype + residual scoring, hierarchical "
         "ranking, top-k with confidence, and conformal sets only if exchangeability is "
         "defensible under molecule-grouped splits.\n"
         "The hard constraint: a class error must be recoverable. At 0.845 class accuracy, a "
         "hard filter would remove the correct molecule for roughly one spectrum in six before "
         "scoring began. Any design that cannot recover from a class error is rejected at the "
         "gate regardless of its mean accuracy.", PURPLE),
        ("PHASE 09 — V5 head-to-head  [RETAINED]",
         "The Tier-1 criteria were frozen in Phase 00 and HAVE NEVER BEEN MEASURED. Phase 05's "
         "0.845 is a per-spectrum number on 5-fold grouped CV; the frozen bar is a per-analyte "
         "number at n = 167 under v7_harness_v1. Dropping the phase that measures the frozen bar "
         "would make the bar unreachable and nullify P-13 in practice.", GREEN),
    ]:
        y = callout(ax, y, title, body, col, fs=7.9)


def p_risks(fig):
    ax = frame(fig)
    y = head(ax, "7 · Scientific risks", "STATED BEFORE THE MEASUREMENT, NOT AFTER")
    y = callout(ax, y, "RISK 1 — BSV2 IS THE FOURTH ATTEMPT AT THE SAME ARCHITECTURAL POSITION",
                "Discovered themes, discovered meta-components and declared evidence axes all "
                "lost class information, and two of them scored better on stability while doing "
                "so. BSV2 differs in its input, and that is a real difference — but the position "
                "is identical and so is the failure mode to watch for.\n"
                "Mitigation: the informativeness floor is pre-registered before the K sweep, K is "
                "chosen on a published Pareto frontier, and 'BSV2 does not improve on Chemistry "
                "Evidence' is an accepted publishable outcome.", RED)
    y = callout(ax, y, "RISK 2 — CIRCULARITY IN THE CHEMISTRY EVIDENCE LAYER (R-01, U-02)",
                "The layer predicts the same sixteen classes that partitioned the Phase 01 "
                "fits. If the partition imprinted itself on the representation, part of 0.845 is "
                "an artefact of the experimental design.\n"
                "Mitigation: DG-06 makes the class-agnostic decomposition control mandatory, and "
                "the gap must be published whatever it shows.", RED)
    y = callout(ax, y, "RISK 3 — 'OPEN-SET REJECTION' DOES NOT YET MEAN NOVELTY (U-03)",
                "Every Phase 05 negative is corruption or structureless signal, four of six drawn "
                "from the same perturbation module used in the robustness study. The engine is "
                "shown to reject spectra corrupted by processes it was separately shown to be "
                "robust against. That is internally consistent and it is not evidence about novel "
                "chemistry.\n"
                "Mitigation: a held-out-chemistry-class experiment in Phase 06. Until it exists, "
                "the phrase used must be 'rejection of degraded and structureless spectra'.", AMBER)
    y = callout(ax, y, "RISK 4 — THE CSM LAYER MAY NOT EARN ITS PLACE OVER THE LSM LAYER (U-08)",
                "Class top-1 0.845 vs 0.848; retention 0.935 vs 0.923. Phase 02 accepted one "
                "merge of 1,225 candidate pairs, so 48 of 49 CSMs are single LSMs and the two "
                "layers are nearly the same object. Nothing measured so far justifies the CSM "
                "layer on performance; the justification is provenance and interpretability, and "
                "that should be stated rather than implied.", AMBER)
    y = callout(ax, y, "RISK 5 — THE CORPUS, NOT THE ARCHITECTURE, MAY BE THE BINDING CONSTRAINT (R-17)",
                "375 spectra and 154 molecules is a small corpus for a 154-way retrieval problem "
                "and for calibration with ten confidence bins. If the remaining phases deliver "
                "small gains, the honest conclusion may be that further architectural work is not "
                "the lever, and Phase 11 corpus expansion is the correct next step. The plan is "
                "explicitly able to reach that conclusion.", AMBER)
    para(ax, y, "A note on band shape (U-07). Amide I and cis C=C both sit near 1650 cm⁻¹ and "
         "window-based reasoning cannot separate them; this is the leading cause of the amide "
         "axis's failure in Phase 05. Band width, asymmetry and splitting carry assignment "
         "information the architecture currently discards. It is the highest-value addition not "
         "on the roadmap.", color=MUTED)


def p_expected(fig):
    ax = frame(fig)
    y = head(ax, "8 · Expected outcomes", "WHAT EACH GATE IS LIKELY TO RETURN, AND WHAT IT WOULD MEAN")
    y = para(ax, y, "Written in advance so that the outcome cannot be reinterpreted afterwards "
             "(P-12). These are expectations, not predictions to be defended.")
    y -= 0.012
    for gate, likely, meaning, col in [
        ("DG-06", "Likely to pass, with a caveat.",
         "The 16-d layer has a larger label space than the 11 declared axes and matches the "
         "evaluation ontology exactly, so exceeding 0.664 is plausible. The caveat is U-02: if "
         "the R-01 control shows a large gap, the gain is partly the class partition talking to "
         "itself, and the report must say so. Passing DG-06 with an uncontrolled R-01 would be "
         "worse than failing it.", GREEN),
        ("DG-07", "More likely to fail than to pass.",
         "Three predecessors in the same position all fell below the floor. BSV2's different "
         "input is a genuine difference, but 16 → K is a compression of an already-compressed "
         "object. If it fails, Chemistry Evidence becomes the terminal interpretable layer, "
         "Phase 08 proceeds without BSV2, and V7 will have four independent measurements saying "
         "the same thing about abstraction over this corpus. That is a stronger scientific "
         "result than a marginal pass.", RED),
        ("DG-08", "Genuinely uncertain, and the most interesting gate.",
         "Molecule top-1 of 0.605 with class accuracy of 0.845 is exactly the configuration in "
         "which a soft prior should help: the chemistry is known far better than the identity. "
         "The risk is not that the prior fails but that it helps on average while harming "
         "specific classes — which is why S-34 requires every harmed class to be named.", AMBER),
        ("DG-09", "Cannot be predicted, and should not be.",
         "The frozen bar is +8 points on fine-16 retrieval over V5's 0.6707. Phase 05's 0.845 "
         "is suggestive but is a different protocol on a different unit of analysis. The honest "
         "position is that the measurement has not been made. If V7 clears it, V5 is replaced; "
         "if not, V5 is retained and the negative result is published in full.", BLUE),
    ]:
        y = callout(ax, y, f"{gate} — {likely}", meaning, col, fs=8.0)
    y -= 0.004
    para(ax, y, "The V7 rebuild has so far produced more negative results than positive ones, "
         "and that is not a failure of the programme. Phase 02 rejected three of four proposed "
         "motif merges. Phase 03 rejected one of five themes. Phase 04.5 discarded its entire "
         "output. Phase 05 left a gate failing rather than relaxing it. The one large positive "
         "result — 0.845 chemistry-class top-1 on molecules the atlas has never seen, with 0.935 "
         "robustness retention and complete provenance — is credible in proportion to how many "
         "opportunities there were to report it dishonestly.", color=MUTED)


def p_inconsistencies(fig):
    ax = frame(fig)
    y = head(ax, "9 · Inconsistencies found and resolved", "INCLUDING ONE THAT HAS STOOD SINCE PHASE 00")
    y = callout(ax, y, "THE MATERIAL ONE — THE FROZEN CRITERIA MEASURE A TASK THE TERMINOLOGY FORBIDS",
                "SUCCESS_CRITERIA.md, frozen in Phase 00: S-01 requires 'CSM/MSS-equivalent fine "
                "top-1 ≥ 0.7507' against a V5 baseline of 0.6707. The baseline table shows the "
                "level it comes from: v5_atlas::v7_fine_16 — retrieval of the correct fine-16 "
                "CHEMISTRY CLASS.\n"
                "TERMINOLOGY_AND_DEFINITIONS.md: chemical class 'is not the inference output. V7 "
                "never predicts class.'\n"
                "Both have stood since Phase 00. They cannot both be right.\n"
                "RESOLUTION — the frozen thresholds stay exactly as they are; P-13 forbids "
                "adjusting them and nothing here does. The terminology was wrong: it conflated "
                "class as a TERMINAL CLAIM (rightly forbidden) with class as SUPERVISION INSIDE "
                "A FIT (rightly forbidden) and class as an INTERMEDIATE PROBABILISTIC EVIDENCE "
                "COORDINATE, which was never the danger and is what the frozen bar has always "
                "measured. P-06 is amended accordingly.\n"
                "This does not make the 16-d layer safe. It resolves the conflict. U-02 remains "
                "the open scientific question.", RED)
    y = para(ax, y, "Smaller inconsistencies, corrected in the same pass:", weight="bold")
    y -= 0.004
    y = table(ax, y, [
        ("found", "correction"),
        ("'Nothing in V7 has been implemented' in two documents", "corrected — six phases complete"),
        ("corpus cited as 167 analytes throughout", "canonical figure is 154 molecules from 375 spectra;"),
        ("", "167 was the pre-audit normalised-name count"),
        ("original vs canonical phase numbering mixed across documents", "canonical numbering used, mapping restated"),
        ("dependency map showed Phase 05 = BSV, Phase 06 = integration", "rewritten"),
        ("phases/ directory names use original numbering", "retained (committed links) with a mapping table"),
        ("Phase 04 GATE_FAILED left as an open item", "recorded as out of scope under Raman-only (A-09/A-17)"),
    ], [0.46, 0.44], lead=0.0155)
    y -= 0.016
    y = callout(ax, y, "ONE ITEM LEFT OPEN FOR THE PROJECT OWNER'S DECISION",
                "The brief that commissioned this update specified the remaining roadmap as "
                "Phases 06, 07 and 08 only. Phase 09 — the V5 head-to-head against the frozen "
                "Tier-1 criteria — has been RETAINED rather than dropped, because the frozen "
                "criteria have never been measured and dropping the phase that measures them "
                "would make the bar unreachable and nullify P-13 in practice. If the intent was "
                "to retire the V5 replacement decision entirely, that is a legitimate call, but "
                "it should be made explicitly and recorded, not effected by omission.", AMBER)
    y = callout(ax, y, "ONE POLICY CONFLICT, RESOLVED IN FAVOUR OF THE MORE RECENT INSTRUCTION",
                "A standing instruction from Phase 02.5 onward is 'PNG only, no SVG'. This "
                "update explicitly requested publication-quality SVG for the architecture "
                "diagrams. Both formats are written: SVG as requested, PNG at 200 dpi so the "
                "PDF embeds the same image the figure script produced and the per-phase "
                "convention is unbroken.", BLUE)


def p_provenance(fig):
    ax = frame(fig)
    y = head(ax, "Provenance", "WHERE EVERY NUMBER IN THIS DOCUMENT COMES FROM")
    y = table(ax, y, [
        ("claim class", "source"),
        ("all V7 phase numbers", "results/v7_rebuild/<phase>/tables/ and artifacts/"),
        ("the six-level hierarchy", "phase04/tables/hierarchy_retrieval_v1.csv"),
        ("Meta Component comparison", "phase04_5/artifacts/verdict_v1.json"),
        ("Phase 05 retrieval, calibration, rejection", "phase05/artifacts/phase05_summary_v1.json"),
        ("robustness retention", "phase05/tables/robustness_summary_v1.csv"),
        ("evidence-axis grounding and sensitivity", "phase05/tables/evidence_axis_*.csv"),
        ("V5 baseline", "phase00/tables/phase00_baseline_metrics.csv (v7_harness_v1)"),
        ("principles and risks", "context/SCIENTIFIC_DESIGN_PRINCIPLES.md, plan/RISK_REGISTER.md"),
    ], [0.40, 0.50], lead=0.0175)
    y -= 0.020
    y = para(ax, y, "Fingerprints", weight="bold")
    y = table(ax, y, [
        ("object", "fingerprint"),
        ("V5 atlas (control, unmodified)", "09ed804a40836f4a05a91ba10900cded"),
        ("LSM registry", "208482d6f7178b5b8f16cace91be55b0"),
        ("CSM dictionary", "0b4aa550ccefed3edabdbde5bae11c8d"),
        ("theme registry (archived)", "f54d4835ffdf8aa2d50a4a203da0e8f4"),
        ("Phase 05 inference engine", "20d8bd99ce71f45a125c6a2b1d719e51"),
    ], [0.40, 0.50], lead=0.0175)
    y -= 0.024
    y = callout(ax, y, "SCOPE OF THIS UPDATE",
                "Documentation only. No algorithm was implemented, no code modified, no "
                "inference changed, nothing retrained. The changes are confined to architecture, "
                "context, planning, terminology, pipeline diagrams, phase definitions, decision "
                "gates and success criteria.\n"
                "No frozen artefact was touched. No frozen success threshold was altered. Every "
                "archived phase retains its complete output.", GREEN)
    para(ax, y, "If a number in this document disagrees with a phase table, the phase table is "
         "correct and this document is wrong.", color=MUTED, fs=8.4)


def p_figure(fig, path, caption):
    fig.patch.set_facecolor("white")
    img = mpimg.imread(path)
    h, w = img.shape[:2]
    box_w, box_h = 0.90 * LAND[0], 0.78 * LAND[1]
    aspect = h / w
    if box_w * aspect <= box_h:
        dw, dh = box_w, box_w * aspect
    else:
        dh, dw = box_h, box_h / aspect
    fw, fh = dw / LAND[0], dh / LAND[1]
    top, bottom = 0.95, 0.13
    ax = fig.add_axes([(1 - fw) / 2, bottom + (top - bottom - fh) / 2, fw, fh])
    ax.imshow(img); ax.axis("off")
    ft = fig.add_axes([0, 0, 1, 1]); ft.axis("off")
    ft.set_xlim(0, 1); ft.set_ylim(0, 1)
    ft.plot([0.05, 0.95], [0.105, 0.105], color=RULE, lw=0.8)
    yy = 0.082
    for line in textwrap.wrap(caption, 150):
        ft.text(0.05, yy, line, fontsize=8.4, color=INK)
        yy -= 0.020


def main() -> int:
    missing = [k for k in CAPTIONS if not list(FIGS.glob(f"{k}_*.png"))]
    if missing:
        print(f"missing figures {missing} — run make_architecture_figures.py first")
        return 1
    with PdfPages(OUT) as pdf:
        for draw in (p_cover, p_contents, p_current, p_what_changed, p_why, p_evidence):
            _page(pdf, draw)
        for key in sorted(CAPTIONS):
            path = sorted(FIGS.glob(f"{key}_*.png"))[0]
            _page(pdf, lambda f, p=path, c=CAPTIONS[key]: p_figure(f, p, c), LAND)
        for draw in (p_roadmap, p_risks, p_expected, p_inconsistencies, p_provenance):
            _page(pdf, draw)
        d = pdf.infodict()
        d["Title"] = "GAIRA V7 — Architecture Update After Phase 05"
        d["Subject"] = ("Supplementary architecture document: current architecture, what "
                        "changed, why, and the evidence for every change")
        d["Keywords"] = "GAIRA V7 Raman CSM chemistry evidence BSV2 architecture"
    n = len(CAPTIONS) + 11
    print(f"  {OUT.relative_to(ROOT.parent)}  ({OUT.stat().st_size / 1e6:.1f} MB, {n} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
