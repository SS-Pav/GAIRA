#!/usr/bin/env python3
"""GAIRA V7 — Phase 09: assemble PHASE_09_FIGURES.pdf.

A complete document rather than a figure dump: narrative sections, a facing explanation page for
every figure, the result tables, a mathematical appendix rendered with mathtext, and the limits.
Every number is read from the committed artifacts — nothing in this file is typed by hand except
prose.
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
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
PH = HERE.parent
REPO = PH.parents[2]
F, A, R, T = PH / "figures", PH / "artifacts", PH / "reports", PH / "tables"
INK, MUTED, RULE = "#1a1a1a", "#6b7280", "#d1d5db"
ACCENT, WARN, GOOD = "#1d4ed8", "#b45309", "#15803d"
PAGE = (11.69, 8.27)
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "mathtext.fontset": "dejavusans"})

_page_no = [0]


# ── page primitives ──────────────────────────────────────────────────────────
def _new(pdf, draw, chrome=True, footer=""):
    _page_no[0] += 1
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    draw(fig)
    if chrome:
        ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.95, 0.030, str(_page_no[0]), fontsize=8, color=MUTED, ha="right")
        ax.text(0.05, 0.030, footer or "GAIRA V7 · Phase 09 · the canonical inference engine",
                fontsize=7.4, color=MUTED)
    pdf.savefig(fig)
    plt.close(fig)


class Body:
    """A simple flowing text column with a y cursor. Blocks are (kind, payload) tuples."""

    def __init__(self, ax, x=0.075, y=0.855, width=138, right=0.925):
        self.ax, self.x, self.y, self.w, self.right = ax, x, y, width, right

    def _t(self, s, size, color, dy, x=None, weight="normal", style="normal", ha="left"):
        self.ax.text(self.x if x is None else x, self.y, s, fontsize=size, color=color,
                     weight=weight, style=style, ha=ha, va="top")
        self.y -= dy

    def h2(self, s):
        self.y -= 0.012
        self._t(s, 12.5, INK, 0.036, weight="bold")

    def h3(self, s):
        self.y -= 0.006
        self._t(s, 10.0, ACCENT, 0.030, weight="bold")

    def p(self, s, size=9.6, color=INK):
        for line in textwrap.wrap(s, self.w):
            self._t(line, size, color, 0.0250)
        self.y -= 0.010

    def lead(self, s):
        for line in textwrap.wrap(s, int(self.w * 0.84)):
            self._t(line, 10.8, INK, 0.029)
        self.y -= 0.012

    def bullet(self, s, marker="—"):
        lines = textwrap.wrap(s, self.w - 6)
        for i, line in enumerate(lines):
            self.ax.text(self.x + (0.0 if i else 0.0), self.y, marker if i == 0 else " ",
                         fontsize=9.2, color=MUTED, va="top")
            self.ax.text(self.x + 0.022, self.y, line, fontsize=9.2, color=INK, va="top")
            self.y -= 0.0235
        self.y -= 0.005

    def kv(self, rows, keyw=0.30, size=9.4):
        for k, v in rows:
            if not k and not v:
                self.y -= 0.014
                continue
            self.ax.text(self.x, self.y, k, fontsize=size, color=MUTED, va="top")
            self.ax.text(self.x + keyw, self.y, v, fontsize=size, color=INK, va="top")
            self.y -= 0.0265
        self.y -= 0.008

    def table(self, header, rows, cols, size=9.0, colors=None):
        self.ax.plot([self.x, self.right], [self.y + 0.016, self.y + 0.016], color=RULE, lw=0.9)
        for c, h in zip(cols, header):
            self.ax.text(self.x + c, self.y, h, fontsize=size, color=MUTED, va="top",
                         weight="bold")
        self.y -= 0.026
        self.ax.plot([self.x, self.right], [self.y + 0.014, self.y + 0.014], color=RULE, lw=0.6)
        for i, row in enumerate(rows):
            col = (colors or {}).get(i, INK)
            for c, cell in zip(cols, row):
                self.ax.text(self.x + c, self.y, str(cell), fontsize=size, color=col, va="top")
            self.y -= 0.0245
        self.ax.plot([self.x, self.right], [self.y + 0.014, self.y + 0.014], color=RULE, lw=0.9)
        self.y -= 0.020

    def eq(self, s, size=13):
        self.y -= 0.008
        self.ax.text(0.5, self.y, s, fontsize=size, color=INK, ha="center", va="top")
        self.y -= 0.062

    def note(self, s, color=WARN, label="Note"):
        lines = textwrap.wrap(s, self.w - 10)
        h = 0.0235 * len(lines) + 0.030
        self.ax.add_patch(plt.Rectangle((self.x - 0.012, self.y - h + 0.020), 0.004, h,
                                        color=color, transform=self.ax.transAxes, clip_on=False))
        self.ax.text(self.x, self.y, label, fontsize=8.2, color=color, weight="bold", va="top")
        self.y -= 0.024
        for line in lines:
            self._t(line, 9.3, INK, 0.0235, style="italic")
        self.y -= 0.014

    def plain(self, s):
        """The plain-English gloss that runs beside every technical statement."""
        self.note(s, color=ACCENT, label="In plain terms")


def text_page(pdf, kicker, title, build, footer=""):
    def draw(fig):
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        if kicker:
            ax.text(0.075, 0.945, kicker.upper(), fontsize=7.8, color=MUTED, va="top",
                    weight="bold")
        ax.text(0.075, 0.925, title, fontsize=16.5, color=INK, va="top", weight="bold")
        ax.plot([0.075, 0.925], [0.888, 0.888], color=RULE, lw=1.0)
        build(Body(ax))
    _new(pdf, draw, footer=footer)


def divider(pdf, number, title, blurb):
    def draw(fig):
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#f8fafc", zorder=-2))
        ax.text(0.10, 0.60, number, fontsize=64, color="#e2e8f0", weight="bold", va="center")
        ax.text(0.10, 0.47, title, fontsize=24, color=INK, weight="bold", va="center")
        ax.plot([0.10, 0.62], [0.425, 0.425], color=RULE, lw=1.2)
        y = 0.385
        for line in textwrap.wrap(blurb, 78):
            ax.text(0.10, y, line, fontsize=10.2, color=MUTED, va="top")
            y -= 0.030
    _new(pdf, draw)


def figure_page(pdf, path, caption):
    img = mpimg.imread(path)
    h, w = img.shape[:2]

    def draw(fig):
        bw, bh = 0.88 * PAGE[0], 0.76 * PAGE[1]
        asp = h / w
        dw, dh = (bw, bw * asp) if bw * asp <= bh else (bh / asp, bh)
        fw, fh = dw / PAGE[0], dh / PAGE[1]
        top, bottom = 0.935, 0.135
        ax = fig.add_axes([(1 - fw) / 2, bottom + (top - bottom - fh) / 2, fw, fh])
        ax.imshow(img); ax.axis("off")
        ft = fig.add_axes([0, 0, 1, 1]); ft.axis("off")
        ft.set_xlim(0, 1); ft.set_ylim(0, 1)
        ft.text(0.05, 0.968, path.stem.split("_")[0].replace("F", "Figure "),
                fontsize=11, color=INK, weight="bold", va="top")
        ft.plot([0.05, 0.95], [0.112, 0.112], color=RULE, lw=0.8)
        yy = 0.094
        for line in textwrap.wrap(caption, 150):
            ft.text(0.05, yy, line, fontsize=8.3, color=INK, va="top")
            yy -= 0.020
    _new(pdf, draw)


# ── per-figure narrative ─────────────────────────────────────────────────────
CAPTIONS = {
    "F01": "The complete GAIRA V7 architecture. Above the rule is offline learning, done once and "
           "then frozen; below it is the canonical inference path fixed by Phases 05-08. The red "
           "panel lists what is deliberately absent, and every exclusion is a measured decision.",
    "F02": "The grounded Raman corpus: 375 spectra, 154 canonical molecules, 16 chemistry "
           "families, three source libraries. 66 molecules have a single spectrum, which sets a "
           "hard ceiling on molecular retrieval.",
    "F03": "Canonical preprocessing stage by stage: crop to 450-1800 cm-1, resample to 676 bins "
           "at 2.0 cm-1, remove the fluorescence baseline by asymmetric least squares, smooth "
           "with Savitzky-Golay, normalise to unit length. Unchanged since V5.",
    "F04": "Six representative Local Spectral Motifs of 50. Each is a non-negative basis spectrum "
           "learned inside one chemistry class, so it describes a pattern that class actually "
           "contains rather than one imposed on it by the corpus average.",
    "F05": "Six representative Consensus Spectral Motifs of 49, diagnostic bands marked in amber. "
           "These 49 numbers are the canonical coordinates that every downstream layer reads.",
    "F06": "The CSM similarity space, one point per molecule, coloured by chemistry family. "
           "VISUALISATION ONLY - UMAP is not used during inference and the distances shown here "
           "are not quantitative.",
    "F07": "A single inference end to end: spectrum, CSM activation, molecular retrieval, "
           "chemistry radar, ordered bar view, and the written interpretation with its confidence "
           "and warnings. This is exactly what the engine returns.",
    "F08": "Chemistry Evidence radars across eight families. Spoke thickness encodes calibrated "
           "confidence. The radius is RELATIVE BIOCHEMICAL EVIDENCE - not a concentration, not an "
           "abundance, not a mixture fraction.",
    "F09": "Three retrieval outcomes: a confident correct identification, a near miss where the "
           "true molecule sits second or third, and a failure. Green marks the true molecule.",
    "F10": "Chemistry-class confusion under molecule-grouped cross-validation. The errors sit "
           "where a spectroscopist would expect them: small classes dominated by the chains and "
           "rings they share with larger ones.",
    "F11": "Per-axis ROC. These curves are IN-SAMPLE - the chemistry map is fitted on all 375 "
           "spectra, as a shipped engine would be. The held-out top-1 of 0.851 is the number to "
           "quote as performance.",
    "F12": "Per-axis precision-recall, with the same in-sample caveat as Figure 11. Under class "
           "imbalance PR is the more honest of the two curves.",
    "F13": "Calibration of the chemistry confidence, the risk-coverage curve that tells an "
           "operator when to abstain, and the separation between confident-correct and "
           "confident-wrong answers.",
    "F14": "Noise robustness of the complete engine across seven perturbations at five levels "
           "each. The radar degrades far more slowly than molecule identity, which is what a "
           "chemistry-level answer should do.",
    "F15": "Failure analysis: per-class chemistry F1, the low-reconstruction tail, and the ten "
           "worst-ranked spectra by name. The engine flags what it cannot explain rather than "
           "answering confidently.",
    "F16": "End-to-end summary. The middle panel is the finding the whole rebuild rests on: "
           "chemistry accuracy on unseen molecules rises from the raw spectrum to the CSM layer "
           "and does not rise further.",
}

EXPLAIN = {
    "F01": ("The map of everything",
            "Read the diagram top to bottom. Everything above the horizontal rule happened once, "
            "during Phases 00-06, and is now frozen: the corpus was assembled, the motif "
            "dictionaries were learned, the chemistry map was fitted. Everything below the rule "
            "happens every time a spectrum arrives.",
            ["A spectrum enters at the bottom left and moves right through five stages. It is "
             "cleaned, described as a mixture of 50 local motifs, re-described as a mixture of 49 "
             "consensus motifs, compared against 154 reference molecules, and summarised as "
             "evidence along 16 chemistry axes.",
             "The red panel is as important as the blue one. Each item in it was built, measured "
             "and rejected: themes lost information, Meta Components lost more, the geometric "
             "coordinate layer produced an improvement that a significance test could not "
             "distinguish from zero. They are absent because of evidence, not taste.",
             "BSV2 sits deliberately off to the side. It is a way of describing the chemistry "
             "output for a human reader; it never feeds back into the answer."],
            "A diagram like this is usually aspirational. This one is enforced: the engine "
            "verifies four cryptographic fingerprints when it loads and refuses to start if any "
            "frozen component has changed."),
    "F02": ("What the engine was built from",
            "Every claim in this document rests on 375 spectra of 154 pure molecules. Knowing the "
            "shape of that corpus is the fastest way to understand both what the engine can do "
            "and where it must stop.",
            ["The class sizes are very uneven: 80 peptide/protein spectra against 3 nucleic-acid "
             "polymer spectra. That imbalance is why performance is reported as macro F1, which "
             "gives every class one vote, alongside plain accuracy.",
             "The replicate histogram is the panel to dwell on. 66 of 154 molecules appear "
             "exactly once. When such a molecule's only spectrum is held out for testing, the "
             "correct answer is not in the reference bank at all, and the engine cannot possibly "
             "retrieve it. That accounts for 68 of the 375 test queries.",
             "Three source libraries contribute. Phase 06.5 measured how much of the spectral "
             "geometry is explained by chemistry versus by which library a spectrum came from: "
             "chemistry wins by a factor of eleven, but the acquisition signal is not zero."],
            "This is a corpus of PURE compounds measured under clean conditions. It contains no "
            "serum, no extracellular vesicles, no SERS substrates and no tissue. Nothing measured "
            "here transfers to those settings without being measured again."),
    "F03": ("Making spectra comparable",
            "Two instruments measuring the same molecule produce different numbers: different "
            "wavenumber points, different intensity scales, different fluorescent backgrounds. "
            "Preprocessing removes those differences and keeps what is chemical.",
            ["Resampling puts every spectrum on the same 676-point ruler from 450 to 1800 cm-1. "
             "Where a spectrum does not reach, the engine writes zeros and says so in a warning. "
             "It never extrapolates, because an invented value would be indistinguishable from a "
             "measured one further downstream.",
             "The baseline stage is the subtle one. Fluorescence adds a broad hump that can dwarf "
             "the peaks. Asymmetric least squares fits a smooth curve that is pulled hard by the "
             "valleys and barely at all by the peaks, so it traces the background rather than the "
             "signal. The asymmetry is the entire trick.",
             "Savitzky-Golay smoothing fits a small cubic in a sliding window instead of taking a "
             "moving average, so it removes jitter without flattening peaks. Finally the spectrum "
             "is scaled to unit length, which discards laser power and integration time."],
            "That last step is why no output of this engine can be read as a concentration. The "
            "absolute scale is deliberately destroyed before anything else happens."),
    "F04": ("Learning the vocabulary",
            "A Raman spectrum of a mixture is a sum of its components' spectra. Non-negative "
            "matrix factorisation takes that literally: it looks for a small set of basis spectra "
            "that can be ADDED, never subtracted, to reproduce what was observed.",
            ["Each Local Spectral Motif is one such basis spectrum. Fifty of them, learned "
             "separately inside each of the 16 chemistry families, so a motif describes a pattern "
             "that its family actually contains rather than an average across unrelated "
             "chemistry.",
             "The non-negativity is a scientific constraint, not a technical convenience. Methods "
             "that allow negative weights - principal component analysis, for one - fit better "
             "but produce components with negative lobes, and there is no such thing as "
             "negative-two-parts glucose in a real mixture.",
             "A typical spectrum uses about ten of the fifty motifs. The dictionary is describing "
             "composition, not memorising curves; if it were memorising, each spectrum would "
             "activate its own private motif and nothing would generalise."],
            "Explained variance averages 0.824 across the corpus, but the minimum is 0.209. The "
            "molecules at the bottom - pyruvate, thymine, malic acid, urea - are small and "
            "high-symmetry, with few strong bands in this window. The engine reports low "
            "confidence on them rather than guessing."),
    "F05": ("From 50 motifs to 49 coordinates",
            "Some local motifs describe the same underlying vibration seen from two different "
            "chemistry families. Phase 02 merged those by building a graph in which motifs are "
            "nodes and an edge means 'these two are the same thing'.",
            ["An edge weight combines seven independent similarity measures using a weighted "
             "GEOMETRIC mean. This matters: under an ordinary average, two motifs that merely "
             "looked alike could be merged on the strength of one channel. Under a geometric "
             "mean, any single near-zero channel drags the whole weight toward zero. It is the "
             "operational form of 'a consensus requires several independent lines of evidence'.",
             "The threshold for cutting the graph was not chosen by eye. It was swept from 0.05 "
             "to 0.90 and tested against a null built by shuffling spectral bands, keeping only "
             "groupings that survive a wide contiguous range of thresholds.",
             "The result is 49 Consensus Spectral Motifs, each carrying its diagnostic band "
             "positions - the amber marks. Those bands are what lets a chemistry conclusion be "
             "traced all the way back to specific wavenumbers."],
            "Compressing 50 motifs to 49 costs essentially nothing in reconstruction (0.8237 to "
            "0.8232) while RAISING replicate consistency from 0.880 to 0.893. That is the whole "
            "reason the layer exists, and it is why the CSM activation - not the LSM activation - "
            "is the canonical representation."),
    "F06": ("The shape of chemistry space",
            "This is a two-dimensional shadow of a 49-dimensional space, produced by UMAP. It is "
            "here to build intuition and for no other purpose.",
            ["Molecules of the same chemistry family land near each other without ever having "
             "been told to. Phase 06.5 quantified this properly: chemistry explains 45.2% of the "
             "distance structure, excitation wavelength 11.8%, and source library 4.0%. Chemistry "
             "wins by a wide margin, but acquisition is not nothing.",
             "Phase 06.5 also asked whether the space contains a natural number of clusters. It "
             "does not. Across four algorithms and seven internal indices, not one index has an "
             "interior optimum - they rise or fall monotonically as the cluster count increases. "
             "The space is a continuum with dense regions, not a set of discrete islands.",
             "That result has a consequence worth stating: the 16 chemistry classes are a curated "
             "cut through a continuum, chosen because chemists find them useful. They are not a "
             "discovery, and the geometry does not validate them."],
            "Do not read distances on this plot quantitatively. UMAP preserves local neighbourhood "
            "structure and distorts global distance freely; two clusters appearing far apart here "
            "may not be far apart in the real space."),
    "F07": ("One complete answer",
            "This is the whole engine on one page, for one spectrum. Six panels, in the order the "
            "engine produces them.",
            ["Top left is the preprocessed spectrum. Next to it, the CSM activation: which of the "
             "49 consensus motifs the spectrum is made of. About ten are active; the rest are "
             "exactly zero.",
             "The retrieval panel ranks the 154 reference molecules by how closely their "
             "activation pattern points in the same direction as this one. Crucially, that "
             "similarity is an inner product, so it decomposes EXACTLY into per-motif "
             "contributions. The engine checks this for every candidate it reports: the parts sum "
             "to the whole to within 1e-9. There is no hidden term.",
             "The radar and the bar chart are two views of the same 16 numbers. The radar is good "
             "for recognising a pattern at a glance; the bars are better for reading off an "
             "ordering. The written interpretation at the bottom right states the confidence and "
             "any warnings."],
            "Confidence is the product of two things: how much of the spectrum the atlas could "
            "explain, and how well the best reference matched. Both must be high. A spectrum the "
            "dictionary cannot express might still land near some molecule by accident, and "
            "multiplying ensures that accident cannot become a confident answer."),
    "F08": ("Reading a radar",
            "Sixteen spokes, one per chemistry family, always in the same order. The distance "
            "along a spoke is the relative evidence for that chemistry. Thicker spokes mean the "
            "engine is more confident.",
            ["Compare the shapes, not the sizes. A sterol produces a radar with one dominant "
             "spoke; an amino acid produces a broader shape with weight spread over several "
             "nitrogen-bearing families. It is the pattern that identifies the chemistry.",
             "The same molecule measured twice produces nearly the same radar - cosine similarity "
             "0.960 across replicates. That reproducibility is what makes the radar usable as a "
             "report to a human rather than a curiosity.",
             "Under noise the radar is the last thing to break. Averaged over 35 perturbation "
             "conditions it retains cosine 0.965 to its clean self, while molecule identity falls "
             "to 0.811. The general answer survives longer than the specific one, which is "
             "exactly the intended behaviour."],
            "The radius is RELATIVE BIOCHEMICAL EVIDENCE. It is not a concentration, not an "
            "abundance, and not a mixture fraction. The absolute scale was destroyed by "
            "normalisation in the very first stage, and a similarity is not a quantity of "
            "material. A tall spoke means 'evidence associated with this chemistry is present', "
            "and nothing more."),
    "F09": ("When retrieval works and when it does not",
            "Three real queries, chosen to show the range. Green marks the true molecule wherever "
            "it appears in the ranking.",
            ["The left panel is the common case: the correct molecule is first, with a clear gap "
             "to the second. This happens for 60.5% of spectra.",
             "The middle panel is a near miss - the true molecule is second or third, usually "
             "behind a close structural relative. A further 15.7% of spectra land here, which is "
             "why top-3 reaches 0.763. For a chemist, a shortlist of three is often more useful "
             "than a single confident guess.",
             "The right panel is a genuine failure. Many of these are structural rather than "
             "algorithmic: 68 of the 375 test queries belong to molecules with only one spectrum "
             "in the corpus, so when that spectrum is held out the answer is simply absent from "
             "the bank."],
            "Those 68 unretrievable spectra are counted as failures. Excluding them would raise "
            "the headline from 0.605 to roughly 0.74, and that number would be misleading - the "
            "engine really did fail to name them."),
    "F10": ("Where the chemistry layer gets confused",
            "Rows are the true class, columns the predicted one. A perfect result would put every "
            "spectrum on the diagonal.",
            ["The off-diagonal mass is not random. Fatty acids are confused with acylglycerols; "
             "free amino acids with peptides; purines with pyrimidines. In each case the two "
             "families share the chemical group that dominates their Raman signature - a long "
             "aliphatic chain, an amide backbone, a nitrogen heterocycle.",
             "The weakest classes are the smallest. small_nitrogenous (7 spectra, F1 0.400) and "
             "phospholipid_sphingolipid (8 spectra, F1 0.556) are both dominated by larger "
             "families they resemble. sterol_steroid and nucleic_acid_polymer reach F1 1.000.",
             "The ontology was deliberately NOT modified in response to this figure. Merging two "
             "classes because a model confuses them would improve the score while destroying the "
             "distinction the score was meant to measure."],
            "This matrix is computed under molecule-grouped cross-validation: every spectrum of a "
            "molecule sits in the same fold, so the model is always predicting a molecule it has "
            "never seen. That is the honest question and it produces the honest 0.851."),
    "F11": ("How separable each chemistry is",
            "A receiver operating characteristic curve asks: as we lower the bar for calling a "
            "spectrum 'this chemistry', how quickly do we pick up true cases relative to false "
            "ones? The area beneath is the chance that a random true case outranks a random "
            "false one.",
            ["Every axis sits near the top-left corner, and the macro-averaged area is 0.999. "
             "Taken at face value that would be an extraordinary result.",
             "It should not be taken at face value. These curves are IN-SAMPLE: the chemistry map "
             "was fitted on all 375 spectra, and is being asked about those same spectra. That is "
             "the right way to describe the model being shipped, and the wrong way to predict "
             "performance on anything new.",
             "The honest comparison is the held-out figure of 0.851 top-1 accuracy, computed with "
             "the model refitted inside each cross-validation fold. The gap between 0.955 "
             "in-sample and 0.851 held-out is precisely the size of the illusion."],
            "This distinction was a defect in this phase before it was a caption. Validation 4 was "
            "originally computed in-sample only, and would have reported 0.955 as the headline. "
            "Gates G15 and G16 were added so it cannot recur."),
    "F12": ("The same question, asked more strictly",
            "Precision-recall curves plot how often the engine is right when it commits, against "
            "how many true cases it finds.",
            ["ROC has a blind spot under class imbalance. With 80 peptide spectra and 3 "
             "nucleic-acid-polymer spectra, a large pool of true negatives keeps the false "
             "positive rate small no matter how the model behaves on the rare class. Precision "
             "has no such escape hatch: a false positive hurts immediately.",
             "For an unevenly sampled corpus like this one, PR is therefore the more honest "
             "picture. Macro average precision is 0.983 - still high, still in-sample.",
             "The classes whose PR curves sag are the same ones the confusion matrix flagged: the "
             "small families that share their strongest bands with larger neighbours."],
            "Same in-sample caveat as Figure 11. Quote 0.851."),
    "F13": ("Knowing when not to answer",
            "Three panels about honesty rather than accuracy.",
            ["The reliability diagram asks: of everything the engine called 80% likely, was it "
             "right 80% of the time? Points on the diagonal mean the confidences can be believed. "
             "Expected calibration error summarises the gap in one number.",
             "Calibration error alone is a trap, and V7 learned it the hard way. A model that "
             "outputs the same base rate for every input has near-perfect calibration and is "
             "completely useless. That is why model selection here uses log loss subject to "
             "FLOORS on sharpness and discrimination - a constant predictor is disqualified "
             "before calibration is consulted at all.",
             "The risk-coverage curve is the practical one. It says: if you only answer when "
             "confidence exceeds some threshold, how often are you right, and how many spectra do "
             "you answer at all? Refusing to answer below a margin of 0.497 keeps 51% of spectra "
             "at 79% accuracy, against 61% at full coverage."],
            "No operating point is baked into the engine. The curve is published so that an "
            "operator can choose one for their own tolerance for being wrong."),
    "F14": ("What happens as the signal degrades",
            "Seven ways a spectrum degrades on a real instrument, each applied at five strengths, "
            "each run through the complete engine from raw spectrum to final radar.",
            ["Detector noise and shot noise barely register - the radar holds above cosine 0.998 "
             "at every level. Preprocessing absorbs high-frequency perturbations before the "
             "projection ever sees them.",
             "Baseline drift and fluorescence are harder, because they attack the stage designed "
             "to remove them. At the most extreme drift the engine still classifies chemistry "
             "correctly 72% of the time and retains radar cosine 0.867.",
             "Band broadening and peak dropout hurt molecule identity most, which makes physical "
             "sense: both destroy the fine band structure that distinguishes one molecule from "
             "its close relative, while leaving the coarse pattern that identifies a family."],
            "Across every condition the ordering holds: radar 0.965 > chemistry 0.889 > molecule "
            "0.811. When evidence weakens, the engine gives up naming the molecule before it "
            "gives up describing the chemistry. That ordering is the architecture's central "
            "promise, and this figure is where it is demonstrated rather than asserted."),
    "F15": ("The engine's own weak points",
            "A system that only shows its successes cannot be trusted. These are the failures, by "
            "name.",
            ["Per-class F1 spans 1.000 down to 0.400. The weak classes are the small ones, and "
             "they are weak in a chemically predictable direction.",
             "The low-reconstruction tail is chemically coherent: pyruvate at 0.209, thymine "
             "around 0.27, malic acid at 0.26, urea at 0.345. These are small, high-symmetry "
             "molecules with few strong bands between 450 and 1800 cm-1. A dictionary built "
             "largely from biopolymers should struggle here, and it does - the prediction and the "
             "observation agree.",
             "The ten worst-ranked spectra are named individually. Several are singleton "
             "molecules whose answer was never in the bank; others, like stearate being called an "
             "acylglycerol, are boundary cases where the spectrum is well explained and the "
             "ontology is what is difficult."],
            "One limit found while writing the regression tests: white noise is NOT reliably "
            "flagged. It reconstructs at explained variance around 0.61, above the 0.50 warning "
            "floor, so the flag fires on only 1 of 20 random spectra. Confidence still separates "
            "it cleanly - noise peaks at 0.495 against a corpus mean of 0.803. Read the "
            "confidence, not the flag."),
    "F16": ("The result the architecture rests on",
            "One page, three claims, each measured the same way on the same corpus.",
            ["The middle panel is the important one. Chemistry accuracy on molecules the model "
             "has never seen rises from 0.608 using the raw spectrum, to 0.850 using the LSM "
             "layer, to 0.855 using the CSM layer - and then falls for every layer built on top: "
             "0.664 for eleven grounded axes, 0.405 for themes, 0.392 for Meta Components.",
             "Four independent attempts to build an abstraction above the CSM each lost "
             "information. A fifth, the geometric coordinate layer of Phase 06.5, produced an "
             "improvement of +0.016 that a paired significance test could not distinguish from "
             "zero (p = 0.180).",
             "So the engine ships the layer where the information actually is, and treats "
             "everything above it - including BSV2 - as a description of that layer rather than a "
             "stage of it."],
            "This is not a claim that the CSM representation is optimal. It is the considerably "
            "stronger and more useful claim that four separate attempts to improve on it, "
            "designed independently and measured identically, all failed."),
}


# ── document ─────────────────────────────────────────────────────────────────
def main() -> int:
    state = json.loads((PH / "PHASE_STATE.json").read_text())
    s = json.loads((A / "phase09_summary_v1.json").read_text())
    figs = sorted(F.glob("F*.png"))
    if not figs:
        print("no figures — run make_figures.py first")
        return 1
    e = s["engine"]
    v1, v2, v3, v4 = (s["validation_1_lsm"], s["validation_2_csm"],
                      s["validation_3_retrieval"], s["validation_4_chemistry"])
    n = s["noise_robustness"]
    per_class = pd.read_csv(T / "csm_per_class_v1.csv")
    ranks = pd.read_csv(T / "retrieval_rank_distribution_v1.csv")
    rc = pd.read_csv(T / "retrieval_risk_coverage_v1.csv")
    gates = pd.read_csv(T / "phase09_gates_v1.csv")
    rob = pd.read_csv(T / "noise_robustness_v1.csv")
    R.mkdir(parents=True, exist_ok=True)
    out = R / "PHASE_09_FIGURES.pdf"

    with PdfPages(out) as pdf:
        # ── cover ────────────────────────────────────────────────────────────
        def cover(fig):
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.text(0.075, 0.90, "GAIRA V7", fontsize=34, color=INK, weight="bold")
            ax.text(0.075, 0.838, "The Canonical Inference Engine", fontsize=19, color=INK)
            ax.plot([0.075, 0.925], [0.805, 0.805], color=RULE, lw=1.3)
            ax.text(0.075, 0.772, "Phase 09 — one frozen path from a Raman spectrum to an "
                    "explainable biochemical interpretation", fontsize=10.5, color=MUTED)
            rows = [
                ("Status", f"{state['status']} — packaging phase, architecture unchanged"),
                ("Engine", f"{e['n_lsms']} LSMs · {e['n_csms']} CSMs · {e['n_molecules']} "
                           f"molecules · {e['n_chemistry_axes']} chemistry axes"),
                ("Atlas fingerprint", e["atlas_fingerprint"]),
                ("Deterministic", str(e["deterministic"])),
                ("", ""),
                ("LSM reconstruction EV", f"{v1['mean_explained_variance']:.4f}   replicate "
                                          f"consistency {v1['replicate_consistency']:.4f}"),
                ("CSM chemistry top-1 / top-3", f"{v2['class_top1']:.4f} / {v2['class_top3']:.4f}"
                                                f"   macro-F1 {v2['macro_f1']:.4f}"),
                ("Molecule top-1 / top-5 / MRR", f"{v3['top1']:.4f} / {v3['top5']:.4f} / "
                                                 f"{v3['mrr']:.4f}"),
                ("Chemistry top-1 (HELD OUT)", f"{v4['fine_top1_heldout']:.4f}   top-3 "
                                               f"{v4['fine_top3_heldout']:.4f}   macro-F1 "
                                               f"{v4['macro_f1_heldout']:.4f}"),
                ("Chemistry top-1 (in-sample)", f"{v4['fine_top1_in_sample']:.4f} — a sanity "
                                                f"check, NOT a performance claim"),
                ("Chemistry calibration ECE", f"{v4['ece']:.4f}"),
                ("Radar reproducibility", f"{v4['radar_reproducibility']:.4f}"),
                ("", ""),
                ("Under perturbation", f"molecule {n['mean_retrieval_top1']:.3f} · chemistry "
                                       f"{n['mean_chemistry_top1']:.3f} · radar cosine "
                                       f"{n['mean_radar_cosine']:.3f}"),
                ("Frozen baseline reproduced", f"{s['baseline_match']} — exactly, all six "
                                               f"retrieval metrics"),
                ("Gates", f"{s['gates']['n'] - s['gates']['failed']} of {s['gates']['n']} PASS"),
                ("", ""),
                ("Scope", "pure Raman reference spectra only"),
                ("Completed", state["finished"][:19].replace("T", " ") + " UTC"),
            ]
            y = 0.720
            for k, v in rows:
                if k:
                    ax.text(0.075, y, k, fontsize=9.0, color=MUTED)
                    ax.text(0.40, y, v, fontsize=9.0, color=INK)
                y -= 0.0288
            ax.text(0.075, 0.075,
                    "The Chemistry Evidence radar shows RELATIVE BIOCHEMICAL EVIDENCE. It is not "
                    "a concentration, not an abundance, and not a mixture statement.\n"
                    "Sources of record: PHASE_09_REPORT.md · PHASE_09_ENGINE_SPEC.md · "
                    "PHASE_09_MATHEMATICAL_APPENDIX.md · PHASE_09_SCIENTIFIC_AUDIT.md · "
                    "PHASE_09_DECISION_GATE.md",
                    fontsize=8.2, color=MUTED)
        _new(pdf, cover, chrome=False)

        # ── contents ─────────────────────────────────────────────────────────
        def contents(b: Body):
            b.p("This document is written to be read start to finish by someone who is not a "
                "spectroscopist. Every technical section is followed by a plain-English gloss "
                "marked in blue. Nothing in it is asserted without a number behind it, and the "
                "numbers all come from the committed artifacts of this phase.")
            b.h2("Part I — What the engine is")
            for t in ("The problem, stated plainly", "The five stages", "What is deliberately "
                      "absent, and why", "How to read a figure in this document"):
                b.bullet(t, marker="·")
            b.h2("Part II — The sixteen figures")
            b.p("Each figure is preceded by a page explaining what it shows, how to read it, and "
                "what it does not license you to conclude.")
            b.h2("Part III — Results")
            for t in ("Validation 1-4 in full", "Per-class behaviour", "Robustness",
                      "Gates"):
                b.bullet(t, marker="·")
            b.h2("Part IV — Mathematics")
            b.p("Every equation the engine executes, each with its intuition.")
            b.h2("Part V — Limits, decision and glossary")
        text_page(pdf, "", "Contents", contents)

        # ── Part I ───────────────────────────────────────────────────────────
        divider(pdf, "I", "What the engine is",
                "A Raman spectrum goes in. A ranked list of candidate molecules, a "
                "sixteen-dimensional chemistry reading, a calibrated confidence and a complete "
                "audit trail come out. One path, no alternatives, no tunable knobs.")

        def p_problem(b: Body):
            b.lead("Shine a laser at a sample and a small fraction of the light comes back with "
                   "its colour shifted. The size of each shift is set by how the molecules in the "
                   "sample vibrate, so the pattern of shifts is a fingerprint of the chemistry "
                   "present. That pattern is a Raman spectrum.")
            b.h3("Why this is hard")
            b.p("A textbook treats a spectrum as a fingerprint to be matched against a library. "
                "Real samples do not cooperate. Three facts govern everything in this "
                "architecture:")
            b.bullet("Spectra are mixtures, not fingerprints. What arrives at the detector is the "
                     "sum of contributions from everything in the illuminated volume.")
            b.bullet("A peak is not a molecule. A band near 1450 cm-1 says a CH2 group is "
                     "bending. Thousands of biological molecules contain CH2 groups.")
            b.bullet("Nearby is not the same. Published assignments frequently match a peak to a "
                     "molecule because the wavenumbers agree to within a few units and the "
                     "biology is plausible. That is not evidence.")
            b.h3("What GAIRA answers instead")
            b.p("Rather than asking 'which molecule is this?', the engine answers a question it "
                "can support: what chemistry does the evidence favour, how strongly, and by way "
                "of which specific spectral features? It still returns ranked molecular "
                "candidates, because a shortlist is useful, but the load-bearing output is the "
                "chemistry reading.")
            b.plain("Think of a doctor reading a blood panel. They rarely conclude 'this is "
                    "precisely molecule X at precisely concentration Y'. They read a pattern and "
                    "say which processes it is consistent with, and how sure they are. This "
                    "engine is built to do the second thing well rather than the first thing "
                    "badly.")
        text_page(pdf, "Part I", "The problem, stated plainly", p_problem)

        def p_stages(b: Body):
            b.lead("Five stages, always in this order, with no branches and no options.")
            b.h3("1 · Preprocess")
            b.p("Crop to 450-1800 cm-1, resample to 676 points spaced 2.0 cm-1 apart, subtract "
                "the fluorescence background by asymmetric least squares, smooth with a "
                "Savitzky-Golay filter, and scale to unit length. This makes any two spectra "
                "comparable and discards the absolute intensity scale.")
            b.h3("2 · Project onto 50 Local Spectral Motifs")
            b.p("Express the spectrum as a non-negative sum of 50 learned basis spectra. About "
                "ten are active for a typical sample. Reported for interpretability; not consumed "
                "by any later stage.")
            b.h3("3 · Project onto 49 Consensus Spectral Motifs")
            b.p("The same operation against the merged, deduplicated dictionary. These 49 numbers "
                "are the canonical representation - the coordinates every downstream layer reads, "
                "and the only ones.")
            b.h3("4 · Retrieve molecules")
            b.p("Compare the activation pattern against 154 reference molecules by cosine "
                "similarity, and rank them. Because a cosine is an inner product, the score "
                "decomposes exactly into per-motif contributions, which is what makes the ranking "
                "explainable rather than merely produced.")
            b.h3("5 · Aggregate to Chemistry Evidence")
            b.p("Collapse the activation into 16 numbers, one per chemistry family, using a "
                "hierarchical model that lets coarse chemistry gently inform fine chemistry "
                "without ever excluding it. Calibrate, and emit the radar with its confidence and "
                "warnings.")
            b.plain("Each stage answers a narrower question than the last: is this spectrum "
                    "clean, what patterns is it made of, which of those patterns are real and "
                    "distinct, what does it resemble, and what does that mean chemically.")
        text_page(pdf, "Part I", "The five stages", p_stages)

        def p_absent(b: Body):
            b.lead("The most informative part of this architecture is what it does not contain. "
                   "Each of the following was built, measured, and rejected on evidence.")
            b.table(
                ["excluded", "measured outcome", "phase"],
                [["Theme layer", "chemistry on unseen molecules fell to 0.405", "03 / 04"],
                 ["Meta Components", "0.392", "04.5"],
                 ["11 grounded axes", "0.664", "04"],
                 ["Latent geometry / coordinates", "molecule +0.016, McNemar p = 0.180", "06.5"],
                 ["Chemistry-aware reranking", "no gain once both arms used the same bank", "08"],
                 ["BSV2 on the inference path", "derived description, never an input", "07"],
                 ["SERS / serum / EV handling", "out of scope; unmeasured in V7", "—"]],
                [0.0, 0.30, 0.72])
            b.p("For comparison, chemistry accuracy on unseen molecules using the raw preprocessed "
                "spectrum is 0.608, and using the CSM activation it is 0.855. Every layer built "
                "ABOVE the CSM performed worse than the CSM itself.")
            b.h3("The failure mode this rebuild kept catching")
            b.p("Seven times across V7, a selection rule was maximised by an answer that was "
                "degenerate rather than good: a calibrator that returned the base rate for every "
                "input scored best on calibration error; a cluster count of 4 scored best on "
                "bootstrap stability because a coarse partition is trivially stable; a programme "
                "count of 16 over a 16-dimensional input scored best because the model had "
                "learned the identity function.")
            b.p("The countermeasure that worked every time was the same: describe what a "
                "degenerate answer would look like BEFORE running the selection, and add a floor "
                "that disqualifies it.")
            b.plain("A metric is a proxy for what you want. Any proxy can be satisfied by "
                    "something you did not want. The discipline is not cleverer metrics — it is "
                    "naming the cheat in advance and forbidding it.")
        text_page(pdf, "Part I", "What is deliberately absent, and why", p_absent)

        def p_howto(b: Body):
            b.lead("Every figure in Part II is preceded by an explanation page laid out the same "
                   "way.")
            b.kv([("Title and opening line", "what the figure is for, in one sentence"),
                  ("Three or four paragraphs", "what each panel shows and how to read it"),
                  ("A blue box", "the plain-English gloss, or the caveat that matters most"),
                  ("The figure itself", "full page, with its formal caption beneath")], keyw=0.26)
            b.h3("Conventions used throughout")
            b.bullet("Wavenumbers are in cm-1 and always span 450 to 1800.")
            b.bullet("'Held out' means molecule-grouped cross-validation: every spectrum of a "
                     "molecule sits in one fold, so the model is always predicting a molecule it "
                     "has never seen. This is the honest number.")
            b.bullet("'In-sample' means the model was fitted on all 375 spectra and asked about "
                     "those same spectra. It describes the object being shipped and is never a "
                     "performance claim. Where both appear, quote the held-out one.")
            b.bullet("'Top-k' is the fraction of queries whose correct answer appeared in the "
                     "first k results. 'Macro F1' averages over classes so a 7-spectrum class "
                     "counts as much as an 80-spectrum one.")
            b.bullet("Radar radius is relative evidence. Never a concentration.")
            b.note("If you read only one figure, read Figure 16. If you read two, add Figure 14.",
                   color=GOOD, label="Where to start")
        text_page(pdf, "Part I", "How to read this document", p_howto)

        # ── Part II ──────────────────────────────────────────────────────────
        divider(pdf, "II", "The sixteen figures",
                "Each figure gets an explanation page and a full page of its own. Figures 1 to 6 "
                "describe how the engine was built; 7 to 9 show it working; 10 to 13 measure it; "
                "14 to 16 stress it and summarise.")
        for p in figs:
            key = p.stem.split("_")[0]
            title, lead, paras, note = EXPLAIN[key]

            def build(b: Body, title=title, lead=lead, paras=paras, note=note):
                b.lead(lead)
                for para in paras:
                    b.p(para)
                b.plain(note)
            text_page(pdf, f"Figure {key.replace('F', '').lstrip('0')}", title, build)
            figure_page(pdf, p, CAPTIONS.get(key, ""))

        # ── Part III ─────────────────────────────────────────────────────────
        divider(pdf, "III", "Results",
                "Four validations across all 375 spectra, the per-class detail behind them, the "
                "robustness sweep, and the sixteen gates.")

        def p_v12(b: Body):
            b.h3("Validation 1 — the LSM layer")
            b.kv([("mean explained variance", f"{v1['mean_explained_variance']:.4f}"),
                  ("minimum explained variance", f"{v1['min_explained_variance']:.4f}  (pyruvate)"),
                  ("mean reconstruction error", f"{v1['mean_reconstruction_error']:.4f}"),
                  ("mean active components", f"{v1['mean_active_components']:.1f} of "
                                             f"{v1['n_lsms']}"),
                  ("mean sparsity", f"{v1['mean_sparsity']:.4f}"),
                  ("replicate consistency", f"{v1['replicate_consistency']:.4f}")])
            b.p("Ten motifs of fifty reconstruct a typical spectrum, and replicates of the same "
                "molecule land in nearly the same place. The dictionary is describing "
                "composition, not memorising curves.")
            b.h3("Validation 2 — the CSM layer")
            b.kv([("chemistry-class top-1", f"{v2['class_top1']:.4f}"),
                  ("chemistry-class top-3", f"{v2['class_top3']:.4f}"),
                  ("macro F1", f"{v2['macro_f1']:.4f}"),
                  ("balanced accuracy", f"{v2['balanced_accuracy']:.4f}"),
                  ("mean explained variance", f"{v2['mean_explained_variance']:.4f}"),
                  ("replicate consistency", f"{v2['replicate_consistency']:.4f}")])
            b.plain("Collapsing 50 motifs to 49 costs almost nothing in reconstruction while "
                    "raising replicate consistency. That trade is the entire justification for "
                    "the CSM layer, and it is why the CSM activation is the canonical "
                    "representation rather than the LSM activation.")
        text_page(pdf, "Part III", "Validations 1 and 2 — representation", p_v12)

        def p_v3(b: Body):
            b.lead("Leave-one-spectrum-out over all 375 spectra against the full 154-molecule "
                   "bank. Every metric reproduces the frozen Phase 05/08 baseline exactly.")
            b.table(["metric", "Phase 09", "frozen baseline"],
                    [["top-1", f"{v3['top1']:.4f}", "0.6053"],
                     ["top-3", f"{v3['top3']:.4f}", "0.7627"],
                     ["top-5", f"{v3['top5']:.4f}", "0.7947"],
                     ["top-10", f"{v3['top10']:.4f}", "0.8107"],
                     ["MRR", f"{v3['mrr']:.4f}", "0.6870"],
                     ["nDCG@5", f"{v3['ndcg5']:.4f}", "0.7112"],
                     ["median rank", f"{v3['median_rank']:.1f}", "—"],
                     ["calibration ECE", f"{v3['ece']:.4f}", "—"],
                     ["discrimination", f"{v3['discrimination']:.4f}", "—"]],
                    [0.0, 0.26, 0.50])
            b.h3("Where the ranks fall")
            b.table(["rank", "spectra", "share"],
                    [[str(int(r.rank_upper)) if r.rank_upper < 1000 else "unretrievable",
                      str(int(r.n)), f"{r.share:.3f}"] for r in ranks.itertuples()],
                    [0.0, 0.20, 0.36])
            b.note("The final row is structural. 66 of 154 molecules have exactly one spectrum, "
                   "so when it is held out the correct answer is absent from the bank entirely. "
                   "These 68 spectra are counted as failures. Excluding them would raise top-1 "
                   "from 0.605 to roughly 0.74 and would misrepresent what the engine did.")
        text_page(pdf, "Part III", "Validation 3 — molecular retrieval", p_v3)

        def p_v4(b: Body):
            b.note("The shipped engine fits its chemistry map on all 375 spectra, as a shipped "
                   "engine should. The in-sample figures describe that object. The number to "
                   "quote as performance is the held-out one.", color=WARN, label="Read this first")
            b.h3("Held out — molecule-grouped cross-validation")
            b.kv([("fine-class top-1", f"{v4['fine_top1_heldout']:.4f}"),
                  ("fine-class top-3", f"{v4['fine_top3_heldout']:.4f}"),
                  ("macro F1", f"{v4['macro_f1_heldout']:.4f}")])
            b.h3("In-sample — a sanity check on the shipped fit")
            b.kv([("fine-class top-1", f"{v4['fine_top1_in_sample']:.4f}"),
                  ("fine-class top-3", f"{v4['fine_top3_in_sample']:.4f}"),
                  ("macro AUC", f"{v4['macro_auc']:.4f}"),
                  ("macro average precision", f"{v4['macro_average_precision']:.4f}")])
            b.h3("Calibration and stability")
            b.kv([("expected calibration error", f"{v4['ece']:.4f}  (in-sample; held-out was "
                                                 f"0.1247 in Phase 06)"),
                  ("Brier score", f"{v4['brier']:.4f}"),
                  ("discrimination", f"{v4['discrimination']:.4f}"),
                  ("radar reproducibility", f"{v4['radar_reproducibility']:.4f}")])
            b.plain("The gap between 0.955 and 0.851 is exactly the illusion that in-sample "
                    "evaluation creates. Phase 09 found this as a defect in its own first draft, "
                    "fixed it, and added two gates so it cannot recur.")
        text_page(pdf, "Part III", "Validation 4 — Chemistry Evidence", p_v4)

        def p_class(b: Body):
            b.lead("Per-class behaviour under molecule-grouped cross-validation. All sixteen "
                   "classes, named explicitly.")
            rows, colors = [], {}
            for i, r in enumerate(per_class.itertuples()):
                rows.append([r._1, str(int(r.n)), f"{r.precision:.3f}", f"{r.recall:.3f}",
                             f"{r.f1:.3f}"])
                colors[i] = GOOD if r.f1 >= 0.90 else (WARN if r.f1 < 0.70 else INK)
            b.table(["chemistry class", "n", "precision", "recall", "F1"], rows,
                    [0.0, 0.30, 0.38, 0.50, 0.62], size=8.4, colors=colors)
            b.p("Green marks F1 at or above 0.90; amber marks below 0.70. The weak classes are "
                "the small ones, and they are weak toward the larger families whose dominant "
                "bands they share.")
            b.note("The ontology was not modified in response to this table. Merging two classes "
                   "because a model confuses them raises the score by removing the distinction "
                   "the score exists to measure.")
        text_page(pdf, "Part III", "Per-class chemistry behaviour", p_class)

        def p_rob(b: Body):
            b.lead("Seven perturbations, five levels each, applied to the raw spectrum and run "
                   "through the complete engine. 35 conditions across all 375 spectra.")
            b.kv([("mean molecule top-1", f"{n['mean_retrieval_top1']:.4f}"),
                  ("mean chemistry top-1", f"{n['mean_chemistry_top1']:.4f}"),
                  ("mean radar cosine", f"{n['mean_radar_cosine']:.4f}")])
            g = (rob.groupby("perturbation")
                    .agg(retrieval=("retrieval_top1", "mean"),
                         chemistry=("chemistry_top1", "mean"),
                         radar=("radar_cosine", "mean"),
                         worst=("radar_cosine", "min")).reset_index())
            b.table(["perturbation", "molecule", "chemistry", "radar", "worst radar"],
                    [[r.perturbation.replace("_", " "), f"{r.retrieval:.3f}",
                      f"{r.chemistry:.3f}", f"{r.radar:.3f}", f"{r.worst:.3f}"]
                     for r in g.itertuples()],
                    [0.0, 0.26, 0.40, 0.54, 0.68])
            b.p("The ordering never reverses. Under every perturbation the radar degrades most "
                "slowly, chemistry next, molecule identity fastest.")
            b.plain("That is the correct ordering for a chemistry-level answer. As evidence "
                    "weakens the engine should lose its grip on WHICH molecule before it loses "
                    "its reading of WHAT KIND of chemistry is present. This table is where that "
                    "design intention becomes a measurement.")
        text_page(pdf, "Part III", "Robustness", p_rob)

        def p_gates(b: Body):
            b.lead(f"{s['gates']['n'] - s['gates']['failed']} of {s['gates']['n']} gates pass. "
                   "Gates are declared before the run and are not relaxed afterwards.")
            b.table(["gate", "status"],
                    [[g.gate, g.status] for g in gates.itertuples()],
                    [0.0, 0.76], size=8.4,
                    colors={i: GOOD if g.status == "PASS" else WARN
                            for i, g in enumerate(gates.itertuples())})
            b.note("G15 and G16 were added during the phase, after an in-sample Validation 4 and "
                   "a pinned calibration temperature were found. Adding a gate in response to a "
                   "defect is permitted; relaxing one in response to a result is not.")
        text_page(pdf, "Part III", "Gates", p_gates)

        # ── Part IV ──────────────────────────────────────────────────────────
        divider(pdf, "IV", "Mathematics",
                "Every equation the engine executes, and every metric used to judge it. Each is "
                "followed by a plain-English reading of what it actually does.")

        def m_pre(b: Body):
            b.h3("Baseline removal — asymmetric least squares")
            b.eq(r"$z = \mathrm{argmin}_z \; \sum_b \omega_b (r_b - z_b)^2 \; + \; "
                 r"\lambda \sum_b (\Delta^2 z_b)^2$")
            b.eq(r"$\omega_b = p \;\; \mathrm{if} \;\; r_b > z_b, \quad "
                 r"\omega_b = 1 - p \;\; \mathrm{otherwise}$", size=11)
            b.p("With lambda = 1e5, p = 0.01 and ten iterations. The corrected spectrum is "
                "max(r - z, 0).")
            b.plain("Two forces are balanced. The first pulls the curve toward the data, but "
                    "asymmetrically: points above the curve get weight 0.01 and points below get "
                    "0.99, so peaks are largely ignored while the valleys between them pull hard. "
                    "The second term, scaled by the large lambda, punishes curvature and forces "
                    "smoothness. What survives is the slow background, not the sharp bands. A "
                    "symmetric fit would be dragged up into the peaks and would subtract away "
                    "real signal.")
            b.h3("Normalisation and quality")
            b.eq(r"$x = \frac{\max(r, 0)}{\|\max(r,0)\|_2 + \varepsilon}, \qquad "
                 r"\mathrm{SNR} = \frac{\max_b x_b}{\mathrm{median}_b |x_{b+1} - x_b| + "
                 r"\varepsilon}$")
            b.plain("Doubling the laser power doubles every intensity, so scaling to unit length "
                    "throws that away and keeps only the shape — which is where chemistry lives. "
                    "It is also why nothing downstream may be read as a concentration. For the "
                    "noise estimate, using the MEDIAN hop rather than the mean stops real peaks, "
                    "which produce a few enormous hops, from inflating it.")
        text_page(pdf, "Part IV", "Preprocessing", m_pre)

        def m_nmf(b: Body):
            b.h3("The motif dictionaries — non-negative matrix factorisation")
            b.eq(r"$\min_{W \geq 0,\; H \geq 0} \; \| X - W H \|_F^2$")
            b.p("H holds the motifs, W their per-spectrum weights, and the Frobenius norm is the "
                "sum of squared entrywise errors. Learned once in Phase 01, within each chemistry "
                "class, and frozen.")
            b.plain("This asks: what small set of building-block spectra can be ADDED TOGETHER, "
                    "never subtracted, to reproduce what was observed? The non-negativity is a "
                    "scientific constraint. A real mixture spectrum is a sum of its components; "
                    "there is no such thing as negative-two-parts glucose. Methods that allow "
                    "negative weights fit better and produce components that mean nothing.")
            b.h3("Merging motifs — the consensus graph")
            b.eq(r"$W_{ij} = \prod_f \left( \max(F^{(f)}_{ij},\, 10^{-3}) \right)^{\alpha_f}, "
                 r"\qquad \sum_f \alpha_f = 1$")
            b.p("Seven similarity features per motif pair: band overlap 0.25, spectral cosine "
                "0.20, peak agreement 0.15, bootstrap co-occurrence 0.15, substitutability 0.10, "
                "activation co-occurrence 0.10, provenance overlap 0.05.")
            b.plain("A geometric mean is a strict form of 'all of the above'. Under an ordinary "
                    "average, a spectral cosine of 0.95 could carry an edge whose every other "
                    "channel was near zero — two motifs that merely look alike would merge. Under "
                    "a geometric mean, any single near-zero channel drags the whole weight toward "
                    "zero. The floor at 1e-3 makes the strongest possible veto a stated constant "
                    "rather than an accident of floating-point representation.")
        text_page(pdf, "Part IV", "Learning the dictionaries", m_nmf)

        def m_proj(b: Body):
            b.h3("Projection")
            b.eq(r"$a = \mathrm{argmin}_{a \geq 0} \; \| x - a^{\top} C \|_2^2$")
            b.p("Solved by non-negative least squares against the frozen 49-motif dictionary C. "
                "Diagnostics, with the reconstruction x-hat = a-transpose C:")
            b.eq(r"$\mathrm{EV} = 1 - \frac{\|x - \hat{x}\|_2^2}{\|x\|_2^2 + \varepsilon}, "
                 r"\qquad \rho = \frac{\|x - \hat{x}\|_2}{\|x\|_2 + \varepsilon}$")
            b.plain("'Which mixture of the 49 frozen motifs best explains this spectrum, given "
                    "that you cannot use a negative amount of anything?' There is NO "
                    "regularisation parameter, and that absence is deliberate: a penalty weight "
                    "would be a quantity that could be tuned per spectrum, and a tunable quantity "
                    "on the inference path is a place where results can be nudged.")
            b.h3("Molecular retrieval")
            b.eq(r"$S_m = \mathrm{clip}\left( \hat{R}_m \cdot \hat{q},\, 0,\, 1 \right) "
                 r"= \sum_{j=1}^{49} \hat{q}_j \, \hat{R}_{mj}$")
            b.plain("Cosine similarity asks whether two activation patterns point in the same "
                    "DIRECTION, ignoring how large they are. The second equality is why the "
                    "engine can explain itself: an inner product is literally a sum of per-motif "
                    "terms, so the score decomposes exactly into 'how much each motif "
                    "contributed'. The engine asserts that the parts sum to the whole within "
                    "1e-9 for every candidate it reports. Nothing is hidden.")
        text_page(pdf, "Part IV", "Projection and retrieval", m_proj)

        def m_chem(b: Body):
            b.p("The selected model is D:A_max_idf with lambda 0.5, chosen by nested "
                "molecule-grouped cross-validation on inner-fold macro F1.")
            b.h3("Fine level")
            b.eq(r"$e_c = w_c \cdot \max_{m \in M_c} S_m, \qquad "
                 r"w_c = \frac{\log(N / |M_c|) + 1}{\mathrm{mean}_{c'}[\log(N/|M_{c'}|)+1]}$")
            b.plain("A class's evidence is the similarity of its SINGLE BEST-MATCHING member, not "
                    "an average — averaging would penalise a chemically diverse class such as "
                    "free amino acids for containing members unlike the query, which is not a "
                    "defect. The inverse-frequency weight offsets the opposite bias: with 80 "
                    "peptide references and 3 nucleic-acid-polymer references, the large class "
                    "would otherwise win simply by having more chances.")
            b.h3("Broad level, soft routing")
            b.eq(r"$e_c \; \leftarrow \; e_c \cdot \left( B_{\beta(c)} \right)^{\lambda}, "
                 r"\qquad \lambda = 0.5$")
            b.plain("Coarse chemistry is easier to get right than fine chemistry, so it is used "
                    "as a HINT — but only as a hint. Because every broad evidence is strictly "
                    "positive, multiplying can never zero a fine class out; a molecule whose "
                    "superclass was misjudged is penalised, not excluded. A hard filter would "
                    "make every coarse error permanently unrecoverable.")
            b.h3("The radar")
            b.eq(r"$\tilde{e} = \frac{e}{\sum_c e_c + \varepsilon}$")
            b.note("This quantity is RELATIVE BIOCHEMICAL EVIDENCE. Not a concentration, not an "
                   "abundance, not a mixture fraction. Normalisation destroyed absolute scale in "
                   "the first stage, and a similarity is not a quantity of material.")
        text_page(pdf, "Part IV", "Chemistry Evidence", m_chem)

        def m_cal(b: Body):
            b.h3("Temperature scaling")
            b.eq(r"$p_c = \frac{\exp(e_c / T)}{\sum_{c'} \exp(e_{c'} / T)}, \qquad T = 0.4538$")
            b.plain("One number rescales how sharply evidence becomes probability. It cannot "
                    "change the RANKING — the winner is the winner at any T — only how confident "
                    "the engine claims to be. That is exactly what is wanted from a calibrator: "
                    "fix the honesty of the numbers without touching the answer.")
            b.h3("Expected calibration error")
            b.eq(r"$\mathrm{ECE} = \sum_{b=1}^{B} \frac{|\mathcal{B}_b|}{n} \, "
                 r"\left| \mathrm{acc}(\mathcal{B}_b) - \mathrm{conf}(\mathcal{B}_b) \right|$")
            b.plain("Of everything the engine called 80% likely, was it right 80% of the time? "
                    "ECE averages that gap. But ECE ALONE IS DANGEROUS: a model that outputs the "
                    "base rate for every input has near-perfect ECE and is useless. V7 hit this "
                    "exact trap in Phase 05. Selection therefore uses log loss or Brier subject "
                    "to floors on sharpness and discrimination, so a constant predictor is "
                    "disqualified before ECE is ever consulted.")
            b.h3("Sharpness, discrimination, risk-coverage")
            b.eq(r"$\mathrm{sharpness} = \mathrm{Var}_i(p_{i,\hat{y}}), \qquad "
                 r"\mathrm{disc} = \Pr(p^{\mathrm{correct}} > p^{\mathrm{wrong}})$")
            b.p(f"At full coverage retrieval accuracy is {rc.accuracy.iloc[0]:.3f}. Abstaining "
                f"below a margin of {rc.threshold.iloc[15]:.3f} keeps "
                f"{rc.coverage.iloc[15]:.0%} of spectra at {rc.accuracy.iloc[15]:.3f}.")
            b.plain("Sharpness asks whether the confidences vary at all; a constant predictor "
                    "scores zero. Discrimination asks whether the engine is more confident when "
                    "it is right than when it is wrong. The risk-coverage curve turns both into "
                    "an operational rule: answer only above this threshold, and here is what you "
                    "get.")
        text_page(pdf, "Part IV", "Calibration", m_cal)

        def m_metrics(b: Body):
            b.h3("Retrieval metrics")
            b.eq(r"$\mathrm{top}_k = \frac{1}{n}\sum_i \mathbb{1}[\mathrm{rank}_i "
                 r"\leq k], \qquad \mathrm{MRR} = \frac{1}{n}\sum_i "
                 r"\frac{1}{\mathrm{rank}_i}$")
            b.plain("Top-k is the blunt question — was the answer in the first k? MRR rewards "
                    "being close: rank 1 scores 1.0, rank 2 scores 0.5, rank 10 scores 0.1, so a "
                    "system that is usually second beats one that is usually tenth.")
            b.h3("Classification metrics")
            b.eq(r"$\mathrm{macro\,F1} = \frac{1}{16}\sum_c \frac{2 P_c R_c}{P_c + R_c}, "
                 r"\qquad \mathrm{balanced\,acc} = \frac{1}{16}\sum_c R_c$")
            b.plain("Plain accuracy is dominated by the big classes — with 80 peptide spectra and "
                    "3 nucleic-acid-polymer spectra, a model could ignore the small classes "
                    "entirely and still look good. Macro F1 gives every class one vote. That is "
                    "why macro F1 (0.811) sits below top-1 (0.851): the engine IS worse on the "
                    "rare classes, and macro F1 refuses to let that hide.")
            b.h3("Uncertainty")
            b.eq(r"$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$")
            b.plain("McNemar's test. When two systems run on the SAME spectra, only the cases "
                    "where they disagree carry information — the ones both get right tell you "
                    "nothing about which is better. This test is the reason Phase 06.5 did not "
                    "adopt the geometry layer: the raw improvement of +0.016 came from six "
                    "spectra, and McNemar returned p = 0.180.")
            b.p("Bootstrap confidence intervals resample MOLECULES, not spectra. Replicate "
                "spectra of glucose are not independent evidence about how well the engine "
                "handles sugars, and treating them as independent would shrink the interval to "
                "something the data does not support.")
        text_page(pdf, "Part IV", "Metrics and uncertainty", m_metrics)

        # ── Part V ───────────────────────────────────────────────────────────
        divider(pdf, "V", "Limits, decision and glossary",
                "What this engine may and may not be used to claim, what the decision gate "
                "concluded, and the vocabulary used throughout.")

        def p_limits(b: Body):
            b.lead("These are constraints on what the output MEANS, not caveats about its "
                   "quality.")
            b.bullet("The radar is relative biochemical evidence. Not a concentration, not an "
                     "abundance, not a mixture decomposition.")
            b.bullet("Pure Raman reference spectra only. Nothing here licenses a SERS, serum, "
                     "plasma, EV, tissue or pathogen claim. Transfer to those regimes is "
                     "unmeasured in V7 and must be established separately.")
            b.bullet("A peak is not a molecule. Retrieval returns ranked candidates with a score "
                     "decomposition, deliberately, and top-1 of 0.605 is the honest figure for "
                     "how often the single best guess is right.")
            b.bullet("The sixteen classes are a curated cut through a continuum. Phase 06.5 "
                     "showed the space has NO preferred cluster count; the ontology is a "
                     "reporting convention that generalises well, not a natural kind.")
            b.bullet("In-sample chemistry figures describe the shipped fit, not expected "
                     "performance on new molecules.")
            b.bullet("The engine cannot tell that the true molecule is absent from its bank. The "
                     "'unknown' and 'outlier' warnings detect unexplained SPECTRA, not unknown "
                     "MOLECULES. Using them as an open-set detector would be a misuse.")
            b.bullet("Class-prior bias remains open. Class sizes range from 3 to 80 spectra and "
                     "the chemistry layer inherits that prior; the inverse-frequency weight "
                     "mitigates it without eliminating it.")
            b.note("Every number in this document comes from the same 375 spectra that built the "
                   "atlas. The robustness study is the closest thing to an external test and it "
                   "is synthetic. The correct reading of Phase 09 is 'the architecture is "
                   "internally sound and faithfully implemented' — not 'the architecture works'. "
                   "The second claim needs data V7 does not have.")
        text_page(pdf, "Part V", "Interpretation limits", p_limits)

        def p_decision(b: Body):
            b.lead("Phase 09 was declared a packaging phase before any code ran. Its gate asks "
                   "whether the frozen architecture is faithfully implemented and honestly "
                   "validated — not whether some change is worth adopting.")
            b.h3("Outcome A — ship the engine and freeze V7")
            b.p("Retrieval reproduces the Phase 05/08 baseline to the digit. The chemistry layer, "
                "the CSM projection and the LSM projection all reproduce their source phases. "
                "Integrating eight phases into one object surfaced no contradiction between them, "
                "which is the strongest single piece of evidence that the chain of decisions "
                "behind the rebuild is sound.")
            b.h3("What is now frozen")
            b.p("Canonical preprocessing; the 50-motif and 49-motif dictionaries; retrieval by "
                "CSM cosine; the Chemistry Evidence model and its calibrator; the confidence and "
                "warning rules; the engine's API. Changing any of them changes a fingerprint and "
                "the engine refuses to load — the freeze is enforced by code, not convention.")
            b.h3("What is not frozen")
            b.p("BSV2, which sits downstream and may be revised freely. Presentation choices, "
                "including where an operator sets an abstention point. And the corpus, whose "
                "expansion is the intended path forward.")
            b.h3("Where the work goes next")
            b.bullet("Corpus. 66 of 154 molecules have a single spectrum, which alone caps "
                     "molecule top-1 at 0.819. This is the largest single lever available and it "
                     "needs no engine change.")
            b.bullet("Transfer. Establish what survives the move from pure Raman to SERS, serum "
                     "and EV — the question GAIRA ultimately exists to answer.")
            b.bullet("Open-set behaviour. Hold out whole molecules from the bank and measure "
                     "whether confidence drops. This is currently an argued limitation rather "
                     "than a measured one, and it is the limitation most likely to matter in use.")
            b.note("Four independent attempts to add a layer above the CSM have now failed on "
                   "measurement. A fifth would be a re-run of the same experiment under a "
                   "different name. The productive next work is scientific, not architectural.",
                   color=GOOD, label="Recommendation")
        text_page(pdf, "Part V", "The decision", p_decision)

        def p_gloss(b: Body):
            b.kv([("Raman spectrum", "the pattern of light-frequency shifts produced by "
                                     "molecular vibrations in a sample"),
                  ("cm-1 (wavenumber)", "the unit of that shift; this engine uses 450 to 1800"),
                  ("asLS", "asymmetric least squares — fits the fluorescence background"),
                  ("Savitzky-Golay", "a smoother that fits small polynomials, preserving peaks"),
                  ("L2 normalisation", "scaling a spectrum to unit length, discarding intensity"),
                  ("NMF", "non-negative matrix factorisation — finds add-only building blocks"),
                  ("LSM", "Local Spectral Motif — one of 50 basis spectra learned per class"),
                  ("CSM", "Consensus Spectral Motif — one of 49 merged motifs; the canonical "
                          "coordinates"),
                  ("activation", "how much of each motif a spectrum contains; never negative"),
                  ("explained variance", "the fraction of a spectrum the dictionary accounts for"),
                  ("cosine similarity", "how closely two patterns point in the same direction"),
                  ("Chemistry Evidence", "the 16 numbers, one per chemistry family, that the "
                                         "radar displays"),
                  ("BSV2", "a compressed description of Chemistry Evidence; NOT on the inference "
                           "path"),
                  ("calibration", "making stated confidence match observed correctness"),
                  ("ECE", "expected calibration error — the average gap between the two"),
                  ("held out", "evaluated on molecules the model never saw during fitting"),
                  ("in-sample", "evaluated on the same data used for fitting; not a performance "
                                "claim"),
                  ("macro F1", "an average over classes that gives small classes equal weight"),
                  ("top-k", "the fraction of queries whose answer appeared in the first k "
                            "results"),
                  ("MRR", "mean reciprocal rank — rewards being close, not just being first"),
                  ("fingerprint", "a hash that changes if a frozen artefact changes at all"),
                  ("gate", "a pass/fail condition declared before a run and never relaxed after")],
                 keyw=0.22, size=8.6)
        text_page(pdf, "Part V", "Glossary", p_gloss)

        def p_colophon(b: Body):
            b.p("Every figure, table and number in this document was generated from the committed "
                "artifacts of Phase 09 by the scripts listed below. No value was typed by hand.")
            b.kv([("engine", "src/gaira/v7/canonical/engine.py"),
                  ("validation", "results/v7_rebuild/phase09/code/run_phase09.py"),
                  ("figures", "results/v7_rebuild/phase09/code/make_figures.py"),
                  ("this document", "results/v7_rebuild/phase09/code/make_pdf.py"),
                  ("tests", "tests/test_v7_phase09.py"),
                  ("", ""),
                  ("atlas fingerprint", e["atlas_fingerprint"]),
                  ("frozen atlas", e["fingerprints"]["atlas"]),
                  ("frozen LSM registry", e["fingerprints"]["lsm"]),
                  ("frozen CSM registry", e["fingerprints"]["csm"]),
                  ("frozen Phase 05 engine", e["fingerprints"]["engine"]),
                  ("", ""),
                  ("completed", state["finished"][:19].replace("T", " ") + " UTC")], keyw=0.28)
            b.h3("Companion documents")
            for d in ("PHASE_09_REPORT.md — the findings",
                      "PHASE_09_ENGINE_SPEC.md — the engine, precisely enough to reimplement",
                      "PHASE_09_MATHEMATICAL_APPENDIX.md — every equation, with intuition",
                      "PHASE_09_DECISION_GATE.md — the criteria and the decision",
                      "PHASE_09_SCIENTIFIC_AUDIT.md — what is strongly supported, weakly "
                      "supported, and unsupported"):
                b.bullet(d, marker="·")
        text_page(pdf, "", "Colophon and provenance", p_colophon)

        d = pdf.infodict()
        d["Title"] = "GAIRA V7 Phase 09 — The canonical inference engine"
        d["Subject"] = "The frozen canonical GAIRA inference engine: 16 figures, validation, "\
                       "mathematics and limits"
        d["Keywords"] = "GAIRA V7 Raman canonical engine inference chemistry evidence radar"
    print(f"  {out.relative_to(REPO)}  ({out.stat().st_size / 1e6:.1f} MB, {_page_no[0]} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
