"""V5 PDF report — abstraction recovery. reportlab + DejaVu. Reads committed V5 tables/figures."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib as _mpl

_FT = Path(_mpl.__file__).parent / "mpl-data/fonts/ttf"
for nm, fn in [("DJ", "DejaVuSans.ttf"), ("DJ-B", "DejaVuSans-Bold.ttf"), ("DJ-O", "DejaVuSans-Oblique.ttf")]:
    pdfmetrics.registerFont(TTFont(nm, str(_FT / fn)))
pdfmetrics.registerFontFamily("DJ", normal="DJ", bold="DJ-B", italic="DJ-O", boldItalic="DJ-B")
FN, FN_B, FN_O = "DJ", "DJ-B", "DJ-O"

BASE = Path("/Users/surajpg/projects/GAIRA/results/v5_rebuild/abstraction_recovery_v5")
OUT = BASE / "GAIRA_Pure_AgSERS_Abstraction_Recovery_V5.pdf"
FP = "09ed804a40836f4a05a91ba10900cded"
S = json.loads((BASE / "artifacts/abstraction_summary.json").read_text())
LAD = pd.read_csv(BASE / "tables/recovery_by_abstraction_level.csv")
CLS = pd.read_csv(BASE / "tables/subclass_classification_results.csv")
FAMB = pd.read_csv(BASE / "tables/family_abstraction_breakdown.csv")

INK = colors.HexColor("#1b2430"); MUTED = colors.HexColor("#5b6472"); BLUE = colors.HexColor("#0072B2")
VERM = colors.HexColor("#D55E00"); GREEN = colors.HexColor("#009E73"); LIGHT = colors.HexColor("#eef2f6")
ss = getSampleStyleSheet()
def _mk(n, **k): return ParagraphStyle(n, parent=ss["Normal"], **k)
TITLE = _mk("t", fontName=FN_B, fontSize=24, leading=28, textColor=INK, spaceAfter=6)
H1 = _mk("h1", fontName=FN_B, fontSize=15, leading=19, textColor=INK, spaceBefore=13, spaceAfter=5)
BODY = _mk("b", fontName=FN, fontSize=9.6, leading=13.8, textColor=INK, spaceAfter=6)
CAP = _mk("c", fontName=FN_O, fontSize=8.2, leading=10.4, textColor=MUTED, spaceAfter=12)
SM = _mk("sm", fontName=FN, fontSize=7.6, leading=9.6, textColor=INK)
SMB = _mk("smb", fontName=FN_B, fontSize=7.6, leading=9.6, textColor=colors.white)
UW = letter[0] - 1.8 * inch
def P(t, s=BODY): return Paragraph(t, s)
def callout(kind, text):
    col = {"note": BLUE, "warn": VERM, "good": GREEN, "take": INK}[kind]
    bg = {"note": colors.HexColor("#f2f7fb"), "warn": colors.HexColor("#fdf3ee"), "good": colors.HexColor("#eef8f4"), "take": INK}[kind]
    tc = colors.white if kind == "take" else INK
    t = Table([[Paragraph(text, _mk("cx", fontName=FN, fontSize=9.3, leading=13, textColor=tc))]], colWidths=[UW])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, col),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)])); return t
def fig(name, cap, maxw=UW, maxh=7.0 * inch):
    p = BASE / "figures" / name
    if not p.exists(): return P(f"[{name} missing]", CAP)
    iw, ih = PILImage.open(p).size; ar = ih / iw; w = maxw; h = w * ar
    if h > maxh: h = maxh; w = h / ar
    return KeepTogether([Image(str(p), width=w, height=h), Spacer(1, 3), P(cap, CAP)])
def table(dfr, cols, headers, cw, fs=7.6):
    cell = _mk("cell", fontName=FN, fontSize=fs, leading=fs + 1.6, textColor=INK)
    head = _mk("head", fontName=FN_B, fontSize=fs, leading=fs + 1.6, textColor=colors.white)
    data = [[Paragraph(h, head) for h in headers]] + [[Paragraph(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]), cell) for c in cols] for _, r in dfr.iterrows()]
    t = Table(data, colWidths=cw, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), INK), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9dee4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4), ("LEFTPADDING", (0, 0), (-1, -1), 4)])); return t

story = []
story += [Spacer(1, 1.5 * inch), P("GAIRA", TITLE),
          P("Pure Ag-SERS Abstraction Recovery (V5)", _mk("st", fontName=FN_B, fontSize=15, leading=19, textColor=BLUE, spaceAfter=12)),
          P("From exact molecular identity to recoverable biochemical abstraction. When exact identity "
            "is lost after Ag-SERS, does the correct broader chemistry — component, motif, subclass, "
            "theme — still survive?", _mk("sub", fontName=FN, fontSize=12, leading=16, textColor=MUTED)),
          Spacer(1, 0.3 * inch)]
cov = Table([[P(f"<b>Frozen atlas fingerprint</b>  {FP} · reproduces V4 exact identity (7/3/4)", SM)],
             [P("Subclass is an EVALUATION OVERLAY, never a new GAIRA axis · SERS validates, never trains", SM)],
             [P("Headline: expected motif/theme often PRESENT (top-3: MSS 40%, theme 49%) but rarely "
                "SPECIFIC (2/48, 1/51); cross-modal class recovery at chance; only functional "
                "perturbation (3) recovers class chemistry beyond exact identity.", SM)]], colWidths=[UW])
cov.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("BOX", (0, 0), (-1, -1), 0.5, BLUE),
    ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("LEFTPADDING", (0, 0), (-1, -1), 12)]))
story += [cov, PageBreak()]

story += [P("1 · Executive summary", H1),
          P("When exact analyte identity is lost after Ag-SERS, does the correct broader chemistry "
            "survive? The expected motif/theme is often PRESENT in the Ag-SERS top-3 (MSS 19/48 = 40%, "
            "theme 25/51 = 49%), but that presence is <b>not analyte-specific</b>.", BODY),
          callout("warn", "Specific, null-and-background-adjusted recovery stays rare at every level "
            "(component 2/51, MSS 2/48, theme 1/51), and cross-modal subclass/family classification is "
            "AT CHANCE (balanced accuracy 0.03–0.18, all permutation p n.s.). <b>Presence ≠ recovery.</b>"),
          callout("good", "A Raman→Raman control proves the taxonomy is separable and that abstraction "
            "DOES help within Raman (subclass 0.23 → family 0.35 → theme 0.42) — the Ag-SERS modality "
            "gap collapses it. The failure is the modality, not the taxonomy or the method."),
          fig("fig02_recovery_by_level.png", "Figure. Graded recovery: PRESENCE (light) rises with "
              "abstraction; SPECIFIC recovery (dark) and classification (grey, at chance) stay low.")]
story.append(PageBreak())

story += [P("2 · The decisive control — abstraction helps in Raman, collapses across the modality gap", H1),
          fig("fig05_classification_control.png", "Figure. Raman→Raman control (green) rises with "
              "abstraction and sits well above chance; Ag-SERS cross-modal (bars) sits near chance."),
          P("This isolates the cause: the chemical taxonomy IS separable and abstraction genuinely "
            "improves recovery — but only within the Raman training modality. The Ag-SERS surface "
            "reshaping (the purine attractor and adsorption selection) destroys the class structure, so "
            "cross-modal subclass/family recovery is at chance.", BODY)]
story.append(PageBreak())

story += [P("3 · Recovery ladder & classification", H1),
          table(LAD, ["level", "tier", "n_recovered", "denominator", "fraction"],
                ["Level", "Tier", "n", "denom", "frac"], [1.5*inch, 2.2*inch, 0.5*inch, 0.7*inch, 0.6*inch], fs=7.2),
          Spacer(1, 6),
          table(CLS[["granularity", "space", "balanced_accuracy", "macro_f1", "accuracy"]],
                ["granularity", "space", "balanced_accuracy", "macro_f1", "accuracy"],
                ["gran", "space", "bal acc", "macro-F1", "acc"], [1.0*inch, 2.4*inch, 0.9*inch, 0.8*inch, 0.8*inch], fs=7.0)]
story.append(PageBreak())

story += [P("4 · Per-analyte ladder & highest recovered level", H1),
          fig("fig03_recovery_ladder.png", "Figure. Per-analyte ladder — dark ✓ specific, light ✓ "
              "present-only, · not tested, u unassigned. Presence-only cells dominate.", maxh=8.2*inch)]
story.append(PageBreak())

story += [P("5 · MSS motif & broad theme recovery", H1),
          fig("fig06_mss_motif_ranking.png", "Figure. MSS expected-motif recovery, null-adjusted (dark = "
              "specific, n=2; light = present).", maxh=6.6*inch),
          fig("fig08_theme_recovery.png", "Figure. Expected-theme rank (25/51 top-3) and enrichment "
              "(1 specific).", maxh=3.4*inch)]
story.append(PageBreak())

story += [P("6 · Purine attractor, perturbation & matrix", H1),
          fig("fig09_purine_correction.png", "Figure. Non-purines that retain expected chemistry despite "
              "purine pull — genuine presence vs attraction.", maxh=3.6*inch),
          fig("fig11_perturbation_overlay.png", "Figure. Functional perturbation recovers class chemistry "
              "static Ag-SERS cannot (adenine, ergothioneine, urate).", maxh=3.2*inch),
          fig("fig12_abstraction_vs_serum.png", "Figure. Pure abstraction recovery does not predict serum "
              "strength — matrix is a separate property.", maxh=3.4*inch)]
story.append(PageBreak())

story += [P("7 · Representative analytes", H1),
          fig("fig10_representative.png", "Figure. Spectrum → themes → evidence for 12 representatives.", maxh=8.4*inch)]
story.append(PageBreak())

story += [P("8 · What GAIRA can & cannot claim · limitations · conclusions", H1),
          callout("note", "<b>Can:</b> report a broad theme/motif is PRESENT (top-3) for ~40–50% (broad "
            "interpretation); specifically recover exact identity for a strong-chemisorber minority "
            "(7/51); functionally validate 3 analytes. <b>Cannot:</b> claim molecular identification "
            "from motif/theme presence; claim class-discriminative subclass/family recovery from pure "
            "Ag-SERS (at chance); infer perturbation/matrix for untested analytes."),
          P("<b>Limitations.</b> Raman-trained atlas; surface-selection effects; purine attractor; low "
            "exact identity; subclass imbalance (15 exploratory singletons); 3 perturbation cases; no "
            "Au-SERS observation model; confidence ≠ identifiability; cross-modal centroid classification "
            "carries the global-shift confound (reported with the Raman control).", BODY),
          P("<b>Conclusions.</b> Broad biochemical presence frequently survives Ag-SERS, but it is not "
            "analyte-specific or class-discriminative — abstraction raises apparent presence via a shared "
            "attractor, while the modality gap collapses genuine class recovery to chance. Specific "
            "recovery beyond a strong-chemisorber minority comes only from functional perturbation. "
            "<b>Recommended next experiment:</b> a learned Raman→SERS observation model, and dynamic "
            "perturbation (DART) as the route to class-specific recovery.", BODY),
          P("<b>Provenance.</b> Frozen atlas " + FP + " unchanged; reproduces V4 identity (7/3/4); "
            "deterministic; method in EVALUATION_HIERARCHY_AND_METRICS.md; overlay provenance in "
            "ANALYTE_CLASSIFICATION_PROVENANCE.md.", BODY)]


def on_page(canvas, doc):
    canvas.saveState(); canvas.setFont(FN, 8); canvas.setFillColor(MUTED)
    canvas.drawString(0.9 * inch, 0.55 * inch, "GAIRA · Pure Ag-SERS Abstraction Recovery (V5)")
    canvas.drawRightString(letter[0] - 0.9 * inch, 0.55 * inch, str(doc.page))
    canvas.setStrokeColor(colors.HexColor("#d9dee4")); canvas.line(0.9 * inch, 0.72 * inch, letter[0] - 0.9 * inch, 0.72 * inch)
    canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=letter, topMargin=0.8 * inch, bottomMargin=0.9 * inch,
                  leftMargin=0.9 * inch, rightMargin=0.9 * inch, title="GAIRA V5 — Abstraction Recovery").build(
    story, onFirstPage=on_page, onLaterPages=on_page)
print("PDF ->", OUT)
