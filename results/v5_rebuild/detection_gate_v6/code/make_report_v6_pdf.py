"""V6 PDF report — detection gate. reportlab + DejaVu. Reads committed V6 tables/figures."""
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

BASE = Path("/Users/surajpg/projects/GAIRA/results/v5_rebuild/detection_gate_v6")
OUT = BASE / "GAIRA_Pure_AgSERS_Evaluation_V6.pdf"
FP = "09ed804a40836f4a05a91ba10900cded"
S = json.loads((BASE / "artifacts/detection_summary.json").read_text())
R = json.loads((BASE / "artifacts/restricted_hierarchy_summary.json").read_text())
LAD = pd.read_csv(BASE / "tables/recovery_detectable_vs_all.csv")
INK = colors.HexColor("#1b2430"); MUTED = colors.HexColor("#5b6472"); BLUE = colors.HexColor("#0072B2")
VERM = colors.HexColor("#D55E00"); GREEN = colors.HexColor("#009E73"); LIGHT = colors.HexColor("#eef2f6")
ss = getSampleStyleSheet()
def _mk(n, **k): return ParagraphStyle(n, parent=ss["Normal"], **k)
TITLE = _mk("t", fontName=FN_B, fontSize=24, leading=28, textColor=INK, spaceAfter=6)
H1 = _mk("h1", fontName=FN_B, fontSize=15, leading=19, textColor=INK, spaceBefore=13, spaceAfter=5)
BODY = _mk("b", fontName=FN, fontSize=9.6, leading=13.8, textColor=INK, spaceAfter=6)
CAP = _mk("c", fontName=FN_O, fontSize=8.2, leading=10.4, textColor=MUTED, spaceAfter=12)
SM = _mk("sm", fontName=FN, fontSize=7.6, leading=9.6, textColor=INK)
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
          P("Pure Ag-SERS Evaluation (V6)", _mk("st", fontName=FN_B, fontSize=15, leading=19, textColor=BLUE, spaceAfter=12)),
          P("A detection gate before recovery — separating measurement failure (invisible on silver) "
            "from representation failure (measured but chemistry not recovered).",
            _mk("sub", fontName=FN, fontSize=12, leading=16, textColor=MUTED)), Spacer(1, 0.3 * inch)]
cov = Table([[P(f"<b>Frozen atlas fingerprint</b>  {FP} · reuses V5 recovery flags unchanged", SM)],
             [P(f"<b>{S['n_pass']}/51 analytes pass the Stage-0 detection gate</b> · {S['n_fail']} undetectable "
                "· thresholds validated before freezing (validate_detection.ipynb)", SM)],
             [P("Among detectable analytes, exact identity ~doubles (14%→23%) and presence rises, but "
                "analyte-specific recovery stays low — the residual failure is representational. A learned "
                "Raman→SERS transfer model is justified for ~11 analytes; the rest need a better substrate.", SM)]], colWidths=[UW])
cov.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("BOX", (0, 0), (-1, -1), 0.5, BLUE),
    ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("LEFTPADDING", (0, 0), (-1, -1), 12)]))
story += [cov, PageBreak()]

story += [P("1 · Executive summary & detection philosophy", H1),
          P("V5 evaluated all 51 analytes equally — including analytes whose Ag-SERS is essentially "
            "blank. That conflates measurement failure (invisible on silver) with representation "
            "failure (measured but chemistry not recovered). V6 inserts a Stage-0 detection gate that "
            "asks a MEASUREMENT question: does this Ag-SERS spectrum contain reproducible analyte "
            "information above noise/background?", BODY),
          callout("take", f"{S['n_pass']}/51 analytes pass; {S['n_fail']} are undetectable. Only "
            "detectable analytes are eligible for identity/motif/theme evaluation — mirroring how "
            "spectroscopy is actually done: confirm signal before interpreting it."),
          fig("fig01_detection_hierarchy.png", "Figure 1. Stage 0 gates every later stage.", maxh=5.6*inch)]
story.append(PageBreak())

story += [P("2 · Detection metrics, thresholds & distribution", H1),
          P("Deterministic weighted Detection Confidence (0–1), NO ML: replicate Pearson (0.45), "
            "Spearman (0.10), peak SNR (0.20), variance concentration (0.15), reproducible peaks (0.10). "
            "Tiers GOOD ≥0.65 / MODERATE ≥0.50 / POOR ≥0.40 / UNDETECTABLE <0.40; pass ≥0.50. Replicate "
            "cosine was rejected — baseline-inflated (0.93–0.99 for all). Validated before freezing: "
            "anchors pass (xanthine 0.99, ergothioneine 0.97, urate 0.89, adenine 0.63) and fail "
            "(glucose 0.43, tyrosine 0.42, oleate 0.33) for adsorption reasons.", BODY),
          fig("fig02_detection_distribution.png", "Figure 2. Detection confidence per analyte with tier bands.")]
story.append(PageBreak())

story += [P("3 · Representative spectra — why analytes pass or fail", H1),
          fig("fig03_representative_spectra.png", "Figure 3. Ag-SERS (blue) · blank (grey) · difference "
              "(red) · peaks (▼). Top row PASS (sharp reproducible peaks); bottom rows FAIL (blank-like).", maxh=7.4*inch)]
story.append(PageBreak())

story += [P("4 · Hierarchical recovery — detectable only", H1),
          table(LAD, ["level", "all_n", "all_frac", "detectable_n", "detectable_frac", "gain"],
                ["Level", "all n", "all frac", "det n", "det frac", "gain"],
                [1.9*inch, 0.7*inch, 0.8*inch, 0.7*inch, 0.8*inch, 0.6*inch], fs=7.4),
          Spacer(1, 6),
          fig("fig05_recovery_detectable.png", "Figure 5. Removing measurement failure lifts every "
              "level, but analyte-specific recovery stays low → the residual is representational.")]
story.append(PageBreak())

story += [P("5 · Transfer-model decision & roadmap", H1),
          fig("fig07_transfer_decision.png", "Figure 7. Decision tree: A measurement-limited · B "
              "representation-limited · C already recoverable.", maxh=4.4*inch),
          fig("fig08_transfer_roadmap.png", "Figure 8. Learned Raman→SERS transfer roadmap — ~11 "
              "'potentially recoverable' are the target set.", maxh=3.4*inch)]
story.append(PageBreak())

story += [P("6 · Recommendations, limitations & conclusions", H1),
          callout("note", "<b>Recommendations.</b> Build a learned Raman→SERS transfer model on the ~11 "
            "detectable, representation-limited analytes (Case B promising); the 29 undetectable analytes "
            "need a better substrate, not a model. Extend dynamic perturbation (DART) — the only route "
            "that recovered class chemistry in V5."),
          P("<b>Limitations.</b> No pure Ag-colloid buffer blank (serum blank used as the Ag background); "
            "5 replicates limit resolution; the weighting is a transparent, validated choice, not unique; "
            "the gate is conservative (creatinine and thymine were identity-recovered yet fall just below "
            "it); Raman-trained atlas; no learned modality model yet.", BODY),
          P("<b>Conclusions.</b> A measurement gate before interpretation cleanly separates 'we can't "
            "see it' from 'we can't recover it.' Half the analytes are simply invisible on this substrate; "
            "among the visible ones recovery improves but specific chemistry still largely fails to "
            "transfer — a real representation gap that a learned model could target for ~11 analytes. "
            "Frozen atlas " + FP + " unchanged; reuses V5 recovery flags; deterministic.", BODY)]


def on_page(canvas, doc):
    canvas.saveState(); canvas.setFont(FN, 8); canvas.setFillColor(MUTED)
    canvas.drawString(0.9 * inch, 0.55 * inch, "GAIRA · Pure Ag-SERS Evaluation (V6)")
    canvas.drawRightString(letter[0] - 0.9 * inch, 0.55 * inch, str(doc.page))
    canvas.setStrokeColor(colors.HexColor("#d9dee4")); canvas.line(0.9 * inch, 0.72 * inch, letter[0] - 0.9 * inch, 0.72 * inch)
    canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=letter, topMargin=0.8 * inch, bottomMargin=0.9 * inch,
                  leftMargin=0.9 * inch, rightMargin=0.9 * inch, title="GAIRA V6 — Pure Ag-SERS Evaluation").build(
    story, onFirstPage=on_page, onLaterPages=on_page)
print("PDF ->", OUT)
