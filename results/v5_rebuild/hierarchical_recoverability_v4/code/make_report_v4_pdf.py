"""V4 PDF report — null-calibrated hierarchical cross-modal validation. reportlab + DejaVu.
Reads committed V4 tables/figures/summary. Additive; frozen atlas unchanged."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
                                PageBreak, KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib as _mpl

_FT = Path(_mpl.__file__).parent / "mpl-data/fonts/ttf"
pdfmetrics.registerFont(TTFont("DJ", str(_FT / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DJ-B", str(_FT / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DJ-O", str(_FT / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DJ", normal="DJ", bold="DJ-B", italic="DJ-O", boldItalic="DJ-B")
FN, FN_B, FN_O = "DJ", "DJ-B", "DJ-O"

BASE = Path("/Users/surajpg/projects/GAIRA/results/v5_rebuild/hierarchical_recoverability_v4")
OUT = BASE / "GAIRA_Hierarchical_Cross_Modal_Validation_V4.pdf"
FP = "09ed804a40836f4a05a91ba10900cded"
S = json.loads((BASE / "artifacts/recoverability_summary.json").read_text())
CNT = pd.read_csv(BASE / "tables/recoverable_analyte_counts.csv")
LVL = pd.read_csv(BASE / "tables/level_null_summary.csv")
DEC = pd.read_csv(BASE / "tables/metric_decision_table.csv")
BY = pd.read_csv(BASE / "tables/recoverable_analytes_by_level.csv")
MP = pd.read_csv(BASE / "tables/matrix_prediction.csv")

INK = colors.HexColor("#1b2430"); MUTED = colors.HexColor("#5b6472")
BLUE = colors.HexColor("#0072B2"); VERM = colors.HexColor("#D55E00"); GREEN = colors.HexColor("#009E73")
LIGHT = colors.HexColor("#eef2f6")
ss = getSampleStyleSheet()
def _mk(n, **k): return ParagraphStyle(n, parent=ss["Normal"], **k)
TITLE = _mk("t", fontName=FN_B, fontSize=25, leading=29, textColor=INK, spaceAfter=6)
H1 = _mk("h1", fontName=FN_B, fontSize=15, leading=19, textColor=INK, spaceBefore=14, spaceAfter=5)
BODY = _mk("b", fontName=FN, fontSize=9.7, leading=14, textColor=INK, spaceAfter=6)
CAP = _mk("c", fontName=FN_O, fontSize=8.3, leading=10.5, textColor=MUTED, spaceAfter=12)
SM = _mk("sm", fontName=FN, fontSize=7.6, leading=9.6, textColor=INK)
SMB = _mk("smb", fontName=FN_B, fontSize=7.6, leading=9.6, textColor=colors.white)
UW = letter[0] - 1.8 * inch


def P(t, s=BODY): return Paragraph(t, s)
def callout(kind, text):
    col = {"note": BLUE, "warn": VERM, "good": GREEN, "take": INK}[kind]
    bg = {"note": colors.HexColor("#f2f7fb"), "warn": colors.HexColor("#fdf3ee"),
          "good": colors.HexColor("#eef8f4"), "take": INK}[kind]
    tc = colors.white if kind == "take" else INK
    t = Table([[Paragraph(text, _mk("cx", fontName=FN, fontSize=9.3, leading=13, textColor=tc))]], colWidths=[UW])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, col),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t
def fig(name, cap, maxw=UW, maxh=7.2 * inch):
    p = BASE / "figures" / name
    if not p.exists(): return P(f"[{name} missing]", CAP)
    iw, ih = PILImage.open(p).size; ar = ih / iw; w = maxw; h = w * ar
    if h > maxh: h = maxh; w = h / ar
    return KeepTogether([Image(str(p), width=w, height=h), Spacer(1, 3), P(cap, CAP)])
def table(dfr, cols, headers, cw, fs=7.6):
    cell = _mk("cell", fontName=FN, fontSize=fs, leading=fs + 1.6, textColor=INK)
    head = _mk("head", fontName=FN_B, fontSize=fs, leading=fs + 1.6, textColor=colors.white)
    data = [[Paragraph(h, head) for h in headers]]
    for _, r in dfr.iterrows():
        data.append([Paragraph(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]), cell) for c in cols])
    t = Table(data, colWidths=cw, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), INK),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9dee4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4)]))
    return t


story = []
# cover
story += [Spacer(1, 1.6 * inch), P("GAIRA", TITLE),
          P("Hierarchical Cross-Modal Validation (V4)", _mk("st", fontName=FN_B, fontSize=16,
            leading=20, textColor=BLUE, spaceAfter=12)),
          P("Null-calibrated recoverability across Raman, Ag-SERS, controlled perturbation and "
            "biological matrix. Every representation metric is calibrated against an analyte-"
            "mismatched null; recovery is defined statistically, never from a raw cosine threshold.",
            _mk("sub", fontName=FN, fontSize=12, leading=16, textColor=MUTED)), Spacer(1, 0.3 * inch)]
cov = Table([[P(f"<b>Frozen atlas fingerprint</b>  {FP}", SM)],
             [P(f"<b>51</b> matched pure analytes · reproduces V3 bit-for-bit · SERS validates, never trains", SM)],
             [P("Recovered analyte-specifically: latent 7/51 · MSS 3/51 · theme 4/51 · perturbation 3/51 · "
                "matrix 9/51 (serum-tested). Raw cosines (MSS 0.74, theme 0.92) are shared background.", SM)]],
            colWidths=[UW])
cov.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("BOX", (0, 0), (-1, -1), 0.5, BLUE),
    ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("LEFTPADDING", (0, 0), (-1, -1), 12)]))
story += [cov, PageBreak()]

# 1 executive summary
story += [P("1 · Executive summary", H1),
          P("Raman → Ag-SERS transfer cannot be described by one score — and V4's central result is "
            "that it must not be described by one <b>level</b> either. Calibrated against an analyte-"
            "mismatched null, <b>analyte-specific recovery is rare at every level</b>.", BODY),
          callout("warn", "The high raw cosines are almost entirely shared background: MSS matched "
            "0.740 vs null 0.732 (separation 0.008); theme 0.918 vs 0.915 (0.002). A high raw cosine "
            "is broad biochemical interpretation, <b>not</b> analyte identity."),
          callout("take", "MSS is <b>not</b> the primary cross-modal metric — its null separation is "
            "smaller than the latent coordinates', and its 3 recovered analytes are a strict subset of "
            "the 7 recovered at the latent level. The strongest evidence is functional perturbation "
            "(3 analytes). The purine attractor appears in the unspiked-serum blank (purine 0.27) "
            "before any analyte is added.")]
story += [fig("fig01_representation_hierarchy.png",
              "Figure 1. The null-calibrated hierarchy. The per-analyte identity signal (matched − "
              "mismatched null) is tiny at every level and smallest for MSS.")]
story.append(PageBreak())

# counts table + recovered lists
story += [P("2 · How many analytes are recoverable at each level?", H1),
          table(CNT, ["level", "n_recovered", "denominator", "fraction", "ci95_low", "ci95_high"],
                ["Level", "n", "denom", "frac", "CI low", "CI high"],
                [1.7*inch, 0.6*inch, 0.8*inch, 0.7*inch, 0.7*inch, 0.7*inch]),
          Spacer(1, 6),
          fig("fig02_recoverable_by_level.png", "Figure 2. Fractions with 95% CI. Matrix denominator = "
              "serum-tested; all others = 51 matched."),
          P("<b>Recovered analyte names</b> (rank-1 + jackknife-stable):", BODY)]
for _, r in BY.iterrows():
    story.append(P(f"• <b>{r.level}</b> ({r.n}): {r.analytes}", SM))
story.append(PageBreak())

story += [P("3 · Per-analyte recovery matrix", H1),
          fig("fig03_recovery_matrix.png", "Figure 3. ✓ recovered · blank not · dot not-tested. Strong "
              "chemisorbers dominate; most analytes are broad-theme only.", maxh=8.6*inch)]
story.append(PageBreak())

story += [P("4 · Matched vs mismatched — which metric carries identity?", H1),
          fig("fig04_matched_vs_null.png", "Figure 4. Matched (blue) and mismatched-null (grey) "
              "distributions overlap heavily at every level — the metrics carry little analyte identity."),
          fig("fig07_broad_vs_identity.png", "Figure 7. Raw theme cosine clusters near 0.9 for all "
              "analytes; the identity residual separates only the strong adsorbers.")]
story.append(PageBreak())

story += [P("5 · MSS is not primary (tested, not assumed)", H1),
          P("MSS motif cosine (0.740) looks like a robust middle layer, but its mismatched null is "
            "0.732 — the 0.74 is background. Only 3/51 clear the null, all already latent-recovered.", BODY),
          fig("fig06_mss_specificity.png", "Figure 6. MSS ranked by null-adjusted specificity (matched − "
              "null95), not raw cosine. Only 3 clear the null."),
          table(LVL, ["level", "matched_median", "null_median", "separation", "n_rank1", "n_recovered"],
                ["Level", "matched", "null", "separation", "rank-1", "recovered"],
                [2.3*inch, 0.8*inch, 0.7*inch, 0.9*inch, 0.6*inch, 0.9*inch])]
story.append(PageBreak())

story += [P("6 · The purine attractor — background control", H1),
          P(f"The unspiked-serum-on-Ag blank projects to purine share "
            f"<b>{S['purine_controls']['serum_blank_purine_theme']}</b> with "
            f"<b>{S['purine_controls']['serum_blank_dominant_theme']}</b> dominant — the attractor is "
            "present before any analyte. Δpurine anti-correlates with latent (r=−0.38, p=0.006) and MSS "
            "(r=−0.40, p=0.003). No pure Ag-colloid buffer blank exists, so the mechanism is not fully "
            "isolated: we describe the attractor as phenomenological, not as Ag binding alone.", BODY),
          fig("fig08_purine_controls.png", "Figure 8. The attractor and its controls.", maxh=6.3*inch)]
story.append(PageBreak())

story += [P("7 · Perturbation & matrix", H1),
          fig("fig09_perturbation.png", "Figure 9. Functional validation for the only three tested "
              "analytes (adenine, ergothioneine dose; uricase directional).", maxh=3.1*inch),
          P("Matrix: no pure metric significantly predicts serum displacement; only overall confidence "
            "correlates (r=0.71, p=5e-9), likely reflecting signal strength, not identity.", BODY),
          fig("fig10_matrix_prediction.png", "Figure 10. Pure-metric predictors of serum recovery with "
              "95% CI. Only confidence is significant.", maxh=4.2*inch)]
story.append(PageBreak())

story += [P("8 · Representative analytes", H1),
          fig("fig11_representative_panels.png", "Figure 11. Spectrum → latent → themes → identity "
              "residual → evidence for seven representative analytes (blue Raman, red Ag-SERS).", maxh=8.4*inch)]
story.append(PageBreak())

story += [P("9 · Metric-selection decision table", H1),
          P("Decided by the null analyses, not prior preference:", BODY),
          table(DEC, ["purpose", "primary_metric", "null_result"], ["Purpose", "Primary metric", "Why (null result)"],
                [2.1*inch, 1.8*inch, UW-3.9*inch], fs=7.4),
          Spacer(1, 6),
          callout("take", "Verdict: latent cosine is the best cross-modal identity cosine (7/51); MSS is "
            "supporting, not primary; raw theme and Spearman are broad-interpretation descriptors; "
            "functional perturbation is the strongest evidence; matrix is a separate property.")]
story.append(PageBreak())

story += [P("10 · Limitations & conclusions", H1),
          P("<b>Limitations.</b> Raman-trained atlas; no learned modality correction; purine mechanism "
            "not fully isolated (no buffer blank); 3 perturbation cases; recovery depends on null "
            "thresholds and the discrete retrieval p-floor (1/51, so BH-FDR is degenerate at N=51); "
            "confidence ≠ analyte identifiability; incomplete Au-SERS grounding; 5 replicates/analyte.", BODY),
          P("<b>Conclusions.</b> Cross-modal biochemical recovery is hierarchical and rare: only the "
            "strong-chemisorber minority retains analyte-specific latent structure (7/51); motif and "
            "theme identity are weaker (3–4/51); raw cosines are broad-interpretation background; the "
            "purine attractor is a background phenomenon; functional perturbation (3 analytes) is the "
            "strongest evidence. GAIRA reports the whole calibrated hierarchy and never calls an "
            "analyte 'detectable' from a raw cosine.", BODY),
          P("<b>Provenance.</b> Frozen atlas " + FP + " verified unchanged; matched values reproduce V3 "
            "(max abs diff 0.0); deterministic; reruns identical. Full method in "
            "METRICS_AND_DECISION_RULES.md; source audit in AUDIT_OF_V3_METRICS.md.", BODY)]


def on_page(canvas, doc):
    canvas.saveState(); canvas.setFont(FN, 8); canvas.setFillColor(MUTED)
    canvas.drawString(0.9 * inch, 0.55 * inch, "GAIRA · Hierarchical Cross-Modal Validation (V4)")
    canvas.drawRightString(letter[0] - 0.9 * inch, 0.55 * inch, str(doc.page))
    canvas.setStrokeColor(colors.HexColor("#d9dee4")); canvas.line(0.9 * inch, 0.72 * inch, letter[0] - 0.9 * inch, 0.72 * inch)
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=letter, topMargin=0.8 * inch, bottomMargin=0.9 * inch,
                        leftMargin=0.9 * inch, rightMargin=0.9 * inch, title="GAIRA V4 — Hierarchical Cross-Modal Validation")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("PDF ->", OUT, "| pages ~", doc.page)
