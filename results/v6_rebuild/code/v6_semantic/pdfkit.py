"""Shared reportlab styling for the V6 document set."""
from __future__ import annotations
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                TableStyle, PageBreak, KeepTogether, ListFlowable, ListItem)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib as _mpl

_FT = Path(_mpl.__file__).parent / "mpl-data/fonts/ttf"
for nm, fn in [("DJ", "DejaVuSans.ttf"), ("DJ-B", "DejaVuSans-Bold.ttf"),
               ("DJ-O", "DejaVuSans-Oblique.ttf"), ("DJ-M", "DejaVuSansMono.ttf")]:
    try:
        pdfmetrics.registerFont(TTFont(nm, str(_FT / fn)))
    except Exception:
        pass
pdfmetrics.registerFontFamily("DJ", normal="DJ", bold="DJ-B", italic="DJ-O", boldItalic="DJ-B")
FN, FN_B, FN_O, FN_M = "DJ", "DJ-B", "DJ-O", "DJ-M"

INK = colors.HexColor("#1b2430"); MUTED = colors.HexColor("#5b6472")
BLUE = colors.HexColor("#0072B2"); VERM = colors.HexColor("#D55E00")
GREEN = colors.HexColor("#009E73"); ORANGE = colors.HexColor("#E69F00")
LIGHT = colors.HexColor("#eef2f6"); RULE = colors.HexColor("#d9dee4")
FP = "09ed804a40836f4a05a91ba10900cded"

_ss = getSampleStyleSheet()
def _mk(n, **k): return ParagraphStyle(n, parent=_ss["Normal"], **k)

TITLE = _mk("t", fontName=FN_B, fontSize=23, leading=27, textColor=INK, spaceAfter=4)
SUB = _mk("sub", fontName=FN, fontSize=12.5, leading=16.5, textColor=BLUE, spaceAfter=10)
H1 = _mk("h1", fontName=FN_B, fontSize=14.5, leading=18, textColor=INK, spaceBefore=15, spaceAfter=5)
H2 = _mk("h2", fontName=FN_B, fontSize=10.6, leading=13.5, textColor=VERM, spaceBefore=9, spaceAfter=3)
H3 = _mk("h3", fontName=FN_B, fontSize=9.6, leading=12.5, textColor=INK, spaceBefore=7, spaceAfter=2)
BODY = _mk("b", fontName=FN, fontSize=9.5, leading=13.7, textColor=INK, spaceAfter=6)
SMALL = _mk("s", fontName=FN, fontSize=8.4, leading=12.0, textColor=MUTED, spaceAfter=5)
CAP = _mk("c", fontName=FN_O, fontSize=8.1, leading=10.4, textColor=MUTED, spaceAfter=13)
MONO = _mk("m", fontName=FN_M, fontSize=7.8, leading=11.0, textColor=INK, spaceAfter=5)
EQ = _mk("eq", fontName=FN_M, fontSize=9.0, leading=14.5, textColor=INK, leftIndent=16,
         spaceBefore=3, spaceAfter=6)
UW = letter[0] - 1.5 * inch


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items, style=BODY):
    return ListFlowable([ListItem(P(t, style), leftIndent=13, value="•") for t in items],
                        bulletType="bullet", start="•", leftIndent=13,
                        bulletFontName=FN, bulletFontSize=8)


def callout(kind, text, title=None):
    col = {"note": BLUE, "warn": VERM, "good": GREEN, "key": INK, "amber": ORANGE}[kind]
    bg = {"note": colors.HexColor("#f2f7fb"), "warn": colors.HexColor("#fdf3ee"),
          "good": colors.HexColor("#eef8f4"), "key": INK, "amber": colors.HexColor("#fdf7e9")}[kind]
    tc = colors.white if kind == "key" else INK
    body = _mk("cx", fontName=FN, fontSize=9.2, leading=13.2, textColor=tc)
    head = _mk("ch", fontName=FN_B, fontSize=9.2, leading=13.2,
               textColor=colors.white if kind == "key" else col, spaceAfter=2)
    cell = ([Paragraph(title, head)] if title else []) + [Paragraph(text, body)]
    t = Table([[cell]], colWidths=[UW])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, col),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def fig(figdir, name, cap, maxh=7.2 * inch):
    p = Path(figdir) / name
    if not p.exists():
        return P(f"[{name} missing]", CAP)
    iw, ih = PILImage.open(p).size
    ar = ih / iw
    w = UW; h = w * ar
    if h > maxh:
        h = maxh; w = h / ar
    return KeepTogether([Spacer(1, 3), Image(str(p), width=w, height=h), Spacer(1, 3), P(cap, CAP)])


def tbl(rows, headers, cw, fs=8.0):
    cell = _mk("cl", fontName=FN, fontSize=fs, leading=fs + 2.4, textColor=INK)
    head = _mk("hd", fontName=FN_B, fontSize=fs, leading=fs + 2.4, textColor=colors.white)
    data = [[Paragraph(str(h), head) for h in headers]] + \
           [[Paragraph(str(c), cell) for c in r] for r in rows]
    t = Table(data, colWidths=cw, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), INK),
                           ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
                           ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
    return t


def build(story, out_path, title, footer_note):
    def _f(canvas, doc):
        canvas.saveState()
        canvas.setFont(FN, 7.4); canvas.setFillColor(MUTED)
        canvas.drawString(0.75 * inch, 0.46 * inch, footer_note)
        canvas.drawRightString(letter[0] - 0.75 * inch, 0.46 * inch, str(doc.page))
        canvas.setStrokeColor(RULE)
        canvas.line(0.75 * inch, 0.62 * inch, letter[0] - 0.75 * inch, 0.62 * inch)
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out_path), pagesize=letter,
                            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.8 * inch,
                            title=title, author="GAIRA V6")
    doc.build(story, onFirstPage=_f, onLaterPages=_f)
    print("wrote", out_path)
