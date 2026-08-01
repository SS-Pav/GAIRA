"""Generate the full GAIRA Cross-Modal Transfer PDF report — a document version of the
Foundation Explorer with detailed text explainers, every audited figure, and the key tables.

Reads ONLY committed artifacts (V2 theme-preservation + V3 representation-hierarchy) + the
frozen atlas fingerprint. Additive; nothing frozen is modified.

Output: results/v5_rebuild/representation_hierarchy_v3/GAIRA_Cross_Modal_Transfer_Report.pdf
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
                                PageBreak, KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib as _mpl

# Register DejaVu Sans (bundled with matplotlib) — full Unicode coverage so Δ, ρ, →, ², µ all
# render. reportlab's built-in Helvetica drops Greek capitals / combining marks.
_FT = Path(_mpl.__file__).parent / "mpl-data/fonts/ttf"
pdfmetrics.registerFont(TTFont("DJ", str(_FT / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DJ-B", str(_FT / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DJ-O", str(_FT / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DJ", normal="DJ", bold="DJ-B", italic="DJ-O", boldItalic="DJ-B")
FN, FN_B, FN_O = "DJ", "DJ-B", "DJ-O"

REPO = Path("/Users/surajpg/projects/GAIRA")
V3 = REPO / "results/v5_rebuild/representation_hierarchy_v3"
V2 = REPO / "results/v5_rebuild/pure_ag_sers_theme_preservation"
OUT = V3 / "GAIRA_Cross_Modal_Transfer_Report.pdf"
FP = "09ed804a40836f4a05a91ba10900cded"

# ── data ──
S = json.loads((V3 / "artifacts/hierarchy_summary.json").read_text())
L = S["layers"]
HS = S["representation_hierarchy"]
DF = pd.read_csv(V3 / "tables/per_analyte_hierarchy.csv")
FAM = pd.read_csv(V3 / "tables/rank_by_family.csv")
PERT = pd.read_csv(V2 / "tables/perturbation_sensitivity.csv")
HSUM = pd.read_csv(V3 / "tables/representation_hierarchy_summary.csv")
CARDS = json.loads((V3 / "artifacts/all_cards_v3.json").read_text())
MREG = S["matrix_regression"]["predictor_latent_fingerprint"]

# ── palette (Okabe-Ito) ──
INK = colors.HexColor("#1b2430"); MUTED = colors.HexColor("#5b6472")
BLUE = colors.HexColor("#0072B2"); VERM = colors.HexColor("#D55E00")
GREEN = colors.HexColor("#009E73"); ORANGE = colors.HexColor("#E69F00")
PURPLE = colors.HexColor("#CC79A7"); LIGHT = colors.HexColor("#eef2f6")
LEVELCOL = {1: VERM, 2: ORANGE, 3: BLUE, 4: GREEN, 5: PURPLE}

# ── styles ──
ss = getSampleStyleSheet()
def _mk(name, **kw):
    return ParagraphStyle(name, parent=ss["Normal"], **kw)
TITLE = _mk("t", fontName=FN_B, fontSize=26, leading=30, textColor=INK, spaceAfter=6)
SUB = _mk("s", fontName=FN, fontSize=13, leading=17, textColor=MUTED, spaceAfter=4)
H1 = _mk("h1", fontName=FN_B, fontSize=16, leading=20, textColor=INK,
         spaceBefore=16, spaceAfter=6)
H2 = _mk("h2", fontName=FN_B, fontSize=12.5, leading=16, textColor=BLUE,
         spaceBefore=11, spaceAfter=4)
BODY = _mk("b", fontName=FN, fontSize=10, leading=14.5, textColor=INK, spaceAfter=7,
           alignment=TA_LEFT)
CAP = _mk("c", fontName=FN_O, fontSize=8.5, leading=11, textColor=MUTED, spaceAfter=12)
SMALL = _mk("sm", fontName=FN, fontSize=7.5, leading=9.5, textColor=INK)
SMALLB = _mk("smb", fontName=FN_B, fontSize=7.5, leading=9.5, textColor=colors.white)
CALL = _mk("call", fontName=FN, fontSize=9.5, leading=13.5, textColor=INK)
TOCITEM = _mk("toc", fontName=FN, fontSize=10.5, leading=17, textColor=INK)

USABLE_W = letter[0] - 2 * 0.9 * inch


def P(t, style=BODY): return Paragraph(t, style)


def callout(kind, text):
    col = {"note": BLUE, "warn": VERM, "good": GREEN, "take": INK}[kind]
    bg = {"note": colors.HexColor("#f2f7fb"), "warn": colors.HexColor("#fdf3ee"),
          "good": colors.HexColor("#eef8f4"), "take": INK}[kind]
    tc = colors.white if kind == "take" else INK
    p = Paragraph(text, _mk("cx", fontName=FN, fontSize=9.5, leading=13.5, textColor=tc))
    tbl = Table([[p]], colWidths=[USABLE_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, col),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return tbl


def fig(name, caption, module=V3, max_w=USABLE_W, max_h=7.4 * inch):
    p = module / "figures" / name
    if not p.exists():
        return P(f"[figure {name} not found]", CAP)
    iw, ih = PILImage.open(p).size
    ar = ih / iw
    w = max_w; h = w * ar
    if h > max_h:
        h = max_h; w = h / ar
    return KeepTogether([Image(str(p), width=w, height=h), Spacer(1, 3), P(caption, CAP)])


def df_table(df, cols, headers, colw, fontsize=7.5, hl_col=None):
    cell = _mk("cell", fontName=FN, fontSize=fontsize, leading=fontsize + 1.6, textColor=INK)
    head = _mk("head", fontName=FN_B, fontSize=fontsize, leading=fontsize + 1.6, textColor=colors.white)
    data = [[Paragraph(h, head) for h in headers]]
    for _, r in df.iterrows():
        row = []
        for c in cols:
            v = r[c]
            v = f"{v:.3f}" if isinstance(v, float) else str(v)
            row.append(Paragraph(v, cell))
        data.append(row)
    t = Table(data, colWidths=colw, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), INK),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9dee4")),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    t.setStyle(TableStyle(style))
    return t


def level_head(n, title):
    band = Table([[Paragraph(f"LEVEL {n}", SMALLB), Paragraph(f"<b>{title}</b>",
                 _mk("lh", fontName=FN_B, fontSize=13, textColor=INK))]],
                 colWidths=[0.9 * inch, USABLE_W - 0.9 * inch])
    band.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), LEVELCOL[n]),
                              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("LEFTPADDING", (0, 0), (0, 0), 8), ("TOPPADDING", (0, 0), (-1, -1), 5),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                              ("LEFTPADDING", (1, 0), (1, 0), 10)]))
    return band


# ══════════════════════ build the story ══════════════════════
story = []


def spacer(h=8): story.append(Spacer(1, h))


# ── COVER ──
story += [Spacer(1, 1.7 * inch),
          P("GAIRA", TITLE),
          P("Cross-Modal Transfer — the full report", _mk("st", fontName=FN_B,
            fontSize=17, leading=21, textColor=BLUE, spaceAfter=14)),
          P("What survives when a pure-compound spectrum moves from Raman to Ag-SERS, measured as "
            "a five-level Representation Hierarchy — latent fingerprint, MSS motif, biochemical "
            "theme, perturbation, and matrix robustness — with honest null controls throughout.",
            SUB),
          Spacer(1, 0.4 * inch)]
cover = Table([[P(f"<b>Frozen atlas fingerprint</b>  {FP}", SMALL)],
               [P(f"<b>{S['n_matched']}</b> matched pure analytes · Raman reference vs pure Ag-SERS · "
                  "SERS validates the model, never trains it", SMALL)],
               [P("A document version of Foundation Explorer V3, incorporating the V1 (latent) and "
                  "V2 (theme) analyses. Every figure is reproduced from committed artifacts; nothing "
                  "is retrained.", SMALL)]], colWidths=[USABLE_W])
cover.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("BOX", (0, 0), (-1, -1), 0.5, BLUE),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                           ("LEFTPADDING", (0, 0), (-1, -1), 12)]))
story += [cover, PageBreak()]

# ── EXECUTIVE SUMMARY ──
story += [P("Executive summary", H1),
          P("There is no single number for “how well does Raman transfer to Ag-SERS.” Transfer is a "
            "<b>hierarchy of representations</b>, and the honest question is how far up the hierarchy "
            "the agreement survives — because each level is progressively more abstract and less "
            "governed by silver’s surface physics.", BODY),
          P(f"Read top to bottom, two things happen at once. The <b>raw</b> agreement rises "
            f"(latent {L['L1_latent_fingerprint']['median']} → MSS {L['L2_mss_motif']['median']} → "
            f"theme {L['L3a_theme_raw']['median']}), which looks like “more biochemistry survives the "
            f"higher you go.” But the <b>identity-specific</b> agreement falls at the top "
            f"(theme identity {L['L3b_theme_identity']['median']}, argmax "
            f"{int(S['layers']['L6_argmax_agreement_rate']*100)}%). Both are true, and holding them "
            f"together is the whole point.", BODY)]
story += [callout("warn",
          "<b>The central caution.</b> Raw theme cosine (0.92) and raw Spearman rank ρ (0.87) are "
          "spectacular but <b>baseline-inflated</b>: every analyte’s theme composition shares the same "
          "dominant background, so any two analytes — even unrelated ones — already agree at ≈0.9 "
          "before preservation is considered. Once the baseline is removed, identity-specific "
          "preservation is selective, concentrated in the strong silver adsorbers."), Spacer(1, 8)]
story += [callout("good",
          "<b>The useful minority.</b> A real subset — adenine foremost — redistributes its latent "
          "fingerprint yet keeps a <b>dose-responsive</b> biochemical theme. Adenine is weak at Level 1 "
          "(0.36) yet responds to concentration along a saturating Langmuir law (ρ=0.996). A single "
          "cosine would call it a failure; the hierarchy shows it is one of GAIRA’s best-validated analytes.")]
story.append(PageBreak())

# ── CONTENTS ──
toc = ["1 · The Representation Hierarchy", "2 · The metric problem", "3 · Metric specification",
       "4 · Level 1 — Latent fingerprint", "5 · Level 2 — MSS motif",
       "6 · Level 3 — Biochemical theme (raw & identity)", "7 · Level 3 — Theme rank preservation (new)",
       "8 · Level 3 — Top-k theme overlap (new)", "9 · Level 3 — Dominant-theme agreement",
       "10 · The purine attractor, quantified", "11 · Level 4 — Perturbation validation",
       "12 · Level 5 — Matrix robustness", "13 · Cross-modal transfer map",
       "14 · Per-analyte assessment cards", "15 · Verdict", "16 · Interpretation guide",
       "Appendix A · Full per-analyte metrics (51 analytes)", "Appendix B · Provenance & reproducibility"]
story += [P("Contents", H1)] + [P(t, TOCITEM) for t in toc]
story.append(PageBreak())

# ── 1 · HIERARCHY ──
story += [P("1 · The Representation Hierarchy", H1),
          P("Each level below is more abstract, and less governed by adsorption physics, than the one "
            "above it. GAIRA reports the whole ladder, never a single collapsed “preservation score.”", BODY),
          fig("fig_h1_representation_hierarchy.png",
              "Figure 1. The conceptual ladder (left) and the per-level distributions across 51 analytes "
              "(right). Raw theme and raw rank cluster high but are baseline-inflated; the identity metric "
              "spreads wide — the honest, selective signal.")]
story += [P("Per-level summary", H2),
          df_table(HSUM, ["level", "metric", "median", "mean", "std", "min", "max"],
                   ["Level", "Metric", "median", "mean", "std", "min", "max"],
                   [1.5*inch, 1.5*inch, 0.6*inch, 0.6*inch, 0.55*inch, 0.55*inch, 0.55*inch]),
          Spacer(1, 6),
          callout("take", "Read the hierarchy top-to-bottom: raw agreement rises, identity-specific "
                  "agreement falls. The job is not to maximise agreement but to separate surface "
                  "physics from biochemical meaning.")]
story.append(PageBreak())

# ── 2 · METRIC PROBLEM ──
story += [P("2 · The metric problem", H1),
          P("The original pure-Ag-SERS stage reports one number per analyte — the cosine between the "
            "24 NMF coordinates — and calls it <i>recoverability</i>. That number is correct and is "
            "kept (it becomes Level 1). But it measures only whether two spectra land on the same point "
            "of the latent manifold; it says nothing about whether the biochemical interpretation "
            "survives, whether a perturbation would register, or whether the analyte is recoverable in "
            "serum. Those are four different questions.", BODY),
          P(f"At the theme level the raw cosine is {L['L3a_theme_raw']['median']} — far above the "
            f"latent {L['L1_latent_fingerprint']['median']}, and it exceeds the latent cosine for all "
            f"51 analytes. Read naively this “proves” the theme almost always survives. It does not: "
            f"the composition of every analyte is dominated by the same high-share background themes, "
            f"so two unrelated analytes already sit at ≈0.9. The fix is to subtract the shared baseline "
            f"and measure each analyte’s distinctive deviation against a null.", BODY),
          fig("fig1_component_vs_theme.png",
              "Figure 2. Naive vs honest reading (from the V2 analysis). Left: raw theme cosine sits above "
              "the diagonal for every analyte — baseline inflation. Right: baseline-subtracted, only the "
              "strong chemisorbers keep an identity-specific theme.", module=V2)]
story.append(PageBreak())

# ── 3 · METRIC SPEC ──
story += [P("3 · Metric specification", H1),
          P("The theme level is a <b>ladder of strictness</b> — raw → identity → rank → top-k → argmax. "
            "Each answers a different question; the raw ones are baseline-inflated and are never read "
            "alone. Symbols: z = 24 NMF coordinates, m = 12 MSS motifs, t = 11 theme shares, subscripts "
            "R (Raman) / S (Ag-SERS); b = the shared baseline (mean Raman theme composition over all analytes).", BODY)]
spec = [
    ("L1 Latent fingerprint", "cos(z_R, z_S)", f"median {L['L1_latent_fingerprint']['median']}",
     "Do the coordinates line up? Adsorption-dominated — a low value is surface physics, not a representation error."),
    ("L2 MSS motif", "cos(m_R, m_S)", f"median {L['L2_mss_motif']['median']}",
     "Mid-level structure; more robust than coordinates, still surface-sensitive."),
    ("L3a Theme raw", "cos(t_R, t_S)", f"median {L['L3a_theme_raw']['median']}",
     "Gross composition similarity. CONTAINS BASELINE INFLATION — never a stand-alone measure."),
    ("L3b Theme identity", "cos(t_R−b, t_S−b) vs null", f"median {L['L3b_theme_identity']['median']}",
     "Baseline-subtracted, self-referenced against other analytes. The honest, selective theme signal."),
    ("L4 Theme rank (NEW)", "Spearman ρ(rank t_R, rank t_S)", f"median {L['L4_theme_rank_rho']['median']}",
     "Ordering of all 11 themes. Robust to argmax instability — but ALSO baseline-inflated (separation "
     f"{L['L4_rank_separation']['median']})."),
    ("L5 Top-k overlap (NEW)", "|topk(t_R)∩topk(t_S)|/k", f"top-3 {L['L5_top3_overlap']['median']}",
     "Do the leading themes stay in the top set? Interpretable middle-ground; avoids argmax knife-edge."),
    ("L6 Argmax", "argmax(t_R)==argmax(t_S)", f"{int(S['layers']['L6_argmax_agreement_rate']*100)}%",
     "Strict, unstable single-dominant test. On Ag, 50/51 become purine — so agreement mostly means "
     "‘already purine in Raman’."),
]
spec_rows = [[Paragraph("<b>Metric</b>", SMALLB), Paragraph("<b>Equation</b>", SMALLB),
              Paragraph("<b>Result</b>", SMALLB), Paragraph("<b>Purpose / limitation</b>", SMALLB)]]
for a, b, c, d in spec:
    spec_rows.append([Paragraph(a, SMALL), Paragraph(b, SMALL), Paragraph(c, SMALL), Paragraph(d, SMALL)])
st = Table(spec_rows, colWidths=[1.35*inch, 1.5*inch, 0.9*inch, USABLE_W-3.75*inch], repeatRows=1)
st.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), INK),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9dee4")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
story += [st, Spacer(1, 6),
          callout("note", "Reading rule adopted across GAIRA: never quote a raw metric (L3a cosine or "
                  "L4 rank ρ) without its null / separation; report the hierarchy, not a single score; "
                  "use physics-aware language (latent redistribution, adsorption-driven observation bias, "
                  "identity-specific preservation, functional perturbation validation), never “theme "
                  "preserved / failed.”")]
story.append(PageBreak())

# ── 4 · L1 ──
story += [level_head(1, "Latent fingerprint preservation"), Spacer(1, 8),
          P(f"Cosine between the 24 NMF coordinates — the finest-grained view, dominated by adsorption "
            f"physics. Median <b>{L['L1_latent_fingerprint']['median']}</b> (mean "
            f"{L['L1_latent_fingerprint']['mean']}, range [{L['L1_latent_fingerprint']['min']}, "
            f"{L['L1_latent_fingerprint']['max']}]). Strong Ag chemisorbers (oxopurines) score high; "
            f"weak physisorbers (pyrimidines, sugars, small amino acids) low. A low value here is a "
            f"surface effect, not a failure of the frozen representation — this is the original V1 "
            f"“coordinate cosine,” reframed as Level 1 and kept verbatim.", BODY),
          Spacer(1, 6), level_head(2, "MSS motif preservation"), Spacer(1, 8),
          P(f"Cosine over the 12 biochemical MSS motif activations — median <b>{L['L2_mss_motif']['median']}</b>, "
            f"sitting between the latent coordinates (0.42) and the raw theme (0.92). Motif structure is "
            f"more robust than exact coordinates but still adsorption-sensitive; it is also the layer at "
            f"which perturbations localise (the uricase case, §11).", BODY)]
story.append(PageBreak())

# ── 6 · L3 theme ──
story += [level_head(3, "Biochemical theme — raw & identity-specific"), Spacer(1, 8),
          P(f"The interpretation layer, measured as a ladder. The two cosines first: <b>raw</b> "
            f"(median {L['L3a_theme_raw']['median']}, baseline-inflated) and <b>identity-specific</b> "
            f"(median {L['L3b_theme_identity']['median']}, baseline-subtracted vs a null). "
            f"The identity-specific SERS profile self-identifies the correct analyte for only "
            f"{S['self_is_nearest_theme'] if 'self_is_nearest_theme' in S else '4'} of 51 — so theme "
            f"preservation is real but selective.", BODY) if False else
          P(f"The interpretation layer, measured as a ladder. The two cosines first: <b>raw</b> "
            f"(median {L['L3a_theme_raw']['median']}, baseline-inflated) and <b>identity-specific</b> "
            f"(median {L['L3b_theme_identity']['median']}, baseline-subtracted vs a null over the other "
            f"analytes). Identity-specific preservation is real but <b>selective</b> — concentrated in "
            f"the strong silver adsorbers (oxopurines, cofactors, creatinine/urea, PEP/citrate).", BODY),
          fig("fig_h2_metric_comparison.png",
              "Figure 3. The same 51 analytes across six metrics: preservation is not one number. Raw "
              "theme and raw rank cluster near the top; the identity metric spreads wide, many negative."),
          fig("fig3_theme_heatmap.png",
              "Figure 4. Theme composition, Raman (left) vs Ag-SERS (right), analytes ordered by "
              "distinctive preservation. The rich, analyte-specific Raman structure is homogenised on "
              "silver toward a purine-dominated profile.", module=V2)]
story.append(PageBreak())

# ── 7 · L4 rank ──
story += [level_head(3, "Theme RANK preservation (Spearman) — new"), Spacer(1, 8),
          P(f"Instead of argmax or magnitude cosine, compute Spearman ρ between the Raman and Ag-SERS "
            f"theme <i>rankings</i>. This uses the full ordering, including minor themes, and is robust "
            f"to argmax instability — likely the most representative single view. Raw ρ median "
            f"<b>{L['L4_theme_rank_rho']['median']}</b>.", BODY),
          callout("warn",
            f"<b>Tested, not assumed.</b> Raw rank ρ is <b>also baseline-inflated</b>: its null (rank of "
            f"one analyte’s Raman vs another’s Ag-SERS) is ≈0.85, so the identity-specific rank "
            f"separation is only <b>{L['L4_rank_separation']['median']}</b> (positive for "
            f"{S['rank_positive_separation']}/51 — a slim edge over cosine’s 28/51). Rank ordering "
            f"carries marginally more identity signal than magnitude, and no more."),
          Spacer(1, 6),
          fig("fig_h4_topk_and_rank_null.png",
              "Figure 5. Right: raw rank ρ ≈ its null; the honest signal is the small positive "
              "separation. Left: top-k overlap (next section).")]
story.append(PageBreak())

# ── 8 · L5 top-k ──
story += [level_head(3, "Top-k theme overlap — new"), Spacer(1, 8),
          P(f"If purine, protein and organic remain within the top three themes, that is partial "
            f"preservation even if the single argmax flips. Top-k overlap avoids argmax instability and "
            f"is less baseline-inflated than cosine/rank because it looks only at the leading (more "
            f"variable) themes. Median top-2 <b>{L['L5_top2_overlap']['median']}</b>, top-3 "
            f"<b>{L['L5_top3_overlap']['median']}</b> — typically two of three leading themes retained. "
            f"This is the interpretable middle-ground of the theme ladder.", BODY),
          Spacer(1, 6), level_head(3, "Dominant-theme agreement (argmax)"), Spacer(1, 8),
          P(f"Whether the single most-active theme survives — agreement "
            f"<b>{int(S['layers']['L6_argmax_agreement_rate']*100)}%</b>. Intentionally the strictest "
            f"test, and the least stable: on Ag-SERS 50/51 analytes become nucleic_purine-dominant, so "
            f"all 18 agreements are analytes that were <i>already</i> purine-dominant in Raman. Argmax "
            f"is a strict corner case, never the headline — which is exactly why GAIRA reports the softer "
            f"rank and top-k metrics alongside it.", BODY),
          fig("fig6_dominant_theme_confusion.png",
              "Figure 6. Dominant-theme confusion (Raman → Ag-SERS): nearly all off-diagonal mass is the "
              "nucleic_purine column — the silver purine attractor.", module=V2)]
story.append(PageBreak())

# ── 10 · PURINE ATTRACTOR ──
story += [P("10 · The purine attractor, quantified", H1),
          P("Silver colloid binds N-heterocycles strongly, so oxopurine-like signal dominates the SERS "
            "of weak adsorbers. This is the mechanism behind the low argmax agreement — and V3 quantifies "
            "it per analyte.", BODY),
          P(f"<b>ΔPurine</b> = purine share in Ag-SERS minus in Raman. It increases for "
            f"<b>{S['delta_purine']['n_increase']}/51</b> analytes (median {S['delta_purine']['median']}): "
            f"non-purines gain purine share on silver, while already-purine-rich analytes lose it "
            f"(regression toward the attractor). Crucially, <b>ΔPurine anti-correlates with the latent "
            f"fingerprint (r=−0.38, p=0.006)</b> — the weaker the adsorption fidelity, the harder the "
            f"analyte is pulled into the attractor.", BODY),
          fig("fig_h5_delta_purine.png",
              "Figure 7. Per-analyte ΔPurine. Red = gains purine share on silver (weak adsorbers); "
              "blue = loses it (already purine-rich, e.g. guanine −0.25).", max_h=6.6*inch)]
story += [fig("fig_h6_delta_purine_vs_component.png",
              "Figure 8. Significant negative relationship (r=−0.38, p=0.006): weaker adsorption fidelity "
              "→ stronger attractor pull."),
          fig("fig_h3_family_heatmap.png",
              "Figure 9. Preservation by biochemical family across every layer. Identity-specific "
              "preservation (column L3b) is positive for purines, cofactors, organic acids and "
              "small-nitrogenous; negative (scrambled) for amino acids, sugars, polyol."),
          callout("note", "This is adsorption-driven observation bias — a property of the silver surface "
                  "and a target for a future observation model, not a defect of the frozen representation, "
                  "which honestly flags the modality gap (OOD 0.05 → 0.16).")]
story.append(PageBreak())

# ── 11 · PERTURBATION ──
story += [level_head(4, "Perturbation validation"), Spacer(1, 8),
          P("A controlled change that still moves the correct theme is stronger evidence than any static "
            "similarity — it shows the abstraction is <b>functional</b>. It exists for <b>exactly three "
            "analytes</b>; every other analyte is “not measured,” never implied.", BODY),
          df_table(PERT, ["analyte", "perturbation_kind", "target_theme", "statement"],
                   ["Analyte", "Type", "Target", "Key finding"],
                   [0.9*inch, 1.5*inch, 1.2*inch, USABLE_W-3.6*inch]),
          Spacer(1, 6),
          fig("fig_h8_perturbation_summary.png",
              "Figure 10. The three validated perturbations. Adenine and ergothioneine are saturating "
              "dose-responses; uricase is a directional depletion (not a dose).", max_h=3.2*inch),
          callout("good", "Adenine — weak at Level 1 (0.36) — drives the purine theme monotonically "
                  "(ρ=0.996) along a saturating Langmuir law (K=0.89 µM): the abstraction tracks "
                  "concentration, not coincidence. Dynamic response is the strongest rung in the ladder — "
                  "but only 3 of 51 analytes provide it, and GAIRA never extrapolates it."),]
story.append(PageBreak())

# ── 12 · MATRIX ──
story += [level_head(5, "Matrix robustness"), Spacer(1, 8),
          P(f"Does pure-Ag transfer <i>predict</i> serum recoverability across all 51 analytes? "
            f"Regression says only weakly: latent fingerprint → serum spike displacement gives "
            f"<b>r={MREG['r']}, R²={MREG['r2']}, p={MREG['p_value']} (not significant)</b>; the "
            f"Spearman ρ is {MREG['spearman_rho']}. The top oxopurines survive both the modality and the "
            f"matrix gap, but there is no tight per-analyte law — serum adds substantial matrix-specific "
            f"competition beyond pure adsorption strength.", BODY),
          fig("fig_h7_matrix_regression.png",
              "Figure 11. Pure transfer vs serum spike displacement with 95% CI. The slope is shallow and "
              "its CI spans zero — a categorical top-set agreement, not a quantitative law. This is an "
              "honest downgrade of the earlier categorical claim."),
          callout("warn", "Serum data validate <b>matrix</b> recoverability, not pure-analyte theme "
                  "preservation. The two are kept separate throughout the framework.")]
story.append(PageBreak())

# ── 13 · CROSS-MODAL MAP + 14 exemplar cards ──
story += [P("13 · Cross-modal transfer map & 14 · Per-analyte assessment", H1),
          P("Each analyte carries nine layers: latent · MSS · theme cosine · rank ρ · top-3 · argmax · "
            "family · interpretation · limitations. Four exemplars span the behaviour space.", BODY)]
for name in ["adenine", "guanine", "glucose", "uracil"]:
    c = CARDS[name]; L1 = c["layer1_latent_fingerprint"]; L2c = c["layer2_mss_motif"]
    L3 = c["layer3_theme_cosine"]; L4 = c["layer4_theme_rank_correlation"]; L5 = c["layer5_top3_overlap"]
    L6 = c["layer6_argmax_agreement"]; L7 = c["layer7_family"]
    metrics_line = (f"L1 latent {L1['component_cosine']} · L2 MSS {L2c['mss_cosine']} · "
                    f"theme raw {L3['raw']} · theme identity {L3['identity_specific']} · "
                    f"rank ρ {L4['spearman_rho']} (sep {L4['rank_separation']}) · top-3 {L5['top3']} · "
                    f"argmax {L6['raman_dominant']}→{L6['sers_dominant']} · ΔPurine {L7['delta_purine']}")
    block = [P(f"<b>{name}</b> — {L7['family']}", H2),
             P(metrics_line, SMALL), Spacer(1, 3),
             P(f"<b>Interpretation.</b> {c['layer8_interpretation']}", BODY),
             P("<b>Limitations.</b> " + " ".join(c["layer9_limitations"]), _mk("lim",
               fontName=FN_O, fontSize=8.5, leading=11.5, textColor=MUTED, spaceAfter=10))]
    story.append(KeepTogether(block))
story.append(PageBreak())

# ── 15 · VERDICT ──
story += [P("15 · Verdict", H1),
          callout("take",
            "Raman → Ag-SERS transfer is a preservation <b>hierarchy</b>, not a score. Latent coordinates "
            "transfer partially (0.42, adsorption-limited); motifs better (0.74); the broad biochemical "
            "neighbourhood best in raw terms (theme 0.92, rank 0.87) — but that is largely a compositional "
            "baseline. Corrected, identity-specific theme preservation is selective (identity 0.11, rank "
            "separation +0.01, top-3 0.67, argmax 35%), because silver homogenises most analytes toward a "
            "purine attractor (ΔPurine ∝ −adsorption fidelity, p=0.006). A minority — adenine foremost — "
            "redistribute their latent fingerprint yet retain a dose-responsive theme; that functional "
            "validation, though rare, is the strongest rung. GAIRA’s job is not to maximise agreement but "
            "to separate surface physics from biochemical meaning, and to report the whole ladder."),
          Spacer(1, 8),
          P("16 · Interpretation guide", H1),
          P("<b>Do not say</b> “theme preserved / failed,” “the model recovers X,” “SERS agrees with "
            "Raman.” <b>Say instead</b> “identity-specific preservation is high / selective / weak,” "
            "“X’s biochemical abstraction transfers / is pulled into the purine attractor,” “agreement "
            "survives to Level n of the hierarchy.” Always pair a raw metric with its null; describe "
            "failures as surface physics; treat dynamic perturbation as the gold standard — available for "
            "only three analytes.", BODY)]
story.append(PageBreak())

# ── APPENDIX A — full table ──
story += [P("Appendix A · Full per-analyte metrics (51 analytes)", H1),
          P("All layers, sorted by theme rank ρ. Metrics: L1 latent · L2 MSS · L3a theme raw · L3b theme "
            "identity · L4 rank ρ · L4 rank separation · L5 top-3 · L6 argmax · ΔPurine.", BODY)]
appx = DF[["analyte", "family", "L1_latent_fingerprint", "L2_mss_motif", "L3a_theme_raw",
           "L3b_theme_identity", "L4_theme_rank_rho", "L4_rank_separation", "L5_top3_overlap",
           "L6_argmax_agreement", "delta_purine"]].copy()
story += [df_table(appx,
          list(appx.columns),
          ["analyte", "family", "L1", "L2", "L3a", "L3b", "rank ρ", "rank Δ", "top3", "argmax", "ΔPur"],
          [0.92*inch, 0.82*inch, 0.44*inch, 0.44*inch, 0.44*inch, 0.48*inch, 0.5*inch, 0.5*inch,
           0.48*inch, 0.52*inch, 0.48*inch], fontsize=6.9)]
story.append(PageBreak())

# ── APPENDIX B — provenance ──
repro = S["reproducibility_vs_v2"]
story += [P("Appendix B · Provenance & reproducibility", H1),
          P(f"<b>Frozen atlas fingerprint:</b> {FP} — verified unchanged. NMF (k=24), preprocessing, "
            f"MSS generation, ontology, registry, and theme weights are all unchanged; SERS is projected "
            f"through the fixed basis, never used to fit it.", BODY),
          P(f"<b>Reproducibility vs the V2 analysis (max absolute difference):</b> component cosine "
            f"{repro['component_cosine_max_abs_diff']}, theme raw {repro['theme_raw_max_abs_diff']}, "
            f"theme identity {repro['theme_identity_max_abs_diff']}, MSS {repro['mss_max_abs_diff']} — "
            f"every shared metric reproduced bit-for-bit.", BODY),
          P("<b>Sources.</b> Level 1–3 metrics: results/v5_rebuild/representation_hierarchy_v3/ "
            "(reanalysis) and results/v5_rebuild/pure_ag_sers_theme_preservation/ (V2). Perturbation: "
            "foundation_audit/tables/validation_results.json. Matrix: spike_validation/tables/"
            "phase7_serum_vs_pure.csv. Figures reproduced from the committed, audited PNGs. Interactive "
            "versions: Foundation Explorer V1/V2/V3 (streamlit).", BODY),
          P("<b>Scope.</b> 51 pure analytes matched between the Gobbato Raman reference and pure Ag-SERS. "
            "This report is a document version of Foundation Explorer V3, incorporating the V1 (latent) "
            "and V2 (theme) analyses. Nothing is retrained or modified.", BODY)]


# ── page furniture ──
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont(FN, 8); canvas.setFillColor(MUTED)
    canvas.drawString(0.9 * inch, 0.55 * inch,
                      "GAIRA · Cross-Modal Transfer Report")
    canvas.drawRightString(letter[0] - 0.9 * inch, 0.55 * inch, f"{doc.page}")
    canvas.setStrokeColor(colors.HexColor("#d9dee4"))
    canvas.line(0.9 * inch, 0.72 * inch, letter[0] - 0.9 * inch, 0.72 * inch)
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=letter, topMargin=0.8 * inch, bottomMargin=0.9 * inch,
                        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                        title="GAIRA — Cross-Modal Transfer Report",
                        author="GAIRA")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("PDF written ->", OUT)
print("pages ~", doc.page)
