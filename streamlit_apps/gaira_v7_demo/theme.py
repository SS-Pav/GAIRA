"""GAIRA V7 Phase 11 — visual system.

Presentation only. Nothing in this module touches a spectrum, a number or the engine; it defines
colours, typography, CSS and the shared Plotly layout so every figure in the demo reads as one
system rather than as a pile of charts.
"""
from __future__ import annotations

# ── palette ──────────────────────────────────────────────────────────────────
BG          = "#0b0e14"      # page
SURFACE     = "#121722"      # card
SURFACE_2   = "#171d2b"      # raised card
STROKE      = "#232b3d"
INK         = "#e8ecf4"      # primary text
INK_2       = "#9aa5bb"      # secondary text
INK_3       = "#63708c"      # tertiary / axis

ACCENT      = "#5b8cff"      # primary — the engine
CYAN        = "#22d3ee"      # representation (LSM / CSM)
VIOLET      = "#a78bfa"      # chemistry
AMBER       = "#fbbf24"      # caution / bands
GREEN       = "#34d399"      # confirmation
ROSE        = "#fb7185"      # residual / warning

# Sixteen chemistry axes, in the frozen CLASS_ORDER. Perceptually spaced, dark-legible.
AXIS_COLORS = [
    "#5b8cff", "#22d3ee", "#34d399", "#a3e635", "#fbbf24", "#fb923c",
    "#fb7185", "#f472b6", "#a78bfa", "#818cf8", "#2dd4bf", "#4ade80",
    "#facc15", "#f87171", "#c084fc", "#60a5fa",
]

FONT = ('-apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, '
        'Helvetica, Arial, sans-serif')
MONO = 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace'


def plotly_layout(**kw) -> dict:
    """The shared Plotly layout. Every figure in the demo starts from this."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=INK_2),
        margin=dict(l=48, r=24, t=62, b=44),
        hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=STROKE,
                        font=dict(family=FONT, size=12, color=INK)),
        xaxis=dict(gridcolor=STROKE, zerolinecolor=STROKE, linecolor=STROKE,
                   tickfont=dict(color=INK_3), title_font=dict(color=INK_3, size=11)),
        yaxis=dict(gridcolor=STROKE, zerolinecolor=STROKE, linecolor=STROKE,
                   tickfont=dict(color=INK_3), title_font=dict(color=INK_3, size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2, size=11),
                    orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        # text="" matters: a title dict without it renders the string "undefined".
        title=dict(text="", x=0, xanchor="left", y=0.97, yanchor="top"),
        transition=dict(duration=420, easing="cubic-in-out"),
    )
    base.update(kw)
    return base


CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  .stApp {{ background:
      radial-gradient(1200px 700px at 12% -8%, #16203a 0%, transparent 55%),
      radial-gradient(900px 600px at 92% 4%, #1b1630 0%, transparent 50%),
      {BG}; color: {INK}; }}

  header[data-testid="stHeader"] {{ background: transparent; }}
  [data-testid="stToolbar"], [data-testid="stDecoration"], footer {{ display: none; }}
  section[data-testid="stSidebar"] {{ display: none; }}

  .block-container {{ padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1340px; }}

  html, body, [class*="css"], .stMarkdown, p, span, div, label {{
      font-family: {FONT}; -webkit-font-smoothing: antialiased; }}

  h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.021em; font-weight: 600; }}
  h1 {{ font-size: 2.05rem; }} h2 {{ font-size: 1.32rem; margin-top: .4rem; }}
  h3 {{ font-size: 1.02rem; color: {INK}; }}
  p, li {{ color: {INK_2}; line-height: 1.62; }}
  code, .mono {{ font-family: {MONO}; font-size: .80rem; color: {CYAN}; }}

  /* ── glass surfaces ─────────────────────────────────────────────────── */
  .glass {{
      background: linear-gradient(160deg, rgba(255,255,255,.055), rgba(255,255,255,.018));
      border: 1px solid rgba(255,255,255,.075);
      border-radius: 18px; padding: 1.25rem 1.45rem;
      backdrop-filter: blur(22px) saturate(150%);
      -webkit-backdrop-filter: blur(22px) saturate(150%);
      box-shadow: 0 18px 46px rgba(0,0,0,.36), inset 0 1px 0 rgba(255,255,255,.055);
  }}
  .glass-tight {{ padding: .9rem 1.1rem; border-radius: 14px; }}

  /* ── hero ───────────────────────────────────────────────────────────── */
  .hero {{ text-align: center; padding: 4.6rem 1rem 2.4rem; }}
  .hero-title {{
      font-size: clamp(4rem, 12vw, 8.2rem); font-weight: 700; line-height: .94;
      letter-spacing: -0.055em; margin: 0;
      background: linear-gradient(105deg, #ffffff 8%, {ACCENT} 46%, {VIOLET} 88%);
      -webkit-background-clip: text; background-clip: text; color: transparent;
      animation: rise .85s cubic-bezier(.2,.7,.2,1) both;
  }}
  .hero-sub {{ font-size: clamp(1.05rem, 2.1vw, 1.5rem); color: {INK}; font-weight: 400;
      margin-top: 1.1rem; letter-spacing: -.012em;
      animation: rise .85s .10s cubic-bezier(.2,.7,.2,1) both; }}
  .hero-lede {{ font-size: 1.02rem; color: {INK_2}; max-width: 660px; margin: 1.05rem auto 0;
      animation: rise .85s .18s cubic-bezier(.2,.7,.2,1) both; }}

  @keyframes rise {{ from {{ opacity:0; transform: translateY(20px); }}
                     to   {{ opacity:1; transform: none; }} }}
  @keyframes fade {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
  .fade-in {{ animation: fade .55s ease both; }}
  .rise-in {{ animation: rise .6s cubic-bezier(.2,.7,.2,1) both; }}

  /* ── stat tiles ─────────────────────────────────────────────────────── */
  .stat {{ text-align: left; }}
  .stat-v {{ font-size: 1.85rem; font-weight: 600; color: {INK}; letter-spacing: -.03em;
             line-height: 1.12; }}
  .stat-l {{ font-size: .74rem; color: {INK_3}; text-transform: uppercase;
             letter-spacing: .085em; margin-top: .28rem; }}
  .stat-s {{ font-size: .80rem; color: {INK_2}; margin-top: .18rem; }}

  /* ── verdict card ───────────────────────────────────────────────────── */
  .verdict {{ border-radius: 20px; padding: 1.7rem 1.9rem;
      border: 1px solid rgba(255,255,255,.09);
      background: linear-gradient(140deg, rgba(91,140,255,.14), rgba(167,139,250,.07) 55%,
                  rgba(255,255,255,.02));
      backdrop-filter: blur(24px); box-shadow: 0 20px 52px rgba(0,0,0,.4); }}
  .verdict-k {{ font-size: .74rem; letter-spacing: .1em; text-transform: uppercase;
                color: {INK_3}; }}
  .verdict-h {{ font-size: 2.0rem; font-weight: 650; letter-spacing: -.03em; color: {INK};
                margin: .3rem 0 .1rem; line-height: 1.15; }}
  .verdict-p {{ font-size: .97rem; color: {INK_2}; line-height: 1.6; margin-top: .55rem; }}

  .pill {{ display: inline-block; padding: .2rem .68rem; border-radius: 999px;
      font-size: .74rem; font-weight: 500; letter-spacing: .012em; margin-right: .4rem; }}
  .pill-ok  {{ background: rgba(52,211,153,.15); color: {GREEN};
               border: 1px solid rgba(52,211,153,.3); }}
  .pill-mid {{ background: rgba(251,191,36,.14); color: {AMBER};
               border: 1px solid rgba(251,191,36,.3); }}
  .pill-low {{ background: rgba(251,113,133,.14); color: {ROSE};
               border: 1px solid rgba(251,113,133,.3); }}
  .pill-neu {{ background: rgba(154,165,187,.12); color: {INK_2};
               border: 1px solid rgba(255,255,255,.1); }}

  /* ── scope banner ───────────────────────────────────────────────────── */
  .scope {{ border-left: 3px solid {AMBER}; background: rgba(251,191,36,.07);
      padding: .78rem 1.05rem; border-radius: 0 12px 12px 0; font-size: .87rem;
      color: {INK_2}; margin: .7rem 0; }}
  .scope b {{ color: {AMBER}; }}
  .note {{ border-left: 3px solid {ACCENT}; background: rgba(91,140,255,.07);
      padding: .78rem 1.05rem; border-radius: 0 12px 12px 0; font-size: .87rem;
      color: {INK_2}; margin: .7rem 0; }}

  /* ── stage checklist ────────────────────────────────────────────────── */
  .stage {{ display:flex; align-items:center; gap:.7rem; padding:.52rem .1rem;
      font-size:.92rem; color:{INK_3}; transition: all .4s ease; }}
  .stage.done {{ color:{INK}; }}
  .stage.active {{ color:{CYAN}; }}
  .stage-dot {{ width:1.35rem; height:1.35rem; border-radius:50%; flex:0 0 auto;
      display:flex; align-items:center; justify-content:center; font-size:.72rem;
      border:1px solid {STROKE}; }}
  .stage.done .stage-dot {{ background: rgba(52,211,153,.16); border-color: rgba(52,211,153,.4);
      color:{GREEN}; }}
  .stage.active .stage-dot {{ background: rgba(34,211,238,.16);
      border-color: rgba(34,211,238,.45); color:{CYAN}; }}

  /* ── streamlit control restyling ────────────────────────────────────── */
  .stButton > button {{
      background: linear-gradient(135deg, {ACCENT}, #6d5cf6); color: #fff; border: 0;
      border-radius: 13px; padding: .68rem 1.7rem; font-weight: 550; font-size: .96rem;
      letter-spacing: -.008em; box-shadow: 0 10px 26px rgba(91,140,255,.30);
      transition: transform .18s cubic-bezier(.2,.7,.2,1), box-shadow .18s ease; }}
  .stButton > button:hover {{ transform: translateY(-2px);
      box-shadow: 0 16px 36px rgba(91,140,255,.42); }}
  .stButton > button:active {{ transform: translateY(0); }}
  .stButton > button[kind="secondary"] {{
      background: rgba(255,255,255,.055); border: 1px solid rgba(255,255,255,.11);
      color: {INK}; box-shadow: none; font-weight: 450; }}
  .stButton > button[kind="secondary"]:hover {{ background: rgba(255,255,255,.1);
      box-shadow: none; }}

  [data-testid="stFileUploaderDropzone"] {{
      background: linear-gradient(160deg, rgba(255,255,255,.05), rgba(255,255,255,.015));
      border: 1.5px dashed rgba(255,255,255,.17); border-radius: 18px; padding: 2.4rem 1rem;
      transition: all .25s ease; }}
  [data-testid="stFileUploaderDropzone"]:hover {{ border-color: {ACCENT};
      background: rgba(91,140,255,.07); }}

  .stTabs [data-baseweb="tab-list"] {{ gap: .3rem; background: transparent;
      border-bottom: 1px solid {STROKE}; }}
  .stTabs [data-baseweb="tab"] {{ background: transparent; color: {INK_3};
      border-radius: 10px 10px 0 0; padding: .55rem 1.05rem; font-size: .92rem; }}
  .stTabs [aria-selected="true"] {{ color: {INK}; background: rgba(255,255,255,.05); }}

  [data-testid="stExpander"] {{ background: rgba(255,255,255,.028);
      border: 1px solid rgba(255,255,255,.075); border-radius: 15px; margin-bottom: .55rem; }}
  [data-testid="stExpander"] summary {{ font-size: .95rem; color: {INK}; padding: .2rem 0; }}
  [data-testid="stExpander"] summary:hover {{ color: {ACCENT}; }}
  [data-testid="stExpander"] details {{ background: transparent; }}
  [data-testid="stExpander"] details > summary {{ background: transparent !important; }}
  [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{ background: transparent; }}
  details[open] > summary {{ border-bottom: 1px solid {STROKE}; margin-bottom: .5rem; }}

  /* radio group used as a segmented control */
  [data-testid="stRadio"] > div {{ gap: .35rem; }}
  [data-testid="stRadio"] label {{ background: rgba(255,255,255,.05);
      border: 1px solid rgba(255,255,255,.09); border-radius: 10px;
      padding: .28rem .8rem !important; color: {INK_2}; }}
  [data-testid="stRadio"] label:hover {{ border-color: {ACCENT}; color: {INK}; }}

  .stDownloadButton > button {{ background: rgba(255,255,255,.055);
      border: 1px solid rgba(255,255,255,.12); color: {INK}; border-radius: 12px;
      font-weight: 450; }}
  .stDownloadButton > button:hover {{ background: rgba(91,140,255,.16);
      border-color: {ACCENT}; }}

  [data-testid="stDataFrame"] {{ border-radius: 13px; overflow: hidden;
      border: 1px solid {STROKE}; }}
  .stSlider [data-baseweb="slider"] div[role="slider"] {{ background: {ACCENT}; }}
  .stSelectbox div[data-baseweb="select"] > div, .stTextInput input {{
      background: rgba(255,255,255,.05); border-color: rgba(255,255,255,.11);
      color: {INK}; border-radius: 11px; }}
  hr {{ border-color: {STROKE}; margin: 1.9rem 0; }}
  .caption {{ font-size: .80rem; color: {INK_3}; }}

  /* ── top navigation ─────────────────────────────────────────────────── */
  .navbar {{ display:flex; align-items:center; justify-content:space-between;
      padding:.55rem 0 1.1rem; border-bottom:1px solid {STROKE}; margin-bottom:1.5rem; }}
  .brand {{ font-weight:650; font-size:1.06rem; letter-spacing:-.02em; color:{INK}; }}
  .brand span {{ color:{INK_3}; font-weight:400; margin-left:.55rem; font-size:.86rem; }}
</style>
"""
