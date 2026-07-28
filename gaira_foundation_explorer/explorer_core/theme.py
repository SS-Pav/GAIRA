"""Visual theme for the GAIRA Foundation Explorer — an interactive review-article look:
light, generous whitespace, a restrained academic palette, serif display headings."""

# palette (academic, print-friendly; matches the audit figures for continuity)
INK = "#16202c"        # near-black text
MUTED = "#4a5563"      # secondary text
FAINT = "#7b8694"      # captions
LINE = "#e3e7ec"       # hairlines
PAPER = "#fdfdfc"      # page surface
CARD = "#ffffff"       # card surface
NAVY = "#2a6f97"       # primary / hero
NAVY_D = "#1d4e6b"
ACCENT = "#0f766e"     # teal secondary
UP = "#b2182b"         # diverging warm pole
DOWN = "#2166ac"       # diverging cool pole
GOOD = "#2f7d4f"
WARN = "#b26a00"
GOLD = "#a9791c"

PAGE_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600;6..72,700&family=Inter:wght@400;500;600;700&display=swap');

:root {{
  --ink:{INK}; --muted:{MUTED}; --faint:{FAINT}; --line:{LINE};
  --paper:{PAPER}; --card:{CARD}; --navy:{NAVY}; --accent:{ACCENT};
  --up:{UP}; --down:{DOWN}; --good:{GOOD};
}}

.stApp {{ background:{PAPER}; }}
.block-container {{ max-width: 1060px; padding-top: 2.2rem; padding-bottom: 5rem; }}

html, body, [class*="css"], .stMarkdown, p, li {{
  font-family:'Inter', -apple-system, system-ui, sans-serif;
  color:{INK}; font-size:1.02rem; line-height:1.68;
}}
h1,h2,h3,h4 {{ font-family:'Newsreader', Georgia, serif; color:{INK};
  letter-spacing:-0.01em; font-weight:600; }}
h1 {{ font-size:2.35rem; line-height:1.14; margin:.1rem 0 .3rem; }}
h2 {{ font-size:1.62rem; margin:2.0rem 0 .5rem; }}
h3 {{ font-size:1.24rem; margin:1.4rem 0 .4rem; }}
a {{ color:{NAVY}; text-decoration:none; border-bottom:1px solid {LINE}; }}

/* eyebrow / section kicker */
.eyebrow {{ font-family:'Inter'; text-transform:uppercase; letter-spacing:.16em;
  font-size:.72rem; font-weight:700; color:{NAVY}; margin-bottom:.35rem; }}
.lead {{ font-family:'Newsreader', serif; font-size:1.28rem; line-height:1.55;
  color:{MUTED}; font-weight:400; margin:.4rem 0 1.1rem; }}

.rule {{ height:1px; background:{LINE}; border:0; margin:2.0rem 0 1.3rem; }}
.section-num {{ display:inline-block; font-family:'Inter'; font-weight:700; color:{NAVY};
  font-size:.9rem; background:{NAVY}12; border:1px solid {NAVY}30; border-radius:6px;
  padding:.05rem .5rem; margin-right:.5rem; }}

/* question banner */
.question {{ background:linear-gradient(180deg,#f4f8fb,#eef4f8); border:1px solid #dce7ef;
  border-left:4px solid {NAVY}; border-radius:8px; padding:.85rem 1.1rem; margin:.4rem 0 1.2rem;
  font-family:'Newsreader',serif; font-size:1.16rem; color:{NAVY_D}; }}
.question b {{ color:{NAVY_D}; }}

/* cards */
.card {{ background:{CARD}; border:1px solid {LINE}; border-radius:12px; padding:1.15rem 1.3rem;
  margin:.5rem 0; box-shadow:0 1px 2px rgba(20,32,44,.04); }}
.card h4 {{ margin:.1rem 0 .5rem; }}

/* figure card scaffolding (Q / method / result / interpretation / takehome) */
.figmeta {{ font-size:.95rem; color:{MUTED}; margin:.1rem 0; }}
.figmeta .tag {{ display:inline-block; min-width:104px; font-weight:700; color:{NAVY};
  font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; }}
.takehome {{ background:#f2f8f4; border:1px solid #cfe6d8; border-radius:8px;
  padding:.6rem .9rem; margin:.5rem 0 .2rem; color:#1c4b30; font-size:.98rem; }}
.takehome b {{ color:#143a24; }}

/* callouts */
.note {{ border-radius:8px; padding:.7rem 1rem; margin:.6rem 0; font-size:.98rem; }}
.note.take {{ background:#f2f8f4; border:1px solid #cfe6d8; color:#1c4b30; }}
.note.caveat {{ background:#fbf4ee; border:1px solid #eeddca; color:#6b4a22; }}
.note.info {{ background:#f3f6fb; border:1px solid #dbe4f0; color:#274060; }}

/* stat tiles */
.stat-row {{ display:flex; gap:.7rem; flex-wrap:wrap; margin:.5rem 0 1rem; }}
.stat {{ flex:1 1 130px; background:{CARD}; border:1px solid {LINE}; border-radius:10px;
  padding:.7rem .85rem; }}
.stat .v {{ font-family:'Newsreader',serif; font-size:1.7rem; font-weight:600; color:{NAVY_D};
  line-height:1.05; }}
.stat .l {{ font-size:.76rem; color:{FAINT}; text-transform:uppercase; letter-spacing:.06em;
  margin-top:.2rem; }}

/* pills / badges */
.pill {{ display:inline-block; font-size:.72rem; font-weight:600; padding:.12rem .55rem;
  border-radius:999px; margin:.1rem .2rem .1rem 0; }}
.pill.train {{ background:#e8f2ea; color:#1c6b3a; border:1px solid #bfe0cb; }}
.pill.val {{ background:#eef2fb; color:#274a86; border:1px solid #cdd9f0; }}
.pill.unused {{ background:#f2f2f2; color:#666; border:1px solid #e0e0e0; }}
.pill.clean {{ background:#e8f2ea; color:#1c6b3a; }}
.pill.mixed {{ background:#fbf1e6; color:#8a5a1c; }}
.pill.hero {{ background:{NAVY}; color:#fff; }}

/* flow diagram */
.flow {{ display:flex; align-items:center; justify-content:center; flex-wrap:wrap;
  gap:.15rem; margin:1rem 0; }}
.flow .node {{ background:{CARD}; border:1px solid {LINE}; border-radius:9px;
  padding:.55rem .8rem; text-align:center; font-size:.9rem; font-weight:600; color:{INK};
  box-shadow:0 1px 2px rgba(20,32,44,.05); }}
.flow .node.hi {{ border-color:{NAVY}; background:{NAVY}0d; color:{NAVY_D}; }}
.flow .node .sub {{ display:block; font-weight:400; font-size:.74rem; color:{FAINT}; }}
.flow .arrow {{ color:{FAINT}; font-size:1.15rem; padding:0 .2rem; }}

/* dataframe + tables a touch tighter */
[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:10px; }}
.small {{ font-size:.86rem; color:{FAINT}; }}
blockquote {{ border-left:3px solid {NAVY}; padding-left:.9rem; color:{MUTED};
  font-family:'Newsreader',serif; font-size:1.08rem; }}

/* sidebar */
[data-testid="stSidebar"] {{ background:#f7f8f6; border-right:1px solid {LINE}; }}
[data-testid="stSidebar"] .stRadio label {{ font-size:.95rem; }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {{ font-size:1.05rem; }}

footer, #MainMenu {{ visibility:hidden; }}
</style>
"""
