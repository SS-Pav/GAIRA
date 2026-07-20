"""Publication ("interactive scientific paper") visual system for the V6 demo.

One restrained palette, position-first encoding, colour used sparingly. Magnitude
uses a single-hue sequential ramp; change uses a diverging blue<->red pair with a
neutral midpoint (never an 11-way categorical cycle across themes — theme identity
is carried by axis position and label text, per data-viz good practice).
"""
from __future__ import annotations

# ── ink & surfaces (light "paper" theme) ──
INK = "#1b2430"          # primary text
MUTED = "#586173"        # secondary text
FAINT = "#93a0b2"        # captions / de-emphasis
SURFACE = "#ffffff"
PANEL = "#f5f7fa"
PANEL_EDGE = "#e4e9f0"
GRID = "#e6ebf2"

# ── single accent (GAIRA deep blue) + soft fill ──
PRIMARY = "#2a6f97"
PRIMARY_SOFT = "#bcd6e6"
SECONDARY = "#3a5a7a"

# ── magnitude (sequential, light -> dark) ──
SEQ = ["#eef4f8", "#cfe1ec", "#a9cbe0", "#6ba7c9", "#2a6f97", "#1c4e6e"]

# ── change (diverging: decrease -> neutral -> increase) ──
DOWN = "#2166ac"         # cool = depletion / decrease
UP = "#b2182b"           # warm = elevation / increase
NEUTRAL = "#c7ced8"

# ── status ink (reserved; never reused as a series colour) ──
GOOD = "#2f7d4f"
WARN = "#b7791f"
BAD = "#b23a48"

# stable, muted tint per biochemical theme — used ONLY for small dots/labels,
# never as the primary quantitative encoding.
THEME_TINT = {
    "nucleic_purine": "#7b4bd1", "nucleic_pyrimidine": "#9b6dde",
    "protein_peptide": "#2a6f97", "aromatic_amino_acid": "#3f8fbf",
    "lipid_acyl": "#c8862a", "sterol_membrane": "#a9741f",
    "saccharide_glycan": "#2f9e8f", "organic_acid_metabolism": "#6aa84f",
    "sulfur_antioxidant": "#c0562b", "heme_porphyrin": "#b23a48",
    "redox_broad": "#7a6f52", "background_matrix": "#9aa4b2", "unknown_mixed": "#b7bfca",
}


def mpl_rc():
    """Matplotlib rcParams for consistent, publication-clean figures."""
    return {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": 150, "figure.dpi": 120,
        "axes.edgecolor": PANEL_EDGE, "axes.linewidth": 0.9,
        "axes.labelcolor": MUTED, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "600",
        "legend.frameon": False, "figure.autolayout": False,
    }


# Streamlit page CSS (clean, paper-like; minimal chrome)
PAGE_CSS = f"""
<style>
.stApp {{ background: {SURFACE}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }}
h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.01em; font-weight: 650; }}
p, li {{ color: {MUTED}; line-height: 1.62; }}
hr {{ border: none; border-top: 1px solid {PANEL_EDGE}; margin: 1.4rem 0; }}
.gaira-kicker {{ color: {PRIMARY}; font-size: 0.80rem; font-weight: 700;
    letter-spacing: 0.09em; text-transform: uppercase; }}
.gaira-lede {{ color: {MUTED}; font-size: 1.06rem; line-height: 1.65; }}
.gaira-caption {{ color: {FAINT}; font-size: 0.86rem; line-height: 1.5; }}
.gaira-card {{ background: {PANEL}; border: 1px solid {PANEL_EDGE};
    border-radius: 12px; padding: 1.0rem 1.15rem; }}
.gaira-caveat {{ background: #fbf7ee; border: 1px solid #ecdfc4;
    border-left: 3px solid {WARN}; border-radius: 8px; padding: 0.7rem 0.95rem;
    color: #6b5d3e; font-size: 0.92rem; }}
.gaira-take {{ background: #eef5f0; border: 1px solid #d3e6da;
    border-left: 3px solid {GOOD}; border-radius: 8px; padding: 0.7rem 0.95rem;
    color: #35543f; font-size: 0.92rem; }}
.gaira-stat {{ font-size: 1.75rem; font-weight: 700; color: {INK}; line-height: 1.1; }}
.gaira-stat-label {{ font-size: 0.80rem; color: {FAINT}; text-transform: uppercase;
    letter-spacing: 0.05em; }}
.gaira-prov {{ color: {FAINT}; font-size: 0.78rem; font-family: ui-monospace, monospace;
    border-top: 1px solid {PANEL_EDGE}; padding-top: 0.7rem; margin-top: 2rem; }}
div[data-testid="stMetricValue"] {{ color: {INK}; }}
</style>
"""
