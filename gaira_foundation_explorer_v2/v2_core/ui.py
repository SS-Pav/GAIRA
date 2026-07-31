"""UI helpers + palette for Foundation Explorer V2. Readable, high-contrast, colorblind-safe
(Okabe-Ito). Kept intentionally small and dependency-light so pages stay testable.
"""
from __future__ import annotations
import streamlit as st

# Okabe-Ito colorblind-safe palette (shared with the figure scripts)
OI = {"black": "#111418", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7",
      "grey": "#7A828C"}
INK = "#1b2430"; MUTED = "#5b6472"; FAINT = "#8a93a0"; LINE = "#e4e8ed"
QUADRANT_COLOR = {
    "Q1 identity preserved (both)": OI["blue"],
    "Q2 latent redistribution, theme survives": OI["green"],
    "Q3 superficial coord match, theme changes": OI["orange"],
    "Q4 poor transfer (both)": OI["verm"]}
LEVEL_COLOR = {1: OI["verm"], 2: OI["blue"], 3: OI["green"], 4: OI["purple"]}


def inject_css():
    st.markdown(f"""
    <style>
      .v2-eyebrow {{ letter-spacing:.14em; text-transform:uppercase; font-size:.72rem;
                    color:{OI['blue']}; font-weight:700; margin-bottom:.2rem }}
      .v2-title {{ font-family:Newsreader,Georgia,serif; font-size:2.0rem; font-weight:600;
                  color:{INK}; line-height:1.12; margin:.1rem 0 .5rem }}
      .v2-lead {{ font-size:1.02rem; color:{MUTED}; line-height:1.6; max-width:60rem }}
      .v2-note {{ border-left:3px solid {OI['blue']}; background:#f2f7fb; padding:.7rem 1rem;
                 border-radius:0 6px 6px 0; margin:1rem 0; color:{INK}; font-size:.92rem }}
      .v2-warn {{ border-left:3px solid {OI['verm']}; background:#fdf3ee; padding:.7rem 1rem;
                 border-radius:0 6px 6px 0; margin:1rem 0; color:{INK}; font-size:.92rem }}
      .v2-good {{ border-left:3px solid {OI['green']}; background:#eef8f4; padding:.7rem 1rem;
                 border-radius:0 6px 6px 0; margin:1rem 0; color:{INK}; font-size:.92rem }}
      .v2-take {{ background:{INK}; color:#f6f8fa; padding:1rem 1.2rem; border-radius:8px;
                 margin:1.2rem 0; font-size:1.0rem; line-height:1.55 }}
      .v2-cap {{ color:{FAINT}; font-size:.82rem; margin:.2rem 0 1.1rem }}
      .stat-wrap {{ display:flex; gap:1.4rem; flex-wrap:wrap; margin:1rem 0 .4rem }}
      .stat b {{ display:block; font-size:1.6rem; color:{INK}; font-family:Newsreader,serif }}
      .stat span {{ color:{FAINT}; font-size:.78rem }}
    </style>""", unsafe_allow_html=True)


def header(eyebrow: str, title: str, lead: str):
    st.markdown(f'<div class="v2-eyebrow">{eyebrow}</div>'
                f'<div class="v2-title">{title}</div>'
                f'<div class="v2-lead">{lead}</div>', unsafe_allow_html=True)
    st.write("")


def note(text: str):
    st.markdown(f'<div class="v2-note">{text}</div>', unsafe_allow_html=True)


def warn(text: str):
    st.markdown(f'<div class="v2-warn">{text}</div>', unsafe_allow_html=True)


def good(text: str):
    st.markdown(f'<div class="v2-good">{text}</div>', unsafe_allow_html=True)


def takehome(text: str):
    st.markdown(f'<div class="v2-take">{text}</div>', unsafe_allow_html=True)


def caption(text: str):
    st.markdown(f'<div class="v2-cap">{text}</div>', unsafe_allow_html=True)


def stats(items):
    """items: list of (value, label)."""
    html = '<div class="stat-wrap">'
    for v, lab in items:
        html += f'<div class="stat"><b>{v}</b><span>{lab}</span></div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


def figure(path, caption_text: str | None = None):
    if path is None:
        st.info("Figure not found — run `code/make_figures.py`.")
        return
    st.image(str(path), use_container_width=True)
    if caption_text:
        caption(caption_text)
