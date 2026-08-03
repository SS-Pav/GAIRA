"""UI helpers + palette for Foundation Explorer V6 (Okabe-Ito, readable, V1-style)."""
from __future__ import annotations
import streamlit as st

OI = {"black": "#111418", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "grey": "#7A828C"}
INK = "#1b2430"; MUTED = "#5b6472"; FAINT = "#8a93a0"
LEVEL_COLOR = {1: OI["verm"], 2: OI["orange"], 3: OI["blue"], 4: OI["green"], 5: OI["purple"]}


def inject_css():
    st.markdown(f"""<style>
      .v6-eyebrow {{letter-spacing:.14em;text-transform:uppercase;font-size:.72rem;color:{OI['blue']};font-weight:700;margin-bottom:.2rem}}
      .v6-title {{font-family:Newsreader,Georgia,serif;font-size:1.95rem;font-weight:600;color:{INK};line-height:1.12;margin:.1rem 0 .5rem}}
      .v6-lead {{font-size:1.02rem;color:{MUTED};line-height:1.6;max-width:62rem}}
      .v6-note {{border-left:3px solid {OI['blue']};background:#f2f7fb;padding:.7rem 1rem;border-radius:0 6px 6px 0;margin:1rem 0;color:{INK};font-size:.92rem}}
      .v6-warn {{border-left:3px solid {OI['verm']};background:#fdf3ee;padding:.7rem 1rem;border-radius:0 6px 6px 0;margin:1rem 0;color:{INK};font-size:.92rem}}
      .v6-good {{border-left:3px solid {OI['green']};background:#eef8f4;padding:.7rem 1rem;border-radius:0 6px 6px 0;margin:1rem 0;color:{INK};font-size:.92rem}}
      .v6-take {{background:{INK};color:#f6f8fa;padding:1rem 1.2rem;border-radius:8px;margin:1.2rem 0;font-size:1.0rem;line-height:1.55}}
      .v6-cap {{color:{FAINT};font-size:.82rem;margin:.2rem 0 1.1rem}}
      .lvl {{display:inline-block;padding:.15rem .5rem;border-radius:4px;color:white;font-size:.78rem;font-weight:700;margin-right:.4rem}}
      .stat-wrap {{display:flex;gap:1.4rem;flex-wrap:wrap;margin:1rem 0 .4rem}}
      .stat b {{display:block;font-size:1.55rem;color:{INK};font-family:Newsreader,serif}}
      .stat span {{color:{FAINT};font-size:.78rem}}
    </style>""", unsafe_allow_html=True)


def header(e, t, l):
    st.markdown(f'<div class="v6-eyebrow">{e}</div><div class="v6-title">{t}</div><div class="v6-lead">{l}</div>', unsafe_allow_html=True)
    st.write("")
def note(t): st.markdown(f'<div class="v6-note">{t}</div>', unsafe_allow_html=True)
def warn(t): st.markdown(f'<div class="v6-warn">{t}</div>', unsafe_allow_html=True)
def good(t): st.markdown(f'<div class="v6-good">{t}</div>', unsafe_allow_html=True)
def takehome(t): st.markdown(f'<div class="v6-take">{t}</div>', unsafe_allow_html=True)
def caption(t): st.markdown(f'<div class="v6-cap">{t}</div>', unsafe_allow_html=True)
def level_badge(n, label): return f'<span class="lvl" style="background:{LEVEL_COLOR[n]}">L{n}</span>{label}'
def stats(items):
    h = '<div class="stat-wrap">'
    for v, lab in items: h += f'<div class="stat"><b>{v}</b><span>{lab}</span></div>'
    st.markdown(h + "</div>", unsafe_allow_html=True)
def figure(path, cap=None):
    if path is None: st.info("Figure not found — run make_figures_v4.py."); return
    st.image(str(path), use_container_width=True)
    if cap: caption(cap)
