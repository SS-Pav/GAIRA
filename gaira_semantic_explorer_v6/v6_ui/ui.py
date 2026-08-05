"""Shared UI primitives for Foundation Explorer V6 — clean, teaching-first."""
from __future__ import annotations
import streamlit as st

INK, MUTED = "#1b2430", "#5b6472"
BLUE, VERM, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"

CSS = """
<style>
  .block-container {padding-top: 2.2rem; max-width: 1180px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  .v6-eyebrow {font-size:.74rem; letter-spacing:.13em; text-transform:uppercase;
               color:#0072B2; font-weight:700; margin-bottom:.15rem;}
  .v6-title {font-size:1.95rem; font-weight:800; line-height:1.15; color:#1b2430; margin:0 0 .35rem 0;}
  .v6-lede {font-size:1.0rem; color:#5b6472; line-height:1.55; margin-bottom:1.1rem;}
  .v6-q {background:#f2f7fb; border-left:3px solid #0072B2; padding:.7rem .95rem;
         border-radius:6px; font-weight:600; color:#1b2430; margin:.5rem 0 1rem 0;}
  .v6-card {background:#fff; border:1px solid #e3e8ee; border-radius:10px;
            padding:.85rem 1rem; margin:.35rem 0;}
  .v6-card h4 {margin:0 0 .35rem 0; font-size:.92rem; color:#1b2430;}
  .v6-card p {margin:0; font-size:.84rem; color:#5b6472; line-height:1.5;}
  .v6-take {background:#1b2430; color:#fff; border-radius:10px; padding:.85rem 1.05rem;
            font-size:.9rem; line-height:1.55; margin:.9rem 0;}
  .v6-take b {color:#7ec8ff;}
  .v6-warn {background:#fdf3ee; border-left:3px solid #D55E00; padding:.7rem .95rem;
            border-radius:6px; font-size:.86rem; color:#1b2430; margin:.6rem 0;}
  .v6-good {background:#eef8f4; border-left:3px solid #009E73; padding:.7rem .95rem;
            border-radius:6px; font-size:.86rem; color:#1b2430; margin:.6rem 0;}
  .v6-cap {font-size:.78rem; color:#5b6472; font-style:italic; margin-top:.2rem;}
  .v6-stat {text-align:center; padding:.55rem .3rem; background:#f7f9fb;
            border-radius:9px; border:1px solid #e8edf2;}
  .v6-stat .v {font-size:1.28rem; font-weight:800; color:#1b2430; line-height:1.1;}
  .v6-stat .l {font-size:.68rem; color:#5b6472; text-transform:uppercase; letter-spacing:.05em;}
  section[data-testid="stSidebar"] {background:#fbfcfd;}
  hr {margin:1.3rem 0; border-color:#e8edf2;}
</style>
"""


def header(eyebrow, title, lede):
    st.markdown(f'<div class="v6-eyebrow">{eyebrow}</div>'
                f'<div class="v6-title">{title}</div>'
                f'<div class="v6-lede">{lede}</div>', unsafe_allow_html=True)


def question(q):
    st.markdown(f'<div class="v6-q">❯ {q}</div>', unsafe_allow_html=True)


def stats(items):
    cols = st.columns(len(items))
    for c, (v, l) in zip(cols, items):
        c.markdown(f'<div class="v6-stat"><div class="v">{v}</div><div class="l">{l}</div></div>',
                   unsafe_allow_html=True)


def card(title, body):
    st.markdown(f'<div class="v6-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def take(text):
    st.markdown(f'<div class="v6-take">{text}</div>', unsafe_allow_html=True)


def warn(text):
    st.markdown(f'<div class="v6-warn">{text}</div>', unsafe_allow_html=True)


def good(text):
    st.markdown(f'<div class="v6-good">{text}</div>', unsafe_allow_html=True)


def figure(path, caption=None):
    if path:
        st.image(path, use_container_width=True)
        if caption:
            st.markdown(f'<div class="v6-cap">{caption}</div>', unsafe_allow_html=True)
    else:
        st.info("figure not generated yet — run results/v6_rebuild/code/p06_figures.py")


def rule():
    st.markdown("<hr/>", unsafe_allow_html=True)


def fmt(v, n=3):
    try:
        return f"{float(v):.{n}f}"
    except Exception:
        return "—"
