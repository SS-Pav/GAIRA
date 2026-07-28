"""Reusable presentation components for the GAIRA Foundation Explorer."""
from __future__ import annotations
import html
import streamlit as st

from . import theme as T
from . import data as D


def inject_css():
    st.markdown(T.PAGE_CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, lead: str):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    st.markdown(f'<div class="lead">{lead}</div>', unsafe_allow_html=True)


def section(num: str, title: str):
    st.markdown(f'<h2><span class="section-num">{num}</span>{title}</h2>', unsafe_allow_html=True)


def rule():
    st.markdown('<hr class="rule">', unsafe_allow_html=True)


def question(text: str):
    st.markdown(f'<div class="question">❖ &nbsp;{text}</div>', unsafe_allow_html=True)


def note(kind: str, text: str):
    st.markdown(f'<div class="note {kind}">{text}</div>', unsafe_allow_html=True)


def takehome(text: str):
    st.markdown(f'<div class="takehome"><b>Take-home.</b> {text}</div>', unsafe_allow_html=True)


def stat_row(items):
    cells = "".join(
        f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>'
        for v, l in items)
    st.markdown(f'<div class="stat-row">{cells}</div>', unsafe_allow_html=True)


def figure_card(fig_name: str, *, question: str = None, method: str = None,
                result: str = None, interpretation: str = None, takehome_text: str = None,
                caption: str = None, width: str = "stretch"):
    """A publication-style figure block: the image (native fullscreen), then a
    structured Question / Method / Result / Interpretation / Take-home read-out."""
    p = D.figure(fig_name)
    if p is None:
        note("caveat", f"Figure <code>{fig_name}</code> not found in the audit.")
        return
    st.image(str(p), width=width)
    if caption:
        st.markdown(f'<div class="small">{caption}</div>', unsafe_allow_html=True)
    rows = []
    for tag, val in [("Question", question), ("Method", method), ("Result", result),
                     ("Interpretation", interpretation)]:
        if val:
            rows.append(f'<div class="figmeta"><span class="tag">{tag}</span>{val}</div>')
    if rows:
        st.markdown("".join(rows), unsafe_allow_html=True)
    if takehome_text:
        takehome(takehome_text)


def flow(nodes, highlight=None, arrow="→"):
    """Horizontal flow diagram. nodes = [(label, sublabel|None), ...]."""
    highlight = highlight or set()
    parts = []
    for i, n in enumerate(nodes):
        label, sub = (n if isinstance(n, (list, tuple)) else (n, None))
        cls = "node hi" if i in highlight or label in highlight else "node"
        subhtml = f'<span class="sub">{sub}</span>' if sub else ""
        parts.append(f'<div class="{cls}">{label}{subhtml}</div>')
        if i < len(nodes) - 1:
            parts.append(f'<div class="arrow">{arrow}</div>')
    st.markdown(f'<div class="flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def pills(items):
    """items = [(text, css_class), ...]"""
    html_ = "".join(f'<span class="pill {c}">{t}</span>' for t, c in items)
    st.markdown(html_, unsafe_allow_html=True)


def report_expander(report_name: str, label: str = "Read the full audit report"):
    with st.expander(f"📄 {label}"):
        st.markdown(D.load_report(report_name))


def card(title: str, body_md: str):
    st.markdown(f'<div class="card"><h4>{title}</h4>', unsafe_allow_html=True)
    st.markdown(body_md)
    st.markdown("</div>", unsafe_allow_html=True)


def fmt(x, nd=3, dash="—"):
    try:
        if x is None:
            return dash
        f = float(x)
        if f == int(f) and abs(f) >= 1:
            return f"{int(f)}"
        return f"{f:.{nd}g}"
    except (TypeError, ValueError):
        return str(x) if x is not None else dash
