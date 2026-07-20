"""Shared Streamlit UI atoms — the consistent chrome every page uses.

Each page follows the same scientific-paper skeleton:
    header (kicker + title + lede)  ->  question  ->  content
    ->  key takeaways  ->  scientific caveats  ->  provenance footer.
"""
from __future__ import annotations
import streamlit as st
from . import theme as T


def inject_css():
    st.markdown(T.PAGE_CSS, unsafe_allow_html=True)


def page_header(kicker, title, lede):
    st.markdown(f'<div class="gaira-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    st.markdown(f'<div class="gaira-lede">{lede}</div>', unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)


def question(text):
    st.markdown(f'<div class="gaira-card"><b>The question this page answers.</b> '
                f'{text}</div>', unsafe_allow_html=True)
    st.write("")


def caption(text):
    st.markdown(f'<div class="gaira-caption">{text}</div>', unsafe_allow_html=True)


def figure(fig, cap=None, interp=None, limits=None):
    """Render a matplotlib figure with the required caption / interpretation / limits."""
    st.pyplot(fig, use_container_width=True)
    import matplotlib.pyplot as plt
    plt.close(fig)
    if cap:
        st.markdown(f'<div class="gaira-caption"><b>Figure.</b> {cap}</div>', unsafe_allow_html=True)
    if interp:
        st.markdown(f'<div class="gaira-caption"><b>Interpretation.</b> {interp}</div>',
                    unsafe_allow_html=True)
    if limits:
        st.markdown(f'<div class="gaira-caption"><b>Limitations.</b> {limits}</div>',
                    unsafe_allow_html=True)


def takeaways(items):
    body = "".join(f"<li>{x}</li>" for x in items)
    st.markdown(f'<div class="gaira-take"><b>Key takeaways.</b><ul style="margin:0.4rem 0 0 0;">'
                f'{body}</ul></div>', unsafe_allow_html=True)


def caveats(items):
    body = "".join(f"<li>{x}</li>" for x in items)
    st.markdown(f'<div class="gaira-caveat"><b>Scientific caveats.</b>'
                f'<ul style="margin:0.4rem 0 0 0;">{body}</ul></div>', unsafe_allow_html=True)


def stat_row(stats):
    """stats: list of (value, label). Renders evenly spaced stat tiles."""
    cols = st.columns(len(stats))
    for col, (val, label) in zip(cols, stats):
        with col:
            st.markdown(f'<div class="gaira-stat">{val}</div>'
                        f'<div class="gaira-stat-label">{label}</div>', unsafe_allow_html=True)


def provenance_footer(stats):
    v = stats["versions"]
    st.markdown(
        f'<div class="gaira-prov">Raman Reference Atlas {v.get("atlas", "v0.1")} · '
        f'fingerprint {stats["fingerprint"][:12]}… · ontology {v.get("biochemical_ontology", "v2")} · '
        f'MSS v1 · BSV v2 · frozen &amp; deterministic — the demo measures this system, '
        f'it does not modify it.</div>', unsafe_allow_html=True)


def scaffold_note(planned):
    """Honest placeholder for a page still being built out (page-by-page cadence)."""
    body = "".join(f"<li>{x}</li>" for x in planned)
    st.markdown(
        f'<div class="gaira-card"><b>This page is scaffolded.</b> The V6 demo is built '
        f'page by page on the frozen engine. This page renders its scientific frame now; '
        f'the interactive panels below are being wired to the engine next.'
        f'<ul style="margin:0.6rem 0 0 0;">{body}</ul></div>', unsafe_allow_html=True)
