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


def debug_panel(bridge, coord, dataset_id, extra=None):
    """Developer/debug view (hidden unless enabled in the sidebar): the selected data
    identifier, inference hash, BSV/MSS/component vectors and cache key (Part 11)."""
    if not st.session_state.get("_debug"):
        return
    from .engine_bridge import cache_key, inference_hash
    import numpy as np
    out, acts = bridge.bsv_and_mss(coord)
    with st.expander("🔧 debug — inference identity & vectors", expanded=False):
        st.markdown(f"- dataset: `{dataset_id}`\n- inference hash: `{inference_hash(coord)}`\n"
                    f"- cache key: `{cache_key()}`")
        if extra:
            st.markdown("\n".join(f"- {k}: `{v}`" for k, v in extra.items()))
        st.markdown(f"- OOD `{out.bsv.ood_score:.3f}` · overall_conf "
                    f"`{out.bsv.overall_confidence:.3f}`")
        st.write("component_coord (24):", np.round(out.bsv.component_coord, 4).tolist())
        st.write("BSV composition:", {k: round(v, 4) for k, v in out.bsv.composition.items()})
        st.write("MSS composition:", {a.id: round(a.composition, 4) for a in acts})


def goto(label):
    """Request a jump to another page (consumed by app.py before the nav radio)."""
    st.session_state["_pending_nav"] = label
    st.rerun()


def related(pairs):
    """Render 'related pages' navigation buttons. pairs: list of page labels."""
    st.markdown('<div class="gaira-caption" style="margin-top:1.5rem;">Related pages</div>',
                unsafe_allow_html=True)
    cols = st.columns(len(pairs))
    for col, label in zip(cols, pairs):
        with col:
            if st.button(f"→ {label}", key=f"rel_{label}", use_container_width=True):
                goto(label)


def scaffold_note(planned):
    """Honest placeholder for a page still being built out (page-by-page cadence)."""
    body = "".join(f"<li>{x}</li>" for x in planned)
    st.markdown(
        f'<div class="gaira-card"><b>This page is scaffolded.</b> The V6 demo is built '
        f'page by page on the frozen engine. This page renders its scientific frame now; '
        f'the interactive panels below are being wired to the engine next.'
        f'<ul style="margin:0.6rem 0 0 0;">{body}</ul></div>', unsafe_allow_html=True)
