"""Compact dark-theme UI primitives for the context-graph explorer."""
from __future__ import annotations

import streamlit as st


CSS = """
<style>
.cge-card {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 10px;
  padding: 14px 16px;
  margin: 4px 0 12px 0;
  color: #c9d1d9;
}
.cge-card-title {
  font-size: 1.0rem;
  font-weight: 600;
  color: #d2a8ff;
  margin-bottom: 4px;
}
.cge-card-sub {
  font-size: 0.82rem;
  color: #8b949e;
  margin-bottom: 6px;
}
.cge-metric {
  display: inline-block;
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 8px;
  padding: 10px 14px;
  margin: 0 8px 8px 0;
  min-width: 130px;
}
.cge-metric-value {
  font-size: 1.4rem; font-weight: 700; color: #79c0ff;
}
.cge-metric-label {
  font-size: 0.74rem; color: #8b949e;
  text-transform: uppercase; letter-spacing: 0.04em;
}
.cge-warn {
  background: #2d1f0a; border: 1px solid #6e4c00;
  color: #f0d98e; border-radius: 8px;
  padding: 10px 14px; margin: 6px 0 12px 0;
}
.cge-interp {
  background: #0e1d2e; border: 1px solid #1f4f8a;
  color: #c9d1d9; border-radius: 8px;
  padding: 10px 14px; margin: 8px 0; font-size: 0.88rem;
}
.cge-interp-title {
  color: #79c0ff; font-weight: 600; margin-bottom: 4px;
}
.cge-divider { border-top: 1px solid #30363d; margin: 14px 0; }
</style>
"""


def inject_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def card(title: str, body_md: str, subtitle: str | None = None) -> None:
    sub = f'<div class="cge-card-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="cge-card">'
        f'<div class="cge-card-title">{title}</div>'
        f'{sub}<div>{body_md}</div></div>',
        unsafe_allow_html=True)


def render_metric_cards(metrics: dict[str, str], cols_per_row: int = 4) -> None:
    items = list(metrics.items())
    for i in range(0, len(items), cols_per_row):
        chunk = items[i:i + cols_per_row]
        cols = st.columns(len(chunk))
        for col, (label, value) in zip(cols, chunk):
            with col:
                st.markdown(
                    f'<div class="cge-metric">'
                    f'<div class="cge-metric-value">{value}</div>'
                    f'<div class="cge-metric-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True)


def warning_card(msg: str) -> None:
    st.markdown(f'<div class="cge-warn">⚠️ {msg}</div>',
                unsafe_allow_html=True)


def interpretation(title: str, body_html: str) -> None:
    st.markdown(
        f'<div class="cge-interp">'
        f'<div class="cge-interp-title">{title}</div>'
        f'<div>{body_html}</div></div>',
        unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<div class="cge-divider"></div>', unsafe_allow_html=True)
