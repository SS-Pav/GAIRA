"""Reusable Streamlit UI primitives — cards, headers, metric rows, warnings.

Visual style: clean dark, scientific. Cards use light borders and muted
foreground colors so the content (data, plots, captions) stays salient.
"""
from __future__ import annotations

import streamlit as st


# ─── styling ────────────────────────────────────────────────────────────────

CARD_CSS = """
<style>
.gaira-card {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 10px;
  padding: 16px 18px;
  margin: 6px 0 14px 0;
  color: #c9d1d9;
}
.gaira-card-title {
  font-size: 1.02rem;
  font-weight: 600;
  color: #d2a8ff;
  margin-bottom: 4px;
}
.gaira-card-subtitle {
  font-size: 0.85rem;
  color: #8b949e;
  margin-bottom: 8px;
}
.gaira-metric {
  display: inline-block;
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 8px;
  padding: 10px 14px;
  margin-right: 10px;
  margin-bottom: 8px;
  min-width: 110px;
  text-align: left;
}
.gaira-metric-value {
  font-size: 1.35rem;
  font-weight: 700;
  color: #79c0ff;
}
.gaira-metric-label {
  font-size: 0.75rem;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.gaira-warn {
  background: #2d1f0a;
  border: 1px solid #6e4c00;
  border-radius: 8px;
  padding: 10px 14px;
  color: #f0d98e;
  margin: 6px 0 12px 0;
}
.gaira-pill {
  display: inline-block;
  background: #1f2933;
  color: #79c0ff;
  border-radius: 12px;
  padding: 2px 10px;
  font-size: 0.78rem;
  margin: 2px 4px 2px 0;
}
.gaira-disabled {
  background: #11151a;
  border: 1px dashed #30363d;
  color: #6e7681;
}
.gaira-step {
  display: inline-block;
  background: #1f2933;
  color: #79c0ff;
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 6px 10px;
  margin-right: 6px;
  margin-bottom: 6px;
  font-size: 0.85rem;
  font-weight: 500;
}
.gaira-arrow {
  color: #6e7681;
  margin: 0 2px;
}
.gaira-divider {
  border-top: 1px solid #30363d;
  margin: 14px 0;
}
.gaira-cluster-card {
  background: #0d1117;
  border-left: 3px solid #d2a8ff;
  padding: 10px 14px;
  margin: 6px 0;
  border-radius: 4px;
}
.gaira-cluster-card-title {
  color: #d2a8ff;
  font-weight: 600;
  font-size: 0.95rem;
}
.gaira-cluster-stat {
  color: #8b949e;
  font-size: 0.82rem;
  margin-top: 2px;
}
.gaira-interp {
  background: #0e1d2e;
  border: 1px solid #1f4f8a;
  color: #c9d1d9;
  border-radius: 8px;
  padding: 10px 14px;
  margin: 8px 0;
  font-size: 0.88rem;
}
.gaira-interp-title {
  color: #79c0ff;
  font-weight: 600;
  margin-bottom: 4px;
}
</style>
"""


def inject_styles() -> None:
    st.markdown(CARD_CSS, unsafe_allow_html=True)


# ─── primitives ─────────────────────────────────────────────────────────────

def section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def card(title: str, body_md: str, subtitle: str | None = None,
         disabled: bool = False) -> None:
    cls = "gaira-card gaira-disabled" if disabled else "gaira-card"
    sub_html = f'<div class="gaira-card-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="{cls}">'
        f'<div class="gaira-card-title">{title}</div>'
        f'{sub_html}'
        f'<div>{body_md}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def artifact_card(title: str, value: str, subtitle: str | None = None) -> None:
    """A small card emphasising a number (e.g. dataset coverage)."""
    sub_html = f'<div class="gaira-card-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="gaira-card">'
        f'<div class="gaira-card-title">{title}</div>'
        f'<div class="gaira-metric-value">{value}</div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def metric_row(metrics: dict[str, str], cols_per_row: int = 4) -> None:
    """Render a dict of {label: value} as a row of inline metric cards."""
    items = list(metrics.items())
    for i in range(0, len(items), cols_per_row):
        chunk = items[i:i + cols_per_row]
        cols = st.columns(len(chunk))
        for col, (label, value) in zip(cols, chunk):
            with col:
                st.markdown(
                    f'<div class="gaira-metric">'
                    f'<div class="gaira-metric-value">{value}</div>'
                    f'<div class="gaira-metric-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def warning_card(message: str) -> None:
    st.markdown(f'<div class="gaira-warn">⚠️ {message}</div>',
                unsafe_allow_html=True)


def pipeline_flow(steps: list[str]) -> None:
    """Inline horizontal pipeline steps with arrows between them."""
    parts = []
    for i, s in enumerate(steps):
        parts.append(f'<span class="gaira-step">{s}</span>')
        if i < len(steps) - 1:
            parts.append('<span class="gaira-arrow">▶</span>')
    st.markdown(
        f'<div style="line-height: 2.2; padding: 10px 0;">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def pill_list(labels: list[str]) -> None:
    parts = [f'<span class="gaira-pill">{lbl}</span>' for lbl in labels]
    st.markdown(f'<div>{"".join(parts)}</div>', unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<div class="gaira-divider"></div>', unsafe_allow_html=True)


def cluster_card(title: str, stats: dict[str, str]) -> None:
    """Compact card listing per-cluster stats (dominant class, members, bands)."""
    rows = "".join(
        f'<div class="gaira-cluster-stat"><strong>{k}:</strong> {v}</div>'
        for k, v in stats.items()
    )
    st.markdown(
        f'<div class="gaira-cluster-card">'
        f'<div class="gaira-cluster-card-title">{title}</div>'
        f'{rows}'
        f'</div>',
        unsafe_allow_html=True,
    )


def interpretation(title: str, body_html: str) -> None:
    """Blue-tinted interpretation panel — for explainer text under figures."""
    st.markdown(
        f'<div class="gaira-interp">'
        f'<div class="gaira-interp-title">{title}</div>'
        f'<div>{body_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
