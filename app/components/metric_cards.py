from __future__ import annotations

import streamlit as st


def render_metric_cards(metrics: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value, help_text) in zip(columns, metrics):
        with column:
            st.markdown(
                f"""
                <div style="border:1px solid #d1d5db;border-radius:14px;padding:16px 18px;background:#ffffff;min-height:110px;">
                  <div style="font-size:0.88rem;color:#4b5563;">{label}</div>
                  <div style="font-size:1.7rem;font-weight:700;color:#111827;margin-top:6px;">{value}</div>
                  <div style="font-size:0.8rem;color:#6b7280;margin-top:8px;">{help_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
