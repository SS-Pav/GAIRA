from __future__ import annotations

import streamlit as st

from app.services.plots import plot_spectrum, plot_spectrum_pair


def render_single_spectrum(
    x_values,
    y_values,
    *,
    title: str,
    color: str = "#1d4ed8",
    apply_baseline_correction: bool = False,
) -> None:
    figure, method = plot_spectrum(
        x_values,
        y_values,
        title=title,
        color=color,
        apply_baseline_correction=apply_baseline_correction,
    )
    st.pyplot(figure, width="stretch")
    if apply_baseline_correction:
        st.caption(
            f"Display-only preprocessing active ({method or 'baseline correction'}). Inference still uses the current backend processed representation."
        )


def render_pair_spectrum(
    left,
    right,
    *,
    show_difference: bool = False,
    apply_baseline_correction: bool = False,
) -> None:
    figure, method = plot_spectrum_pair(
            left["x"],
            left["y"],
            right["x"],
            right["y"],
            left_label=left["label"],
            right_label=right["label"],
            show_difference=show_difference,
            apply_baseline_correction=apply_baseline_correction,
    )
    st.pyplot(figure, width="stretch")
    if apply_baseline_correction:
        st.caption(
            f"Display-only preprocessing active ({method or 'baseline correction'}). Inference and holdout metrics still use the current backend processed representation."
        )
