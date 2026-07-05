"""Helper module — emergent-paths table renderer reusable from any tab.

The actual table is computed in `utils.load_context_data.build_emergent_paths`
and surfaced in Tab 4 (hierarchical_context_graph). This module exposes a
thin renderer in case a future tab wants to reuse the same view.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render_emergent_paths(paths: pd.DataFrame, top_n: int = 30,
                            short_labels: dict[str, str] | None = None) -> None:
    if paths is None or paths.empty:
        st.info("No emergent paths available.")
        return
    short = short_labels or {}
    view = paths.head(top_n).copy()
    view["dataset"] = view["dataset"].map(lambda d: short.get(d, d[:24]))
    cols_show = ["sample_type", "dataset", "specific_condition",
                  "bsv_axis", "dom_direction", "n_events",
                  "mean_abs_effect", "consistency", "path_score",
                  "top_mss", "confidence_tier"]
    cols_show = [c for c in cols_show if c in view.columns]
    st.dataframe(view[cols_show], use_container_width=True, hide_index=True)
