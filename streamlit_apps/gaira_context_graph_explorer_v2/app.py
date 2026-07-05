"""GAIRA Context Graph Explorer · v2 — Streamlit entry point.

Run:
    streamlit run streamlit_apps/gaira_context_graph_explorer_v2/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from components import ui_blocks as ui                      # noqa: E402
from components import overview                              # noqa: E402
from components import condition_axis_graph                  # noqa: E402
from components import condition_specific_graphs             # noqa: E402
from components import hierarchical_context_graph            # noqa: E402
from components import mss_transfer_graph                    # noqa: E402
from components import sample_type_comparison                # noqa: E402
from components import context_embedding_tab                 # noqa: E402
from utils.load_context_data import load_v2_context          # noqa: E402


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    app_cfg = _load_yaml(APP_DIR / "config" / "app_config.yaml")

    st.set_page_config(
        page_title=app_cfg.get("app", {}).get("title",
                                                "Context Graph Explorer · v2"),
        layout=app_cfg.get("app", {}).get("layout", "wide"),
        initial_sidebar_state="expanded",
    )

    ui.inject_styles()

    paths = app_cfg.get("paths", {})
    context_root = paths.get("context_root",
                              "/Volumes/SSD_Rad/GAIRA_BUILD/"
                              "gaira_base_4_context_graph_discovery_v1")
    short_labels = app_cfg.get("dataset_short_labels", {})
    ctx = load_v2_context(context_root, str(APP_DIR), short_labels)

    # Sidebar
    with st.sidebar:
        st.markdown(f"### {app_cfg['app']['title']}")
        st.caption(app_cfg["app"].get("version", "v2"))
        st.markdown("---")
        st.caption("Source")
        st.code(context_root, language="text")
        ev = ctx.get("events_v2")
        if ev is not None and "specific_condition" in ev.columns:
            mapped = (ev["specific_condition"] != "unmapped").sum()
            total = len(ev)
            st.caption(f"events mapped: **{mapped}/{total}**  "
                       f"({mapped/total*100:.0f}%)")
            st.caption(f"specific conditions: "
                       f"**{ev['specific_condition'].nunique()}**")
        st.markdown("---")
        st.caption("Tabs")
        st.markdown(
            "- 1 · Overview\n"
            "- 2 · Condition → Axis programs\n"
            "- 3 · Specific Condition Explorer\n"
            "- 4 · Hierarchical Context Flow\n"
            "- 5 · MSS Transfer · candidate layer\n"
            "- 6 · EV vs Serum Comparison\n"
            "- 7 · Context Embeddings")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "1 · Overview",
        "2 · Condition → Axis",
        "3 · Specific condition",
        "4 · Hierarchical flow",
        "5 · MSS transfer",
        "6 · EV vs Serum",
        "7 · Context embeddings",
    ])
    with tab1:
        overview.render(ctx, app_cfg)
    with tab2:
        condition_axis_graph.render(ctx, app_cfg)
    with tab3:
        condition_specific_graphs.render(ctx, app_cfg)
    with tab4:
        hierarchical_context_graph.render(ctx, app_cfg)
    with tab5:
        mss_transfer_graph.render(ctx, app_cfg)
    with tab6:
        sample_type_comparison.render(ctx, app_cfg)
    with tab7:
        context_embedding_tab.render(ctx, app_cfg)


if __name__ == "__main__":
    main()
