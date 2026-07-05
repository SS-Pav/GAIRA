"""GAIRA Context Graph Explorer — Streamlit entry point.

Run:
    streamlit run streamlit_apps/gaira_context_graph_explorer/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from components import ui_blocks as ui                      # noqa: E402
from components import overview_tab                          # noqa: E402
from components import condition_axis_graph                  # noqa: E402
from components import hierarchical_context_graph            # noqa: E402
from components import mss_transfer_graph                    # noqa: E402
from components import context_embedding_tab                 # noqa: E402
from components import evidence_tables                       # noqa: E402
from utils.load_context_data import load_all_context         # noqa: E402


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    app_cfg = _load_yaml(APP_DIR / "config" / "app_config.yaml")

    st.set_page_config(
        page_title=app_cfg.get("app", {}).get("title", "Context Graph Explorer"),
        layout=app_cfg.get("app", {}).get("layout", "wide"),
        initial_sidebar_state="expanded",
    )

    ui.inject_styles()

    paths = app_cfg.get("paths", {})
    context_root = paths.get("context_root",
                              "/Volumes/SSD_Rad/GAIRA_BUILD/"
                              "gaira_base_4_context_graph_discovery_v1")
    ctx = load_all_context(context_root)

    # Sidebar
    with st.sidebar:
        st.markdown(f"### {app_cfg['app']['title']}")
        st.caption(app_cfg["app"].get("version", "v1"))
        st.markdown("---")
        st.caption("Source")
        st.code(context_root, language="text")
        # Quick artifact health
        present = sum(1 for k, v in ctx.items()
                       if k != "_root" and v is not None)
        total = sum(1 for k in ctx if k != "_root")
        st.caption(f"tables loaded: **{present}/{total}**")
        st.markdown("---")
        st.caption("Tabs")
        st.markdown(
            "- 1 · Overview\n"
            "- 2 · Condition → Axis network\n"
            "- 3 · Hierarchical Sankey\n"
            "- 4 · MSS transfer graph\n"
            "- 5 · Context embedding")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1 · Overview",
        "2 · Condition → Axis",
        "3 · Hierarchical context",
        "4 · MSS transfer",
        "5 · Context embedding",
    ])
    with tab1:
        overview_tab.render(ctx, app_cfg)
        with st.expander("🔍 Explore raw evidence events", expanded=False):
            evidence_tables.render_events_subset(ctx.get("events"),
                                                   max_rows=400)
    with tab2:
        condition_axis_graph.render(ctx, app_cfg)
    with tab3:
        hierarchical_context_graph.render(ctx, app_cfg)
    with tab4:
        mss_transfer_graph.render(ctx, app_cfg)
    with tab5:
        context_embedding_tab.render(ctx, app_cfg)


if __name__ == "__main__":
    main()
