"""GAIRA Command Center — Streamlit app entry point (Tabs 1+2 v1).

Run:
    streamlit run streamlit_apps/gaira_command_center/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
import streamlit as st

# Make this app's local packages importable when run from project root
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from components import ui_blocks as ui            # noqa: E402
from components import overview_tab               # noqa: E402
from components import motif_mss_bsv_tab          # noqa: E402
from utils.artifact_loader import (               # noqa: E402
    ensure_manifest, build_manifest, write_manifest,
)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _resolve_manifest(app_cfg: dict, force_rebuild: bool = False) -> tuple[dict, Path]:
    paths = app_cfg.get("paths", {})
    build_root = Path(paths.get("build_root", "/Volumes/SSD_Rad/GAIRA_BUILD"))
    manifest_path = APP_DIR / paths.get("manifest", "config/artifact_manifest.yaml")
    phases = app_cfg.get("artifact_phase_folders", [])
    if force_rebuild:
        manifest = build_manifest(build_root, phases)
        write_manifest(manifest, manifest_path)
        return manifest, manifest_path
    manifest = ensure_manifest(build_root, phases, manifest_path, rebuild=False)
    return manifest, manifest_path


def main() -> None:
    app_cfg = _load_yaml(APP_DIR / "config" / "app_config.yaml")
    evidence_cfg = _load_yaml(APP_DIR / "config" / "evidence_layers.yaml")

    st.set_page_config(
        page_title=app_cfg.get("app", {}).get("title", "GAIRA Command Center"),
        layout=app_cfg.get("app", {}).get("layout", "wide"),
        initial_sidebar_state="expanded",
    )

    ui.inject_styles()

    # Top-level config dict for tab-render contracts
    top_cfg = {
        "title": app_cfg.get("app", {}).get("title", "GAIRA Command Center"),
        "subtitle": app_cfg.get("app", {}).get("subtitle", ""),
        "version": app_cfg.get("app", {}).get("version", "v1"),
        "roadmap": app_cfg.get("roadmap", []),
        "paths": app_cfg.get("paths", {}),
        "artifact_phase_folders": app_cfg.get("artifact_phase_folders", []),
    }

    # Sidebar
    with st.sidebar:
        st.markdown(f"### {top_cfg['title']}")
        st.caption(top_cfg["version"])
        st.markdown("---")
        st.caption("Artifact manifest")
        if st.button("🔄 Rebuild manifest", use_container_width=True):
            manifest, manifest_path = _resolve_manifest(app_cfg, force_rebuild=True)
            st.success(f"Rebuilt → {manifest['summary']['artifacts_total']} artifacts")
        else:
            manifest, manifest_path = _resolve_manifest(app_cfg, force_rebuild=False)
        s = manifest.get("summary", {})
        st.caption(
            f"phases: {s.get('phases_present', 0)}/{s.get('phases_present', 0) + s.get('phases_missing', 0)} | "
            f"csv {s.get('csv_total', 0)} · png {s.get('png_total', 0)} · md {s.get('md_total', 0)}"
        )
        st.markdown("---")
        st.caption("Tabs implemented:")
        st.markdown("- ✅ Overview / Evidence Stack")
        st.markdown("- ✅ Motif · MSS · BSV")
        st.caption("Coming next:")
        for item in top_cfg["roadmap"]:
            st.markdown(f"- 🔜 {item['title']}")

    tab1, tab2 = st.tabs([
        "Overview / Evidence Stack",
        "Motif · MSS · BSV",
    ])

    with tab1:
        overview_tab.render(top_cfg, evidence_cfg, manifest, manifest_path)
    with tab2:
        motif_mss_bsv_tab.render(top_cfg, evidence_cfg, manifest, manifest_path)


if __name__ == "__main__":
    main()
