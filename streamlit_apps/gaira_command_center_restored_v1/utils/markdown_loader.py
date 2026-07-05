"""Safe markdown loading helpers."""
from __future__ import annotations

from pathlib import Path
import streamlit as st


def load_markdown_safe(path: str | Path,
                       show_warning: bool = False) -> str | None:
    p = Path(path) if path is not None else None
    if p is None or not p.exists():
        if show_warning:
            st.info(f"🗂️ Markdown not found: `{path}`")
        return None
    try:
        return p.read_text()
    except Exception as e:
        st.warning(f"Could not read markdown `{p.name}`: {e}")
        return None


def display_markdown_safe(path: str | Path,
                          fallback_msg: str | None = None,
                          expander_label: str | None = None) -> bool:
    text = load_markdown_safe(path, show_warning=False)
    if text is None:
        st.info(f"🗂️ {fallback_msg or f'Markdown not found: `{path}`'}")
        return False
    if expander_label is not None:
        with st.expander(expander_label, expanded=False):
            st.markdown(text)
    else:
        st.markdown(text)
    return True
