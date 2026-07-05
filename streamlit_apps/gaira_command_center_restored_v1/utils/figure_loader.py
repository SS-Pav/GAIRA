"""Safe image loading helpers."""
from __future__ import annotations

from pathlib import Path
import streamlit as st


def load_image_safe(path: str | Path, caption: str | None = None,
                    use_container_width: bool = True,
                    fallback_msg: str | None = None) -> bool:
    """Render an image if it exists, else show a soft warning card.

    Returns True if rendered, False otherwise.
    """
    p = Path(path) if path is not None else None
    if p is None or not p.exists():
        msg = fallback_msg or f"Artifact not found: `{path}`"
        st.info(f"🗂️ {msg}")
        return False
    try:
        st.image(str(p), caption=caption, use_container_width=use_container_width)
        return True
    except Exception as e:
        st.warning(f"Could not render image `{p.name}`: {e}")
        return False
