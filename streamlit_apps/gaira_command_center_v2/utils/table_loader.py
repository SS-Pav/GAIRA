"""Safe CSV loading helpers."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st


def load_csv_safe(path: str | Path,
                  show_warning: bool = False) -> pd.DataFrame | None:
    """Return a DataFrame, or None if missing/unreadable.

    show_warning=True surfaces a Streamlit info card when the file is missing.
    """
    p = Path(path) if path is not None else None
    if p is None or not p.exists():
        if show_warning:
            st.info(f"🗂️ Table not found: `{path}`")
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.warning(f"Could not read CSV `{p.name}`: {e}")
        return None


def display_csv_safe(path: str | Path, caption: str | None = None,
                     max_rows: int | None = None,
                     show_warning: bool = True) -> bool:
    """Render a CSV as a Streamlit table. Returns True if rendered."""
    df = load_csv_safe(path, show_warning=show_warning)
    if df is None:
        return False
    if caption:
        st.caption(caption)
    if max_rows is not None and len(df) > max_rows:
        st.dataframe(df.head(max_rows), use_container_width=True, hide_index=True)
        st.caption(f"showing {max_rows}/{len(df)} rows")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
    return True
