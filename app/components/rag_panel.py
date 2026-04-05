from __future__ import annotations

from typing import Any

import streamlit as st


def render_rag_panel(layer_name: str, layer_payload: dict[str, Any], *, key_prefix: str) -> None:
    st.caption(f"{layer_payload['doc_count']} docs • {layer_payload['chunk_count']} chunks")
    documents = layer_payload["documents"]
    if not documents:
        st.info("No support/context documents available.")
        return
    options = {f"{row['title']} [{row['kind']}]": row for row in documents}
    selected_label = st.selectbox(
        f"{layer_name} documents",
        list(options.keys()),
        key=f"{key_prefix}_doc",
    )
    row = options[selected_label]
    st.markdown(f"**{row['kind']}**  ")
    st.caption(row["source_dataset_id"])
    st.write(row["snippet"] or "No snippet available.")
