from __future__ import annotations

from typing import Any

import streamlit as st


def render_layer_explorer(layer_name: str, layer_data: dict[str, Any], *, key_prefix: str) -> None:
    for bucket_name, datasets in layer_data.items():
        with st.expander(f"{bucket_name} ({len(datasets)})", expanded=True):
            for dataset in datasets:
                st.markdown(f"**{dataset['display_name']}**  ")
                if dataset.get("dataset_kind") == "support_only":
                    st.caption(f"{dataset.get('n_docs', 0):,} support docs • no spectra")
                elif dataset.get("dataset_kind") == "reference_spectra":
                    st.caption(f"{dataset['n_spectra']:,} reference spectra")
                else:
                    st.caption(f"{dataset['n_spectra']:,} spectra")
                if dataset["items"]:
                    preview_rows = []
                    for item in dataset["items"][:16]:
                        preview_rows.append(
                            {
                                "class/type": item.get("class_label", ""),
                                "family": item.get("subclass_label", item.get("experiment_family", item.get("type_bucket", item.get("family_label", "")))),
                                "count": item.get("n_spectra", 0),
                            }
                        )
                    st.dataframe(preview_rows, hide_index=True, width="stretch", key=f"{key_prefix}_{dataset['dataset_id']}")
