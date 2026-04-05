from __future__ import annotations

import streamlit as st

from app.components.metric_cards import render_metric_cards
from app.services.catalog import get_global_kpis


def main() -> None:
    st.set_page_config(page_title="GAIRAM v1", page_icon="🧪", layout="wide")
    st.title("GAIRAM v1")
    st.caption("Deterministic Raman / SERS interpretation demo with structured evidence, routing-aware reasoning, and calibrated caution.")

    kpis = get_global_kpis()
    render_metric_cards(
        [
            ("Live datasets", f"{kpis['total_live_datasets']}", "Grounding, EV, and serum layers"),
            ("Live spectra", f"{kpis['total_live_spectra']:,}", "Current SSD_Rad-backed archive"),
            ("Support / context docs", f"{kpis['total_support_context_docs']}", "Retrievable literature and context notes"),
            ("Holdout status", "Holdout preserved", kpis["holdout_status"]),
        ]
    )

    st.markdown(
        """
        ## Demo pages

        - **Knowledge Map**: what GAIRAM v1 contains across grounding, EV, serum, and the support/context stack.
        - **Routing Engine**: how sample metadata drives family-aware context/support routing before interpretation.
        - **Inference Playground**: side-by-side general vs query-aware inference on representative spectra.
        - **HCC Holdout**: paired calibrated differential interpretation on unseen serum holdout data.
        """
    )


if __name__ == "__main__":
    main()
