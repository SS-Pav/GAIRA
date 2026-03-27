from __future__ import annotations

import streamlit as st

from app.components.metric_cards import render_metric_cards
from app.components.spectrum_plot import render_pair_spectrum
from app.services.holdout import get_holdout_banner_metrics, get_holdout_pair
from app.services.plots import plot_signed_bars, plot_theme_bars


st.set_page_config(page_title="GAIRAM v1 • HCC Holdout", page_icon="🧬", layout="wide")
st.title("GAIRAM v1 • HCC Holdout")
st.caption("Unseen serum holdout evaluation shown through paired calibrated differential interpretation. This dataset remains holdout-only and is not part of live GAIRAM v1 serum routing.")

metrics = get_holdout_banner_metrics()
render_metric_cards(
    [
        ("Holdout mode", "Safe path only", "Evaluated through isolated holdout artifacts"),
        ("Positive confidence", f"{metrics.get('mean_positive_confidence', 0.0):.3f}", "Calibrated differential view"),
        ("Effect size", f"{metrics.get('mean_abs_theme_effect_size', 0.0):.3f}", "Theme-space separation summary"),
        ("Silhouette", f"{metrics.get('theme_space_silhouette', 0.0):.3f}", "Exploratory theme-space structure"),
    ]
)

mode = st.radio("Pair mode", ["representative pair", "class-mean pair"], horizontal=True)
show_difference = st.toggle("Show difference curve", value=False)
apply_visual_baseline = st.toggle(
    "Apply baseline correction (visual only)",
    value=True,
    help="Display-only preprocessing. The calibrated differential outputs still come from the stored backend holdout artifacts.",
)
pair = get_holdout_pair(mode)

left_col, right_col = st.columns([1.05, 1.0], gap="large")

with left_col:
    st.markdown("### Paired spectrum comparison")
    render_pair_spectrum(
        pair["left"],
        pair["right"],
        show_difference=show_difference,
        apply_baseline_correction=apply_visual_baseline,
    )
    st.caption(f"{pair['left']['label']} vs {pair['right']['label']}")

with right_col:
    st.markdown("### Calibrated differential interpretation")
    st.info(pair["summary_text"])
    shared_col, shifted_col = st.columns(2, gap="large")
    with shared_col:
        st.pyplot(
            plot_theme_bars(
                [value.replace("_associated", "").replace("_", " ").title() for value in pair["shared_background"]["theme_name"].tolist()],
                pair["shared_background"]["shared_background_score"].astype(float).tolist(),
                title="Shared background themes",
                color="#475569",
                x_label="Shared background score",
            ),
            width="stretch",
        )
    with shifted_col:
        shifted_df = pair["h0t_shifted"][["theme_name", "h0t_minus_ctr"]].copy()
        shifted_df = shifted_df.sort_values("h0t_minus_ctr", ascending=False)
        st.pyplot(
            plot_signed_bars(
                [value.replace("_associated", "").replace("_", " ").title() for value in shifted_df["theme_name"].tolist()],
                shifted_df["h0t_minus_ctr"].astype(float).tolist(),
                title="H0T vs CTR shifted themes",
            ),
            width="stretch",
        )

    st.markdown("**Differential evidence summary**")
    for row in pair["differential_evidence"].head(6).to_dict("records"):
        direction = "H0T-leaning" if float(row.get("count_diff", 0.0)) > 0 else "CTR-leaning"
        st.markdown(
            f"<div style='border:1px solid #d7dde6;border-radius:10px;padding:0.6rem 0.8rem;margin-bottom:0.45rem;'>"
            f"<div style='font-weight:600;'>{row.get('label', 'evidence item')}</div>"
            f"<div style='font-size:0.86rem;color:#475569;'>{row.get('evidence_family', 'evidence')} • {direction}</div>"
            f"<div style='font-size:0.82rem;color:#334155;'>count diff {float(row.get('count_diff', 0.0)):.1f} • abs diff {float(row.get('abs_count_diff', 0.0)):.1f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.warning(
        "Exploratory calibrated differential interpretation only. GAIRAM v1 is separating shared serum background from shifted signal on unseen holdout data; it is not making a diagnosis. Matrix, probe, and specificity cautions still apply."
    )

with st.expander("Advanced / internal view", expanded=False):
    st.markdown("**Shared background table**")
    st.dataframe(
        pair["shared_background"][["theme_name", "shared_background_score", "mean_group_shift", "mean_comparison_score"]],
        hide_index=True,
        width="stretch",
    )
    st.markdown("**Shifted themes table**")
    st.dataframe(pair["h0t_shifted"][["theme_name", "h0t_minus_ctr"]], hide_index=True, width="stretch")
    st.markdown("**Differential evidence table**")
    st.dataframe(pair["differential_evidence"], hide_index=True, width="stretch")
    if pair["left"]["summary_row"] is not None and pair["right"]["summary_row"] is not None:
        st.markdown("**Representative sample summaries**")
        st.dataframe(
            [
                {
                    "class_label": pair["left"]["summary_row"]["class_label"],
                    "dominant_themes": pair["left"]["summary_row"]["dominant_themes"],
                    "global_caveats": pair["left"]["summary_row"]["global_caveats"],
                    "what_not_to_claim": pair["left"]["summary_row"]["what_not_to_claim"],
                },
                {
                    "class_label": pair["right"]["summary_row"]["class_label"],
                    "dominant_themes": pair["right"]["summary_row"]["dominant_themes"],
                    "global_caveats": pair["right"]["summary_row"]["global_caveats"],
                    "what_not_to_claim": pair["right"]["summary_row"]["what_not_to_claim"],
                },
            ],
            hide_index=True,
            width="stretch",
        )
