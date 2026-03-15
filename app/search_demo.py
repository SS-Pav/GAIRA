import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gaira.search import PeakSearchEngine


def list_query_files() -> list[Path]:
    """Collect query CSV files from the test and evaluation folders."""
    search_dirs = [
        PROJECT_ROOT / "data" / "processed" / "test_queries",
        PROJECT_ROOT / "data" / "processed" / "eval_queries",
    ]
    csv_files: list[Path] = []
    for folder in search_dirs:
        if folder.exists():
            csv_files.extend(sorted(folder.glob("*.csv")))
    return csv_files


def build_line_chart(data_df: pd.DataFrame, title: str, color_field: str = "series") -> dict:
    """Create a simple Vega-Lite line chart without extra plotting libraries."""
    return {
        "title": title,
        "mark": {"type": "line"},
        "encoding": {
            "x": {"field": "wavenumber", "type": "quantitative"},
            "y": {"field": "intensity", "type": "quantitative"},
            "color": {"field": color_field, "type": "nominal"},
        },
        "data": {"values": data_df.to_dict(orient="records")},
    }


def build_overlay_peak_chart(spectrum_df: pd.DataFrame, peaks_df: pd.DataFrame, title: str) -> dict:
    """Overlay detected peaks on the query spectrum."""
    return {
        "title": title,
        "layer": [
            {
                "mark": {"type": "line"},
                "data": {"values": spectrum_df.assign(series="Query").to_dict(orient="records")},
                "encoding": {
                    "x": {"field": "wavenumber", "type": "quantitative"},
                    "y": {"field": "intensity", "type": "quantitative"},
                    "color": {"field": "series", "type": "nominal"},
                },
            },
            {
                "mark": {"type": "point", "filled": True, "size": 55},
                "data": {"values": peaks_df.to_dict(orient="records")},
                "encoding": {
                    "x": {"field": "peak_cm", "type": "quantitative"},
                    "y": {"field": "peak_intensity", "type": "quantitative"},
                    "color": {"value": "#d62728"},
                    "tooltip": [
                        {"field": "peak_rank", "type": "ordinal"},
                        {"field": "peak_cm", "type": "quantitative"},
                        {"field": "peak_intensity", "type": "quantitative"},
                    ],
                },
            },
        ],
    }


def build_matched_peaks_table(
    engine: PeakSearchEngine,
    query_peaks_df: pd.DataFrame,
    ref_id: str,
) -> pd.DataFrame:
    """Match query peaks to the closest reference peaks within the active tolerance."""
    reference_peaks_df = engine._get_reference_peaks()
    candidate_peaks_df = reference_peaks_df[reference_peaks_df["ref_id"] == ref_id].copy()

    matched_rows: list[dict] = []
    for row in query_peaks_df.itertuples(index=False):
        if candidate_peaks_df.empty:
            continue

        candidate_peaks_df["delta_cm"] = (candidate_peaks_df["peak_cm"] - float(row.peak_cm)).abs()
        within_tolerance_df = candidate_peaks_df[
            candidate_peaks_df["delta_cm"] <= engine.peak_tolerance_cm
        ].copy()
        if within_tolerance_df.empty:
            continue

        best_match = within_tolerance_df.sort_values("delta_cm").iloc[0]
        matched_rows.append(
            {
                "query_peak_cm": float(row.peak_cm),
                "closest_reference_peak_cm": float(best_match["peak_cm"]),
                "delta_cm": float(best_match["delta_cm"]),
                "query_peak_intensity": float(row.peak_intensity),
                "reference_rel_intensity": float(best_match["rel_intensity"]),
            }
        )

    return pd.DataFrame(matched_rows)


def load_uploaded_query(uploaded_file) -> tuple[Path, str]:
    """Persist an uploaded query so the existing search pipeline can read it."""
    upload_dir = PROJECT_ROOT / "data" / "processed" / "eval_queries"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / uploaded_file.name
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path, uploaded_file.name


def main() -> None:
    st.set_page_config(page_title="GAIRA Search Demo", layout="wide")
    st.title("GAIRA Search Demo")
    st.caption("Stable baseline Raman peak search with full-spectrum reranking.")

    db_path = PROJECT_ROOT / "data" / "gaira.duckdb"
    if not db_path.exists():
        st.error("DuckDB database not found at data/gaira.duckdb.")
        return

    engine = PeakSearchEngine(db_path=db_path)
    available_queries = list_query_files()

    st.sidebar.header("Query Selection")
    source_mode = st.sidebar.radio(
        "Choose query source",
        ["Local query file", "Upload CSV"],
    )

    selected_query_path: Path | None = None
    query_label = ""

    if source_mode == "Local query file":
        if not available_queries:
            st.error("No local query CSV files were found in test_queries or eval_queries.")
            return

        query_options = {
            str(path.relative_to(PROJECT_ROOT)): path for path in available_queries
        }
        query_label = st.sidebar.selectbox("Local query CSV", list(query_options.keys()))
        selected_query_path = query_options[query_label]
    else:
        uploaded_file = st.sidebar.file_uploader("Upload query CSV", type=["csv"])
        if uploaded_file is None:
            st.info("Upload a CSV file to run the search demo.")
            return
        selected_query_path, query_label = load_uploaded_query(uploaded_file)

    if selected_query_path is None:
        st.info("Select a query file to continue.")
        return

    try:
        results = engine.run(selected_query_path)
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        return

    query_df = results["query_spectrum_df"]
    query_peaks_df = results["query_peaks_df"]

    st.subheader("Query Metadata")
    metadata_cols = st.columns(4)
    metadata_cols[0].metric("Query file", query_label)
    metadata_cols[1].metric(
        "Spectral range",
        f"{query_df['wavenumber'].min():.0f}-{query_df['wavenumber'].max():.0f}",
    )
    metadata_cols[2].metric("Points", int(len(query_df)))
    metadata_cols[3].metric("Detected peaks", int(len(query_peaks_df)))

    st.subheader("Query Spectrum")
    st.vega_lite_chart(
        build_overlay_peak_chart(query_df, query_peaks_df, "Query spectrum with detected peaks"),
        use_container_width=True,
    )

    st.subheader("Search Results")
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("**Top reranked ref-level candidates**")
        st.dataframe(results["candidate_df_reranked"].head(10), width="stretch")
    with right_col:
        st.markdown("**Top reranked component-level candidates**")
        st.dataframe(results["candidate_df_component_reranked"].head(10), width="stretch")

    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.markdown("**Top biochemical classes**")
        st.dataframe(results["class_df_reranked"].head(10), width="stretch")
    with lower_right:
        st.markdown("**Interpretation**")
        st.write(results["interpretation_text"])

    reranked_df = results["candidate_df_reranked"].copy()
    if reranked_df.empty:
        st.warning("No reranked candidates were returned for this query.")
        return

    st.subheader("Candidate Inspection")
    candidate_labels = [
        f"{row.ref_id} | {row.component}" for row in reranked_df.head(10).itertuples(index=False)
    ]
    selected_label = st.selectbox("Choose a top reranked reference", candidate_labels)
    selected_ref_id = selected_label.split(" | ", maxsplit=1)[0]

    reference_df = engine.load_reference_spectrum(selected_ref_id)
    interpolated_reference_df = engine.interpolate_reference_to_query_grid(
        reference_df,
        query_df["wavenumber"],
    )
    overlay_df = pd.concat(
        [
            query_df.assign(series="Query"),
            interpolated_reference_df.assign(series=selected_ref_id),
        ],
        ignore_index=True,
    )

    st.vega_lite_chart(
        build_line_chart(overlay_df, "Query vs selected reference overlay"),
        use_container_width=True,
    )

    matched_peaks_df = build_matched_peaks_table(engine, query_peaks_df, selected_ref_id)
    st.markdown("**Matched peaks for selected candidate**")
    if matched_peaks_df.empty:
        st.info("No query peaks matched this reference within the current tolerance.")
    else:
        st.dataframe(matched_peaks_df, width="stretch")


if __name__ == "__main__":
    main()
