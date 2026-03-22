from pathlib import Path
import sys

import duckdb
import pandas as pd
import streamlit as st


st.set_page_config(page_title="GAIRA Dataset Dashboard", layout="wide")


def load_datasets() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.config import get_database_path

    db_path = get_database_path()

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with duckdb.connect(str(db_path), read_only=True) as connection:
        return connection.execute("SELECT * FROM datasets").fetchdf()


st.title("GAIRA — GenAI Raman Analysis")
st.caption("Phase 1: Raman Dataset Registry")

try:
    datasets_df = load_datasets()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except duckdb.Error:
    st.error(
        "The datasets table is not available yet. Run `python scripts/load_registry.py` first."
    )
    st.stop()

total_datasets = len(datasets_df)
unique_sample_types = datasets_df["sample_type"].nunique()
unique_modalities = datasets_df["modality"].nunique()

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Total datasets", total_datasets)
metric_2.metric("Unique sample types", unique_sample_types)
metric_3.metric("Unique modalities", unique_modalities)

st.subheader("Dataset Registry")
st.dataframe(datasets_df, width="stretch")

sample_type_counts = (
    datasets_df["sample_type"].value_counts().rename_axis("sample_type").reset_index(name="count")
)
modality_counts = (
    datasets_df["modality"].value_counts().rename_axis("modality").reset_index(name="count")
)

chart_1, chart_2 = st.columns(2)

with chart_1:
    st.subheader("Datasets by Sample Type")
    st.bar_chart(sample_type_counts.set_index("sample_type"))

with chart_2:
    st.subheader("Datasets by Modality")
    st.bar_chart(modality_counts.set_index("modality"))
