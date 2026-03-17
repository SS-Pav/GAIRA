from pathlib import Path

import duckdb
import pandas as pd


DATASET_CONTEXT_COLUMNS = [
    "dataset_id",
    "dataset_family",
    "context_level",
    "biosample_type",
    "measurement_mode",
    "default_substrate_type",
    "default_substrate_material",
    "instrument_context",
    "default_preprocessing_family",
    "notes",
]

SUBCLASS_CONTEXT_COLUMNS = [
    "dataset_id",
    "subclass_label",
    "context_level",
    "biosample_type",
    "measurement_mode",
    "substrate_type",
    "substrate_material",
    "substrate_batch_id",
    "probe_family",
    "spectral_axis_family",
    "cross_domain_intensity_comparable",
    "preprocessing_family",
    "notes",
]


def load_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing context seed file: {path}")

    df = pd.read_csv(path)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    return df[required_columns].copy()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "gaira.duckdb"
    context_dir = project_root / "data" / "raw" / "context"
    dataset_context_path = context_dir / "dataset_domain_context_v1.csv"
    subclass_context_path = context_dir / "subclass_domain_context_v1.csv"

    dataset_df = load_csv(dataset_context_path, DATASET_CONTEXT_COLUMNS)
    subclass_df = load_csv(subclass_context_path, SUBCLASS_CONTEXT_COLUMNS)

    dataset_ids = sorted(set(dataset_df["dataset_id"].tolist()) | set(subclass_df["dataset_id"].tolist()))

    with duckdb.connect(str(db_path)) as connection:
        for dataset_id in dataset_ids:
            connection.execute(
                "DELETE FROM dataset_domain_context WHERE dataset_id = ?",
                [dataset_id],
            )
            connection.execute(
                "DELETE FROM subclass_domain_context WHERE dataset_id = ?",
                [dataset_id],
            )

        connection.register("dataset_context_df", dataset_df)
        connection.execute("INSERT INTO dataset_domain_context SELECT * FROM dataset_context_df")
        connection.unregister("dataset_context_df")

        connection.register("subclass_context_df", subclass_df)
        connection.execute("INSERT INTO subclass_domain_context SELECT * FROM subclass_context_df")
        connection.unregister("subclass_context_df")

    print(f"Inserted dataset_domain_context rows: {len(dataset_df)}")
    print(f"Inserted subclass_domain_context rows: {len(subclass_df)}")
    print("Datasets covered:")
    for dataset_id in dataset_ids:
        print(f"  {dataset_id}")


if __name__ == "__main__":
    main()
