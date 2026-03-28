from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


MASTER_X = np.arange(400.0, 1801.0, 1.0, dtype=float)
RNG = np.random.default_rng(7)
DEFAULT_MAX_PER_DATASET = 350
CURRENT_VERSION_MAP = {
    "serum_ag_colloids_grounding": "v1_crop400_1800_interp1_vector",
    "adenine_sers_control": "v1_crop400_1800_interp1_vector",
    "metabolite_sers63_support": "v1_crop500_1800_interp1_vector",
    "amino_acid_raman_grounding": "v1_crop400_1800_interp1_vector",
    "small2023_ev": "v1_crop670_1800_interp1_minmax",
    "shine_ev_sers": "v1_crop450_1800_interp1_minmax",
    "diabetes_plasma_ev_sers": "v1_crop500_1600_interp1_minmax",
    "serum_ag_colloids": "v1_crop400_1800_interp1_minmax",
    "serum_protocol_comparison": "v1_crop400_1800_interp1_minmax",
    "cspp_serum": "v1_crop400_1800_interp1_minmax",
    "ergothioneine_serum": "v1_crop400_1800_interp1_minmax",
    "covid_serum_raman": "v1_crop400_1800_interp1_minmax",
    "cca_hcc_lm_serum_sers": "v1_crop400_1800_interp1_minmax",
}


def parse_json_array(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def align_to_master_grid(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    order = np.argsort(x_values)
    x_sorted = x_values[order]
    y_sorted = y_values[order]
    return np.interp(MASTER_X, x_sorted, y_sorted, left=0.0, right=0.0).astype(np.float32)


def choose_version(version_series: pd.Series, dataset_id: str) -> str:
    versions = sorted(set(version_series.astype(str).tolist()))
    poly3_versions = [version for version in versions if "poly3_vector" in version]
    if poly3_versions:
        return poly3_versions[0]
    current_version = CURRENT_VERSION_MAP.get(dataset_id)
    if current_version and current_version in versions:
        return current_version
    v1_versions = [version for version in versions if version.startswith("v1_")]
    return v1_versions[0] if v1_versions else versions[0]


def sample_rows(df: pd.DataFrame, max_per_dataset: int) -> pd.DataFrame:
    if len(df) <= max_per_dataset:
        return df.copy()
    sample_indices = RNG.choice(df.index.to_numpy(), size=max_per_dataset, replace=False)
    return df.loc[np.sort(sample_indices)].copy()


def infer_sample_type(dataset_id: str, registry_df: pd.DataFrame) -> str:
    subset = registry_df[registry_df["dataset_id"] == dataset_id]
    if subset.empty:
        return "serum"
    sample_type_text = str(subset.iloc[0]["sample_type"]).lower()
    if "extracellular vesicle" in sample_type_text:
        return "ev"
    if "serum" in sample_type_text:
        return "serum"
    return "grounding"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.embedding.runtime import LOCAL_OUTPUT_ROOT, add_common_io_args

    parser = argparse.ArgumentParser(description="Build the contrastive embedding dataset from DuckDB.")
    add_common_io_args(
        parser,
        default_run_name="embedding_v2",
        default_root=LOCAL_OUTPUT_ROOT,
        include_dataset_path=False,
    )
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=DEFAULT_MAX_PER_DATASET,
        help="Maximum number of processed spectra to sample per dataset before adding class summaries.",
    )
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.embedding.metadata import family_label_for_dataset, hard_negative_scope, semantic_group
    from gaira.embedding.runtime import resolve_output_dir
    from gaira.config import get_database_path

    args = parse_args()
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = get_database_path()
    registry_df = pd.read_csv(project_root / "data/registry/datasets.csv")
    with duckdb.connect(str(db_path), read_only=True) as connection:
        biosample_versions = connection.execute(
            """
            SELECT dataset_id, processing_version
            FROM biosample_processed_spectra
            """
        ).fetchdf()
        grounding_versions = connection.execute(
            """
            SELECT dataset_id, processing_version
            FROM grounding_processed_spectra
            """
        ).fetchdf()

        biosample_version_map = {
            dataset_id: choose_version(subset["processing_version"], dataset_id)
            for dataset_id, subset in biosample_versions.groupby("dataset_id")
        }
        grounding_version_map = {
            dataset_id: choose_version(subset["processing_version"], dataset_id)
            for dataset_id, subset in grounding_versions.groupby("dataset_id")
        }
        rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []

        for dataset_id, version in biosample_version_map.items():
            sample_type = infer_sample_type(dataset_id, registry_df)
            subset = connection.execute(
                """
                SELECT
                  p.processed_id AS sample_key,
                  p.dataset_id,
                  p.processing_version,
                  p.wavenumbers_json,
                  p.intensity_json,
                  m.class_label,
                  m.subclass_label
                FROM biosample_processed_spectra p
                JOIN biosample_metadata m USING (biosample_id, dataset_id)
                WHERE p.dataset_id = ?
                  AND p.processing_version = ?
                ORDER BY p.biosample_id
                """,
                [dataset_id, version],
            ).fetchdf()
            selected = sample_rows(subset, args.max_per_dataset)
            for row in selected.to_dict(orient="records"):
                rows.append(
                    {
                        "sample_key": row["sample_key"],
                        "dataset_id": dataset_id,
                        "sample_type": sample_type,
                        "label_optional": row["class_label"],
                        "record_kind": "processed_spectrum",
                        "processing_version": row["processing_version"],
                        "subclass_label": row["subclass_label"],
                        "wavenumbers_json": row["wavenumbers_json"],
                        "intensity_json": row["intensity_json"],
                    }
                )
            summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "sample_type": sample_type,
                    "record_kind": "processed_spectrum",
                    "selected_count": int(len(selected)),
                    "available_count": int(len(subset)),
                    "processing_version": version,
                }
            )

            summary_subset = connection.execute(
                """
                SELECT
                  summary_id AS sample_key,
                  dataset_id,
                  processing_version,
                  mean_wavenumbers_json AS wavenumbers_json,
                  mean_intensity_json AS intensity_json,
                  class_label,
                  subclass_label
                FROM biosample_class_summary
                WHERE dataset_id = ?
                  AND processing_version = ?
                """,
                [dataset_id, version],
            ).fetchdf()
            for row in summary_subset.to_dict(orient="records"):
                rows.append(
                    {
                        "sample_key": row["sample_key"],
                        "dataset_id": row["dataset_id"],
                        "sample_type": sample_type,
                        "label_optional": row["class_label"],
                        "record_kind": "class_summary",
                        "processing_version": row["processing_version"],
                        "subclass_label": row["subclass_label"],
                        "wavenumbers_json": row["wavenumbers_json"],
                        "intensity_json": row["intensity_json"],
                    }
                )
            summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "sample_type": sample_type,
                    "record_kind": "class_summary",
                    "selected_count": int(len(summary_subset)),
                    "available_count": int(len(summary_subset)),
                    "processing_version": version,
                }
            )

        for dataset_id, version in grounding_version_map.items():
            subset = connection.execute(
                """
                SELECT
                  processed_id AS sample_key,
                  dataset_id,
                  processing_version,
                  wavenumbers_json,
                  intensity_json,
                  class_label,
                  experiment_family AS subclass_label
                FROM grounding_processed_spectra
                WHERE dataset_id = ?
                  AND processing_version = ?
                ORDER BY grounding_id
                """,
                [dataset_id, version],
            ).fetchdf()
            selected = sample_rows(subset, args.max_per_dataset)
            for row in selected.to_dict(orient="records"):
                rows.append(
                    {
                        "sample_key": row["sample_key"],
                        "dataset_id": dataset_id,
                        "sample_type": "grounding",
                        "label_optional": row["class_label"],
                        "record_kind": "processed_spectrum",
                        "processing_version": row["processing_version"],
                        "subclass_label": row["subclass_label"],
                        "wavenumbers_json": row["wavenumbers_json"],
                        "intensity_json": row["intensity_json"],
                    }
                )
            summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "sample_type": "grounding",
                    "record_kind": "processed_spectrum",
                    "selected_count": int(len(selected)),
                    "available_count": int(len(subset)),
                    "processing_version": version,
                }
            )

            summary_subset = connection.execute(
                """
                SELECT
                  summary_id AS sample_key,
                  dataset_id,
                  processing_version,
                  mean_wavenumbers_json AS wavenumbers_json,
                  mean_intensity_json AS intensity_json,
                  class_label,
                  experiment_family AS subclass_label
                FROM grounding_class_summary
                WHERE dataset_id = ?
                  AND processing_version = ?
                """,
                [dataset_id, version],
            ).fetchdf()
            for row in summary_subset.to_dict(orient="records"):
                rows.append(
                    {
                        "sample_key": row["sample_key"],
                        "dataset_id": row["dataset_id"],
                        "sample_type": "grounding",
                        "label_optional": row["class_label"],
                        "record_kind": "class_summary",
                        "processing_version": row["processing_version"],
                        "subclass_label": row["subclass_label"],
                        "wavenumbers_json": row["wavenumbers_json"],
                        "intensity_json": row["intensity_json"],
                    }
                )
            summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "sample_type": "grounding",
                    "record_kind": "class_summary",
                    "selected_count": int(len(summary_subset)),
                    "available_count": int(len(summary_subset)),
                    "processing_version": version,
                }
            )

    metadata_df = pd.DataFrame(rows)
    X = np.vstack(
        [
            align_to_master_grid(parse_json_array(row["wavenumbers_json"]), parse_json_array(row["intensity_json"]))
            for row in metadata_df.to_dict(orient="records")
        ]
    )

    np.savez(
        output_dir / "embedding_dataset.npz",
        X=X.astype(np.float32),
        dataset_ids=metadata_df["dataset_id"].astype(str).to_numpy(),
        sample_types=metadata_df["sample_type"].astype(str).to_numpy(),
        labels_optional=metadata_df["label_optional"].fillna("").astype(str).to_numpy(),
        family_labels=np.asarray(
            [
                family_label_for_dataset(dataset_id, sample_type)
                for dataset_id, sample_type in zip(
                    metadata_df["dataset_id"].astype(str),
                    metadata_df["sample_type"].astype(str),
                    strict=False,
                )
            ],
            dtype=object,
        ),
        semantic_groups=np.asarray(
            [
                semantic_group(dataset_id, sample_type, label_optional)
                for dataset_id, sample_type, label_optional in zip(
                    metadata_df["dataset_id"].astype(str),
                    metadata_df["sample_type"].astype(str),
                    metadata_df["label_optional"].fillna("").astype(str),
                    strict=False,
                )
            ],
            dtype=object,
        ),
        hard_negative_scopes=np.asarray(
            [
                hard_negative_scope(dataset_id, sample_type, label_optional)
                for dataset_id, sample_type, label_optional in zip(
                    metadata_df["dataset_id"].astype(str),
                    metadata_df["sample_type"].astype(str),
                    metadata_df["label_optional"].fillna("").astype(str),
                    strict=False,
                )
            ],
            dtype=object,
        ),
        record_kinds=metadata_df["record_kind"].astype(str).to_numpy(),
        processing_versions=metadata_df["processing_version"].astype(str).to_numpy(),
        sample_keys=metadata_df["sample_key"].astype(str).to_numpy(),
        subclasses=metadata_df["subclass_label"].fillna("").astype(str).to_numpy(),
        master_x=MASTER_X.astype(np.float32),
    )

    dataset_summary = pd.DataFrame(summary_rows).drop_duplicates().reset_index(drop=True)
    dataset_summary.to_csv(output_dir / "dataset_summary.csv", index=False)

    print(f"Saved embedding dataset: {output_dir / 'embedding_dataset.npz'}")
    print(f"Samples: {X.shape[0]}")
    print(f"Input length: {X.shape[1]}")


if __name__ == "__main__":
    main()
