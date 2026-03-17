from pathlib import Path

import duckdb


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "gaira.duckdb"

    with duckdb.connect(str(db_path), read_only=True) as connection:
        dataset_count = connection.execute(
            "SELECT COUNT(*) FROM dataset_domain_context"
        ).fetchone()[0]
        subclass_count = connection.execute(
            "SELECT COUNT(*) FROM subclass_domain_context"
        ).fetchone()[0]

        dataset_preview = connection.execute(
            """
            SELECT
                dataset_id,
                dataset_family,
                biosample_type,
                measurement_mode,
                default_substrate_type,
                default_substrate_material,
                substrate_vendor,
                instrument_context,
                default_preprocessing_family
            FROM dataset_domain_context
            ORDER BY dataset_id
            """
        ).fetchdf()

        subclass_preview = connection.execute(
            """
            SELECT
                dataset_id,
                subclass_label,
                measurement_mode,
                substrate_type,
                substrate_material,
                substrate_vendor,
                substrate_batch_id,
                probe_family,
                spectral_axis_family,
                cross_domain_intensity_comparable,
                preprocessing_family
            FROM subclass_domain_context
            ORDER BY dataset_id, subclass_label
            """
        ).fetchdf()

    print(f"dataset_domain_context row count: {dataset_count}")
    print(f"subclass_domain_context row count: {subclass_count}")
    print("\nDataset-level context preview:")
    print(dataset_preview.to_string(index=False))
    print("\nSubclass/domain-level context preview:")
    print(subclass_preview.to_string(index=False))


if __name__ == "__main__":
    main()
