import csv
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = (
        project_root / "data" / "processed" / "eval_queries" / "eval_manifest.csv"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "query_id": "collagen_example",
            "query_path": "data/processed/test_queries/collagen_example_query.csv",
            "expected_component": "collagen",
            "expected_class": "Proteins",
            "notes": "Starter evaluation query copied from RamanBioLib example data.",
        }
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "query_path",
                "expected_component",
                "expected_class",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created starter evaluation manifest at {manifest_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()
