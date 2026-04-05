from pathlib import Path
import csv


SPECTRA_TYPE_ORDER = {
    "RAW_SPECTRA": 0,
    "PEAK_TABLE": 1,
}


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def _to_int(value: str) -> int:
    try:
        return int(str(value).strip())
    except ValueError:
        return 0


def _priority_for_ingestion(record: dict[str, str]) -> bool:
    return (
        record.get("spectra_type") == "RAW_SPECTRA"
        or _to_int(record.get("data_quality_score", "0")) >= 2
    )


def load_usable_records(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        records = [row for row in reader if _to_bool(row.get("usable_for_gaira", ""))]

    for record in records:
        record["priority_for_ingestion"] = str(_priority_for_ingestion(record))

    records.sort(
        key=lambda record: (
            SPECTRA_TYPE_ORDER.get(record.get("spectra_type", ""), 99),
            -_to_int(record.get("data_quality_score", "0")),
        )
    )
    return records


def print_records(records: list[dict[str, str]]) -> None:
    if not records:
        print("No usable GAIRA discovery records found.")
        return

    for index, record in enumerate(records, start=1):
        print("----------------------------------")
        print(f"INDEX: {index}")
        print()
        print("TITLE:")
        print(record.get("title", ""))
        print()
        print("SOURCE:")
        print(record.get("source", ""))
        print()
        print("YEAR:")
        print(record.get("year", ""))
        print()
        print("DOMAIN:")
        print(record.get("domain_tag", ""))
        print()
        print("SPECTRA TYPE:")
        print(record.get("spectra_type", ""))
        print()
        print("DATA QUALITY SCORE:")
        print(record.get("data_quality_score", ""))
        print()
        print("HAS FILES:")
        print(record.get("has_files", ""))
        print()
        print("FILE TYPES:")
        print(record.get("file_types", ""))
        print()
        print("LINKS:")
        print(record.get("file_links", ""))
        print()
        print("PRIORITY FOR INGESTION:")
        print(record.get("priority_for_ingestion", ""))
        print()
    print("----------------------------------")


def save_shortlist(records: list[dict[str, str]], output_path: Path) -> None:
    fieldnames = [
        "title",
        "doi",
        "source",
        "year",
        "domain_tag",
        "has_files",
        "file_types",
        "is_open_access",
        "spectra_type",
        "usable_for_gaira",
        "priority_for_ingestion",
        "data_quality_score",
        "abstract",
        "file_links",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "discovery" / "processed" / "discovery_registry.csv"
    output_path = project_root / "data" / "discovery" / "processed" / "discovery_top_usable.csv"

    if not input_path.exists():
        print(f"Discovery registry not found: {input_path}")
        return

    records = load_usable_records(input_path)
    print_records(records)
    save_shortlist(records, output_path)
    print(f"Saved {len(records)} shortlisted records to {output_path}")


if __name__ == "__main__":
    main()
