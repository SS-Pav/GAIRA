from pathlib import Path
import csv
import json
import sys
from collections import Counter

import requests


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.discovery.discovery_utils import (
        deduplicate_records,
        normalize_record,
        simple_relevance_filter,
    )
    from gaira.discovery.source_clients import (
        query_crossref,
        query_europe_pmc,
        query_zenodo,
    )

    queries = [
        "Raman serum",
        "SERS bacteria",
        "Raman extracellular vesicles",
        "Raman metabolite spectra",
        "SERS biofluid",
    ]
    raw_output_path = project_root / "data" / "discovery" / "raw" / "discovery_raw.json"
    processed_output_path = (
        project_root / "data" / "discovery" / "processed" / "discovery_registry.csv"
    )

    raw_records: list[dict[str, object]] = []
    normalized_records: list[dict[str, str]] = []
    source_functions = [
        ("crossref", query_crossref),
        ("europe_pmc", query_europe_pmc),
        ("zenodo", query_zenodo),
    ]

    print("Starting GAIRA discovery run")
    for query in queries:
        print(f"Running query: {query}")
        for source_name, source_function in source_functions:
            print(f"  Fetching from {source_name}")
            try:
                records = source_function(query)
            except requests.RequestException as exc:
                print(f"  Request failed for {source_name}: {exc}")
                records = []

            print(f"  Retrieved {len(records)} records from {source_name}")
            raw_records.append(
                {
                    "query": query,
                    "source": source_name,
                    "records": records,
                }
            )
            for record in records:
                normalized_records.append(normalize_record(record, source_name))

    print(f"Collected {len(normalized_records)} normalized records before filtering")
    filtered_records = [record for record in normalized_records if simple_relevance_filter(record)]
    print(f"Records after filtering: {len(filtered_records)}")

    deduplicated_records = deduplicate_records(filtered_records)
    deduplicated_records.sort(
        key=lambda record: (-int(record.get("data_quality_score", 0)), not bool(record.get("has_files")))
    )

    print(f"Total records collected: {len(normalized_records)}")
    print(f"Records after deduplication: {len(deduplicated_records)}")

    domain_counts = Counter(record.get("domain_tag", "OTHER") for record in deduplicated_records)
    print("Domain distribution:")
    for domain_tag in sorted(domain_counts):
        print(f"  {domain_tag}: {domain_counts[domain_tag]}")

    spectra_type_counts = Counter(record.get("spectra_type", "NO_DATA") for record in deduplicated_records)
    print("Spectra type distribution:")
    for spectra_type in sorted(spectra_type_counts):
        print(f"  {spectra_type}: {spectra_type_counts[spectra_type]}")

    has_files_count = sum(1 for record in deduplicated_records if record.get("has_files"))
    has_files_pct = (100.0 * has_files_count / len(deduplicated_records)) if deduplicated_records else 0.0
    print(f"Records with files: {has_files_count}/{len(deduplicated_records)} ({has_files_pct:.1f}%)")

    usable_for_gaira_count = sum(
        1 for record in deduplicated_records if record.get("usable_for_gaira")
    )
    usable_for_gaira_pct = (
        100.0 * usable_for_gaira_count / len(deduplicated_records)
        if deduplicated_records
        else 0.0
    )
    print(
        f"Usable for GAIRA: {usable_for_gaira_count}/{len(deduplicated_records)} "
        f"({usable_for_gaira_pct:.1f}%)"
    )

    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_output_path.parent.mkdir(parents=True, exist_ok=True)

    with raw_output_path.open("w", encoding="utf-8") as handle:
        json.dump(raw_records, handle, indent=2, ensure_ascii=False)
    print(f"Wrote raw discovery dump to {raw_output_path}")

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
        "data_quality_score",
        "abstract",
        "file_links",
    ]
    with processed_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in deduplicated_records:
            row = {field: record.get(field, "") for field in fieldnames}
            row["file_types"] = "; ".join(record.get("file_types", []))
            writer.writerow(row)
    print(f"Wrote processed discovery registry to {processed_output_path}")


if __name__ == "__main__":
    main()
