from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse
import sys

import requests


ZENODO_QUERIES = [
    "raman serum spectra",
    "sers bacteria spectra",
    "raman extracellular vesicle spectra",
    "raman metabolite spectra",
]
PMC_QUERIES = [
    "raman serum supplementary",
    "sers exosome supplementary",
    "raman extracellular vesicles supplementary",
]
FIGSHARE_QUERIES = [
    "raman spectra dataset",
    "sers spectra dataset",
    "raman metabolite spectra",
]


def _infer_source_type_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "zenodo.org" in host:
        return "zenodo"
    if "figshare.com" in host:
        return "figshare"
    if "europepmc.org" in host or "pmc" in host:
        return "pmc"
    if "github.com" in host:
        return "github"
    return host or "html"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.targeted.acquisition import (
        classify_downloaded_file,
        discover_page_file_links,
        download_candidate_file,
        filter_target_urls,
        generate_search_queries,
        initialize_content_hashes,
        search_duckduckgo_html,
        search_page_has_anomaly,
        write_registry,
    )
    from gaira.targeted.source_harvesters import (
        harvest_figshare,
        harvest_pmc,
        harvest_zenodo,
    )

    raw_html_dir = project_root / "data" / "targeted" / "raw_html"
    files_dir = project_root / "data" / "targeted" / "files"
    processed_dir = project_root / "data" / "targeted" / "processed"
    registry_path = processed_dir / "targeted_registry.csv"

    search_results: list[dict[str, object]] = []
    registry_rows: list[dict[str, object]] = []
    source_candidate_counts: Counter[str] = Counter()
    source_download_counts: Counter[str] = Counter()
    source_raw_counts: Counter[str] = Counter()
    seen_file_urls: set[str] = set()
    content_hashes = initialize_content_hashes(files_dir)

    write_registry(registry_rows, registry_path)

    print("Starting GAIRA targeted acquisition run")

    duckduckgo_blocked_queries = 0
    for query in generate_search_queries():
        print(f"Searching: {query}")
        try:
            query_results = search_duckduckgo_html(query, raw_html_dir)
        except requests.RequestException as exc:
            print(f"  Search failed: {exc}")
            continue

        print(f"  Retrieved {len(query_results)} search results")
        if search_page_has_anomaly(query, raw_html_dir):
            duckduckgo_blocked_queries += 1
            print("  DuckDuckGo returned an anti-bot challenge page")
        search_results.extend(query_results)

    candidate_pages = filter_target_urls(search_results)
    print(f"Total URLs scraped: {len(search_results)}")
    print(f"Total candidate pages: {len(candidate_pages)}")
    if duckduckgo_blocked_queries:
        print(f"DuckDuckGo challenge pages encountered: {duckduckgo_blocked_queries}")

    html_candidates: list[dict[str, object]] = []
    for page in candidate_pages:
        page_url = str(page.get("url", ""))
        source_type = _infer_source_type_from_url(page_url)
        print(f"Scanning page: {page_url}")
        try:
            file_links, priority_boost = discover_page_file_links(page, raw_html_dir)
        except requests.RequestException as exc:
            print(f"  Page fetch failed: {exc}")
            continue

        print(f"  Found {len(file_links)} candidate file links")
        for file_link in file_links:
            if file_link in seen_file_urls:
                continue
            seen_file_urls.add(file_link)
            html_candidates.append(
                {
                    "file_url": file_link,
                    "source_url": page_url,
                    "source_type": source_type,
                    "extraction_method": "html",
                    "priority_boost": priority_boost,
                }
            )

    harvester_tasks = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for query in ZENODO_QUERIES:
            harvester_tasks.append(("zenodo", query, executor.submit(harvest_zenodo, query)))
        for query in FIGSHARE_QUERIES:
            harvester_tasks.append(("figshare", query, executor.submit(harvest_figshare, query)))
        for query in PMC_QUERIES:
            harvester_tasks.append(("pmc", query, executor.submit(harvest_pmc, query)))

        api_candidates: list[dict[str, object]] = []
        for source_name, query, future in harvester_tasks:
            try:
                harvested = future.result()
            except requests.RequestException as exc:
                print(f"{source_name} harvester failed for '{query}': {exc}")
                continue
            except Exception as exc:
                print(f"{source_name} harvester failed for '{query}': {exc}")
                continue

            print(f"{source_name} harvester query '{query}' yielded {len(harvested)} file candidates")
            for candidate in harvested:
                file_url = str(candidate.get("file_url", ""))
                if not file_url or file_url in seen_file_urls:
                    continue
                seen_file_urls.add(file_url)
                api_candidates.append(candidate)

    all_candidates = html_candidates + api_candidates
    all_candidates.sort(key=lambda candidate: (not bool(candidate.get("priority_boost")), str(candidate.get("file_url", ""))))

    for candidate in all_candidates:
        file_url = str(candidate.get("file_url", ""))
        source_url = str(candidate.get("source_url", ""))
        source_type = str(candidate.get("source_type", "html"))
        extraction_method = str(candidate.get("extraction_method", "html"))
        source_candidate_counts[source_type] += 1

        try:
            downloaded = download_candidate_file(
                file_url=file_url,
                source_url=source_url,
                files_dir=files_dir,
                priority_boost=bool(candidate.get("priority_boost")),
                source_type=source_type,
                extraction_method=extraction_method,
                content_hashes=content_hashes,
            )
        except requests.RequestException as exc:
            print(f"File download failed ({source_type}): {exc}")
            continue

        if downloaded is None:
            continue

        spectra_type = classify_downloaded_file(
            downloaded["local_path"],
            str(downloaded["file_type"]),
        )
        source_download_counts[source_type] += 1
        if spectra_type == "RAW_SPECTRA":
            source_raw_counts[source_type] += 1
        registry_rows.append(
            {
                "file_name": downloaded["file_name"],
                "source_url": downloaded["source_url"],
                "file_type": downloaded["file_type"],
                "spectra_type": spectra_type,
                "size_kb": downloaded["size_kb"],
                "source_type": downloaded["source_type"],
                "extraction_method": downloaded["extraction_method"],
            }
        )
        write_registry(registry_rows, registry_path)

    write_registry(registry_rows, registry_path)

    total_downloads = len(registry_rows)
    raw_spectra_count = sum(1 for row in registry_rows if row["spectra_type"] == "RAW_SPECTRA")
    table_count = sum(1 for row in registry_rows if row["spectra_type"] == "TABLE")

    print(f"Total files downloaded: {total_downloads}")
    print(f"RAW_SPECTRA count: {raw_spectra_count}")
    print(f"TABLE count: {table_count}")
    print("Files per source:")
    for source_type in sorted(source_candidate_counts):
        print(f"  {source_type}: {source_download_counts[source_type]}")
    print("RAW_SPECTRA per source:")
    for source_type in sorted(source_candidate_counts):
        print(f"  {source_type}: {source_raw_counts[source_type]}")
    print("Success rate per source:")
    for source_type in sorted(source_candidate_counts):
        candidate_count = source_candidate_counts[source_type]
        success_rate = (
            100.0 * source_download_counts[source_type] / candidate_count
            if candidate_count
            else 0.0
        )
        print(
            f"  {source_type}: {source_download_counts[source_type]}/{candidate_count} "
            f"({success_rate:.1f}%)"
        )
    print(f"Wrote targeted registry to {registry_path}")


if __name__ == "__main__":
    main()
