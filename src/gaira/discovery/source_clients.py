"""Lightweight source clients for GAIRA discovery."""

from urllib.parse import urlparse
from typing import Any

import requests


CROSSREF_ENDPOINT = "https://api.crossref.org/works"
EUROPE_PMC_ENDPOINT = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
ZENODO_ENDPOINT = "https://zenodo.org/api/records"
REQUEST_TIMEOUT = 30
DEFAULT_ROWS = 25


def _safe_get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _first_text(value: Any) -> str:
    if isinstance(value, list) and value:
        first_item = value[0]
        if isinstance(first_item, str):
            return first_item.strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def _extract_crossref_link(item: dict[str, Any]) -> str:
    resource = item.get("resource") or {}
    primary = resource.get("primary") or {}
    if isinstance(primary, dict) and primary.get("URL"):
        return str(primary["URL"])

    link_entries = item.get("link") or []
    for entry in link_entries:
        if isinstance(entry, dict) and entry.get("URL"):
            return str(entry["URL"])

    if item.get("URL"):
        return str(item["URL"])
    return ""


def _extract_link_extensions(links: list[dict[str, Any]]) -> list[str]:
    extensions: list[str] = []
    for entry in links:
        if not isinstance(entry, dict):
            continue
        url = entry.get("URL")
        if not url:
            continue
        suffix = urlparse(str(url)).path.rsplit(".", 1)
        if len(suffix) == 2 and suffix[1]:
            extension = suffix[1].lower()
            if extension not in extensions:
                extensions.append(extension)
    return extensions


def query_crossref(query: str, rows: int = DEFAULT_ROWS) -> list[dict[str, Any]]:
    params = {"query": query, "rows": rows}
    payload = _safe_get(CROSSREF_ENDPOINT, params=params)
    items = payload.get("message", {}).get("items", [])

    records: list[dict[str, Any]] = []
    for item in items:
        license_entries = item.get("license") or []
        link_entries = item.get("link") or []
        published_parts = (
            item.get("published-print", {}).get("date-parts")
            or item.get("published-online", {}).get("date-parts")
            or item.get("created", {}).get("date-parts")
            or []
        )
        year = None
        if published_parts and published_parts[0]:
            year = published_parts[0][0]

        records.append(
            {
                "title": _first_text(item.get("title")),
                "doi": item.get("DOI") or "",
                "year": year,
                "abstract": item.get("abstract") or "",
                "link": _extract_crossref_link(item),
                "license_urls": [
                    str(entry.get("URL"))
                    for entry in license_entries
                    if isinstance(entry, dict) and entry.get("URL")
                ],
                "is_open_access": bool(license_entries),
                "file_types": _extract_link_extensions(link_entries),
                "raw_source_id": item.get("DOI") or item.get("URL") or "",
            }
        )

    return records


def query_europe_pmc(query: str, rows: int = DEFAULT_ROWS) -> list[dict[str, Any]]:
    params = {
        "query": query,
        "format": "json",
        "pageSize": rows,
        "resultType": "core",
    }
    payload = _safe_get(EUROPE_PMC_ENDPOINT, params=params)
    items = payload.get("resultList", {}).get("result", [])

    records: list[dict[str, Any]] = []
    for item in items:
        records.append(
            {
                "title": item.get("title") or "",
                "pmid": item.get("pmid") or "",
                "doi": item.get("doi") or "",
                "year": item.get("pubYear") or "",
                "abstract": item.get("abstractText") or "",
                "open_access": item.get("isOpenAccess") == "Y",
                "raw_source_id": item.get("id") or item.get("pmid") or item.get("doi") or "",
            }
        )

    return records


def query_zenodo(query: str, rows: int = DEFAULT_ROWS) -> list[dict[str, Any]]:
    params = {"q": query, "size": rows}
    payload = _safe_get(ZENODO_ENDPOINT, params=params)
    items = payload.get("hits", {}).get("hits", [])

    records: list[dict[str, Any]] = []
    for item in items:
        metadata = item.get("metadata") or {}
        files = item.get("files") or []

        file_names: list[str] = []
        download_links: list[str] = []
        for file_entry in files:
            key = file_entry.get("key") or file_entry.get("filename")
            if key:
                file_names.append(str(key))

            links = file_entry.get("links") or {}
            if links.get("self"):
                download_links.append(str(links["self"]))

        records.append(
            {
                "title": metadata.get("title") or "",
                "doi": metadata.get("doi") or item.get("doi") or "",
                "year": str(metadata.get("publication_date") or "")[:4],
                "files": file_names,
                "download_links": download_links,
                "description": metadata.get("description") or "",
                "raw_source_id": item.get("id") or metadata.get("doi") or "",
            }
        )

    return records
