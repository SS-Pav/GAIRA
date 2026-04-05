"""Source-specific harvesters for targeted acquisition."""

from __future__ import annotations

from urllib.parse import urljoin

from bs4 import BeautifulSoup

from gaira.targeted.acquisition import (
    prioritize_supplementary_url,
    request_json,
    request_post_json,
    request_xml_text,
)


ZENODO_ENDPOINT = "https://zenodo.org/api/records"
FIGSHARE_SEARCH_ENDPOINT = "https://api.figshare.com/v2/articles/search"
FIGSHARE_ARTICLE_ENDPOINT = "https://api.figshare.com/v2/articles/{article_id}"
EUROPE_PMC_SEARCH_ENDPOINT = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPE_PMC_FULLTEXT_ENDPOINT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
TARGET_FILE_EXTENSIONS = (".csv", ".xlsx", ".txt", ".zip")
PMC_TARGET_FILE_EXTENSIONS = (".csv", ".xlsx", ".zip")


def _is_target_file(url: str) -> bool:
    lowered = url.lower()
    return lowered.endswith(TARGET_FILE_EXTENSIONS)


def _is_pmc_target_file(url: str) -> bool:
    lowered = url.lower()
    return lowered.endswith(PMC_TARGET_FILE_EXTENSIONS)


def _boost_from_text(*values: str) -> bool:
    combined = " ".join(values).lower()
    return "raman" in combined or "sers" in combined


def _make_candidate(
    file_url: str,
    source_url: str,
    source_type: str,
    extraction_method: str,
    title: str,
    description: str,
) -> dict[str, object]:
    priority_boost = _boost_from_text(title, description) or prioritize_supplementary_url(file_url)
    return {
        "file_url": file_url,
        "source_url": source_url,
        "source_type": source_type,
        "extraction_method": extraction_method,
        "priority_boost": priority_boost,
    }


def harvest_zenodo(query: str) -> list[dict[str, object]]:
    payload = request_json(ZENODO_ENDPOINT, params={"q": query, "size": 25})
    hits = payload.get("hits", {}).get("hits", [])

    candidates: list[dict[str, object]] = []
    for hit in hits:
        metadata = hit.get("metadata") or {}
        title = str(metadata.get("title") or "")
        description = str(metadata.get("description") or "")
        source_url = str(hit.get("links", {}).get("self_html") or metadata.get("doi_url") or "")
        for file_entry in hit.get("files") or []:
            file_url = str(file_entry.get("links", {}).get("self") or "")
            if not file_url or not _is_target_file(file_url):
                continue
            candidates.append(
                _make_candidate(
                    file_url=file_url,
                    source_url=source_url or file_url,
                    source_type="zenodo",
                    extraction_method="api",
                    title=title,
                    description=description,
                )
            )
    return candidates


def harvest_figshare(query: str) -> list[dict[str, object]]:
    search_payload = request_post_json(
        FIGSHARE_SEARCH_ENDPOINT,
        json_payload={"search_for": query},
        params={"page_size": 25},
    )
    article_summaries = search_payload if isinstance(search_payload, list) else []

    candidates: list[dict[str, object]] = []
    for article_summary in article_summaries:
        article_id = article_summary.get("id")
        if article_id is None:
            continue
        article = request_json(FIGSHARE_ARTICLE_ENDPOINT.format(article_id=article_id))
        title = str(article.get("title") or "")
        description = str(article.get("description") or "")
        source_url = str(article.get("url_public_html") or article.get("figshare_url") or "")
        for file_entry in article.get("files") or []:
            file_url = str(file_entry.get("download_url") or "")
            if not file_url or not _is_target_file(file_url):
                continue
            candidates.append(
                _make_candidate(
                    file_url=file_url,
                    source_url=source_url or file_url,
                    source_type="figshare",
                    extraction_method="api",
                    title=title,
                    description=description,
                )
            )
    return candidates


def harvest_pmc(query: str) -> list[dict[str, object]]:
    payload = request_json(
        EUROPE_PMC_SEARCH_ENDPOINT,
        params={
            "query": f"({query}) AND OPEN_ACCESS:Y",
            "format": "json",
            "pageSize": 25,
            "resultType": "core",
        },
    )
    results = payload.get("resultList", {}).get("result", [])

    candidates: list[dict[str, object]] = []
    for result in results:
        pmcid = result.get("pmcid")
        if not pmcid:
            continue

        title = str(result.get("title") or "")
        abstract = str(result.get("abstractText") or "")
        source_url = f"https://europepmc.org/article/PMC/{pmcid}"
        xml_text = request_xml_text(EUROPE_PMC_FULLTEXT_ENDPOINT.format(pmcid=pmcid))
        soup = BeautifulSoup(xml_text, "xml")

        supplementary_links: list[str] = []
        for tag in soup.find_all(["supplementary-material", "ext-link", "media"]):
            href = (
                tag.get("xlink:href")
                or tag.get("href")
                or tag.get("mimetype")
                or ""
            )
            href = str(href).strip()
            if not href:
                continue
            full_url = urljoin(source_url, href)
            lowered = full_url.lower()
            if _is_pmc_target_file(full_url):
                supplementary_links.append(full_url)

        for link in supplementary_links:
            candidates.append(
                _make_candidate(
                    file_url=link,
                    source_url=source_url,
                    source_type="pmc",
                    extraction_method="api",
                    title=title,
                    description=abstract,
                )
            )

    return candidates
