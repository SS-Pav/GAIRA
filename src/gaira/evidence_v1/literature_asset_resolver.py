from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import duckdb
import requests
from bs4 import BeautifulSoup

from gaira.evidence_v1.constants import (
    DB_PATH,
    LITERATURE_ASSET_RESOLUTION_ASSET_ROOT,
    LITERATURE_ASSET_RESOLUTION_REPORT_ROOT,
    LITERATURE_ASSET_RESOLUTION_TABLES_ROOT,
    LITERATURE_PIPELINE_ASSET_ROOT,
    ensure_literature_asset_resolution_output_dirs,
)
from gaira.evidence_v1.schema import (
    initialize_schema,
    reset_literature_asset_resolution_tables,
)


REQUEST_TIMEOUT = 25
TARGET_LIMIT = 12
PAPER_CORPUS_AUDIT_PATH = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/"
    "gaira_remaining_paper_controlled_ingest_v1/tables/paper_corpus_audit.csv"
)
MANUSCRIPT_ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/gaira_literature_corpus/manuscripts")
SUPPLEMENTARY_ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/gaira_literature_corpus/supplementary")
SOURCE_DATA_ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/gaira_literature_corpus/source_data")

HEADERS = {
    "User-Agent": "GAIRA-asset-resolver/1.0 (targeted literature asset resolution)",
    "Accept": "text/html,application/pdf,application/xml;q=0.9,*/*;q=0.8",
}


@dataclass
class PaperTarget:
    paper_id: str
    title: str
    doi: str
    source: str
    final_score: float
    queue_status: str
    selected_for_ingestion: bool
    asset_ready: bool
    queue_notes: str
    remote_url: str
    manuscript_pdf_path: str
    supplementary_files_json: str
    source_data_links_json: str


@dataclass
class ResolvedAsset:
    resolved_asset_id: str
    paper_id: str
    asset_type: str
    source_url: str
    local_path: str
    file_type: str
    resolution_method: str
    resolution_status: str
    checksum_sha256: str
    manuscript_or_si: str
    notes: str


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _normalize_doi(value: str | None) -> str:
    if not value:
        return ""
    doi = value.strip().lower()
    doi = doi.removeprefix("https://doi.org/")
    doi = doi.removeprefix("http://doi.org/")
    doi = doi.removeprefix("doi:")
    return doi


def _normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _title_key(value: str) -> str:
    words = [word for word in _normalize_title(value).split() if word not in {"the", "a", "an", "of", "and", "for", "using"}]
    return " ".join(words[:20])


def _checksum(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _file_type_from_name(value: str) -> str:
    suffix = Path(urlparse(value).path).suffix.lower().lstrip(".")
    return suffix or "unknown"


def _paper_targets(connection: duckdb.DuckDBPyConnection, limit: int = TARGET_LIMIT) -> list[PaperTarget]:
    rows = connection.sql(
        f"""
        SELECT paper_id, title, doi, source, final_score, queue_status, selected_for_ingestion, asset_ready, notes,
               remote_url, manuscript_pdf_path, supplementary_files_json, source_data_links_json
        FROM (
          SELECT
            c.paper_id,
            c.title,
            c.doi,
            c.source,
            t.final_score,
            q.queue_status,
            q.selected_for_ingestion,
            q.asset_ready,
            q.notes,
            COALESCE(a.remote_url, '') AS remote_url,
            COALESCE(a.manuscript_pdf_path, '') AS manuscript_pdf_path,
            COALESCE(a.supplementary_files_json, '[]') AS supplementary_files_json,
            COALESCE(a.source_data_links_json, '[]') AS source_data_links_json,
            ROW_NUMBER() OVER (PARTITION BY c.paper_id ORDER BY a.asset_id) AS rn
          FROM literature.candidate_papers c
          JOIN literature.paper_triage t USING (paper_id)
          JOIN literature.processing_queue q USING (paper_id)
          LEFT JOIN literature.paper_assets a USING (paper_id)
          WHERE q.selected_for_ingestion = TRUE
             OR (q.queue_status = 'pending' AND t.final_score >= 0.80)
        )
        WHERE rn = 1
        ORDER BY selected_for_ingestion DESC, final_score DESC, paper_id
        LIMIT {limit}
        """
    ).fetchall()
    return [
        PaperTarget(
            paper_id=row[0],
            title=row[1],
            doi=_normalize_doi(row[2]),
            source=row[3],
            final_score=float(row[4]),
            queue_status=row[5],
            selected_for_ingestion=bool(row[6]),
            asset_ready=bool(row[7]),
            queue_notes=row[8] or "",
            remote_url=row[9] or "",
            manuscript_pdf_path=row[10] or "",
            supplementary_files_json=row[11] or "[]",
            source_data_links_json=row[12] or "[]",
        )
        for row in rows
    ]


def _load_local_title_maps() -> tuple[list[Path], list[Path], list[Path], dict[str, dict]]:
    manuscripts = [path for path in MANUSCRIPT_ROOT.glob("*.pdf") if path.is_file() and not path.name.startswith("._")]
    supplementary = [path for path in SUPPLEMENTARY_ROOT.glob("*") if path.is_file() and not path.name.startswith("._")]
    source_data = [path for path in SOURCE_DATA_ROOT.glob("*") if path.is_file() and not path.name.startswith("._")]
    audit_map: dict[str, dict] = {}
    if PAPER_CORPUS_AUDIT_PATH.exists():
        with PAPER_CORPUS_AUDIT_PATH.open() as handle:
            for row in csv.DictReader(handle):
                for key in {
                    _title_key(row.get("title", "")),
                    _title_key(Path(row.get("basename", "")).stem),
                    _title_key(row.get("source_id", "")),
                }:
                    if key:
                        audit_map[key] = row
    return manuscripts, supplementary, source_data, audit_map


def _best_local_match(title: str, paths: list[Path], threshold: float = 0.78) -> Path | None:
    target = _title_key(title)
    best_path = None
    best_score = 0.0
    for path in paths:
        score = 0.0
        candidate_key = _title_key(path.stem)
        if target and candidate_key:
            score = sum(1 for word in target.split() if word in candidate_key.split()) / max(1, len(set(target.split())))
        if score > best_score:
            best_score = score
            best_path = path
    if best_path and best_score >= threshold:
        return best_path
    return None


def _resolve_doi(doi: str) -> tuple[str, bool, str]:
    if not doi:
        return "", False, "not_available"
    try:
        response = requests.get(
            f"https://doi.org/{doi}",
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
    except Exception:
        return "", False, "failed"
    if response.url and "doi.org/" not in response.url:
        return response.url, True, "resolved"
    return response.url or "", False, "landing_page_only"


def _fetch_html(url: str) -> tuple[str, int, str]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    except Exception:
        return "", 0, url
    if "html" not in (response.headers.get("content-type") or ""):
        return "", response.status_code, response.url
    return response.text, response.status_code, response.url


def _pmcid_from_text(text: str) -> str:
    match = re.search(r"(PMC\d{5,})", text or "")
    return match.group(1) if match else ""


def _extract_links_from_article(url: str, html: str) -> tuple[list[str], list[str], list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    manuscript_links: list[str] = []
    supplementary_links: list[str] = []
    source_data_links: list[str] = []

    for anchor in soup.find_all("a", href=True):
        href = urljoin(url, anchor["href"])
        label = " ".join(anchor.get_text(" ", strip=True).split()).lower()
        low = f"{label} {href.lower()}"
        if "pdf" in low and href not in manuscript_links:
            manuscript_links.append(href)
        if any(term in low for term in ("supplement", "supporting", "source data", "xlsx", "zip", "docx", "doc")):
            if any(term in low for term in ("source data", "dataset", "xlsx", "csv", "zip")):
                if href not in source_data_links:
                    source_data_links.append(href)
            if href not in supplementary_links:
                supplementary_links.append(href)
    return manuscript_links, supplementary_links, source_data_links


def _download_binary(url: str, destination: Path) -> tuple[str, str, str]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    except Exception:
        return "failed", "", ""
    content_type = response.headers.get("content-type") or ""
    file_type = _file_type_from_name(response.url or url)
    if "application/pdf" in content_type or "application/vnd" in content_type or file_type in {"pdf", "xlsx", "xls", "zip", "docx", "doc", "csv"}:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        return "resolved_downloaded", str(destination), file_type
    if response.status_code == 200 and "html" in content_type:
        if "Preparing to download" in response.text or "cloudpmc-viewer-pow" in response.text:
            return "resolved_link_only", "", file_type or "pdf"
        return "landing_page_only", "", file_type
    if response.status_code in {401, 403}:
        return "inaccessible", "", file_type
    if response.status_code == 404:
        return "not_available", "", file_type
    return "failed", "", file_type


def _asset_id(prefix: str, paper_id: str, index: int) -> str:
    return f"{prefix}_{paper_id}_{index:03d}"


def _attempt_manuscript_resolution(
    target: PaperTarget,
    manuscripts: list[Path],
    audit_map: dict[str, dict],
) -> tuple[list[ResolvedAsset], str, str, str, list[str]]:
    resolved: list[ResolvedAsset] = []
    notes: list[str] = []
    manuscript_local_path = ""
    canonical_url = ""
    manuscript_status = "not_available"

    if target.manuscript_pdf_path and Path(target.manuscript_pdf_path).exists():
        path = Path(target.manuscript_pdf_path)
        resolved.append(
            ResolvedAsset(
                resolved_asset_id=_asset_id("asset", target.paper_id, 1),
                paper_id=target.paper_id,
                asset_type="manuscript_pdf",
                source_url=target.remote_url,
                local_path=str(path),
                file_type="pdf",
                resolution_method="existing_pipeline_asset",
                resolution_status="duplicate_existing",
                checksum_sha256=_checksum(path),
                manuscript_or_si="manuscript",
                notes="Reused manuscript already downloaded by the acquisition pipeline.",
            )
        )
        return resolved, target.remote_url, "duplicate_existing", str(path), notes

    local_match = _best_local_match(target.title, manuscripts)
    if local_match is None:
        audit_row = audit_map.get(_title_key(target.title))
        if audit_row:
            candidate_path = MANUSCRIPT_ROOT / audit_row["basename"]
            if candidate_path.exists():
                local_match = candidate_path
    if local_match is not None:
        resolved.append(
            ResolvedAsset(
                resolved_asset_id=_asset_id("asset", target.paper_id, 1),
                paper_id=target.paper_id,
                asset_type="manuscript_pdf",
                source_url="",
                local_path=str(local_match),
                file_type="pdf",
                resolution_method="local_corpus_match",
                resolution_status="duplicate_existing",
                checksum_sha256=_checksum(local_match),
                manuscript_or_si="manuscript",
                notes="Linked an existing manuscript from the local GAIRA literature corpus.",
            )
        )
        return resolved, "", "duplicate_existing", str(local_match), notes

    doi_url, doi_resolved, doi_note = _resolve_doi(target.doi)
    if doi_url:
        canonical_url = doi_url
        notes.append(f"doi:{doi_note}")
    candidate_urls: list[tuple[str, str]] = []
    if target.remote_url:
        candidate_urls.append(("existing_remote_url", target.remote_url))
    if canonical_url:
        candidate_urls.append(("doi_redirect", canonical_url))

    pmcid = _pmcid_from_text(f"{target.remote_url} {canonical_url}")
    if pmcid:
        candidate_urls.append(("pmc_article", f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"))

    seen_urls: set[str] = set()
    next_index = 1
    for method, url in candidate_urls:
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        html, status_code, final_url = _fetch_html(url)
        if final_url and not canonical_url:
            canonical_url = final_url
        if status_code in {401, 403}:
            manuscript_status = "inaccessible"
            notes.append(f"{method}:inaccessible")
            continue
        if html:
            manuscript_links, _, _ = _extract_links_from_article(final_url or url, html)
            if "/full" in (final_url or url) and "frontiersin.org" in (final_url or url):
                manuscript_links.insert(0, (final_url or url).replace("/full", "/pdf"))
            for manuscript_url in manuscript_links[:6]:
                destination = LITERATURE_ASSET_RESOLUTION_ASSET_ROOT / target.paper_id / "manuscript" / f"manuscript_{next_index}.{_file_type_from_name(manuscript_url)}"
                status, local_path, file_type = _download_binary(manuscript_url, destination)
                resolved.append(
                    ResolvedAsset(
                        resolved_asset_id=_asset_id("asset", target.paper_id, next_index),
                        paper_id=target.paper_id,
                        asset_type="manuscript_pdf",
                        source_url=manuscript_url,
                        local_path=local_path,
                        file_type=file_type or "pdf",
                        resolution_method=method,
                        resolution_status=status,
                        checksum_sha256=_checksum(Path(local_path)) if local_path else "",
                        manuscript_or_si="manuscript",
                        notes=f"Resolved from article page {final_url or url}.",
                    )
                )
                next_index += 1
                if status in {"resolved_downloaded", "duplicate_existing"}:
                    manuscript_status = status
                    manuscript_local_path = local_path
                    return resolved, canonical_url, manuscript_status, manuscript_local_path, notes
                if status == "resolved_link_only" and manuscript_status not in {"resolved_downloaded", "duplicate_existing"}:
                    manuscript_status = "resolved_link_only"
            if manuscript_links and manuscript_status == "not_available":
                manuscript_status = "landing_page_only"
            continue
        if final_url and final_url.endswith(".pdf"):
            destination = LITERATURE_ASSET_RESOLUTION_ASSET_ROOT / target.paper_id / "manuscript" / "manuscript.pdf"
            status, local_path, file_type = _download_binary(final_url, destination)
            resolved.append(
                ResolvedAsset(
                    resolved_asset_id=_asset_id("asset", target.paper_id, next_index),
                    paper_id=target.paper_id,
                    asset_type="manuscript_pdf",
                    source_url=final_url,
                    local_path=local_path,
                    file_type=file_type or "pdf",
                    resolution_method=method,
                    resolution_status=status,
                    checksum_sha256=_checksum(Path(local_path)) if local_path else "",
                    manuscript_or_si="manuscript",
                    notes="Direct manuscript attempt from canonical or existing remote URL.",
                )
            )
            next_index += 1
            if status in {"resolved_downloaded", "duplicate_existing"}:
                return resolved, canonical_url or final_url, status, local_path, notes
            if status == "resolved_link_only":
                manuscript_status = "resolved_link_only"

    if not canonical_url and target.remote_url:
        canonical_url = target.remote_url
    return resolved, canonical_url, manuscript_status, manuscript_local_path, notes


def _attempt_supporting_assets(
    target: PaperTarget,
    canonical_url: str,
    source_data_paths: list[Path],
) -> tuple[list[ResolvedAsset], str, list[str], str, list[str], list[str]]:
    resolved: list[ResolvedAsset] = []
    supplementary_status = "not_available"
    source_data_status = "not_available"
    supplementary_local_paths: list[str] = []
    source_data_local_paths: list[str] = []
    notes: list[str] = []

    existing_supp = [path for path in json.loads(target.supplementary_files_json or "[]") if Path(path).exists()]
    for index, path_string in enumerate(existing_supp, start=1):
        path = Path(path_string)
        resolved.append(
            ResolvedAsset(
                resolved_asset_id=_asset_id("supp", target.paper_id, index),
                paper_id=target.paper_id,
                asset_type="supplementary",
                source_url="",
                local_path=str(path),
                file_type=_file_type_from_name(str(path)),
                resolution_method="existing_local_supplementary",
                resolution_status="duplicate_existing",
                checksum_sha256=_checksum(path),
                manuscript_or_si="si",
                notes="Reused an already available local supplementary asset.",
            )
        )
        supplementary_local_paths.append(str(path))
    if supplementary_local_paths:
        supplementary_status = "duplicate_existing"

    existing_source = [path for path in json.loads(target.source_data_links_json or "[]") if Path(path).exists()]
    for index, path_string in enumerate(existing_source, start=1):
        path = Path(path_string)
        resolved.append(
            ResolvedAsset(
                resolved_asset_id=_asset_id("srcdata", target.paper_id, index),
                paper_id=target.paper_id,
                asset_type="source_data",
                source_url="",
                local_path=str(path),
                file_type=_file_type_from_name(str(path)),
                resolution_method="existing_local_source_data",
                resolution_status="duplicate_existing",
                checksum_sha256=_checksum(path),
                manuscript_or_si="si",
                notes="Reused an already available local source-data asset.",
            )
        )
        source_data_local_paths.append(str(path))
    if source_data_local_paths:
        source_data_status = "duplicate_existing"

    if canonical_url:
        html, status_code, final_url = _fetch_html(canonical_url)
        if html:
            _, supplementary_links, source_data_links = _extract_links_from_article(final_url or canonical_url, html)
            index = 100
            for link in supplementary_links[:8]:
                destination = LITERATURE_ASSET_RESOLUTION_ASSET_ROOT / target.paper_id / "supplementary" / f"supp_{index}.{_file_type_from_name(link)}"
                status, local_path, file_type = _download_binary(link, destination)
                resolved.append(
                    ResolvedAsset(
                        resolved_asset_id=_asset_id("supp", target.paper_id, index),
                        paper_id=target.paper_id,
                        asset_type="supplementary",
                        source_url=link,
                        local_path=local_path,
                        file_type=file_type or _file_type_from_name(link),
                        resolution_method="article_page_parse",
                        resolution_status=status,
                        checksum_sha256=_checksum(Path(local_path)) if local_path else "",
                        manuscript_or_si="si",
                        notes=f"Supplementary candidate extracted from {final_url or canonical_url}.",
                    )
                )
                index += 1
                if local_path:
                    supplementary_local_paths.append(local_path)
                if status in {"resolved_downloaded", "duplicate_existing"}:
                    supplementary_status = "resolved_downloaded"
                elif status == "resolved_link_only" and supplementary_status == "not_available":
                    supplementary_status = "resolved_link_only"
            for link in source_data_links[:8]:
                destination = LITERATURE_ASSET_RESOLUTION_ASSET_ROOT / target.paper_id / "source_data" / f"source_{index}.{_file_type_from_name(link)}"
                status, local_path, file_type = _download_binary(link, destination)
                resolved.append(
                    ResolvedAsset(
                        resolved_asset_id=_asset_id("srcdata", target.paper_id, index),
                        paper_id=target.paper_id,
                        asset_type="source_data",
                        source_url=link,
                        local_path=local_path,
                        file_type=file_type or _file_type_from_name(link),
                        resolution_method="article_page_parse",
                        resolution_status=status,
                        checksum_sha256=_checksum(Path(local_path)) if local_path else "",
                        manuscript_or_si="si",
                        notes=f"Source-data candidate extracted from {final_url or canonical_url}.",
                    )
                )
                index += 1
                if local_path:
                    source_data_local_paths.append(local_path)
                if status in {"resolved_downloaded", "duplicate_existing"}:
                    source_data_status = "resolved_downloaded"
                elif status == "resolved_link_only" and source_data_status == "not_available":
                    source_data_status = "resolved_link_only"
        elif status_code in {401, 403}:
            notes.append("support_assets:inaccessible")

    if supplementary_status == "not_available":
        local_supp_match = _best_local_match(target.title, [path for path in SUPPLEMENTARY_ROOT.glob("*") if path.is_file() and not path.name.startswith("._")], threshold=0.62)
        if local_supp_match:
            supplementary_status = "duplicate_existing"
            supplementary_local_paths.append(str(local_supp_match))
            resolved.append(
                ResolvedAsset(
                    resolved_asset_id=_asset_id("supp", target.paper_id, 999),
                    paper_id=target.paper_id,
                    asset_type="supplementary",
                    source_url="",
                    local_path=str(local_supp_match),
                    file_type=_file_type_from_name(str(local_supp_match)),
                    resolution_method="local_corpus_match",
                    resolution_status="duplicate_existing",
                    checksum_sha256=_checksum(local_supp_match),
                    manuscript_or_si="si",
                    notes="Linked a local supplementary file by title similarity.",
                )
            )

    local_source_match = _best_local_match(target.title, source_data_paths, threshold=0.60)
    if local_source_match and source_data_status == "not_available":
        source_data_status = "duplicate_existing"
        source_data_local_paths.append(str(local_source_match))
        resolved.append(
            ResolvedAsset(
                resolved_asset_id=_asset_id("srcdata", target.paper_id, 999),
                paper_id=target.paper_id,
                asset_type="source_data",
                source_url="",
                local_path=str(local_source_match),
                file_type=_file_type_from_name(str(local_source_match)),
                resolution_method="local_corpus_match",
                resolution_status="duplicate_existing",
                checksum_sha256=_checksum(local_source_match),
                manuscript_or_si="si",
                notes="Linked a local source-data file by title similarity.",
            )
        )

    return resolved, supplementary_status, sorted(set(supplementary_local_paths)), source_data_status, sorted(set(source_data_local_paths)), notes


def _readiness_status(manuscript_path: str, supplementary_paths: list[str], source_data_paths: list[str]) -> str:
    ready_count = sum(bool(item) for item in [manuscript_path, supplementary_paths, source_data_paths])
    if ready_count >= 2:
        return "multi_asset_ready"
    if manuscript_path:
        return "manuscript_ready"
    if supplementary_paths:
        return "si_ready"
    if source_data_paths:
        return "source_data_ready"
    return "not_ready"


def _update_existing_tables(
    connection: duckdb.DuckDBPyConnection,
    target: PaperTarget,
    canonical_url: str,
    manuscript_status: str,
    manuscript_path: str,
    supplementary_paths: list[str],
    source_data_paths: list[str],
    readiness_status: str,
    notes: str,
) -> None:
    rows = connection.sql(
        "SELECT COUNT(*) FROM literature.paper_assets WHERE paper_id = ?",
        params=[target.paper_id],
    ).fetchone()[0]
    if rows == 0:
        connection.execute(
            """
            INSERT INTO literature.paper_assets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                f"asset_{target.paper_id}_manuscript",
                target.paper_id,
                target.source,
                "manuscript_pdf",
                manuscript_path,
                json.dumps(supplementary_paths),
                json.dumps(source_data_paths),
                False,
                False,
                canonical_url,
                manuscript_path,
                "pdf" if manuscript_path else "",
                manuscript_status,
                notes,
            ],
        )
    else:
        connection.execute(
            """
            UPDATE literature.paper_assets
            SET manuscript_pdf_path = ?,
                supplementary_files_json = ?,
                source_data_links_json = ?,
                remote_url = ?,
                local_path = ?,
                download_status = ?,
                notes = ?
            WHERE paper_id = ?
            """,
            [
                manuscript_path,
                json.dumps(supplementary_paths),
                json.dumps(source_data_paths),
                canonical_url,
                manuscript_path,
                manuscript_status,
                notes,
                target.paper_id,
            ],
        )
    connection.execute(
        """
        UPDATE literature.processing_queue
        SET asset_ready = ?, notes = ?
        WHERE paper_id = ?
        """,
        [readiness_status != "not_ready", notes, target.paper_id],
    )


def run_literature_asset_resolver(db_path: Path = DB_PATH) -> dict[str, int]:
    ensure_literature_asset_resolution_output_dirs()
    manuscripts, _supplementary, source_data_paths, audit_map = _load_local_title_maps()

    connection = duckdb.connect(str(db_path))
    try:
        initialize_schema(connection)
        reset_literature_asset_resolution_tables(connection)

        targets = _paper_targets(connection)
        paper_rows: list[dict] = []
        manuscript_rows: list[dict] = []
        supplementary_rows: list[dict] = []
        source_data_rows: list[dict] = []
        asset_rows: list[ResolvedAsset] = []

        for target in targets:
            manuscript_assets, canonical_url, manuscript_status, manuscript_path, manuscript_notes = _attempt_manuscript_resolution(
                target,
                manuscripts,
                audit_map,
            )
            support_assets, supplementary_status, supplementary_paths, source_data_status, source_data_local_paths, support_notes = _attempt_supporting_assets(
                target,
                canonical_url or target.remote_url,
                source_data_paths,
            )
            readiness_status = _readiness_status(manuscript_path, supplementary_paths, source_data_local_paths)
            note_text = "; ".join(note for note in [target.queue_notes, *manuscript_notes, *support_notes] if note)

            asset_rows.extend(manuscript_assets)
            asset_rows.extend(support_assets)

            connection.execute(
                """
                INSERT INTO literature.paper_asset_resolution VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    target.paper_id,
                    canonical_url,
                    bool(canonical_url),
                    manuscript_status,
                    supplementary_status,
                    source_data_status,
                    manuscript_path,
                    json.dumps(supplementary_paths),
                    json.dumps(source_data_local_paths),
                    readiness_status,
                    note_text,
                ],
            )

            _update_existing_tables(
                connection,
                target,
                canonical_url,
                manuscript_status,
                manuscript_path,
                supplementary_paths,
                source_data_local_paths,
                readiness_status,
                note_text,
            )

            paper_rows.append(
                {
                    "paper_id": target.paper_id,
                    "title": target.title,
                    "doi": target.doi,
                    "final_score": target.final_score,
                    "queue_status": target.queue_status,
                    "selected_for_ingestion": target.selected_for_ingestion,
                    "canonical_article_url": canonical_url,
                    "doi_resolved": bool(canonical_url),
                    "manuscript_asset_status": manuscript_status,
                    "supplementary_asset_status": supplementary_status,
                    "source_data_asset_status": source_data_status,
                    "manuscript_local_path": manuscript_path,
                    "supplementary_local_paths_json": json.dumps(supplementary_paths),
                    "source_data_local_paths_json": json.dumps(source_data_local_paths),
                    "readiness_status": readiness_status,
                    "asset_resolution_notes": note_text,
                }
            )
            manuscript_rows.append(
                {
                    "paper_id": target.paper_id,
                    "title": target.title,
                    "canonical_article_url": canonical_url,
                    "manuscript_asset_status": manuscript_status,
                    "manuscript_local_path": manuscript_path,
                    "notes": note_text,
                }
            )
            supplementary_rows.append(
                {
                    "paper_id": target.paper_id,
                    "title": target.title,
                    "supplementary_asset_status": supplementary_status,
                    "supplementary_local_paths_json": json.dumps(supplementary_paths),
                    "notes": note_text,
                }
            )
            source_data_rows.append(
                {
                    "paper_id": target.paper_id,
                    "title": target.title,
                    "source_data_asset_status": source_data_status,
                    "source_data_local_paths_json": json.dumps(source_data_local_paths),
                    "notes": note_text,
                }
            )

        if asset_rows:
            connection.executemany(
                """
                INSERT INTO literature.resolved_assets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        asset.resolved_asset_id,
                        asset.paper_id,
                        asset.asset_type,
                        asset.source_url,
                        asset.local_path,
                        asset.file_type,
                        asset.resolution_method,
                        asset.resolution_status,
                        asset.checksum_sha256,
                        asset.manuscript_or_si,
                        asset.notes,
                    )
                    for asset in asset_rows
                ],
            )
        connection.commit()

        resolved_asset_dicts = [
            {
                "resolved_asset_id": asset.resolved_asset_id,
                "paper_id": asset.paper_id,
                "asset_type": asset.asset_type,
                "source_url": asset.source_url,
                "local_path": asset.local_path,
                "file_type": asset.file_type,
                "resolution_method": asset.resolution_method,
                "resolution_status": asset.resolution_status,
                "checksum_sha256": asset.checksum_sha256,
                "manuscript_or_si": asset.manuscript_or_si,
                "notes": asset.notes,
            }
            for asset in asset_rows
        ]

        _write_csv(
            LITERATURE_ASSET_RESOLUTION_TABLES_ROOT / "top_ranked_asset_resolution_attempts.csv",
            list(paper_rows[0].keys()) if paper_rows else ["paper_id"],
            paper_rows,
        )
        _write_csv(
            LITERATURE_ASSET_RESOLUTION_TABLES_ROOT / "manuscript_resolution_summary.csv",
            list(manuscript_rows[0].keys()) if manuscript_rows else ["paper_id"],
            manuscript_rows,
        )
        _write_csv(
            LITERATURE_ASSET_RESOLUTION_TABLES_ROOT / "supplementary_resolution_summary.csv",
            list(supplementary_rows[0].keys()) if supplementary_rows else ["paper_id"],
            supplementary_rows,
        )
        _write_csv(
            LITERATURE_ASSET_RESOLUTION_TABLES_ROOT / "source_data_resolution_summary.csv",
            list(source_data_rows[0].keys()) if source_data_rows else ["paper_id"],
            source_data_rows,
        )
        _write_csv(
            LITERATURE_ASSET_RESOLUTION_TABLES_ROOT / "resolved_asset_inventory.csv",
            list(resolved_asset_dicts[0].keys()) if resolved_asset_dicts else ["resolved_asset_id"],
            resolved_asset_dicts,
        )
        _write_csv(
            LITERATURE_ASSET_RESOLUTION_TABLES_ROOT / "ingestion_ready_papers.csv",
            list(paper_rows[0].keys()) if paper_rows else ["paper_id"],
            [row for row in paper_rows if row["readiness_status"] != "not_ready"],
        )
        _write_csv(
            LITERATURE_ASSET_RESOLUTION_TABLES_ROOT / "unresolved_high_priority_papers.csv",
            list(paper_rows[0].keys()) if paper_rows else ["paper_id"],
            [row for row in paper_rows if row["readiness_status"] == "not_ready"],
        )
        _write_csv(
            LITERATURE_ASSET_RESOLUTION_TABLES_ROOT / "duplicate_asset_links.csv",
            list(resolved_asset_dicts[0].keys()) if resolved_asset_dicts else ["resolved_asset_id"],
            [row for row in resolved_asset_dicts if row["resolution_status"] == "duplicate_existing"],
        )

        manuscript_resolved = sum(
            1
            for row in paper_rows
            if row["manuscript_asset_status"] in {"resolved_downloaded", "duplicate_existing"}
        )
        supplementary_resolved = sum(
            1
            for row in resolved_asset_dicts
            if row["asset_type"] == "supplementary" and row["resolution_status"] in {"resolved_downloaded", "duplicate_existing", "resolved_link_only"}
        )
        source_data_resolved = sum(
            1
            for row in resolved_asset_dicts
            if row["asset_type"] == "source_data" and row["resolution_status"] in {"resolved_downloaded", "duplicate_existing", "resolved_link_only"}
        )
        ingestion_ready = sum(1 for row in paper_rows if row["readiness_status"] != "not_ready")

        failure_counter: dict[str, int] = {}
        for row in paper_rows:
            if row["readiness_status"] == "not_ready":
                key = row["manuscript_asset_status"]
                failure_counter[key] = failure_counter.get(key, 0) + 1

        implementation_lines = [
            "# Implementation Note",
            "",
            "This pass adds a targeted asset resolver on top of the existing literature candidate/triage/queue tables.",
            "It does not rerun broad discovery and does not ingest evidence.",
            "",
            "Resolution order:",
            "1. Reuse existing local manuscript/SI/source-data assets when present.",
            "2. Resolve DOI to a canonical article URL.",
            "3. Parse structured article pages for manuscript PDF, supplementary, and source-data links.",
            "4. Download only directly accessible binary assets; otherwise record `resolved_link_only`, `landing_page_only`, or `inaccessible` honestly.",
        ]
        (LITERATURE_ASSET_RESOLUTION_REPORT_ROOT / "implementation_note.md").write_text("\n".join(implementation_lines))

        assessment_lines = [
            "# Current State Assessment",
            "",
            f"- Top-ranked candidates attempted: `{len(paper_rows)}`",
            f"- Manuscript PDFs resolved locally: `{manuscript_resolved}`",
            f"- Supplementary assets resolved: `{supplementary_resolved}`",
            f"- Source-data assets resolved: `{source_data_resolved}`",
            f"- Papers now ingestion-ready: `{ingestion_ready}`",
            "",
            "Dominant remaining failure modes:",
        ]
        for key, value in sorted(failure_counter.items(), key=lambda item: (-item[1], item[0])):
            assessment_lines.append(f"- `{key}`: `{value}`")
        assessment_lines.extend(
            [
                "",
                "This resolver improved the acquisition layer where OA or existing local assets were available.",
                "The main remaining blockers are publisher anti-bot / access-denied pages and cases where only an article landing page was reachable without a directly downloadable manuscript or SI file.",
                "",
                f"Re-run of controlled ingestion is {'justified' if ingestion_ready > 0 else 'not yet justified'} for the newly ingestion-ready subset.",
            ]
        )
        (LITERATURE_ASSET_RESOLUTION_REPORT_ROOT / "current_state_assessment.md").write_text("\n".join(assessment_lines))

        return {
            "attempted_candidates": len(paper_rows),
            "manuscript_pdfs_resolved": manuscript_resolved,
            "supplementary_assets_resolved": supplementary_resolved,
            "source_data_assets_resolved": source_data_resolved,
            "ingestion_ready_papers": ingestion_ready,
        }
    finally:
        connection.close()


if __name__ == "__main__":
    print(json.dumps(run_literature_asset_resolver(), indent=2, sort_keys=True))
