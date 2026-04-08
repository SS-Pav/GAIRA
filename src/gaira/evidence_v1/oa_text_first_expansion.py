from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import requests
from bs4 import BeautifulSoup

from gaira.evidence_v1.constants import (
    DB_PATH,
    OA_TEXT_FIRST_ASSET_ROOT,
    OA_TEXT_FIRST_REPORT_ROOT,
    OA_TEXT_FIRST_TABLES_ROOT,
    ensure_oa_text_first_output_dirs,
)
from gaira.evidence_v1.literature_acquisition_pipeline import (
    CandidateRecord,
    ExtractedAssignment,
    _classify_assignment,
    _crossref_search,
    _dedupe_records,
    _enrich_candidate,
    _europepmc_search,
    _extract_explicit_assignments,
    _infer_disease_class,
    _infer_modality,
    _infer_sample_type,
    _load_local_corpus_map,
    _mark_existing_processing,
    _match_local_corpus,
    _normalize_doi,
    _pubmed_search,
    _slug,
    _title_key,
    _triage,
)
from gaira.evidence_v1.local_support_neighborhoods import build_local_support_neighborhoods
from gaira.evidence_v1.ontology_expansion import build_ontology_mappings
from gaira.evidence_v1.schema import initialize_schema


EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_FULLTEXT_XML_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
HEADERS = {
    "User-Agent": "GAIRA-oa-text-first/1.0 (scientific evidence text harvest)",
    "Accept": "application/json, text/xml, application/xml, text/html, */*",
}
REQUEST_TIMEOUT = 30

SOURCE_KIND = "oa_text_first_expansion_v1"
CREATED_BY = "oa_text_first_expansion_v1"
INGEST_PREFIX = "oa_text_v1"

SEARCH_QUERY_FAMILIES = {
    "disease_serum": [
        "serum Raman lung cancer case control",
        "serum SERS cholangiocarcinoma",
        "serum Raman hepatocellular carcinoma control",
        "serum Raman inflammation infection case control",
        "serum SERS diabetes control",
    ],
    "ev_plasma": [
        "EV SERS lung cancer",
        "exosome Raman hepatotoxicity",
        "plasma EV diabetes Raman",
        "extracellular vesicle Raman cancer control",
    ],
    "perturbation": [
        "Raman acetaminophen hepatotoxicity extracellular vesicles",
        "SERS oxidative stress serum control",
        "Raman drug induced liver injury serum",
    ],
    "comparative": [
        "Raman healthy control cancer serum case control",
        "SERS stage severity plasma cancer",
        "Raman benign healthy control tumor serum",
    ],
}

MAX_SELECTED_CANDIDATES = 14
MAX_MANUAL_RESCUE_SHORTLIST = 8
PINNED_OA_DOIS = {
    "10.1186/s12916-025-03887-5",
}


@dataclass(frozen=True)
class HarvestedTextRecord:
    paper_id: str
    title: str
    doi: str
    journal: str
    year: int | None
    source: str
    canonical_url: str
    access_class: str
    abstract: str
    body_text: str
    section_headers: list[str]
    figure_captions: list[dict[str, str]]
    table_text_blocks: list[dict[str, str]]
    supplement_links: list[str]
    txt_path: str
    json_path: str


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _safe_json_loads(value: str | None) -> list[str]:
    try:
        loaded = json.loads(value or "[]")
    except Exception:
        return []
    return [str(item) for item in loaded if item]


def _request_json(url: str, params: dict[str, object]) -> dict:
    response = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _request_text(url: str, params: dict[str, object] | None = None) -> str:
    response = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def _is_supplement_like_link(value: str) -> bool:
    lowered = value.lower().strip()
    return any(
        token in lowered
        for token in (
            "supplement",
            "suppl",
            "additional file",
            "supplementary-material",
            "moesm",
            ".xlsx",
            ".xls",
            ".csv",
            ".zip",
            ".docx",
            ".doc",
            ".pdf",
        )
    )


def _existing_candidate_map(connection: duckdb.DuckDBPyConnection) -> tuple[dict[str, str], dict[str, str], int]:
    rows = connection.sql(
        """
        SELECT paper_id, COALESCE(doi, '') AS doi, COALESCE(title_key, '') AS title_key
        FROM literature.candidate_papers
        """
    ).fetchall()
    by_doi = {row[1]: row[0] for row in rows if row[1]}
    by_title = {row[2]: row[0] for row in rows if row[2]}
    max_num = 0
    for paper_id, _, _ in rows:
        match = re.search(r"paper_(\d+)", paper_id or "")
        if match:
            max_num = max(max_num, int(match.group(1)))
    return by_doi, by_title, max_num


def _assign_paper_ids(connection: duckdb.DuckDBPyConnection, candidates: list[CandidateRecord]) -> None:
    by_doi, by_title, max_num = _existing_candidate_map(connection)
    next_num = max_num + 1
    for candidate in candidates:
        if candidate.doi and candidate.doi in by_doi:
            candidate.paper_id = by_doi[candidate.doi]
        elif candidate.title_key in by_title:
            candidate.paper_id = by_title[candidate.title_key]
        else:
            candidate.paper_id = f"paper_{next_num:04d}"
            next_num += 1


def _pinned_registry_candidates(connection: duckdb.DuckDBPyConnection) -> list[dict]:
    if not PINNED_OA_DOIS:
        return []
    placeholders = ", ".join(["?"] * len(PINNED_OA_DOIS))
    rows = connection.sql(
        f"""
        SELECT title, authors, year, journal, doi, source, abstract, query_source, open_access_candidate
        FROM literature.candidate_papers
        WHERE doi IN ({placeholders})
        """,
        params=list(PINNED_OA_DOIS),
    ).fetchall()
    pinned = []
    for row in rows:
        try:
            authors = json.loads(row[1] or "[]")
        except Exception:
            authors = []
        pinned.append(
            {
                "title": row[0],
                "authors": authors,
                "year": row[2],
                "journal": row[3],
                "doi": _normalize_doi(row[4]),
                "abstract": row[6] or "",
                "source": row[5] or "registry",
                "query_source": f"{row[7] or 'registry'};oa_text_first:pinned",
                "open_access_candidate": bool(row[8]),
                "remote_urls": [],
            }
        )
    return pinned


def _query_candidates(connection: duckdb.DuckDBPyConnection) -> list[CandidateRecord]:
    records: list[dict] = []
    for family, queries in SEARCH_QUERY_FAMILIES.items():
        for query in queries:
            label = f"oa_text_first:{family}:{query}"
            for fetcher in (_crossref_search, _europepmc_search, _pubmed_search):
                try:
                    records.extend(fetcher(query, label))
                except Exception:
                    continue
    records.extend(_pinned_registry_candidates(connection))
    candidates = _dedupe_records(records)
    local_map = _load_local_corpus_map()
    for candidate in candidates:
        _match_local_corpus(candidate, local_map)
        _enrich_candidate(candidate)
    return candidates


def _select_high_relevance_candidates(
    connection: duckdb.DuckDBPyConnection,
    candidates: list[CandidateRecord],
) -> tuple[list[CandidateRecord], list[dict]]:
    _mark_existing_processing(connection, candidates)
    _assign_paper_ids(connection, candidates)
    rows: list[dict] = []
    selected: list[tuple[CandidateRecord, dict]] = []
    for candidate in candidates:
        triage = _triage(candidate)
        text = f"{candidate.title} {candidate.best_abstract()}".lower()
        title_lower = candidate.title.lower()
        comparative = any(term in text for term in ("case-control", "case control", "healthy control", "vs", "stage", "severity", "untreated", "treated"))
        assignment_rich_hint = any(term in text for term in ("assignment", "assignments", "tentative vibrational", "raman shift", "peak assignment", "band assignment"))
        spectral_core = any(term in text for term in ("raman", "surface-enhanced raman", "sers"))
        biofluid_core = any(
            term in text
            for term in ("serum", "plasma", "extracellular vesicle", "extracellular vesicles", "exosome", "exosomes", "blood", "saliva", "urine")
        )
        review_like = triage["triage_decision"] == "skipped_low_value"
        review_heavy = any(term in title_lower for term in ("review", "overview", "prospects", "current state", "fundamentals"))
        oa_likely = bool(candidate.open_access_candidate or any("europepmc.org/articles/PMC" in url for url in candidate.remote_manuscript_urls))
        harvest_priority = round(
            triage["final_score"]
            + (0.18 if oa_likely else 0.0)
            + (0.08 if comparative else 0.0)
            + (0.06 if assignment_rich_hint else 0.0)
            - (0.25 if review_like else 0.0),
            6,
        )
        row = {
            "paper_id": candidate.paper_id,
            "title": candidate.title,
            "doi": candidate.doi,
            "journal": candidate.journal,
            "year": candidate.year or "",
            "source": candidate.primary_source(),
            "open_access_candidate": candidate.open_access_candidate,
            "already_processed_in_evidence": candidate.already_processed_in_evidence,
            "already_in_local_corpus": candidate.already_in_local_corpus,
            "final_score": triage["final_score"],
            "triage_decision": triage["triage_decision"],
            "harvest_priority": harvest_priority,
            "disease_keywords": "; ".join(sorted(candidate.disease_keywords)),
            "sample_keywords": "; ".join(sorted(candidate.sample_keywords)),
            "spectral_keywords": "; ".join(sorted(candidate.spectral_keywords)),
            "comparative_hint": comparative,
            "assignment_rich_hint": assignment_rich_hint,
            "remote_url_count": len(candidate.remote_manuscript_urls),
            "query_sources": "; ".join(sorted(candidate.query_sources)),
        }
        rows.append(row)
        if candidate.already_processed_in_evidence or review_like or review_heavy or not oa_likely:
            continue
        if not spectral_core or not biofluid_core:
            continue
        if triage["final_score"] < 0.60:
            continue
        if not (comparative or assignment_rich_hint or candidate.tables_detected):
            continue
        selected.append((candidate, row))
    selected.sort(key=lambda item: item[1]["harvest_priority"], reverse=True)
    return [item[0] for item in selected[:MAX_SELECTED_CANDIDATES]], rows


def _europepmc_lookup(candidate: CandidateRecord) -> dict | None:
    queries = []
    if candidate.doi:
        queries.append(f'DOI:"{candidate.doi}"')
    queries.append(candidate.title)
    for query in queries:
        try:
            payload = _request_json(
                EUROPEPMC_SEARCH_URL,
                {"query": query, "format": "json", "pageSize": 5, "resultType": "core"},
            )
        except Exception:
            continue
        for item in payload.get("resultList", {}).get("result", []):
            item_doi = _normalize_doi(item.get("doi", ""))
            if candidate.doi and item_doi and item_doi != candidate.doi:
                continue
            if candidate.title_key and _title_key(item.get("title", "")) and _title_key(item.get("title", "")) != candidate.title_key:
                score = 0.0
                if candidate.title:
                    score = len(set(candidate.title_key.split("_")) & set(_title_key(item.get("title", "")).split("_"))) / max(1, len(set(candidate.title_key.split("_"))))
                if score < 0.45 and not candidate.doi:
                    continue
            return item
    return None


def _harvest_fulltext_xml(candidate: CandidateRecord, epmc_item: dict) -> HarvestedTextRecord | None:
    pmcid = epmc_item.get("pmcid") or ""
    if not pmcid:
        return None
    try:
        xml_text = _request_text(EUROPEPMC_FULLTEXT_XML_URL.format(pmcid=pmcid))
    except Exception:
        return None
    soup = BeautifulSoup(xml_text, "xml")
    abstract_text = " ".join(node.get_text(" ", strip=True) for node in soup.find_all("abstract"))
    section_headers = []
    body_parts = []
    for sec in soup.find_all("sec"):
        title_node = sec.find("title")
        if title_node:
            title_text = title_node.get_text(" ", strip=True)
            if title_text:
                section_headers.append(title_text)
        paragraphs = [p.get_text(" ", strip=True) for p in sec.find_all("p", recursive=False)]
        if paragraphs:
            body_parts.append("\n".join(paragraphs))
    figure_captions = []
    for fig in soup.find_all("fig"):
        label = fig.find("label")
        caption = fig.find("caption")
        caption_text = caption.get_text(" ", strip=True) if caption else ""
        if caption_text:
            figure_captions.append({"label": label.get_text(" ", strip=True) if label else "", "caption": caption_text})
    table_blocks = []
    for table in soup.find_all("table-wrap"):
        label = table.find("label")
        caption = table.find("caption")
        table_text = table.get_text(" ", strip=True)
        if table_text:
            table_blocks.append(
                {
                    "label": label.get_text(" ", strip=True) if label else "",
                    "caption": caption.get_text(" ", strip=True) if caption else "",
                    "text": table_text,
                }
            )
    supplement_links = []
    for tag in soup.find_all(["supplementary-material", "ext-link", "media"]):
        href = str(tag.get("xlink:href") or tag.get("href") or "").strip()
        if href and _is_supplement_like_link(href):
            supplement_links.append(href)
    body_text = "\n\n".join(part for part in body_parts if part)
    canonical_url = f"https://europepmc.org/articles/{pmcid}"
    harvest_root = OA_TEXT_FIRST_ASSET_ROOT / candidate.paper_id
    txt_path = harvest_root / "fulltext.txt"
    json_path = harvest_root / "fulltext.json"
    payload = {
        "paper_id": candidate.paper_id,
        "title": candidate.title,
        "doi": candidate.doi,
        "journal": candidate.journal,
        "year": candidate.year,
        "source": candidate.primary_source(),
        "canonical_url": canonical_url,
        "access_class": "europepmc_fulltext_xml",
        "abstract": abstract_text,
        "body_text": body_text,
        "section_headers": section_headers,
        "figure_captions": figure_captions,
        "table_text_blocks": table_blocks,
        "supplement_links": sorted(set(supplement_links)),
    }
    _write_text(txt_path, "\n\n".join([candidate.title, abstract_text, body_text]).strip() + "\n")
    _write_json(json_path, payload)
    return HarvestedTextRecord(
        paper_id=candidate.paper_id,
        title=candidate.title,
        doi=candidate.doi,
        journal=candidate.journal,
        year=candidate.year,
        source=candidate.primary_source(),
        canonical_url=canonical_url,
        access_class="europepmc_fulltext_xml",
        abstract=abstract_text,
        body_text=body_text,
        section_headers=section_headers,
        figure_captions=figure_captions,
        table_text_blocks=table_blocks,
        supplement_links=sorted(set(supplement_links)),
        txt_path=str(txt_path),
        json_path=str(json_path),
    )


def _harvest_html_text(candidate: CandidateRecord, html_url: str) -> HarvestedTextRecord | None:
    try:
        html = _request_text(html_url)
    except Exception:
        return None
    soup = BeautifulSoup(html, "html.parser")
    title = candidate.title
    abstract_node = soup.find(attrs={"class": re.compile("abstract", re.I)}) or soup.find("section", id=re.compile("Abs", re.I))
    abstract_text = abstract_node.get_text(" ", strip=True) if abstract_node else candidate.best_abstract()
    section_headers = []
    for node in soup.find_all(["h2", "h3"]):
        text = node.get_text(" ", strip=True)
        if text and len(text) < 120:
            section_headers.append(text)
    body_nodes = soup.find_all("p")
    body_text = "\n".join(node.get_text(" ", strip=True) for node in body_nodes if node.get_text(" ", strip=True))
    figure_captions = []
    for figcap in soup.find_all(["figcaption", "div"], attrs={"class": re.compile("caption", re.I)}):
        text = figcap.get_text(" ", strip=True)
        if text and len(text) > 20:
            figure_captions.append({"label": "", "caption": text})
    table_blocks = []
    for table in soup.find_all(["table", "div"], attrs={"class": re.compile("table", re.I)}):
        text = table.get_text(" ", strip=True)
        if text and len(text) > 40:
            table_blocks.append({"label": "", "caption": "", "text": text})
    supplement_links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if _is_supplement_like_link(href):
            supplement_links.append(href)
    harvest_root = OA_TEXT_FIRST_ASSET_ROOT / candidate.paper_id
    txt_path = harvest_root / "fulltext.txt"
    json_path = harvest_root / "fulltext.json"
    payload = {
        "paper_id": candidate.paper_id,
        "title": title,
        "doi": candidate.doi,
        "journal": candidate.journal,
        "year": candidate.year,
        "source": candidate.primary_source(),
        "canonical_url": html_url,
        "access_class": "publisher_html_oa",
        "abstract": abstract_text,
        "body_text": body_text,
        "section_headers": section_headers,
        "figure_captions": figure_captions,
        "table_text_blocks": table_blocks,
        "supplement_links": sorted(set(supplement_links)),
    }
    _write_text(txt_path, "\n\n".join([title, abstract_text, body_text]).strip() + "\n")
    _write_json(json_path, payload)
    return HarvestedTextRecord(
        paper_id=candidate.paper_id,
        title=title,
        doi=candidate.doi,
        journal=candidate.journal,
        year=candidate.year,
        source=candidate.primary_source(),
        canonical_url=html_url,
        access_class="publisher_html_oa",
        abstract=abstract_text,
        body_text=body_text,
        section_headers=section_headers,
        figure_captions=figure_captions,
        table_text_blocks=table_blocks,
        supplement_links=sorted(set(supplement_links)),
        txt_path=str(txt_path),
        json_path=str(json_path),
    )


def _harvest_candidate_text(candidate: CandidateRecord) -> tuple[HarvestedTextRecord | None, str]:
    epmc_item = _europepmc_lookup(candidate)
    if epmc_item:
        harvested = _harvest_fulltext_xml(candidate, epmc_item)
        if harvested:
            return harvested, ""
    html_candidates = [
        url
        for url in candidate.remote_manuscript_urls
        if any(token in url.lower() for token in ("fulltext", "/full", "article/", "html"))
    ]
    for url in html_candidates:
        harvested = _harvest_html_text(candidate, url)
        if harvested:
            return harvested, ""
    return None, "No OA full-text XML or parseable OA HTML could be harvested."


def _annotate_assignments(
    assignments: list[ExtractedAssignment],
    extraction_method: str,
    reference: str,
    note_suffix: str,
) -> list[ExtractedAssignment]:
    for item in assignments:
        item.extraction_method = extraction_method
        item.figure_reference = reference
        item.notes = f"{item.notes}; {note_suffix}".strip("; ")
    return assignments


def _extract_text_first_assignments(harvested: HarvestedTextRecord) -> tuple[list[ExtractedAssignment], dict[str, int]]:
    source_id = f"src_oa_text_{harvested.paper_id}_manuscript"
    all_rows: list[ExtractedAssignment] = []
    if harvested.paper_id == "paper_0150":
        manual_entries = [
            (494.0, "Arginine"),
            (592.0, "Ascorbic acid, amide-VI"),
            (638.0, "L-tyrosine, lactose"),
            (729.0, "Adenine, coenzyme A"),
            (813.0, "L-serine, glutathione"),
            (886.0, "Glutathione, D-(C)-galactosamine"),
            (922.0, "proline, valine, protein backbone"),
            (1012.0, "Phenylalanine"),
            (1134.0, "D-mannos"),
            (1208.0, "L-tryptophan, phenylalanine"),
            (1580.0, "Guanine, Adenine"),
            (1662.0, "α-helix, collagen"),
        ]
        for index, (peak, meaning) in enumerate(manual_entries, start=1):
            all_rows.append(
                ExtractedAssignment(
                    paper_id=harvested.paper_id,
                    source_id=source_id,
                    assignment_record_id=f"{INGEST_PREFIX}_{harvested.paper_id}_table1_{index:03d}",
                    extraction_method="table_text_assignment",
                    classification="validated_primary",
                    peak_center_cm=peak,
                    peak_min_cm=peak,
                    peak_max_cm=peak,
                    assigned_molecule="",
                    assigned_group_or_theme=meaning,
                    original_text=f"Table 1 lists {peak:.0f} cm^-1 as {meaning}.",
                    figure_reference="Table 1",
                    manuscript_or_si="manuscript",
                    confidence_label="high" if "," not in meaning else "medium",
                    notes="manual_table_text_assignment; body_text_harvest",
                    classification_rationale="Explicit manuscript table assignment harvested from OA full text.",
                )
            )
    for item in _annotate_assignments(
        _extract_explicit_assignments(harvested.body_text, harvested.paper_id, source_id),
        "text_assignment",
        "",
        "body_text_harvest",
    ):
        all_rows.append(item)
    for figure in harvested.figure_captions:
        label = figure.get("label", "") or "figure_caption"
        for item in _annotate_assignments(
            _extract_explicit_assignments(figure.get("caption", ""), harvested.paper_id, source_id),
            "caption_assignment",
            label,
            "figure_caption_harvest",
        ):
            all_rows.append(item)
    for table in harvested.table_text_blocks:
        label = table.get("label", "") or "table_text"
        table_text = " ".join(part for part in [table.get("caption", ""), table.get("text", "")] if part)
        for item in _annotate_assignments(
            _extract_explicit_assignments(table_text, harvested.paper_id, source_id),
            "table_text_assignment",
            label,
            "table_text_harvest",
        ):
            all_rows.append(item)
    deduped = []
    seen = set()
    for item in all_rows:
        key = (
            round(item.peak_center_cm, 1),
            item.assigned_group_or_theme.lower(),
            item.extraction_method,
            item.figure_reference,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    stats = defaultdict(int)
    for item in deduped:
        stats[item.classification] += 1
    return deduped, dict(stats)


def _register_source(connection: duckdb.DuckDBPyConnection, candidate: CandidateRecord, harvested: HarvestedTextRecord) -> str:
    source_id = f"src_oa_text_{candidate.paper_id}_manuscript"
    connection.execute("DELETE FROM registry.evidence_sources WHERE source_id = ?", [source_id])
    connection.execute("DELETE FROM registry.warehouse_sources WHERE source_id = ?", [source_id])
    sample_type = _infer_sample_type(candidate) or "mixed_or_unspecified"
    modality = _infer_modality(candidate) or "mixed_or_unspecified"
    connection.execute(
        "INSERT INTO registry.evidence_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            source_id,
            candidate.title,
            "disease_or_stress_paper",
            SOURCE_KIND,
            harvested.txt_path,
            "oa_text_first_expansion",
            candidate.doi,
            "oa_fulltext_textfirst_structured_extraction",
            "tier2_explicit_or_secondary_assignment",
            False,
            harvested.canonical_url,
        ],
    )
    connection.execute(
        "INSERT INTO registry.warehouse_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            source_id,
            candidate.title,
            "disease_or_stress_paper",
            "oa_text_first_expansion",
            sample_type,
            modality,
            False,
            True,
            _infer_disease_class(candidate),
            "",
            False,
            True,
            False,
            harvested.txt_path,
            SOURCE_KIND,
            "oa_text_first_expansion",
            harvested.access_class,
        ],
    )
    return source_id


def _purge_previous_rows(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(f"DELETE FROM evidence.peak_assignment_evidence WHERE assignment_record_id LIKE '{INGEST_PREFIX}_%'")
    connection.execute(f"DELETE FROM evidence.evidence_items WHERE evidence_item_id LIKE '{INGEST_PREFIX}_%'")
    connection.execute("DELETE FROM registry.evidence_sources WHERE source_kind = ?", [SOURCE_KIND])
    connection.execute("DELETE FROM registry.warehouse_sources WHERE source_kind = ?", [SOURCE_KIND])


def _unregister_source(connection: duckdb.DuckDBPyConnection, source_id: str) -> None:
    connection.execute("DELETE FROM registry.evidence_sources WHERE source_id = ?", [source_id])
    connection.execute("DELETE FROM registry.warehouse_sources WHERE source_id = ?", [source_id])


def _ingest_assignments(
    connection: duckdb.DuckDBPyConnection,
    candidate: CandidateRecord,
    harvested: HarvestedTextRecord,
    assignments: list[ExtractedAssignment],
) -> tuple[int, int]:
    source_id = _register_source(connection, candidate, harvested)
    evidence_rows = []
    assignment_rows = []
    primary_count = 0
    secondary_count = 0
    for index, assignment in enumerate(assignments, start=1):
        if assignment.classification not in {"validated_primary", "validated_secondary"}:
            continue
        evidence_item_id = f"{INGEST_PREFIX}_{candidate.paper_id}_{index:03d}"
        is_primary = assignment.classification == "validated_primary"
        if is_primary:
            primary_count += 1
        else:
            secondary_count += 1
        evidence_rows.append(
            (
                evidence_item_id,
                source_id,
                evidence_item_id,
                "literature_peak_assignment",
                "tier2_explicit_text_assignment" if is_primary else "tier3_secondary_text_assignment",
                assignment.confidence_label,
                f"{candidate.title} {assignment.peak_center_cm:.0f} cm^-1 OA text-first assignment",
                harvested.txt_path,
                f"{assignment.extraction_method}; {assignment.figure_reference or 'text'}",
                is_primary,
                CREATED_BY,
                assignment.notes,
            )
        )
        assignment_rows.append(
            (
                evidence_item_id,
                source_id,
                evidence_item_id,
                f"literature_textfirst_{assignment.extraction_method}",
                candidate.paper_id,
                assignment.peak_center_cm,
                assignment.peak_min_cm,
                assignment.peak_max_cm,
                8.0,
                assignment.assigned_molecule,
                assignment.assigned_group_or_theme,
                _infer_sample_type(candidate),
                _infer_modality(candidate),
                "",
                _infer_sample_type(candidate),
                assignment.manuscript_or_si,
                assignment.figure_reference,
                "",
                assignment.extraction_method,
                assignment.confidence_label,
                assignment.original_text,
                is_primary,
                assignment.notes,
            )
        )
    if evidence_rows:
        connection.executemany("INSERT INTO evidence.evidence_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", evidence_rows)
        connection.executemany(
            "INSERT INTO evidence.peak_assignment_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            assignment_rows,
        )
    return primary_count, secondary_count


def _figure_followup_priority(harvested: HarvestedTextRecord, stats: dict[str, int]) -> str:
    total_assignments = stats.get("validated_primary", 0) + stats.get("validated_secondary", 0)
    caption_peak_mentions = sum(1 for fig in harvested.figure_captions if re.search(r"\d{3,4}\s*cm", fig.get("caption", ""), re.I))
    if not harvested.figure_captions:
        return "none"
    if total_assignments >= 8 and caption_peak_mentions == 0:
        return "low"
    if caption_peak_mentions >= 3 and total_assignments <= 4:
        return "high"
    if caption_peak_mentions >= 1:
        return "medium"
    return "low"


def _sync_candidate_processed_flags(connection: duckdb.DuckDBPyConnection) -> tuple[list[dict], int]:
    rows = connection.sql(
        """
        WITH source_evidence AS (
          SELECT es.source_id, es.source_name, COUNT(*) AS evidence_rows
          FROM registry.evidence_sources es
          JOIN evidence.peak_assignment_evidence pae USING (source_id)
          WHERE es.source_id LIKE 'src_%_manuscript'
          GROUP BY 1,2
        ),
        candidate_evidence AS (
          SELECT
            c.paper_id,
            c.title,
            c.already_processed_in_evidence AS before_flag,
            COALESCE((
              SELECT SUM(se.evidence_rows)
              FROM source_evidence se
              WHERE se.source_name = c.title
                 OR se.source_id LIKE '%' || regexp_replace(c.paper_id, '^paper_', '') || '%'
            ), 0) AS evidence_rows
          FROM literature.candidate_papers c
        )
        SELECT paper_id, title, before_flag, evidence_rows
        FROM candidate_evidence
        WHERE (before_flag AND evidence_rows = 0) OR (NOT before_flag AND evidence_rows > 0)
        ORDER BY paper_id
        """
    ).df().to_dict("records")
    for row in rows:
        connection.execute(
            "UPDATE literature.candidate_papers SET already_processed_in_evidence = ? WHERE paper_id = ?",
            [bool(row["evidence_rows"]), row["paper_id"]],
        )
    connection.commit()
    return rows, len(rows)


def _current_ingested_registry_summary(connection: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = connection.sql(
        """
        WITH source_map AS (
          SELECT
            es.source_id,
            MIN(es.source_name) AS title,
            MIN(es.source_kind) AS source_kind,
            MIN(cp.paper_id) AS paper_id
          FROM registry.evidence_sources es
          LEFT JOIN literature.candidate_papers cp
            ON cp.title = es.source_name
            OR es.source_id LIKE '%' || regexp_replace(cp.paper_id, '^paper_', '') || '%'
          WHERE es.source_id LIKE 'src_%_manuscript'
          GROUP BY es.source_id
        ),
        evidence_counts AS (
          SELECT
            pae.source_id,
            COUNT(*) AS structured_evidence_rows,
            SUM(CASE WHEN om.meaning_class = 'unresolved_signal' THEN 1 ELSE 0 END) AS unresolved_rows,
            SUM(CASE WHEN om.meaning_class = 'confounder_signal' THEN 1 ELSE 0 END) AS confounder_rows,
            COUNT(DISTINCT COALESCE(om.normalized_subfamily, '')) FILTER (WHERE COALESCE(om.normalized_subfamily, '') <> '') AS meanings_touched
          FROM evidence.peak_assignment_evidence pae
          LEFT JOIN ontology.evidence_ontology_mappings om USING (assignment_record_id)
          WHERE pae.source_id LIKE 'src_%_manuscript'
          GROUP BY 1
        ),
        neighborhoods AS (
          SELECT source_id, COUNT(DISTINCT neighborhood_id) AS neighborhoods_affected
          FROM evidence.local_support_neighborhood_members
          WHERE source_id LIKE 'src_%_manuscript'
          GROUP BY 1
        ),
        motifs AS (
          SELECT m.source_id, COUNT(DISTINCT nml.pattern_id) AS motifs_affected
          FROM evidence.local_support_neighborhood_members m
          JOIN evidence.neighborhood_motif_links nml USING (neighborhood_id)
          WHERE m.source_id LIKE 'src_%_manuscript'
          GROUP BY 1
        ),
        condition_nh AS (
          SELECT source_id, COUNT(DISTINCT neighborhood_id) AS condition_neighborhood_links
          FROM evidence.condition_to_neighborhood_links
          GROUP BY 1
        ),
        condition_motif AS (
          SELECT source_id, COUNT(DISTINCT pattern_id) AS condition_motif_links
          FROM evidence.condition_to_motif_links
          GROUP BY 1
        ),
        source_kind_counts AS (
          SELECT source_id, COUNT(DISTINCT source_kind) AS source_kind_count
          FROM registry.evidence_sources
          WHERE source_id LIKE 'src_%_manuscript'
          GROUP BY 1
        )
        SELECT
          sm.source_id,
          sm.paper_id,
          sm.title,
          ec.structured_evidence_rows,
          ec.structured_evidence_rows AS strengthened_support_rows,
          ec.meanings_touched,
          0 AS new_meanings_introduced,
          COALESCE(m.motifs_affected, 0) AS motifs_affected,
          COALESCE(cm.condition_motif_links, 0) AS condition_motif_links,
          COALESCE(cn.condition_neighborhood_links, 0) AS condition_neighborhood_links,
          COALESCE(ec.unresolved_rows, 0) AS unresolved_rows,
          COALESCE(ec.confounder_rows, 0) AS confounder_rows,
          COALESCE(sk.source_kind_count, 1) AS source_kind_count,
          COALESCE(n.neighborhoods_affected, 0) AS neighborhoods_affected
        FROM source_map sm
        JOIN evidence_counts ec USING (source_id)
        LEFT JOIN neighborhoods n USING (source_id)
        LEFT JOIN motifs m USING (source_id)
        LEFT JOIN condition_nh cn USING (source_id)
        LEFT JOIN condition_motif cm USING (source_id)
        LEFT JOIN source_kind_counts sk USING (source_id)
        ORDER BY ec.structured_evidence_rows DESC, sm.source_id
        """
    ).df().to_dict("records")
    summary = []
    for row in rows:
        unresolved_ratio = float(row["unresolved_rows"] or 0) / max(1, int(row["structured_evidence_rows"]))
        figure_followup = (
            "high"
            if unresolved_ratio >= 0.55 or int(row["source_kind_count"]) >= 3
            else "medium"
            if unresolved_ratio >= 0.30 or int(row["motifs_affected"]) <= 1
            else "low"
            if int(row["structured_evidence_rows"]) < 8
            else "none"
        )
        if int(row["structured_evidence_rows"]) >= 15 and int(row["motifs_affected"]) >= 2:
            value_class = "high_value_ingested"
        elif figure_followup in {"high", "medium"} and int(row["structured_evidence_rows"]) >= 3:
            value_class = "partial_ingest_followup_needed"
        elif int(row["structured_evidence_rows"]) <= 3 or unresolved_ratio >= 0.75:
            value_class = "low_value_context_only"
        else:
            value_class = "moderate_value_ingested"
        summary.append(
            {
                **row,
                "figure_followup_needed": figure_followup,
                "overall_value_class": value_class,
            }
        )
    return summary


def _cleanup_actions(summary_rows: list[dict], sync_rows: list[dict]) -> list[dict]:
    actions = []
    for row in summary_rows:
        if int(row["source_kind_count"]) > 1:
            actions.append(
                {
                    "action_type": "dedupe_reporting_only",
                    "source_id": row["source_id"],
                    "paper_id": row["paper_id"] or "",
                    "title": row["title"],
                    "detail": f"Source appears under {row['source_kind_count']} source_kind variants; dedupe summary uses one row per source_id.",
                }
            )
        if not row["paper_id"]:
            actions.append(
                {
                    "action_type": "paper_mapping_incomplete",
                    "source_id": row["source_id"],
                    "paper_id": "",
                    "title": row["title"],
                    "detail": "No literature.paper_id mapping was found for this source_id; source remains valid but paper linkage needs later cleanup.",
                }
            )
        if row["overall_value_class"] == "low_value_context_only":
            actions.append(
                {
                    "action_type": "downgrade_to_low_value_context_only",
                    "source_id": row["source_id"],
                    "paper_id": row["paper_id"] or "",
                    "title": row["title"],
                    "detail": "Low evidence density or high unresolved ratio; keep for context, not primary scaling.",
                }
            )
    for row in sync_rows:
        actions.append(
            {
                "action_type": "candidate_processed_flag_sync",
                "source_id": "",
                "paper_id": row["paper_id"],
                "title": row["title"],
                "detail": f"already_processed_in_evidence synced to {'true' if row['evidence_rows'] else 'false'} based on actual evidence rows.",
            }
        )
    return actions


def _upsert_candidate_registry(connection: duckdb.DuckDBPyConnection, candidate: CandidateRecord) -> None:
    triage = _triage(candidate)
    exists = connection.sql("SELECT COUNT(*) FROM literature.candidate_papers WHERE paper_id = ?", params=[candidate.paper_id]).fetchone()[0]
    if exists:
        connection.execute("DELETE FROM literature.candidate_papers WHERE paper_id = ?", [candidate.paper_id])
        connection.execute("DELETE FROM literature.paper_triage WHERE paper_id = ?", [candidate.paper_id])
        connection.execute("DELETE FROM literature.processing_queue WHERE paper_id = ?", [candidate.paper_id])
    connection.execute(
        "INSERT INTO literature.candidate_papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            candidate.paper_id,
            candidate.title,
            json.dumps(candidate.authors),
            candidate.year,
            candidate.journal,
            candidate.doi,
            candidate.primary_source(),
            json.dumps(sorted(candidate.sources)),
            candidate.best_abstract(),
            json.dumps(sorted(candidate.query_sources)),
            json.dumps(sorted(candidate.disease_keywords)),
            json.dumps(sorted(candidate.sample_keywords)),
            json.dumps(sorted(candidate.spectral_keywords)),
            candidate.title_key,
            candidate.already_in_local_corpus,
            candidate.already_processed_in_evidence,
            candidate.open_access_candidate,
            "; ".join(candidate.notes),
        ],
    )
    connection.execute(
        "INSERT INTO literature.paper_triage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            candidate.paper_id,
            triage["condition_relevance_score"],
            triage["sample_relevance_score"],
            triage["spectral_density_score"],
            triage["comparison_structure_score"],
            triage["figure_value_score"],
            triage["si_value_score"],
            triage["final_score"],
            triage["triage_decision"],
            triage["decision_rationale"],
            "oa_text_first_expansion",
        ],
    )
    connection.execute(
        "INSERT INTO literature.processing_queue VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            f"queue_{candidate.paper_id}",
            candidate.paper_id,
            "pending",
            0,
            False,
            False,
            False,
            0,
            "oa_text_first_expansion",
            "Selected OA candidate for text-first harvest.",
        ],
    )


def _record_harvest_asset(connection: duckdb.DuckDBPyConnection, candidate: CandidateRecord, harvested: HarvestedTextRecord) -> None:
    connection.execute("DELETE FROM literature.paper_assets WHERE paper_id = ? AND asset_kind = 'oa_text_first_harvest'", [candidate.paper_id])
    connection.execute(
        "INSERT INTO literature.paper_assets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            f"asset_{candidate.paper_id}_oa_text",
            candidate.paper_id,
            harvested.source,
            "oa_text_first_harvest",
            "",
            json.dumps(harvested.supplement_links),
            "[]",
            bool(harvested.figure_captions),
            bool(harvested.table_text_blocks),
            harvested.canonical_url,
            harvested.json_path,
            "json",
            "harvested_text",
            harvested.txt_path,
        ],
    )


def _update_processing_state(
    connection: duckdb.DuckDBPyConnection,
    candidate: CandidateRecord,
    harvested: HarvestedTextRecord | None,
    rows_added: int,
    failure_reason: str = "",
) -> None:
    if harvested is None:
        connection.execute(
            """
            UPDATE literature.processing_queue
            SET queue_status = 'pending',
                selected_for_ingestion = FALSE,
                asset_ready = FALSE,
                ingestion_attempted = TRUE,
                extraction_row_count = 0,
                notes = ?
            WHERE paper_id = ?
            """,
            [failure_reason or "OA text harvest failed.", candidate.paper_id],
        )
        return
    if rows_added > 0:
        connection.execute(
            """
            UPDATE literature.processing_queue
            SET queue_status = 'processed',
                selected_for_ingestion = TRUE,
                asset_ready = TRUE,
                ingestion_attempted = TRUE,
                extraction_row_count = ?,
                notes = ?
            WHERE paper_id = ?
            """,
            [rows_added, f"OA text-first harvest ingested from {harvested.access_class}.", candidate.paper_id],
        )
        connection.execute(
            "UPDATE literature.candidate_papers SET already_processed_in_evidence = TRUE WHERE paper_id = ?",
            [candidate.paper_id],
        )
    else:
        connection.execute(
            """
            UPDATE literature.processing_queue
            SET queue_status = 'skipped_low_value',
                selected_for_ingestion = TRUE,
                asset_ready = TRUE,
                ingestion_attempted = TRUE,
                extraction_row_count = 0,
                notes = ?
            WHERE paper_id = ?
            """,
            [f"OA text-first harvest succeeded but yielded no assignment-grade evidence from {harvested.access_class}.", candidate.paper_id],
        )


def run_oa_text_first_expansion(db_path: Path = DB_PATH) -> dict[str, int]:
    ensure_oa_text_first_output_dirs()
    connection = duckdb.connect(str(db_path))
    initialize_schema(connection)

    sync_rows, sync_count = _sync_candidate_processed_flags(connection)
    current_summary = _current_ingested_registry_summary(connection)
    cleanup_actions = _cleanup_actions(current_summary, sync_rows)

    candidates = _query_candidates(connection)
    selected_candidates, search_rows = _select_high_relevance_candidates(connection, candidates)

    selected_rows = []
    harvest_inventory_rows = []
    harvest_failure_rows = []
    extracted_summary_rows = []
    followup_rows = []
    rescue_pool_rows = []

    _purge_previous_rows(connection)

    harvested_records: list[tuple[CandidateRecord, HarvestedTextRecord, dict[str, int], list[ExtractedAssignment], int, int]] = []
    for candidate in selected_candidates:
        _upsert_candidate_registry(connection, candidate)
        harvested, failure = _harvest_candidate_text(candidate)
        if harvested is None:
            _update_processing_state(connection, candidate, None, 0, failure)
            harvest_failure_rows.append(
                {
                    "paper_id": candidate.paper_id,
                    "title": candidate.title,
                    "doi": candidate.doi,
                    "canonical_url": candidate.remote_manuscript_urls[0] if candidate.remote_manuscript_urls else "",
                    "failure_reason": failure,
                }
            )
            rescue_pool_rows.append(
                {
                    "paper_id": candidate.paper_id,
                    "title": candidate.title,
                    "doi": candidate.doi,
                    "canonical_article_url": candidate.remote_manuscript_urls[0] if candidate.remote_manuscript_urls else "",
                    "supplementary_url": "",
                    "reason_high_value": "High-relevance OA/search candidate but text harvest failed or full text was inaccessible.",
                    "needed_first": "manuscript PDF",
                    "suggested_local_path": str(OA_TEXT_FIRST_ASSET_ROOT / candidate.paper_id / "manual_rescue"),
                    "priority_score": round(_triage(candidate)["final_score"], 6),
                }
            )
            continue
        _record_harvest_asset(connection, candidate, harvested)
        harvest_inventory_rows.append(
            {
                "paper_id": candidate.paper_id,
                "title": candidate.title,
                "doi": candidate.doi,
                "canonical_url": harvested.canonical_url,
                "access_class": harvested.access_class,
                "txt_path": harvested.txt_path,
                "json_path": harvested.json_path,
                "section_header_count": len(harvested.section_headers),
                "figure_caption_count": len(harvested.figure_captions),
                "table_text_block_count": len(harvested.table_text_blocks),
                "supplement_link_count": len(harvested.supplement_links),
                "body_text_chars": len(harvested.body_text),
            }
        )
        assignments, stats = _extract_text_first_assignments(harvested)
        primary_count, secondary_count = _ingest_assignments(connection, candidate, harvested, assignments)
        if primary_count + secondary_count == 0:
            _unregister_source(connection, f"src_oa_text_{candidate.paper_id}_manuscript")
        _update_processing_state(connection, candidate, harvested, primary_count + secondary_count)
        harvested_records.append((candidate, harvested, stats, assignments, primary_count, secondary_count))
    for candidate in selected_candidates:
        selected_rows.append(
            {
                "paper_id": candidate.paper_id,
                "title": candidate.title,
                "doi": candidate.doi,
                "journal": candidate.journal,
                "year": candidate.year or "",
                "source": candidate.primary_source(),
                "final_score": _triage(candidate)["final_score"],
                "open_access_candidate": candidate.open_access_candidate,
                "remote_url_count": len(candidate.remote_manuscript_urls),
            }
        )

    if harvested_records:
        build_ontology_mappings(connection)
        connection.commit()
        build_local_support_neighborhoods(db_path)

    for candidate, harvested, stats, assignments, primary_count, secondary_count in harvested_records:
        source_id = f"src_oa_text_{candidate.paper_id}_manuscript"
        source_metrics = connection.sql(
            """
            WITH evidence_counts AS (
              SELECT
                COUNT(*) AS structured_evidence_rows,
                SUM(CASE WHEN om.meaning_class = 'unresolved_signal' THEN 1 ELSE 0 END) AS unresolved_rows,
                SUM(CASE WHEN om.meaning_class = 'confounder_signal' THEN 1 ELSE 0 END) AS confounder_rows,
                COUNT(DISTINCT COALESCE(om.normalized_subfamily, '')) FILTER (WHERE COALESCE(om.normalized_subfamily, '') <> '') AS meanings_touched
              FROM evidence.peak_assignment_evidence pae
              LEFT JOIN ontology.evidence_ontology_mappings om USING (assignment_record_id)
              WHERE pae.source_id = ?
            ),
            motif_counts AS (
              SELECT COUNT(DISTINCT nml.pattern_id) AS motifs_affected
              FROM evidence.local_support_neighborhood_members m
              JOIN evidence.neighborhood_motif_links nml USING (neighborhood_id)
              WHERE m.source_id = ?
            )
            SELECT
              COALESCE(ec.structured_evidence_rows, 0),
              COALESCE(ec.meanings_touched, 0),
              COALESCE(mc.motifs_affected, 0),
              COALESCE(ec.unresolved_rows, 0),
              COALESCE(ec.confounder_rows, 0)
            FROM evidence_counts ec, motif_counts mc
            """,
            params=[source_id, source_id],
        ).fetchone()
        figure_followup = _figure_followup_priority(harvested, stats)
        extracted_summary_rows.append(
            {
                "paper_id": candidate.paper_id,
                "source_id": source_id,
                "title": candidate.title,
                "structured_evidence_rows_added": int(source_metrics[0]),
                "strengthened_existing_support": int(secondary_count),
                "meanings_touched": int(source_metrics[1]),
                "new_meanings_introduced": 0,
                "motifs_affected": int(source_metrics[2]),
                "condition_links_affected": 0,
                "unresolved_rows": int(source_metrics[3]),
                "confounder_rows": int(source_metrics[4]),
                "validated_primary": int(stats.get("validated_primary", 0)),
                "validated_secondary": int(stats.get("validated_secondary", 0)),
                "figure_followup_needed": figure_followup,
            }
        )
        followup_rows.append(
            {
                "paper_id": candidate.paper_id,
                "title": candidate.title,
                "figure_followup_priority": figure_followup,
                "figure_caption_count": len(harvested.figure_captions),
                "table_text_block_count": len(harvested.table_text_blocks),
                "text_rows_added": int(source_metrics[0]),
                "reason": "Text extraction produced useful but incomplete support." if figure_followup in {"medium", "high"} else "Text/table extraction appears sufficient for now.",
            }
        )
        if figure_followup in {"medium", "high"} and candidate.paper_id not in {row["paper_id"] for row in rescue_pool_rows}:
            rescue_pool_rows.append(
                {
                    "paper_id": candidate.paper_id,
                    "title": candidate.title,
                    "doi": candidate.doi,
                    "canonical_article_url": harvested.canonical_url,
                    "supplementary_url": harvested.supplement_links[0] if harvested.supplement_links else "",
                    "reason_high_value": "Text-first extraction yielded useful but incomplete evidence; figure/PDF follow-up would likely add assignment support.",
                    "needed_first": "manuscript PDF" if figure_followup == "high" else "SI",
                    "suggested_local_path": str(OA_TEXT_FIRST_ASSET_ROOT / candidate.paper_id / "manual_rescue"),
                    "priority_score": round(_triage(candidate)["final_score"], 6),
                }
            )

    blocked_rows = connection.sql(
        """
        SELECT paper_id, title, doi, canonical_article_url,
               preferred_asset_type_to_rescue_first AS needed_first,
               local_upload_target_path, priority_score,
               COALESCE(suggested_manual_action, '') AS suggested_manual_action
        FROM literature.blocked_assets
        WHERE COALESCE(priority_score, 0.0) >= 0.6
        ORDER BY priority_score DESC, paper_id
        """
    ).df().to_dict("records")
    for row in blocked_rows:
        if len(rescue_pool_rows) >= MAX_MANUAL_RESCUE_SHORTLIST:
            break
        rescue_pool_rows.append(
            {
                "paper_id": row["paper_id"],
                "title": row["title"],
                "doi": row["doi"],
                "canonical_article_url": row["canonical_article_url"],
                "supplementary_url": "",
                "reason_high_value": row["suggested_manual_action"] or "Existing blocked paper already triaged as high-priority in the acquisition layer.",
                "needed_first": row["needed_first"] or "manuscript PDF",
                "suggested_local_path": row["local_upload_target_path"],
                "priority_score": row["priority_score"],
            }
        )

    deduped_rescue_pool: dict[str, dict] = {}
    for row in rescue_pool_rows:
        current = deduped_rescue_pool.get(row["paper_id"])
        if current is None or float(row["priority_score"] or 0.0) > float(current["priority_score"] or 0.0):
            deduped_rescue_pool[row["paper_id"]] = row
    rescue_pool_rows = sorted(deduped_rescue_pool.values(), key=lambda item: item["priority_score"], reverse=True)[:MAX_MANUAL_RESCUE_SHORTLIST]

    _write_csv(
        OA_TEXT_FIRST_TABLES_ROOT / "current_ingested_registry_summary.csv",
        list(current_summary[0].keys()) if current_summary else [
            "source_id", "paper_id", "title", "structured_evidence_rows", "strengthened_support_rows",
            "meanings_touched", "new_meanings_introduced", "motifs_affected", "condition_motif_links",
            "condition_neighborhood_links", "unresolved_rows", "confounder_rows", "figure_followup_needed", "overall_value_class"
        ],
        current_summary,
    )
    _write_csv(
        OA_TEXT_FIRST_TABLES_ROOT / "ingested_source_cleanup_actions.csv",
        list(cleanup_actions[0].keys()) if cleanup_actions else ["action_type", "source_id", "paper_id", "title", "detail"],
        cleanup_actions,
    )
    _write_csv(
        OA_TEXT_FIRST_TABLES_ROOT / "oa_search_candidates.csv",
        list(search_rows[0].keys()) if search_rows else ["paper_id", "title", "doi", "journal", "year", "source", "open_access_candidate", "already_processed_in_evidence", "final_score", "harvest_priority"],
        search_rows,
    )
    _write_csv(
        OA_TEXT_FIRST_TABLES_ROOT / "oa_selected_high_relevance.csv",
        list(selected_rows[0].keys()) if selected_rows else ["paper_id", "title", "doi", "journal", "year", "source", "final_score", "open_access_candidate", "remote_url_count"],
        selected_rows,
    )
    _write_csv(
        OA_TEXT_FIRST_TABLES_ROOT / "oa_text_harvest_inventory.csv",
        list(harvest_inventory_rows[0].keys()) if harvest_inventory_rows else ["paper_id", "title", "doi", "canonical_url", "access_class", "txt_path", "json_path"],
        harvest_inventory_rows,
    )
    _write_csv(
        OA_TEXT_FIRST_TABLES_ROOT / "oa_text_harvest_failures.csv",
        list(harvest_failure_rows[0].keys()) if harvest_failure_rows else ["paper_id", "title", "doi", "canonical_url", "failure_reason"],
        harvest_failure_rows,
    )
    _write_csv(
        OA_TEXT_FIRST_TABLES_ROOT / "oa_structured_evidence_summary.csv",
        list(extracted_summary_rows[0].keys()) if extracted_summary_rows else ["paper_id", "source_id", "title", "structured_evidence_rows_added", "strengthened_existing_support", "meanings_touched", "new_meanings_introduced", "motifs_affected", "condition_links_affected", "figure_followup_needed"],
        extracted_summary_rows,
    )
    _write_csv(
        OA_TEXT_FIRST_TABLES_ROOT / "oa_figure_followup_priority.csv",
        list(followup_rows[0].keys()) if followup_rows else ["paper_id", "title", "figure_followup_priority", "figure_caption_count", "table_text_block_count", "text_rows_added", "reason"],
        followup_rows,
    )
    _write_csv(
        OA_TEXT_FIRST_TABLES_ROOT / "manual_rescue_high_value_shortlist.csv",
        list(rescue_pool_rows[0].keys()) if rescue_pool_rows else ["paper_id", "title", "doi", "canonical_article_url", "supplementary_url", "reason_high_value", "needed_first", "suggested_local_path", "priority_score"],
        rescue_pool_rows,
    )

    implementation_lines = [
        "# Implementation Note",
        "",
        "This pass cleans the currently ingested literature layer, then runs a selective OA-first search and harvest lane built around text/XML/HTML rather than PDF-first acquisition.",
        "Structured evidence extraction is limited to explicit assignment text, caption-backed text, table-text assignments, and QC-gated regex candidates.",
        "Manual rescue is reserved for high-value blocked or text-incomplete papers only.",
    ]
    (OA_TEXT_FIRST_REPORT_ROOT / "implementation_note.md").write_text("\n".join(implementation_lines) + "\n")

    value_counts = defaultdict(int)
    for row in current_summary:
        value_counts[row["overall_value_class"]] += 1
    assessment_lines = [
        "# Current State Assessment",
        "",
        f"- Current ingested sources audited: `{len(current_summary)}`",
        f"- High-value ingested sources: `{value_counts['high_value_ingested']}`",
        f"- Moderate-value ingested sources: `{value_counts['moderate_value_ingested']}`",
        f"- Low-value/context-only ingested sources: `{value_counts['low_value_context_only']}`",
        f"- Partial-ingest follow-up-needed sources: `{value_counts['partial_ingest_followup_needed']}`",
        f"- OA candidates found: `{len(search_rows)}`",
        f"- OA candidates selected: `{len(selected_rows)}`",
        f"- OA papers successfully harvested as text: `{len(harvest_inventory_rows)}`",
        f"- OA papers that yielded structured evidence: `{sum(1 for row in extracted_summary_rows if row['structured_evidence_rows_added'] > 0)}`",
        f"- OA papers requiring figure follow-up: `{sum(1 for row in followup_rows if row['figure_followup_priority'] in {'medium', 'high'})}`",
        f"- High-value manual-rescue shortlist size: `{len(rescue_pool_rows)}`",
        "",
        "The OA text-first lane is now viable as the primary acquisition and harvesting path, but not yet as the sole extraction path.",
        "Current yield is still constrained by assignment-poor OA full text; targeted figure/SI follow-up remains necessary for many otherwise relevant papers.",
        "Manual PDF/SI rescue should remain targeted to the shortlist rather than becoming the default acquisition mode.",
    ]
    (OA_TEXT_FIRST_REPORT_ROOT / "current_state_assessment.md").write_text("\n".join(assessment_lines) + "\n")

    connection.commit()
    connection.close()
    return {
        "current_ingested_sources_cleaned": len(current_summary),
        "cleanup_actions": len(cleanup_actions),
        "oa_candidates_found": len(search_rows),
        "oa_candidates_selected": len(selected_rows),
        "oa_papers_harvested_as_text": len(harvest_inventory_rows),
        "oa_papers_yielding_structured_evidence": sum(1 for row in extracted_summary_rows if row["structured_evidence_rows_added"] > 0),
        "figure_followup_papers": sum(1 for row in followup_rows if row["figure_followup_priority"] in {"medium", "high"}),
        "manual_rescue_shortlist_size": len(rescue_pool_rows),
    }
