from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import duckdb
import requests

from gaira.evidence_v1.constants import (
    DB_PATH,
    LITERATURE_PIPELINE_ASSET_ROOT,
    LITERATURE_PIPELINE_REPORT_ROOT,
    LITERATURE_PIPELINE_TABLES_ROOT,
    ensure_literature_pipeline_output_dirs,
)
from gaira.evidence_v1.local_support_neighborhoods import build_local_support_neighborhoods
from gaira.evidence_v1.ontology_expansion import build_ontology_mappings
from gaira.evidence_v1.schema import initialize_schema, reset_literature_tables


CROSSREF_URL = "https://api.crossref.org/works"
EUROPEPMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ZENODO_URL = "https://zenodo.org/api/records"
FIGSHARE_URL = "https://api.figshare.com/v2/articles"

REQUEST_TIMEOUT = 25
SEARCH_ROWS = 8
TOP_ASSET_DISCOVERY_LIMIT = 25
TOP_INGEST_LIMIT = 6
INGEST_PREFIX = "litacq_v1"
SOURCE_KIND = "literature_acquisition_pipeline_v1"
CREATED_BY = "literature_acquisition_pipeline_v1"

PAPER_CORPUS_AUDIT_PATH = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/"
    "gaira_remaining_paper_controlled_ingest_v1/tables/paper_corpus_audit.csv"
)
MANUSCRIPT_ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/gaira_literature_corpus/manuscripts")
SUPPLEMENTARY_ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/gaira_literature_corpus/supplementary")

HEADERS = {
    "User-Agent": "GAIRA-literature-acquisition/1.0 (scientific evidence acquisition pipeline)",
    "Accept": "application/json, text/plain, */*",
}

QUERY_FAMILIES = {
    "A_disease_sample_raman": [
        "serum SERS cholangiocarcinoma",
        "EV Raman lung cancer",
        "plasma SERS diabetes",
        "serum Raman hepatocellular carcinoma",
        "exosome SERS hepatotoxicity",
    ],
    "B_condition_control": [
        "Raman healthy control cancer serum",
        "SERS case control plasma disease",
        "EV Raman healthy control tumor",
    ],
    "C_perturbation_stress": [
        "Raman acetaminophen hepatotoxicity EV",
        "SERS oxidative stress serum",
        "Raman drug induced liver injury exosome",
    ],
    "D_review_low_priority": [
        "Raman serum cancer review",
        "SERS extracellular vesicle review",
    ],
}

DISEASE_KEYWORDS = {
    "cancer",
    "carcinoma",
    "tumor",
    "hcc",
    "cca",
    "hepatotoxicity",
    "dili",
    "inflammation",
    "infection",
    "sepsis",
    "diabetes",
    "metabolic",
    "obesity",
    "fibrosis",
    "liver disease",
    "oxidative stress",
    "injury",
    "healthy",
    "control",
    "benign",
    "stage",
    "severity",
    "progression",
    "treatment",
    "sjogrens",
    "nephropathy",
}
SAMPLE_KEYWORDS = {
    "serum",
    "plasma",
    "extracellular vesicle",
    "extracellular vesicles",
    "ev",
    "exosome",
    "exosomes",
    "saliva",
    "urine",
    "blood",
    "tissue",
}
SPECTRAL_KEYWORDS = {
    "raman",
    "sers",
    "peak",
    "band",
    "assignment",
    "assignments",
    "spectra",
    "spectrum",
    "biomarker",
    "figure",
    "table",
    "supplementary",
    "source data",
}
COMPARISON_KEYWORDS = {
    "case control",
    "case-control",
    "disease vs healthy",
    "healthy control",
    "control",
    "versus",
    "vs.",
    "vs ",
    "stage",
    "severity",
    "longitudinal",
    "dose response",
    "dose-response",
    "perturbation",
}
LOW_VALUE_TERMS = {
    "review",
    "overview",
    "baseline removal",
    "preprocessing",
    "machine learning review",
    "method",
    "benchmark",
    "protocol",
}
ASSIGNMENT_VERBS = ("assigned to", "attributed to", "represents", "represent", "corresponds to", "associated with")
NOISE_TERMS = ("doi", "copyright", "creativecommons", "license", "figure", "fig.", "table", "http", "www.", "et al")
TENTATIVE_TERMS = ("tentative", "possibly", "possible", "may be", "might be", "likely", "putative", "region", "broad")
SUBFAMILY_HINT_TERMS = (
    "amide",
    "protein",
    "lipid",
    "dna",
    "rna",
    "nucleic",
    "phenylalanine",
    "tyrosine",
    "tryptophan",
    "glycogen",
    "carbohydrate",
    "glycan",
    "carotenoid",
    "adenine",
    "guanine",
    "cytosine",
    "thymine",
    "citrate",
)


@dataclass
class CandidateRecord:
    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    journal: str
    doi: str
    sources: set[str] = field(default_factory=set)
    abstracts: list[str] = field(default_factory=list)
    query_sources: set[str] = field(default_factory=set)
    title_key: str = ""
    disease_keywords: set[str] = field(default_factory=set)
    sample_keywords: set[str] = field(default_factory=set)
    spectral_keywords: set[str] = field(default_factory=set)
    open_access_candidate: bool = False
    local_manuscript_path: str = ""
    supplementary_files: list[str] = field(default_factory=list)
    remote_manuscript_urls: list[str] = field(default_factory=list)
    source_data_links: list[str] = field(default_factory=list)
    figures_detected: bool = False
    tables_detected: bool = False
    already_in_local_corpus: bool = False
    already_processed_in_evidence: bool = False
    notes: list[str] = field(default_factory=list)

    def primary_source(self) -> str:
        source_priority = ["europepmc", "pubmed", "crossref"]
        for item in source_priority:
            if item in self.sources:
                return item
        return sorted(self.sources)[0] if self.sources else "unknown"

    def best_abstract(self) -> str:
        for text in self.abstracts:
            if text and text.strip():
                return text.strip()
        return ""


@dataclass
class ExtractedAssignment:
    paper_id: str
    source_id: str
    assignment_record_id: str
    extraction_method: str
    classification: str
    peak_center_cm: float
    peak_min_cm: float
    peak_max_cm: float
    assigned_molecule: str
    assigned_group_or_theme: str
    original_text: str
    figure_reference: str
    manuscript_or_si: str
    confidence_label: str
    notes: str
    classification_rationale: str = ""


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


def _keyword_hits(text: str, keywords: Iterable[str]) -> set[str]:
    lowered = text.lower()
    hits = set()
    for keyword in keywords:
        if len(keyword) <= 3 and keyword.isalpha():
            if re.search(rf"\b{re.escape(keyword)}\b", lowered):
                hits.add(keyword)
        elif keyword in lowered:
            hits.add(keyword)
    return hits


def _request_json(url: str, params: dict[str, object] | None = None) -> dict:
    response = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _crossref_search(query: str, query_source: str) -> list[dict]:
    payload = _request_json(
        CROSSREF_URL,
        {
            "query.bibliographic": query,
            "rows": SEARCH_ROWS,
            "sort": "relevance",
        },
    )
    results = []
    for item in payload.get("message", {}).get("items", []):
        title = " ".join(item.get("title", [])) if item.get("title") else ""
        authors = []
        for author in item.get("author", [])[:8]:
            name = " ".join(part for part in [author.get("given", ""), author.get("family", "")] if part).strip()
            if name:
                authors.append(name)
        year = None
        date_parts = item.get("published-print", item.get("published-online", item.get("issued", {}))).get("date-parts", [])
        if date_parts and date_parts[0] and date_parts[0][0] is not None:
            year = int(date_parts[0][0])
        results.append(
            {
                "title": title,
                "authors": authors,
                "year": year,
                "journal": " ".join(item.get("container-title", [])) if item.get("container-title") else "",
                "doi": _normalize_doi(item.get("DOI", "")),
                "abstract": re.sub(r"<[^>]+>", " ", item.get("abstract", "") or "").strip(),
                "source": "crossref",
                "query_source": query_source,
                "open_access_candidate": any(link.get("content-type") == "application/pdf" for link in item.get("link", []) or []),
                "remote_urls": [link.get("URL", "") for link in item.get("link", []) or [] if link.get("URL")],
            }
        )
    return results


def _europepmc_search(query: str, query_source: str) -> list[dict]:
    payload = _request_json(
        EUROPEPMC_URL,
        {
            "query": query,
            "format": "json",
            "pageSize": SEARCH_ROWS,
            "resultType": "core",
        },
    )
    results = []
    for item in payload.get("resultList", {}).get("result", []):
        authors = [part.strip() for part in (item.get("authorString") or "").split(",") if part.strip()]
        doi = _normalize_doi(item.get("doi", ""))
        remote_urls = []
        pmcid = item.get("pmcid") or ""
        if pmcid:
            remote_urls.append(f"https://europepmc.org/articles/{pmcid}?pdf=render")
        results.append(
            {
                "title": item.get("title", "") or "",
                "authors": authors[:8],
                "year": int(item["pubYear"]) if item.get("pubYear") else None,
                "journal": item.get("journalTitle", "") or "",
                "doi": doi,
                "abstract": item.get("abstractText", "") or "",
                "source": "europepmc",
                "query_source": query_source,
                "open_access_candidate": str(item.get("isOpenAccess", "")).lower() == "y" or bool(pmcid),
                "remote_urls": remote_urls,
            }
        )
    return results


def _pubmed_search(query: str, query_source: str) -> list[dict]:
    search_payload = _request_json(
        PUBMED_ESEARCH_URL,
        {"db": "pubmed", "retmode": "json", "retmax": SEARCH_ROWS, "term": query},
    )
    ids = search_payload.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    summary_payload = _request_json(
        PUBMED_ESUMMARY_URL,
        {"db": "pubmed", "retmode": "json", "id": ",".join(ids)},
    )
    abstract_xml = requests.get(
        PUBMED_EFETCH_URL,
        params={"db": "pubmed", "retmode": "xml", "id": ",".join(ids)},
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    abstract_xml.raise_for_status()
    abstract_text = abstract_xml.text
    abstract_lookup: dict[str, str] = {}
    for match in re.finditer(r"<PubmedArticle>.*?<PMID[^>]*>(\d+)</PMID>.*?<Abstract>(.*?)</Abstract>", abstract_text, re.S):
        pmid = match.group(1)
        abstract = re.sub(r"<[^>]+>", " ", match.group(2))
        abstract_lookup[pmid] = re.sub(r"\s+", " ", abstract).strip()

    results = []
    for pmid in ids:
        item = summary_payload.get("result", {}).get(pmid, {})
        authors = []
        for author in item.get("authors", [])[:8]:
            if author.get("name"):
                authors.append(author["name"])
        article_ids = item.get("articleids", []) or []
        doi = ""
        for article_id in article_ids:
            if article_id.get("idtype") == "doi":
                doi = _normalize_doi(article_id.get("value", ""))
                break
        results.append(
            {
                "title": item.get("title", "") or "",
                "authors": authors,
                "year": int((item.get("pubdate") or "")[:4]) if (item.get("pubdate") or "")[:4].isdigit() else None,
                "journal": item.get("fulljournalname", "") or "",
                "doi": doi,
                "abstract": abstract_lookup.get(pmid, ""),
                "source": "pubmed",
                "query_source": query_source,
                "open_access_candidate": any(article_id.get("idtype") == "pmc" for article_id in article_ids),
                "remote_urls": [],
            }
        )
    return results


def _search_zenodo(query: str) -> list[dict]:
    try:
        payload = _request_json(ZENODO_URL, {"q": query, "page": 1, "size": 5})
    except Exception:
        return []
    hits = payload.get("hits", {}).get("hits", [])
    rows = []
    for item in hits:
        metadata = item.get("metadata", {})
        rows.append(
            {
                "title": metadata.get("title", "") or "",
                "doi": _normalize_doi(metadata.get("doi", "")),
                "record_url": item.get("links", {}).get("html", ""),
                "files": [file_item.get("links", {}).get("self", "") for file_item in item.get("files", []) if file_item.get("links", {}).get("self")],
            }
        )
    return rows


def _search_figshare(query: str) -> list[dict]:
    try:
        payload = _request_json(
            FIGSHARE_URL,
            {"search_for": query, "page": 1, "page_size": 5, "order_direction": "desc"},
        )
    except Exception:
        return []
    rows = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "title": item.get("title", "") or "",
                "doi": _normalize_doi(item.get("doi", "")),
                "record_url": item.get("url_public_html", "") or "",
                "files": [],
            }
        )
    return rows


def _dedupe_records(records: list[dict]) -> list[CandidateRecord]:
    deduped: list[CandidateRecord] = []
    for record in records:
        title = (record.get("title") or "").strip()
        if not title:
            continue
        doi = _normalize_doi(record.get("doi", ""))
        title_key = _title_key(title)
        lowered_title = title.lower()
        if (
            lowered_title.startswith("figure ")
            or lowered_title.startswith("table ")
            or lowered_title.startswith("review for ")
            or lowered_title.startswith("supplementary")
            or "/fig-" in doi
            or "/review" in doi
            or doi.endswith(".s001")
            or doi.endswith(".s002")
            or doi.endswith(".s003")
        ):
            continue
        matched = None
        if doi:
            for candidate in deduped:
                if candidate.doi and candidate.doi == doi:
                    matched = candidate
                    break
        if matched is None:
            for candidate in deduped:
                if title_key == candidate.title_key:
                    matched = candidate
                    break
                if SequenceMatcher(None, title_key, candidate.title_key).ratio() >= 0.93:
                    matched = candidate
                    break
        if matched is None:
            candidate = CandidateRecord(
                paper_id=f"paper_{len(deduped)+1:04d}",
                title=title,
                authors=list(record.get("authors") or []),
                year=record.get("year"),
                journal=record.get("journal") or "",
                doi=doi,
                title_key=title_key,
            )
            deduped.append(candidate)
            matched = candidate
        matched.sources.add(record.get("source", "unknown"))
        matched.query_sources.add(record.get("query_source", ""))
        if record.get("abstract"):
            matched.abstracts.append(record["abstract"])
        if not matched.authors and record.get("authors"):
            matched.authors = list(record["authors"])
        if not matched.journal and record.get("journal"):
            matched.journal = record["journal"]
        if matched.year is None and record.get("year"):
            matched.year = record["year"]
        if not matched.doi and doi:
            matched.doi = doi
        matched.open_access_candidate = matched.open_access_candidate or bool(record.get("open_access_candidate"))
        for url in record.get("remote_urls") or []:
            if url and url not in matched.remote_manuscript_urls:
                matched.remote_manuscript_urls.append(url)
    return deduped


def _load_local_corpus_map() -> dict[str, dict]:
    if not PAPER_CORPUS_AUDIT_PATH.exists():
        return {}
    with PAPER_CORPUS_AUDIT_PATH.open() as handle:
        rows = list(csv.DictReader(handle))
    mapping = {}
    for row in rows:
        basename = row.get("basename", "")
        candidate_keys = {
            _title_key(Path(basename).stem),
            _title_key(row.get("title", "")),
            _title_key(row.get("source_id", "")),
        }
        for key in candidate_keys:
            if key:
                mapping[key] = row
    return mapping


def _match_local_corpus(candidate: CandidateRecord, local_map: dict[str, dict]) -> None:
    best_score = 0.0
    best_row = None
    for key, row in local_map.items():
        score = SequenceMatcher(None, candidate.title_key, key).ratio()
        if score > best_score:
            best_score = score
            best_row = row
    if best_row and best_score >= 0.62:
        candidate.already_in_local_corpus = True
        candidate.local_manuscript_path = str(MANUSCRIPT_ROOT / best_row["basename"])
        candidate.notes.append(f"matched_local_corpus:{best_row['basename']}")
        basename = Path(best_row["basename"]).stem.lower()
        supp_matches = []
        for path in SUPPLEMENTARY_ROOT.glob("*"):
            if not path.is_file() or path.name.startswith("._"):
                continue
            name = path.name.lower()
            author_hint = basename.split("_")[0]
            year_hint_match = re.search(r"_(20\d{2})", basename)
            year_hint = year_hint_match.group(1) if year_hint_match else ""
            if author_hint in name and (not year_hint or year_hint in name):
                supp_matches.append(str(path))
        candidate.supplementary_files.extend(sorted(set(supp_matches)))


def _existing_processed_titles(connection: duckdb.DuckDBPyConnection) -> list[str]:
    rows = connection.sql(
        """
        SELECT DISTINCT source_name
        FROM registry.warehouse_sources
        WHERE source_id LIKE 'src_%_manuscript'
        """
    ).fetchall()
    return [_title_key(row[0]) for row in rows if row[0]]


def _mark_existing_processing(connection: duckdb.DuckDBPyConnection, candidates: list[CandidateRecord]) -> None:
    processed_titles = _existing_processed_titles(connection)
    processed_source_ids = {
        row[0]
        for row in connection.sql(
            """
            SELECT DISTINCT source_id
            FROM evidence.peak_assignment_evidence
            WHERE source_id LIKE 'src_%_manuscript'
            """
        ).fetchall()
    }
    for candidate in candidates:
        if candidate.local_manuscript_path:
            basename = Path(candidate.local_manuscript_path).stem.lower()
            if any(SequenceMatcher(None, candidate.title_key, key).ratio() >= 0.68 for key in processed_titles):
                candidate.already_processed_in_evidence = True
                continue
            # direct basename mapping if possible
            row_key = _title_key(Path(candidate.local_manuscript_path).stem)
            if any(SequenceMatcher(None, row_key, key).ratio() >= 0.9 for key in processed_titles):
                candidate.already_processed_in_evidence = True
        if candidate.title_key in processed_titles:
            candidate.already_processed_in_evidence = True


def _enrich_candidate(candidate: CandidateRecord) -> None:
    text = f"{candidate.title} {candidate.best_abstract()}"
    candidate.disease_keywords = _keyword_hits(text, DISEASE_KEYWORDS)
    candidate.sample_keywords = _keyword_hits(text, SAMPLE_KEYWORDS)
    candidate.spectral_keywords = _keyword_hits(text, SPECTRAL_KEYWORDS)
    lowered = text.lower()
    candidate.figures_detected = "figure" in lowered or "fig." in lowered
    candidate.tables_detected = "table" in lowered


def _asset_discovery_for_candidate(candidate: CandidateRecord) -> None:
    search_text = candidate.doi or candidate.title
    for row in _search_zenodo(search_text):
        title_score = SequenceMatcher(None, candidate.title_key, _title_key(row["title"])).ratio()
        if candidate.doi and row["doi"] == candidate.doi or title_score >= 0.70:
            if row["record_url"]:
                candidate.source_data_links.append(row["record_url"])
            for file_url in row["files"]:
                if file_url not in candidate.source_data_links:
                    candidate.source_data_links.append(file_url)
    for row in _search_figshare(search_text):
        title_score = SequenceMatcher(None, candidate.title_key, _title_key(row["title"])).ratio()
        if candidate.doi and row["doi"] == candidate.doi or title_score >= 0.72:
            if row["record_url"]:
                candidate.source_data_links.append(row["record_url"])
    candidate.source_data_links = sorted(set(candidate.source_data_links))


def _score_condition(candidate: CandidateRecord) -> float:
    score = min(1.0, len(candidate.disease_keywords) / 4.0)
    title = candidate.title.lower()
    if any(term in title for term in ("cancer", "carcinoma", "diabetes", "hepatotoxicity", "injury", "syndrome", "nephropathy")):
        score += 0.2
    return min(1.0, score)


def _score_sample(candidate: CandidateRecord) -> float:
    score = min(1.0, len(candidate.sample_keywords) / 3.0)
    title = candidate.title.lower()
    if any(term in title for term in ("serum", "plasma", "ev", "exosome", "blood", "urine", "saliva")):
        score += 0.2
    return min(1.0, score)


def _score_spectral(candidate: CandidateRecord) -> float:
    score = min(1.0, len(candidate.spectral_keywords) / 5.0)
    if candidate.figures_detected:
        score += 0.15
    if candidate.tables_detected:
        score += 0.15
    if candidate.supplementary_files or candidate.source_data_links:
        score += 0.1
    return min(1.0, score)


def _score_comparison(candidate: CandidateRecord) -> float:
    text = f"{candidate.title} {candidate.best_abstract()}".lower()
    hits = sum(1 for keyword in COMPARISON_KEYWORDS if keyword in text)
    return min(1.0, hits / 3.0)


def _score_figure_value(candidate: CandidateRecord) -> float:
    score = 0.0
    if candidate.figures_detected:
        score += 0.4
    if any(keyword in candidate.spectral_keywords for keyword in {"peak", "band", "assignment", "figure"}):
        score += 0.3
    if candidate.already_in_local_corpus or candidate.remote_manuscript_urls:
        score += 0.2
    return min(1.0, score)


def _score_si_value(candidate: CandidateRecord) -> float:
    score = 0.0
    if candidate.supplementary_files:
        score += 0.6
    if candidate.source_data_links:
        score += 0.3
    if "supplementary" in candidate.best_abstract().lower():
        score += 0.1
    return min(1.0, score)


def _infer_sample_type(candidate: CandidateRecord) -> str:
    sample_hits = {term.lower() for term in candidate.sample_keywords}
    if "serum" in sample_hits:
        return "serum"
    if "plasma" in sample_hits:
        return "plasma"
    if "ev" in sample_hits or "extracellular vesicle" in sample_hits or "extracellular vesicles" in sample_hits:
        return "ev"
    if "exosome" in sample_hits or "exosomes" in sample_hits:
        return "ev"
    if "saliva" in sample_hits:
        return "saliva"
    if "urine" in sample_hits:
        return "urine"
    if "blood" in sample_hits:
        return "blood"
    if "tissue" in sample_hits:
        return "tissue"
    return ""


def _infer_modality(candidate: CandidateRecord) -> str:
    text = f"{candidate.title} {candidate.best_abstract()}".lower()
    if "sers" in text:
        return "sers"
    if "raman" in text:
        return "raman"
    return ""


def _infer_disease_class(candidate: CandidateRecord) -> str:
    text = f"{candidate.title} {candidate.best_abstract()}".lower()
    if any(term in text for term in ("cancer", "carcinoma", "tumor", "hcc", "cca", "benign")):
        return "cancer"
    if any(term in text for term in ("diabetes", "metabolic", "obesity", "nephropathy")):
        return "metabolic_disease"
    if any(term in text for term in ("hepatotoxicity", "dili", "drug induced", "injury")):
        return "drug_induced_toxicity"
    if any(term in text for term in ("infection", "sepsis", "inflammation", "sjogrens")):
        return "infection_inflammation"
    if "healthy" in text or "control" in text:
        return "healthy_control"
    return ""


def _infer_stress_class(candidate: CandidateRecord) -> str:
    text = f"{candidate.title} {candidate.best_abstract()}".lower()
    if "oxidative stress" in text:
        return "oxidative_stress"
    if "hepatotoxicity" in text or "drug induced" in text or "dili" in text:
        return "drug_perturbation"
    return ""


def _triage(candidate: CandidateRecord) -> dict:
    condition = _score_condition(candidate)
    sample = _score_sample(candidate)
    spectral = _score_spectral(candidate)
    comparison = _score_comparison(candidate)
    figure = _score_figure_value(candidate)
    si = _score_si_value(candidate)
    final = round((0.4 * condition) + (0.2 * sample) + (0.2 * spectral) + (0.2 * comparison), 6)
    lowered_title = candidate.title.lower()
    lowered_text = f"{candidate.title} {candidate.best_abstract()}".lower()
    low_value = any(term in lowered_title for term in LOW_VALUE_TERMS)
    review_like = any(term in lowered_text for term in ("systematic review", "meta-analysis", "narrative review", "review article"))
    asset_ready = bool(candidate.local_manuscript_path or candidate.remote_manuscript_urls)
    if candidate.already_processed_in_evidence:
        decision = "processed"
        rationale = "Candidate already exists in the structured evidence warehouse."
    elif low_value or review_like:
        decision = "skipped_low_value"
        rationale = "Review/methods-like candidate retained for discovery context only, not controlled evidence ingestion."
    elif final >= 0.74 and asset_ready:
        decision = "selected_for_ingestion"
        rationale = "High disease/sample/spectral score with available manuscript asset."
    elif final >= 0.55:
        decision = "pending"
        rationale = "Relevant candidate, but asset readiness or evidence density is not strong enough for automatic ingestion."
    else:
        decision = "skipped_low_value"
        rationale = "Insufficient disease/sample/comparison relevance for controlled ingestion."
    return {
        "condition_relevance_score": round(condition, 6),
        "sample_relevance_score": round(sample, 6),
        "spectral_density_score": round(spectral, 6),
        "comparison_structure_score": round(comparison, 6),
        "figure_value_score": round(figure, 6),
        "si_value_score": round(si, 6),
        "final_score": final,
        "triage_decision": decision,
        "decision_rationale": rationale,
    }


def _classify_assignment(meaning: str, original_text: str, extraction_method: str) -> tuple[str, str, str]:
    cleaned = re.sub(r"\s+", " ", meaning).strip(" ,;:.")
    lowered = cleaned.lower()
    original_lower = original_text.lower()
    if (
        len(cleaned) < 4
        or sum(character.isdigit() for character in cleaned) >= max(3, len(cleaned) // 3)
        or any(term in original_lower for term in NOISE_TERMS)
        or cleaned.count("(") != cleaned.count(")")
    ):
        return "reject_noise", "low", "Noisy or non-assignment fragment."
    if extraction_method == "text_regex" and not any(term in lowered for term in SUBFAMILY_HINT_TERMS):
        return "mention_only", "low", "Peak mention lacks a stable biochemical meaning phrase."
    if any(term in lowered for term in TENTATIVE_TERMS):
        return "validated_secondary", "low", "Assignment wording is plausible but tentative or region-level."
    if extraction_method == "table_assignment":
        return "validated_secondary", "low", "Structured table-like assignment retained as secondary support."
    return "validated_primary", "medium", "Explicit assignment phrase with usable biochemical wording."


def _download_asset(url: str, destination: Path) -> str:
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except Exception:
        return "failed"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return "downloaded"


def _extract_pdf_text(path: Path) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except Exception:
        return ""


def _extract_explicit_assignments(text: str, paper_id: str, source_id: str) -> list[ExtractedAssignment]:
    rows: list[ExtractedAssignment] = []
    seen = set()
    for match in re.finditer(
        r"(?P<peak>\d{3,4}(?:\.\d+)?)\s*(?:cm[\-−–]?\s*1|cm[\-−–]?1|cm-1|cm−1|cm–1)\s*(?:[^.\n]{0,80}?)(?:assigned to|attributed to|represents?|corresponds to|associated with)\s+(?P<meaning>[^.;\n]{3,120})",
        text,
        re.I,
    ):
        peak = float(match.group("peak"))
        meaning = re.sub(r"\s+", " ", match.group("meaning")).strip(" ,;")
        original = re.sub(r"\s+", " ", match.group(0)).strip()
        key = (round(peak, 2), meaning.lower())
        if key in seen:
            continue
        seen.add(key)
        classification, confidence, rationale = _classify_assignment(meaning, original, "text_assignment")
        rows.append(
            ExtractedAssignment(
                paper_id=paper_id,
                source_id=source_id,
                assignment_record_id=f"{INGEST_PREFIX}_{paper_id}_{len(rows)+1:03d}",
                extraction_method="text_assignment",
                classification=classification,
                peak_center_cm=peak,
                peak_min_cm=peak,
                peak_max_cm=peak,
                assigned_molecule="",
                assigned_group_or_theme=meaning[:120],
                original_text=original[:500],
                figure_reference="",
                manuscript_or_si="manuscript",
                confidence_label=confidence,
                notes="explicit_text_assignment_auto",
                classification_rationale=rationale,
            )
        )
    # fallback table-like rows
    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        match = re.match(r"^(?P<peak>\d{3,4}(?:\.\d+)?)\s+(?P<meaning>[A-Za-z][A-Za-z0-9 ,()=/\-\u2013\u2014]{3,80})$", cleaned)
        if not match:
            continue
        peak = float(match.group("peak"))
        meaning = match.group("meaning").strip(" ,;")
        if len(meaning.split()) > 12 or re.search(r"\d{2,}", meaning):
            continue
        key = (round(peak, 2), meaning.lower())
        if key in seen:
            continue
        seen.add(key)
        classification, confidence, rationale = _classify_assignment(meaning, cleaned, "table_assignment")
        rows.append(
            ExtractedAssignment(
                paper_id=paper_id,
                source_id=source_id,
                assignment_record_id=f"{INGEST_PREFIX}_{paper_id}_{len(rows)+1:03d}",
                extraction_method="table_assignment",
                classification=classification,
                peak_center_cm=peak,
                peak_min_cm=peak,
                peak_max_cm=peak,
                assigned_molecule="",
                assigned_group_or_theme=meaning[:120],
                original_text=cleaned[:500],
                figure_reference="table_like_text",
                manuscript_or_si="manuscript",
                confidence_label=confidence,
                notes="table_like_assignment_auto",
                classification_rationale=rationale,
            )
        )
    for match in re.finditer(
        r"(?P<peak>\d{3,4}(?:\.\d+)?)\s*(?:cm[\-−–]?\s*1|cm[\-−–]?1|cm-1|cm−1|cm–1)(?P<context>[^.;\n]{0,120})",
        text,
        re.I,
    ):
        peak = float(match.group("peak"))
        context = re.sub(r"\s+", " ", match.group("context")).strip(" ,;")
        if not context:
            continue
        key = (round(peak, 2), context.lower())
        if key in seen:
            continue
        if any(verb in context.lower() for verb in ASSIGNMENT_VERBS):
            continue
        if not any(term in context.lower() for term in SUBFAMILY_HINT_TERMS):
            continue
        seen.add(key)
        classification, confidence, rationale = _classify_assignment(context, context, "text_regex")
        rows.append(
            ExtractedAssignment(
                paper_id=paper_id,
                source_id=source_id,
                assignment_record_id=f"{INGEST_PREFIX}_{paper_id}_{len(rows)+1:03d}",
                extraction_method="text_regex",
                classification=classification,
                peak_center_cm=peak,
                peak_min_cm=peak,
                peak_max_cm=peak,
                assigned_molecule="",
                assigned_group_or_theme=context[:120],
                original_text=context[:500],
                figure_reference="",
                manuscript_or_si="manuscript",
                confidence_label=confidence,
                notes="regex_candidate_auto",
                classification_rationale=rationale,
            )
        )
    return rows


def _source_id_for_candidate(candidate: CandidateRecord) -> str:
    author = _slug(candidate.authors[0].split()[-1] if candidate.authors else "paper")
    year = str(candidate.year or "unknown")
    title_piece = _slug(candidate.title)[:24]
    return f"src_{author}_{year}_{title_piece}_manuscript"


def _insert_candidate_source(connection: duckdb.DuckDBPyConnection, candidate: CandidateRecord, source_id: str, manuscript_path: str, note: str) -> None:
    exists = connection.sql(
        f"SELECT COUNT(*) FROM registry.evidence_sources WHERE source_id = {source_id!r}"
    ).fetchone()[0]
    if exists:
        return
    connection.execute(
        "INSERT INTO registry.evidence_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            source_id,
            candidate.title,
            "disease_or_stress_paper",
            SOURCE_KIND,
            manuscript_path,
            "literature_acquisition_pipeline",
            candidate.doi,
            "auto_discovered_literature_peak_assignments",
            "tier2_explicit_or_secondary_assignment",
            False,
            note,
        ],
    )
    connection.execute(
        "INSERT INTO registry.warehouse_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            source_id,
            candidate.title,
            "disease_or_stress_paper",
            "auto_acquired_literature_support",
            _infer_sample_type(candidate) or "mixed_or_unspecified",
            _infer_sample_type(candidate),
            _infer_modality(candidate),
            False,
            True,
            _infer_disease_class(candidate),
            _infer_stress_class(candidate),
            False,
            True,
            False,
            manuscript_path,
            SOURCE_KIND,
            "literature_acquisition_pipeline",
            note,
        ],
    )


def _purge_existing_acquisition_rows(connection: duckdb.DuckDBPyConnection, source_id: str) -> None:
    connection.execute(
        """
        DELETE FROM evidence.peak_assignment_evidence
        WHERE source_id = ?
          AND assignment_origin LIKE 'literature_auto_%'
        """,
        [source_id],
    )
    connection.execute(
        """
        DELETE FROM evidence.evidence_items
        WHERE source_id = ?
          AND created_by = ?
        """,
        [source_id, CREATED_BY],
    )


def _ingest_assignments(connection: duckdb.DuckDBPyConnection, candidate: CandidateRecord, assignments: list[ExtractedAssignment], manuscript_path: str) -> tuple[int, int]:
    source_id = _source_id_for_candidate(candidate)
    _insert_candidate_source(connection, candidate, source_id, manuscript_path, "Auto-ingested from literature acquisition pipeline after explicit-assignment validation.")
    _purge_existing_acquisition_rows(connection, source_id)
    evidence_rows = []
    assignment_rows = []
    primary_count = 0
    secondary_count = 0
    for assignment in assignments:
        if assignment.classification not in {"validated_primary", "validated_secondary"}:
            continue
        evidence_item_id = assignment.assignment_record_id
        is_primary = assignment.classification == "validated_primary"
        if is_primary:
            primary_count += 1
        else:
            secondary_count += 1
        evidence_rows.append(
            (
                evidence_item_id,
                source_id,
                assignment.assignment_record_id,
                "literature_peak_assignment",
                "tier2_explicit_text_assignment" if is_primary else "tier3_secondary_text_assignment",
                assignment.confidence_label,
                f"{candidate.title} {assignment.peak_center_cm:.0f} cm^-1 auto assignment",
                manuscript_path,
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
                assignment.assignment_record_id,
                f"literature_auto_{assignment.extraction_method}",
                _slug(candidate.title)[:40],
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
    connection.executemany(
        "INSERT INTO evidence.evidence_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        evidence_rows,
    )
    connection.executemany(
        "INSERT INTO evidence.peak_assignment_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        assignment_rows,
    )
    return primary_count, secondary_count


def run_literature_acquisition_pipeline(db_path: Path = DB_PATH) -> dict[str, object]:
    ensure_literature_pipeline_output_dirs()
    connection = duckdb.connect(str(db_path))
    try:
        initialize_schema(connection)
        reset_literature_tables(connection)

        records = []
        query_rows = []
        for family, queries in QUERY_FAMILIES.items():
            for query in queries:
                query_label = f"{family}:{query}"
                for fetcher in (_crossref_search, _europepmc_search, _pubmed_search):
                    try:
                        records.extend(fetcher(query, query_label))
                    except Exception as exc:
                        query_rows.append({"query_family": family, "query": query, "source": fetcher.__name__, "status": "failed", "note": str(exc)[:200]})
                    else:
                        query_rows.append({"query_family": family, "query": query, "source": fetcher.__name__, "status": "ok", "note": ""})

        candidates = _dedupe_records(records)
        local_map = _load_local_corpus_map()
        for candidate in candidates:
            _match_local_corpus(candidate, local_map)
            _enrich_candidate(candidate)
        _mark_existing_processing(connection, candidates)

        top_candidates = sorted(
            candidates,
            key=lambda item: (
                len(item.disease_keywords),
                len(item.sample_keywords),
                len(item.spectral_keywords),
                item.open_access_candidate,
            ),
            reverse=True,
        )[:TOP_ASSET_DISCOVERY_LIMIT]
        for candidate in top_candidates:
            _asset_discovery_for_candidate(candidate)

        triage_rows = []
        queue_rows = []
        candidate_rows = []
        asset_rows = []
        extracted_assignment_rows: list[dict] = []
        ingestion_summary_rows = []

        sorted_candidates = []
        for candidate in candidates:
            triage = _triage(candidate)
            sorted_candidates.append((candidate, triage))
        sorted_candidates.sort(key=lambda item: item[1]["final_score"], reverse=True)

        selected_candidates = []
        for rank, (candidate, triage) in enumerate(sorted_candidates, start=1):
            paper_id = candidate.paper_id
            candidate_rows.append(
                {
                    "paper_id": paper_id,
                    "title": candidate.title,
                    "authors": "; ".join(candidate.authors),
                    "year": candidate.year or "",
                    "journal": candidate.journal,
                    "doi": candidate.doi,
                    "source": candidate.primary_source(),
                    "source_list_json": json.dumps(sorted(candidate.sources)),
                    "abstract": candidate.best_abstract(),
                    "query_source": "; ".join(sorted(candidate.query_sources)),
                    "disease_keywords_detected": "; ".join(sorted(candidate.disease_keywords)),
                    "sample_keywords_detected": "; ".join(sorted(candidate.sample_keywords)),
                    "spectral_keywords_detected": "; ".join(sorted(candidate.spectral_keywords)),
                    "title_key": candidate.title_key,
                    "already_in_local_corpus": candidate.already_in_local_corpus,
                    "already_processed_in_evidence": candidate.already_processed_in_evidence,
                    "open_access_candidate": candidate.open_access_candidate,
                    "notes": "; ".join(candidate.notes),
                }
            )
            triage_rows.append({"paper_id": paper_id, **triage, "notes": ""})

            asset_ready = bool(candidate.local_manuscript_path or candidate.remote_manuscript_urls)
            queue_status = triage["triage_decision"]
            selected_for_ingestion = queue_status == "selected_for_ingestion" and len(selected_candidates) < TOP_INGEST_LIMIT
            if selected_for_ingestion:
                selected_candidates.append(candidate)
            queue_rows.append(
                {
                    "queue_id": f"queue_{rank:04d}",
                    "paper_id": paper_id,
                    "queue_status": queue_status,
                    "rank_order": rank,
                    "selected_for_ingestion": selected_for_ingestion,
                    "asset_ready": asset_ready,
                    "ingestion_attempted": False,
                    "extraction_row_count": 0,
                    "selection_reason": triage["decision_rationale"],
                    "notes": "",
                }
            )

            all_asset_urls = []
            if candidate.local_manuscript_path:
                all_asset_urls.append(candidate.local_manuscript_path)
            all_asset_urls.extend(candidate.remote_manuscript_urls)
            asset_rows.append(
                {
                    "asset_id": f"asset_{paper_id}_manuscript",
                    "paper_id": paper_id,
                    "source_provider": candidate.primary_source(),
                    "asset_kind": "manuscript_pdf",
                    "manuscript_pdf_path": candidate.local_manuscript_path,
                    "supplementary_files_json": json.dumps(candidate.supplementary_files),
                    "source_data_links_json": json.dumps(candidate.source_data_links),
                    "figures_detected": candidate.figures_detected,
                    "tables_detected": candidate.tables_detected,
                    "remote_url": candidate.remote_manuscript_urls[0] if candidate.remote_manuscript_urls else "",
                    "local_path": candidate.local_manuscript_path,
                    "file_type": "pdf" if candidate.local_manuscript_path or candidate.remote_manuscript_urls else "",
                    "download_status": "local_existing" if candidate.local_manuscript_path else ("remote_discovered" if candidate.remote_manuscript_urls else "not_found"),
                    "notes": "; ".join(all_asset_urls[:3]),
                }
            )

        # persist literature tables before ingestion
        connection.executemany(
            "INSERT INTO literature.candidate_papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["paper_id"],
                    row["title"],
                    row["authors"],
                    int(row["year"]) if row["year"] != "" else None,
                    row["journal"],
                    row["doi"],
                    row["source"],
                    row["source_list_json"],
                    row["abstract"],
                    row["query_source"],
                    row["disease_keywords_detected"],
                    row["sample_keywords_detected"],
                    row["spectral_keywords_detected"],
                    row["title_key"],
                    row["already_in_local_corpus"],
                    row["already_processed_in_evidence"],
                    row["open_access_candidate"],
                    row["notes"],
                )
                for row in candidate_rows
            ],
        )
        connection.executemany(
            "INSERT INTO literature.paper_triage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["paper_id"],
                    row["condition_relevance_score"],
                    row["sample_relevance_score"],
                    row["spectral_density_score"],
                    row["comparison_structure_score"],
                    row["figure_value_score"],
                    row["si_value_score"],
                    row["final_score"],
                    row["triage_decision"],
                    row["decision_rationale"],
                    row["notes"],
                )
                for row in triage_rows
            ],
        )
        connection.executemany(
            "INSERT INTO literature.paper_assets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["asset_id"],
                    row["paper_id"],
                    row["source_provider"],
                    row["asset_kind"],
                    row["manuscript_pdf_path"],
                    row["supplementary_files_json"],
                    row["source_data_links_json"],
                    row["figures_detected"],
                    row["tables_detected"],
                    row["remote_url"],
                    row["local_path"],
                    row["file_type"],
                    row["download_status"],
                    row["notes"],
                )
                for row in asset_rows
            ],
        )
        connection.executemany(
            "INSERT INTO literature.processing_queue VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["queue_id"],
                    row["paper_id"],
                    row["queue_status"],
                    row["rank_order"],
                    row["selected_for_ingestion"],
                    row["asset_ready"],
                    row["ingestion_attempted"],
                    row["extraction_row_count"],
                    row["selection_reason"],
                    row["notes"],
                )
                for row in queue_rows
            ],
        )
        connection.commit()

        # controlled ingestion for selected papers only
        for queue_row in queue_rows:
            if not queue_row["selected_for_ingestion"]:
                continue
            candidate = next(item for item in candidates if item.paper_id == queue_row["paper_id"])
            manuscript_path = ""
            if candidate.local_manuscript_path:
                manuscript_path = candidate.local_manuscript_path
            elif candidate.remote_manuscript_urls:
                destination = LITERATURE_PIPELINE_ASSET_ROOT / f"{candidate.paper_id}.pdf"
                status = _download_asset(candidate.remote_manuscript_urls[0], destination)
                if status == "downloaded":
                    manuscript_path = str(destination)
                    for asset_row in asset_rows:
                        if asset_row["paper_id"] == candidate.paper_id and asset_row["asset_kind"] == "manuscript_pdf":
                            asset_row["download_status"] = "downloaded"
                            asset_row["local_path"] = manuscript_path
                            asset_row["manuscript_pdf_path"] = manuscript_path
            if not manuscript_path:
                queue_row["ingestion_attempted"] = True
                queue_row["notes"] = "asset_not_downloadable"
                continue
            text = _extract_pdf_text(Path(manuscript_path))
            if not text:
                queue_row["ingestion_attempted"] = True
                queue_row["notes"] = "pdftotext_failed_or_empty"
                continue
            extracted = _extract_explicit_assignments(text, candidate.paper_id, _source_id_for_candidate(candidate))
            for assignment in extracted:
                extracted_assignment_rows.append(
                    {
                        "paper_id": candidate.paper_id,
                        "source_id": _source_id_for_candidate(candidate),
                        "assignment_record_id": assignment.assignment_record_id,
                        "classification": assignment.classification,
                        "classification_rationale": assignment.classification_rationale,
                        "extraction_method": assignment.extraction_method,
                        "peak_center_cm": assignment.peak_center_cm,
                        "assigned_group_or_theme": assignment.assigned_group_or_theme,
                        "original_text": assignment.original_text,
                        "manuscript_or_si": assignment.manuscript_or_si,
                        "ingested_to_evidence": assignment.classification in {"validated_primary", "validated_secondary"},
                    }
                )
            validated = [assignment for assignment in extracted if assignment.classification in {"validated_primary", "validated_secondary"}]
            if not validated:
                queue_row["notes"] = "no_valid_explicit_assignments_found"
                queue_row["ingestion_attempted"] = True
                continue
            primary_count, secondary_count = _ingest_assignments(connection, candidate, validated, manuscript_path)
            queue_row["ingestion_attempted"] = True
            queue_row["extraction_row_count"] = len(validated)
            queue_row["queue_status"] = "processed"
            queue_row["notes"] = f"ingested_primary={primary_count};secondary={secondary_count}"
            ingestion_summary_rows.append(
                {
                    "paper_id": candidate.paper_id,
                    "title": candidate.title,
                    "source_id": _source_id_for_candidate(candidate),
                    "primary_rows_ingested": primary_count,
                    "secondary_rows_ingested": secondary_count,
                    "validated_primary_candidates": sum(1 for assignment in extracted if assignment.classification == "validated_primary"),
                    "validated_secondary_candidates": sum(1 for assignment in extracted if assignment.classification == "validated_secondary"),
                    "mention_only_candidates": sum(1 for assignment in extracted if assignment.classification == "mention_only"),
                    "reject_noise_candidates": sum(1 for assignment in extracted if assignment.classification == "reject_noise"),
                    "manuscript_path": manuscript_path,
                    "notes": queue_row["notes"],
                }
            )

        # refresh queue and assets after ingestion
        connection.execute("DELETE FROM literature.paper_assets")
        connection.execute("DELETE FROM literature.processing_queue")
        connection.executemany(
            "INSERT INTO literature.paper_assets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["asset_id"],
                    row["paper_id"],
                    row["source_provider"],
                    row["asset_kind"],
                    row["manuscript_pdf_path"],
                    row["supplementary_files_json"],
                    row["source_data_links_json"],
                    row["figures_detected"],
                    row["tables_detected"],
                    row["remote_url"],
                    row["local_path"],
                    row["file_type"],
                    row["download_status"],
                    row["notes"],
                )
                for row in asset_rows
            ],
        )
        connection.executemany(
            "INSERT INTO literature.processing_queue VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["queue_id"],
                    row["paper_id"],
                    row["queue_status"],
                    row["rank_order"],
                    row["selected_for_ingestion"],
                    row["asset_ready"],
                    row["ingestion_attempted"],
                    row["extraction_row_count"],
                    row["selection_reason"],
                    row["notes"],
                )
                for row in queue_rows
            ],
        )
        connection.commit()

        if any(row["ingested_to_evidence"] for row in extracted_assignment_rows):
            build_ontology_mappings(connection)
            connection.commit()
            build_local_support_neighborhoods(db_path)

        # outputs
        _write_csv(
            LITERATURE_PIPELINE_TABLES_ROOT / "candidate_papers.csv",
            list(candidate_rows[0].keys()) if candidate_rows else ["paper_id"],
            candidate_rows,
        )
        _write_csv(
            LITERATURE_PIPELINE_TABLES_ROOT / "paper_triage_summary.csv",
            list(triage_rows[0].keys()) if triage_rows else ["paper_id"],
            triage_rows,
        )
        _write_csv(
            LITERATURE_PIPELINE_TABLES_ROOT / "processing_queue.csv",
            list(queue_rows[0].keys()) if queue_rows else ["queue_id"],
            queue_rows,
        )
        _write_csv(
            LITERATURE_PIPELINE_TABLES_ROOT / "extracted_assignments.csv",
            list(extracted_assignment_rows[0].keys()) if extracted_assignment_rows else ["paper_id"],
            extracted_assignment_rows,
        )
        _write_csv(
            LITERATURE_PIPELINE_TABLES_ROOT / "ingestion_summary.csv",
            list(ingestion_summary_rows[0].keys()) if ingestion_summary_rows else ["paper_id"],
            ingestion_summary_rows,
        )

        discovery_lines = [
            "# Discovery Coverage",
            "",
            f"- Queries executed: `{sum(len(items) for items in QUERY_FAMILIES.values())}`",
            f"- Candidate papers after deduplication: `{len(candidate_rows)}`",
            f"- Candidates already in local corpus: `{sum(1 for row in candidate_rows if row['already_in_local_corpus'])}`",
            f"- Candidates already processed in evidence: `{sum(1 for row in candidate_rows if row['already_processed_in_evidence'])}`",
            f"- Candidates with supplementary/source-data asset leads: `{sum(1 for row in asset_rows if row['supplementary_files_json'] != '[]' or row['source_data_links_json'] != '[]')}`",
            "",
            "Primary discovery used Crossref, Europe PMC, and PubMed. Zenodo and Figshare were used in the asset-enrichment stage rather than as the main deduplicated paper source.",
        ]
        discovery_lines.append("")
        discovery_lines.append("Query execution status:")
        for row in query_rows:
            discovery_lines.append(
                f"- `{row['query_family']}` via `{row['source']}` on `{row['query']}`: `{row['status']}`"
                + (f" ({row['note']})" if row["note"] else "")
            )
        (LITERATURE_PIPELINE_REPORT_ROOT / "discovery_coverage.md").write_text("\n".join(discovery_lines))

        triage_log = ["# Triage Decision Log", ""]
        for row in sorted(queue_rows, key=lambda item: item["rank_order"])[:40]:
            triage_log.append(
                f"- `{row['paper_id']}` `{row['queue_status']}` rank `{row['rank_order']}`: {row['selection_reason']}"
            )
        (LITERATURE_PIPELINE_REPORT_ROOT / "triage_decision_log.md").write_text("\n".join(triage_log))

        ingestion_lines = [
            "# Ingestion Quality Report",
            "",
            f"- Selected for ingestion: `{sum(1 for row in queue_rows if row['selected_for_ingestion'])}`",
            f"- Ingestion attempted: `{sum(1 for row in queue_rows if row['ingestion_attempted'])}`",
            f"- Assignment candidates extracted: `{len(extracted_assignment_rows)}`",
            f"- Assignment rows ingested: `{sum(1 for row in extracted_assignment_rows if row.get('ingested_to_evidence'))}`",
            f"- Papers actually processed into evidence: `{len(ingestion_summary_rows)}`",
            "",
            "Only explicit text/table-style assignment statements were eligible for ingestion. Figure-only candidates remained in queue and did not bypass the evidence gate.",
            "Regex-like candidates were classified into `validated_primary`, `validated_secondary`, `mention_only`, or `reject_noise`; only validated rows affected the evidence layer.",
        ]
        (LITERATURE_PIPELINE_REPORT_ROOT / "ingestion_quality_report.md").write_text("\n".join(ingestion_lines))

        return {
            "candidate_count": len(candidate_rows),
            "selected_for_ingestion": sum(1 for row in queue_rows if row["selected_for_ingestion"]),
            "ingestion_attempted": sum(1 for row in queue_rows if row["ingestion_attempted"]),
            "assignment_candidates_extracted": len(extracted_assignment_rows),
            "assignment_rows_ingested": sum(1 for row in extracted_assignment_rows if row.get("ingested_to_evidence")),
            "papers_ingested": len(ingestion_summary_rows),
        }
    finally:
        connection.close()


if __name__ == "__main__":
    print(json.dumps(run_literature_acquisition_pipeline(), indent=2, sort_keys=True))
