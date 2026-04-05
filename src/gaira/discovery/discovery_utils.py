"""Utilities for discovery record normalization and filtering."""

import re
from typing import Any


RAMAN_TERMS = ("raman", "sers")
BIOLOGY_TERMS = (
    "serum",
    "plasma",
    "blood",
    "extracellular vesicle",
    "ev",
    "exosome",
    "cell",
    "tissue",
    "biopsy",
    "bacteria",
    "virus",
    "pathogen",
    "metabolite",
    "lipid",
    "protein",
    "glycan",
)
DATA_SIGNAL_TERMS = (
    "spectra",
    "spectrum",
    "dataset",
    "data",
    "supplementary",
    "analysis",
)
DOMAIN_RULES = (
    ("EV", ("extracellular vesicle", "ev", "exosome")),
    ("SERUM", ("serum", "plasma", "blood")),
    ("PATHOGEN", ("bacteria", "virus", "pathogen")),
    ("MOLECULE", ("metabolite", "lipid", "protein", "glycan")),
    ("TISSUE", ("tissue", "biopsy", "histology")),
)
DATA_FILE_TYPES = {"csv", "xlsx", "txt"}
IMAGE_FILE_TYPES = {"pdf", "png", "jpg"}
LIKELY_FILE_TERMS = ("supplementary", "dataset", "data available")
PDF_ONLY_TYPES = {"pdf"}
PEAK_TABLE_TERMS = ("peak", "assignment", "band")
GAIRA_USABLE_SPECTRA_TYPES = {"RAW_SPECTRA", "PEAK_TABLE"}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _normalize_file_types(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, list):
        candidates = values
    else:
        candidates = [values]

    normalized: list[str] = []
    for value in candidates:
        text = str(value).strip().lower().lstrip(".")
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _contains_term(text: str, term: str) -> bool:
    if len(term) <= 3:
        return re.search(rf"\b{re.escape(term)}\b", text) is not None
    return term in text


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(_contains_term(text, term) for term in terms)


def _extract_file_types_from_links(links_text: str) -> list[str]:
    matches = re.findall(r"\.([a-z0-9]{2,5})(?:\b|[?#])", links_text.lower())
    file_types: list[str] = []
    for extension in matches:
        if extension not in file_types:
            file_types.append(extension)
    return file_types


def normalize_record(record: dict[str, Any], source: str) -> dict[str, Any]:
    title = _normalize_text(record.get("title"))
    doi = _normalize_text(record.get("doi"))
    abstract = _normalize_text(record.get("abstract") or record.get("description"))
    year = _normalize_text(record.get("year"))

    raw_file_links = record.get("file_links")
    if raw_file_links is None:
        if source == "crossref":
            raw_file_links = [record.get("link")] if record.get("link") else []
        elif source == "zenodo":
            raw_file_links = record.get("download_links") or []
        else:
            raw_file_links = []

    file_links = _normalize_text(raw_file_links)
    file_types = _normalize_file_types(record.get("file_types"))
    if not file_types and source == "zenodo":
        file_types = _normalize_file_types(
            [
                str(file_name).rsplit(".", 1)[1]
                for file_name in record.get("files") or []
                if "." in str(file_name)
            ]
        )
    if not file_types and file_links:
        file_types = _extract_file_types_from_links(file_links)

    abstract_lower = abstract.lower()
    has_files = bool(record.get("files")) or bool(file_links) or bool(file_types)
    if not has_files and _contains_any(abstract_lower, LIKELY_FILE_TERMS):
        has_files = True

    is_open_access = bool(record.get("is_open_access"))
    if source == "crossref" and not is_open_access:
        is_open_access = bool(record.get("license_urls")) or "/doi/pdf/" in file_links.lower()

    raw_source_id = _normalize_text(record.get("raw_source_id"))
    if not raw_source_id:
        fallback_id = record.get("pmid") or record.get("link") or title
        raw_source_id = _normalize_text(fallback_id)

    normalized = {
        "title": title,
        "doi": doi,
        "source": source,
        "year": year,
        "abstract": abstract,
        "file_links": file_links,
        "has_files": has_files,
        "file_types": file_types,
        "is_open_access": is_open_access,
        "raw_source_id": raw_source_id,
    }
    normalized["domain_tag"] = assign_domain_tag(normalized)
    normalized["spectra_type"] = classify_spectra_type(normalized)
    normalized["usable_for_gaira"] = normalized["spectra_type"] in GAIRA_USABLE_SPECTRA_TYPES
    normalized["data_quality_score"] = score_data_quality(normalized)
    return normalized


def simple_relevance_filter(record: dict[str, Any]) -> bool:
    combined_text = " ".join(
        [record.get("title", ""), record.get("abstract", "")]
    ).lower()
    has_raman_term = _contains_any(combined_text, RAMAN_TERMS)
    has_biology_term = _contains_any(combined_text, BIOLOGY_TERMS)
    has_data_signal = _contains_any(combined_text, DATA_SIGNAL_TERMS)
    return has_raman_term and has_biology_term and has_data_signal


def assign_domain_tag(record: dict[str, Any]) -> str:
    combined_text = " ".join(
        [record.get("title", ""), record.get("abstract", "")]
    ).lower()
    for domain_tag, terms in DOMAIN_RULES:
        if _contains_any(combined_text, terms):
            return domain_tag
    return "OTHER"


def classify_spectra_type(record: dict[str, Any]) -> str:
    file_types = {file_type.lower() for file_type in record.get("file_types", [])}
    combined_text = " ".join(
        [record.get("title", ""), record.get("abstract", "")]
    ).lower()

    if file_types & DATA_FILE_TYPES:
        return "RAW_SPECTRA"
    if _contains_any(combined_text, PEAK_TABLE_TERMS):
        return "PEAK_TABLE"
    if file_types & IMAGE_FILE_TYPES:
        return "IMAGE_PLOT"
    return "NO_DATA"


def score_data_quality(record: dict[str, Any]) -> int:
    score = 0
    file_types = {file_type.lower() for file_type in record.get("file_types", [])}
    abstract = record.get("abstract", "").lower()
    source = record.get("source", "").lower()

    if file_types & DATA_FILE_TYPES:
        score += 3
    if source == "zenodo":
        score += 2
    if "supplementary" in abstract:
        score += 1
    if file_types and file_types.issubset(PDF_ONLY_TYPES):
        score -= 2

    return score


def deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    seen_fuzzy_titles: set[str] = set()

    for record in records:
        doi_key = record.get("doi", "").strip().lower()
        title_key = record.get("title", "").strip().lower()
        fuzzy_title_key = title_key[:80]

        if doi_key and doi_key in seen_dois:
            continue
        if title_key and title_key in seen_titles:
            continue
        if fuzzy_title_key and fuzzy_title_key in seen_fuzzy_titles:
            continue

        if doi_key:
            seen_dois.add(doi_key)
        if title_key:
            seen_titles.add(title_key)
            seen_fuzzy_titles.add(fuzzy_title_key)

        deduplicated.append(record)

    return deduplicated
