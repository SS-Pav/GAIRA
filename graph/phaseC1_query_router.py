"""GAIRA Query Router v1.6 — scope-aware comparative queries.

Supports sample-type scope, domain scope, and comparator type parsing.
No LLM. Pure keyword matching and pattern rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ParsedQuery:
    raw: str
    query_type: str        # condition | compare | one_vs_rest | peak | theme | chemistry | unknown
    query_mode: str = "single"  # single | pairwise | one_vs_rest
    entities: list[str] = field(default_factory=list)
    comparator: str = ""
    peak_cm: float | None = None
    # Scope fields (NEW in C1.6)
    sample_scope: str = ""       # serum | plasma | EV | tissue | ... | "" (unscoped)
    domain_scope: str = ""       # liver | cancer | infectious | ... | "" (unscoped)
    scope_mode: str = "broad"    # broad | same_matrix | same_matrix_domain | all_sources_same_matrix
    confidence: str = "high"
    notes: str = ""


# ── vocabularies ────────────────────────────────────────────

_CONDITIONS = {
    "hcc": "HCC", "hepatocellular": "HCC",
    "nafld": "NAFLD_NASH", "nash": "NAFLD_NASH", "masld": "NAFLD_NASH",
    "mash": "NAFLD_NASH", "fatty liver": "NAFLD_NASH",
    "fibrosis": "fibrosis", "cirrhosis": "cirrhosis",
    "dili": "DILI_hepatotoxicity", "hepatotoxicity": "DILI_hepatotoxicity",
    "hepatitis": "hepatitis",
    "cholangiocarcinoma": "cholangiocarcinoma", "cca": "cholangiocarcinoma",
    "liver cancer": "liver_cancer", "liver tumor": "liver_cancer",
    "ovarian cancer": "ovarian_cancer", "ovarian": "ovarian_cancer",
    "prostate cancer": "prostate_cancer", "prostate": "prostate_cancer",
    "breast cancer": "breast_cancer", "lung cancer": "lung_cancer",
    "leukemia": "leukemia", "depression": "depression",
    "bacterial": "bacterial_identification", "bacteria": "bacterial_identification",
    "infection": "bacterial_identification", "pathogen": "bacterial_identification",
    "healthy": "healthy_control", "control": "healthy_control",
    "normal": "healthy_control",
}

_SAMPLE_TYPES = {
    "serum": "serum", "plasma": "plasma", "ev": "EV", "exosome": "EV",
    "tissue": "tissue", "saliva": "saliva", "urine": "urine",
    "cell line": "cell_line", "pathogen": "pathogen",
}

_DOMAINS = {
    "liver": "liver", "hepatic": "liver",
    "cancer": "cancer", "oncology": "cancer",
    "infectious": "infectious", "bacterial": "infectious",
    "neurological": "neurological", "cardiovascular": "cardiovascular",
}

_THEMES = {
    "protein": "protein", "peptide": "protein",
    "lipid": "lipid", "phospholipid": "lipid",
    "nucleic acid": "nucleic acid", "dna": "nucleic acid (DNA)",
    "rna": "nucleic acid (RNA)", "nucleotide": "nucleic acid",
    "carbohydrate": "carbohydrate", "saccharide": "carbohydrate",
    "amino acid": "amino acid", "membrane": "membrane", "metabolite": "metabolite",
}

_CHEMISTRY = {
    "amide": "amide", "amide i": "amide I", "amide ii": "amide II",
    "amide iii": "amide III", "carbonyl": "carbonyl", "c=o": "carbonyl",
    "disulfide": "disulfide", "s-s": "disulfide",
    "phosphate": "phosphate", "p-o": "phosphate",
    "c-h": "C-H", "c-c": "C-C", "c-n": "C-N",
    "ring": "ring", "aromatic": "ring",
    "methylene": "methylene", "ch2": "methylene",
    "stretching": "stretching", "bending": "bending",
}


def _find_condition(text: str) -> str | None:
    for kw in sorted(_CONDITIONS, key=len, reverse=True):
        if kw in text:
            return _CONDITIONS[kw]
    return None


def _find_sample_type(text: str) -> str:
    for kw in sorted(_SAMPLE_TYPES, key=len, reverse=True):
        if kw in text:
            return _SAMPLE_TYPES[kw]
    return ""


def _find_domain(text: str) -> str:
    for kw in sorted(_DOMAINS, key=len, reverse=True):
        if kw in text:
            return _DOMAINS[kw]
    return ""


def _determine_scope_mode(text: str, sample: str, domain: str) -> str:
    """Determine the comparator scope mode from query phrasing."""
    if "within" in text and domain:
        return "same_matrix_domain" if sample else "same_domain"
    if "across all" in text and sample:
        return "all_sources_same_matrix"
    if sample and domain:
        return "same_matrix_domain"
    if sample:
        return "same_matrix"
    return "broad"


def route_query(raw: str) -> ParsedQuery:
    text = raw.strip().lower()

    # ── peak query ──────────────────────────────────────────
    peak_match = re.search(r"(\d{3,4})\s*(?:cm|peak|wavenumber)", text)
    if not peak_match:
        peak_match = re.search(r"peak\s+(?:at\s+)?(\d{3,4})", text)
    if peak_match:
        return ParsedQuery(raw=raw, query_type="peak", peak_cm=float(peak_match.group(1)),
                           notes="peak query")

    # ── extract scope modifiers BEFORE condition matching ───
    sample = _find_sample_type(text)
    domain = _find_domain(text)

    # ── comparative: "compare X vs Y" / "X vs Y" ───────────
    vs_match = re.search(r"(?:compare\s+)?(.+?)\s+vs\.?\s+(.+)", text)
    if vs_match:
        left_text = vs_match.group(1).strip()
        right_text = vs_match.group(2).strip()
        left_cond = _find_condition(left_text)
        right_cond = _find_condition(right_text)
        if left_cond and right_cond:
            # Check for sample type in the comparator phrase too
            if not sample:
                sample = _find_sample_type(right_text)
            scope_mode = _determine_scope_mode(text, sample, domain)
            scope_notes = []
            if sample:
                scope_notes.append(f"sample={sample}")
            if domain:
                scope_notes.append(f"domain={domain}")
            scope_notes.append(f"scope={scope_mode}")
            return ParsedQuery(
                raw=raw, query_type="condition", query_mode="pairwise",
                entities=[left_cond], comparator=right_cond,
                sample_scope=sample, domain_scope=domain, scope_mode=scope_mode,
                notes=f"pairwise: {left_cond} vs {right_cond} | {', '.join(scope_notes)}",
            )

    # ── one-vs-rest ─────────────────────────────────────────
    ovr_match = re.search(r"(?:enriched in|distinguishes?|specific to|unique to)\s+(.+?)(?:\s+(?:vs|from|compared)\s+(?:rest|others?|everything))?$", text)
    if ovr_match:
        cond = _find_condition(ovr_match.group(1).strip())
        if cond:
            scope_mode = _determine_scope_mode(text, sample, domain)
            return ParsedQuery(
                raw=raw, query_type="condition", query_mode="one_vs_rest",
                entities=[cond], sample_scope=sample, domain_scope=domain,
                scope_mode=scope_mode,
                notes=f"one-vs-rest: {cond} | scope={scope_mode}",
            )

    # ── single condition ────────────────────────────────────
    cond = _find_condition(text)
    if cond:
        return ParsedQuery(raw=raw, query_type="condition", query_mode="single",
                           entities=[cond], sample_scope=sample, domain_scope=domain,
                           notes=f"condition: {cond}")

    # ── theme ───────────────────────────────────────────────
    for kw, canonical in _THEMES.items():
        if kw in text:
            return ParsedQuery(raw=raw, query_type="theme", entities=[canonical],
                               notes=f"theme: {kw}")

    # ── chemistry (longest match first) ─────────────────────
    for kw, canonical in sorted(_CHEMISTRY.items(), key=lambda x: -len(x[0])):
        if kw in text:
            return ParsedQuery(raw=raw, query_type="chemistry", entities=[canonical],
                               notes=f"chemistry: {kw}")

    return ParsedQuery(raw=raw, query_type="unknown", confidence="low",
                       notes="no entity found; try conditions, peaks, or themes")
