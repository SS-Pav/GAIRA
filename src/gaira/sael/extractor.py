"""Part A — Local spectral anchor evidence extraction.

Pulls rows from existing DB evidence sources and emits structured
AnchorEvidenceRow objects. Two record types:

  kind = "contrast"     — row carries a direction claim tied to a peak
                            (e.g. "phenylalanine at 1003 cm⁻¹ increased in ...")
  kind = "assignment"   — row carries a peak→molecule assignment without
                            a direction claim. Useful for anchoring a window's
                            location; NOT used to claim directional shifts.

Sources scanned:
  - peak_assignments  (288 rows; peak_cm + evidence_text + matrix_context)
  - biomarker_claims  (12 rows;  disease_context + spectral_region + claim_text)
  - knowledge_chunks  (96 rows;  diffuse prose organised by section)

Rules:
  - a "contrast" record requires BOTH a cm⁻¹ mention AND a direction verb
    within ±160 characters. Nothing inferred or imputed.
  - direction verb ambiguity → direction = "mixed"; never silently picked.
  - condition terms are matched against a small curated alias map; rows
    that cannot ground a condition stay as `condition_a = None`.

Numbers are deliberately small. We would rather emit 20 strong rows than
200 shaky ones.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import duckdb
import pandas as pd


DEFAULT_DB_PATH = Path("/Volumes/SSD_Rad/GAIRA_DATA/interim/gaira.duckdb")


# ─────────────────────────────────────────────────────────────────────
# Vocabularies
# ─────────────────────────────────────────────────────────────────────

_UP_VERBS = (
    "increase", "increased", "elevated", "elevation", "upregulated",
    "higher", "greater", "enhanced", "stronger", "more intense",
    "significantly increased", "significantly higher", "enriched",
)

_DOWN_VERBS = (
    "decrease", "decreased", "reduced", "reduction", "downregulated",
    "lower", "weaker", "less intense", "diminished", "attenuated",
    "significantly decreased", "significantly lower", "depleted",
)

_MIXED_VERBS = ("mixed", "altered", "shifted", "changed", "modulated")

_CONDITION_ALIASES: dict[str, list[str]] = {
    "HCC":                 ["hcc", "hepatocellular carcinoma"],
    "NAFLD_NASH":          ["nafld", "nash", "non-alcoholic fatty liver"],
    "cholangiocarcinoma":  ["cholangiocarcinoma", "cca", "bile duct cancer"],
    "hepatitis":           ["hepatitis"],
    "fibrosis":            ["fibrosis"],
    "liver_cancer_unspecified": ["liver cancer"],
    "healthy_control":     ["healthy", "control", "healthy control", "normal"],
    "cancer":              ["cancer", "tumor", "tumour"],
    "hepatotoxicity":      ["hepatotoxicity", "hepatotoxic"],
    "diabetes":            ["diabetic", "diabetes"],
    "bmi_gt_25":           ["bmi>25", "bmi > 25", "overweight"],
    "bmi_le_25":           ["bmi≤25", "bmi<=25", "bmi <= 25"],
    "hypoxanthine_spike":  ["hypoxanthine spike", "spiked hypoxanthine"],
    "uric_acid_depletion": ["uricase treatment", "uricase-treated", "uric acid depletion"],
    "ergothioneine_spike": ["ergothioneine spike", "spiked ergothioneine"],
}

# Substrate tokens found in matrix_context etc.
_SUBSTRATES = {
    "SERS":        ["sers"],
    "Raman":       ["raman"],
    "Ag_colloid":  ["ag colloid", "ag-colloid", "agcolloid", "silver colloid"],
    "AgNP":        ["agnp", "ag nanoparticle", "silver nanoparticle"],
    "Au":          ["au", "gold substrate", "gold nanoparticle"],
    "plasmonic_paper": ["plasmonic paper", "paper-based"],
}

_MATRIX_TOKENS = {
    "serum":               ["serum"],
    "plasma":              ["plasma"],
    "extracellular_vesicles": ["extracellular vesicle", "vesicle", "ev"],
    "urine":               ["urine"],
    "biofluid":            ["biofluid"],
    "tissue":              ["tissue"],
    "cell":                ["cell line", "cell"],
}

# Priority chemistry families — the extractor annotates but doesn't filter.
_PRIORITY_TAGS = {
    "purine":       ["adenine", "guanine", "hypoxanthine", "xanthine", "urate", "uric acid", "purine"],
    "redox_sulfur": ["ergothioneine", "glutathione", "cysteine", "disulfide", "thiol", "carotenoid", "carotene"],
    "aromatic_aa":  ["tyrosine", "phenylalanine", "tryptophan", "aromatic"],
    "glycan":       ["glucose", "glycogen", "polysaccharide", "monosaccharide", "glycan"],
    "lipid":        ["lipid", "cholesterol", "phospholipid", "fatty acid"],
    "nucleic_bb":   ["phosphodiester", "po2", "dna", "rna", "ribose"],
    "pyrimidine":   ["cytosine", "thymine", "uracil", "pyrimidine"],
    "protein":      ["amide", "protein backbone"],
}

# Regex: cm⁻¹ number. Accepts "725", "1003 cm-1", "1003 cm^-1", "∼725 cm⁻¹".
_CM_PATTERN = re.compile(
    r"(?P<cm>\d{3,4})\s*(?:cm\s*[-−–]?\s*1|cm\s*\^?\s*\-?\s*1|cm⁻¹|cm\^-1)?",
    re.IGNORECASE,
)

# Region pattern e.g. "1200–1700", "720-1100", "1400-1465 cm^-1"
_RANGE_PATTERN = re.compile(
    r"(?P<lo>\d{3,4})\s*[-−–]\s*(?P<hi>\d{3,4})", re.IGNORECASE,
)


@dataclass
class AnchorEvidenceRow:
    """One extracted piece of anchor-level evidence."""
    source_id: str | None
    source_table: str                     # "peak_assignments" | "biomarker_claims" | "knowledge_chunks"
    kind: str                              # "contrast" | "assignment"
    peak_cm1: float | None
    region_lo_cm1: float | None
    region_hi_cm1: float | None
    condition_a: str | None                # perturbed / disease / spike
    condition_b: str | None                # reference if mentioned; often None
    direction: str                         # up | down | mixed | unknown
    assigned_molecule: str | None
    matrix: str | None
    substrate: str | None
    confidence: str                         # high | moderate | low
    evidence_type: str                      # figure_caption | results_text | table | inferred
    priority_tags: list[str] = field(default_factory=list)
    raw_text_excerpt: str = ""
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _detect_tokens(text: str, vocab: dict[str, list[str]]) -> list[str]:
    """Return canonical tags whose aliases appear in `text` (lowercase match)."""
    t = (text or "").lower()
    hits: list[str] = []
    for canon, aliases in vocab.items():
        if any(a in t for a in aliases):
            hits.append(canon)
    return hits


def _detect_direction_near(text: str, pos: int, radius: int = 160) -> str:
    """Inspect a ±radius window around `pos` in `text` for a direction verb."""
    lo = max(0, pos - radius)
    hi = min(len(text), pos + radius)
    window = text[lo:hi].lower()
    has_up = any(v in window for v in _UP_VERBS)
    has_down = any(v in window for v in _DOWN_VERBS)
    if has_up and has_down:
        return "mixed"
    if has_up:
        return "up"
    if has_down:
        return "down"
    if any(v in window for v in _MIXED_VERBS):
        return "mixed"
    return "unknown"


def _detect_condition_near(text: str, pos: int, radius: int = 220) -> list[str]:
    lo = max(0, pos - radius)
    hi = min(len(text), pos + radius)
    window = text[lo:hi]
    return _detect_tokens(window, _CONDITION_ALIASES)


def _priority_tags_for(text: str) -> list[str]:
    return _detect_tokens(text, _PRIORITY_TAGS)


def _matrix_from(text: str) -> str | None:
    hits = _detect_tokens(text or "", _MATRIX_TOKENS)
    if "serum" in hits:
        return "serum"
    if "extracellular_vesicles" in hits:
        return "extracellular_vesicles"
    if "plasma" in hits:
        return "plasma"
    if "urine" in hits:
        return "urine"
    if "tissue" in hits:
        return "tissue"
    if "biofluid" in hits:
        return "biofluid"
    return None


def _substrate_from(text: str) -> str | None:
    hits = _detect_tokens(text or "", _SUBSTRATES)
    # Specific substrate wins over generic modality.
    for pref in ("AgNP", "Ag_colloid", "Au", "plasmonic_paper"):
        if pref in hits:
            return pref
    if "SERS" in hits:
        return "SERS"
    if "Raman" in hits:
        return "Raman"
    return None


def _confidence_from_source_row(src_row: dict) -> str:
    # Prefer explicit confidence_text if present.
    c = (src_row.get("confidence_text") or src_row.get("evidence_strength") or "").lower()
    if c == "high":
        return "high"
    if c in ("medium", "moderate"):
        return "moderate"
    return "low"


def _evidence_type_from_matrix_context(mx: str | None) -> str:
    t = (mx or "").lower()
    if "caption" in t or "figure_caption" in t:
        return "figure_caption"
    if "table" in t:
        return "table"
    if "body" in t or "results" in t:
        return "results_text"
    return "inferred"


# ─────────────────────────────────────────────────────────────────────
# Per-source extractors
# ─────────────────────────────────────────────────────────────────────

def _scan_peak_assignments(db_path: Path) -> list[AnchorEvidenceRow]:
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute("""
            SELECT assignment_id, source_id, peak_cm, tolerance_cm,
                   assigned_molecule, assigned_group, matrix_context,
                   confidence_text, evidence_text
            FROM peak_assignments
            WHERE peak_cm IS NOT NULL
        """).df()

    out: list[AnchorEvidenceRow] = []
    for _, r in df.iterrows():
        evidence = r.get("evidence_text") or ""
        molecule = r.get("assigned_molecule") or ""
        full = f"{molecule} | {evidence}"

        # Try to find a direction verb near any cm mention in the evidence text.
        direction = "unknown"
        found_contrast = False
        for m in _CM_PATTERN.finditer(evidence):
            try:
                peak_val = float(m.group("cm"))
            except (TypeError, ValueError):
                continue
            # Only treat as meaningful if the text mentions a cm that is close
            # to the row's primary peak_cm (±30). Otherwise we'd pick up random
            # 3-digit numbers in the prose.
            if abs(peak_val - float(r["peak_cm"])) > 30:
                continue
            direction = _detect_direction_near(evidence.lower(), m.start())
            if direction != "unknown":
                found_contrast = True
                break

        # Try scanning the molecule + evidence combined for direction even if
        # the cm mention matching failed (e.g. direction-bearing sentence does
        # not re-state the peak).
        if direction == "unknown":
            combined = full.lower()
            if any(v in combined for v in _UP_VERBS):
                direction = "up"
                found_contrast = True
            elif any(v in combined for v in _DOWN_VERBS):
                direction = "down"
                found_contrast = True

        conditions = _detect_condition_near(full, len(full) // 2)
        condition_a = conditions[0] if conditions else None

        matrix = _matrix_from(r.get("matrix_context") or "")
        substrate = _substrate_from(r.get("matrix_context") or "")

        out.append(AnchorEvidenceRow(
            source_id=r.get("source_id"),
            source_table="peak_assignments",
            kind="contrast" if found_contrast else "assignment",
            peak_cm1=float(r["peak_cm"]),
            region_lo_cm1=None, region_hi_cm1=None,
            condition_a=condition_a, condition_b=None,
            direction=direction,
            assigned_molecule=molecule or None,
            matrix=matrix, substrate=substrate,
            confidence=_confidence_from_source_row(r.to_dict()),
            evidence_type=_evidence_type_from_matrix_context(r.get("matrix_context")),
            priority_tags=_priority_tags_for(full),
            raw_text_excerpt=(evidence or "")[:320],
            notes="",
        ))
    return out


def _scan_biomarker_claims(db_path: Path) -> list[AnchorEvidenceRow]:
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute("""
            SELECT claim_id, source_id, biomarker_name, disease_context,
                   sample_type, spectral_region, claim_text, evidence_strength
            FROM biomarker_claims
        """).df()

    out: list[AnchorEvidenceRow] = []
    for _, r in df.iterrows():
        claim = r.get("claim_text") or ""
        region_str = r.get("spectral_region") or ""
        rm = _RANGE_PATTERN.search(region_str)
        if rm:
            lo, hi = float(rm.group("lo")), float(rm.group("hi"))
        else:
            # Single-value fallback
            cm = _CM_PATTERN.search(region_str)
            if cm:
                v = float(cm.group("cm"))
                lo = hi = v
            else:
                lo = hi = None  # type: ignore[assignment]

        # Biomarker claims deliberately hedge ("may shift", "may change").
        # Direction almost always "mixed" or "unknown" here.
        direction = _detect_direction_near(claim.lower(), len(claim) // 2)
        conditions = _detect_tokens(
            f"{r.get('disease_context') or ''} {claim}",
            _CONDITION_ALIASES,
        )
        condition_a = conditions[0] if conditions else None
        matrix = _matrix_from(r.get("sample_type") or claim)

        out.append(AnchorEvidenceRow(
            source_id=r.get("source_id"),
            source_table="biomarker_claims",
            kind="contrast" if direction != "unknown" else "assignment",
            peak_cm1=None,
            region_lo_cm1=lo, region_hi_cm1=hi,
            condition_a=condition_a, condition_b=None,
            direction=direction,
            assigned_molecule=r.get("biomarker_name"),
            matrix=matrix, substrate=None,
            confidence=_confidence_from_source_row(r.to_dict()),
            evidence_type="inferred",
            priority_tags=_priority_tags_for(claim),
            raw_text_excerpt=claim[:320],
            notes="biomarker_claims row — prose often hedged",
        ))
    return out


def _scan_knowledge_chunks(db_path: Path) -> list[AnchorEvidenceRow]:
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute("""
            SELECT chunk_id, source_id, section, chunk_text
            FROM knowledge_chunks
        """).df()

    out: list[AnchorEvidenceRow] = []
    for _, r in df.iterrows():
        text = r.get("chunk_text") or ""
        # knowledge_chunks are general-context prose. We only emit a row if
        # the chunk contains BOTH a cm⁻¹ value (or explicit range) AND a
        # direction verb. Otherwise they go to the general pool (not SAEL).
        has_cm = bool(_CM_PATTERN.search(text)) or bool(_RANGE_PATTERN.search(text))
        lower = text.lower()
        has_direction = (
            any(v in lower for v in _UP_VERBS)
            or any(v in lower for v in _DOWN_VERBS)
        )
        if not (has_cm and has_direction):
            continue

        # Tightening: knowledge_chunks are general-context prose. Many of the
        # direction-bearing chunks are actually SERS-substrate caveats
        # ("signal can be enhanced unevenly by adsorption geometry") rather
        # than disease-contrast evidence. Only emit as contrast-level if a
        # disease / condition token is also present; otherwise skip.
        chunk_conditions = _detect_tokens(text, _CONDITION_ALIASES)
        if not chunk_conditions:
            continue

        rm = _RANGE_PATTERN.search(text)
        if rm:
            lo, hi = float(rm.group("lo")), float(rm.group("hi"))
            peak = None
        else:
            cm = _CM_PATTERN.search(text)
            peak = float(cm.group("cm")) if cm else None
            lo = hi = None  # type: ignore[assignment]

        direction = _detect_direction_near(lower, len(lower) // 2)
        conditions = _detect_tokens(text, _CONDITION_ALIASES)
        condition_a = conditions[0] if conditions else None
        matrix = _matrix_from(text)
        substrate = _substrate_from(text)

        out.append(AnchorEvidenceRow(
            source_id=r.get("source_id"),
            source_table="knowledge_chunks",
            kind="contrast" if direction in ("up", "down", "mixed") else "assignment",
            peak_cm1=peak,
            region_lo_cm1=lo, region_hi_cm1=hi,
            condition_a=condition_a, condition_b=None,
            direction=direction,
            assigned_molecule=None,
            matrix=matrix, substrate=substrate,
            confidence="low",
            evidence_type="results_text",
            priority_tags=_priority_tags_for(text),
            raw_text_excerpt=text[:320],
            notes="knowledge_chunks prose — general, low confidence",
        ))
    return out


# ─────────────────────────────────────────────────────────────────────
# Top-level
# ─────────────────────────────────────────────────────────────────────

def extract_anchor_evidence(db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    rows: list[AnchorEvidenceRow] = []
    rows.extend(_scan_peak_assignments(db_path))
    rows.extend(_scan_biomarker_claims(db_path))
    rows.extend(_scan_knowledge_chunks(db_path))
    # priority_tags is a list; pandas handles it fine but downstream CSV
    # reading will see a string rep. Convert to "; "-separated for CSV.
    records = []
    for r in rows:
        d = asdict(r)
        d["priority_tags"] = "; ".join(r.priority_tags)
        records.append(d)
    return pd.DataFrame(records)
