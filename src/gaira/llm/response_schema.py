"""
Structured response container and normalization for GAIRA LFM outputs.

Normalizes formatting variation across different Gemini models so
GAIRA's parser behaves consistently. Does NOT rewrite scientific content.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# Canonical section headers that GAIRA expects
_CANONICAL_SECTIONS = [
    ("answer_summary",      "Summary"),
    ("biochemical_themes",  "Biochemical Themes"),
    ("strongest_evidence",  "Strongest Evidence"),
    ("supporting_evidence", "Supporting Evidence"),
    ("caveats",             "Caveats"),
    ("confidence_notes",    "Confidence Notes"),
]

# Patterns to match section headers in various markdown styles
# Handles: ### Summary, ## Summary, **Summary**, **Summary:**, Summary:
_HEADER_PATTERNS = {
    attr: re.compile(
        r"(?:^|\n)\s*(?:#{1,3}\s+|\*\*)" + re.escape(label) + r"(?:\*\*)?[\s:]*",
        re.IGNORECASE,
    )
    for attr, label in _CANONICAL_SECTIONS
}


def normalize_response_text(raw: str) -> str:
    """Normalize raw LLM output so section headers are consistent.

    Standardizes all heading variants to '### Header' format.
    Does not alter body content.
    """
    text = raw

    for attr, label in _CANONICAL_SECTIONS:
        pattern = _HEADER_PATTERNS[attr]

        def _replacer(m, lbl=label):
            prefix = "\n" if m.group(0)[0] == "\n" else ""
            return f"{prefix}### {lbl}\n"

        text = pattern.sub(_replacer, text, count=1)

    return text


@dataclass
class GAIRAResponse:
    """Container for a GAIRA text query response."""

    raw_text: str
    answer_summary: str = ""
    biochemical_themes: str = ""
    strongest_evidence: str = ""
    supporting_evidence: str = ""
    caveats: str = ""
    confidence_notes: str = ""
    parse_success: bool = False

    @classmethod
    def from_raw(cls, raw_text: str) -> GAIRAResponse:
        """Parse a structured response from raw LLM output.

        Normalizes formatting first, then extracts sections.
        Falls back gracefully if sections can't be found.
        """
        normalized = normalize_response_text(raw_text)
        resp = cls(raw_text=raw_text)

        # Find all section positions using the canonical ### Header pattern
        positions = []
        for attr, label in _CANONICAL_SECTIONS:
            pattern = re.compile(rf"^### {re.escape(label)}\s*$", re.MULTILINE | re.IGNORECASE)
            match = pattern.search(normalized)
            if match:
                positions.append((match.start(), match.end(), attr))

        if not positions:
            return resp

        positions.sort(key=lambda x: x[0])

        for i, (start, end, attr) in enumerate(positions):
            if i + 1 < len(positions):
                content = normalized[end:positions[i + 1][0]]
            else:
                content = normalized[end:]

            content = content.strip()
            setattr(resp, attr, content)

        resp.parse_success = len(positions) >= 3

        # Fill missing sections with a conservative placeholder
        for attr, label in _CANONICAL_SECTIONS:
            if not getattr(resp, attr):
                setattr(resp, attr, "")

        return resp

    def summary_display(self) -> str:
        """Short display string for terminal output."""
        if self.parse_success:
            parts = [f"SUMMARY: {self.answer_summary[:200]}"]
            if self.biochemical_themes:
                parts.append(f"THEMES: {self.biochemical_themes[:200]}")
            if self.caveats:
                parts.append(f"CAVEATS: {self.caveats[:200]}")
            return "\n\n".join(parts)
        return self.raw_text[:500]
