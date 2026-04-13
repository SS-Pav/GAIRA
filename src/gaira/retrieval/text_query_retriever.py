"""
GAIRA text query retriever — local, file-backed, evidence-tiered.

Loads curated GAIRA evidence documents, splits them into sections,
scores by IDF-weighted keyword overlap, then applies evidence-tier
weighting and section-level demotion/promotion so that atomic evidence
outranks meta-summaries.

Phase 4: tier-aware, diversity-capped, inspectable.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from gaira.retrieval.source_registry import (
    GROUNDED_TITLE_PATTERNS,
    META_TITLE_PATTERNS,
    SOURCE_REGISTRY,
    TIER_WEIGHTS,
    SourceEntry,
)


@dataclass
class RetrievedItem:
    """A single retrieved evidence section."""
    text: str
    source: str
    title: str = ""
    retrieval_score: float = 0.0
    source_tier: str = ""
    source_display_name: str = ""  # human-readable source label


@dataclass
class SourceDocument:
    """A loaded document split into sections."""
    path: str
    tier: str
    reason: str
    display_name: str = ""
    sections: list[dict] = field(default_factory=list)


def _split_markdown_sections(text: str) -> list[dict]:
    """Split markdown into sections by ## or ### headers."""
    lines = text.split("\n")
    sections: list[dict] = []
    current_title = ""
    current_lines: list[str] = []

    for line in lines:
        if re.match(r"^#{1,3}\s+", line):
            body = "\n".join(current_lines).strip()
            if body and len(body) > 30:
                sections.append({"title": current_title, "text": body})
            current_title = re.sub(r"^#{1,3}\s+", "", line).strip()
            current_lines = []
        else:
            current_lines.append(line)

    body = "\n".join(current_lines).strip()
    if body and len(body) > 30:
        sections.append({"title": current_title, "text": body})

    return sections


def _tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _section_tier_adjustment(title: str, doc_tier: str) -> float:
    """Return a multiplier adjustment for a section based on its title.

    Demotes meta/planning sections. Promotes grounded sections even in
    lower-tier documents. Returns a multiplier (0.3 – 1.4).
    """
    title_lower = title.lower()

    # Demote meta/planning/verdict sections
    for pat in META_TITLE_PATTERNS:
        if pat in title_lower:
            return 0.3

    # Promote grounded sections in any document
    for pat in GROUNDED_TITLE_PATTERNS:
        if pat in title_lower:
            return 1.4 if doc_tier != "grounded_evidence" else 1.0

    return 1.0


class TextQueryRetriever:
    """Evidence-tiered keyword retriever over curated GAIRA documents."""

    def __init__(self):
        self.documents: list[SourceDocument] = []
        # (section_dict, source_path, doc_tier, display_name)
        self._all_sections: list[tuple[dict, str, str, str]] = []
        self._idf: dict[str, float] = {}
        self._loaded = False

    def load_sources(self) -> int:
        """Load all sources from the registry. Returns section count."""
        self.documents = []
        self._all_sections = []

        for entry in SOURCE_REGISTRY:
            path = entry.full_path
            if not path.exists():
                continue
            text = path.read_text(errors="replace")
            sections = _split_markdown_sections(text)
            dname = entry.display_name or entry.path.split("/")[-1].replace(".md", "")
            doc = SourceDocument(
                path=entry.path, tier=entry.tier,
                reason=entry.reason, display_name=dname, sections=sections,
            )
            self.documents.append(doc)
            for sec in sections:
                self._all_sections.append((sec, entry.path, entry.tier, dname))

        self._build_idf()
        self._loaded = True
        return len(self._all_sections)

    def _build_idf(self):
        n = len(self._all_sections)
        if n == 0:
            return
        df: dict[str, int] = {}
        for sec, _, _, _ in self._all_sections:
            tokens = set(_tokenize(sec["text"] + " " + sec.get("title", "")))
            for t in tokens:
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log(n / count) for t, count in df.items()}

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        max_per_source: int = 3,
    ) -> list[RetrievedItem]:
        """Retrieve top-k sections with tier weighting and diversity cap.

        Scoring:
          raw_score   = Σ IDF(token) for matches + 2x title bonus
          tier_weight = TIER_WEIGHTS[doc_tier]
          sec_adjust  = section-level promotion/demotion (0.3 – 1.4)
          final_score = raw_score × tier_weight × sec_adjust

        Diversity: at most max_per_source items from the same document.
        """
        if not self._loaded:
            self.load_sources()

        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        scored: list[tuple[float, float, dict, str, str, str]] = []

        for sec, source, tier, dname in self._all_sections:
            text_tokens = set(_tokenize(sec["text"]))
            title_tokens = set(_tokenize(sec.get("title", "")))

            raw = 0.0
            for qt in query_tokens:
                idf = self._idf.get(qt, 0.0)
                if qt in text_tokens:
                    raw += idf
                if qt in title_tokens:
                    raw += idf * 2.0

            if raw <= 0:
                continue

            tier_w = TIER_WEIGHTS.get(tier, 0.5)
            sec_adj = _section_tier_adjustment(sec.get("title", ""), tier)
            final = raw * tier_w * sec_adj

            scored.append((final, raw, sec, source, tier, dname))

        scored.sort(key=lambda x: -x[0])

        # Diversity cap: limit items per source document
        results: list[RetrievedItem] = []
        source_counts: dict[str, int] = {}

        for final, raw, sec, source, tier, dname in scored:
            if len(results) >= top_k:
                break

            count = source_counts.get(source, 0)
            if count >= max_per_source:
                continue
            source_counts[source] = count + 1

            text = sec["text"]
            if len(text) > 1500:
                text = text[:1500] + " [...]"

            results.append(RetrievedItem(
                text=text,
                source=source,
                title=sec.get("title", ""),
                retrieval_score=round(final, 3),
                source_tier=tier,
                source_display_name=dname,
            ))

        return results

    def source_summary(self) -> list[dict]:
        """Return summary of loaded sources for inspection."""
        return [
            {
                "path": doc.path,
                "tier": doc.tier,
                "reason": doc.reason,
                "n_sections": len(doc.sections),
            }
            for doc in self.documents
        ]
