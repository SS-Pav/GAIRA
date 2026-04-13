"""
Section-to-evidence linker for GAIRA responses.

Maps each parsed response section to the retrieved evidence items that
most likely support it, using lexical overlap scoring. This is a
transparent heuristic — not causal attribution.
"""
from __future__ import annotations

import re
from typing import Sequence

from gaira.llm.response_schema import GAIRAResponse
from gaira.retrieval.text_query_retriever import RetrievedItem


SECTION_FIELDS = [
    ("answer_summary", "Summary"),
    ("biochemical_themes", "Biochemical Themes"),
    ("strongest_evidence", "Strongest Evidence"),
    ("supporting_evidence", "Supporting Evidence"),
    ("caveats", "Caveats"),
    ("confidence_notes", "Confidence Notes"),
]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _overlap_score(section_tokens: set[str], evidence_tokens: set[str]) -> float:
    """Jaccard-like overlap weighted toward the section side.

    Returns fraction of section tokens found in the evidence text,
    which measures "how much of this section's vocabulary is covered
    by this evidence item."
    """
    if not section_tokens:
        return 0.0
    return len(section_tokens & evidence_tokens) / len(section_tokens)


def link_sections_to_evidence(
    parsed: GAIRAResponse,
    retrieved_items: Sequence[RetrievedItem],
    top_k: int = 3,
) -> dict[str, list[dict]]:
    """Link each response section to its top supporting evidence items.

    For each section, scores all evidence items by lexical overlap and
    returns the top-k matches with metadata.

    Args:
        parsed: The parsed GAIRA response.
        retrieved_items: Evidence items from the retriever.
        top_k: Max supporting items per section.

    Returns:
        Dict mapping section field name to list of support dicts:
        [{"evidence_idx": int, "title": str, "source": str,
          "tier": str, "support_score": float, "preview": str}]
    """
    # Pre-tokenize evidence
    ev_tokens = []
    for item in retrieved_items:
        tokens = _tokenize(item.text + " " + (item.title or ""))
        ev_tokens.append(tokens)

    links: dict[str, list[dict]] = {}

    for field, label in SECTION_FIELDS:
        section_text = getattr(parsed, field, "")
        if not section_text:
            links[field] = []
            continue

        section_tok = _tokenize(section_text)
        # Remove very common words that add noise
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "of",
                      "to", "and", "or", "for", "with", "this", "that", "from",
                      "on", "by", "as", "it", "not", "but", "be", "at", "has",
                      "have", "had", "can", "may", "its", "also", "than", "more",
                      "most", "such", "which", "these", "those", "been", "will"}
        section_tok -= stopwords

        scored = []
        for i, item in enumerate(retrieved_items):
            score = _overlap_score(section_tok, ev_tokens[i])
            if score > 0.05:  # minimum threshold
                scored.append((score, i, item))

        scored.sort(key=lambda x: -x[0])

        supports = []
        for score, idx, item in scored[:top_k]:
            preview = item.text[:120].replace("\n", " ")
            supports.append({
                "evidence_idx": idx,
                "title": item.title or "(untitled)",
                "source": item.source,
                "tier": item.source_tier,
                "support_score": round(score, 3),
                "preview": preview,
            })

        links[field] = supports

    return links
