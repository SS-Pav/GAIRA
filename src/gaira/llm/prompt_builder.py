"""
GAIRA prompt construction for text queries.

Builds a structured prompt that enforces GAIRA scientific voice:
- discussion-section style, not generic LLM summary
- biochemical themes over exact molecule claims
- uncertainty-aware but not over-defensive
- evidence provenance
"""
from __future__ import annotations

from typing import Sequence


SYSTEM_PREAMBLE = """\
You are GAIRA, a domain-aware Raman/SERS biochemical interpretation engine.
Write like the discussion section of a high-quality scientific paper.

STRICT RULES:
- Interpret in terms of biochemical THEMES and subfamilies, not exact molecules.
- A single Raman peak can map to multiple molecular origins. Never claim
  one peak = one molecule without multi-source support.
- Be decisive where evidence is strong. Be cautious where it is weaker.
- If evidence items conflict, state the disagreement directly.
- Prefer region-based reasoning over exact-wavenumber matching.
- Do not overuse hedging words ("however", "nevertheless", "it should be noted").
- Do not introduce spectral validation, cosine alignment, single-lab holdout,
  or cross-dataset transfer caveats unless the query explicitly asks about
  spectral measurement, validation, or dataset comparison.
"""

# Output section instructions — scientific discussion style
OUTPUT_INSTRUCTIONS = """\
## INSTRUCTIONS

Respond using EXACTLY these section headers, in EXACTLY this order.
Do not add extra sections. Do not rename or reorder them.
Write in a scientific discussion style — synthesized, direct, not a list dump.

### Summary
2-3 sentences stating what the evidence supports. Be direct.

### Biochemical Themes
Group findings into biochemical themes (e.g. membrane lipid changes,
protein backbone shifts, nucleic acid signatures, redox alterations).
Use theme-level language. Mention representative spectral regions
(e.g. 1000-1010 cm⁻¹) where they strengthen the interpretation.

### Strongest Evidence
The best-supported findings. State what the evidence says clearly,
cite the source type or provenance, and note which BSV components
are implicated.

### Supporting Evidence
Additional context consistent with the interpretation but less
certain or less directly grounded.

### Caveats
At most 3 concise caveats relevant to this specific query.
Only mention limitations that actually apply to the evidence retrieved.
Do NOT add generic disclaimers about spectral validation or cross-dataset transfer.

### Confidence Notes
Assess evidence convergence: do grounding sources and literature agree?
Which themes/BSV axes are best supported? Which are weaker?
Does this track known disease biology or suggest tension?
Do NOT frame this as a validation report — frame it as scientific
confidence in the interpretation."""


def build_prompt(
    user_query: str,
    evidence: Sequence[str | dict],
    provenance: Sequence[str] | None = None,
    caveats: Sequence[str] | None = None,
    domain_context: str | None = None,
) -> str:
    """Build a GAIRA-formatted prompt for the LLM.

    Args:
        user_query: The user's text question.
        evidence: Retrieved evidence items (strings or dicts with 'text' key).
        provenance: Source citations for the evidence.
        caveats: Known limitations or warnings.
        domain_context: Optional extra context (e.g. sample type, substrate).

    Returns:
        A single formatted prompt string.
    """
    sections = [SYSTEM_PREAMBLE.strip()]

    # User query
    sections.append(f"## USER QUESTION\n\n{user_query}")

    # Domain context
    if domain_context:
        sections.append(f"## DOMAIN CONTEXT\n\n{domain_context}")

    # Evidence
    evidence_lines = []
    for i, item in enumerate(evidence, 1):
        if isinstance(item, str):
            evidence_lines.append(f"{i}. {item}")
        elif isinstance(item, dict):
            parts = []
            if item.get("title"):
                parts.append(f"[{item['title']}]")
            parts.append(item.get("text", str(item)))
            if item.get("source"):
                parts.append(f"(Source: {item['source']})")
            evidence_lines.append(f"{i}. {' '.join(parts)}")
        else:
            evidence_lines.append(f"{i}. {item}")

    sections.append("## RETRIEVED EVIDENCE\n\n" + "\n".join(evidence_lines))

    # Provenance
    if provenance:
        prov_lines = [f"- {s}" for s in provenance]
        sections.append("## PROVENANCE\n\n" + "\n".join(prov_lines))

    # Caveats
    if caveats:
        caveat_lines = [f"- {c}" for c in caveats]
        sections.append("## KNOWN CAVEATS\n\n" + "\n".join(caveat_lines))

    sections.append(OUTPUT_INSTRUCTIONS)

    return "\n\n---\n\n".join(sections)
