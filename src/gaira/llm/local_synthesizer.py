"""
Local GAIRA response synthesizer — no external LLM required.

Generates a structured GAIRA response from pipeline outputs using
deterministic rules and templates. For development/testing when
Gemini API access is paused or rate-limited.
"""
from __future__ import annotations

from typing import Sequence

from gaira.retrieval.text_query_retriever import RetrievedItem
from gaira.retrieval.motif_theme_mapper import THEME_DISPLAY


# Evidence tier display names (covers both current and legacy tier names)
_TIER_LABEL = {
    "grounding_component": "grounding evidence",
    "evidence_rules": "evidence rules",
    "context_source": "domain context",
    "benchmark_summary": "benchmark analysis",
    "analysis_summary": "analysis summary",
    "meta_summary": "summary context",
    # Legacy
    "grounded_evidence": "grounding evidence",
    "domain_context": "domain context",
    "spectral_query": "analysis summary",
}

# BSV component display names
_BSV_DISPLAY = {
    "membrane_lipid": "membrane lipid",
    "protein_backbone": "protein backbone",
    "aromatic_amino_acid": "aromatic amino acid",
    "purine_nucleotide": "purine nucleotide",
    "pyrimidine_nucleotide": "pyrimidine nucleotide",
    "glycan_carbohydrate": "glycan/carbohydrate",
    "redox_metabolite": "redox metabolite",
    "nucleic_acid_backbone": "nucleic acid backbone",
}


def synthesize_local_response(
    query: str,
    retrieved_items: Sequence[RetrievedItem],
    motifs_themes_bsv: dict,
    literature_bsv_profile: dict,
    evidence_packet: dict | None = None,
) -> dict:
    """Synthesize a structured GAIRA response locally.

    Returns a dict with the 6 canonical section fields, compatible
    with GAIRAResponse.
    """
    themes = motifs_themes_bsv.get("themes", [])
    motifs = motifs_themes_bsv.get("motifs", [])
    bsv_links = motifs_themes_bsv.get("bsv_links", {})
    conditions = literature_bsv_profile.get("conditions", [])
    profiles = literature_bsv_profile.get("profiles", {})
    has_bsv = literature_bsv_profile.get("available", False)

    # Tier counts
    tier_counts: dict[str, int] = {}
    for item in retrieved_items:
        t = item.source_tier or "unknown"
        tier_counts[t] = tier_counts.get(t, 0) + 1

    grounded_n = (tier_counts.get("grounding_component", 0)
                  + tier_counts.get("evidence_rules", 0)
                  + tier_counts.get("grounded_evidence", 0))
    total_n = len(retrieved_items)

    # ── Summary ────────────────────────────────────────────────────
    if conditions:
        cond_str = ", ".join(c.replace("_", " ") for c in conditions)
    else:
        cond_str = "the queried condition"

    top_themes = [t["display"] for t in themes[:3]]
    theme_str = ", ".join(top_themes) if top_themes else "multiple biochemical categories"

    support_level = "strong" if grounded_n >= 3 else "moderate" if grounded_n >= 1 else "limited"

    summary = (
        f"The GAIRA evidence base provides {support_level} literature-grounded support "
        f"for biochemical characterization of {cond_str}. "
        f"The dominant themes supported by retrieved evidence are {theme_str}. "
        f"This interpretation is based on {total_n} retrieved evidence items "
        f"from the curated GAIRA registry."
    )

    # ── Biochemical Themes ─────────────────────────────────────────
    theme_lines = []
    for t in themes:
        motif_str = ", ".join(t["motifs"][:3])
        theme_lines.append(
            f"- **{t['display']}** — supported by {t['evidence_count']} evidence items. "
            f"Key motifs: {motif_str}."
        )
    if not theme_lines:
        theme_lines.append("No strongly supported biochemical themes detected in retrieved evidence.")

    biochemical_themes = "\n".join(theme_lines)

    # ── Strongest Evidence ─────────────────────────────────────────
    strong_lines = []

    # Top motifs with peaks
    for m in motifs[:4]:
        peaks = m.get("peaks", "")
        theme_name = THEME_DISPLAY.get(m["theme"], m["theme"])
        peak_str = f" ({peaks})" if peaks else ""
        strong_lines.append(
            f"- **{m['name']}**{peak_str}: detected in {m['hit_count']} evidence items, "
            f"mapping to the {theme_name} theme."
        )

    # BSV profile highlights
    if has_bsv:
        for cond, prof in profiles.items():
            top_axes = sorted(prof.items(), key=lambda x: -x[1])[:3]
            top_axes = [(k, v) for k, v in top_axes if v > 0.1]
            if top_axes:
                axes_str = ", ".join(
                    f"{_BSV_DISPLAY.get(k, k)} ({v:.2f})" for k, v in top_axes
                )
                strong_lines.append(
                    f"- Literature-grounded BSV profile for **{cond.replace('_', ' ')}** "
                    f"shows strongest support on: {axes_str}."
                )

    if not strong_lines:
        strong_lines.append("No assignment-grade evidence strongly supported in current retrieval.")

    strongest_evidence = "\n".join(strong_lines)

    # ── Supporting Evidence ────────────────────────────────────────
    support_lines = []

    # Context/benchmark sources
    context_items = [it for it in retrieved_items
                     if it.source_tier in ("context_source", "domain_context",
                                           "benchmark_summary", "analysis_summary")]
    if context_items:
        sources = set(it.source_display_name or it.source.split("/")[-1] for it in context_items)
        support_lines.append(
            f"- Domain and benchmark context drawn from: {', '.join(sorted(sources))}."
        )

    # Secondary themes
    for t in themes[3:]:
        support_lines.append(
            f"- **{t['display']}**: {t['evidence_count']} evidence items, "
            f"providing secondary support."
        )

    # BSV weaker axes
    if has_bsv:
        for cond, prof in profiles.items():
            weak_axes = [k for k, v in prof.items() if 0 < v <= 0.25]
            if weak_axes:
                axes_str = ", ".join(_BSV_DISPLAY.get(k, k) for k in weak_axes[:3])
                support_lines.append(
                    f"- Weaker BSV support for {cond.replace('_', ' ')}: {axes_str}."
                )

    if not support_lines:
        support_lines.append("Limited additional context available from current retrieval.")

    supporting_evidence = "\n".join(support_lines)

    # ── Caveats ────────────────────────────────────────────────────
    caveat_lines = [
        "- Raman/SERS peak assignments are many-to-many: a single spectral region "
        "can correspond to multiple molecular origins.",
    ]

    # Only add substrate caveat if relevant motifs are present
    substrate_relevant = any(
        m["name"] in ("PO2 stretch", "CH2/CH3 deformation", "phenylalanine")
        for m in motifs
    )
    if substrate_relevant:
        caveat_lines.append(
            "- Substrate and sample-preparation differences across studies "
            "may shift relative signal expression for some motifs."
        )

    # Add weak-support caveat only if some components are poorly supported
    if has_bsv:
        for cond, prof in profiles.items():
            zero_axes = [k for k, v in prof.items() if v == 0]
            if zero_axes and len(zero_axes) <= 4:
                axes_str = ", ".join(_BSV_DISPLAY.get(k, k) for k in zero_axes[:3])
                caveat_lines.append(
                    f"- Some BSV components ({axes_str}) have no literature support "
                    f"for {cond.replace('_', ' ')} in the current registry."
                )
                break

    caveats = "\n".join(caveat_lines[:3])

    # ── Confidence Notes ───────────────────────────────────────────
    conf_parts = []

    if grounded_n >= 3 and len(themes) >= 3:
        conf_parts.append(
            "Evidence convergence is good: multiple grounding sources and "
            "literature context support the same biochemical themes."
        )
    elif grounded_n >= 1:
        conf_parts.append(
            "Evidence convergence is moderate: grounding evidence is present "
            "but the interpretation relies partly on contextual and benchmark sources."
        )
    else:
        conf_parts.append(
            "Evidence convergence is limited: the interpretation draws primarily "
            "from contextual and summary sources rather than direct grounding evidence."
        )

    if top_themes:
        conf_parts.append(
            f"Best-supported themes: {', '.join(top_themes)}."
        )

    n_sources = len(set(it.source for it in retrieved_items))
    conf_parts.append(f"Evidence drawn from {n_sources} distinct source documents.")

    confidence_notes = " ".join(conf_parts)

    return {
        "answer_summary": summary,
        "biochemical_themes": biochemical_themes,
        "strongest_evidence": strongest_evidence,
        "supporting_evidence": supporting_evidence,
        "caveats": caveats,
        "confidence_notes": confidence_notes,
    }
