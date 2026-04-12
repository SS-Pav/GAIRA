"""GAIRA Explanation Templates v1.7 — motif differential + stability display.

No LLM. Pure deterministic formatting.
"""

from __future__ import annotations
from graph.phaseC1_scoring import ScoredResult, ScoredTheme, ScoredMotif, MotifDifferential, ComparatorSummary
from graph.phaseC1_query_engine import GraphResult


def _safe(name: str) -> str:
    return name.replace("_", " ").replace("~", "").replace("*", "")


def format_explanation(scored: ScoredResult, graph_result: GraphResult) -> dict:
    is_comp = scored.query_mode in ("pairwise", "one_vs_rest")
    type_labels = {
        "condition": "Condition Query (comparative)" if is_comp else "Condition Query (associative)",
        "peak": "Peak Interpretation Query", "theme": "Biochemical Theme Query",
        "chemistry": "Chemistry-to-Biology Query",
    }

    sections = {
        "A_query_understanding": {
            "query_type": type_labels.get(scored.query_type, scored.query_type),
            "matched_entity": _safe(scored.matched_entity),
            "comparator": _safe(scored.comparator) if scored.comparator else None,
            "query_mode": scored.query_mode,
            "evidence_rows_found": scored.evidence_count,
            "independent_sources": scored.source_count,
            "inferred_sample_type": _safe(graph_result.inferred_sample_type or "unknown"),
        },
        "B_grounding": {
            "matched_motifs": len(scored.motif_summary),
            "biochemical_themes_found": len(scored.themes),
            "biomolecules_found": len(scored.top_biomolecules),
            "functional_groups_found": len(scored.top_functional_groups),
        },
        "C_graph_expansion": {
            "traversal_summary": _traversal(scored),
            "direct_support_edges": graph_result.direct_support_count,
            "inferred_support_edges": graph_result.inferred_support_count,
        },
    }

    if scored.comparator_summary:
        cs = scored.comparator_summary
        sections["C2_comparator_summary"] = {
            "query_condition": _safe(cs.query_condition),
            "comparator_condition": _safe(cs.comparator_condition),
            "inferred_sample_type": _safe(cs.inferred_sample_type),
            "query_evidence": cs.query_evidence, "query_sources": cs.query_sources,
            "comparator_evidence": cs.comparator_evidence, "comparator_sources": cs.comparator_sources,
            "comparator_adequacy": cs.comparator_adequacy,
            "evidence_balance": cs.evidence_balance,
        }

    if is_comp:
        sections["D_enriched_themes"] = [_fmt(t) for t in scored.themes if t.interpretation == "enriched"][:4]
        sections["D2_associated_themes"] = [_fmt(t) for t in scored.themes if t.interpretation == "associated"][:4]
        sections["D3_shared_themes"] = [_fmt(t) for t in scored.themes if t.interpretation == "shared"][:3]
        sections["D4_depleted_themes"] = [_fmt(t) for t in scored.themes if t.interpretation == "depleted"][:3]
    else:
        sections["D_top_themes"] = [_fmt(t) for t in scored.themes[:4]]
        sections["D2_secondary_themes"] = [_fmt(t) for t in scored.themes[4:8]]

    sections["E_top_motifs"] = [_fmt_m(m) for m in scored.motif_summary[:8]]

    # NEW C1.7: Motif differentials with stability
    if scored.motif_differentials:
        sections["E2_motif_differentials"] = [_fmt_md(d) for d in scored.motif_differentials[:12]]
        # Stability summary
        stable = [d for d in scored.motif_differentials if d.stability_label == "STABLE"]
        mixed = [d for d in scored.motif_differentials if d.stability_label == "MIXED"]
        unstable = [d for d in scored.motif_differentials if d.stability_label == "UNSTABLE"]
        insuff = [d for d in scored.motif_differentials if d.stability_label == "INSUFFICIENT"]
        sections["E3_stability_summary"] = {
            "stable": [_safe(d.subfamily or d.motif_id) for d in stable],
            "mixed": [_safe(d.subfamily or d.motif_id) for d in mixed],
            "unstable": [_safe(d.subfamily or d.motif_id) for d in unstable],
            "insufficient": [_safe(d.subfamily or d.motif_id) for d in insuff],
        }

    sections["F_biomolecules"] = scored.top_biomolecules[:6]
    sections["G_functional_groups"] = scored.top_functional_groups[:6]
    sections["H_supporting_evidence"] = scored.evidence_sample
    sections["I_caveats"] = scored.caveats or ["No significant caveats."]
    return sections


def _traversal(s):
    e = _safe(s.matched_entity)
    c = _safe(s.comparator) if s.comparator else ""
    if s.query_mode == "pairwise":
        return f"Compared '{e}' vs '{c}' -> {s.evidence_count} query rows"
    if s.query_mode == "one_vs_rest":
        return f"'{e}' vs all other conditions -> {s.evidence_count} query rows"
    if s.query_type == "condition":
        return f"Condition '{e}' -> {len(s.motif_summary)} motifs -> {s.evidence_count} rows"
    if s.query_type == "peak":
        return f"Peak ~{e} cm-1 -> {s.evidence_count} evidence rows"
    if s.query_type == "theme":
        return f"Theme '{e}' -> {s.evidence_count} evidence rows"
    if s.query_type == "chemistry":
        return f"FG '{e}' -> inferred themes + direct assignments"
    return ""


def _fmt(t: ScoredTheme) -> dict:
    return {
        "theme": _safe(t.theme), "final_score": t.final_score,
        "support_score": t.support_score, "specificity_score": t.specificity_score,
        "confidence": t.confidence, "is_broad": t.is_broad,
        "interpretation": t.interpretation,
        "query_direct": t.query_direct, "query_sources": t.query_sources,
        "comp_direct": t.comp_direct, "comp_sources": t.comp_sources,
        "enrichment_ratio": t.enrichment_ratio,
        "norm_enrichment": t.norm_enrichment_ratio,
        "evidence_balance": t.evidence_balance,
        "coverage_flag": t.coverage_flag,
        "direct_evidence": t.direct_count, "motif_links": t.motif_count,
        "source_diversity": t.source_diversity,
    }


def _fmt_m(m: ScoredMotif) -> dict:
    return {
        "motif_id": m.motif_id, "subfamily": _safe(m.subfamily),
        "family": _safe(m.family), "member_count": m.member_count,
        "enrichment": m.condition_enrichment, "interpretation": m.motif_interpretation,
        "comparator_members": m.comparator_members, "coverage_flag": m.coverage_flag,
    }


def _fmt_md(d: MotifDifferential) -> dict:
    return {
        "motif": _safe(d.subfamily) or d.motif_id,
        "q_direct": d.q_direct, "c_direct": d.c_direct,
        "norm_ratio": d.norm_ratio, "balance": d.balance,
        "interpretation": d.interpretation,
        "studies_total": d.studies_total,
        "studies_enriched": d.studies_enriched,
        "studies_shared": d.studies_shared,
        "studies_depleted": d.studies_depleted,
        "stability": d.stability_label,
        "agreement": d.directional_agreement,
        "flag": d.coverage_flag,
    }


def render_text_explanation(sections: dict) -> str:
    lines = []
    a = sections["A_query_understanding"]
    lines.append(f"### Query: {a['query_type']}")
    el = f"Entity: {a['matched_entity']}"
    if a.get("comparator"):
        el += f"  vs  {a['comparator']}"
    lines.append(el)
    lines.append(f"Evidence: {a['evidence_rows_found']} rows, {a['independent_sources']} sources")
    lines.append(f"Sample type: {a.get('inferred_sample_type','unknown')}")
    lines.append("")

    if "C2_comparator_summary" in sections:
        cs = sections["C2_comparator_summary"]
        lines.append("### Comparator Summary")
        lines.append(f"Query: {cs['query_condition']} ({cs['query_evidence']} rows, {cs['query_sources']} sources)")
        lines.append(f"Comparator: {cs['comparator_condition']} ({cs['comparator_evidence']} rows, {cs['comparator_sources']} sources)")
        lines.append(f"Adequacy: {cs['comparator_adequacy']} | Balance: {cs['evidence_balance']:.2f}")
        lines.append("")

    c = sections["C_graph_expansion"]
    lines.append(f"### Traversal")
    lines.append(c["traversal_summary"])
    lines.append("")

    if "D_enriched_themes" in sections:
        _render_comp(lines, "Enriched", sections.get("D_enriched_themes", []))
        _render_comp(lines, "Associated", sections.get("D2_associated_themes", []))
        _render_comp(lines, "Shared", sections.get("D3_shared_themes", []))
        _render_comp(lines, "Depleted", sections.get("D4_depleted_themes", []))
    else:
        _render_single(lines, "Top Themes", sections.get("D_top_themes", []))
        if sections.get("D2_secondary_themes"):
            _render_single(lines, "Secondary", sections["D2_secondary_themes"])

    if sections.get("E_top_motifs"):
        lines.append("### Top Motifs")
        for m in sections["E_top_motifs"][:5]:
            comp = f" (comp: {m['comparator_members']})" if m.get("comparator_members") else ""
            flag = f" [{m['coverage_flag']}]" if m.get("coverage_flag") else ""
            lines.append(f"- {m['subfamily']} ({m['family']}) {m['member_count']} members{comp} [{m.get('interpretation','')}]{flag}")
        lines.append("")

    if sections.get("E2_motif_differentials"):
        lines.append("### Motif Differential + Stability")
        for d in sections["E2_motif_differentials"][:10]:
            stab = d.get("stability","")
            agr = d.get("agreement",0)
            flag = f" [{d['flag']}]" if d.get("flag") else ""
            lines.append(
                f"- {d['motif']} Q={d['q_direct']} C={d['c_direct']} norm={d['norm_ratio']}x "
                f"[{d['interpretation']}] studies={d['studies_total']} "
                f"(E{d['studies_enriched']}/S{d['studies_shared']}/D{d['studies_depleted']}) "
                f"{stab} ({agr}){flag}"
            )
        lines.append("")

    if sections.get("E3_stability_summary"):
        ss = sections["E3_stability_summary"]
        lines.append("### Stability Summary")
        if ss["stable"]: lines.append(f"- STABLE: {', '.join(ss['stable'])}")
        if ss["mixed"]: lines.append(f"- MIXED: {', '.join(ss['mixed'])}")
        if ss["unstable"]: lines.append(f"- UNSTABLE: {', '.join(ss['unstable'])}")
        if ss["insufficient"]: lines.append(f"- INSUFFICIENT: {', '.join(ss['insufficient'])}")
        lines.append("")

    if sections.get("F_biomolecules"):
        lines.append("### Biomolecules: " + ", ".join(f"{_safe(b['biomolecule'])} ({b['count']})" for b in sections["F_biomolecules"]))
        lines.append("")

    lines.append("### Sample Evidence")
    for ev in sections.get("H_supporting_evidence", [])[:5]:
        lines.append(f"- {ev.get('peak_cm','')} cm-1: {_safe(ev.get('meaning',''))[:70]}")
    lines.append("")

    lines.append("### Caveats")
    for cav in sections.get("I_caveats", []):
        lines.append(f"- {cav}")
    return "\n".join(lines)


def _render_comp(lines, title, themes):
    if not themes:
        return
    lines.append(f"### {title}")
    for t in themes:
        broad = " (broad)" if t.get("is_broad") else ""
        flag = f" [{t['coverage_flag']}]" if t.get("coverage_flag") else ""
        ne = t.get('norm_enrichment', 0)
        ne_str = f" norm={ne}x" if ne else ""
        lines.append(
            f"- {t['theme']}{broad} score={t['final_score']} [{t['confidence'].upper()}]{flag}  "
            f"Q={t['query_direct']}d/{t['query_sources']}s  C={t['comp_direct']}d/{t['comp_sources']}s  "
            f"raw={t['enrichment_ratio']}x{ne_str}  balance={t.get('evidence_balance',0):.2f}"
        )
    lines.append("")


def _render_single(lines, title, themes):
    if not themes:
        return
    lines.append(f"### {title}")
    for t in themes:
        broad = " (broad)" if t.get("is_broad") else ""
        lines.append(f"- {t['theme']}{broad} score={t['final_score']} [{t['confidence'].upper()}] direct={t['direct_evidence']} sources={t['source_diversity']}")
    lines.append("")
