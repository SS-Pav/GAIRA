"""
GAIRA trust graph builder — GAIRA-native reasoning flow (Phase 7A).

6-column staged graph:
  Query → Evidence Lanes → Motifs → Themes → BSV → Output
"""
from __future__ import annotations

from typing import Sequence

from gaira.retrieval.text_query_retriever import RetrievedItem


TIER_COLORS = {
    "grounding_component": "#2ECC71",
    "evidence_rules":      "#27AE60",
    "context_source":      "#F39C12",
    "benchmark_summary":   "#9B59B6",
    "analysis_summary":    "#3498DB",
    "meta_summary":        "#95A5A6",
    # Legacy aliases (kept for compatibility)
    "grounded_evidence":   "#2ECC71",
    "domain_context":      "#F39C12",
    "spectral_query":      "#3498DB",
}

TIER_DISPLAY = {
    "grounding_component": "Grounding",
    "evidence_rules":      "Rules",
    "context_source":      "Context",
    "benchmark_summary":   "Benchmark",
    "analysis_summary":    "Analysis",
    "meta_summary":        "Meta",
    # Legacy aliases
    "grounded_evidence":   "Grounding",
    "domain_context":      "Context",
    "spectral_query":      "Analysis",
}

NODE_TYPE_COLORS = {
    "query":   "#ECF0F1",
    "motif":   "#E8D44D",
    "theme":   "#1ABC9C",
    "bsv":     "#3498DB",
    "output":  "#E74C3C",
}

BSV_COMPONENTS = [
    "membrane_lipid", "protein_backbone", "aromatic_amino_acid",
    "purine_nucleotide", "pyrimidine_nucleotide", "glycan_carbohydrate",
    "redox_metabolite", "nucleic_acid_backbone",
]

BSV_SHORT = {
    "membrane_lipid": "Lipid",
    "protein_backbone": "Protein",
    "aromatic_amino_acid": "Aromatic AA",
    "purine_nucleotide": "Purine",
    "pyrimidine_nucleotide": "Pyrimidine",
    "glycan_carbohydrate": "Glycan",
    "redox_metabolite": "Redox",
    "nucleic_acid_backbone": "Nuc. Backbone",
}


def build_trust_graph(
    query: str,
    retrieved_items: Sequence[RetrievedItem],
    motif_theme_map: dict,
    bsv_profile: dict | None = None,
) -> dict:
    """Build a 6-column GAIRA-native trust graph.

    Columns:
      0: Query
      1: Evidence (grouped by tier: grounding / literature / context)
      2: Motifs
      3: Themes
      4: BSV components (only those with evidence support)
      5: Output

    Args:
        query: User's text query.
        retrieved_items: Evidence from retriever.
        motif_theme_map: Output from map_evidence_to_motifs_themes_bsv().
        bsv_profile: Output from build_literature_bsv_profile() (optional).

    Returns:
        dict with nodes, edges, summary, column_labels
    """
    nodes = []
    edges = []

    # ── Col 0: Query ───────────────────────────────────────────────
    nodes.append({
        "id": "query", "label": "Query", "column": 0, "row": 0,
        "node_type": "query", "color": NODE_TYPE_COLORS["query"],
        "detail": query,
    })

    # ── Col 1: Evidence lanes (grounding first, then context/summary) ──
    from gaira.retrieval.source_registry import GROUNDING_TIERS, LITERATURE_TIERS
    grounding_tiers = GROUNDING_TIERS
    literature_tiers = LITERATURE_TIERS

    import re as _re

    def _clean_section_name(title: str) -> str:
        """Extract the component/section name from titles like '3. aromatic_amino_acid'."""
        cleaned = _re.sub(r"^\d+\.\s*", "", title).strip()
        return cleaned if cleaned else title

    def _ev_label(item, tier: str) -> str:
        """Visible graph label for an evidence node."""
        if tier in grounding_tiers and item.title:
            # For grounding nodes, show the actual component/section name
            return _clean_section_name(item.title)[:24]
        # For context/summary nodes, use the display name
        dname = item.source_display_name
        if dname:
            return dname[:24]
        return item.source.split("/")[-1].replace(".md", "")[:24]

    # Source-type explainers for hover text
    _EXPLAINERS = {
        "BSV Component Definitions": "Grounding component definition from GAIRA's BSV framework.",
        "BSV v2 Revision Notes": "Scientific rationale for BSV component naming and design.",
        "BSV Scoring Logic": "Rules governing how motifs contribute to BSV component scores.",
        "BSV Confidence Rules": "Criteria for component confidence labels (strong/moderate/weak).",
        "BSV Graph Structure": "How the BSV evidence graph is structured.",
        "Contrast Schema": "Schema for recording directional peak changes (up/down) across conditions.",
        "Peak Matching Rules": "Rules for matching reported peaks to existing evidence (±5 cm⁻¹ tolerance).",
        "Harmonization Rules": "Rules for harmonizing magnitude and significance across studies.",
        "Trust Tier Rules": "Definitions of evidence support tiers (strong/moderate/weak/insufficient).",
        "Condition Classification": "Categories for classifying conditions (biological vs method vs context).",
        "Serum SERS Context": "GAIRA domain-context source for interpreting serum Raman/SERS evidence.",
        "EV SERS Context": "GAIRA domain-context source for interpreting extracellular-vesicle SERS evidence.",
        "Liver Contrast Summary": "GAIRA synthesis of directional contrast evidence across liver conditions.",
        "Contrast Pilot Summary": "Results from the initial contrast evidence extraction pilot.",
        "Directionality Audit": "GAIRA audit assessing whether the literature provides usable directional assignments.",
        "Liver Evidence Limits": "Known failure modes and limitations of the liver evidence corpus.",
        "Landscape v2 Summary": "GAIRA landscape v2 BSV matrix construction across conditions.",
        "Landscape v3 Summary": "GAIRA landscape v3 condition-level evidence structure.",
        "Landscape v4 BSV Deltas": "GAIRA landscape v4 condition-specific BSV deltas vs healthy baseline.",
        "Landscape v5 Decomposition": "GAIRA landscape v5 subcomponent decomposition and confidence scoring.",
        "Landscape v5.1 Hardening": "Inference hardening: bounded confidence, penalty logic.",
        "HCC Spectral Query v1": "GAIRA result summary from the first HCC spectral-query validation run.",
        "BSV Delta Plot Notes": "Notes on BSV delta visualization methodology.",
        "BSV Output Format": "BSV output data format specification.",
        "BSV Table Format": "BSV v2 side-by-side comparison table format.",
        "Contrast Schema Proposal": "Proposed contrast evidence schema (deferred for future use).",
        "Contrast Integration Notes": "Integration hooks for contrast data into GAIRA graph.",
        "Domain Context Template": "Template for domain measurement/provenance context.",
        "Landscape v3 BSV Bugfix": "Technical fix details for landscape v3 BSV computation.",
        "Evidence Ontology (Phase C)": "Evidence stabilization: 198 meanings, 7 L1 / 21 L2 hierarchy, ontology normalization.",
        "Extraction Pilot (Phase G)": "OA extraction pilot: 20 papers, 164 assignment-grade evidence rows, modality labels.",
        "CCA BSV Analysis (v3)": "GAIRA CCA dataset multi-condition BSV analysis.",
        "CCA Refined Query (v3.2)": "GAIRA CCA refined-band spectral query results.",
        "CCA Robustness (v3.3)": "GAIRA CCA robustness test using randomization and bootstrap.",
        "Substrate Sensitivity": "GAIRA analysis of Au vs AgNP substrate sensitivity across BSV axes.",
        "CCA Substrate Refinement": "GAIRA CCA substrate-aware refinement results.",
    }

    def _ev_explainer(item) -> str:
        """Plain-language explanation for hover text."""
        dname = item.source_display_name or ""
        if dname in _EXPLAINERS:
            return _EXPLAINERS[dname]
        tier = item.source_tier or ""
        if tier in grounding_tiers:
            return "Grounding component definition from GAIRA's structured evidence layer."
        return "Retrieved from GAIRA's curated structured evidence corpus."

    def _ev_node(i, item, lane, row_val):
        nid = f"ev_{i}"
        tier = item.source_tier or "meta_summary"
        return {
            "id": nid, "label": _ev_label(item, tier), "column": 1, "row": row_val,
            "node_type": tier,
            "color": TIER_COLORS.get(tier, "#95A5A6"),
            "lane": lane,
            "detail": item.text[:300],
            "meta": {
                "title": item.title or "(untitled)",
                "source": item.source,
                "source_path": item.source,
                "display_name": item.source_display_name,
                "tier": TIER_DISPLAY.get(tier, tier),
                "score": item.retrieval_score,
                "explainer": _ev_explainer(item),
            },
        }

    row = 0
    for i, item in enumerate(retrieved_items):
        if (item.source_tier or "meta_summary") not in grounding_tiers:
            continue
        nodes.append(_ev_node(i, item, "grounding", row))
        edges.append({"from": "query", "to": f"ev_{i}", "weight": 1.0})
        row += 1

    for i, item in enumerate(retrieved_items):
        if (item.source_tier or "meta_summary") not in literature_tiers:
            continue
        nodes.append(_ev_node(i, item, "literature", row + 1))  # +1 gap
        edges.append({"from": "query", "to": f"ev_{i}", "weight": 0.7})
        row += 1

    # ── Col 2: Motifs ──────────────────────────────────────────────
    motifs = motif_theme_map.get("motifs", [])
    for mi, m in enumerate(motifs):
        mid = f"motif_{mi}"
        peaks = m.get("peaks", "")
        detail = f"Peaks: {peaks}\nTheme: {m['theme']}\nEvidence: {m['hit_count']} items"
        nodes.append({
            "id": mid, "label": m["name"][:20], "column": 2, "row": mi,
            "node_type": "motif", "color": NODE_TYPE_COLORS["motif"],
            "detail": detail,
            "peaks": peaks,
        })
        # Connect evidence → motif (limit to top 3 to reduce clutter)
        ev_for_motif = m["evidence_indices"][:3]
        for ev_idx in ev_for_motif:
            edges.append({"from": f"ev_{ev_idx}", "to": mid, "weight": 1.0})

    # ── Col 3: Themes ──────────────────────────────────────────────
    themes = motif_theme_map.get("themes", [])
    for ti, t in enumerate(themes):
        tid = f"theme_{ti}"
        nodes.append({
            "id": tid, "label": t["display"][:20], "column": 3, "row": ti,
            "node_type": "theme", "color": NODE_TYPE_COLORS["theme"],
            "detail": f"Motifs: {', '.join(t['motifs'][:4])}. Evidence: {t['evidence_count']}",
        })
        # Connect motifs → theme
        for mi, m in enumerate(motifs):
            if m["theme"] == t["name"]:
                edges.append({"from": f"motif_{mi}", "to": tid, "weight": 1.0})

    # ── Col 4: BSV components ──────────────────────────────────────
    bsv_links = motif_theme_map.get("bsv_links", {})
    active_bsv = [c for c in BSV_COMPONENTS if c in bsv_links]
    for bi, comp in enumerate(active_bsv):
        bid = f"bsv_{comp}"
        score_str = ""
        if bsv_profile and bsv_profile.get("available"):
            # Show the first condition's score
            for cond, prof in bsv_profile.get("profiles", {}).items():
                val = prof.get(comp, 0)
                if val > 0:
                    score_str = f" ({val:.2f})"
                break

        nodes.append({
            "id": bid, "label": BSV_SHORT.get(comp, comp) + score_str,
            "column": 4, "row": bi,
            "node_type": "bsv", "color": NODE_TYPE_COLORS["bsv"],
            "detail": f"{comp}: {len(bsv_links[comp])} evidence items",
        })
        # Connect theme → BSV
        for ti, t in enumerate(themes):
            if t["bsv_component"] == comp:
                edges.append({"from": f"theme_{ti}", "to": bid, "weight": 1.0})

    # ── Col 5: Output ──────────────────────────────────────────────
    conditions = bsv_profile.get("conditions", []) if bsv_profile else []
    output_label = "BSV Profile" if conditions else "Answer"

    nodes.append({
        "id": "output", "label": output_label, "column": 5, "row": 0,
        "node_type": "output", "color": NODE_TYPE_COLORS["output"],
        "detail": f"Conditions: {', '.join(conditions)}" if conditions else "Synthesized answer",
    })
    for bi, comp in enumerate(active_bsv):
        edges.append({"from": f"bsv_{comp}", "to": "output", "weight": 0.8})

    # ── Summary (using canonical bucket counting) ───────────────
    from gaira.retrieval.confidence_composer import _count_buckets
    buckets = _count_buckets(retrieved_items)

    summary = {
        "total_evidence": len(retrieved_items),
        "grounded_count": buckets["grounded"],
        "context_count": buckets["context"],
        "benchmark_count": buckets["benchmark"],
        "n_motifs": len(motifs),
        "n_themes": len(themes),
        "n_bsv_active": len(active_bsv),
        "conditions": conditions,
    }

    return {
        "nodes": nodes,
        "edges": edges,
        "summary": summary,
        "column_labels": ["Query", "Evidence", "Motifs", "Themes", "BSV", "Output"],
    }


def build_per_condition_traversals(
    query: str,
    retrieved_items: Sequence[RetrievedItem],
    motif_theme_map: dict,
    bsv_profile: dict | None = None,
    conditions: list[str] | None = None,
    retriever=None,
) -> list[dict]:
    """Build separate trust graph traversals for each condition in a comparison query.

    For each condition:
    1. Filters evidence to items mentioning that condition
    2. Supplements with grounding component definitions if the BSV profile
       shows active components (ensures motif chains are populated)
    3. Builds an independent motif→theme→BSV traversal

    Args:
        retriever: Optional TextQueryRetriever for supplemental grounding retrieval.
    """
    from gaira.retrieval.motif_theme_mapper import map_evidence_to_motifs_themes_bsv
    from gaira.retrieval.literature_bsv_builder import BSV_COMPONENTS

    if not conditions:
        conditions = (bsv_profile or {}).get("conditions", [])

    if len(conditions) <= 1:
        return []

    # Condition keyword variants for filtering
    _CONDITION_KEYWORDS = {
        "HCC": ["hcc", "hepatocellular"],
        "healthy_control": ["healthy", "control", "baseline", "normal"],
        "cholangiocarcinoma": ["cca", "cholangiocarcinoma"],
        "NAFLD_NASH": ["nafld", "nash", "fatty liver"],
        "liver_cancer_unspecified": ["lm", "liver metast", "liver cancer"],
        "hepatitis": ["hepatitis"],
        "fibrosis": ["fibrosis"],
    }

    # Build index of BSV component definition sections directly from the
    # retriever's loaded documents. Each condition will get only its
    # top-scoring components, ensuring different motif chains.
    import re as _re
    all_component_defs: dict[str, RetrievedItem] = {}
    if retriever is not None:
        for doc in retriever.documents:
            if doc.display_name != "BSV Component Definitions":
                continue
            for sec in doc.sections:
                comp_name = _re.sub(r"^\d+\.\s*", "", sec.get("title", "")).strip()
                if comp_name in BSV_COMPONENTS:
                    text = sec["text"]
                    if len(text) > 1500:
                        text = text[:1500] + " [...]"
                    all_component_defs[comp_name] = RetrievedItem(
                        text=text,
                        source=doc.path,
                        title=sec.get("title", ""),
                        retrieval_score=0.0,
                        source_tier=doc.tier,
                        source_display_name=doc.display_name,
                    )

    traversals = []
    for cond in conditions:
        keywords = _CONDITION_KEYWORDS.get(cond, [cond.lower().replace("_", " ")])
        cond_items = []

        # Include retrieved items that match this condition or are grounding
        for item in retrieved_items:
            text_lower = (item.text + " " + (item.title or "")).lower()
            if item.source_tier in ("grounding_component", "evidence_rules"):
                cond_items.append(item)
            elif any(kw in text_lower for kw in keywords):
                cond_items.append(item)

        # Condition-specific grounding supplement:
        # Add BSV component definitions only for components that are active
        # (score > 0) in this condition's literature profile. This ensures
        # each condition gets different motif-bearing evidence.
        if all_component_defs and bsv_profile and bsv_profile.get("available"):
            profiles = bsv_profile.get("profiles", {})
            cond_profile = profiles.get(cond, {})
            # Sort by score descending, take top 4 active components
            active_comps = sorted(
                [(comp, score) for comp, score in cond_profile.items() if score > 0],
                key=lambda x: -x[1],
            )[:4]
            seen_titles = {it.title for it in cond_items}
            for comp_name, _ in active_comps:
                if comp_name in all_component_defs:
                    comp_item = all_component_defs[comp_name]
                    if comp_item.title not in seen_titles:
                        cond_items.append(comp_item)
                        seen_titles.add(comp_item.title)

        # Deduplicate
        seen_ids = set()
        deduped = []
        for item in cond_items:
            item_id = id(item)
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                deduped.append(item)
        cond_items = deduped

        # Build per-condition motif map
        cond_mtb = map_evidence_to_motifs_themes_bsv(cond_items)

        # Build per-condition BSV profile
        cond_bsv = None
        if bsv_profile and bsv_profile.get("available"):
            profiles = bsv_profile.get("profiles", {})
            deltas = bsv_profile.get("deltas", {})
            if cond in profiles:
                cond_bsv = {
                    "conditions": [cond],
                    "profiles": {cond: profiles[cond]},
                    "deltas": {cond: deltas[cond]} if cond in deltas else {},
                    "available": True,
                    "notes": "",
                }

        cond_graph = build_trust_graph(
            query=f"{query} [{cond.replace('_', ' ')}]",
            retrieved_items=cond_items,
            motif_theme_map=cond_mtb,
            bsv_profile=cond_bsv,
        )

        traversals.append({
            "condition": cond,
            "graph": cond_graph,
            "n_evidence": len(cond_items),
        })

    return traversals
