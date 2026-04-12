"""GAIRA BSV v1 Graph Explainer — traces BSV components back to motifs and evidence.

Every BSV component is graph-explainable. This module generates explanations.
"""

from __future__ import annotations

from graph.bsv_v1_scoring import BSVVector, BSVComparison, BSVComponent, _COMPONENT_MAP


def explain_component(bsv: BSVVector, component_name: str) -> dict:
    """Generate a graph-backed explanation for a BSV component."""
    comp = next((c for c in bsv.components if c.name == component_name), None)
    if not comp:
        return {"error": f"Component '{component_name}' not found"}

    comp_def = _COMPONENT_MAP.get(component_name, {})
    contributing_subfamilies = list(comp_def.get("subfamilies", set()))[:8]
    contributing_families = list(comp_def.get("families", set()))[:5]

    return {
        "component": component_name,
        "normalized_score": comp.normalized_score,
        "raw_score": comp.raw_score,
        "motif_count": comp.motif_count,
        "contributing_motifs": comp.contributing_motifs,
        "dominant_stability": comp.dominant_stability,
        "coverage": comp.coverage_note,
        "mapped_subfamilies": contributing_subfamilies,
        "mapped_families": contributing_families,
        "neo4j_query": _neo4j_query(component_name, comp_def),
    }


def _neo4j_query(component_name: str, comp_def: dict) -> str:
    """Generate a Neo4j Cypher query to inspect a component's supporting subgraph."""
    subfamilies = list(comp_def.get("subfamilies", set()))[:5]
    if not subfamilies:
        families = list(comp_def.get("families", set()))[:3]
        if families:
            fam = families[0]
            return f"""MATCH (m:Motif) WHERE toLower(m.family) CONTAINS '{fam}'
WITH m
MATCH (e:EvidenceRow)-[:PART_OF_MOTIF]->(m)
MATCH (e)-[:HAS_ASSIGNMENT]->(a:Assignment)
OPTIONAL MATCH (a)-[:DIRECTLY_SUPPORTS_THEME]->(t:BiochemicalTheme)
RETURN m, e, a, t LIMIT 30"""
        return f"// No specific subfamilies mapped for {component_name}"

    sf_list = "', '".join(subfamilies[:3])
    return f"""MATCH (m:Motif) WHERE m.subfamily IN ['{sf_list}']
WITH m
MATCH (e:EvidenceRow)-[:PART_OF_MOTIF]->(m)
MATCH (e)-[:HAS_ASSIGNMENT]->(a:Assignment)
OPTIONAL MATCH (a)-[:DIRECTLY_SUPPORTS_THEME]->(t:BiochemicalTheme)
RETURN m, e, a, t LIMIT 30"""


def generate_bsv_explanation(comparison: BSVComparison) -> list[dict]:
    """Generate explanations for all BSV components in a comparison."""
    explanations = []
    q = comparison.query_bsv

    for comp in q.components:
        exp = explain_component(q, comp.name)

        # Add delta info if comparison exists
        if comparison.delta_components:
            delta = next((d for d in comparison.delta_components if d["component"] == comp.name), None)
            if delta:
                exp["delta"] = delta["delta"]
                exp["delta_direction"] = delta["direction"]

        # Confidence
        exp["confidence"] = comp.confidence

        # Generate plain-English summary
        name_clean = comp.name.replace("_", " ")
        conf_tag = f" [{comp.confidence}]" if comp.confidence else ""
        if comp.coverage_note == "absent":
            exp["summary"] = f"{name_clean}: no contributing motifs found.{conf_tag}"
        elif comp.normalized_score > 0.7:
            exp["summary"] = f"{name_clean} is strongly represented (score={comp.normalized_score}), supported by {comp.motif_count} motifs ({', '.join(comp.contributing_motifs[:3])}). Stability: {comp.dominant_stability}.{conf_tag}"
        elif comp.normalized_score > 0.3:
            exp["summary"] = f"{name_clean} has moderate representation (score={comp.normalized_score}), {comp.motif_count} contributing motifs. Stability: {comp.dominant_stability}."
        else:
            exp["summary"] = f"{name_clean} has weak representation (score={comp.normalized_score}). {'No motifs.' if comp.motif_count == 0 else f'{comp.motif_count} sparse motifs.'}"

        explanations.append(exp)

    return explanations
