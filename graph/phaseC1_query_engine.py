"""GAIRA Graph Query Engine v1.6 — scope-aware comparator with sample-type/domain filtering.

Returns raw graph result dicts for downstream scoring and explanation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from neo4j import GraphDatabase

from graph.phaseC1_query_router import ParsedQuery


_DEFAULT_URI = "bolt://localhost:7687"
_DEFAULT_USER = "neo4j"
_DEFAULT_PASS = "gaira2026"


def _get_driver(uri=_DEFAULT_URI, user=_DEFAULT_USER, password=_DEFAULT_PASS):
    return GraphDatabase.driver(uri, auth=(user, password))


@dataclass
class GraphResult:
    query_type: str
    query_mode: str
    cypher: str
    viz_cypher: str
    matched_entity: str
    comparator: str = ""
    inferred_sample_type: str = ""
    scope_mode: str = "broad"
    sample_scope: str = ""
    domain_scope: str = ""
    scope_fallback: str = ""   # "" if no fallback, else describes what happened
    motifs: list[dict] = field(default_factory=list)
    themes: list[dict] = field(default_factory=list)
    biomolecules: list[dict] = field(default_factory=list)
    functional_groups: list[dict] = field(default_factory=list)
    direct_support_count: int = 0
    inferred_support_count: int = 0
    total_evidence_rows: int = 0
    source_count: int = 0
    raw_records: list[dict] = field(default_factory=list)
    comparator_records: list[dict] = field(default_factory=list)
    comparator_evidence_count: int = 0
    comparator_source_count: int = 0


# ── Cypher builder ──────────────────────────────────────────

def _build_condition_cypher(entity_param: str, sample_scope: str = "", domain_scope: str = "") -> str:
    """Build a condition query Cypher, optionally filtered by sample type and domain."""
    sample_clause = ""
    if sample_scope:
        sample_clause = f"\nMATCH (e)-[:FROM_SAMPLE_TYPE]->(st:SampleType) WHERE toLower(st.name) = toLower('{sample_scope}')"

    # Domain filtering: if domain is 'liver', restrict to sources containing 'liver' in name
    domain_clause = ""
    if domain_scope:
        domain_clause = f"\nMATCH (e)<-[:SUPPORTS]-(dsrc:Source) WHERE toLower(dsrc.name) CONTAINS toLower('{domain_scope}')"

    return f"""
MATCH (c:Condition)
WHERE toLower(c.name) CONTAINS toLower({entity_param})
WITH c
OPTIONAL MATCH (c)<-[:LINKED_TO]-(m:Motif)<-[:PART_OF_MOTIF]-(e:EvidenceRow)-[:HAS_ASSIGNMENT]->(a:Assignment){sample_clause}{domain_clause}
OPTIONAL MATCH (a)-[:DIRECTLY_SUPPORTS_THEME]->(t:BiochemicalTheme)
OPTIONAL MATCH (a)-[:DIRECTLY_SUPPORTS_BIOMOLECULE]->(b:Biomolecule)
OPTIONAL MATCH (a)-[:HAS_FUNCTIONAL_GROUP]->(fg:FunctionalGroup)
OPTIONAL MATCH (e)-[:REFERS_TO_PEAK]->(pk:Peak)
OPTIONAL MATCH (e)<-[:SUPPORTS]-(src:Source)
OPTIONAL MATCH (e)-[:FROM_SAMPLE_TYPE]->(stout:SampleType)
RETURN c.name AS condition,
       m.node_id AS motif_id, m.subfamily AS motif_subfamily, m.family AS motif_family,
       e.node_id AS evidence_id, e.peak_cm AS peak_cm, e.assignment_level AS level,
       a.cleaned_meaning AS meaning,
       t.name AS theme, b.name AS biomolecule, fg.name AS functional_group,
       pk.wavenumber_cm AS wavenumber, src.name AS source, stout.name AS sample_type
LIMIT 500
"""


_CONDITION_VIZ = """
MATCH (c:Condition)
WHERE toLower(c.name) CONTAINS toLower($entity)
WITH c
MATCH (c)<-[:LINKED_TO]-(m:Motif)<-[:PART_OF_MOTIF]-(e:EvidenceRow)-[:HAS_ASSIGNMENT]->(a:Assignment)
OPTIONAL MATCH (a)-[:DIRECTLY_SUPPORTS_THEME]->(t:BiochemicalTheme)
RETURN c, m, e, a, t LIMIT 60
"""

_COMPARATIVE_VIZ = """
MATCH (c1:Condition) WHERE toLower(c1.name) CONTAINS toLower($entity)
MATCH (c2:Condition) WHERE toLower(c2.name) CONTAINS toLower($comparator)
WITH c1, c2
MATCH (c1)<-[:LINKED_TO]-(m1:Motif)<-[:PART_OF_MOTIF]-(e1:EvidenceRow)-[:HAS_ASSIGNMENT]->(a1:Assignment)
OPTIONAL MATCH (a1)-[:DIRECTLY_SUPPORTS_THEME]->(t1:BiochemicalTheme)
RETURN c1, c2, m1, e1, a1, t1 LIMIT 60
"""

_PEAK_CYPHER = """
MATCH (pk:Peak)
WHERE pk.wavenumber_cm >= $peak_low AND pk.wavenumber_cm <= $peak_high
WITH pk
MATCH (e:EvidenceRow)-[:REFERS_TO_PEAK]->(pk)
MATCH (e)-[:HAS_ASSIGNMENT]->(a:Assignment)
OPTIONAL MATCH (a)-[:DIRECTLY_SUPPORTS_THEME]->(t:BiochemicalTheme)
OPTIONAL MATCH (a)-[:DIRECTLY_SUPPORTS_BIOMOLECULE]->(b:Biomolecule)
OPTIONAL MATCH (a)-[:HAS_FUNCTIONAL_GROUP]->(fg:FunctionalGroup)
OPTIONAL MATCH (e)-[:OBSERVED_IN]->(c:Condition)
OPTIONAL MATCH (e)<-[:SUPPORTS]-(src:Source)
RETURN pk.wavenumber_cm AS wavenumber,
       e.node_id AS evidence_id, e.peak_cm AS peak_cm, e.assignment_level AS level,
       a.cleaned_meaning AS meaning,
       t.name AS theme, b.name AS biomolecule, fg.name AS functional_group,
       c.name AS condition, src.name AS source
LIMIT 300
"""

_PEAK_VIZ = """
MATCH (pk:Peak) WHERE pk.wavenumber_cm >= $peak_low AND pk.wavenumber_cm <= $peak_high
WITH pk
MATCH (e:EvidenceRow)-[:REFERS_TO_PEAK]->(pk)
MATCH (e)-[:HAS_ASSIGNMENT]->(a:Assignment)
OPTIONAL MATCH (a)-[:DIRECTLY_SUPPORTS_THEME]->(t:BiochemicalTheme)
RETURN pk, e, a, t LIMIT 40
"""

_THEME_CYPHER = """
MATCH (t:BiochemicalTheme) WHERE toLower(t.name) CONTAINS toLower($entity)
WITH t
MATCH (a:Assignment)-[:DIRECTLY_SUPPORTS_THEME]->(t)
MATCH (e:EvidenceRow)-[:HAS_ASSIGNMENT]->(a)
OPTIONAL MATCH (a)-[:DIRECTLY_SUPPORTS_BIOMOLECULE]->(b:Biomolecule)
OPTIONAL MATCH (a)-[:HAS_FUNCTIONAL_GROUP]->(fg:FunctionalGroup)
OPTIONAL MATCH (e)-[:OBSERVED_IN]->(c:Condition)
OPTIONAL MATCH (e)<-[:SUPPORTS]-(src:Source)
RETURN t.name AS theme, e.node_id AS evidence_id, e.peak_cm AS peak_cm, e.assignment_level AS level,
       a.cleaned_meaning AS meaning, b.name AS biomolecule, fg.name AS functional_group,
       c.name AS condition, src.name AS source
LIMIT 300
"""

_THEME_VIZ = """
MATCH (t:BiochemicalTheme) WHERE toLower(t.name) CONTAINS toLower($entity)
WITH t
MATCH (a:Assignment)-[:DIRECTLY_SUPPORTS_THEME]->(t)
MATCH (e:EvidenceRow)-[:HAS_ASSIGNMENT]->(a)
RETURN t, a, e LIMIT 40
"""

_CHEMISTRY_CYPHER = """
MATCH (fg:FunctionalGroup) WHERE toLower(fg.name) CONTAINS toLower($entity)
WITH fg
OPTIONAL MATCH (fg)-[inf:INFERRED_SUPPORTS_THEME]->(t:BiochemicalTheme)
OPTIONAL MATCH (a:Assignment)-[:HAS_FUNCTIONAL_GROUP]->(fg)
OPTIONAL MATCH (a)-[:DIRECTLY_SUPPORTS_THEME]->(dt:BiochemicalTheme)
OPTIONAL MATCH (a)-[:DIRECTLY_SUPPORTS_BIOMOLECULE]->(b:Biomolecule)
OPTIONAL MATCH (e:EvidenceRow)-[:HAS_ASSIGNMENT]->(a)
RETURN fg.name AS functional_group,
       t.name AS inferred_theme, inf.weight AS inferred_weight, inf.confidence AS inferred_confidence,
       dt.name AS direct_theme, b.name AS biomolecule,
       e.node_id AS evidence_id, e.peak_cm AS peak_cm, a.cleaned_meaning AS meaning
LIMIT 300
"""

_CHEMISTRY_VIZ = """
MATCH (fg:FunctionalGroup) WHERE toLower(fg.name) CONTAINS toLower($entity)
WITH fg
OPTIONAL MATCH (fg)-[r:INFERRED_SUPPORTS_THEME]->(t:BiochemicalTheme)
RETURN fg, t, r LIMIT 30
"""


def _infer_sample_type(records: list[dict]) -> str:
    st_counts = Counter(r.get("sample_type") for r in records if r.get("sample_type"))
    return st_counts.most_common(1)[0][0] if st_counts else "unknown"


def _build_result(parsed, records, cypher, viz, comparator_records=None, scope_fallback=""):
    themes = Counter(r.get("theme") or r.get("direct_theme") for r in records if r.get("theme") or r.get("direct_theme"))
    bios = Counter(r.get("biomolecule") for r in records if r.get("biomolecule"))
    fgs = Counter(r.get("functional_group") for r in records if r.get("functional_group"))
    motifs_set = {}
    for r in records:
        mid = r.get("motif_id")
        if mid and mid not in motifs_set:
            motifs_set[mid] = {"motif_id": mid, "subfamily": r.get("motif_subfamily",""), "family": r.get("motif_family","")}
    sources = set(r.get("source") for r in records if r.get("source"))
    evidence_ids = set(r.get("evidence_id") for r in records if r.get("evidence_id"))
    direct = sum(1 for r in records if r.get("theme") or r.get("direct_theme"))
    inferred = sum(1 for r in records if r.get("inferred_theme"))
    entity_str = parsed.entities[0] if parsed.entities else (str(parsed.peak_cm) if parsed.peak_cm else "")

    comp_evidence = set(r.get("evidence_id") for r in (comparator_records or []) if r.get("evidence_id"))
    comp_sources = set(r.get("source") for r in (comparator_records or []) if r.get("source"))

    viz_display = viz
    for k, v in [("entity", entity_str), ("comparator", parsed.comparator),
                 ("peak_low", str(int(parsed.peak_cm or 0)-5)), ("peak_high", str(int(parsed.peak_cm or 0)+5))]:
        viz_display = viz_display.replace(f"${k}", f"'{v}'" if not str(v).lstrip('-').isdigit() else str(v))

    return GraphResult(
        query_type=parsed.query_type, query_mode=parsed.query_mode,
        cypher=cypher, viz_cypher=viz_display,
        matched_entity=entity_str, comparator=parsed.comparator,
        inferred_sample_type=_infer_sample_type(records),
        scope_mode=parsed.scope_mode, sample_scope=parsed.sample_scope,
        domain_scope=parsed.domain_scope, scope_fallback=scope_fallback,
        motifs=list(motifs_set.values()),
        themes=[{"theme": t, "count": c} for t, c in themes.most_common(10)],
        biomolecules=[{"biomolecule": b, "count": c} for b, c in bios.most_common(10)],
        functional_groups=[{"functional_group": f, "count": c} for f, c in fgs.most_common(10)],
        direct_support_count=direct, inferred_support_count=inferred,
        total_evidence_rows=len(evidence_ids), source_count=len(sources),
        raw_records=records, comparator_records=comparator_records or [],
        comparator_evidence_count=len(comp_evidence), comparator_source_count=len(comp_sources),
    )


def execute_query(parsed, uri=_DEFAULT_URI, user=_DEFAULT_USER, password=_DEFAULT_PASS):
    driver = _get_driver(uri, user, password)
    sample = parsed.sample_scope
    domain = parsed.domain_scope
    scope_fallback = ""

    if parsed.query_type == "condition":
        entity = parsed.entities[0] if parsed.entities else ""
        with driver.session() as session:
            # Build scoped query
            cypher = _build_condition_cypher("$entity", sample, domain)
            records = [dict(r) for r in session.run(cypher, entity=entity)]

            # Fallback: if scoped query returns too few results, broaden
            if len(set(r.get("evidence_id") for r in records if r.get("evidence_id"))) < 3:
                if sample or domain:
                    scope_fallback = f"scoped query ({sample or ''} / {domain or ''}) returned < 3 rows; broadened to unscoped"
                    cypher = _build_condition_cypher("$entity")
                    records = [dict(r) for r in session.run(cypher, entity=entity)]

            comp_records = []
            if parsed.query_mode == "pairwise" and parsed.comparator:
                comp_cypher = _build_condition_cypher("$entity", sample, domain)
                comp_records = [dict(r) for r in session.run(comp_cypher, entity=parsed.comparator)]
                # Fallback for comparator too
                if len(set(r.get("evidence_id") for r in comp_records if r.get("evidence_id"))) < 2:
                    if sample or domain:
                        scope_fallback += "; comparator broadened"
                        comp_cypher = _build_condition_cypher("$entity")
                        comp_records = [dict(r) for r in session.run(comp_cypher, entity=parsed.comparator)]

            elif parsed.query_mode == "one_vs_rest":
                ovr_cypher = f"""
                MATCH (c:Condition) WHERE NOT toLower(c.name) CONTAINS toLower($entity)
                WITH c
                OPTIONAL MATCH (c)<-[:LINKED_TO]-(m:Motif)<-[:PART_OF_MOTIF]-(e:EvidenceRow)-[:HAS_ASSIGNMENT]->(a:Assignment)
                OPTIONAL MATCH (a)-[:DIRECTLY_SUPPORTS_THEME]->(t:BiochemicalTheme)
                OPTIONAL MATCH (a)-[:DIRECTLY_SUPPORTS_BIOMOLECULE]->(b:Biomolecule)
                OPTIONAL MATCH (a)-[:HAS_FUNCTIONAL_GROUP]->(fg:FunctionalGroup)
                OPTIONAL MATCH (e)<-[:SUPPORTS]-(src:Source)
                RETURN c.name AS condition,
                       m.node_id AS motif_id, m.subfamily AS motif_subfamily, m.family AS motif_family,
                       e.node_id AS evidence_id, e.peak_cm AS peak_cm, e.assignment_level AS level,
                       a.cleaned_meaning AS meaning, t.name AS theme, b.name AS biomolecule,
                       fg.name AS functional_group, src.name AS source
                LIMIT 1000
                """
                comp_records = [dict(r) for r in session.run(ovr_cypher, entity=entity)]

        viz = _COMPARATIVE_VIZ if parsed.comparator else _CONDITION_VIZ
        driver.close()
        return _build_result(parsed, records, cypher, viz, comp_records, scope_fallback)

    elif parsed.query_type == "peak":
        pk = parsed.peak_cm or 1000
        with driver.session() as session:
            records = [dict(r) for r in session.run(_PEAK_CYPHER, peak_low=int(pk)-5, peak_high=int(pk)+5)]
        driver.close()
        return _build_result(parsed, records, _PEAK_CYPHER, _PEAK_VIZ)

    elif parsed.query_type == "theme":
        entity = parsed.entities[0] if parsed.entities else ""
        with driver.session() as session:
            records = [dict(r) for r in session.run(_THEME_CYPHER, entity=entity)]
        driver.close()
        return _build_result(parsed, records, _THEME_CYPHER, _THEME_VIZ)

    elif parsed.query_type == "chemistry":
        entity = parsed.entities[0] if parsed.entities else ""
        with driver.session() as session:
            records = [dict(r) for r in session.run(_CHEMISTRY_CYPHER, entity=entity)]
        driver.close()
        return _build_result(parsed, records, _CHEMISTRY_CYPHER, _CHEMISTRY_VIZ)

    driver.close()
    return GraphResult(query_type="unknown", query_mode="single", cypher="", viz_cypher="", matched_entity="")
