# GAIRA Phase C1 — Visual Inspection Queries

These queries are designed for Neo4j Browser. Each produces a meaningful subgraph visualization.

## 1. HCC Disease Module
```cypher
MATCH (c:Condition {name: 'HCC'})<-[:LINKED_TO]-(m:Motif)<-[:PART_OF_MOTIF]-(e:EvidenceRow)-[:HAS_ASSIGNMENT]->(a:Assignment)-[:DIRECTLY_SUPPORTS_THEME]->(t:BiochemicalTheme)
RETURN c, m, e, a, t LIMIT 50
```
**What to see**: Red Condition hub → green Motifs → yellow Evidence → gray Assignments → purple Themes

## 2. Peak 1005 (Phenylalanine)
```cypher
MATCH (pk:Peak {wavenumber_cm: 1005})<-[:REFERS_TO_PEAK]-(e:EvidenceRow)-[:HAS_ASSIGNMENT]->(a:Assignment)
OPTIONAL MATCH (a)-[:DIRECTLY_SUPPORTS_THEME]->(t:BiochemicalTheme)
OPTIONAL MATCH (a)-[:DIRECTLY_SUPPORTS_BIOMOLECULE]->(b:Biomolecule)
RETURN pk, e, a, t, b LIMIT 30
```
**What to see**: Red Peak hub → multiple Evidence rows converging → protein/amino acid themes

## 3. Lipid Theme Network
```cypher
MATCH (t:BiochemicalTheme {name: 'lipid'})<-[:DIRECTLY_SUPPORTS_THEME]-(a:Assignment)<-[:HAS_ASSIGNMENT]-(e:EvidenceRow)-[:OBSERVED_IN]->(c:Condition)
RETURN t, a, e, c LIMIT 40
```
**What to see**: Purple lipid theme → assignments → evidence → conditions (which diseases show lipid changes)

## 4. Amide I Chemistry Bridge
```cypher
MATCH (fg:FunctionalGroup {name: 'amide I'})-[r:INFERRED_SUPPORTS_THEME]->(t:BiochemicalTheme)
RETURN fg, r, t
UNION
MATCH (fg:FunctionalGroup {name: 'amide I'})<-[:HAS_FUNCTIONAL_GROUP]-(a:Assignment)-[:DIRECTLY_SUPPORTS_THEME]->(t:BiochemicalTheme)
RETURN fg, a AS r, t LIMIT 15
```
**What to see**: Orange FG node bridging to purple themes via both inferred (mapping) and direct (evidence) edges

## 5. Multi-Source Agreement
```cypher
MATCH (pk:Peak {wavenumber_cm: 1005})<-[:REFERS_TO_PEAK]-(e:EvidenceRow)<-[:SUPPORTS]-(s:Source)<-[:HAS_SOURCE]-(p:Paper)
RETURN pk, e, s, p LIMIT 20
```
**What to see**: How many independent papers support the same peak — the foundation of GAIRA confidence scoring
