# BSV v1 — Graph-Backed Explanation

## Principle
Every BSV component must be traceable to graph evidence. The explainer generates per-component summaries with:
- Contributing motifs (by name)
- Motif count and stability
- Coverage assessment
- Neo4j Cypher query for visual inspection

## Example Explanation

For `aromatic_amino_acid` with score 1.0:
> "Aromatic amino acid is strongly represented (score=1.0), supported by 3 motifs (tryptophan, phenylalanine, tyrosine). Stability: STABLE."

For `membrane_lipid` with score 0.3:
> "Membrane lipid has moderate representation (score=0.3), 1 contributing motif. Stability: MIXED."

## Neo4j Inspection Queries

### Inspect a BSV component's supporting subgraph:
```cypher
MATCH (m:Motif) WHERE m.subfamily IN ['tryptophan', 'phenylalanine', 'tyrosine']
WITH m
MATCH (e:EvidenceRow)-[:PART_OF_MOTIF]->(m)
MATCH (e)-[:HAS_ASSIGNMENT]->(a:Assignment)
OPTIONAL MATCH (a)-[:DIRECTLY_SUPPORTS_THEME]->(t:BiochemicalTheme)
RETURN m, e, a, t LIMIT 30
```

### See which conditions link to a component's motifs:
```cypher
MATCH (m:Motif)-[:LINKED_TO]->(c:Condition)
WHERE m.subfamily IN ['adenine', 'guanine']
RETURN m.subfamily, c.name, count(*) AS links
ORDER BY links DESC
```
