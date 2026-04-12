# BSV v2 — Graph Explanation Panel Notes

## Change
Each BSV component now has an expandable explanation panel showing:
1. **Summary sentence** — plain-English description of the component's state
2. **Confidence label** — strong/moderate/weak/exploratory with colored icon
3. **Contributing motifs** — the actual motif subfamilies that drove the score
4. **Delta** — if comparative, the direction and magnitude of the shift
5. **Neo4j Cypher query** — for inspecting the component's subgraph in Neo4j Browser

## Example
> 🟢 **Aromatic Amino Acid — score 1.00**
> "aromatic amino acid is strongly represented (score=1.0), supported by 3 motifs (tryptophan, phenylalanine, tyrosine). Stability: STABLE. [strong]"
> 
> Motifs: tryptophan, phenylalanine, tyrosine
> Delta: +0.245 (up)
> ```cypher
> MATCH (m:Motif) WHERE m.subfamily IN ['tryptophan', 'phenylalanine', 'tyrosine'] ...
> ```

## Design Principle
The user should be able to answer "why is this component high/low?" by reading the explanation without leaving Streamlit. The Neo4j query is for deeper inspection.
