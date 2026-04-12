# Neo4j Inspection Instructions

## The embedded graph preview in the GAIRA demo is simplified.

For full-depth graph inspection:

## Step 1: Open Neo4j Browser
Navigate to http://localhost:7474 in your web browser.

## Step 2: Copy the Cypher Query
In the GAIRA demo, expand the "Neo4j Visualization Query" panel and copy the Cypher query.

## Step 3: Paste and Run
Paste the query into Neo4j Browser's query bar and press play.

## Step 4: Switch to Graph View
Click the "Graph" tab (not "Table" or "Text") to see the visual graph.

## Step 5: Explore
- **Double-click** a node to expand its neighborhood
- **Click** a node to see its properties
- **Drag** nodes to rearrange the layout
- Use the node label selector (database icon) to customize captions and colors

## Recommended Caption Settings
- EvidenceRow → `peak_cm`
- Assignment → `cleaned_meaning`
- All name-bearing nodes → `name`

## Deeper Queries
For exploration beyond the demo query, try:
```cypher
// Expand from a specific motif
MATCH (m:Motif {subfamily: 'tryptophan'})-[:LINKED_TO]->(c:Condition)
OPTIONAL MATCH (e:EvidenceRow)-[:PART_OF_MOTIF]->(m)
RETURN m, c, e LIMIT 40

// Find all evidence for a specific peak
MATCH (pk:Peak {wavenumber_cm: 1005})<-[:REFERS_TO_PEAK]-(e:EvidenceRow)
RETURN pk, e LIMIT 30
```
