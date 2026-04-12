# Phase C1.4 — Sample Type Resolution

## Current Behavior
- Sample type is **inferred** from retrieved evidence, not filtered at query time
- The dominant sample type in the query's evidence pool is reported in:
  - The comparator summary block
  - The query understanding section
  - The full text explanation

## Inference Logic
```python
sample_type_counts = Counter(record["sample_type"] for record in records if record["sample_type"])
dominant = sample_type_counts.most_common(1)[0][0]
```

## Reporting
The inferred sample type appears in the UI as:
- "Dominant sample type: **serum**"

## Limitations
- No query-time filtering (yet). "Compare HCC serum vs healthy serum" is parsed correctly but both sides retrieve all evidence, not just serum evidence
- Mixed matrices (e.g., serum + tissue + EV all present) are not explicitly warned about unless a future enhancement is added

## Future
Query-time sample-type filtering would require Cypher modification:
```cypher
MATCH (e:EvidenceRow)-[:FROM_SAMPLE_TYPE]->(st:SampleType {name: $sample_type})
```
This is architecturally ready but not yet connected to the router.
