# GAIRA_SERUM_CONTEXT

`GAIRA_SERUM_CONTEXT` is a lightweight domain-specific interpretive overlay for serum SERS work.

Purpose:
- keep serum-specific interpretation separate from shared grounding retrieval
- let `GAIRA_GROUNDING` answer what evidence exists
- let `GAIRA_SERUM_CONTEXT` answer how that evidence should be read in serum

Current v1 sources:
- `hcc_serum` benchmark summary
- `hcc_serum` paper-comparison summary
- existing serum dataset and subclass context rows
- `serum_ag_colloids_grounding` family structure
- `serum_ag_colloids_literature_grounding` literature-support chunks

Current v1 scope:
- benchmark and paper-summary context
- serum-specific caveats
- serum-specific evidence-tiering note
- small set of grounded band notes

Non-goals for v1:
- no large external corpus
- no semantic embedding retriever
- no changes to shared grounding ranking
- no changes to serum benchmarks
