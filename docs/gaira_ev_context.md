# GAIRA_EV_CONTEXT

`GAIRA_EV_CONTEXT` is a lightweight domain-specific interpretive overlay for EV SERS work.

Purpose:
- keep EV-specific interpretation separate from shared grounding retrieval
- let `GAIRA_GROUNDING` answer what evidence exists
- let `GAIRA_EV_CONTEXT` answer how that evidence should be read in EV-domain work

Current v1 sources:
- `small2023_ev` embedding benchmark summaries and validation summaries
- current default-embedding status note
- existing EV dataset and subclass context rows
- finalized diabetes plasma EV weak-label framing

Current v1 scope:
- `small2023_ev` benchmark hierarchy and benchmark caveats
- EV dataset-framing notes for `small2023_ev`, `shine_ev_sers`, and `diabetes_plasma_ev_sers`
- EV cross-substrate comparability caveat
- weak-label interpretation note for the diabetes plasma EV dataset

Non-goals for v1:
- no large external EV literature corpus
- no semantic embedding retriever
- no changes to shared grounding ranking
- no changes to EV benchmarks or default embedding selection
