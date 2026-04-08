# Developer Note: Literature Ops Console

The Streamlit console is intentionally thin and wraps existing GAIRA logic rather than reimplementing the literature pipeline.

Wrapped / reused components:

- `src/gaira/evidence_v1/literature_acquisition_pipeline.py`
  - `_crossref_search`
  - `_europepmc_search`
  - `_pubmed_search`
  - `_dedupe_records`
  - `_match_local_corpus`
  - `_mark_existing_processing`
  - `_triage`
- `src/gaira/evidence_v1/literature_asset_truth_oa.py`
  - `_validate_local_asset`
  - `_queue_partition_for_row`
  - `_upsert_candidate`
- `scripts/evidence_v1/run_oa_ready_controlled_ingest.py`
  - wrapped by app action for selected OA-ready papers
- `scripts/evidence_v1/run_ready_paper_controlled_ingest.py`
  - wrapped as the broader existing ready-paper pipeline

New app-specific code:

- `src/gaira/evidence_v1/literature_ops_console.py`
  - DuckDB access helpers
  - candidate-queue operational-state derivation
  - duplicate/existing paper detection
  - manual upload + readiness refresh
  - post-ingestion state refresh
  - row-level OA fetch wrapper on top of the OA truth-validation path
  - live evidence registry summaries
  - subprocess wrappers for ingestion triggers
- `streamlit_apps/literature_ops_console.py`
  - internal Streamlit UI
  - two-registry operator workflow
  - table-first row-action workflow with inline expanders instead of primary selectbox inspection

Important implementation choices:

- no credential handling or login automation
- duplicate detection is not fuzzy-only; DOI exact match is checked first
- asset resolution and evidence ingestion are treated as separate operator steps
- Candidate Queue shows only non-ingested papers and uses front-facing actionable states:
  - `READY NOW`
  - `OA FETCHABLE`
  - `NEEDS RESOLUTION`
  - `BLOCKED`
  - `LOW VALUE`
  - `DISCARDED`
- Live Evidence Registry shows only ingested papers/sources that already contributed structured evidence
- manual uploads update `literature.paper_asset_resolution`, `literature.blocked_assets`, `literature.processing_queue`, `literature.queue_partition`, and local-corpus presence on `literature.candidate_papers`
- the app does not add a new ingestion pipeline; it triggers the existing ready/OA ingest scripts and then refreshes candidate/live registry state
- undo ingest is not yet fully implemented; the UI exposes rollback-ready metadata at source level but does not perform destructive rollback
- the activity log is built from currently available DB signals rather than a new persistent ops-log schema
