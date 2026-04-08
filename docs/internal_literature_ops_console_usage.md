# Internal Literature Ops Console

Run from the GAIRA repo root:

```bash
PYTHONPATH=src .venv/bin/streamlit run streamlit_apps/literature_ops_console.py
```

What it is for:

- use the Candidate Queue as the main work surface for papers not yet ingested
- use the Live Evidence Registry as the main record of papers that already changed the warehouse
- search structured discovery sources for new papers
- see duplicate/existing status immediately
- rescue blocked papers by uploading externally obtained manuscript/SI/source-data files
- trigger existing controlled-ingestion pipelines only after assets are truth-validated
- inspect paper-level evidence impact from the Live Evidence Registry

What it is not for:

- public demo use
- publisher login automation
- broad uncontrolled ingestion
- bypassing the current evidence/ontology/neighborhood/motif pipelines

Operational notes:

- search uses the existing Crossref / Europe PMC / PubMed helpers
- duplicate detection uses DOI exact match first, then title similarity fallback plus local-corpus / processed / blocked / queue checks
- the main workflow is: `Candidate Queue -> rescue/fetch assets -> ingest paper -> Live Evidence Registry`
- the UI is table-first and row-action driven; primary operations no longer depend on paper dropdown selectors
- Candidate Queue states are operator-facing and actionable:
  - `READY NOW`
  - `OA FETCHABLE`
  - `NEEDS RESOLUTION`
  - `BLOCKED`
  - `LOW VALUE`
  - `DISCARDED`
- manual rescue uploads write files locally, validate them truthfully, refresh readiness, refresh queue placement, and move rescued papers out of the active blocked lane
- only papers with genuinely usable assets should be treated as ready to ingest
- row-level OA fetch now reuses the existing OA truth-validation path for one paper at a time
- row-level ingest is available directly from each `READY NOW` row, with optional bulk ingest for selected ready rows
- OA ingest actions call the existing `run_oa_ready_controlled_ingest.py` pipeline
- non-OA ready-paper ingest remains available through the existing `run_ready_paper_controlled_ingest.py` wrapper

Top-level tabs:

- `Overview`: synchronized counts and definitions
- `Candidate Queue`: primary operator table and per-paper action surface
- `Search`: structured discovery with duplicate/existing awareness
- `Blocked Rescue`: manual asset rescue only
- `Live Evidence Registry`: one row per ingested paper/source with impact summaries
- `Advanced / Debug`: ready-only queue, asset inventory, and activity/debug views
