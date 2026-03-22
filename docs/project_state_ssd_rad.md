# GAIRA Project State on SSD_Rad

Current active GAIRA data root:

- `/Volumes/SSD_Rad/GAIRA_DATA`

Current live database:

- `/Volumes/SSD_Rad/GAIRA_DATA/interim/gaira.duckdb`

Holdout datasets:

- `hcc_serum`
  - registry status: `holdout`
  - pack membership: `GAIRA_SERUM`
  - default behavior: skipped in rebuild, download, and ingest unless explicitly requested
  - current SSD_Rad state: not ingested by design

Recovered local knowledge package:

- `raman_knowledge_core`
  - current raw package: `/Volumes/SSD_Rad/GAIRA_DATA/raw/raman_knowledge_core/`
  - recovery basis: reconstructed from the older local GAIRA DuckDB before SSD_Rad verification
  - current live counts:
    - `knowledge_sources`: `5`
    - `peak_assignments`: `84`
    - `biomarker_claims`: `12`
    - `confounder_notes`: `12`
    - `knowledge_chunks`: `96`
    - `semantic_regions`: `11`
    - `dataset_context`: `1`

Current pack readiness:

- `GAIRA_EV`: ready
  - active members: `small2023_ev`, `shine_ev_sers`, `diabetes_plasma_ev_sers`
- `GAIRA_SERUM`: usable with caveat
  - active rebuilt members: `serum_ag_colloids`, `serum_protocol_comparison`, `cspp_serum`, `ergothioneine_serum`
  - holdout member: `hcc_serum`
- `GAIRA_GROUNDING`: ready
  - direct grounding: `ramanbiolib`, `serum_ag_colloids_grounding`
  - support grounding: `serum_ag_colloids_literature_grounding`, `sers_fingerprint_workingpaper_support`, `sers24_metabolite_support`

Current context readiness:

- `GAIRA_EV_CONTEXT`: usable with caveat
- `GAIRA_SERUM_CONTEXT`: usable with caveat

Current inference readiness:

- shared grounding retrieval: ready
- domain-aware reranking: ready
- integrated inference: ready

Operational note:

- Active rebuild and ingestion paths are SSD_Rad-backed.
- Legacy forensic/recovery scripts may still mention older local artifacts, but the current live GAIRA path is SSD_Rad-only.
