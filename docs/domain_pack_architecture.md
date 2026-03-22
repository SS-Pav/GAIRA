# Domain Pack Architecture

GAIRA remains a single shared platform, but now carries an explicit domain-pack layer that separates:

1. biosample foundation packs used for inference and benchmarking
2. shared grounding packs used for molecular attribution, analog matching, and interpretation support

Current packs:

- `GAIRA_EV`
  - role: `ev_foundation_pack`
  - intended sample types: extracellular vesicles
  - datasets:
    - `small2023_ev`
    - `shine_ev_sers`
    - `diabetes_plasma_ev_sers`
  - default embedding: `small2023_ev_v1`

- `GAIRA_SERUM`
  - role: `serum_foundation_pack`
  - intended sample types: serum
  - datasets:
    - `hcc_serum` (holdout; skipped by default)
    - `serum_ag_colloids`
    - `serum_protocol_comparison`
    - `cspp_serum`
    - `ergothioneine_serum`
  - holdout datasets:
    - `hcc_serum`
  - default embedding: `none_yet`

- `GAIRA_GROUNDING`
  - role: `shared_grounding_pack`
  - status: `active_scaffold`
  - datasets:
    - `ramanbiolib`
    - `serum_ag_colloids_grounding`
    - `serum_ag_colloids_literature_grounding`
    - `sers_fingerprint_workingpaper_support`
    - `sers24_metabolite_support`
  - default embedding: none

Design intent:

- Keep shared infrastructure, search, ingestion, and dataset registries in one repo.
- Allow biosample packs to stay focused on domain-specific datasets and embeddings.
- Keep shared grounding assets separate from biosample inference packs.
- Make pack membership, intended scope, and default embeddings explicit in configuration.

Current scope:

- This layer is configuration only.
- It does not introduce dynamic routing, inference switching, or pack-specific pipeline behavior.
- It does not change RamanBioLib search behavior, SHINE behavior, serum behavior, or any benchmark outputs.

Relationship to embedding registry:

- The embedding registry still defines benchmark roles and per-dataset default embeddings.
- The domain-pack registry sits above that and defines which embedding a biosample pack should treat as its default working embedding.
- Today, `GAIRA_EV` points to `small2023_ev_v1`.
- `GAIRA_GROUNDING` intentionally does not define an embedding.

Grounding versus inference:

- `GAIRA_EV` and `GAIRA_SERUM` are biosample packs.
- `GAIRA_GROUNDING` is a shared support layer for molecular and literature-backed interpretation.
- `ramanbiolib` should now be treated as a grounding asset rather than as a member of biosample packs.
- `hcc_serum` is retained in the serum pack as a holdout evaluation asset, but current default rebuild,
  download, and ingest flows skip it unless explicitly requested.

Next use:

- This three-pack architecture is ready for future serum and EV expansion while keeping grounding resources explicit and reusable across domains.
