# Domain Pack Architecture

GAIRA remains a single shared platform, but now carries an explicit domain-pack layer for pack-scoped defaults and dataset grouping.

Current packs:

- `GAIRA_EV`
  - role: `ev_foundation_pack`
  - intended sample types: extracellular vesicles
  - datasets:
    - `ramanbiolib`
    - `small2023_ev`
    - `shine_ev_sers`
    - `diabetes_plasma_ev_sers`
  - default embedding: `small2023_ev_v1`

- `GAIRA_SERUM`
  - role: `serum_foundation_pack`
  - status: scaffold only
  - intended sample types: serum
  - datasets:
    - `ramanbiolib`
  - default embedding: `none_yet`

Design intent:

- Keep shared infrastructure, search, and ingestion in one repo.
- Allow future domain-specific foundation packs without splitting GAIRA into separate codebases.
- Make pack membership, intended sample scope, and default embedding explicit in configuration.

Current scope:

- This layer is configuration only.
- It does not introduce dynamic routing, inference switching, or pack-specific pipeline behavior.
- It does not change RamanBioLib search behavior, SHINE behavior, or any existing benchmark outputs.

Relationship to embedding registry:

- The embedding registry still defines benchmark roles and per-dataset default embeddings.
- The domain-pack registry sits above that and defines which embedding a pack should treat as its default working embedding.
- Today, `GAIRA_EV` points to `small2023_ev_v1`.

Shared datasets:

- `ramanbiolib` is intentionally shared across packs as the analog-reference base layer.

Next use:

- This architecture is ready for future serum dataset onboarding and future pack-specific inference selection logic if GAIRA needs it later.
