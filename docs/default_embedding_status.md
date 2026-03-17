# Default Embedding Status

GAIRA now carries an explicit embedding registry for benchmark-role selection.

Current `small2023_ev` hierarchy:

- `small2023_ev_v1`
  - role: `production_default`
  - meaning: default working GAIRA embedding for downstream inference, search, and interpretation
- `small2023_ev_v2`
  - role: `research_upper_bound`
  - meaning: strong transductive supervised upper-bound benchmark, not the default deployment-like choice
- `small2023_ev_v3`
  - role: `strict_transfer_test`
  - meaning: strict source-only negative-result benchmark kept as a stress test, not a downstream default

Why v1 is the default:

- v1 is the current project-approved working default for downstream use.
- v2 remains valuable, but it is explicitly labeled as a transductive upper-bound rather than a default.
- v3 remains valuable as a strict deployment-style negative result.

This pass only adds explicit configuration and documentation. It does not change:

- RamanBioLib search behavior
- SHINE ingestion or explanation behavior
- `small2023_ev` ingestion or processed-spectrum generation
- any benchmark output files already generated
