## GAIRA Domain Context v1

This seed layer stores lightweight measurement and provenance context separate from search logic and separate from the knowledge-layer `dataset_context` table.

Files:
- `data/raw/context/dataset_domain_context_v1.csv`
- `data/raw/context/subclass_domain_context_v1.csv`

### `dataset_domain_context_v1.csv`
Required columns:
- `dataset_id`
- `dataset_family`
- `context_level`
- `biosample_type`
- `measurement_mode`
- `default_substrate_type`
- `default_substrate_material`
- `substrate_vendor`
- `instrument_context`
- `default_preprocessing_family`
- `notes`

Use this for dataset-wide statements such as:
- reference vs biosample
- Raman vs SERS
- whether the dataset spans multiple probe families
- broad instrument/substrate uncertainty

### `subclass_domain_context_v1.csv`
Required columns:
- `dataset_id`
- `subclass_label`
- `context_level`
- `biosample_type`
- `measurement_mode`
- `substrate_type`
- `substrate_material`
- `substrate_vendor`
- `substrate_batch_id`
- `probe_family`
- `spectral_axis_family`
- `cross_domain_intensity_comparable`
- `preprocessing_family`
- `notes`

Use this for raw-family or subclass-specific provenance such as:
- probe family
- substrate batch identity if grounded
- raw axis family
- whether raw intensity values are comparable across families
- preprocessing family

Guidelines:
- keep values cautious and provenance-preserving
- prefer `unknown_*` or `not_applicable_or_unknown` instead of inventing unsupported substrate facts
- do not encode richer biological meaning than the released files support
- this layer is for routing, interpretation context, and future invariant embedding experiments
