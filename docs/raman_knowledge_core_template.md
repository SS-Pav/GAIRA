# Raman Knowledge Core Template

`raman_knowledge_core` is a local curated knowledge package for GAIRA's first structured Raman knowledge layer.

Place the files under:

`/Volumes/SSD_Rad/GAIRA_DATA/raw/raman_knowledge_core`

Required files:

`sources.csv`

Required columns:
- `source_id`
- `dataset_id`
- `source_type`
- `title`
- `authors`
- `year`
- `journal`
- `doi`
- `url`
- `citation`
- `license`
- `notes`

Purpose:
- One row per paper, review, textbook chapter, dataset note, or curated source package.
- This is the provenance table for all other knowledge rows.

`peak_assignments.csv`

Required columns:
- `assignment_id`
- `source_id`
- `dataset_id`
- `peak_cm`
- `tolerance_cm`
- `assigned_molecule`
- `assigned_group`
- `matrix_context`
- `confidence_text`
- `evidence_text`

Purpose:
- One row per literature-derived Raman peak assignment.
- `assigned_group` should stay broad when the evidence is broad, for example `proteins`, `lipids`, `nucleic acids`, `amide`, `phenyl ring`, `carbohydrates`.
- `evidence_text` should be short and traceable to the source.

`biomarker_claims.csv`

Required columns:
- `claim_id`
- `source_id`
- `dataset_id`
- `biomarker_name`
- `disease_context`
- `sample_type`
- `spectral_region`
- `claim_text`
- `evidence_strength`
- `notes`

Purpose:
- One row per cautious biomarker or spectral-association claim.
- Keep claims descriptive and source-grounded.

`confounder_notes.csv`

Required columns:
- `confounder_id`
- `source_id`
- `dataset_id`
- `confounder_name`
- `applies_to`
- `note_text`
- `mitigation_text`

Purpose:
- One row per confounder or interpretation caveat.
- Examples include substrate effects, fluorescence, hemoglobin contamination, protein background, drying artifacts, or matrix overlap.

Optional file:

`knowledge_chunks.csv`

Required columns if present:
- `chunk_id`
- `source_id`
- `dataset_id`
- `section`
- `chunk_text`
- `chunk_order`
- `page_label`
- `metadata_json`

Purpose:
- Retrieval-ready short text chunks for later RAG workflows.
- This file is optional in v1.

Optional but strongly recommended file:

`semantic_regions.csv`

Required columns if present:
- `region_id`
- `dataset_id`
- `region_label`
- `region_min_cm`
- `region_max_cm`
- `dominant_group`
- `secondary_groups`
- `typical_examples`
- `interpretation_note`
- `caution_note`

Purpose:
- One row per curated, explicit Raman interpretation region.
- This table is the preferred semantic-region ontology for cautious downstream interpretation.
- Keep the ontology compact and broad. Favor scientist-readable region concepts over dense peak lists.

Optional but strongly recommended file:

`dataset_context.csv`

Required columns if present:
- `context_id`
- `dataset_id`
- `target_dataset_id`
- `modality`
- `sample_type`
- `measurement_state`
- `substrate_type`
- `enhancement_mode`
- `known_biases`
- `region_caution_450_700`
- `region_caution_700_900`
- `region_caution_900_1100`
- `region_caution_1100_1300`
- `region_caution_1300_1500`
- `region_caution_1500_1700`
- `interpretation_note`
- `do_not_overclaim_note`

Purpose:
- One row per target dataset that needs explicit acquisition-context interpretation guidance.
- This is where SERS-, EV-, serum-, or substrate-conditioned caution can be stored in a structured form.
- Keep wording broad, cautious, and dataset-aware rather than overly specific.

Suggested use:
- `target_dataset_id = shine_ev_sers`
- `modality = SERS`
- `sample_type = extracellular vesicles`
- region caution fields should stay compact and directly usable by deterministic downstream reports.

Suggested examples:
- low-wavenumber mixed biosample region
- nucleic-acid / choline / ring mode region
- aromatic + phosphate + carbohydrate overlap region
- amide III / carbohydrate / lipid overlap region
- CH deformation lipid-protein overlap region
- aromatic amino-acid / base overlap region
- amide I protein-rich region

Notes:
- Use UTF-8 CSV files with headers.
- Extra columns are allowed, but GAIRA v1 will read only the required columns above.
- `dataset_id` should be set to `raman_knowledge_core`. The parser will also enforce that value during ingestion.
- Do not include fabricated claims or unsupported peak assignments.
