# Biosample Ingestion Template

Use this checklist when onboarding the first real biosample dataset into GAIRA.

Required fields:

- `dataset_name`
- `dataset_id`
- `dataset_family` = `biosample`
- `source_type`
- `source_location`
- `raw_storage_folder`
- `expected_format`
- `biosample_type`
- `disease_context`
- `labels_expected`
- `spectra_format`
- `parser_name`

Helpful details:

- what files contain metadata
- what files contain spectra
- whether spectra are full arrays, x-y pairs, or peak lists
- whether sample IDs, patient IDs, and replicates are present
- what class labels are expected
- what preprocessing is already applied
- any known missing fields or quirks in the raw files
