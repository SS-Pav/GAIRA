# Knowledge Ingestion Template

Use this checklist when onboarding the first real literature or RAG source into GAIRA.

Required fields:

- `dataset_name`
- `dataset_id`
- `dataset_family` = `knowledge`
- `source_type`
- `source_location`
- `raw_storage_folder`
- `expected_format`
- `parser_name`

Helpful details:

- whether the source is a paper, review, textbook, or curated notes
- source metadata available: title, authors, year, journal, DOI, URL, citation
- whether text needs chunking
- whether peak assignments are tabular
- whether biomarker claims are explicit or narrative
- whether confounder notes are explicit or inferred
- page labels or section boundaries if available
