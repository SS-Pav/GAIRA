# GAIRA Search Baseline

## Stable baseline

The current GAIRA search baseline does the following:

1. Load a query CSV spectrum.
2. Crop the spectrum to 450-1800 cm-1.
3. Min-max normalize intensity to 0-1.
4. Detect query peaks.
5. Retrieve matching reference peaks from DuckDB.
6. Score peak-based reference candidates.
7. Rerank top reference candidates using full-spectrum similarity from `reference_spectra`.
8. Aggregate reranked reference candidates to the component level.
9. Summarize biochemical classes and generate a short interpretation.

## Current outputs

The stable search engine returns:

- `query_spectrum_df`
- `query_peaks_df`
- `peak_matches_df`
- `candidate_df_ref`
- `candidate_df_similarity`
- `candidate_df_reranked`
- `candidate_df_component_reranked`
- `class_df_reranked`
- `interpretation_text`

## Intentionally removed from active use

The following experimental layers are not part of the active baseline:

- hard biochemical class filtering
- soft class-prior reranking
- diagnostic peak reranking
- derivative-based similarity weighting
- region-specific similarity weighting
- confuser-aware reranking
- contrastive reranking

## Current collagen example result

For `data/processed/test_queries/collagen_example_query.csv`:

- ref-level collagen rank = 2
- component-level collagen rank = 2
