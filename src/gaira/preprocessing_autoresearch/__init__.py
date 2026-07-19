"""GAIRA V5 Stage B0 — Preprocessing AutoResearch for Raman/Ag-SERS comparability.

A constrained, leakage-safe, multi-objective search over physically reasonable
preprocessing and Ag-SERS background-correction pipelines. Reuses
src/gaira/preprocessing (grid, resample, ASLS, Savitzky-Golay, normalizers),
src/gaira/representation (retrieval, metrics) and src/gaira/evidence (dataset,
split machinery). Nothing here modifies Stage A/B results or historical pipelines.
"""
