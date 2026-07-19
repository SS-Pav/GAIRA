"""GAIRA V5 Foundation — Raman-only biochemical foundation model.

Canonical observation domain is Raman. Ag-SERS / Au-SERS / DART are future
observation domains and are excluded here. Builds a frozen biochemical manifold
from pure Raman reference analytes (C1-C2), discovers emergent biochemical axes
(C3), derives the Biochemical State Vector (C4) and Molecular Spectral Signatures
(C5), and projects held-out and biological Raman spectra into the frozen space
without retraining (C6-C7).
"""
